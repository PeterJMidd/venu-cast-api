# -*- coding: utf-8 -*-
"""dataSights watcher: run each active rule's SQL over the lake marts; breaches
become tasks (one per rule per period, AI-written context). Rules are written
against the mart_* views (mart_venue_daily etc.)."""
import datetime as dt
import json
import logging
import os
import re

import tk_ai
import tk_calendar
import tk_db
import tk_templates
import lake_reader

LOG = logging.getLogger("tk_watcher")

# Supabase-side view materialised into the DuckDB cache so rules can join
# actuals (mart_venue_daily) to the Venu Cast base case. Venue names match
# the mart exactly ("Yo-Chi Albert St").
FORECAST_TABLE = "forecast_venue_daily"

CONTEXT_SYSTEM = (
    "You write the description for a task raised automatically by a data watcher at "
    "Yo-Chi (AU frozen-yoghurt chain). You get the rule and the breaching rows. Write "
    "plain text (no markdown): 2-3 sentences explaining what the data shows and why it "
    "was flagged (name venues and numbers), then a short 'First steps:' list of 2-3 "
    "concrete actions. Factual tone, no drama. Money is AUD net of GST."
)


def _sync_forecast(log=LOG.info):
    """Materialise taskapp.v_forecast_venue_daily (base-case net sales per venue,
    trailing 14 days) as a parquet table in the lake cache so watch rules can
    reference it like any lake table."""
    rows = tk_db.get("v_forecast_venue_daily", {"select": "venue,d,forecast_sales"})
    if not rows:
        log("forecast sync: view returned no rows - keeping any previous copy")
        return 0
    import duckdb
    tdir = os.path.join(lake_reader.CACHE_DIR, "tables", FORECAST_TABLE)
    os.makedirs(tdir, exist_ok=True)
    path = os.path.join(tdir, "data.parquet").replace("\\", "/")
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE t (venue VARCHAR, d DATE, forecast_sales DOUBLE)")
        con.executemany("INSERT INTO t VALUES (?,?,?)",
                        [(r["venue"], r["d"], r["forecast_sales"]) for r in rows])
        con.execute("COPY t TO '%s' (FORMAT PARQUET)" % path)
    finally:
        con.close()
    with open(os.path.join(tdir, "_meta.json"), "w") as f:
        json.dump({"table": FORECAST_TABLE, "rows": len(rows),
                   "source": "taskapp.v_forecast_venue_daily",
                   "synced_at": dt.datetime.utcnow().isoformat() + "Z"}, f)
    log("forecast sync: %d venue-day rows" % len(rows))
    return len(rows)


def run():
    period_id, first = tk_templates.ensure_period()
    rules = tk_db.get("watcher_rules", {"active": "eq.true", "select": "*"})
    if not rules:
        return {"rules": 0, "tasks_created": 0, "task_ids": []}

    # Sync every table the rules reference (FROM/JOIN identifiers; CTE names
    # harmlessly match nothing in blob)
    referenced = set()
    for rule in rules:
        referenced.update(re.findall(r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_][A-Za-z0-9_]*)\"?",
                                     rule["check_sql"], re.I))
    lake_reader.sync(extra_tables=referenced, log=LOG.info)
    if FORECAST_TABLE in referenced:
        try:
            _sync_forecast()
        except Exception:
            LOG.exception("forecast sync failed - forecast rules will error")
    created, task_ids, errors = 0, [], []
    for rule in rules:
        try:
            result = lake_reader.query(rule["check_sql"], max_rows=50)
            breaches = result["rows"]
            if breaches:
                # AI context from the breach rows
                user = "Rule: %s\n%s\n\nBreaching rows (columns: %s):\n%s" % (
                    rule["name"], rule.get("description") or "",
                    ", ".join(result["columns"]),
                    json.dumps(breaches, separators=(",", ":"))[:6000],
                )
                try:
                    desc = tk_ai.text(CONTEXT_SYSTEM, user, max_tokens=800)
                except Exception as e:
                    LOG.warning("AI context failed for %s: %s", rule["name"], e)
                    desc = "Automatic flag: %d row(s) breached this rule.\n%s" % (
                        len(breaches), json.dumps(breaches[:10], indent=1))
                due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=2))
                inserted = tk_db.insert("tasks", [{
                    "project_id": rule["project_id"],
                    "period_id": period_id,
                    "watcher_rule_id": rule["id"],
                    "title": "[Watch] %s" % rule["name"],
                    "description": desc,
                    "priority": rule.get("priority", "high"),
                    "assignee_id": rule.get("assignee_id"),
                    "due_date": due.isoformat(),
                    "source": "watcher",
                }], on_conflict="watcher_rule_id,period_id", ignore_duplicates=True,
                    returning=True)
                # duplicates (same rule+period) come back empty - only NEW tasks count
                for row in inserted or []:
                    task_ids.append(row["id"])
                    created += 1
            tk_db.patch("watcher_rules", {"id": "eq." + rule["id"]},
                        {"last_run_at": dt.datetime.utcnow().isoformat() + "Z"})
        except Exception as e:
            LOG.exception("rule %s failed", rule.get("name"))
            errors.append({"rule": rule.get("name"), "error": str(e)[:200]})
    return {"rules": len(rules), "tasks_created": created, "task_ids": task_ids,
            "errors": errors}
