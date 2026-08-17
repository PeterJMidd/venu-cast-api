# -*- coding: utf-8 -*-
"""Data contracts: declarative freshness + volume SLAs on the tables the
cockpit depends on.

Before this, staleness was policed by two hand-written watcher rules, so when
mart_venue_daily silently stopped at 13-Aug the 16-Aug report just printed zero
sales and nobody was told. Now every critical table declares how fresh it must
be and how many rows it should carry, one job checks them all each morning
BEFORE the reports run, and a single task lists every breach.

Read-only against the lake; writes only its own status back to
taskapp.data_contracts (+ one dedup'd task when something breaches)."""
import datetime as dt
import logging

import lake_reader
import tk_calendar
import tk_db

LOG = logging.getLogger("tk_contracts")

AEST = dt.timezone(dt.timedelta(hours=10))
DATA_WATCH = "Data watch"


def _freshness_sql(c):
    """latest date + recent row count, ignoring nonsense future dates (the Xero
    view carries rows dated 2802)."""
    return (
        "SELECT max(d) AS latest, "
        "sum(CASE WHEN d >= current_date - %d THEN 1 ELSE 0 END) AS recent_rows, "
        "count(*) AS total_rows "
        "FROM (SELECT %s AS d FROM %s) AS t "
        "WHERE d IS NULL OR d <= current_date + 1"
        % (int(c.get("recent_days") or 7), c["date_expr"], c["table_name"]))


def _expected_latest(c, today):
    """Oldest acceptable 'latest' date for this contract."""
    allowed = int(c.get("max_staleness_days") or 2)
    if c.get("weekdays_only"):
        d = today
        while not tk_calendar.is_business_day(d):
            d -= dt.timedelta(days=1)
        return d - dt.timedelta(days=allowed)
    return today - dt.timedelta(days=allowed)


def _project_id(name):
    rows = tk_db.get("projects", {"name": "eq." + name, "select": "id", "limit": "1"})
    return rows[0]["id"] if rows else None


def check(raise_task=True, today=None):
    today = today or dt.datetime.now(AEST).date()
    contracts = tk_db.get("data_contracts", {"active": "eq.true", "select": "*",
                                             "order": "table_name"})
    breaches, checked, results = [], 0, []
    for c in contracts:
        checked += 1
        label = c.get("label") or c["table_name"]
        status, detail = "ok", ""
        try:
            out = lake_reader.query(_freshness_sql(c), max_rows=1)
            row = dict(zip(out["columns"], out["rows"][0])) if out["rows"] else {}
            latest = row.get("latest")
            recent = int(row.get("recent_rows") or 0)
            total = int(row.get("total_rows") or 0)
            want_latest = _expected_latest(c, today)
            min_rows = int(c.get("min_rows_recent") or 0)
            problems = []
            if not latest:
                problems.append("no dated rows at all (%d rows in table)" % total)
            else:
                latest_d = dt.date.fromisoformat(str(latest)[:10])
                if latest_d < want_latest:
                    problems.append("stale: latest %s, expected %s or newer (%d days behind)"
                                    % (latest_d, want_latest, (today - latest_d).days))
            if min_rows and recent < min_rows:
                problems.append("thin: %d rows in last %sd, expected >= %d"
                                % (recent, c.get("recent_days") or 7, min_rows))
            if problems:
                status, detail = "breach", "; ".join(problems)
            else:
                detail = "latest %s, %d rows/%sd" % (
                    str(latest)[:10], recent, c.get("recent_days") or 7)
        except Exception as e:
            status = "error"
            detail = str(e)[:250]
            LOG.exception("contract check failed for %s", c["table_name"])
        results.append({"table": c["table_name"], "label": label,
                        "status": status, "detail": detail})
        if status != "ok":
            breaches.append({"label": label, "table": c["table_name"],
                             "status": status, "detail": detail})
        try:
            tk_db.patch("data_contracts", {"id": "eq." + c["id"]},
                        {"last_status": status, "last_detail": detail[:500],
                         "last_checked_at": dt.datetime.utcnow().isoformat() + "Z"})
        except Exception:
            LOG.exception("could not record contract status for %s", c["table_name"])

    task_id = None
    if raise_task and breaches:
        ref = "contract:" + today.isoformat()
        body = "\n".join("- %s (%s): %s" % (b["label"], b["table"], b["detail"])
                         for b in breaches)
        pid = _project_id(DATA_WATCH)
        if pid:
            try:
                rows = tk_db.insert("tasks", [{
                    "project_id": pid,
                    "title": "[Data] %d data contract breach%s"
                             % (len(breaches), "" if len(breaches) == 1 else "es"),
                    "description": "Checked before this morning's reports.\n\n" + body
                                   + "\n\nUntil these clear, any report reading these "
                                     "tables is understating or missing data.",
                    "priority": "high",
                    "source": "watcher",
                    "due_date": today.isoformat(),
                    "external_ref": ref,
                }], on_conflict="external_ref", ignore_duplicates=True, returning=True)
                task_id = rows[0]["id"] if rows else None
            except Exception:
                LOG.exception("could not raise contract breach task")
    return {"checked": checked, "breaches": breaches, "results": results,
            "task_id": task_id}


def run():
    return check(raise_task=True)
