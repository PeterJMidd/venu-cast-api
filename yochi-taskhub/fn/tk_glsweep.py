# -*- coding: utf-8 -*-
"""Month-end GL sweep: analyse EVERY account's activity for the just-closed month
vs the prior 3 months, flag accounts needing review (materiality + behaviour
rules), AI-analyse the top tier into individual close-review tasks with
recommended steps, and attach the full scorecard workbook to a summary task."""
import datetime as dt
import json
import logging

import tk_ai
import tk_calendar
import tk_db
import tk_email
import tk_agent
import lake_reader

LOG = logging.getLogger("tk_glsweep")

CLOSE_PROJECT = "aaaaaaaa-0000-0000-0000-000000000001"
MAX_TASKS = 25  # individual AI-analysed tasks; the rest go to the scorecard

STATS_SQL = '''WITH tx AS (
  SELECT "Account Code" AS code, "Account" AS name, "Account Type" AS atype,
         CAST("Date" AS DATE) AS d, CAST("Net Amount" AS DOUBLE) AS net
  FROM XeroAccountTransactionsMasterView
  WHERE CAST("Date" AS DATE) >= date_trunc('month', current_date) - INTERVAL 4 MONTH
),
pm AS (SELECT date_trunc('month', current_date) - INTERVAL 1 MONTH AS m0),
agg AS (
  SELECT code, name, atype,
    sum(CASE WHEN d >= (SELECT m0 FROM pm) AND d < (SELECT m0 FROM pm) + INTERVAL 1 MONTH THEN net ELSE 0 END) AS cur_net,
    sum(CASE WHEN d >= (SELECT m0 FROM pm) AND d < (SELECT m0 FROM pm) + INTERVAL 1 MONTH THEN 1 ELSE 0 END) AS cur_txns,
    sum(CASE WHEN d < (SELECT m0 FROM pm) THEN net ELSE 0 END) / 3.0 AS avg3_net,
    sum(CASE WHEN d < (SELECT m0 FROM pm) THEN 1 ELSE 0 END) / 3.0 AS avg3_txns
  FROM tx GROUP BY code, name, atype
)
SELECT code, name, atype, round(cur_net,0) AS cur_net, cur_txns,
       round(avg3_net,0) AS avg3_net, round(avg3_txns,1) AS avg3_txns,
       CASE
         WHEN abs(cur_net) > 5000 AND avg3_txns = 0 THEN 'new activity in dormant account'
         WHEN abs(cur_net) > 5000 AND avg3_net <> 0 AND sign(cur_net) <> sign(avg3_net) THEN 'sign flip vs prior months'
         WHEN abs(cur_net) > 5000 AND abs(avg3_net) > 500 AND abs(cur_net - avg3_net) / abs(avg3_net) > 0.5 THEN 'movement >50% vs 3-month average'
         WHEN cur_txns = 0 AND avg3_txns >= 3 THEN 'expected activity missing this month'
         ELSE NULL END AS flag
FROM agg
WHERE flag IS NOT NULL
ORDER BY abs(cur_net) DESC'''

TXNS_SQL_TEMPLATE = '''WITH pm AS (SELECT date_trunc('month', current_date) - INTERVAL 1 MONTH AS m0)
SELECT * FROM (
  SELECT "Account Code" AS code, CAST("Date" AS DATE) AS d, "Source" AS source,
         "Contact" AS contact, "Description" AS description,
         round(CAST("Net Amount" AS DOUBLE),0) AS net,
         row_number() OVER (PARTITION BY "Account Code"
                            ORDER BY abs(CAST("Net Amount" AS DOUBLE)) DESC) AS rn
  FROM XeroAccountTransactionsMasterView
  WHERE CAST("Date" AS DATE) >= (SELECT m0 FROM pm)
    AND CAST("Date" AS DATE) < (SELECT m0 FROM pm) + INTERVAL 1 MONTH
    AND "Account Code" IN (%s)
) WHERE rn <= 5 ORDER BY code, rn'''

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "accounts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "analysis": {"type": "string", "description": "2-3 sentences: what moved and the most likely explanation"},
                    "close_steps": {"type": "array", "items": {"type": "string"},
                                    "description": "2-4 concrete review/close steps"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["code", "analysis", "close_steps", "priority"],
            },
        }
    },
    "required": ["accounts"],
}

ANALYSIS_SYSTEM = (
    "You are reviewing flagged general-ledger accounts for Yo-Chi's month-end close. For "
    "each account you get: type, the closed month's net movement and transaction count, the "
    "prior 3-month averages, the flag reason, and the largest transactions. For EACH account "
    "return: analysis (what moved, most likely explanation - e.g. intercompany funding round, "
    "payroll timing, clearing account backlog, one-off journal), close_steps (concrete: what "
    "to reconcile/verify/journal and to what evidence), and priority (critical = could "
    "misstate the accounts materially; high = must clear before sign-off; medium/low = "
    "hygiene). Large intercompany loan or clearing-account swings around month end are often "
    "legitimate funding/settlement patterns - say what would CONFIRM that rather than "
    "assuming error. Money is AUD."
)


def _prior_period():
    first_this = dt.date.today().replace(day=1)
    prior = (first_this - dt.timedelta(days=1)).replace(day=1)
    tk_db.insert("periods", [{"period_month": prior.isoformat(),
                              "label": prior.strftime("%b %Y")}],
                 on_conflict="period_month", ignore_duplicates=True)
    rows = tk_db.get("periods", {"period_month": "eq." + prior.isoformat(), "select": "id"})
    return rows[0]["id"], prior


def run(force=False):
    today = dt.date.today()
    if not force and today != tk_calendar.business_day_of_month(today.year, today.month, 1):
        return {"skipped": "runs on the first business day of the month"}

    period_id, prior = _prior_period()
    lake_reader.sync(extra_tables=["XeroAccountTransactionsMasterView"], log=LOG.info)
    stats = lake_reader.query(STATS_SQL, max_rows=500)
    flagged = [dict(zip(stats["columns"], r)) for r in stats["rows"]]
    if not flagged:
        return {"flagged": 0, "tasks_created": 0}

    top = flagged[:MAX_TASKS]
    codes = ",".join("'%s'" % str(a["code"]).replace("'", "") for a in top)
    txns = lake_reader.query(TXNS_SQL_TEMPLATE % codes, max_rows=200)
    tx_by_code = {}
    for r in txns["rows"]:
        row = dict(zip(txns["columns"], r))
        tx_by_code.setdefault(str(row["code"]), []).append(
            {k: row[k] for k in ("d", "source", "contact", "description", "net")})

    payload = [{**a, "top_transactions": tx_by_code.get(str(a["code"]), [])} for a in top]
    # batch the analysis (8 accounts/call) so long generations never truncate
    analyses = {}
    for i in range(0, len(payload), 8):
        batch = payload[i:i + 8]
        try:
            out = tk_ai.structured(
                ANALYSIS_SYSTEM,
                "Closed month: %s\n\nFlagged accounts:\n%s" % (
                    prior.strftime("%B %Y"), json.dumps(batch, separators=(",", ":"))[:40000]),
                "analyse_accounts", ANALYSIS_SCHEMA, max_tokens=4000)
            for a in out.get("accounts", []):
                analyses[str(a.get("code", "")).strip()] = a
        except Exception:
            LOG.exception("analysis batch %d failed", i // 8)

    # dedup: existing GL review tasks for this period
    existing = {t["title"] for t in tk_db.get(
        "tasks", {"period_id": "eq." + period_id, "select": "title",
                  "title": "like.GL review:*"})}
    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id,email", "limit": "1"})[0]
    due = tk_calendar.roll_forward(today + dt.timedelta(days=2))
    created, task_ids = 0, []
    for a in top:
        title = "GL review: %s %s" % (a["code"], a["name"])
        if title in existing:
            continue
        ai = analyses.get(str(a["code"]).strip())
        desc = (
            "Flag: %s\n%s month movement: $%s over %s txns (3-month avg: $%s / %s txns)\n\n"
            "Analysis: %s\n\nRecommended close steps:\n%s"
        ) % (
            a["flag"], prior.strftime("%b %Y"),
            "{:,.0f}".format(a["cur_net"]), int(a["cur_txns"]),
            "{:,.0f}".format(a["avg3_net"]), a["avg3_txns"],
            ai["analysis"] if ai else "(AI analysis unavailable)",
            "\n".join("- " + s for s in (ai["close_steps"] if ai else
                                         ["Reconcile the movement to source evidence"])),
        )
        rows = tk_db.insert("tasks", [{
            "project_id": CLOSE_PROJECT,
            "period_id": period_id,
            "title": title,
            "description": desc,
            "priority": (ai or {}).get("priority", "medium"),
            "assignee_id": admin["id"],
            "due_date": due.isoformat(),
            "source": "watcher",
        }], returning=True)
        task_ids.append(rows[0]["id"])
        created += 1

    # summary task with the full scorecard workbook
    summary_title = "GL sweep — %s close" % prior.strftime("%b %Y")
    if summary_title not in {t["title"] for t in tk_db.get(
            "tasks", {"period_id": "eq." + period_id, "select": "title"})}:
        rows = tk_db.insert("tasks", [{
            "project_id": CLOSE_PROJECT, "period_id": period_id,
            "title": summary_title,
            "description": ("Full-ledger sweep for the %s close: %d accounts flagged of the "
                            "whole GL; %d individual review tasks raised (see 'GL review:' "
                            "tasks). Complete scorecard attached." % (
                                prior.strftime("%b %Y"), len(flagged), created)),
            "priority": "high", "assignee_id": admin["id"],
            "due_date": due.isoformat(), "source": "watcher",
        }], returning=True)
        summary_id = rows[0]["id"]
        xlsx = tk_agent._xlsx_bytes({"GL sweep scorecard": stats})
        tk_agent._upload(summary_id, "GL_sweep_%s.xlsx" % prior.strftime("%Y-%m"), xlsx,
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         admin["id"])

    tk_email.send(admin["email"], "GL sweep — %s: %d flagged, %d tasks" % (
        prior.strftime("%b %Y"), len(flagged), created),
        "<p>The month-end GL sweep analysed the full ledger for %s.</p>"
        "<ul><li>%d accounts flagged for review</li><li>%d individual review tasks created "
        "with AI analysis and close steps</li><li>Full scorecard attached to the "
        "'GL sweep' task in TaskHub</li></ul>" % (prior.strftime("%B %Y"), len(flagged), created))
    return {"flagged": len(flagged), "tasks_created": created, "task_ids": task_ids}
