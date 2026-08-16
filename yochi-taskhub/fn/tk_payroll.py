# -*- coding: utf-8 -*-
"""Payroll control loop (Mondays 07:05):
  A. Payrun reconciliation: GL wages postings vs Tanda timesheet cost for the
     last 2 complete weeks - variance beyond tolerance means a payrun differs
     from what was worked/approved (the class of failure behind the missed WA
     pay). Tolerance is generous (7.5%) because super/allowances timing differ.
  B. Approval hygiene: PENDING timesheet shifts older than 7 days.
Flags raise ONE open [Payroll] task per issue type. Language: areas to review."""
import datetime as dt
import logging

import tk_calendar
import tk_db
import lake_reader

LOG = logging.getLogger("tk_payroll")

PEOPLE_PROJECT = "aaaaaaaa-0000-0000-0000-000000000020"
VARIANCE_PCT = 7.5
PENDING_AGE_DAYS = 7
PENDING_THRESHOLD = 25


def _task(title, desc):
    if tk_db.get("tasks", {"title": "eq." + title, "status": "neq.done",
                           "select": "id", "limit": "1"}):
        return False
    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id", "limit": "1"})
    due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=2))
    tk_db.insert("tasks", [{
        "project_id": PEOPLE_PROJECT, "title": title, "description": desc,
        "priority": "high",
        "assignee_id": admin[0]["id"] if admin else None,
        "due_date": due.isoformat(), "source": "watcher"}])
    return True


def run():
    lake_reader.sync(extra_tables=["XeroAccountTransactionsMasterView",
                                   "TandaTimesheetShifts"], log=LOG.info)
    created, checks = 0, {}

    # A. GL wages vs Tanda cost, last 2 complete Mon-Sun weeks
    try:
        r = lake_reader.query("""
            WITH wk AS (SELECT date_trunc('week', current_date) w0),
            tanda AS (SELECT date_trunc('week', CAST(date AS DATE)) w,
                             round(sum(TRY_CAST(cost AS DOUBLE)),0) tanda_cost
                      FROM TandaTimesheetShifts
                      WHERE CAST(date AS DATE) >= (SELECT w0 FROM wk) - INTERVAL 14 DAY
                        AND CAST(date AS DATE) < (SELECT w0 FROM wk)
                      GROUP BY 1),
            gl AS (SELECT date_trunc('week', CAST("Date" AS DATE)) w,
                          round(sum(TRY_CAST("Net Amount" AS DOUBLE)),0) gl_wages
                   FROM XeroAccountTransactionsMasterView
                   WHERE (lower("Account") LIKE '%wage%' OR lower("Account") LIKE '%salar%')
                     AND lower("Account") NOT LIKE '%superann%'
                     AND CAST("Date" AS DATE) >= (SELECT w0 FROM wk) - INTERVAL 14 DAY
                     AND CAST("Date" AS DATE) < (SELECT w0 FROM wk)
                   GROUP BY 1)
            SELECT CAST(t.w AS DATE) wk, t.tanda_cost, coalesce(g.gl_wages,0) gl_wages,
                   round(100.0*(coalesce(g.gl_wages,0)-t.tanda_cost)/nullif(t.tanda_cost,0),1) var_pct
            FROM tanda t LEFT JOIN gl g USING (w) ORDER BY 1""", max_rows=4)
        checks["payrun_weeks"] = r["rows"]
        bad = [row for row in r["rows"]
               if row[1] and row[1] > 100000 and abs(row[4] or 100) > VARIANCE_PCT]
        if bad:
            lines = "\n".join("- wk %s: Tanda $%s vs GL wages $%s (%s%%)" % (
                str(row[0])[:10], "{:,.0f}".format(row[1]),
                "{:,.0f}".format(row[2]), row[4]) for row in bad)
            if _task("[Payroll] Payrun vs Tanda variance to review",
                     "Weekly payroll reconciliation - GL wages postings differ from "
                     "Tanda timesheet cost beyond %.1f%% tolerance:\n%s\nFirst steps:\n"
                     "- Check payrun export completeness for those weeks (approved-but-"
                     "not-exported timesheets)\n- Check GL posting timing/accounts\n"
                     "These are areas to review, not conclusions." % (VARIANCE_PCT, lines)):
                created += 1
    except Exception:
        LOG.exception("payrun rec failed")

    # B. stale PENDING timesheets
    try:
        r = lake_reader.query("""
            SELECT count(*) n, min(CAST(date AS DATE)) oldest
            FROM TandaTimesheetShifts
            WHERE upper(status) = 'PENDING'
              AND CAST(date AS DATE) < current_date - %d""" % PENDING_AGE_DAYS,
            max_rows=1)
        n, oldest = r["rows"][0]
        checks["stale_pending"] = int(n or 0)
        if (n or 0) > PENDING_THRESHOLD:
            if _task("[Payroll] %d timesheet shifts pending approval >%dd" % (n, PENDING_AGE_DAYS),
                     "There are %d timesheet shifts still PENDING approval, the oldest "
                     "from %s. Unapproved shifts risk missed or late pays.\nFirst steps:\n"
                     "- Chase venue leaders via the daily deck's unapproved matrix\n"
                     "- Confirm none belong to a closed pay period" % (n, str(oldest)[:10])):
                created += 1
    except Exception:
        LOG.exception("pending check failed")

    return {"checks": checks, "tasks_created": created}
