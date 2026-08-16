# -*- coding: utf-8 -*-
"""Daily revenue assurance: every POS trading dollar must reach the GL, fast.
Checks (run 10:45, after the morning lake top-up):
  A. POS -> GL sales journals: for each of the last 5 settled days (D-3..D-7,
     leaving 2 days' posting lag), compare POS net sales (mart) to revenue
     postings in the GL; flag days with no journal or a gap > 3%.
  B. Adyen settlement clearing: open 'Receivable - Adyen' balances older than
     5 days = settlements not clearing.
Breaches raise ONE open [Revenue] task each (dedup while open). The July close
found $19.7M of unposted sales journals at WD+1 - this catches it same-week."""
import datetime as dt
import logging

import tk_calendar
import tk_db
import lake_reader

LOG = logging.getLogger("tk_revenue")

FINCTRL_PROJECT = "aaaaaaaa-0000-0000-0000-000000000003"  # Month-end close lives in pillar 3
CLOSE_PROJECT = "aaaaaaaa-0000-0000-0000-000000000001"
GAP_PCT = 3.0
ADYEN_AGE_DAYS = 5


def _task(title, desc, priority="critical"):
    if tk_db.get("tasks", {"title": "eq." + title, "status": "neq.done",
                           "select": "id", "limit": "1"}):
        return False
    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id", "limit": "1"})
    due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=1))
    tk_db.insert("tasks", [{
        "project_id": CLOSE_PROJECT, "title": title, "description": desc,
        "priority": priority,
        "assignee_id": admin[0]["id"] if admin else None,
        "due_date": due.isoformat(), "source": "watcher"}])
    return True


def run():
    lake_reader.sync(extra_tables=["XeroAccountTransactionsMasterView", "Invoices"],
                     log=LOG.info)
    issues, created = [], 0

    # A. POS vs GL revenue by day (settled window D-7..D-3)
    try:
        r = lake_reader.query("""
            WITH mx AS (SELECT max(CAST("date" AS DATE)) m FROM mart_venue_daily),
            pos AS (SELECT CAST("date" AS DATE) d, round(sum(net_sales),0) pos_net
                    FROM mart_venue_daily
                    WHERE CAST("date" AS DATE) BETWEEN (SELECT m FROM mx) - INTERVAL 7 DAY
                                                   AND (SELECT m FROM mx) - INTERVAL 3 DAY
                    GROUP BY 1),
            gl AS (SELECT CAST("Date" AS DATE) d,
                          round(sum(-TRY_CAST("Net Amount" AS DOUBLE)),0) gl_rev
                   FROM XeroAccountTransactionsMasterView
                   WHERE lower("Account Type") LIKE '%revenue%'
                     AND CAST("Date" AS DATE) BETWEEN (SELECT m FROM mx) - INTERVAL 7 DAY
                                                  AND (SELECT m FROM mx) - INTERVAL 3 DAY
                   GROUP BY 1)
            SELECT pos.d, pos.pos_net, coalesce(gl.gl_rev,0) gl_rev,
                   round(100.0*(coalesce(gl.gl_rev,0)-pos.pos_net)/nullif(pos.pos_net,0),1) gap_pct
            FROM pos LEFT JOIN gl USING (d) ORDER BY pos.d""", max_rows=10)
        bad = [row for row in r["rows"]
               if row[1] and (row[2] == 0 or abs(row[3] or 100) > GAP_PCT)]
        if bad:
            lines = "\n".join("- %s: POS $%s vs GL revenue $%s (%s%%)" % (
                str(row[0])[:10], "{:,.0f}".format(row[1]),
                "{:,.0f}".format(row[2]), row[3]) for row in bad)
            issues.append("sales_journals")
            if _task("[Revenue] Sales journals missing or short vs POS",
                     "Daily revenue assurance: GL revenue does not match POS net sales "
                     "for settled days (2-day posting lag already allowed):\n%s\n"
                     "First steps:\n- Check the daily sales journal run for those days\n"
                     "- If journals are posted to other accounts, tell Peter so the "
                     "screen's account filter is widened" % lines):
                created += 1
    except Exception:
        LOG.exception("POS vs GL check failed")
        issues.append("sales_journals_error")

    # B. Adyen settlement clearing
    try:
        r = lake_reader.query("""
            SELECT round(sum(TRY_CAST(amountdue AS DOUBLE)),0) open_amt,
                   min(CAST(date AS DATE)) oldest
            FROM Invoices
            WHERE type = 1 AND TRY_CAST(amountdue AS DOUBLE) > 0
              AND lower(contact_name) LIKE '%adyen%'""", max_rows=1)
        open_amt, oldest = r["rows"][0]
        if open_amt and oldest and (dt.date.today() -
                dt.date.fromisoformat(str(oldest)[:10])).days > ADYEN_AGE_DAYS:
            issues.append("adyen_clearing")
            if _task("[Revenue] Adyen settlements not clearing",
                     "Open Adyen settlement receivable $%s with the oldest item dated %s "
                     "(older than %d days). Card takings may not be landing in the bank "
                     "or the clearing entries are not being posted.\nFirst steps:\n"
                     "- Check the Adyen payout report vs bank\n- Post/chase the clearing "
                     "entries" % ("{:,.0f}".format(open_amt), str(oldest)[:10],
                                  ADYEN_AGE_DAYS)):
                created += 1
    except Exception:
        LOG.exception("Adyen check failed")

    return {"issues": issues, "tasks_created": created}
