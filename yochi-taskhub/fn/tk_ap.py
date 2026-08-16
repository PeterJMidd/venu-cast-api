# -*- coding: utf-8 -*-
"""AP autopilot (Mondays 07:25):
  A. Received-not-invoiced: Restoke purchasing lines received without an
     invoice number - Sally's biggest EOM time sink, now a standing list.
  B. Duplicate screen at entry: same supplier + amount + date bill groups in
     the last 14 days, before they get paid.
  C. Proposed payment run: open AP due in the next 7 days by supplier, ready
     for approval.
Email to admins + [AP] tasks for material items."""
import datetime as dt
import logging

import tk_calendar
import tk_db
import tk_email
import lake_reader

LOG = logging.getLogger("tk_ap")

FINOPS_PROJECT = "aaaaaaaa-0000-0000-0000-000000000019"
RNI_TASK_THRESHOLD = 25000


def run(email=True):
    lake_reader.sync(extra_tables=["restoke_purchasing", "Invoices"], log=LOG.info)

    rni = lake_reader.query("""
        SELECT supplier, venue, round(sum(TRY_CAST(total AS DOUBLE)),0) amt, count(*) lines
        FROM restoke_purchasing
        WHERE received_date IS NOT NULL AND received_date <> ''
          AND (invoice_number IS NULL OR trim(invoice_number) = '')
          AND CAST(received_date AS DATE) >= current_date - 45
          AND CAST(received_date AS DATE) < current_date - 3
        GROUP BY 1,2 HAVING sum(TRY_CAST(total AS DOUBLE)) > 500
        ORDER BY 3 DESC LIMIT 25""", max_rows=25)

    dups = lake_reader.query("""
        SELECT contact_name, CAST(date AS DATE) d,
               round(TRY_CAST(total AS DOUBLE),2) amt, count(*) n,
               string_agg(invoicenumber, ' | ') invs
        FROM Invoices
        WHERE type = 0 AND CAST(date AS DATE) >= current_date - 14
          AND TRY_CAST(total AS DOUBLE) > 200
        GROUP BY 1,2,3 HAVING count(*) > 1
        ORDER BY 3 DESC LIMIT 15""", max_rows=15)

    payrun = lake_reader.query("""
        SELECT contact_name, round(sum(TRY_CAST(amountdue AS DOUBLE)),0) due_amt,
               count(*) bills, min(CAST(duedate AS DATE)) first_due
        FROM Invoices
        WHERE type = 0 AND status = 3 AND TRY_CAST(amountdue AS DOUBLE) > 0
          AND CAST(duedate AS DATE) <= current_date + 7
        GROUP BY 1 ORDER BY 2 DESC LIMIT 30""", max_rows=30)

    rni_total = sum(r[2] for r in rni["rows"]) if rni["rows"] else 0
    payrun_total = sum(r[1] for r in payrun["rows"]) if payrun["rows"] else 0

    created = 0
    if rni_total > RNI_TASK_THRESHOLD:
        title = "[AP] Received-not-invoiced — $%s to accrue/chase" % "{:,.0f}".format(rni_total)
        if not tk_db.get("tasks", {"title": "eq." + title, "status": "neq.done",
                                   "select": "id", "limit": "1"}):
            admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                           "select": "id", "limit": "1"})
            due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=3))
            lines = "\n".join("- %s @ %s: $%s (%s lines)" % (r[0], r[1],
                              "{:,.0f}".format(r[2]), r[3]) for r in rni["rows"][:15])
            tk_db.insert("tasks", [{
                "project_id": FINOPS_PROJECT, "title": title,
                "description": ("Stock received in Restoke with no invoice recorded "
                                "(3-45 days old):\n%s\nFirst steps:\n- Chase the "
                                "supplier statements for these\n- Accrue what's still "
                                "uninvoiced at month end" % lines),
                "priority": "high",
                "assignee_id": admin[0]["id"] if admin else None,
                "due_date": due.isoformat(), "source": "watcher"}])
            created += 1

    if email:
        def tbl(headers, rows, empty="Nothing flagged."):
            if not rows:
                return "<p style='color:#9ca3af;font-size:12px'>%s</p>" % empty
            return ("<table cellpadding=4 border=1 style='border-collapse:collapse;"
                    "font-size:11.5px'><tr>%s</tr>%s</table>") % (
                "".join("<th>%s</th>" % h for h in headers),
                "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r)
                        for r in rows))
        html = ("<p><b>AP autopilot — week of %s</b></p>"
                "<p><b>A. Received, not invoiced</b> (total $%s — accrual/chase list)</p>%s"
                "<p><b>B. Possible duplicate bills entered (last 14 days) — review before payment</b></p>%s"
                "<p><b>C. Proposed payment run — open AP due within 7 days (total $%s)</b></p>%s"
                "<p style='color:#6b7280;font-size:11px'>Screens over Restoke purchasing "
                "and Xero bills. Items are 'to review', not conclusions.</p>") % (
            dt.date.today().isoformat(), "{:,.0f}".format(rni_total),
            tbl(["Supplier", "Venue", "$", "Lines"], rni["rows"]),
            tbl(["Supplier", "Date", "$", "Count", "Invoice #s"], dups["rows"]),
            "{:,.0f}".format(payrun_total),
            tbl(["Supplier", "$ due", "Bills", "First due"], payrun["rows"]))
        for a in tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                        "select": "email"}):
            tk_email.send(a["email"], "AP autopilot — RNI $%s · payrun $%s · %d dup group(s)" % (
                "{:,.0f}".format(rni_total), "{:,.0f}".format(payrun_total),
                len(dups["rows"])), html)

    return {"rni_total": rni_total, "rni_suppliers": len(rni["rows"]),
            "dup_groups": len(dups["rows"]), "payrun_total": payrun_total,
            "payrun_suppliers": len(payrun["rows"]), "tasks_created": created}
