# -*- coding: utf-8 -*-
"""13-week rolling cash forecast (v1, assumption-based - every lever is an env
knob). Weekly buckets from the Monday of the current week:

  opening   = group Bank accounts, latest consolidated balance sheet (lake)
  receipts  = Venu Cast base-case net sales x 1.10 GST x RECEIPTS_FACTOR
  AP        = max(open payables due that week, trailing-8-week actual AP paid
              run-rate from Invoices.fullypaidondate); run-rate only if the
              aged view is unavailable
  payroll   = forecast sales x PAYROLL_PCT x 0.80 (net of PAYG withheld -
              PAYG+GST are remitted in the ATO line)
  ATO       = ATO_MONTHLY in the week containing the 21st
  super     = month sales x PAYROLL_PCT x SUPER_RATE in the week of the 28th

Writes one generation to taskapp.cash_forecast, emails admins on Mondays, and
raises a [Cash] task if any closing balance falls below CASH_FLOOR."""
import datetime as dt
import json
import logging
import os

import tk_calendar
import tk_db
import tk_email
import lake_reader

LOG = logging.getLogger("tk_cash")

DATA_WATCH_PROJECT = "aaaaaaaa-0000-0000-0000-000000000005"
WEEKS = 13


def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _monday(d):
    return d - dt.timedelta(days=d.weekday())


def _opening_cash():
    r = lake_reader.query("""
        WITH b AS (SELECT CAST(Date AS DATE) d, TRY_CAST(Amount AS DOUBLE) amt
                   FROM XeroConsolidationBalanceSheetGroupReportView
                   WHERE AccountType = 'Bank'),
        m AS (SELECT max(d) md FROM b)
        SELECT (SELECT md FROM m) AS as_at, round(sum(amt),0) AS cash
        FROM b WHERE d = (SELECT md FROM m)""", max_rows=1)
    as_at, cash = r["rows"][0]
    return float(cash or 0), str(as_at)


def _ap_run_rate():
    r = lake_reader.query("""
        WITH w AS (
          SELECT date_trunc('week', CAST(fullypaidondate AS DATE)) wk,
                 sum(TRY_CAST(amountpaid AS DOUBLE)) paid
          FROM Invoices
          WHERE type = 0 AND fullypaidondate IS NOT NULL
            AND CAST(fullypaidondate AS DATE) >= current_date - 63
            AND CAST(fullypaidondate AS DATE) < date_trunc('week', current_date)
          GROUP BY 1)
        SELECT round(avg(paid),0) FROM w""", max_rows=1)
    v = r["rows"][0][0]
    return float(v or 0)


def _ap_due_by_week():
    """Open payables by due week from the latest aged snapshot; {} on failure
    (schema quirks in old partitions) - the run-rate then carries the model."""
    try:
        r = lake_reader.query("""
            WITH ap AS (
              SELECT CAST(Date AS DATE) snap, CAST(DueDate AS DATE) due,
                     TRY_CAST(NetAmountConverted AS DOUBLE) amt
              FROM XeroConsolidationAgedPayableDetailReportView)
            SELECT CASE WHEN due < date_trunc('week', current_date) + INTERVAL 7 DAY
                        THEN date_trunc('week', current_date)
                        ELSE date_trunc('week', due) END wk,
                   round(sum(amt),0)
            FROM ap
            WHERE snap = (SELECT max(snap) FROM ap)
              AND due < current_date + 98
            GROUP BY 1""", max_rows=30)
        return {str(row[0])[:10]: float(row[1] or 0) for row in r["rows"]}
    except Exception:
        LOG.exception("aged payables unavailable - using run-rate only")
        return {}


def build():
    lake_reader.sync(extra_tables=[
        "XeroConsolidationBalanceSheetGroupReportView", "Invoices",
        "XeroConsolidationAgedPayableDetailReportView"], log=LOG.info)

    receipts_factor = _env("CASH_RECEIPTS_FACTOR", 1.0)
    payroll_pct = _env("CASH_PAYROLL_PCT", 0.28)
    super_rate = _env("CASH_SUPER_RATE", 0.12)
    ato_monthly = _env("CASH_ATO_MONTHLY", 2900000)
    floor = _env("CASH_FLOOR", 3000000)

    opening, as_at = _opening_cash()
    ap_avg = _ap_run_rate()
    ap_due = _ap_due_by_week()
    fc = {r["week_start"]: float(r["forecast_sales"] or 0)
          for r in tk_db.get("v_forecast_weekly", {"select": "week_start,forecast_sales"})}

    start = _monday(dt.date.today())
    weeks, closing = [], opening
    for i in range(WEEKS):
        wk = start + dt.timedelta(weeks=i)
        days = [wk + dt.timedelta(days=d) for d in range(7)]
        sales = fc.get(wk.isoformat(), 0)
        receipts = sales * 1.10 * receipts_factor
        ap = max(ap_due.get(wk.isoformat(), 0), ap_avg)
        payroll = sales * payroll_pct * 0.80
        ato = ato_monthly if any(d.day == 21 for d in days) else 0
        # super: one monthly remittance (week of the 28th) ~= a month of wages
        sup = sales * 4.33 * payroll_pct * super_rate \
            if any(d.day == 28 for d in days) else 0
        net = receipts - ap - payroll - ato - sup
        closing += net
        weeks.append({"week_start": wk.isoformat(), "receipts": round(receipts),
                      "ap": round(ap), "payroll": round(payroll),
                      "ato_super": round(ato + sup), "net": round(net),
                      "closing": round(closing)})

    # working-capital markers for the cockpit (open AR/AP + rough DSO/DPO)
    try:
        wc = lake_reader.query("""
            SELECT round(sum(CASE WHEN type=1 THEN TRY_CAST(amountdue AS DOUBLE) END),0),
                   round(sum(CASE WHEN type=0 THEN TRY_CAST(amountdue AS DOUBLE) END),0)
            FROM Invoices WHERE status = 3""", max_rows=1)
        ar_open, ap_open = (float(v or 0) for v in wc["rows"][0])
    except Exception:
        LOG.exception("working capital query failed")
        ar_open = ap_open = 0
    weekly_sales_avg = (sum(fc.values()) / max(len(fc), 1)) if fc else 0
    dso = round(ar_open / (weekly_sales_avg / 7), 1) if weekly_sales_avg else None
    dpo = round(ap_open / (ap_avg / 7), 1) if ap_avg else None

    # accuracy tracking: last generation's week-1 forecast vs what actually
    # happened (receipts proxy = mart sales x1.1; AP = bills actually paid)
    accuracy = None
    try:
        last_mon = (dt.date.today() - dt.timedelta(days=dt.date.today().weekday() + 7))
        prev = tk_db.get("cash_forecast", {
            "week_start": "eq." + last_mon.isoformat(),
            "order": "generated_at.desc", "limit": "1",
            "select": "receipts,ap"})
        if prev:
            act = lake_reader.query("""
                SELECT round(sum(net_sales)*1.1,0),
                       (SELECT round(sum(TRY_CAST(amountpaid AS DOUBLE)),0) FROM Invoices
                        WHERE type=0 AND CAST(fullypaidondate AS DATE) >= DATE '%s'
                          AND CAST(fullypaidondate AS DATE) < DATE '%s' + INTERVAL 7 DAY)
                FROM mart_venue_daily
                WHERE CAST("date" AS DATE) >= DATE '%s'
                  AND CAST("date" AS DATE) < DATE '%s' + INTERVAL 7 DAY""" % (
                last_mon, last_mon, last_mon, last_mon), max_rows=1)
            act_rec, act_ap = (float(v or 0) for v in act["rows"][0])
            f_rec, f_ap = float(prev[0]["receipts"]), float(prev[0]["ap"])
            accuracy = {"week": last_mon.isoformat(),
                        "receipts_forecast": round(f_rec), "receipts_actual": round(act_rec),
                        "receipts_err_pct": round(100 * (f_rec - act_rec) / act_rec, 1) if act_rec else None,
                        "ap_forecast": round(f_ap), "ap_actual": round(act_ap),
                        "ap_err_pct": round(100 * (f_ap - act_ap) / act_ap, 1) if act_ap else None}
    except Exception:
        LOG.exception("accuracy tracking failed (non-fatal)")

    assumptions = {"opening_cash": round(opening), "bank_as_at": as_at,
                   "last_week_accuracy": accuracy,
                   "ar_open": round(ar_open), "ap_open": round(ap_open),
                   "dso_days": dso, "dpo_days": dpo,
                   "ap_weekly_run_rate": round(ap_avg),
                   "receipts_factor": receipts_factor, "payroll_pct": payroll_pct,
                   "super_rate": super_rate, "ato_monthly": ato_monthly,
                   "floor": floor, "aged_payables_used": bool(ap_due)}
    tk_db.insert("cash_forecast", [{**w, "assumptions": assumptions if i == 0 else None}
                                   for i, w in enumerate(weeks)])
    trough = min(weeks, key=lambda w: w["closing"])
    return {"weeks": weeks, "assumptions": assumptions, "opening": round(opening),
            "trough": trough}


def run(email=True):
    out = build()
    trough = out["trough"]
    floor = out["assumptions"]["floor"]
    breach = trough["closing"] < floor

    if breach:
        title = "[Cash] 13-week forecast dips below floor"
        if not tk_db.get("tasks", {"title": "eq." + title, "status": "neq.done",
                                   "select": "id", "limit": "1"}):
            admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                           "select": "id", "limit": "1"})
            due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=1))
            tk_db.insert("tasks", [{
                "project_id": DATA_WATCH_PROJECT,
                "title": title,
                "description": ("The 13-week cash model forecasts a trough of $%s in the "
                                "week of %s, below the $%s floor. Opening cash $%s (bank "
                                "as at %s). Review the assumptions (env CASH_*) and the "
                                "AP/ATO timing for that week.\nFirst steps:\n- Open the "
                                "cash card on /home\n- Check large AP due and ATO timing "
                                "that week\n- Consider deferring discretionary payments") % (
                    "{:,.0f}".format(trough["closing"]), trough["week_start"],
                    "{:,.0f}".format(floor), "{:,.0f}".format(out["opening"]),
                    out["assumptions"]["bank_as_at"]),
                "priority": "critical",
                "assignee_id": admin[0]["id"] if admin else None,
                "due_date": due.isoformat(),
                "source": "watcher",
            }])

    if email:
        rows = "".join(
            "<tr><td>%s</td><td align=right>%s</td><td align=right>%s</td>"
            "<td align=right>%s</td><td align=right>%s</td>"
            "<td align=right><b%s>%s</b></td></tr>" % (
                w["week_start"], "{:,.0f}".format(w["receipts"]),
                "{:,.0f}".format(w["ap"]), "{:,.0f}".format(w["payroll"]),
                "{:,.0f}".format(w["ato_super"]),
                " style='color:#b91c1c'" if w["closing"] < floor else "",
                "{:,.0f}".format(w["closing"]))
            for w in out["weeks"])
        a = out["assumptions"]
        html = ("<p><b>13-week cash forecast</b> — opening $%s (bank as at %s). "
                "Trough: <b>$%s</b> week of %s%s.</p>"
                "<table cellpadding=5 style='border-collapse:collapse;font-size:12.5px' border=1>"
                "<tr><th>Week</th><th>Receipts</th><th>AP</th><th>Payroll</th>"
                "<th>ATO/Super</th><th>Closing</th></tr>%s</table>"
                "<p style='color:#6b7280;font-size:11.5px'>v1 assumptions: receipts = base-case "
                "sales +GST x%.2f; payroll %.0f%% of sales (80%% cash); AP = max(due, $%s/wk "
                "run-rate); ATO $%s on the 21st; super monthly. Tune via CASH_* app settings.</p>") % (
            "{:,.0f}".format(out["opening"]), a["bank_as_at"],
            "{:,.0f}".format(trough["closing"]), trough["week_start"],
            " — <b style='color:#b91c1c'>BELOW FLOOR</b>" if breach else "",
            rows, a["receipts_factor"], a["payroll_pct"] * 100,
            "{:,.0f}".format(a["ap_weekly_run_rate"]), "{:,.0f}".format(a["ato_monthly"]))
        for adm in tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                          "select": "email"}):
            tk_email.send(adm["email"], "Cash 13-week — trough $%s (%s)" % (
                "{:,.0f}".format(trough["closing"]), trough["week_start"]), html)

    return {"weeks": len(out["weeks"]), "opening": out["opening"],
            "trough": trough, "breach": breach,
            "aged_payables_used": out["assumptions"]["aged_payables_used"]}
