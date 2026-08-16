# -*- coding: utf-8 -*-
"""MMR pack auto-assembly: on WD+3 each month, build the finance section of
the Monthly Management Report as a PDF - prior-month P&L by account group,
state trading + LFL, venue league, food cost, cash position, AI-drafted
commentary ([CHECK]-flagged where a human must verify) - email it and attach
it to a review task. The draft arrives before anyone asks for it."""
import datetime as dt
import io
import json
import logging

import tk_ai
import tk_calendar
import tk_db
import tk_email
import lake_reader

LOG = logging.getLogger("tk_mmr")

FINCTRL_CLOSE = "aaaaaaaa-0000-0000-0000-000000000001"


def build():
    lake_reader.sync(extra_tables=["XeroConsolidationProfitAndLossGroupReportView"],
                     log=LOG.info)
    today = dt.date.today()
    m0 = today.replace(day=1) - dt.timedelta(days=1)
    m0 = m0.replace(day=1)  # first day of prior month
    m_label = m0.strftime("%B %Y")

    pl = lake_reader.query("""
        SELECT AccountGroupName, round(sum(TRY_CAST(Amount AS DOUBLE)),0) amt
        FROM XeroConsolidationProfitAndLossGroupReportView
        WHERE CAST(Date AS DATE) >= DATE '%s'
          AND CAST(Date AS DATE) < DATE '%s' + INTERVAL 1 MONTH
        GROUP BY 1 ORDER BY 2 DESC""" % (m0, m0), max_rows=40)
    states = lake_reader.query("""
        SELECT state, round(sum(net_sales),0) sales, round(avg(lfl_pct)*100,1) lfl
        FROM mart_venue_daily
        WHERE CAST("date" AS DATE) >= DATE '%s'
          AND CAST("date" AS DATE) < DATE '%s' + INTERVAL 1 MONTH
        GROUP BY 1 ORDER BY 2 DESC""" % (m0, m0), max_rows=10)
    league = lake_reader.query("""
        SELECT venue, round(sum(net_sales),0) sales, round(avg(lfl_pct)*100,1) lfl
        FROM mart_venue_daily
        WHERE CAST("date" AS DATE) >= DATE '%s'
          AND CAST("date" AS DATE) < DATE '%s' + INTERVAL 1 MONTH
        GROUP BY 1 ORDER BY 2 DESC""" % (m0, m0), max_rows=100)
    food = lake_reader.query("""
        SELECT venue, food_cost_pct FROM mart_restoke_food_cost
        WHERE month = '%s' ORDER BY TRY_CAST(food_cost_pct AS DOUBLE) DESC
        LIMIT 100""" % m0.strftime("%Y-%m"), max_rows=100)

    cash = tk_db.get("cash_forecast", {"order": "generated_at.desc,week_start",
                                       "limit": "13", "select": "week_start,closing,assumptions"})
    cash_line = ""
    if cash:
        a = next((r["assumptions"] for r in cash if r.get("assumptions")), {})
        trough = min(cash, key=lambda w: float(w["closing"]))
        cash_line = "Opening cash $%s; 13-week trough $%s (week of %s)." % (
            a.get("opening_cash"), trough["closing"], trough["week_start"])

    numbers = {"month": m_label,
               "pl_by_group": pl["rows"], "states": states["rows"],
               "top10": league["rows"][:10], "bottom10": league["rows"][-10:],
               "food_worst10": food["rows"][:10], "cash": cash_line}
    commentary = tk_ai.text(
        "Draft the finance commentary for Yo-Chi's Monthly Management Report. Board "
        "tone, factual, <=350 words: 1) Trading (group + states + LFL), 2) P&L shape "
        "(biggest account-group movements), 3) Margin watch (food cost outliers), "
        "4) Cash, 5) Watch items. Mark anything needing a human number-check with "
        "[CHECK]. Money is AUD net of GST. Award items are 'areas to review'.",
        json.dumps(numbers, default=str)[:30000], max_tokens=1200)

    esc = lambda s: str(s or "").replace("&", "&amp;").replace("<", "&lt;")
    def tbl(headers, rows):
        return ("<table><tr>%s</tr>%s</table>") % (
            "".join("<th>%s</th>" % h for h in headers),
            "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % esc(c) for c in r)
                    for r in rows))
    html = """<html><head><style>
      @page { size: A4; margin: 1.6cm; }
      body { font-family: Helvetica, Arial, sans-serif; color:#1f2937; font-size:9pt; }
      h1 { font-size:13pt; color:#1d683d; border-bottom:2px solid #dbf2e4; margin:16px 0 6px; }
      table { width:100%%; border-collapse:collapse; font-size:8pt; margin-bottom:8px; }
      th { background:#f0faf4; text-align:left; padding:3px 5px; border:1px solid #e3e7ee; }
      td { padding:3px 5px; border:1px solid #e3e7ee; }
      .cover { background:#1d683d; color:#fff; padding:16px 20px; border-radius:8px; }
      .comm { white-space:pre-wrap; background:#f9fafb; padding:10px; border-radius:6px; }
    </style></head><body>
    <div class="cover"><div style="font-size:16pt;font-weight:bold">MMR — Finance section (draft)</div>
    <div style="font-size:9pt">%s · auto-assembled %s · review [CHECK] items before circulation</div></div>
    <h1>Commentary (draft)</h1><div class="comm">%s</div>
    <h1>P&amp;L by account group — %s</h1>%s
    <h1>Trading by state</h1>%s
    <h1>Venue league — top 10</h1>%s
    <h1>Venue league — bottom 10</h1>%s
    <h1>Food cost — highest 10 (purchases basis)</h1>%s
    <h1>Cash</h1><p>%s</p>
    </body></html>""" % (
        m_label, dt.date.today().strftime("%d %b %Y"), esc(commentary),
        m_label, tbl(["Account group", "$"], pl["rows"]),
        tbl(["State", "Sales $", "LFL %"], states["rows"]),
        tbl(["Venue", "Sales $", "LFL %"], league["rows"][:10]),
        tbl(["Venue", "Sales $", "LFL %"], league["rows"][-10:]),
        tbl(["Venue", "Food %"], food["rows"][:10]),
        esc(cash_line or "No cash forecast generation found."))
    from xhtml2pdf import pisa
    buf = io.BytesIO()
    if pisa.CreatePDF(html, dest=buf, encoding="utf-8").err:
        raise RuntimeError("MMR pdf failed")
    return buf.getvalue(), m_label


def run(force=False, email=True):
    today = dt.date.today()
    if not force and today != tk_calendar.business_day_of_month(today.year, today.month, 3):
        return {"skipped": "runs on WD+3"}
    pdf, m_label = build()
    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id,email"})
    title = "MMR finance pack draft — %s" % m_label
    task_id = None
    if not tk_db.get("tasks", {"title": "eq." + title, "select": "id", "limit": "1"}):
        rows = tk_db.insert("tasks", [{
            "project_id": FINCTRL_CLOSE, "title": title,
            "description": ("Auto-assembled MMR finance section for %s (attached): "
                            "commentary draft, P&L by group, state trading, venue "
                            "league, food cost, cash. Review the [CHECK] items, edit, "
                            "and drop into the MMR pack." % m_label),
            "priority": "high", "assignee_id": admin[0]["id"] if admin else None,
            "due_date": tk_calendar.roll_forward(
                today + dt.timedelta(days=2)).isoformat(),
            "source": "watcher"}], returning=True)
        task_id = rows[0]["id"]
        try:
            import tk_agent
            tk_agent._upload(task_id, "MMR_finance_%s.pdf" % m_label.replace(" ", "_"),
                             pdf, "application/pdf", admin[0]["id"] if admin else None)
        except Exception:
            LOG.exception("MMR attachment failed")
    if email:
        for a in admin:
            tk_email.send(a["email"], "MMR finance pack draft — %s" % m_label,
                          "<p>The auto-assembled MMR finance section for <b>%s</b> is "
                          "attached to the task '%s' in TaskHub (PDF). Review the "
                          "[CHECK] items before circulating.</p>" % (m_label, title))
    return {"month": m_label, "task_id": task_id, "pdf_bytes": len(pdf)}
