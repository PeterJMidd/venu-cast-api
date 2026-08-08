# -*- coding: utf-8 -*-
"""Weekly spend-control screen over the purchasing and POS-exception line items:
  A. price CREEP  - SKU median price last complete week vs trailing 8-week median
  B. price DISPERSION - same SKU, same week, spread across venues
  C. ingredient INTENSITY - venue spend on a SKU per $ of sales vs peer median
  D. POS integrity - venue void/discount/refund $ vs sales, plus top operators
     (language: 'to review' - a heuristic screen, never an accusation)
Findings are emailed to admins; the top price alerts become tasks. Credit and
rebate lines (unit price < 30%% of the SKU median) are excluded from A/B."""
import datetime as dt
import logging

import tk_calendar
import tk_db
import tk_email
import lake_reader

LOG = logging.getLogger("tk_price")

DATA_WATCH_PROJECT = "aaaaaaaa-0000-0000-0000-000000000005"
CREEP_PCT = 8.0        # flag price up more than this vs trailing median
MIN_WEEK_SPEND = 2000  # ignore tiny SKUs
DISPERSION_RATIO = 1.25
MAX_TASKS = 3


def _monday(d):
    return d - dt.timedelta(days=d.weekday())


def build():
    lake_reader.sync(extra_tables=["restoke_purchasing",
                                   "PolygonRedcatExceptionsReport"], log=LOG.info)
    today = dt.date.today()
    this_mon = _monday(today)
    wk = this_mon - dt.timedelta(days=7)

    creep = lake_reader.query("""
        WITH p AS (
          SELECT supplier, product, uom,
                 date_trunc('week', CAST(COALESCE(received_date,date) AS DATE)) w,
                 TRY_CAST(unit_price AS DOUBLE) up, TRY_CAST(total AS DOUBLE) amt
          FROM restoke_purchasing
          WHERE CAST(COALESCE(received_date,date) AS DATE) >= DATE '%s' - INTERVAL 63 DAY
            AND CAST(COALESCE(received_date,date) AS DATE) < DATE '%s'
            AND TRY_CAST(unit_price AS DOUBLE) > 0),
        med AS (SELECT supplier, product, uom, median(up) m FROM p GROUP BY 1,2,3),
        clean AS (SELECT p.* FROM p JOIN med USING (supplier, product, uom)
                  WHERE p.up > med.m * 0.3),
        cur AS (SELECT supplier, product, uom, median(up) cur_p, sum(amt) spend
                FROM clean WHERE w = DATE '%s' GROUP BY 1,2,3),
        base AS (SELECT supplier, product, uom, median(up) base_p
                 FROM clean WHERE w < DATE '%s' GROUP BY 1,2,3)
        SELECT c.supplier, c.product, c.uom, round(b.base_p,2), round(c.cur_p,2),
               round(100.0*(c.cur_p/b.base_p - 1),1) AS pct, round(c.spend,0)
        FROM cur c JOIN base b USING (supplier, product, uom)
        WHERE c.spend > %d AND c.cur_p > b.base_p * (1 + %f/100.0)
        ORDER BY c.spend * (c.cur_p/b.base_p - 1) DESC LIMIT 15""" % (
        wk, this_mon, wk, wk, MIN_WEEK_SPEND, CREEP_PCT), max_rows=15)

    disp = lake_reader.query("""
        WITH p AS (
          SELECT supplier, product, uom, venue,
                 TRY_CAST(unit_price AS DOUBLE) up, TRY_CAST(total AS DOUBLE) amt
          FROM restoke_purchasing
          WHERE CAST(COALESCE(received_date,date) AS DATE) >= DATE '%s'
            AND CAST(COALESCE(received_date,date) AS DATE) < DATE '%s'
            AND TRY_CAST(unit_price AS DOUBLE) > 0),
        med AS (SELECT supplier, product, uom, median(up) m FROM p GROUP BY 1,2,3),
        clean AS (SELECT p.* FROM p JOIN med USING (supplier, product, uom)
                  WHERE p.up > med.m * 0.3)
        SELECT supplier, product, uom, count(DISTINCT venue) venues,
               round(median(up),2) med_p, round(max(up),2) max_p,
               round(sum(amt),0) spend
        FROM clean GROUP BY 1,2,3
        HAVING count(DISTINCT venue) >= 10 AND max(up) > median(up) * %f
           AND sum(amt) > 3000
        ORDER BY spend DESC LIMIT 10""" % (wk, this_mon, DISPERSION_RATIO),
        max_rows=10)

    intensity = lake_reader.query("""
        WITH s AS (SELECT venue, sum(net_sales) sales FROM mart_venue_daily
                   WHERE CAST("date" AS DATE) >= DATE '%s' - INTERVAL 21 DAY
                     AND CAST("date" AS DATE) < DATE '%s' GROUP BY 1),
        pp AS (SELECT venue, product, sum(TRY_CAST(total AS DOUBLE)) spend
               FROM restoke_purchasing
               WHERE CAST(COALESCE(received_date,date) AS DATE) >= DATE '%s' - INTERVAL 21 DAY
                 AND CAST(COALESCE(received_date,date) AS DATE) < DATE '%s'
               GROUP BY 1,2 HAVING sum(TRY_CAST(total AS DOUBLE)) > 1000),
        norm AS (SELECT regexp_replace(regexp_replace(lower(regexp_replace(regexp_replace(pp.venue,'[^A-Za-z0-9]','','g'),'^[Yy]o[-]?[Cc]hi','')),'street$','st'),'square$','') nv,
                        pp.product, pp.spend FROM pp),
        sn AS (SELECT regexp_replace(regexp_replace(lower(regexp_replace(regexp_replace(venue,'[^A-Za-z0-9]','','g'),'^[Yy]o[-]?[Cc]hi','')),'street$','st'),'square$','') nv,
                      venue, sales FROM s WHERE sales > 50000),
        j AS (SELECT sn.venue, norm.product, norm.spend / sn.sales AS ratio
              FROM norm JOIN sn USING (nv)),
        peer AS (SELECT product, median(ratio) med_r, count(*) n FROM j GROUP BY 1
                 HAVING count(*) >= 15)
        SELECT j.venue, j.product, round(1000*j.ratio,1) AS per_1k_sales,
               round(1000*peer.med_r,1) AS peer_per_1k,
               round(j.ratio/peer.med_r,1) AS x_peer
        FROM j JOIN peer USING (product)
        WHERE j.ratio > peer.med_r * 1.5
        ORDER BY j.ratio/peer.med_r DESC LIMIT 12""" % (wk, this_mon, wk, this_mon),
        max_rows=12)

    pos = lake_reader.query("""
        WITH e AS (SELECT Store, "User", Severity, TRY_CAST(Amount AS DOUBLE) amt
                   FROM PolygonRedcatExceptionsReport
                   WHERE CAST(Date AS DATE) >= DATE '%s' AND CAST(Date AS DATE) < DATE '%s'),
        s AS (SELECT venue, sum(net_sales) sales FROM mart_venue_daily
              WHERE CAST("date" AS DATE) >= DATE '%s' AND CAST("date" AS DATE) < DATE '%s'
              GROUP BY 1),
        by_store AS (SELECT e.Store, round(sum(abs(e.amt)),0) exc, count(*) n FROM e GROUP BY 1)
        SELECT b.Store, b.exc, b.n, round(s.sales,0) sales,
               round(100.0*b.exc/nullif(s.sales,0),2) pct
        FROM by_store b JOIN s ON s.venue = b.Store
        WHERE s.sales > 20000
        ORDER BY b.exc/nullif(s.sales,0) DESC LIMIT 10""" % (wk, this_mon, wk, this_mon),
        max_rows=10)

    return {"week": wk.isoformat(),
            "creep": creep["rows"], "dispersion": disp["rows"],
            "intensity": intensity["rows"], "pos": pos["rows"]}


def run(email=True):
    out = build()
    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id,email"})
    created = 0
    due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=3))
    for r in out["creep"][:MAX_TASKS]:
        supplier, product, uom, base_p, cur_p, pct, spend = r
        title = "[Price] %s — %s +%s%%" % (product[:60], supplier[:30], pct)
        if tk_db.get("tasks", {"title": "eq." + title, "select": "id", "limit": "1"}):
            continue
        tk_db.insert("tasks", [{
            "project_id": DATA_WATCH_PROJECT, "title": title,
            "description": ("Price integrity screen (week of %s): %s from %s moved from "
                            "$%s to $%s per %s (+%s%%) vs the trailing 8-week median, on "
                            "$%s of weekly spend.\nFirst steps:\n- Check the supplier "
                            "notice/contract for an agreed increase\n- If unagreed, query "
                            "the supplier\n- If agreed, note it for the next COGS review") % (
                out["week"], product, supplier, base_p, cur_p, uom, pct,
                "{:,.0f}".format(spend)),
            "priority": "high",
            "assignee_id": admin[0]["id"] if admin else None,
            "due_date": due.isoformat(), "source": "watcher",
        }])
        created += 1

    if email:
        def table(headers, rows, fmt):
            if not rows:
                return "<p style='color:#9ca3af;font-size:12px'>Nothing flagged.</p>"
            return ("<table cellpadding=4 border=1 style='border-collapse:collapse;"
                    "font-size:11.5px'><tr>%s</tr>%s</table>") % (
                "".join("<th>%s</th>" % h for h in headers),
                "".join("<tr>%s</tr>" % fmt(r) for r in rows))
        td = lambda *vs: "".join("<td>%s</td>" % v for v in vs)
        html = ("<p><b>Spend control screen</b> — week of %s.</p>"
                "<p><b>A. Price creep vs 8-week median</b> (%d task(s) raised)</p>%s"
                "<p><b>B. Cross-venue price dispersion (same SKU, same week)</b></p>%s"
                "<p><b>C. Ingredient intensity — spend per $1k sales vs peer median (4 wks)</b></p>%s"
                "<p><b>D. POS exceptions (voids/discounts/refunds) as %% of sales — items to review</b></p>%s"
                "<p style='color:#6b7280;font-size:11px'>Heuristic screens over Restoke "
                "purchasing lines and Redcat POS exceptions. Credit/rebate lines excluded. "
                "All items are 'to review', not conclusions.</p>") % (
            out["week"], created,
            table(["Supplier", "Product", "UoM", "Was", "Now", "+%", "Wk spend"],
                  out["creep"], lambda r: td(*r)),
            table(["Supplier", "Product", "UoM", "Venues", "Median", "Max", "Wk spend"],
                  out["dispersion"], lambda r: td(*r)),
            table(["Venue", "Product", "$/1k sales", "Peer", "× peer"],
                  out["intensity"], lambda r: td(*r)),
            table(["Venue", "Exceptions $", "Count", "Sales", "%"],
                  out["pos"], lambda r: td(*r)))
        for a in admin:
            tk_email.send(a["email"], "Spend control — wk %s: %d creep, %d dispersion, "
                          "%d intensity flags" % (out["week"], len(out["creep"]),
                                                  len(out["dispersion"]),
                                                  len(out["intensity"])), html)
    return {"week": out["week"], "creep": len(out["creep"]),
            "dispersion": len(out["dispersion"]), "intensity": len(out["intensity"]),
            "pos_rows": len(out["pos"]), "tasks_created": created}
