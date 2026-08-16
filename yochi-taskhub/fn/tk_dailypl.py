# -*- coding: utf-8 -*-
"""Continuous close: daily estimated MTD P&L (the 'P&L pulse'). Month-end
becomes confirming a number watched all month.
  revenue  = mart net sales MTD (actual)
  COGS     = Restoke purchases received MTD (actual, purchases basis)
  labour   = mart labour where healthy; scaled up for missing coverage
  opex     = run-rate: trailing-3-complete-months GL expenses excl labour/COGS,
             per-day x days elapsed
  est EBITDA = revenue - COGS - labour - opex   (estimate, clearly labelled)
Stores one taskapp.pl_pulse row per day; the cockpit renders the latest."""
import datetime as dt
import logging

import tk_db
import lake_reader

LOG = logging.getLogger("tk_dailypl")


def run():
    lake_reader.sync(extra_tables=["XeroAccountTransactionsMasterView",
                                   "restoke_purchasing"], log=LOG.info)
    r = lake_reader.query("""
        WITH mx AS (SELECT max(CAST("date" AS DATE)) m FROM mart_venue_daily),
        bounds AS (SELECT date_trunc('month',(SELECT m FROM mx)) m0, (SELECT m FROM mx) m1),
        rev AS (SELECT sum(net_sales) v,
                       sum(CASE WHEN labour_cost>0 AND labour_cost<net_sales THEN labour_cost END) lab,
                       sum(CASE WHEN labour_cost>0 AND labour_cost<net_sales THEN net_sales END) lab_cov_sales
                FROM mart_venue_daily
                WHERE CAST("date" AS DATE) BETWEEN (SELECT m0 FROM bounds) AND (SELECT m1 FROM bounds)),
        cogs AS (SELECT sum(TRY_CAST(total AS DOUBLE)) v FROM restoke_purchasing
                 WHERE CAST(COALESCE(received_date,date) AS DATE)
                       BETWEEN (SELECT m0 FROM bounds) AND (SELECT m1 FROM bounds)),
        opex3 AS (SELECT sum(TRY_CAST("Net Amount" AS DOUBLE)) v
                  FROM XeroAccountTransactionsMasterView
                  WHERE lower("Account Type") LIKE '%expense%'
                    AND lower("Account") NOT LIKE '%wage%'
                    AND lower("Account") NOT LIKE '%salar%'
                    AND lower("Account") NOT LIKE '%superann%'
                    AND lower("Account") NOT LIKE '%purchase%'
                    AND lower("Account") NOT LIKE '%cost of%'
                    AND CAST("Date" AS DATE) >= date_trunc('month',(SELECT m0 FROM bounds)) - INTERVAL 3 MONTH
                    AND CAST("Date" AS DATE) < (SELECT m0 FROM bounds))
        SELECT (SELECT m1 FROM bounds) AS asof,
               CAST((SELECT m0 FROM bounds) AS DATE) AS m0,
               round((SELECT v FROM rev),0) revenue,
               round((SELECT v FROM cogs),0) cogs,
               round((SELECT lab FROM rev),0) labour_seen,
               round((SELECT lab_cov_sales FROM rev),0) labour_cov_sales,
               round((SELECT v FROM opex3),0) opex3m""", max_rows=1)
    (asof, m0, revenue, cogs, labour_seen, labour_cov_sales, opex3m) = r["rows"][0]
    revenue = float(revenue or 0)
    cogs = float(cogs or 0)
    asof_d = dt.date.fromisoformat(str(asof)[:10])
    m0_d = dt.date.fromisoformat(str(m0)[:10])
    days_mtd = (asof_d - m0_d).days + 1

    # labour: scale healthy-coverage labour% onto full revenue; else 28% assumption
    cov = float(labour_cov_sales or 0)
    if cov > revenue * 0.4 and labour_seen:
        labour_pct = float(labour_seen) / cov
        labour_est, labour_basis = revenue * labour_pct, "actual %.1f%% scaled" % (labour_pct * 100)
    else:
        labour_est, labour_basis = revenue * 0.28, "28% assumption (feed coverage low)"

    days_3m = ((m0_d - dt.timedelta(days=1)) - (m0_d - dt.timedelta(days=1)).replace(day=1)
               ).days  # not used; keep simple below
    opex_per_day = float(opex3m or 0) / 91.0
    opex_est = opex_per_day * days_mtd
    ebitda_est = revenue - cogs - labour_est - opex_est

    budget_sales = None
    try:
        import tk_budget
        bud = tk_budget.daily_group()
        days = [(m0_d + dt.timedelta(days=i)).isoformat() for i in range(days_mtd)]
        vals = [bud.get(d) for d in days]
        if all(v is not None for v in vals):
            budget_sales = round(sum(vals))
    except Exception:
        LOG.exception("budget for pulse failed")

    data = {"asof": asof_d.isoformat(), "days_mtd": days_mtd,
            "revenue": round(revenue), "cogs": round(cogs),
            "cogs_pct": round(100 * cogs / revenue, 1) if revenue else None,
            "labour_est": round(labour_est), "labour_basis": labour_basis,
            "labour_pct": round(100 * labour_est / revenue, 1) if revenue else None,
            "opex_est": round(opex_est),
            "ebitda_est": round(ebitda_est),
            "ebitda_pct": round(100 * ebitda_est / revenue, 1) if revenue else None,
            "budget_sales": budget_sales,
            "sales_vs_budget_pct": (round(100 * (revenue - budget_sales) / budget_sales, 1)
                                    if budget_sales else None)}
    # replace today's row (re-runs during the day keep the freshest numbers)
    import urllib.parse
    import urllib.request
    import os
    import json as _json
    qs = urllib.parse.urlencode({"day": "eq." + dt.date.today().isoformat()})
    req = urllib.request.Request(
        "%s/rest/v1/pl_pulse?%s" % (os.environ["SUPABASE_URL"].rstrip("/"), qs),
        headers={"apikey": os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                 "Authorization": "Bearer " + os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                 "Accept-Profile": "taskapp", "Content-Profile": "taskapp"},
        method="DELETE")
    urllib.request.urlopen(req, timeout=30).read()
    tk_db.insert("pl_pulse", [{"day": dt.date.today().isoformat(), "data": data}])
    return data
