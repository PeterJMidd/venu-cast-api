# -*- coding: utf-8 -*-
"""Nightly metric marts: curated summary tables built from the Parquet lake.

Marts are written back into the lake as ordinary tables (tables/mart_*/data.parquet
+ _meta.json), so the query app picks them up through the existing catalog with no
special handling. Definitions live in catalog/metrics.md (uploaded by build_all)
and are injected into the agent's system prompt.

v1 marts:
  mart_venue_daily    venue x day: net sales (excl GST/surcharge/gift cards), txns,
                      ATV, LFL vs same weekday LY, labour cost (+20% on-cost), labour %
  mart_venue_monthly  venue x month rollup of the daily mart
  mart_reviews_weekly venue x ISO week: rating avg/count, negative count, competitor avg

Sources are downloaded from blob to a temp dir and queried with DuckDB.
"""
import datetime as dt
import io
import json
import logging
import os
import tempfile

import duckdb
from azure.storage.blob import BlobServiceClient, ContentSettings

LOG = logging.getLogger("marts")
CONTAINER = os.environ.get("LAKE_CONTAINER", "datasights-lake")

SOURCES = ["PolygonRedcatNetSalesByStoreDailyView", "Redcat_PY_Sales_Mar_26",
           "Venue_Master", "RestokeLaborCost", "ReviewTrackersReviews",
           "ReviewTrackersCompetitorReviews"]

METRICS_MD = """# Yo-Chi metric dictionary (sanctioned definitions)

These definitions are authoritative - always use them, and prefer the mart_* tables
(pre-computed nightly with these rules) over re-deriving from raw tables.

- **Net sales**: AUD excl GST, excl POS surcharge and gift-card sales
  (PolygonRedcatNetSalesByStoreDailyView is already clean; do NOT re-divide by 1.01).
- **LFL (like-for-like)**: same store, this day vs the SAME WEEKDAY last year
  (date - 364 days). Store must trade on both days to count. lfl_pct = ts/ly - 1.
  LY history comes from Redcat_PY_Sales_Mar_26 (Oct-2024 onward).
- **ATV**: net sales / transaction count.
- **Labour cost**: wages from RestokeLaborCost x 1.20 (the 20% on-cost convention).
- **Labour %**: labour cost (incl on-cost) / net sales, same venue+day.
- **Venue dimension**: Venue_Master; join on store name match key (strip 'Yo-Chi',
  lowercase, alphanumeric only). Address_State = state; Franchise flags franchisees.
- **Reviews**: ReviewTrackersReviews (Google); negative = rating <= 3.
- **Fast Food Award items are NEVER 'breaches'** - say 'areas/items to review'.

Mart tables:
- mart_venue_daily(venue, store_id, state, date, net_sales, txns, atv,
  ly_net_sales, lfl_pct, labour_cost, labour_pct)
- mart_venue_monthly(venue, state, month, net_sales, txns, atv, ly_net_sales,
  lfl_pct, labour_cost, labour_pct, trading_days)
- mart_reviews_weekly(venue, state, week_start, reviews, avg_rating, negative_reviews,
  competitor_avg_rating)

RESTOKE PURCHASING DATA (from the Restoke Analytics API, refreshed 3 AM daily):
- restoke_purchasing, restoke_invoices, restoke_orders, restoke_orders_sent,
  restoke_orders_received, restoke_sales, restoke_recipes, restoke_unmatched_sales
  (one row per line item; numeric fields are STRINGS - use TRY_CAST(x AS DOUBLE)).
- mart_restoke_food_cost(venue, month, net_sales, purchases, food_cost_pct, ...):
  **Actual food cost %% = purchases / net sales** (the reliable operational metric).
- **COGS / theoretical / actual-vs-theoretical are NOT available**: Yo-Chi does not
  record stocktakes in Restoke, and self-serve Servings aren't recipe-costed. Do NOT
  attempt a theoretical food cost from restoke_sales.total_cost - it's ~99%% uncosted
  and misleading. Use mart_restoke_food_cost (purchases-based) for food cost questions.
  Food safety/procedures live in Celsi/Chi-Check, not Restoke.
"""

MART_SQL = {
    "mart_venue_daily": """
        WITH sales AS (
            SELECT StoreName AS venue,
                   regexp_replace(regexp_replace(lower(StoreName), 'yo-?chi', '', 'g'),
                                  '[^a-z0-9]', '', 'g') AS vkey,
                   StoreID AS store_id, TxnDate AS date,
                   SUM(NetSales) AS net_sales, SUM(TxnCount) AS txns
            FROM net_daily GROUP BY 1, 2, 3, 4
        ), ly AS (
            -- LY same weekday: prior-year daily history (Redcat_PY_Sales_Mar_26,
            -- Oct-2024..Apr-2026) unioned with the clean daily view for later dates;
            -- prefer the clean view when both exist
            SELECT vkey, date, MAX(ly_net_sales) AS ly_net_sales FROM (
                SELECT regexp_replace(regexp_replace(lower(StoreName), 'yo-?chi', '', 'g'),
                                      '[^a-z0-9]', '', 'g') AS vkey,
                       CAST(CAST(TxnDate AS DATE) + INTERVAL 364 DAY AS VARCHAR)[:10] AS date,
                       SUM(NetSales) AS ly_net_sales
                FROM py_hist GROUP BY 1, 2
                UNION ALL
                SELECT regexp_replace(regexp_replace(lower(StoreName), 'yo-?chi', '', 'g'),
                                      '[^a-z0-9]', '', 'g') AS vkey,
                       CAST(CAST(TxnDate AS DATE) + INTERVAL 364 DAY AS VARCHAR)[:10] AS date,
                       SUM(NetSales) AS ly_net_sales
                FROM net_daily GROUP BY 1, 2
            ) GROUP BY 1, 2
        ), labour AS (
            SELECT regexp_replace(regexp_replace(lower(venue), 'yo-?chi', '', 'g'),
                                  '[^a-z0-9]', '', 'g') AS vkey,
                   CAST(date AS VARCHAR)[:10] AS ld,
                   SUM(TRY_CAST(total AS DOUBLE)) * 1.20 AS labour_cost
            FROM labor GROUP BY 1, 2
        ), vm AS (
            SELECT regexp_replace(regexp_replace(lower(RC_Loc_Name), 'yo-?chi', '', 'g'),
                                  '[^a-z0-9]', '', 'g') AS vkey,
                   any_value(Address_State) AS state
            FROM venue_master GROUP BY 1
        )
        SELECT s.venue, s.store_id, vm.state, s.date,
               ROUND(s.net_sales, 2) AS net_sales, s.txns,
               ROUND(s.net_sales / NULLIF(s.txns, 0), 2) AS atv,
               ROUND(ly.ly_net_sales, 2) AS ly_net_sales,
               ROUND(s.net_sales / NULLIF(ly.ly_net_sales, 0) - 1, 4) AS lfl_pct,
               ROUND(l.labour_cost, 2) AS labour_cost,
               ROUND(l.labour_cost / NULLIF(s.net_sales, 0), 4) AS labour_pct
        FROM sales s
        LEFT JOIN ly ON ly.vkey = s.vkey AND ly.date = s.date
        LEFT JOIN labour l ON l.vkey = s.vkey AND l.ld = s.date
        LEFT JOIN vm ON vm.vkey = s.vkey
        ORDER BY s.date, s.venue
    """,
    "mart_venue_monthly": """
        SELECT venue, any_value(state) AS state, date[:7] AS month,
               ROUND(SUM(net_sales), 2) AS net_sales, SUM(txns) AS txns,
               ROUND(SUM(net_sales) / NULLIF(SUM(txns), 0), 2) AS atv,
               ROUND(SUM(ly_net_sales), 2) AS ly_net_sales,
               ROUND(SUM(net_sales) FILTER (ly_net_sales IS NOT NULL)
                     / NULLIF(SUM(ly_net_sales), 0) - 1, 4) AS lfl_pct,
               ROUND(SUM(labour_cost), 2) AS labour_cost,
               ROUND(SUM(labour_cost) / NULLIF(SUM(net_sales) FILTER (labour_cost IS NOT NULL), 0), 4) AS labour_pct,
               COUNT(*) AS trading_days
        FROM mart_venue_daily GROUP BY venue, month ORDER BY month, venue
    """,
    "mart_reviews_weekly": """
        WITH r AS (
            SELECT location_name AS venue, location_state AS state,
                   CAST(date_trunc('week', CAST(published_at AS DATE)) AS VARCHAR)[:10] AS week_start,
                   rating
            FROM reviews WHERE published_at IS NOT NULL
        ), c AS (
            SELECT CAST(date_trunc('week', CAST(published_at AS DATE)) AS VARCHAR)[:10] AS week_start,
                   AVG(rating) AS competitor_avg_rating
            FROM comp_reviews WHERE published_at IS NOT NULL GROUP BY 1
        )
        SELECT r.venue, any_value(r.state) AS state, r.week_start,
               COUNT(*) AS reviews, ROUND(AVG(r.rating), 2) AS avg_rating,
               COUNT(*) FILTER (r.rating <= 3) AS negative_reviews,
               ROUND(any_value(c.competitor_avg_rating), 2) AS competitor_avg_rating
        FROM r LEFT JOIN c ON c.week_start = r.week_start
        GROUP BY r.venue, r.week_start ORDER BY r.week_start, r.venue
    """,
}

VIEW_MAP = {"net_daily": "PolygonRedcatNetSalesByStoreDailyView",
            "py_hist": "Redcat_PY_Sales_Mar_26",
            "venue_master": "Venue_Master", "labor": "RestokeLaborCost",
            "reviews": "ReviewTrackersReviews",
            "comp_reviews": "ReviewTrackersCompetitorReviews"}


def _svc():
    return BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])


def build_all():
    svc = _svc()
    cc = svc.get_container_client(CONTAINER)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        # pull source tables local
        for src in SOURCES:
            d = os.path.join(tmp, src)
            os.makedirs(d, exist_ok=True)
            n = 0
            for b in cc.list_blobs(name_starts_with="tables/%s/" % src):
                if b.name.endswith(".parquet"):
                    with open(os.path.join(d, os.path.basename(b.name)), "wb") as f:
                        cc.download_blob(b.name).readinto(f)
                    n += 1
            if n == 0:
                raise RuntimeError("no parquet for source %s" % src)
        con = duckdb.connect(":memory:")
        for alias, src in VIEW_MAP.items():
            con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s/*.parquet', union_by_name=true)"
                        % (alias, os.path.join(tmp, src).replace("\\", "/")))
        for mart, sql in MART_SQL.items():
            con.execute("CREATE OR REPLACE TABLE %s AS %s" % (mart, sql))
            out = os.path.join(tmp, mart + ".parquet")
            con.execute("COPY %s TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)"
                        % (mart, out.replace("\\", "/")))
            with open(out, "rb") as f:
                cc.upload_blob("tables/%s/data.parquet" % mart, f, overwrite=True)
            cols = [r[0] for r in con.execute("DESCRIBE %s" % mart).fetchall()]
            rows = con.execute("SELECT COUNT(*) FROM %s" % mart).fetchone()[0]
            meta = {"view": mart, "mode": "snapshot", "files": ["data.parquet"],
                    "columns": cols, "rows": rows, "source_rows": rows,
                    "note": "nightly metric mart - see catalog/metrics.md for definitions",
                    "exported_at": dt.datetime.now().isoformat()}
            cc.upload_blob("tables/%s/_meta.json" % mart, json.dumps(meta, indent=1),
                           overwrite=True,
                           content_settings=ContentSettings(content_type="application/json"))
            results[mart] = rows
            LOG.info("mart %s: %d rows", mart, rows)
        con.close()
    cc.upload_blob("catalog/metrics.md", METRICS_MD, overwrite=True,
                   content_settings=ContentSettings(content_type="text/markdown"))
    # rebuild the global catalog so the new marts appear
    import lake_export
    n = lake_export.rebuild_catalog(cc)
    return {"marts": results, "catalog_tables": n}
