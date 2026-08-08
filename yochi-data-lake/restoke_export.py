# -*- coding: utf-8 -*-
"""Extract all Restoke Analytics API reports to Azure Blob (container: restoke-export).

Restoke exposes row-level JSON reports at analytics.restoke.ai/api/v1/reports/<name>/
scoped by an HQ Restaurant API key + restaurant_ids + a calendar date range. This
pulls every report for every venue, month by month, and writes both raw JSON and
columnar Parquet so the data is easy to re-extract or query.

Layout (container restoke-export):
  reports/<report>/<YYYY-MM>.parquet      columnar, one file per report-month
  reports/<report>/<YYYY-MM>.json         raw API rows (audit / exact re-extract)
  venues.json                             id -> name map (live)
  catalog.json                            manifest: reports, months, row counts, freshness

Design notes:
  - restaurant_ids is REQUIRED (HQ key with no ids returns 0 rows); we batch all
    real venue ids (TEST venues excluded) BATCH_VENUES at a time and concatenate.
  - Monthly windows: transactional reports (invoices/orders/purchasing/labor) return
    all rows dated in the window; aggregate reports (sales/recipes/unmatched) return
    the period aggregate for that window - both are tagged with query_month.
  - Resumable: a month already written is skipped unless it's within REFRESH_MONTHS
    of today (recent months can still change as invoices are approved).

Config (app settings): RESTOKE_KEY, BLOB_CONNECTION_STRING, RESTOKE_CONTAINER
(default restoke-export), RESTOKE_START_MONTH (default 2024-01), REFRESH_MONTHS
(default 2), RESTOKE_BATCH_VENUES (default 25).
"""
import calendar
import datetime as dt
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient, ContentSettings

LOG = logging.getLogger("restoke_export")
API = "https://analytics.restoke.ai/api/v1"
REPORTS = ["invoices", "labor-cost", "orders-received", "orders-sent", "orders",
           "purchasing", "recipes", "sales", "unmatched-sales"]
CONTAINER = os.environ.get("RESTOKE_CONTAINER", "restoke-export")
BATCH_VENUES = int(os.environ.get("RESTOKE_BATCH_VENUES", "25"))
REFRESH_MONTHS = int(os.environ.get("REFRESH_MONTHS", "2"))
START_MONTH = os.environ.get("RESTOKE_START_MONTH", "2024-01")
HTTP_TIMEOUT = 180


def _key():
    k = os.environ.get("RESTOKE_KEY")
    if not k:
        raise RuntimeError("RESTOKE_KEY not configured")
    return k


def _get(path, params):
    url = "%s/%s?%s" % (API, path.strip("/"), urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={"X-Restaurant-Key": _key(),
                                               "Accept": "application/json"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last


def list_venues(include_test=False):
    data = _get("venues/", {})
    venues = data if isinstance(data, list) else data.get("venues", [])
    out = []
    for v in venues:
        name = str(v.get("name", ""))
        if not include_test and name.upper().startswith("TEST"):
            continue
        out.append({"id": v["id"], "name": name})
    return out


def _svc():
    return BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])


def _months(start_month, end_month):
    y, m = int(start_month[:4]), int(start_month[5:7])
    ey, em = int(end_month[:4]), int(end_month[5:7])
    while (y, m) <= (ey, em):
        last = calendar.monthrange(y, m)[1]
        yield ("%04d-%02d" % (y, m), "%04d-%02d-01" % (y, m), "%04d-%02d-%02d" % (y, m, last))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _fetch_report_month(report, venue_ids, start_date, end_date):
    rows = []
    for i in range(0, len(venue_ids), BATCH_VENUES):
        batch = venue_ids[i:i + BATCH_VENUES]
        data = _get("reports/%s/" % report, {
            "startDate": start_date, "endDate": end_date,
            "restaurant_ids": ",".join(str(v) for v in batch)})
        rows.extend(data.get("rows", []) if isinstance(data, dict) else [])
    return rows


def _upload(cc, path, data, ctype):
    cc.upload_blob(path, data, overwrite=True,
                   content_settings=ContentSettings(content_type=ctype))


def run(start_month=None, end_month=None, only=None):
    """Export reports for [start_month, end_month]. Returns a summary dict."""
    svc = _svc()
    try:
        svc.create_container(CONTAINER)
    except Exception:
        pass
    cc = svc.get_container_client(CONTAINER)

    venues = list_venues()
    venue_ids = [v["id"] for v in venues]
    _upload(cc, "venues.json", json.dumps(venues, indent=1), "application/json")

    today = dt.date.today()
    start_month = start_month or START_MONTH
    end_month = end_month or today.strftime("%Y-%m")
    reports = [r for r in REPORTS if not only or r in only]
    # months within REFRESH_MONTHS of now are always re-pulled
    refresh_from = (today.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")
    recent = set()
    yy, mm = today.year, today.month
    for _ in range(REFRESH_MONTHS):
        recent.add("%04d-%02d" % (yy, mm))
        yy, mm = (yy - 1, 12) if mm == 1 else (yy, mm - 1)

    existing = {b.name for b in cc.list_blobs(name_starts_with="reports/")}
    summary = {"venues": len(venues), "reports": {}, "errors": []}
    t0 = time.time()

    for report in reports:
        totals = {"months": 0, "rows": 0, "skipped": 0}
        for label, sd, ed in _months(start_month, end_month):
            ppath = "reports/%s/%s.parquet" % (report, label)
            if ppath in existing and label not in recent:
                totals["skipped"] += 1
                continue
            try:
                rows = _fetch_report_month(report, venue_ids, sd, ed)
            except Exception as e:
                summary["errors"].append("%s %s: %s" % (report, label, str(e)[:200]))
                LOG.exception("fetch failed %s %s", report, label)
                continue
            if not rows:
                continue
            for r in rows:
                r["query_month"] = label
            df = pd.DataFrame(rows)
            buf = io.BytesIO()
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
            _upload(cc, ppath, buf.getvalue(), "application/octet-stream")
            _upload(cc, "reports/%s/%s.json" % (report, label),
                    json.dumps(rows, default=str), "application/json")
            totals["months"] += 1
            totals["rows"] += len(rows)
            LOG.info("restoke %s %s: %d rows", report, label, len(rows))
        summary["reports"][report] = totals

    catalog = {
        "source": "Restoke Analytics API (analytics.restoke.ai/api/v1)",
        "generated_at": dt.datetime.now().isoformat(),
        "container": CONTAINER, "venues": len(venues),
        "range": {"start": start_month, "end": end_month},
        "reports": summary["reports"],
        "layout": "reports/<report>/<YYYY-MM>.parquet (+ .json raw)",
        "errors": summary["errors"],
    }
    _upload(cc, "catalog.json", json.dumps(catalog, indent=1), "application/json")

    # register the reports as agent-queryable lake tables + build the food-cost mart
    try:
        summary["lake"] = register_lake_tables_and_marts(svc, cc)
    except Exception as e:
        summary["errors"].append("register/mart: %s" % str(e)[:200])
        LOG.exception("register_lake_tables_and_marts failed")

    summary["secs"] = round(time.time() - t0)
    summary["finished_at"] = dt.datetime.now().isoformat()
    return summary


# ---- make Restoke data queryable in the agent + food-cost mart ----

def _table_name(report):
    return "restoke_" + report.replace("-", "_")


def register_lake_tables_and_marts(svc, rc):
    """Copy each report's parquet into datasights-lake/tables/restoke_<report>/ (so the
    agent auto-syncs them as tables), then build mart_restoke_food_cost. Rebuilds the
    lake catalog so everything shows up."""
    import glob
    import tempfile

    import duckdb
    import pyarrow.parquet as _pq

    dl = svc.get_container_client("datasights-lake")
    done = {}
    with tempfile.TemporaryDirectory() as tmp:
        for report in REPORTS:
            table = _table_name(report)
            local_dir = os.path.join(tmp, table)
            os.makedirs(local_dir, exist_ok=True)
            months, rows, cols = [], 0, None
            for b in rc.list_blobs(name_starts_with="reports/%s/" % report):
                if not b.name.endswith(".parquet"):
                    continue
                mon = os.path.basename(b.name)
                data = rc.download_blob(b.name).readall()
                with open(os.path.join(local_dir, mon), "wb") as f:
                    f.write(data)
                # copy into the lake tables area for the agent
                _upload(dl, "tables/%s/%s" % (table, mon), data, "application/octet-stream")
                months.append(mon)
            files = glob.glob(os.path.join(local_dir, "*.parquet"))
            if not files:
                continue
            for f in files:
                md = _pq.read_metadata(f)
                rows += md.num_rows
                if cols is None:
                    cols = list(_pq.read_schema(f).names)
            meta = {"view": table, "mode": "partitioned", "files": [os.path.basename(f) for f in files],
                    "columns": cols, "rows": rows, "source_rows": rows,
                    "date_col": None, "note": "Restoke Analytics API report (monthly)",
                    "exported_at": dt.datetime.now().isoformat()}
            _upload(dl, "tables/%s/_meta.json" % table, json.dumps(meta, indent=1), "application/json")
            done[table] = rows

        # mart_restoke_food_cost: actual food cost % = purchases / net sales (venue x month).
        # NB: theoretical/COGS not built - Restoke has no stocktake data for these venues.
        con = duckdb.connect(":memory:")
        sp = os.path.join(tmp, "restoke_sales", "*.parquet").replace("\\", "/")
        pp = os.path.join(tmp, "restoke_purchasing", "*.parquet").replace("\\", "/")
        con.execute("""CREATE TABLE mart AS
            WITH s AS (SELECT venue, query_month AS month,
                              SUM(TRY_CAST(total_sales_ex_tax AS DOUBLE)) net_sales,
                              COUNT(*) n_sales_lines
                       FROM read_parquet('%s', union_by_name=true) GROUP BY 1,2),
                 p AS (SELECT venue, query_month AS month,
                              SUM(TRY_CAST(total AS DOUBLE)) purchases,
                              COUNT(*) n_purchase_lines
                       FROM read_parquet('%s', union_by_name=true) GROUP BY 1,2)
            SELECT s.venue, s.month,
                   ROUND(s.net_sales,2) net_sales,
                   ROUND(p.purchases,2) purchases,
                   ROUND(100*p.purchases/NULLIF(s.net_sales,0),2) food_cost_pct,
                   s.n_sales_lines, p.n_purchase_lines
            FROM s JOIN p ON p.venue=s.venue AND p.month=s.month
            WHERE s.net_sales > 0 ORDER BY s.month, s.venue""" % (sp, pp))
        out = os.path.join(tmp, "mart.parquet")
        con.execute("COPY mart TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)" % out.replace("\\", "/"))
        mcols = [r[0] for r in con.execute("DESCRIBE mart").fetchall()]
        mrows = con.execute("SELECT COUNT(*) FROM mart").fetchone()[0]
        con.close()
        with open(out, "rb") as f:
            _upload(dl, "tables/mart_restoke_food_cost/data.parquet", f.read(),
                    "application/octet-stream")
        _upload(dl, "tables/mart_restoke_food_cost/_meta.json", json.dumps({
            "view": "mart_restoke_food_cost", "mode": "snapshot", "files": ["data.parquet"],
            "columns": mcols, "rows": mrows, "source_rows": mrows,
            "note": "Actual food cost %% = purchases/net sales (venue x month). Theoretical/COGS "
                    "unavailable - Restoke has no stocktake data for these venues.",
            "exported_at": dt.datetime.now().isoformat()}, indent=1), "application/json")
        done["mart_restoke_food_cost"] = mrows

    import lake_export
    n = lake_export.rebuild_catalog(dl)
    return {"tables": done, "catalog_tables": n}
