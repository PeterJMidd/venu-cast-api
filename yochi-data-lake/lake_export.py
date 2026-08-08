# -*- coding: utf-8 -*-
"""Nightly incremental sync: dataSights warehouse -> datasights-lake Parquet container.

Cloud port of the local backfill exporter (datasights-lake/export_lake.py):
  - snapshot views: re-exported in full every run (streamed to temp file, memory-safe)
  - partitioned views: only the trailing REFRESH_MONTHS months are re-exported
    (+ the part-null chunk); history months are immutable
  - mystery-shopping CSVs: mirrored into lake tables incl. daily-snapshot history
  - catalog/catalog.json rebuilt at the end from all _meta.json files

Config via app settings: DS_SQL_*, BLOB_CONNECTION_STRING, LAKE_CONTAINER,
REFRESH_MONTHS (default 2), SNAPSHOT_MAX (default 300000).
"""
import datetime as dt
import decimal
import io
import json
import logging
import os
import re
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pymssql
from azure.storage.blob import BlobServiceClient, ContentSettings

LOG = logging.getLogger("lake_export")
CONTAINER = os.environ.get("LAKE_CONTAINER", "datasights-lake")
SNAPSHOT_MAX = int(os.environ.get("SNAPSHOT_MAX", "300000"))
REFRESH_MONTHS = int(os.environ.get("REFRESH_MONTHS", "2"))
FETCH_BATCH = 50_000
EARLY_CUTOFF = "2015-01-01"

DENYLIST = {
    "RestokeOrganisation", "TandaOrganisation", "CelsiOrganisation",
    "PolygonRedcatOrganisation", "PolygonRedcatReportEndpoints",
    "ReviewTrackersAccounts",
}
FORCE_SNAPSHOT = {"PolygonRedcatMemberDetailsReport"}

MYSTERY_SRC = "mystery-shopping"
MYSTERY_NAMES = {
    "Category_Averages": "MysteryShoppingCategoryAverages",
    "Site_Comparisons": "MysteryShoppingSiteComparisons",
}


def connect():
    return pymssql.connect(
        server=os.environ["DS_SQL_SERVER"], user=os.environ["DS_SQL_USER"],
        password=os.environ["DS_SQL_PASSWORD"], database=os.environ["DS_SQL_DB"],
        timeout=600, login_timeout=30)


def blob_service():
    return BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])


def _normalise(df):
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().head(50)
            if len(sample) and isinstance(sample.iloc[0], decimal.Decimal):
                df[c] = df[c].astype(float)
            elif len(sample) and isinstance(sample.iloc[0], (bytes, bytearray)):
                df[c] = None
    return df


def stream_query_to_parquet(conn, sql, out_path):
    """Run sql, stream row batches into a parquet file. Returns (n_rows, columns)."""
    cur = conn.cursor()
    cur.execute("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    writer, schema, total = None, None, 0
    try:
        while True:
            batch = cur.fetchmany(FETCH_BATCH)
            if not batch:
                break
            df = _normalise(pd.DataFrame.from_records(batch, columns=cols))
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema, compression="zstd")
            else:
                table = table.cast(schema, safe=False)
            writer.write_table(table)
            total += len(df)
    finally:
        if writer is not None:
            writer.close()
    return total, cols


def upload_file(cc, path, local, content_type="application/octet-stream"):
    with open(local, "rb") as f:
        cc.upload_blob(path, f, overwrite=True,
                       content_settings=ContentSettings(content_type=content_type))


def upload_bytes(cc, path, data, content_type="application/json"):
    cc.upload_blob(path, data, overwrite=True,
                   content_settings=ContentSettings(content_type=content_type))


def month_windows(n_back):
    """Trailing n_back month windows (label, lo, hi) ending this month."""
    today = dt.date.today()
    y, m = today.year, today.month
    out = []
    for _ in range(n_back):
        nxt_y, nxt_m = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append(("%04d-%02d" % (y, m), "%04d-%02d-01" % (y, m), "%04d-%02d-01" % (nxt_y, nxt_m)))
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return list(reversed(out))


def _export_one(conn, cc, entry, tmpdir):
    """entry = catalog view dict (from lake _meta / initial profile)."""
    name = entry["view"]
    dcol = entry.get("date_col")
    mode = ("snapshot" if entry.get("mode") == "snapshot" or not dcol
            or name in FORCE_SNAPSHOT else "partitioned")
    tmp = os.path.join(tmpdir, "x.parquet")
    if mode == "snapshot":
        n, cols = stream_query_to_parquet(conn, "SELECT * FROM [%s] WITH (NOLOCK)" % name, tmp)
        if n == 0 and not os.path.exists(tmp):
            return {"view": name, "mode": mode, "rows": 0, "skipped": "empty"}
        upload_file(cc, "tables/%s/data.parquet" % name, tmp)
        rows = n
    else:
        rows = 0
        windows = month_windows(REFRESH_MONTHS) + [("null", None, None)]
        for label, lo, hi in windows:
            if lo is None:
                sql = "SELECT * FROM [{v}] WITH (NOLOCK) WHERE [{c}] IS NULL".format(v=name, c=dcol)
            else:
                sql = ("SELECT * FROM [{v}] WITH (NOLOCK) WHERE [{c}] >= '{lo}' AND [{c}] < '{hi}'"
                       ).format(v=name, c=dcol, lo=lo, hi=hi)
            n, cols = stream_query_to_parquet(conn, sql, tmp)
            if n:
                upload_file(cc, "tables/%s/part-%s.parquet" % (name, label), tmp)
                rows += n
    # refresh _meta.json (keep original fields, update freshness)
    meta = dict(entry)
    meta.update({"mode": mode, "exported_at": dt.datetime.now().isoformat(),
                 "last_sync_rows": rows})
    upload_bytes(cc, "tables/%s/_meta.json" % name, json.dumps(meta, indent=1))
    return {"view": name, "mode": mode, "rows": rows}


def sync_mystery(svc, cc):
    src = svc.get_container_client(MYSTERY_SRC)
    current, history = {}, {k: [] for k in MYSTERY_NAMES}
    for b in src.list_blobs():
        data = src.download_blob(b.name).readall()
        df = pd.read_csv(io.BytesIO(data))
        m = re.match(r"archive/(\d{4}-\d{2}-\d{2})_(Category_Averages|Site_Comparisons)\.csv", b.name)
        if m:
            df.insert(0, "snapshot_date", m.group(1))
            history[m.group(2)].append(df)
        else:
            for key in MYSTERY_NAMES:
                if key in b.name:
                    current[key] = df
    results = []
    for key, table in MYSTERY_NAMES.items():
        frames = ([("", current.get(key))] if key in current else []) + \
                 ([("History", pd.concat(history[key], ignore_index=True))] if history[key] else [])
        for suffix, df in frames:
            if df is None:
                continue
            buf = io.BytesIO()
            pq.write_table(pa.Table.from_pandas(_normalise(df), preserve_index=False),
                           buf, compression="zstd")
            tname = table + suffix
            upload_bytes(cc, "tables/%s/data.parquet" % tname, buf.getvalue(),
                         "application/octet-stream")
            meta = {"view": tname, "mode": "snapshot", "files": ["data.parquet"],
                    "columns": list(df.columns), "rows": len(df),
                    "source": "mystery-shopping container",
                    "exported_at": dt.datetime.now().isoformat()}
            upload_bytes(cc, "tables/%s/_meta.json" % tname, json.dumps(meta, indent=1))
            results.append({"view": tname, "rows": len(df)})
    return results


def rebuild_catalog(cc):
    tables = []
    for b in cc.list_blobs(name_starts_with="tables/"):
        if b.name.endswith("/_meta.json"):
            try:
                tables.append(json.loads(cc.download_blob(b.name).readall()))
            except Exception:
                LOG.exception("bad meta: %s", b.name)
    manifest = {"generated_at": dt.datetime.now().isoformat(),
                "container": CONTAINER,
                "tables": sorted(tables, key=lambda t: t.get("view", ""))}
    upload_bytes(cc, "catalog/catalog.json", json.dumps(manifest, indent=1))
    return len(tables)


def backfill(view, m_from, m_to, force=False):
    """One-off history backfill for a partitioned view, month by month, oldest first.
    Skips months whose blob already exists (unless force). Resumable across calls."""
    svc = blob_service()
    cc = svc.get_container_client(CONTAINER)
    cat = json.loads(cc.download_blob("catalog/catalog.json").readall())
    entry = next((t for t in cat["tables"] if t.get("view") == view), None)
    if not entry or not entry.get("date_col"):
        return {"error": "view not in catalog or has no date column: %s" % view}
    dcol = entry["date_col"]
    existing = {b.name for b in cc.list_blobs(name_starts_with="tables/%s/" % view)}

    def months(a, b):
        y, m = int(a[:4]), int(a[5:7])
        ey, em = int(b[:4]), int(b[5:7])
        while (y, m) <= (ey, em):
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            yield "%04d-%02d" % (y, m), "%04d-%02d-01" % (y, m), "%04d-%02d-01" % (ny, nm)
            y, m = ny, nm

    done, skipped, rows_total = [], [], 0
    conn = connect()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "x.parquet")
        for label, lo, hi in months(m_from, m_to):
            path = "tables/%s/part-%s.parquet" % (view, label)
            if path in existing and not force:
                skipped.append(label)
                continue
            sql = ("SELECT * FROM [{v}] WITH (NOLOCK) WHERE [{c}] >= '{lo}' AND [{c}] < '{hi}'"
                   ).format(v=view, c=dcol, lo=lo, hi=hi)
            try:
                n, _cols = stream_query_to_parquet(conn, sql, tmp)
            except Exception:
                LOG.exception("backfill %s %s failed; reconnecting", view, label)
                try:
                    conn.close()
                except Exception:
                    pass
                conn = connect()
                n, _cols = stream_query_to_parquet(conn, sql, tmp)
            if n:
                upload_file(cc, path, tmp)
                rows_total += n
            done.append("%s:%d" % (label, n))
            LOG.info("backfill %s %s: %d rows", view, label, n)
    conn.close()
    return {"view": view, "done": done, "skipped": skipped, "rows": rows_total}


def run(only=None):
    """Full nightly sync. Returns a result summary dict."""
    svc = blob_service()
    cc = svc.get_container_client(CONTAINER)
    cat = json.loads(cc.download_blob("catalog/catalog.json").readall())
    entries = [t for t in cat["tables"]
               if t.get("view") not in DENYLIST
               and not t.get("view", "").startswith("MysteryShopping")]
    if only:
        entries = [t for t in entries if t["view"] in only]
    results, errors = [], []
    conn = connect()
    with tempfile.TemporaryDirectory() as tmpdir:
        for e in entries:
            try:
                r = _export_one(conn, cc, e, tmpdir)
                results.append(r)
                LOG.info("synced %s (%s rows)", r["view"], r["rows"])
            except Exception as ex:
                errors.append({"view": e.get("view"), "error": str(ex)[:400]})
                LOG.exception("sync failed: %s", e.get("view"))
                try:
                    conn.close()
                except Exception:
                    pass
                conn = connect()
    conn.close()
    try:
        results.extend(sync_mystery(svc, cc))
    except Exception as ex:
        errors.append({"view": "mystery-shopping", "error": str(ex)[:400]})
        LOG.exception("mystery sync failed")
    n_tables = rebuild_catalog(cc)
    return {"synced": len(results), "errors": errors, "catalog_tables": n_tables,
            "finished_at": dt.datetime.now().isoformat()}
