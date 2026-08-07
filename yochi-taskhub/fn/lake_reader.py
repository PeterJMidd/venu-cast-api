# -*- coding: utf-8 -*-
"""Trimmed lake reader for the watcher (vendored from datasights-lake/queryapp).
Syncs only the mart tables + catalog into /tmp and runs read-only DuckDB SQL."""
import glob
import json
import os
import re
import time

CACHE_DIR = os.environ.get("LAKE_CACHE_DIR", "/tmp/taskhub_lake")
SYNC_PREFIXES = ("tables/mart_", "catalog/")
CONTAINER = "datasights-lake"

_GUARD = re.compile(r"^\s*(WITH|SELECT)\b", re.I)
_FORBIDDEN = re.compile(
    r"\b(ATTACH|COPY|EXPORT|INSTALL|LOAD|CREATE|INSERT|UPDATE|DELETE|DROP|ALTER|"
    r"PRAGMA|SET|CALL|IMPORT|read_csv|read_json|read_parquet|read_text|read_blob|"
    r"parquet_scan|glob|getenv|http)\b"
    # DuckDB implicit file scans: FROM 'path/file.csv' bypasses the read_* names
    # (double-quoted = identifier, which is fine)
    r"|(?:FROM|JOIN)\s*'", re.I)


def sync(extra_tables=None, log=print):
    """Etag-diff mirror of the mart tables (plus any extra named tables that
    watcher rules reference) into CACHE_DIR."""
    from azure.storage.blob import BlobServiceClient
    os.makedirs(CACHE_DIR, exist_ok=True)
    state_path = os.path.join(CACHE_DIR, ".sync_state.json")
    state = {}
    if os.path.exists(state_path):
        try:
            state = json.load(open(state_path))
        except Exception:
            state = {}
    svc = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"], read_timeout=300, connection_timeout=60)
    cc = svc.get_container_client(CONTAINER)
    t0 = time.time()
    downloaded = total = 0

    def _save_state():
        with open(state_path, "w") as f:
            json.dump(state, f)

    prefixes = list(SYNC_PREFIXES) + sorted(
        "tables/%s/" % t for t in (extra_tables or []) if re.fullmatch(r"[A-Za-z0-9_]+", t)
    )
    for prefix in prefixes:
        for b in cc.list_blobs(name_starts_with=prefix):
            total += 1
            local = os.path.join(CACHE_DIR, b.name.replace("/", os.sep))
            if state.get(b.name) == b.etag and os.path.exists(local):
                continue
            os.makedirs(os.path.dirname(local), exist_ok=True)
            for attempt in (1, 2):
                try:
                    with open(local, "wb") as f:
                        cc.download_blob(b.name, max_concurrency=2).readinto(f)
                    state[b.name] = b.etag
                    downloaded += 1
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    log("retrying %s after: %s" % (b.name, str(e)[:100]))
            if downloaded % 10 == 0:
                _save_state()  # keep progress across timeouts on big first syncs
    _save_state()
    log("lake sync: %d/%d blobs in %.1fs" % (downloaded, total, time.time() - t0))
    return downloaded, total


def query(sql, max_rows=200):
    """Read-only DuckDB over the synced mart parquet files."""
    import duckdb
    if not _GUARD.match(sql or ""):
        raise ValueError("only SELECT/WITH queries are allowed")
    if _FORBIDDEN.search(sql or ""):
        raise ValueError("query contains a forbidden keyword")
    con = duckdb.connect(":memory:")
    try:
        for meta_path in glob.glob(os.path.join(CACHE_DIR, "tables", "*", "_meta.json")):
            tdir = os.path.dirname(meta_path)
            name = os.path.basename(tdir)
            if not glob.glob(os.path.join(tdir, "*.parquet")):
                continue
            pattern = os.path.join(tdir, "*.parquet").replace("\\", "/")
            con.execute('CREATE VIEW "%s" AS SELECT * FROM read_parquet(\'%s\', union_by_name=true)'
                        % (name, pattern))
        res = con.execute(sql)
        cols = [d[0] for d in res.description]
        rows = res.fetchmany(max_rows)
        return {"columns": cols, "rows": [[_jsonable(v) for v in r] for r in rows]}
    finally:
        con.close()


def _jsonable(v):
    import datetime as dt
    import decimal
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, bytes):
        return None
    if isinstance(v, float) and (v != v):
        return None
    return v
