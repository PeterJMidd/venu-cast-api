# -*- coding: utf-8 -*-
"""Read-only DuckDB access to the datasights-lake blob.

TWO MODES, same public API (sync + query), so callers never change:

  DIRECT (default, LAKE_DIRECT!=0) - DuckDB's azure extension reads the Parquet
  straight out of blob storage (`az://datasights-lake/tables/<t>/*.parquet`).
  Only the row groups a query actually needs cross the wire, so a 2.7M-row
  table costs milliseconds instead of a multi-minute download. sync() becomes
  a no-op and only the tables a query REFERENCES get wired up.

  LOCAL (fallback) - the original etag-diff mirror into /tmp + file scans. Used
  automatically whenever direct mode can't be established (extension install
  blocked, no egress, bad credentials), so a failure degrades to "slow" and
  never to "broken".

Guard: user SQL must be SELECT/WITH and may not contain file/extension/DDL
keywords. INSTALL/LOAD/CREATE SECRET run only in this module's own setup path,
never through query()."""
import glob
import json
import logging
import os
import re
import time

LOG = logging.getLogger("lake_reader")

CACHE_DIR = os.environ.get("LAKE_CACHE_DIR", "/tmp/taskhub_lake")
EXT_DIR = os.environ.get("DUCKDB_EXT_DIR", "/tmp/duckdb_ext")
SYNC_PREFIXES = ("tables/mart_", "catalog/")
CONTAINER = "datasights-lake"

# Tables that live outside the datasights-lake/tables/ convention. `documents`
# is the SharePoint + Canva mirror in docs-lake, chunked and indexed - the
# agreements, deeds, board packs and entity papers - so a question can be
# answered from what we WROTE, not only from what we transacted. The embedding
# column is deliberately not exposed: a chunk carries a 1k-float vector and
# SELECT * would blow the result size for no analytical gain.
EXTRA_VIEWS = {
    "documents": ("SELECT path, title, ext, mtime, chunk_id, page, text FROM "
                  "read_parquet('az://docs-lake/index/parts/*.parquet', "
                  "union_by_name=true)"),
}
DIRECT = os.environ.get("LAKE_DIRECT", "1") != "0"
MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "1GB")
THREADS = os.environ.get("DUCKDB_THREADS", "2")

_GUARD = re.compile(r"^\s*(WITH|SELECT)\b", re.I)
_FORBIDDEN = re.compile(
    r"\b(ATTACH|COPY|EXPORT|INSTALL|LOAD|CREATE|INSERT|UPDATE|DELETE|DROP|ALTER|"
    r"PRAGMA|SET|CALL|IMPORT|SECRET|read_csv|read_json|read_parquet|read_text|"
    r"read_blob|parquet_scan|glob|getenv|http)\b"
    # DuckDB implicit file scans: FROM 'path/file.csv' bypasses the read_* names
    # (double-quoted = identifier, which is fine)
    r"|(?:FROM|JOIN)\s*'", re.I)

# table identifiers a query reads; CTE names/aliases fall out naturally because
# they are intersected against the tables that actually exist in the lake
_TABLE_RE = re.compile(r'(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', re.I)

_direct_state = {"checked": False, "ok": False}
_known_cache = {"at": 0.0, "names": set()}


def check_sql(sql):
    if not _GUARD.match(sql or ""):
        raise ValueError("only SELECT/WITH queries are allowed")
    if _FORBIDDEN.search(sql or ""):
        raise ValueError("query contains a forbidden keyword")


def referenced_tables(sql):
    return set(_TABLE_RE.findall(sql or ""))


# ------------------------------------------------------------------ blob bits
def _container():
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"], read_timeout=300, connection_timeout=60)
    return svc.get_container_client(CONTAINER)


def known_tables(cc=None, max_age=600):
    """Table names present in the lake. A delimiter walk returns just the table
    prefixes (~150 names in <1s) - the catalog is NOT authoritative here: it
    only carries feed_* tables and the compliance register."""
    now = time.time()
    if _known_cache["names"] and now - _known_cache["at"] < max_age:
        return _known_cache["names"]
    names = set()
    try:
        cc = cc or _container()
        for p in cc.walk_blobs(name_starts_with="tables/", delimiter="/"):
            parts = p.name.split("/")
            if len(parts) > 1 and parts[1]:
                names.add(parts[1])
    except Exception:
        LOG.exception("could not enumerate lake tables")
    _known_cache.update({"at": now, "names": names})
    return names


def _exists_in_blob(cc, name):
    try:
        return next(iter(cc.list_blobs(name_starts_with="tables/%s/" % name)), None) is not None
    except Exception:
        return False


# ---------------------------------------------------------------- direct mode
def _new_con():
    import duckdb
    con = duckdb.connect(":memory:")
    for stmt in ("SET memory_limit='%s'" % MEMORY_LIMIT,
                 "SET threads=%s" % int(THREADS),
                 # the progress bar writes to stderr - noise in App Insights
                 "SET enable_progress_bar=false"):
        try:
            con.execute(stmt)
        except Exception:
            pass
    return con


def _load_azure(con):
    """INSTALL/LOAD the azure extension. wwwroot is read-only on Azure so we
    point the extension dir at /tmp first; if that dir is unusable (some local
    Windows paths) fall back to DuckDB's default location rather than failing
    over to slow mode."""
    last = None
    for use_dir in (True, False):
        try:
            if use_dir:
                os.makedirs(EXT_DIR, exist_ok=True)
                con.execute("SET extension_directory='%s'" % EXT_DIR.replace("\\", "/"))
            con.execute("INSTALL azure")
            con.execute("LOAD azure")
            return
        except Exception as e:
            last = e
            try:
                con.execute("RESET extension_directory")
            except Exception:
                pass
    raise last


def _direct_con(sql, cc=None):
    """Connection with views over only the lake tables this SQL references."""
    con = _new_con()
    _load_azure(con)
    con.execute("CREATE OR REPLACE SECRET lake (TYPE AZURE, CONNECTION_STRING '%s')"
                % os.environ["BLOB_CONNECTION_STRING"].replace("'", "''"))
    wanted = referenced_tables(sql)
    for name in sorted(wanted & set(EXTRA_VIEWS)):
        con.execute('CREATE VIEW "%s" AS %s' % (name, EXTRA_VIEWS[name]))
    wanted = wanted - set(EXTRA_VIEWS)
    if wanted:
        cc = cc or _container()
        available = known_tables(cc)
        for name in sorted(wanted):
            if name not in available and not _exists_in_blob(cc, name):
                continue  # CTE name, alias, or genuinely missing table
            con.execute(
                'CREATE VIEW "%s" AS SELECT * FROM '
                "read_parquet('az://%s/tables/%s/*.parquet', union_by_name=true)"
                % (name, CONTAINER, name))
    return con


def _direct_ok():
    """One probe per worker: can we install the extension and read the lake?"""
    if not DIRECT:
        return False
    if _direct_state["checked"]:
        return _direct_state["ok"]
    _direct_state["checked"] = True
    try:
        con = _direct_con("SELECT 1")
        try:
            con.execute("SELECT 1").fetchall()
            _direct_state["ok"] = True
            LOG.info("lake: direct blob query mode active (no download)")
        finally:
            con.close()
    except Exception as e:
        LOG.warning("lake: direct mode unavailable (%s) - using local cache",
                    str(e)[:200])
        _direct_state["ok"] = False
    return _direct_state["ok"]


# ----------------------------------------------------------------- local mode
def _sync_local(extra_tables=None, log=print):
    """Etag-diff mirror of the mart tables (plus any extra named tables a
    caller references) into CACHE_DIR."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    state_path = os.path.join(CACHE_DIR, ".sync_state.json")
    state = {}
    if os.path.exists(state_path):
        try:
            state = json.load(open(state_path))
        except Exception:
            state = {}
    cc = _container()
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


def _local_con(sql):
    con = _new_con()
    for meta_path in glob.glob(os.path.join(CACHE_DIR, "tables", "*", "_meta.json")):
        tdir = os.path.dirname(meta_path)
        name = os.path.basename(tdir)
        if not glob.glob(os.path.join(tdir, "*.parquet")):
            continue
        pattern = os.path.join(tdir, "*.parquet").replace("\\", "/")
        con.execute('CREATE VIEW "%s" AS SELECT * FROM read_parquet(\'%s\', union_by_name=true)'
                    % (name, pattern))
    return con


def _cached_tables():
    return {os.path.basename(os.path.dirname(p))
            for p in glob.glob(os.path.join(CACHE_DIR, "tables", "*", "_meta.json"))}


# -------------------------------------------------------------- public surface
def sync(extra_tables=None, log=print):
    """No-op in direct mode (nothing to download); the etag mirror otherwise."""
    if _direct_ok():
        log("lake sync: skipped - direct blob query mode")
        return 0, 0
    return _sync_local(extra_tables=extra_tables, log=log)


def _fetch(con, sql, max_rows):
    res = con.execute(sql)
    cols = [d[0] for d in res.description]
    rows = res.fetchmany(max_rows)
    return {"columns": cols, "rows": [[_jsonable(v) for v in r] for r in rows]}


# A bad query is the caller's bug and fails identically in both modes - falling
# back would just repeat it after a pointless multi-minute download.
_SQL_ERRORS = ("ParserException", "CatalogException", "BinderException",
               "ConversionException", "SyntaxException", "InvalidInputException")


def query(sql, max_rows=200):
    """Read-only DuckDB over the lake. Direct blob scan when available, local
    parquet cache otherwise (self-healing: syncs what the query needs)."""
    check_sql(sql)
    if _direct_ok():
        con = None
        try:
            con = _direct_con(sql)
            return _fetch(con, sql, max_rows)
        except Exception as e:
            if type(e).__name__ in _SQL_ERRORS:
                raise  # real SQL/schema error - surface it, don't mask it
            LOG.warning("direct query failed (%s) - retrying via local cache",
                        str(e)[:200])
        finally:
            if con is not None:
                con.close()
    missing = {t for t in referenced_tables(sql) if re.fullmatch(r"[A-Za-z0-9_]+", t)} \
        - _cached_tables()
    if missing:
        try:
            _sync_local(extra_tables=sorted(missing), log=LOG.info)
        except Exception:
            LOG.exception("on-demand local sync failed")
    con = _local_con(sql)
    try:
        return _fetch(con, sql, max_rows)
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
