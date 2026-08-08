# -*- coding: utf-8 -*-
"""Extract all OpCentral / Chi Central API data to Azure Blob (container: opcentral-export).

OpCentral (yochicentral.api.opcentral.com.au) exposes the org's people, locations,
audits, training, forms, manuals, policy/news sign-offs and support tickets as JSON.
Auth is a single HQ API key in the `x-api-key` header. This pulls every bulk endpoint
and writes both raw JSON (exact re-extract / audit) and columnar Parquet, then registers
each as an agent-queryable lake table (opc_*).

Layout (container opcentral-export):
  raw/<name>.json            latest raw API response (list of records)
  <name>.parquet             columnar, flattened (nested fields -> JSON strings)
  catalog.json               manifest: endpoints, row counts, freshness, errors

The lake tables (datasights-lake/tables/opc_<name>/) are rebuilt so the public agent
and the MCP connector can query them immediately.

Response shapes handled:
  - bare list          -> used directly            (userlist, workplacelist, results)
  - Laravel paginated  -> {data:[...], current_page, last_page, next_page_url} -> walk pages

Config (app settings): OPCENTRAL_KEY (secret), BLOB_CONNECTION_STRING,
  OPCENTRAL_CONTAINER (default opcentral-export), OPCENTRAL_BASE
  (default https://yochicentral.api.opcentral.com.au), OPCENTRAL_FORM_SUBMISSIONS
  (default 1 - also pull each form's submissions).

PRIVACY: opc_users carries staff PII (dob, home_address_1/2, phone). It is written in
full to honour "pull all data". Set OPCENTRAL_MASK_USER_PII=1 to drop those columns from
the agent-facing parquet/table (the raw JSON always stays complete in blob).
"""
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

LOG = logging.getLogger("opcentral_export")
BASE = os.environ.get("OPCENTRAL_BASE", "https://yochicentral.api.opcentral.com.au")
CONTAINER = os.environ.get("OPCENTRAL_CONTAINER", "opcentral-export")
HTTP_TIMEOUT = int(os.environ.get("OPCENTRAL_HTTP_TIMEOUT", "300"))
PER_PAGE = 500
# Per-endpoint wall-clock budget for paginated walks; 0 = unlimited. Protects the 3 AM
# job from pathologically slow endpoints (news/signoff/list returns ~60s/page). If a walk
# is cut short, the truncation is recorded in the summary + catalog (never silent).
PAGE_BUDGET = int(os.environ.get("OPCENTRAL_PAGE_BUDGET_SECS", "600"))
MASK_USER_PII = os.environ.get("OPCENTRAL_MASK_USER_PII", "0") == "1"
PII_COLS = ("dob", "home_address_1", "home_address_2")

# name -> (method, path, params, paginated, nightly)
# nightly=False -> excluded from the scheduled 3 AM run and from a default manual run;
# pull it explicitly with only={name}. news_signoff is 287k rows at ~60s/page (~10h full),
# so it is on-demand only, not nightly.
ENDPOINTS = [
    ("users",             "GET",  "/userlist",                        {"include_deleted": 1}, False, True),
    ("locations",         "GET",  "/workplacelist",                   {"include_deleted": 1}, False, True),
    ("audit_results",     "POST", "/public/audit/results",            {},                     False, True),
    ("program_progress",  "GET",  "/training/program/all/results",    {},                     False, True),
    ("activity_progress", "GET",  "/training/results",                {},                     False, True),
    ("workshop_sessions", "GET",  "/v1/training/workshop/session/list", {"include_past": 1},  True,  True),
    ("forms",             "GET",  "/v1/form/formlist",                {"include_archived": 1}, True, True),
    ("policy_signoff",    "GET",  "/policy/signoff/list",             {},                     True,  True),
    ("manuals",           "GET",  "/v1/opdocs/manuals/all",           {},                     True,  True),
    ("support_tickets",   "GET",  "/public/v1/support/list",          {},                     True,  True),
    ("news_signoff",      "GET",  "/news/signoff/list",               {"include_archived": 1}, True, False),
]


def _key():
    k = os.environ.get("OPCENTRAL_KEY")
    if not k:
        raise RuntimeError("OPCENTRAL_KEY not configured")
    return k


def _req(method, path, params=None, body=None):
    url = "%s/%s" % (BASE, path.strip("/"))
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    headers = {"x-api-key": _key(), "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            time.sleep(4 * (attempt + 1))
    raise last


def _fetch(name, method, path, params, paginated, budget=None):
    """Return (rows, truncated) for one endpoint (walks Laravel pages if paginated).
    `budget` seconds caps a paginated walk; truncated=True means pages remained unread."""
    if not paginated:
        d = _req(method, path, params or {}, body={} if method == "POST" else None)
        if isinstance(d, list):
            return d, False
        if isinstance(d, dict) and isinstance(d.get("data"), list):
            return d["data"], False
        return (d.get("results", []) if isinstance(d, dict) else []), False
    rows, page, truncated = [], 1, False
    t0 = time.time()
    while True:
        p = dict(params or {}); p["per_page"] = PER_PAGE; p["page"] = page
        d = _req(method, path, p)
        if isinstance(d, list):
            rows.extend(d); break
        chunk = d.get("data", []) if isinstance(d, dict) else []
        rows.extend(chunk)
        last_page = d.get("last_page") if isinstance(d, dict) else 1
        if not chunk or not last_page or page >= last_page or not d.get("next_page_url"):
            break
        if budget and (time.time() - t0) > budget:
            truncated = True
            LOG.warning("opcentral %s: page budget %ss hit at page %d/%s (%d rows) - truncated",
                        name, budget, page, last_page, len(rows))
            break
        page += 1
    return rows, truncated


def _fetch_form_submissions(forms):
    """For every form, pull its submission list (paginated per form_id)."""
    out = []
    for f in forms:
        fid = f.get("id") or f.get("form_id")
        if fid is None:
            continue
        try:
            subs, _ = _fetch("form_submissions", "GET",
                             "/v1/form/submissionlist/%s" % fid, {}, True, budget=PAGE_BUDGET)
        except Exception as e:
            LOG.warning("form %s submissions failed: %s", fid, str(e)[:150])
            continue
        for s in subs:
            if isinstance(s, dict):
                s.setdefault("form_id", fid)
        out.extend(subs)
    return out


def _flat(rows):
    """JSON-stringify nested (dict/list) values so the frame is flat and DuckDB-friendly."""
    flat = []
    for r in rows:
        if not isinstance(r, dict):
            flat.append({"value": json.dumps(r, default=str)}); continue
        o = {}
        for k, v in r.items():
            o[k] = json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
        flat.append(o)
    return flat


def _to_parquet(rows):
    df = pd.DataFrame(_flat(rows))
    buf = io.BytesIO()
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
    except Exception:
        # last-resort: coerce every column to string so mixed types can't break the write
        df = df.astype("string")
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
    return buf.getvalue(), list(df.columns), len(df)


def _svc():
    # Force chunked multi-block uploads (max_single_put_size) with generous timeouts so
    # large parquet (activity_progress ~tens of MB) doesn't fail on one big PUT.
    return BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"],
        max_single_put_size=8 * 1024 * 1024, max_block_size=8 * 1024 * 1024,
        connection_timeout=600, read_timeout=600)


def _upload(cc, path, data, ctype):
    last = None
    for attempt in range(4):
        try:
            cc.upload_blob(path, data, overwrite=True, max_concurrency=4,
                           content_settings=ContentSettings(content_type=ctype))
            return
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last


def run(only=None):
    """Pull OpCentral endpoints to blob + register lake tables. Returns a summary.
    Default (only=None): every nightly=True endpoint (+ form submissions). Pass
    only={name,...} to pull specific endpoints regardless of the nightly flag."""
    svc = _svc()
    try:
        svc.create_container(CONTAINER)
    except Exception:
        pass
    cc = svc.get_container_client(CONTAINER)

    t0 = time.time()
    summary = {"source": "OpCentral API (%s)" % BASE, "endpoints": {}, "errors": [],
               "truncated": []}
    forms_rows = None
    plan = list(ENDPOINTS)
    if os.environ.get("OPCENTRAL_FORM_SUBMISSIONS", "1") == "1":
        plan.append(("form_submissions", "GET", "__forms__", {}, True, True))

    for name, method, path, params, paginated, nightly in plan:
        if only is None and not nightly:
            continue
        if only and name not in only:
            continue
        try:
            if path == "__forms__":
                rows = _fetch_form_submissions(forms_rows or [])
            else:
                rows, trunc = _fetch(name, method, path, params, paginated, budget=PAGE_BUDGET)
                if trunc:
                    summary["truncated"].append(name)
                if name == "forms":
                    forms_rows = rows
            if name == "users" and MASK_USER_PII:
                for r in rows:
                    if isinstance(r, dict):
                        for c in PII_COLS:
                            r.pop(c, None)
            _upload(cc, "raw/%s.json" % name, json.dumps(rows, default=str),
                    "application/json")
            if rows:
                data, cols, n = _to_parquet(rows)
                _upload(cc, "%s.parquet" % name, data, "application/octet-stream")
            else:
                cols, n = [], 0
            summary["endpoints"][name] = {"rows": n, "path": path if path != "__forms__"
                                          else "/v1/form/submissionlist/{form_id}"}
            LOG.info("opcentral %s: %d rows", name, n)
        except Exception as e:
            summary["errors"].append("%s: %s" % (name, str(e)[:200]))
            LOG.exception("opcentral fetch failed: %s", name)

    catalog = {
        "source": summary["source"],
        "generated_at": dt.datetime.now().isoformat(),
        "container": CONTAINER,
        "auth": "x-api-key header (HQ key)",
        "endpoints": summary["endpoints"],
        "layout": "raw/<name>.json + <name>.parquet ; lake tables opc_<name>",
        "pii_note": "opc_users holds staff PII; masked=%s" % MASK_USER_PII,
        "truncated": summary["truncated"],
        "errors": summary["errors"],
    }
    _upload(cc, "catalog.json", json.dumps(catalog, indent=1), "application/json")

    try:
        summary["lake"] = register_lake_tables(svc, cc, summary["endpoints"])
    except Exception as e:
        summary["errors"].append("register: %s" % str(e)[:200])
        LOG.exception("register_lake_tables failed")

    summary["secs"] = round(time.time() - t0)
    summary["finished_at"] = dt.datetime.now().isoformat()
    return summary


def register_lake_tables(svc, cc, endpoints):
    """Copy each <name>.parquet into datasights-lake/tables/opc_<name>/ so the agent +
    MCP connector auto-sync them, write _meta.json, and rebuild the lake catalog."""
    import tempfile

    import pyarrow.parquet as _pq

    dl = svc.get_container_client("datasights-lake")
    done = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in endpoints:
            blob = "%s.parquet" % name
            try:
                data = cc.download_blob(blob).readall()
            except Exception:
                continue
            table = "opc_" + name
            local = os.path.join(tmp, blob)
            with open(local, "wb") as f:
                f.write(data)
            cols = list(_pq.read_schema(local).names)
            rows = _pq.read_metadata(local).num_rows
            _upload(dl, "tables/%s/data.parquet" % table, data, "application/octet-stream")
            note = "OpCentral %s (full snapshot, refreshed nightly)." % name
            if name == "users":
                note += (" Staff PII (dob/home_address) %s."
                         % ("MASKED" if MASK_USER_PII else "INCLUDED"))
            _upload(dl, "tables/%s/_meta.json" % table, json.dumps({
                "view": table, "mode": "snapshot", "files": ["data.parquet"],
                "columns": cols, "rows": rows, "source_rows": rows,
                "date_col": None, "note": note,
                "exported_at": dt.datetime.now().isoformat()}, indent=1), "application/json")
            done[table] = rows

    import lake_export
    n = lake_export.rebuild_catalog(dl)
    return {"tables": done, "catalog_tables": n}
