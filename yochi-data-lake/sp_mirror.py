# -*- coding: utf-8 -*-
"""Mirror SharePoint 'Shared Documents' DOCUMENT files into blob (container docs-lake).

Scope: document types only (DOC_EXTS) - media/design working files are excluded on
purpose (the library is ~956 GB, of which ~115 GB is documents).

Layout:  docs-lake/sharepoint/<folder path>/<name>
State:   docs-lake/sharepoint/_state.json   {link, phase, files_done, updated_at}
  - phase 'walk': initial full crawl, resumable via the delta nextLink cursor
  - phase 'delta': nightly incremental via the stored deltaLink

Copy path: server-side upload_blob_from_url from the item's pre-authed Graph
download URL - bytes never flow through the function. Skip logic: existing blob
with matching source etag (kept in blob metadata) is not re-copied.

run_mirror(minutes) processes until the time budget is used, saves the cursor,
and returns progress - call repeatedly until phase flips to 'delta'.
"""
import datetime as dt
import json
import logging
import os
import time
import urllib.parse
import urllib.request

from azure.storage.blob import BlobServiceClient

LOG = logging.getLogger("sp_mirror")
GRAPH = "https://graph.microsoft.com/v1.0"
CONTAINER = os.environ.get("DOCS_CONTAINER", "docs-lake")
STATE_BLOB = "sharepoint/_state.json"

# NOTE: 'msg' deliberately excluded - saved Outlook emails are not mirrored
# (owner decision 2026-07: email content stays out of the blob lake).
DOC_EXTS = {"pdf", "xlsx", "xlsm", "xls", "docx", "doc", "pptx", "ppt",
            "csv", "txt", "md", "rtf", "vsdx", "one"}
MAX_BYTES = 500 * 1024 * 1024  # skip >500MB single files


def _token():
    data = urllib.parse.urlencode({
        "client_id": os.environ["CLIENT_ID"], "client_secret": os.environ["CLIENT_SECRET"],
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        "https://login.microsoftonline.com/%s/oauth2/v2.0/token" % os.environ["TENANT_ID"], data=data)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["access_token"]


def _get(url, tok):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def _site_drive_root(tok):
    site = _get(GRAPH + "/sites/" + os.environ.get(
        "SHAREPOINT_SITE", "embraceyochi.sharepoint.com:/sites/YochiTeamSharepoint"), tok)
    return site["id"]


def _svc():
    return BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])


def _load_state(cc):
    try:
        return json.loads(cc.download_blob(STATE_BLOB).readall())
    except Exception:
        return {}


def _save_state(cc, state):
    state["updated_at"] = dt.datetime.utcnow().isoformat()
    cc.upload_blob(STATE_BLOB, json.dumps(state), overwrite=True)


def _dest_name(item):
    path = (item.get("parentReference", {}).get("path") or "")
    rel = urllib.parse.unquote(path.split("root:", 1)[-1].lstrip("/"))
    name = item["name"]
    full = ("sharepoint/" + (rel + "/" if rel else "") + name)
    # blob names: no backslashes; collapse doubled slashes
    return full.replace("\\", "/").replace("//", "/")


def _existing_etags(cc):
    out = {}
    for b in cc.list_blobs(name_starts_with="sharepoint/", include=["metadata"]):
        et = (b.metadata or {}).get("src_etag")
        if et:
            out[b.name] = et
    return out


def run_mirror(minutes=100):
    """Process the crawl/delta for up to `minutes`, then save cursor and return."""
    deadline = time.time() + minutes * 60
    svc = _svc()
    try:
        svc.create_container(CONTAINER)
    except Exception:
        pass
    cc = svc.get_container_client(CONTAINER)
    state = _load_state(cc)
    tok = _token()
    tok_at = time.time()
    sid = _site_drive_root(tok)

    url = state.get("link")
    phase = state.get("phase", "walk")
    if not url:
        url = (GRAPH + "/sites/%s/drive/root/delta"
               "?$select=id,name,size,file,folder,parentReference,eTag&$top=500" % sid)
        phase = "walk"

    existing = _existing_etags(cc) if phase == "walk" else {}
    copied = skipped = errors = pages = 0
    out_of_time = False
    t0 = time.time()

    while url and time.time() < deadline:
        if time.time() - tok_at > 2400:  # refresh token every 40 min
            tok = _token()
            tok_at = time.time()
        try:
            d = _get(url, tok)
        except Exception:
            LOG.exception("page fetch failed; retrying with fresh token")
            tok = _token()
            tok_at = time.time()
            d = _get(url, tok)
        for it in d.get("value", []):
            # A delta page can carry entries that are not usable files: items
            # removed since the last cursor (@removed, not always 'deleted'),
            # and entries with a file facet but no name. Indexing one of those
            # raised KeyError and killed the whole run - which is why this
            # mirror stopped on 17 Jul 2026 and quietly went a month stale.
            if "file" not in it or it.get("deleted") or "@removed" in it:
                continue
            name = it.get("name")
            if not name:
                continue
            ext = name.rsplit(".", 1)[1].lower() if "." in name else ""
            if ext not in DOC_EXTS or it.get("size", 0) > MAX_BYTES:
                continue
            dest = _dest_name(it)
            etag = it.get("eTag", "")
            if existing.get(dest) == etag:
                skipped += 1
                continue
            # Delta re-sends an item whenever ANY property changes, so without
            # this the same unchanged file is downloaded again on every run. A
            # HEAD against the copy we already hold costs a fraction of a
            # re-download, and makes a resumed page cheap rather than wasteful.
            if phase != "walk" and etag:
                try:
                    props = cc.get_blob_client(dest).get_blob_properties()
                    if (props.metadata or {}).get("src_etag") == etag.replace('"', ""):
                        skipped += 1
                        continue
                except Exception:
                    pass                       # not there yet - copy it below
            if time.time() >= deadline:
                # The budget used to be checked only between pages, so a page
                # of 500 files ran hours past it. Stop mid-page and resume from
                # this same page next time - the etag checks above make the
                # replayed items nearly free.
                out_of_time = True
                break
            try:
                # NOTE: no $select here - it strips the @microsoft.graph.downloadUrl annotation
                item = _get(GRAPH + "/sites/%s/drive/items/%s" % (sid, it["id"]), tok)
                dl = item.get("@microsoft.graph.downloadUrl")
                if not dl:
                    errors += 1
                    continue
                bc = cc.get_blob_client(dest)
                bc.upload_blob_from_url(dl, overwrite=True)
                bc.set_blob_metadata({"src_etag": etag.replace('"', ""),
                                      "src_id": it["id"]})
                copied += 1
            except Exception as e:
                errors += 1
                if errors <= 20:
                    LOG.warning("copy failed %s: %s", dest, str(e)[:200])
        if out_of_time:
            _save_state(cc, {"link": url, "phase": phase,
                             "files_done": state.get("files_done", 0) + copied + skipped})
            return {"phase": phase, "complete": False, "out_of_time": True,
                    "pages": pages, "copied": copied, "skipped": skipped,
                    "errors": errors, "secs": round(time.time() - t0)}
        pages += 1
        nxt = d.get("@odata.nextLink")
        delta = d.get("@odata.deltaLink")
        if nxt:
            url = nxt
            if pages % 5 == 0:
                _save_state(cc, {"link": url, "phase": phase,
                                 "files_done": state.get("files_done", 0) + copied + skipped})
        else:
            # crawl complete -> store deltaLink for incremental syncs
            _save_state(cc, {"link": delta, "phase": "delta",
                             "files_done": state.get("files_done", 0) + copied + skipped})
            return {"phase": "delta", "complete": True, "pages": pages, "copied": copied,
                    "skipped": skipped, "errors": errors, "secs": round(time.time() - t0)}

    _save_state(cc, {"link": url, "phase": phase,
                     "files_done": state.get("files_done", 0) + copied + skipped})
    return {"phase": phase, "complete": False, "pages": pages, "copied": copied,
            "skipped": skipped, "errors": errors, "secs": round(time.time() - t0)}


# ------------------------------------------------------------ reconciliation

# Folders whose contents must actually BE here, checked against SharePoint
# itself rather than against the delta cursor.
RECONCILE_PREFIXES = [
    "ALL-SHARES/Finance/GROUP STRUCTURE & CORPORATE",
    "ALL-SHARES/1.International Confidential",
    "ALL-SHARES/Finance/MONTHLY MANAGEMENT REPORTS",
    "ALL-SHARES/ACCOUNTS/Audit - Accounts",
]


def _walk_folder(sid, path, tok, out, deadline, depth=0):
    """Every file under a SharePoint folder, depth-first."""
    if depth > 6 or time.time() >= deadline:
        return
    url = (GRAPH + "/sites/%s/drive/root:/%s:/children"
           "?$select=id,name,size,file,folder,parentReference,eTag&$top=200"
           % (sid, urllib.parse.quote(path)))
    while url and time.time() < deadline:
        try:
            d = _get(url, tok)
        except Exception as e:
            LOG.warning("reconcile: cannot list %s: %s", path[:80], str(e)[:120])
            return
        for it in d.get("value", []):
            name = it.get("name")
            if not name:
                continue
            if "folder" in it:
                _walk_folder(sid, path + "/" + name, tok, out, deadline, depth + 1)
                continue
            if "file" not in it:
                continue
            ext = name.rsplit(".", 1)[1].lower() if "." in name else ""
            if ext not in DOC_EXTS or it.get("size", 0) > MAX_BYTES:
                continue
            out.append(it)
        url = d.get("@odata.nextLink")


def run_reconcile(prefixes=None, minutes=20):
    """Copy anything SharePoint has that the blob does not.

    The delta cursor is not evidence. It reported "caught up" while an entire
    folder created on 8 Aug 2026 - the Entity Register - had never been
    copied, with no error logged: a change can be missed and delta will never
    mention it again, because from its point of view nothing has changed since.
    So for the folders that matter, ask SharePoint what it actually holds and
    fill the gaps. Cheap (hundreds of files), and it self-heals silent loss."""
    deadline = time.time() + minutes * 60
    svc = _svc()
    cc = svc.get_container_client(CONTAINER)
    tok = _token()
    sid = _site_drive_root(tok)
    checked = copied = missing = errors = 0
    gaps = []
    for prefix in (prefixes or RECONCILE_PREFIXES):
        if time.time() >= deadline:
            break
        items = []
        _walk_folder(sid, prefix, tok, items, deadline)
        for it in items:
            checked += 1
            dest = _dest_name(it)
            etag = (it.get("eTag") or "").replace('"', "")
            try:
                props = cc.get_blob_client(dest).get_blob_properties()
                if not etag or (props.metadata or {}).get("src_etag") == etag:
                    continue
            except Exception:
                pass                       # absent - copy it
            missing += 1
            if time.time() >= deadline:
                break
            try:
                item = _get(GRAPH + "/sites/%s/drive/items/%s" % (sid, it["id"]), tok)
                dl = item.get("@microsoft.graph.downloadUrl")
                if not dl:
                    errors += 1
                    continue
                bc = cc.get_blob_client(dest)
                bc.upload_blob_from_url(dl, overwrite=True)
                bc.set_blob_metadata({"src_etag": etag, "src_id": it["id"]})
                copied += 1
                if len(gaps) < 25:
                    gaps.append(dest)
            except Exception as e:
                errors += 1
                LOG.warning("reconcile copy failed %s: %s", dest[:90], str(e)[:150])
    LOG.info("reconcile: %d checked, %d missing, %d copied, %d errors",
             checked, missing, copied, errors)
    return {"checked": checked, "missing": missing, "copied": copied,
            "errors": errors, "examples": gaps,
            "secs": round(time.time() - deadline + minutes * 60)}
