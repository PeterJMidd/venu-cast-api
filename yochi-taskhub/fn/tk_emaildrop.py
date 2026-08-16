# -*- coding: utf-8 -*-
"""Email -> task pipeline (the no-admin-consent alternative to the Outlook
add-in): a Power Automate flow saves each flagged email as a small JSON file
into SharePoint 'ALL-SHARES/Finance/SYSTEMS/TaskHub Email Drop'; this module
polls the folder every 15 minutes via Graph (read-only - the app registration
already has that), AI-routes each new email to a pillar/project, and creates
the task. Dedup = tasks.external_ref 'email:<message id or filename>' (full
unique index), so files never need moving or deleting."""
import datetime as dt
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

import tk_ai
import tk_db
import tk_register

LOG = logging.getLogger("tk_emaildrop")

DROP_PATH = os.environ.get(
    "EMAILDROP_PATH", "ALL-SHARES/Finance/SYSTEMS/TaskHub Email Drop")
MAX_PER_RUN = 20
MAX_AGE_DAYS = 14
FALLBACK_PROJECT = "aaaaaaaa-0000-0000-0000-000000000019"  # Finance operations

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "project": {"type": "string",
                    "description": "EXACT project name from the provided list"},
        "priority": {"type": "string",
                     "enum": ["low", "medium", "high", "critical"]},
        "due_date": {"type": "string",
                     "description": "YYYY-MM-DD if the email implies a deadline, else empty"},
        "title": {"type": "string",
                  "description": "sharp task title (imperative, <=90 chars); "
                                 "default to the subject cleaned of Re:/Fwd:"}},
    "required": ["project", "priority", "due_date", "title"]}


def _graph(path):
    tok = tk_register._token()
    req = urllib.request.Request("https://graph.microsoft.com/v1.0" + path,
                                 headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _site_id():
    return _graph("/sites/" + tk_register.SP_HOST_PATH)["id"]


def _list_drop(site_id):
    path = "/sites/%s/drive/root:/%s:/children?$select=id,name,size,createdDateTime&$top=200" % (
        site_id, urllib.parse.quote(DROP_PATH))
    try:
        return _graph(path).get("value", [])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # folder not synced up yet
        raise


def _download(site_id, item_id):
    tok = tk_register._token()
    req = urllib.request.Request(
        "https://graph.microsoft.com/v1.0/sites/%s/drive/items/%s/content" % (site_id, item_id),
        headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _ref(mail, fname):
    mid = (mail.get("messageId") or "").strip()
    return "email:" + (mid if mid else fname)


def _classify(mail, projects):
    names = [p["name"] for p in projects]
    cats = {c["id"]: c["name"] for c in tk_db.get("categories", {"select": "id,name"})}
    listing = "\n".join("- %s (pillar: %s)" % (p["name"], cats.get(p["category_id"], ""))
                        for p in projects)
    out = tk_ai.structured(
        "Route an email the CFO of Yo-Chi (AU frozen-yoghurt chain) flagged as a "
        "task into the right finance project. Pick the SINGLE best project from "
        "the list (exact name). Priority: critical only for regulator/bank/board "
        "deadlines or money at risk now; high for dated commitments; else medium. "
        "Due date only when the email clearly implies one. Today is %s."
        % dt.date.today().isoformat(),
        "PROJECTS:\n%s\n\nEMAIL:\nSubject: %s\nFrom: %s\nReceived: %s\nBody:\n%s"
        % (listing, mail.get("subject", ""), mail.get("from", ""),
           mail.get("received", ""), (mail.get("body") or "")[:2000]),
        "route_email", CLASSIFY_SCHEMA, max_tokens=500)
    if out.get("project") not in names:
        out["project"] = None
    return out


def run():
    site_id = _site_id()
    items = _list_drop(site_id)
    if items is None:
        return {"skipped": "drop folder not found in SharePoint yet"}
    cutoff = (dt.datetime.utcnow() - dt.timedelta(days=MAX_AGE_DAYS)).isoformat()
    jsons = sorted([i for i in items if i["name"].lower().endswith(".json")
                    and i.get("createdDateTime", "") >= cutoff],
                   key=lambda i: i.get("createdDateTime", ""), reverse=True)[:MAX_PER_RUN]
    if not jsons:
        return {"files": 0, "created": 0}
    existing = {t["external_ref"] for t in tk_db.get(
        "tasks", {"external_ref": "like.email:*", "select": "external_ref"})}
    profiles = tk_db.get("profiles", {"active": "eq.true",
                                      "select": "id,email,role"})
    by_email = {p["email"].lower(): p["id"] for p in profiles}
    admin_uid = next((p["id"] for p in profiles if p["role"] == "admin"), None)
    projects = tk_db.get("projects", {"select": "id,name,category_id"})
    by_name = {p["name"]: p["id"] for p in projects}

    created, skipped, errors = [], 0, []
    for it in jsons:
        try:
            mail = json.loads(_download(site_id, it["id"]).decode("utf-8-sig"))
            ref = _ref(mail, it["name"])
            if ref in existing:
                skipped += 1
                continue
            c = _classify(mail, projects)
            desc = "From email - %s (%s)\n\n%s" % (
                mail.get("from", ""), mail.get("received", ""),
                (mail.get("body") or "")[:2500])
            row = {
                "project_id": by_name.get(c.get("project")) or FALLBACK_PROJECT,
                "title": (c.get("title") or mail.get("subject") or "Email task")[:200],
                "description": desc,
                "priority": c.get("priority", "medium"),
                "due_date": c.get("due_date") or None,
                "assignee_id": by_email.get((mail.get("owner_email") or "").lower(),
                                            admin_uid),
                "external_ref": ref,
                "created_by": admin_uid,
            }
            tk_db.insert("tasks", [row], on_conflict="external_ref",
                         ignore_duplicates=True)
            existing.add(ref)
            created.append({"title": row["title"], "project": c.get("project")})
        except Exception as e:
            LOG.exception("email drop file %s failed", it["name"])
            errors.append({"file": it["name"], "error": str(e)[:200]})
    out = {"files": len(jsons), "created": len(created), "already_tasks": skipped,
           "tasks": created}
    if errors:
        out["errors"] = errors
    return out
