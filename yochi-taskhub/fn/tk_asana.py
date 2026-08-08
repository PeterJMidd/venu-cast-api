# -*- coding: utf-8 -*-
"""Cloud Asana -> TaskHub sync (replaces Part A of the desktop scheduled task).
Pulls Peter's open Asana tasks via the REST API (ASANA_PAT), filters bot junk,
maps each to a TaskHub project + urgency, inserts with external_ref dedup, and
marks TaskHub copies done when completed in Asana. Credentials found in notes
are always redacted."""
import datetime as dt
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request

import tk_db

LOG = logging.getLogger("tk_asana")

API = "https://app.asana.com/api/1.0"
PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"

P = {n: "aaaaaaaa-0000-0000-0000-0000000000%02d" % i for n, i in {
    "close": 1, "bas": 2, "rep": 3, "fl": 11, "tx": 12, "uk": 13, "sea": 14,
    "pipe": 15, "intltax": 16, "entity": 17, "cfo": 18, "finops": 19, "people": 20,
}.items()}

JUNK = (re.compile(r"^Task \d+$"),
        re.compile(r"^(Please fill out|Alert: Asana invitation|Consider delegating|"
                   r"Review SLA compliance report|Review weekly status snapshot)"))
STAT_KW = ("bas ", "bas compliance", "fbt", "payroll tax", "superannuation",
           "tax registration", "taxation compliance", "qsr award", "vat", "gst filing")
INTL_KW = ("transfer pricing", "international structure", "fx exposure", "repatriat",
           "shareholder agreement", "jv doc", "withhold")
PAY_KW = ("payroll", "hire ", "recruit", "salary", "salaries", "wage", "proda",
          "unpaid leave", "employment")
REP_KW = ("power bi", "report", "dashboard", "datasights", "restoke", "api",
          "stocktake export", "redcat", "data ")


def _get(path, params=None):
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    req = urllib.request.Request(API + path + qs, headers={
        "Authorization": "Bearer " + os.environ["ASANA_PAT"]})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _fetch_open():
    ws = os.environ.get("ASANA_WORKSPACE", "1180506962903903")
    tasks, offset = [], None
    for _ in range(12):
        params = {"assignee": "me", "workspace": ws, "completed_since": "now",
                  "limit": "100",
                  "opt_fields": "gid,name,due_on,notes,projects.name,permalink_url"}
        if offset:
            params["offset"] = offset
        page = _get("/tasks", params)
        tasks += page.get("data", [])
        nxt = page.get("next_page") or {}
        offset = nxt.get("offset")
        if not offset:
            break
    return tasks


def _map_project(t):
    name = (t.get("name") or "").lower()
    projs = " | ".join((p.get("name") or "") for p in (t.get("projects") or [])).lower()
    both = name + " " + projs
    if "cfo" in projs and "100 day" in projs:
        if any(k in name for k in STAT_KW):
            return P["bas"]
        if any(k in name for k in INTL_KW):
            return P["intltax"]
        return P["cfo"]
    if "end of month timeline" in projs or re.search(r"\b\d+\. (january|february|march|april|may|june|july|august|september|october|november|december) 20\d\d", projs):
        return P["close"]
    if "international tax" in projs:
        return P["intltax"]
    if "franchisor compliance" in projs:
        return P["entity"]
    if "uk" in projs or "notting hill" in both:
        return P["uk"]
    if any(k in both for k in ("coral gables", "lincoln road", "miami", "florida")):
        return P["fl"]
    if "texas" in both:
        return P["tx"]
    if any(k in both for k in ("sea ", "central world", "singapore")):
        return P["sea"]
    if "new market launch" in projs or "all markets" in name:
        return P["pipe"]
    if any(k in both for k in PAY_KW):
        return P["people"]
    if any(k in both for k in STAT_KW):
        return P["bas"]
    if any(k in both for k in REP_KW):
        return P["rep"]
    return P["finops"]


def _priority_due(t):
    today = dt.date.today()
    due = None
    if t.get("due_on"):
        try:
            due = dt.date.fromisoformat(t["due_on"])
        except ValueError:
            due = None
    urgent = "urgent" in (t.get("name") or "").lower()
    if urgent:
        prio = "critical"
    elif due and (due <= today + dt.timedelta(days=7)) and due >= today - dt.timedelta(days=30):
        prio = "high"
    elif due and today < due <= today + dt.timedelta(days=60):
        prio = "medium"
    else:
        prio = "medium"
    keep_due = due is not None and due >= today - dt.timedelta(days=30)
    return prio, (due.isoformat() if keep_due else None), \
        (t.get("due_on") if (due and not keep_due) else None)


def _redact(text):
    text = re.sub(r"(?im)(password|passcode|pwd)\s*[:=]?\s*\S+",
                  r"\1: (credential redacted - see Asana)", text)
    return text


def run():
    existing = {r["external_ref"]: r for r in tk_db.get(
        "tasks", {"external_ref": "like.asana:*",
                  "select": "external_ref,id,status"})}
    fetched = _fetch_open()
    open_gids = set()
    rows, skipped = [], 0
    for t in fetched:
        name = (t.get("name") or "").strip()
        gid = t["gid"]
        open_gids.add(gid)
        if any(p.search(name) for p in JUNK):
            skipped += 1
            continue
        if ("asana:" + gid) in existing:
            continue
        proj = _map_project(t)
        prio, due, stale_due = _priority_due(t)
        desc = _redact((t.get("notes") or "").strip())[:1500]
        if stale_due:
            desc += "\n\n(Asana due date was %s - re-date as needed.)" % stale_due
        desc += "\n\nImported from Asana: %s" % t.get("permalink_url", "")
        rows.append({"project_id": proj, "title": name[:200], "description": desc.strip(),
                     "priority": prio, "assignee_id": PETER, "due_date": due,
                     "source": "manual", "external_ref": "asana:" + gid})
    inserted = []
    if rows:
        inserted = tk_db.insert("tasks", rows, on_conflict="external_ref",
                                ignore_duplicates=True, returning=True) or []

    # close TaskHub copies whose Asana original is done (open list no longer has it)
    closed = 0
    candidates = [r for ref, r in existing.items()
                  if r["status"] != "done" and ref.split(":", 1)[1] not in open_gids]
    for r in candidates[:60]:
        gid = r["external_ref"].split(":", 1)[1]
        try:
            detail = _get("/tasks/%s" % gid, {"opt_fields": "completed"})
            if detail.get("data", {}).get("completed"):
                tk_db.patch("tasks", {"id": "eq." + r["id"]},
                            {"status": "done",
                             "completed_at": dt.datetime.utcnow().isoformat() + "Z"})
                closed += 1
        except urllib.error.HTTPError as e:
            if e.code == 404:  # deleted in Asana - leave the TaskHub copy alone
                continue
            LOG.warning("close-check %s -> %s", gid, e.code)
    return {"fetched": len(fetched), "junk_skipped": skipped,
            "imported": len(inserted), "closed": closed}
