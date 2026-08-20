# -*- coding: utf-8 -*-
"""The complete compliance register.

Yochi_Group_Compliance_Calendar_FY27.xlsx is the register of record. It holds
12 monthly tabs (Jul-26..Jun-27, 679 obligation-instances over 155 refs, each
ticked against the entities it applies to) plus an Exceptions tab of things
already missed. This module makes that whole thing operational:

  SYNC       every row of every month becomes a compliance_items row
  TASKS      every item inside the rolling window becomes a TaskHub task,
             deduped on external_ref, so nothing lives only in a spreadsheet
  STATUS     task status flows back onto the register item, and the workbook
             cells we WOULD write are computed and held ready (see WRITEBACK)
  REVIEW     one call does the lot and reports what changed - this is what the
             'Review & update' button on the compliance page runs

WRITEBACK: pushing status back into the workbook needs Graph Sites.ReadWrite.All,
which this app registration does not have consent for (probe returns 403
accessDenied). Everything up to the write is built and the pending cell values
are stored on each item, so switching it on is a one-line change once consent
lands. Until then the register in TaskHub carries live status and the workbook
stays as the humans left it - we never silently half-write a source of truth."""
import calendar
import datetime as dt
import io
import json
import logging
import os
import re
import urllib.parse
import urllib.request

import tk_db

LOG = logging.getLogger("tk_compliance")

AEST = dt.timezone(dt.timedelta(hours=10))
SP_HOST_PATH = "embraceyochi.sharepoint.com:/sites/YochiTeamSharepoint"
FILE_PATH = ("ALL-SHARES/Finance/GROUP STRUCTURE & CORPORATE/"
             "Entity Register - Aug 2026/00. Group/"
             "Yochi_Group_Compliance_Calendar_FY27.xlsx")
LIVE_LINK = ("https://embraceyochi.sharepoint.com/sites/YochiTeamSharepoint/"
             "_layouts/15/Doc.aspx?sourcedoc=%7B8D2B72BA-36BB-47F8-91EB-"
             "0C2C4F2D6D0E%7D&file=Yochi_Group_Compliance_Calendar_FY27.xlsx"
             "&action=default&mobileredirect=true")
PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"

WINDOW_AHEAD = 360          # rolling year of obligations
WINDOW_BACK = 60            # recently-passed ones stay visible until closed

HDR_ROW = 5                 # 1-based row of the column headers on a month tab
COL = {"ref": 0, "category": 1, "obligation": 2, "what_to_do": 3,
       "authority": 4, "frequency": 5, "due_text": 6, "owner": 7, "risk": 8,
       "status": 29, "completed_date": 30, "evidence": 31, "notes": 32}
ENTITY_FIRST, ENTITY_LAST = 9, 28

MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
P = lambda n: "aaaaaaaa-0000-0000-0000-0000000000%02d" % n
# category number -> project. Same mapping the old register sync used, so
# existing creg: tasks keep landing where the team already expects them.
CAT_PROJECT = {1: P(17), 2: P(17), 3: P(2), 4: P(2), 5: P(2), 6: P(2),
               7: P(21), 8: P(22), 9: P(22), 10: P(14), 11: P(11),
               12: P(13), 13: P(16), 14: P(16), 15: P(21), 16: P(21),
               17: P(21)}
DEFAULT_PROJECT = P(17)


def _token():
    body = urllib.parse.urlencode({
        "client_id": os.environ["CLIENT_ID"],
        "client_secret": os.environ["CLIENT_SECRET"],
        "grant_type": "client_credentials",
        "scope": "https://graph.microsoft.com/.default"}).encode()
    req = urllib.request.Request(
        "https://login.microsoftonline.com/%s/oauth2/v2.0/token" % os.environ["TENANT_ID"],
        data=body)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["access_token"]


def download():
    h = {"Authorization": "Bearer " + _token()}
    req = urllib.request.Request(
        "https://graph.microsoft.com/v1.0/sites/%s" % SP_HOST_PATH, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        site_id = json.loads(r.read().decode())["id"]
    url = "https://graph.microsoft.com/v1.0/sites/%s/drive/root:/%s:/content" % (
        site_id, urllib.parse.quote(FILE_PATH))
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
        return r.read()


def _txt(v):
    if v is None:
        return ""
    if isinstance(v, dt.datetime):
        return v.date().isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    return re.sub(r"\s+", " ", str(v)).strip()


def _sheet_period(name):
    """'Sep-26' -> (2026, 9). None for the non-month tabs."""
    m = re.match(r"([A-Za-z]{3})[-\s]?(\d{2,4})$", (name or "").strip())
    if not m or m.group(1).lower() not in MONTHS:
        return None
    y = int(m.group(2))
    return (2000 + y if y < 100 else y, MONTHS[m.group(1).lower()])


def _eom(y, m):
    return dt.date(y, m, calendar.monthrange(y, m)[1])


def _due(due_text, y, m):
    """A real date from the workbook's relative phrasing.

    The Due date column is written for humans ('By the 15th of the following
    month', 'Q3', 'Ongoing - month-end check'), so resolve it against the month
    tab the row sits on and record HOW it was resolved."""
    t = (due_text or "").lower()
    exact = re.search(r"(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", t)
    if exact:
        try:
            return dt.date(int(exact.group(3)), MONTHS[exact.group(2)],
                           int(exact.group(1))), "explicit date in the workbook"
        except ValueError:
            pass
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if iso:
        try:
            return dt.date(*map(int, iso.groups())), "explicit date in the workbook"
        except ValueError:
            pass
    nxt = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+of the (?:following|next|2nd|second)\s+month", t)
    if nxt:
        day = int(nxt.group(1))
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        if "2nd" in t or "second" in t:
            ny, nm = (ny + 1, 1) if nm == 12 else (ny, nm + 1)
        return dt.date(ny, nm, min(day, calendar.monthrange(ny, nm)[1])), \
            "day-of-following-month from '%s'" % (due_text or "")[:40]
    within = re.search(r"within\s+(\d{1,2})\s+days?\s+of\s+(?:the\s+)?month[\s-]?end", t)
    if within:
        return _eom(y, m) + dt.timedelta(days=int(within.group(1))), \
            "month end + %s days from '%s'" % (within.group(1), (due_text or "")[:34])
    day = re.search(r"(?:by the|typically|on the)\s+(\d{1,2})(?:st|nd|rd|th)", t)
    if day:
        d = int(day.group(1))
        return dt.date(y, m, min(d, calendar.monthrange(y, m)[1])), \
            "day-of-month from '%s'" % (due_text or "")[:40]
    return _eom(y, m), "month end (the row's month tab)"


def _catnum(category):
    m = re.match(r"\s*(\d{1,2})\s*[.)]", category or "")
    return int(m.group(1)) if m else None


_ENTITY_PROJECT = ((("texas",), P(12)), (("florida",), P(11)),
                   (("uk", "united kingdom", "yo-chi uk", "i love yo-chi"), P(13)),
                   (("singapore", "asia pte", "sea pte"), P(14)))


def _project_for(category, entities):
    n = _catnum(category)
    pid = CAT_PROJECT.get(n, DEFAULT_PROJECT)
    if n == 11 and "texas" in (entities or "").lower():
        pid = P(12)                     # US foreign statutory splits by state
    if n is None:
        # document-review rows have no category number - route by the entity
        e = (entities or "").lower()
        for words, proj in _ENTITY_PROJECT:
            if any(w in e for w in words):
                return proj
    return pid


def parse_calendar(data):
    """Every obligation-instance on every month tab."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    items = []
    for name in wb.sheetnames:
        per = _sheet_period(name)
        if not per:
            continue
        y, m = per
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) <= HDR_ROW:
            continue
        hdr = [_txt(h) for h in rows[HDR_ROW - 1]]
        for raw in rows[HDR_ROW:]:
            ref = _txt(raw[COL["ref"]] if len(raw) > COL["ref"] else "")
            obligation = _txt(raw[COL["obligation"]] if len(raw) > COL["obligation"] else "")
            if not ref or not obligation:
                continue                # category banner rows carry no obligation
            get = lambda k: _txt(raw[COL[k]]) if len(raw) > COL[k] else ""
            ents = [hdr[i] for i in range(ENTITY_FIRST, min(ENTITY_LAST + 1, len(raw)))
                    if _txt(raw[i]) and i < len(hdr)]
            due, basis = _due(get("due_text"), y, m)
            items.append({
                "ref": ref, "period": "%04d-%02d" % (y, m), "stream": "calendar",
                "category": get("category"), "obligation": obligation,
                "what_to_do": get("what_to_do"), "authority": get("authority"),
                "frequency": get("frequency"), "due_text": get("due_text"),
                "due_date": due.isoformat(), "due_basis": basis,
                "owner": get("owner"), "risk": get("risk"),
                "entities": ", ".join(ents)[:500],
                "sheet_status": get("status"), "completed_date": get("completed_date"),
                "evidence": get("evidence"), "notes": get("notes"),
            })
    return items


def parse_exceptions(data):
    """The Exceptions tab - obligations already missed, with an action each."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    if "Exceptions" not in wb.sheetnames:
        return []
    rows = list(wb["Exceptions"].iter_rows(values_only=True))
    hdr_i = next((i for i, r in enumerate(rows)
                  if r and _txt(r[0]).lower() == "#"), 3)
    hdr = [_txt(h).lower() for h in rows[hdr_i]]

    def col(*names):
        for n in names:
            for i, h in enumerate(hdr):
                if h.startswith(n):
                    return i
        return None

    ci = {k: col(*v) for k, v in {
        "severity": ("severity",), "entity": ("entity",), "country": ("country",),
        "area": ("area",), "what": ("what appears", "what "),
        "date": ("statutory", "date"), "position": ("position",),
        "action": ("action",), "owner": ("owner",), "source": ("source",),
    }.items()}
    out = []
    for raw in rows[hdr_i + 1:]:
        ref = _txt(raw[0] if raw else "")
        if not re.match(r"^E\d+", ref, re.I):
            continue
        g = lambda k: _txt(raw[ci[k]]) if ci.get(k) is not None and ci[k] < len(raw) else ""
        what = g("what")
        if not what:
            continue
        due, basis = _exception_due(g("date"))
        out.append({
            "ref": ref, "period": "EXC", "stream": "exception",
            "category": "Exception - %s" % (g("area") or "compliance"),
            "obligation": what[:400],
            "what_to_do": g("action"), "authority": g("source"),
            "frequency": "One-off", "due_text": g("date"),
            "due_date": due.isoformat(), "due_basis": basis,
            "owner": g("owner"), "risk": g("severity"),
            "entities": ", ".join(x for x in (g("entity"), g("country")) if x),
            "sheet_status": g("position"), "completed_date": "",
            "evidence": g("source"), "notes": g("position"),
            "severity": g("severity"), "action": g("action"),
        })
    return out


def _exception_due(text):
    """Exceptions are already late; the statutory date is the due date."""
    t = (text or "").lower()
    m = re.search(r"(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", t)
    if m:
        try:
            return dt.date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))), \
                "statutory date on the Exceptions tab"
        except ValueError:
            pass
    today = dt.datetime.now(AEST).date()
    return today, "no date given - treated as due now"


ITEM_COLS = ("ref", "period", "stream", "category", "obligation", "what_to_do",
             "authority", "frequency", "due_text", "due_date", "due_basis",
             "owner", "risk", "entities", "sheet_status", "completed_date",
             "evidence", "notes", "severity", "action")


def sync_items(items):
    """Upsert the register. Rows that vanish from the workbook go inactive
    rather than being deleted - we keep the audit trail."""
    # every row must carry the SAME keys: PostgREST rejects a bulk insert whose
    # objects differ ("All object keys must match"), so absent fields are
    # explicit nulls rather than omitted
    now = dt.datetime.utcnow().isoformat() + "Z"
    rows = [dict({k: (it.get(k) if it.get(k) not in ("",) else None)
                  for k in ITEM_COLS}, active=True, updated_at=now)
            for it in items]
    for i in range(0, len(rows), 200):
        tk_db.insert("compliance_items", rows[i:i + 200],
                     on_conflict="ref,period", merge_duplicates=True)
    seen = {(r["ref"], r["period"]) for r in rows}
    stale = [x for x in tk_db.get("compliance_items",
                                  {"active": "eq.true", "select": "id,ref,period"})
             if (x["ref"], x["period"]) not in seen]
    for x in stale:
        tk_db.patch("compliance_items", {"id": "eq." + x["id"]}, {"active": False})
    return {"upserted": len(rows), "deactivated": len(stale)}


# ------------------------------------------------------------------ tasks

_RISK_PRIORITY = {"critical": "critical", "high": "high", "medium": "medium",
                  "low": "low"}
_DONE_WORDS = ("complete", "done", "closed", "lodged", "filed", "paid",
               "submitted", "n/a", "not applicable")


def _priority(item):
    r = (item.get("risk") or "").strip().lower()
    for k, v in _RISK_PRIORITY.items():
        if r.startswith(k):
            return v
    return "medium"


def _sheet_says_done(item):
    s = (item.get("sheet_status") or "").strip().lower()
    return bool(s) and any(s.startswith(w) for w in _DONE_WORDS)


def _task_ref(item):
    if item["stream"] == "exception":
        return "cexc:%s" % item["ref"]
    if item["stream"] == "document":
        return "cdoc:%s" % item["ref"]
    return "creg:%s:%s" % (item["ref"], item["period"])


def _title(item):
    if item["stream"] == "document":
        return "[Register gap] %s" % item["obligation"][:120]
    if item["stream"] == "exception":
        return "[%s] %s — %s" % (item["ref"], (item.get("severity") or "Exception"),
                                 item["obligation"][:110])
    y, m = item["period"].split("-")
    mon = dt.date(int(y), int(m), 1).strftime("%b %y")
    return "%s — %s (%s)" % (item["ref"], item["obligation"][:120], mon)


def _description(item):
    bits = [item.get("what_to_do") or item.get("obligation") or ""]
    for label, key in (("Authority", "authority"), ("Frequency", "frequency"),
                       ("Due per workbook", "due_text"), ("Applies to", "entities"),
                       ("Owner", "owner"), ("Risk", "risk"),
                       ("Action required", "action"),
                       ("Position at last review", "notes")):
        v = item.get(key)
        if v:
            bits.append("%s: %s" % (label, v))
    bits.append("Due date derived: %s" % item.get("due_basis", ""))
    bits.append("")
    bits.append("Register ref %s%s — Yochi_Group_Compliance_Calendar_FY27.xlsx, "
                "%s tab. The register is the source of truth; this task is kept "
                "in step with it by the daily compliance sync." % (
                    item["ref"],
                    "" if item["stream"] == "exception" else " / " + item["period"],
                    "Exceptions" if item["stream"] == "exception" else item["period"]))
    bits.append(LIVE_LINK)
    return "\n".join(b for b in bits if b)


def _assignee(owner):
    o = (owner or "").lower()
    return PETER if ("peter" in o or "cfo" in o) else None


def in_window(item, today):
    # exceptions and document gaps are open findings, not scheduled work: they
    # stay in scope however old they are, or the oldest misses - exactly the
    # ones that matter - would quietly age out of the board
    if item.get("stream") in ("exception", "document"):
        return True
    if not item.get("due_date"):
        return False
    d = dt.date.fromisoformat(item["due_date"])
    return today - dt.timedelta(days=WINDOW_BACK) <= d <= today + dt.timedelta(days=WINDOW_AHEAD)


def reconcile_tasks(items, today=None, apply=True):
    """Every in-window item gets a task; existing ones are corrected, not
    duplicated. Items the workbook already marks complete never raise a task."""
    today = today or dt.datetime.now(AEST).date()
    want = [i for i in items if in_window(i, today) and not _sheet_says_done(i)]
    refs = [_task_ref(i) for i in want]

    existing = {}
    for i in range(0, len(refs), 100):
        chunk = refs[i:i + 100]
        for t in tk_db.get("tasks", {
                "external_ref": "in.(%s)" % ",".join('"%s"' % r for r in chunk),
                "select": "id,external_ref,title,due_date,status,priority,project_id"}):
            existing[t["external_ref"]] = t

    added, amended, closed = [], [], []
    new_rows = []
    for item in want:
        ref = _task_ref(item)
        spec = {
            "project_id": _project_for(item.get("category"), item.get("entities")),
            "title": _title(item), "description": _description(item),
            "priority": _priority(item), "due_date": item["due_date"],
            "source": "watcher", "external_ref": ref,
            "assignee_id": _assignee(item.get("owner")),
        }
        cur = existing.get(ref)
        if not cur:
            new_rows.append(spec)
            added.append({"ref": ref, "title": spec["title"], "due": spec["due_date"]})
            continue
        item["task_id"], item["task_status"] = cur["id"], cur["status"]
        if cur["status"] == "done":
            continue
        diff = []
        if (cur.get("due_date") or "") != spec["due_date"]:
            diff.append("due %s -> %s" % (cur.get("due_date") or "(none)", spec["due_date"]))
        if (cur.get("title") or "") != spec["title"]:
            diff.append("title updated")
        if (cur.get("priority") or "") != spec["priority"]:
            diff.append("priority %s -> %s" % (cur.get("priority"), spec["priority"]))
        if diff:
            if apply:
                tk_db.patch("tasks", {"id": "eq." + cur["id"]},
                            {"title": spec["title"], "due_date": spec["due_date"],
                             "priority": spec["priority"],
                             "description": spec["description"]})
            amended.append({"ref": ref, "title": spec["title"],
                            "changes": "; ".join(diff)})

    if apply and new_rows:
        for i in range(0, len(new_rows), 100):
            tk_db.insert("tasks", new_rows[i:i + 100], on_conflict="external_ref",
                         ignore_duplicates=True)

    # the workbook marking something complete closes the task: the register is
    # the source of truth, and this is the sync direction that is not blocked
    done_refs = [_task_ref(i) for i in items
                 if _sheet_says_done(i) and in_window(i, today)]
    for i in range(0, len(done_refs), 100):
        chunk = done_refs[i:i + 100]
        for t in tk_db.get("tasks", {
                "external_ref": "in.(%s)" % ",".join('"%s"' % r for r in chunk),
                "status": "neq.done", "select": "id,external_ref,title"}):
            if apply:
                tk_db.patch("tasks", {"id": "eq." + t["id"]},
                            {"status": "done",
                             "completed_at": dt.datetime.utcnow().isoformat() + "Z"})
            closed.append({"ref": t["external_ref"], "title": t["title"][:80]})
    return {"added": added, "amended": amended, "closed": closed,
            "in_window": len(want)}


def status_flow(items, apply=True):
    """Task status back onto the register item, and the workbook cell values we
    would write if Sites.ReadWrite.All consent were in place."""
    refs = {_task_ref(i): i for i in items}
    keys = list(refs)
    ready = 0
    for i in range(0, len(keys), 100):
        chunk = keys[i:i + 100]
        for t in tk_db.get("tasks", {
                "external_ref": "in.(%s)" % ",".join('"%s"' % r for r in chunk),
                "select": "id,external_ref,status,completed_at"}):
            item = refs.get(t["external_ref"])
            if not item:
                continue
            item["task_id"], item["task_status"] = t["id"], t["status"]
            if t["status"] == "done" and not _sheet_says_done(item):
                item["writeback"] = "Complete|%s" % (t.get("completed_at") or "")[:10]
                ready += 1
    if apply:
        # only the rows that actually moved: a PATCH per item was 700+ round
        # trips every run, most of them writing back what was already stored
        stored = {(r["ref"], r["period"]): r for r in tk_db.get(
            "compliance_items",
            {"select": "ref,period,task_id,task_status,writeback", "limit": "5000"})}
        changed = 0
        for item in items:
            if not item.get("task_id"):
                continue
            was = stored.get((item["ref"], item["period"])) or {}
            new_vals = {"task_id": item["task_id"],
                        "task_status": item.get("task_status"),
                        "writeback": item.get("writeback")}
            if all((was.get(k) or None) == (v or None) for k, v in new_vals.items()):
                continue
            tk_db.patch("compliance_items",
                        {"ref": "eq." + item["ref"], "period": "eq." + item["period"]},
                        new_vals)
            changed += 1
        LOG.info("status flow: %d item(s) changed", changed)
    return {"writeback_ready": ready}


# ------------------------------------------------- review against the deeds

# The legal documents that define what we are actually bound to. The register
# is a secondary source - it was written FROM these, so these are what we check
# it against.
SOURCE_DOCS = [
    "00. Group/Yochi Group - Compliance and Risk Checklist - Aug 2026.docx",
    "00. Group/Yochi Group - Issues Actions and Observations - Aug 2026.docx",
    "00. Group/Exec summary of License and MFA agreement for UK, USA, Sing.docx",
    "00. Group/Executive summary for the JV" + chr(39) + "s UK, USA, Sing.docx",
    "00. Group/Executive Summary of Yochi Intellectual Property License Agreement.docx",
    "04. Yochi Franchising Pty Ltd/Executive Summary of Yochi Franchise Agreement.docx",
    "04. Yochi Franchising Pty Ltd/Yochi Franchise Agreement Template "
    "(pre 1 November 2025).V1.docx",
    "04. Yochi Franchising Pty Ltd/Yo-chi Disclosure Document - (pre 1 Nov 2025).docx",
    "10. Yochi Texas LLC/Executive Summary of the JV Company Agreement.docx",
    "01. Yochi Pty Ltd/Yochi Pty Ltd - Entity Overview - Aug 2026.docx",
    "02. Embrace the Chi Pty Ltd/Embrace the Chi Pty Ltd - Entity Overview - Aug 2026.docx",
    "03. Marroarchi Pty Ltd/Marroarchi Pty Ltd - Entity Overview - Aug 2026.docx",
    "04. Yochi Franchising Pty Ltd/Yochi Franchising Pty Ltd - Entity Overview - Aug 2026.docx",
    "05. Yochi IP Pty Ltd/Yochi IP Pty Ltd - Entity Overview - Aug 2026.docx",
    "06. Yochi Asia Pte Ltd/Yochi Asia Pte Ltd - Entity Overview - Aug 2026.docx",
    "07. Yochi SEA Pte Ltd/Yochi SEA Pte Ltd - Entity Overview - Aug 2026.docx",
    "08. Yochi USA LLC/Yochi USA LLC - Entity Overview - Aug 2026.docx",
    "09. Yochi Florida LLC/Yochi Florida LLC - Entity Overview - Aug 2026.docx",
    "10. Yochi Texas LLC/Yochi Texas LLC - Entity Overview - Aug 2026.docx",
    "11. Yo-Chi UK Holdings Limited/Yo-Chi UK Holdings Limited - Entity Overview - Aug 2026.docx",
    "12. I Love Yo-Chi Ltd/I Love Yo-Chi Ltd - Entity Overview - Aug 2026.docx",
]
DOC_ROOT = ("ALL-SHARES/Finance/GROUP STRUCTURE & CORPORATE/"
            "Entity Register - Aug 2026/")

GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obligation": {"type": "string", "description":
                                   "the recurring obligation, as an instruction"},
                    "entity": {"type": "string"},
                    "authority": {"type": "string", "description":
                                  "the clause, act or regulator that imposes it"},
                    "frequency": {"type": "string"},
                    "quote": {"type": "string", "description":
                              "short verbatim quote from THIS document that "
                              "imposes the obligation"},
                    "why_not_covered": {"type": "string", "description":
                                        "why the listed register refs do not "
                                        "already cover it"},
                    "materiality": {"type": "string",
                                    "enum": ["high", "medium", "low"]},
                },
                "required": ["obligation", "entity", "authority", "frequency",
                             "quote", "why_not_covered", "materiality"]}}},
    "required": ["gaps"]}

GAP_SYSTEM = (
    "You audit a group compliance register against the legal documents it was "
    "written from. You are given the register's existing obligations and ONE "
    "document. Find obligations the DOCUMENT imposes that the register does "
    "NOT already cover.\n\n"
    "Rules, in order of importance:\n"
    "1. Every gap MUST quote the document verbatim. No quote, no gap.\n"
    "2. If an existing ref covers it - even loosely, even under a different "
    "name - it is NOT a gap. Prefer saying nothing to padding the list.\n"
    "3. Only recurring or dated obligations. Background facts, commercial "
    "terms and history are not obligations.\n"
    "4. Never invent a clause, date, act or regulator that is not in the text.\n"
    "Return an empty list when the register already covers the document.")


def _docx_text(blob, limit=18000):
    """Paragraph text from a .docx without adding a dependency - the body is
    one XML part inside the zip."""
    import zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
    except Exception:
        return ""
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    txt = re.sub(r"<[^>]+>", "", xml)
    txt = txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()[:limit]


def _site_id(h):
    req = urllib.request.Request(
        "https://graph.microsoft.com/v1.0/sites/%s" % SP_HOST_PATH, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["id"]


def _fetch(path):
    h = {"Authorization": "Bearer " + _token()}
    url = "https://graph.microsoft.com/v1.0/sites/%s/drive/root:/%s:/content" % (
        _site_id(h), urllib.parse.quote(DOC_ROOT + path))
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
        return r.read()


def _register_digest(items):
    """The register condensed to one line per ref - what the auditor is told
    already exists."""
    seen, lines = set(), []
    for i in items:
        if i["ref"] in seen:
            continue
        seen.add(i["ref"])
        lines.append("%s | %s | %s | %s" % (
            i["ref"], (i.get("category") or "")[:34],
            (i.get("obligation") or "")[:90], (i.get("frequency") or "")[:22]))
    return "\n".join(sorted(lines))


def document_review(items, docs=None, apply=True):
    """Read the deeds and entity overviews and report obligations the register
    does not carry. Never raises - a document that will not open is reported."""
    import tk_ai
    digest = _register_digest(items)
    docs = docs or SOURCE_DOCS
    gaps, errors, read = [], [], 0
    for path in docs:
        name = path.split("/")[-1]
        try:
            text = _docx_text(_fetch(path))
            if len(text) < 200:
                errors.append({"doc": name, "error": "no readable text"})
                continue
            read += 1
            out = tk_ai.structured(
                GAP_SYSTEM,
                "REGISTER - obligations already covered (ref | category | "
                "obligation | frequency):\n%s\n\n=== DOCUMENT: %s ===\n%s"
                % (digest, name, text),
                "report_gaps", GAP_SCHEMA, max_tokens=2000)
            found = (out or {}).get("gaps", []) or []
            for g in found:
                g["document"] = name
                gaps.append(g)
            LOG.info("document review %s -> %d gaps", name, len(found))
        except Exception as e:
            LOG.exception("document review failed for %s", name)
            errors.append({"doc": name, "error": str(e)[:200]})

    if apply and gaps:
        rows = {}
        today = dt.datetime.now(AEST).date()
        for g in gaps:
            slug = re.sub(r"[^a-z0-9]+", "-",
                          (g.get("obligation") or "").lower())[:36].strip("-")
            row = {
                "ref": "DOC-" + slug, "period": "DOC", "stream": "document",
                "category": "Document review - not in the workbook",
                "obligation": (g.get("obligation") or "")[:400],
                "what_to_do": (g.get("why_not_covered") or "")[:400],
                "authority": (g.get("authority") or "")[:300],
                "frequency": (g.get("frequency") or "")[:80],
                "due_text": "from " + g.get("document", ""),
                "due_date": (today + dt.timedelta(days=21)).isoformat(),
                "due_basis": "document review - needs a decision, not a deadline",
                "risk": g.get("materiality", "medium"),
                "entities": (g.get("entity") or "")[:200],
                "sheet_status": "Not in the workbook",
                "evidence": (g.get("quote") or "")[:900],
                "notes": "Found in %s" % g.get("document", ""),
                "severity": g.get("materiality"),
                "active": True,
                "updated_at": dt.datetime.utcnow().isoformat() + "Z"}
            rows[(row["ref"], row["period"])] = row
        tk_db.insert("compliance_items", list(rows.values()),
                     on_conflict="ref,period", merge_duplicates=True)
    return {"gaps": gaps, "errors": errors, "documents_read": read,
            "documents_attempted": len(docs)}


# ------------------------------------------------------------- the button

def review(apply=True, documents=False, today=None, ran_by="timer"):
    """One pass over everything: re-read the workbook, refresh the register,
    push work into tasks, pull status back, and (optionally) re-audit the
    legal documents. This is what 'Review & update' runs."""
    today = today or dt.datetime.now(AEST).date()
    data = download()
    items = parse_calendar(data) + parse_exceptions(data)

    synced = sync_items(items) if apply else {"upserted": 0, "deactivated": 0}
    tasks = reconcile_tasks(items, today=today, apply=apply)
    status = status_flow(items, apply=apply)

    docs = {"gaps": [], "errors": [], "documents_read": 0}
    if documents:
        docs = document_review(items, apply=apply)

    overdue = [i for i in items
               if i.get("due_date") and i["due_date"] < today.isoformat()
               and not _sheet_says_done(i)]
    flags = []
    if docs["gaps"]:
        flags.append({"kind": "document-gap",
                      "detail": "%d obligation(s) in the legal documents are "
                                "not in the workbook" % len(docs["gaps"])})
    for e in docs["errors"]:
        flags.append({"kind": "document-unreadable",
                      "detail": "%s: %s" % (e["doc"], e["error"])})
    if status["writeback_ready"]:
        flags.append({"kind": "writeback-blocked",
                      "detail": "%d item(s) are complete in TaskHub and would be "
                                "written back to the workbook, but the app has no "
                                "SharePoint write consent (Sites.ReadWrite.All)"
                                % status["writeback_ready"]})

    res = {
        "date": today.isoformat(),
        "items": len(items),
        "calendar_items": sum(1 for i in items if i["stream"] == "calendar"),
        "exceptions": sum(1 for i in items if i["stream"] == "exception"),
        "refs": len({i["ref"] for i in items}),
        "in_window": tasks["in_window"],
        "overdue": len(overdue),
        "synced": synced,
        "added": tasks["added"], "amended": tasks["amended"],
        "closed": tasks["closed"],
        "writeback_ready": status["writeback_ready"],
        "document_gaps": docs["gaps"], "document_errors": docs["errors"],
        "documents_read": docs.get("documents_read", 0),
        "flags": flags, "applied": bool(apply), "live_link": LIVE_LINK,
    }
    if apply:
        try:
            tk_db.insert("compliance_runs", [{
                "ran_by": ran_by, "items": len(items),
                "tasks_added": len(tasks["added"]),
                "tasks_amended": len(tasks["amended"]),
                "tasks_linked": len(tasks["closed"]),
                "writeback_ready": status["writeback_ready"],
                "flags": flags,
                "detail": {k: res[k] for k in
                           ("calendar_items", "exceptions", "refs", "in_window",
                            "overdue", "documents_read")}}])
        except Exception:
            LOG.exception("could not record the compliance run")
    LOG.info("compliance review: %d items, %d added, %d amended, %d closed, "
             "%d gaps", len(items), len(tasks["added"]), len(tasks["amended"]),
             len(tasks["closed"]), len(docs["gaps"]))
    return res


def summary():
    """Register state for the compliance page - no SharePoint round trip."""
    today = dt.datetime.now(AEST).date()
    rows = tk_db.get("compliance_items", {
        "active": "eq.true", "select":
        "ref,period,stream,category,obligation,authority,frequency,due_date,"
        "due_text,owner,risk,entities,sheet_status,task_id,task_status,"
        "writeback,evidence,notes,severity",
        "order": "due_date.asc", "limit": "2000"})
    horizon = (today + dt.timedelta(days=WINDOW_AHEAD)).isoformat()
    live = [r for r in rows if (r.get("due_date") or "") <= horizon]
    open_items = [r for r in live if (r.get("task_status") or "todo") != "done"]
    return {
        "live_link": LIVE_LINK,
        "counts": {
            "total": len(rows),
            "calendar": sum(1 for r in rows if r["stream"] == "calendar"),
            "exceptions": sum(1 for r in rows if r["stream"] == "exception"),
            "document_gaps": sum(1 for r in rows if r["stream"] == "document"),
            "in_window": len(live),
            "open": len(open_items),
            "done": len(live) - len(open_items),
            "overdue": sum(1 for r in open_items
                           if (r.get("due_date") or "") < today.isoformat()),
            "writeback_pending": sum(1 for r in rows if r.get("writeback")),
        },
        "last_run": (tk_db.get("compliance_runs", {
            "select": "ran_at,ran_by,items,tasks_added,tasks_amended,"
                      "tasks_linked,writeback_ready,flags",
            "order": "ran_at.desc", "limit": "1"}) or [None])[0],
        "items": rows,
    }


def _html(res):
    def table(items, cols, empty="none"):
        if not items:
            return ("<tr><td colspan='%d' style='padding:8px;color:#888'>%s</td></tr>"
                    % (len(cols), empty))
        return "".join(
            "<tr>" + "".join(
                "<td style='padding:6px 10px;border-top:1px solid #eee;"
                "vertical-align:top'>%s</td>" % str(it.get(c, "") or "")[:200]
                for c in cols) + "</tr>" for it in items[:40])

    h = ["<div style=\"font-family:-apple-system,Segoe UI,Arial,sans-serif;"
         "max-width:900px;color:#222\">",
         "<h2 style='margin:0 0 4px'>Compliance register</h2>",
         "<div style='color:#666;font-size:13px;margin-bottom:14px'>%s &middot; "
         "%d obligations (%d calendar, %d exceptions) over %d refs &middot; "
         "%d in the rolling year</div>" % (
             res["date"], res["items"], res["calendar_items"], res["exceptions"],
             res["refs"], res["in_window"]),
         "<div style='font-size:15px;margin-bottom:16px'><b>%d</b> tasks added "
         "&nbsp;|&nbsp; <b>%d</b> amended &nbsp;|&nbsp; <b>%d</b> closed from the "
         "workbook &nbsp;|&nbsp; <b>%d</b> overdue</div>" % (
             len(res["added"]), len(res["amended"]), len(res["closed"]),
             res["overdue"])]
    for title, items, cols in (
            ("Added", res["added"], ("title", "due")),
            ("Amended", res["amended"], ("title", "changes")),
            ("Closed because the workbook says complete", res["closed"], ("ref", "title")),
            ("In the deeds but not the workbook", res["document_gaps"],
             ("obligation", "entity", "authority", "document")),
            ("Flags", res["flags"], ("kind", "detail"))):
        h.append("<h3 style='margin:18px 0 6px;font-size:14px'>%s</h3>"
                 "<table style='border-collapse:collapse;width:100%%;"
                 "font-size:13px'>%s</table>" % (title, table(items, cols)))
    h.append("<p style='margin-top:18px;font-size:12px;color:#888'>Register of "
             "record: <a href='%s'>Yochi_Group_Compliance_Calendar_FY27.xlsx</a>"
             " &middot; status in TaskHub is live; writing status back into the "
             "workbook is waiting on SharePoint write consent.</p></div>"
             % res["live_link"])
    return "".join(h)


def run(apply=True, documents=False, email=True, ran_by="timer"):
    res = review(apply=apply, documents=documents, ran_by=ran_by)
    if email:
        try:
            import tk_email
            if tk_email.enabled():
                tk_email.send(
                    os.environ.get("COMPLIANCE_TO", "peterm@yochi.com.au"),
                    "Compliance register - %d added, %d amended, %d overdue" % (
                        len(res["added"]), len(res["amended"]), res["overdue"]),
                    _html(res))
                res["emailed"] = True
        except Exception:
            LOG.exception("compliance email failed (the review itself succeeded)")
    return res
