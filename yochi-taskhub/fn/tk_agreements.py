# -*- coding: utf-8 -*-
"""Signed agreements check: Yochi_Signed_Agreements_Tracker_v2.xlsx is the
SOURCE OF TRUTH for the international deals, so every day we read it LIVE from
SharePoint (Graph) and make TaskHub agree with it.

Three things happen, in the language Peter asked for:
  ADDED    an obligation the tracker implies but TaskHub has no task for
  AMENDED  a task whose due date or wording no longer matches the tracker
  FLAGGED  gaps in the tracker itself (TBC/blank), obligations already overdue,
           tracker fields that CHANGED since yesterday, and open tasks whose
           obligation has disappeared from the tracker

The tracker is never written to - it is the source of truth, not a workspace.
Asana is not touched. Dedup/idempotency = tasks.external_ref 'agmt:<slug>:<kind>'.
Runs daily 07:45 AEST (before the 08:30 work report) + run_agreements."""
import datetime as dt
import hashlib
import io
import json
import logging
import os
import re
import urllib.parse
import urllib.request

import tk_db

LOG = logging.getLogger("tk_agreements")

AEST = dt.timezone(dt.timedelta(hours=10))
SP_HOST_PATH = "embraceyochi.sharepoint.com:/sites/YochiTeamSharepoint"
FILE_PATH = ("ALL-SHARES/1.International Confidential/1.Signed Countries/"
             "Exec Summaries (all mkts)/Yochi_Signed_Agreements_Tracker_v2.xlsx")
SHEET = "Signed Agreements"
HEADER_ROW = 5                      # 1-based; data starts at 6
PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"

# Territory -> the project its tasks belong in. Keyed on a keyword because the
# tracker's territory cell is free text (the SEA row is a 9-country paragraph).
TERRITORIES = [
    # (match keyword, slug, short label, project_id)
    ("sea ex", "sea", "SEA ex-Singapore", "aaaaaaaa-0000-0000-0000-000000000014"),
    ("texas", "texas", "Texas (USA)", "aaaaaaaa-0000-0000-0000-000000000012"),
    ("florida", "florida", "Florida (USA)", "aaaaaaaa-0000-0000-0000-000000000011"),
    ("united kingdom", "uk", "United Kingdom", "aaaaaaaa-0000-0000-0000-000000000013"),
    ("singapore", "singapore", "Singapore", "aaaaaaaa-0000-0000-0000-000000000014"),
]

# Columns we watch for day-over-day change. Anything here that moves is
# reported; the ones marked material also raise a flag task.
WATCH = {
    "date_of_signing": ("Date of signing", True),
    "structure": ("Structure", True),
    "ownership": ("Yo Chi Ownership %", True),
    "royalty": ("Royalty %", True),
    "yr1": ("Yr1", True),
    "yr2": ("Yr2", True),
    "yr3": ("Y3", True),
    "window_end": ("No Cost Support Window End Date", True),
    "window": ("No Cost Support Window", False),
    "partner": ("Partner", False),
    "initial_capital": ("Initial Capital Total", False),
    "funded": ("Current what we have funded", False),
    "future_funding": ("Future funding required", True),
    "mff": ("Initial MFF/License", False),
    "dev_fee": ("Development Fee upon opening", False),
    "controls": ("Controls - IE full approval needed.", False),
}

_BLANK = ("", "tbc", "tba", "n/a", "na", "none", "-", "?", "unknown")
_DATE_FMTS = ("%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y")


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


def _download():
    h = {"Authorization": "Bearer " + _token()}
    req = urllib.request.Request(
        "https://graph.microsoft.com/v1.0/sites/%s" % SP_HOST_PATH, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        site_id = json.loads(r.read().decode())["id"]
    url = "https://graph.microsoft.com/v1.0/sites/%s/drive/root:/%s:/content" % (
        site_id, urllib.parse.quote(FILE_PATH))
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read()


def _txt(v):
    if v is None:
        return ""
    if isinstance(v, (dt.datetime, dt.date)):
        return v.date().isoformat() if isinstance(v, dt.datetime) else v.isoformat()
    return re.sub(r"\s+", " ", str(v)).strip()


def _missing(s):
    return _txt(s).lower() in _BLANK


def _date(s):
    """A real date out of a cell that might be a date, '2026-09-14 00:00:00',
    'Ended 14th January 2026', or 'TBC'. None when there isn't one."""
    if isinstance(s, dt.datetime):
        return s.date()
    if isinstance(s, dt.date):
        return s
    t = _txt(s)
    if not t:
        return None
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t, flags=re.I)   # 14th -> 14
    t = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "", t).strip()
    m = re.search(r"\d{4}-\d{2}-\d{2}", t)
    if m:
        try:
            return dt.date.fromisoformat(m.group(0))
        except ValueError:
            pass
    m = re.search(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", t)
    cand = m.group(0) if m else t
    for fmt in _DATE_FMTS:
        try:
            return dt.datetime.strptime(cand, fmt).date()
        except ValueError:
            pass
    return None


def _num(s):
    m = re.search(r"\d+(?:\.\d+)?", _txt(s))
    return float(m.group(0)) if m else None


def _plus_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:                      # 29 Feb
        return d.replace(year=d.year + n, day=28)


def parse(data):
    """The tracker as a list of dicts, one per signed territory."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    ws = wb[SHEET] if SHEET in wb.sheetnames else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [_txt(h) for h in rows[HEADER_ROW - 1]]
    idx = {}
    for key, (label, _material) in WATCH.items():
        idx[key] = next((i for i, h in enumerate(hdr) if h.lower() == label.lower()), None)
    out = []
    for raw in rows[HEADER_ROW:]:
        terr = _txt(raw[0])
        if not terr:
            continue
        low = terr.lower()
        hit = next((t for t in TERRITORIES if t[0] in low), None)
        if not hit:
            LOG.warning("tracker row not mapped to a project: %s", terr[:60])
        rec = {"territory": terr, "slug": hit[1] if hit else re.sub(r"[^a-z]+", "-", low)[:20],
               "label": hit[2] if hit else terr[:40],
               "project_id": hit[3] if hit else None, "mapped": bool(hit)}
        for key, i in idx.items():
            rec[key] = _txt(raw[i]) if i is not None and i < len(raw) else ""
        out.append(rec)
    return out


_STOP = {"the", "and", "for", "with", "per", "from", "that", "this", "are",
         "was", "not", "yo", "chi", "yochi", "usa", "ex"}


def _tokens(title):
    return {w for w in re.findall(r"[a-z0-9]+", (title or "").lower())
            if len(w) > 2 and w not in _STOP}


def _adopt_candidate(spec, open_tasks):
    """An existing unmanaged task that already covers this obligation.

    Without this the first run would raise a second copy of work Peter already
    has on the board (his 'SEA future funding: SGD 400k' task is exactly the
    obligation the tracker implies). Tasks already owned by another sync -
    Asana, the compliance register - are never touched.
    """
    want = _tokens(spec["title"])
    if not want:
        return None
    best, best_score = None, 0.0
    for t in open_tasks:
        if t.get("external_ref") or t.get("project_id") != spec["project_id"]:
            continue
        have = _tokens(t.get("title"))
        if not have:
            continue
        score = len(want & have) / float(min(len(want), len(have)))
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 0.6 else None


def _expected(rec, today):
    """The obligations this territory's row implies, as task specs."""
    label, slug, pid = rec["label"], rec["slug"], rec["project_id"]
    if not pid:
        return []
    src = ("Source of truth: Yochi_Signed_Agreements_Tracker_v2.xlsx, "
           "'%s' tab, %s row. This task is kept in step with that file by the "
           "daily agreements check - if the tracker changes, this task is "
           "amended to match." % (SHEET, label))
    out = []

    # 1. no-cost support window ending (or already ended)
    end = _date(rec.get("window_end"))
    if end:
        overdue = end < today
        out.append({
            "kind": "support-window",
            "project_id": pid,
            "title": ("Support window ENDED %s - confirm %s support is now charged"
                      % (end.strftime("%d %b %Y"), label)) if overdue else
                     ("Support window ends %s - move %s to chargeable support"
                      % (end.strftime("%d %b %Y"), label)),
            "description": (
                "No-cost support window: %s\nWindow end per tracker: %s\n\n"
                "%s\n\nCheck: is support being charged from the day after the "
                "window ends, and does the JV/licensee know the rate?\n\n%s") % (
                    rec.get("window") or "(not stated)", end.isoformat(),
                    "This window has ALREADY ENDED - any support given since is "
                    "likely unbilled." if overdue else
                    "Raise the fee schedule before the window closes.",
                    src),
            "priority": "high" if overdue else "medium",
            # Anchored to the tracker date, never to 'today': a due date of
            # today would be rewritten every morning, so the task would churn
            # daily AND never present as overdue.
            "due_date": (end if overdue else end - dt.timedelta(days=30)).isoformat(),
        })

    # 2. store development targets, reviewed on each anniversary of signing
    signed = _date(rec.get("date_of_signing"))
    if signed:
        for n, key in ((1, "yr1"), (2, "yr2"), (3, "yr3")):
            target = _num(rec.get(key))
            if target is None:
                continue
            anniv = _plus_years(signed, n)
            if not (today - dt.timedelta(days=365) <= anniv <= today + dt.timedelta(days=180)):
                continue
            out.append({
                "kind": "target-yr%d" % n,
                "project_id": pid,
                "title": "Yr%d store target - %s: %s stores by %s" % (
                    n, label, ("%g" % target), anniv.strftime("%d %b %Y")),
                "description": (
                    "Development obligation from the signed agreement.\n\n"
                    "Signed: %s\nYr%d target: %s stores\nMeasured at: %s\n\n"
                    "Check open store count against the target, and if it is "
                    "short, what the agreement allows us to do about it.\n\n%s") % (
                        signed.isoformat(), n, ("%g" % target), anniv.isoformat(), src),
                "priority": "high" if anniv <= today else "medium",
                "due_date": anniv.isoformat(),
            })

    # 3. funding still to be called
    fund = rec.get("future_funding") or ""
    if fund and not _missing(fund) and not fund.lower().startswith("none"):
        out.append({
            "kind": "funding",
            "project_id": pid,
            "title": "Future funding required - %s: %s" % (label, fund[:80]),
            "description": ("The tracker records funding still to be provided.\n\n"
                            "Future funding required: %s\nInitial capital: %s\n"
                            "Funded to date: %s\n\nConfirm timing, entity and "
                            "approval path for the call.\n\n%s") % (
                                fund, rec.get("initial_capital") or "(blank)",
                                rec.get("funded") or "(blank)", src),
            "priority": "medium",
            "due_date": (today + dt.timedelta(days=30)).isoformat(),
        })

    # 4. gaps in the source of truth itself
    gaps = [WATCH[k][0] for k in ("window_end", "date_of_signing", "royalty",
                                  "ownership", "yr1")
            if _missing(rec.get(k))]
    if gaps:
        out.append({
            "kind": "tracker-gap",
            "project_id": pid,
            "title": "Tracker gap - %s: %s" % (label, ", ".join(gaps)[:90]),
            "description": (
                "These fields are blank or TBC in the source of truth, so no "
                "obligation can be derived from them:\n\n%s\n\nFill them in the "
                "tracker (not here) and this task closes itself off the next "
                "morning.\n\n%s") % (
                    "\n".join("- " + g for g in gaps), src),
            "priority": "medium",
            "due_date": (today + dt.timedelta(days=7)).isoformat(),
        })
    for spec in out:
        # Only a date the TRACKER anchors may be re-amended. 'funding' and
        # 'tracker-gap' are due today+N, so comparing them every morning would
        # re-amend the same task forever - they are set once, at creation.
        spec["anchored"] = (spec["kind"] == "support-window"
                            or spec["kind"].startswith("target-yr"))
        spec["external_ref"] = "agmt:%s:%s" % (slug, spec["kind"])
        spec["source"] = "watcher"
        spec["assignee_id"] = PETER
    return out


TASK_COLS = ("project_id", "title", "description", "priority", "due_date",
             "source", "assignee_id", "external_ref")


def _row(spec):
    """The spec as a tasks row - 'kind' is our own bookkeeping, not a column."""
    return {k: v for k, v in spec.items() if k in TASK_COLS}


def _snapshot_diff(records, today):
    """What changed in the tracker since the last run."""
    changes = []
    # newest-first, and keep the FIRST row per slug: a dict comprehension over
    # a desc-ordered list would keep the LAST (oldest) one, which would compare
    # today against the oldest snapshot on file and re-report the same change
    # every morning.
    prev = {}
    for r in tk_db.get("agreement_snapshots",
                       {"select": "slug,data,captured_at",
                        "order": "captured_at.desc", "limit": "500"}):
        prev.setdefault(r["slug"], r)
    for rec in records:
        old = (prev.get(rec["slug"]) or {}).get("data") or {}
        if not old:
            continue                      # first sighting is not a change
        for key, (label, material) in WATCH.items():
            a, b = _txt(old.get(key)), _txt(rec.get(key))
            if a != b:
                changes.append({"territory": rec["label"], "field": label,
                                "old": a or "(blank)", "new": b or "(blank)",
                                "material": material})
    return changes


def _save_snapshot(records):
    rows = [{"slug": r["slug"], "territory": r["label"],
             "data": {k: r.get(k) for k in WATCH},
             "hash": hashlib.sha256(json.dumps(
                 {k: r.get(k) for k in WATCH}, sort_keys=True).encode()).hexdigest()[:16]}
            for r in records]
    if rows:
        tk_db.insert("agreement_snapshots", rows)


def check(apply=True, today=None):
    """Reconcile TaskHub against the tracker. apply=False is a dry run."""
    today = today or dt.datetime.now(AEST).date()
    records = parse(_download())
    added, amended, flags, linked = [], [], [], []

    for rec in records:
        if not rec["mapped"]:
            flags.append({"kind": "unmapped", "territory": rec["territory"][:60],
                          "detail": "no TaskHub project mapped for this territory - "
                                    "tasks cannot be kept in step until it is added "
                                    "to TERRITORIES in tk_agreements.py"})

    expected = [s for rec in records for s in _expected(rec, today)]
    by_ref = {s["external_ref"]: s for s in expected}

    existing = {t["external_ref"]: t for t in tk_db.get(
        "tasks", {"external_ref": "like.agmt:*", "select":
                  "id,external_ref,title,due_date,status,project_id,priority"})}

    # unmanaged open tasks in the territory projects, for adoption matching
    pids = sorted({s["project_id"] for s in expected if s.get("project_id")})
    open_tasks = tk_db.get("tasks", {
        "project_id": "in.(%s)" % ",".join(pids), "status": "neq.done",
        "external_ref": "is.null",
        "select": "id,title,project_id,due_date,external_ref,status"}) if pids else []
    claimed = set()

    for ref, spec in by_ref.items():
        cur = existing.get(ref)
        if not cur:
            twin = _adopt_candidate(spec, [t for t in open_tasks
                                           if t["id"] not in claimed])
            if twin:
                claimed.add(twin["id"])
                if apply:
                    patch = {"external_ref": ref, "description": spec["description"]}
                    if spec["anchored"]:
                        patch["due_date"] = spec["due_date"]
                    tk_db.patch("tasks", {"id": "eq." + twin["id"]}, patch)
                linked.append({"ref": ref, "title": twin["title"],
                               "now": spec["title"], "due": spec["due_date"]})
                continue
            if apply:
                tk_db.insert("tasks", [_row(spec)], on_conflict="external_ref",
                             ignore_duplicates=True)
            added.append({"ref": ref, "title": spec["title"],
                          "due": spec["due_date"]})
            continue
        if cur["status"] == "done":
            continue
        diff = []
        if spec["anchored"] and (cur.get("due_date") or "") != spec["due_date"]:
            diff.append("due %s -> %s" % (cur.get("due_date") or "(none)", spec["due_date"]))
        if (cur.get("title") or "") != spec["title"]:
            diff.append("title now '%s'" % spec["title"][:70])
        if (cur.get("priority") or "") != spec["priority"]:
            diff.append("priority %s -> %s" % (cur.get("priority"), spec["priority"]))
        if diff:
            if apply:
                patch = {"title": spec["title"], "priority": spec["priority"],
                         "description": spec["description"]}
                if spec["anchored"]:
                    patch["due_date"] = spec["due_date"]
                tk_db.patch("tasks", {"id": "eq." + cur["id"]}, patch)
            amended.append({"ref": ref, "title": spec["title"],
                            "changes": "; ".join(diff)})

    for ref, cur in existing.items():
        if ref not in by_ref and cur["status"] != "done":
            flags.append({"kind": "orphan", "territory": ref,
                          "detail": "open task '%s' no longer matches anything in "
                                    "the tracker - close it or restore the row"
                                    % (cur.get("title") or "")[:80]})

    changes = _snapshot_diff(records, today)
    for c in changes:
        flags.append({"kind": "changed", "territory": c["territory"],
                      "detail": "%s: %s -> %s%s" % (c["field"], c["old"][:60],
                                                    c["new"][:60],
                                                    "  (material)" if c["material"] else "")})
    if apply:
        try:
            _save_snapshot(records)
        except Exception:
            LOG.exception("snapshot save failed - change detection resumes next run")

    material = [c for c in changes if c["material"]]
    if apply and material:
        body = "\n".join("- %s | %s: %s -> %s" % (c["territory"], c["field"],
                                                  c["old"][:80], c["new"][:80])
                         for c in material)
        try:
            tk_db.insert("tasks", [{
                "project_id": "aaaaaaaa-0000-0000-0000-000000000021",
                "title": "[Agreements] %d material change%s in the signed agreements tracker"
                         % (len(material), "" if len(material) == 1 else "s"),
                "description": ("The source of truth changed since yesterday:\n\n%s\n\n"
                                "Check whether anything downstream (tasks, tax "
                                "positions, board reporting) needs to follow.") % body,
                "priority": "high", "source": "watcher",
                "due_date": today.isoformat(),
                "assignee_id": PETER,
                "external_ref": "agmt:changes:" + today.isoformat(),
            }], on_conflict="external_ref", ignore_duplicates=True)
        except Exception:
            LOG.exception("could not raise tracker-change task")

    return {"date": today.isoformat(), "territories": len(records),
            "expected": len(expected), "added": added, "amended": amended,
            "linked": linked,
            "flags": flags, "changes": changes, "applied": bool(apply),
            "records": [{k: r.get(k) for k in ("label", "slug", "mapped")}
                        for r in records]}


def _html(res):
    def rows(items, cols):
        if not items:
            return "<tr><td colspan='%d' style='padding:8px;color:#888'>none</td></tr>" % len(cols)
        return "".join("<tr>" + "".join(
            "<td style='padding:6px 10px;border-top:1px solid #eee;vertical-align:top'>%s</td>"
            % (str(it.get(c, "") or "")[:220]) for c in cols) + "</tr>" for it in items)

    h = ["<div style=\"font-family:-apple-system,Segoe UI,Arial,sans-serif;"
         "max-width:860px;color:#222\">",
         "<h2 style='margin:0 0 4px'>Signed agreements check</h2>",
         "<div style='color:#666;font-size:13px;margin-bottom:14px'>%s &middot; %d "
         "territories &middot; %d obligations derived from the tracker</div>" % (
             res["date"], res["territories"], res["expected"]),
         "<div style='font-size:15px;margin-bottom:16px'>"
         "<b>%d</b> added &nbsp;|&nbsp; <b>%d</b> linked &nbsp;|&nbsp; <b>%d</b> "
         "amended &nbsp;|&nbsp; <b>%d</b> flagged</div>"
         % (len(res["added"]), len(res["linked"]), len(res["amended"]),
            len(res["flags"]))]
    for title, items, cols in (
            ("Added", res["added"], ("title", "due")),
            ("Linked to work already on the board", res["linked"], ("title", "now")),
            ("Amended", res["amended"], ("title", "changes")),
            ("Flagged", res["flags"], ("kind", "territory", "detail"))):
        h.append("<h3 style='margin:18px 0 6px;font-size:14px'>%s</h3>"
                 "<table style='border-collapse:collapse;width:100%%;font-size:13px'>%s</table>"
                 % (title, rows(items, cols)))
    h.append("<p style='color:#888;font-size:12px;margin-top:18px'>Source of truth: "
             "Yochi_Signed_Agreements_Tracker_v2.xlsx on the team SharePoint. This "
             "check only reads it - fix data in the tracker and the tasks follow the "
             "next morning.</p></div>")
    return "".join(h)


def run(apply=True, email=True):
    res = check(apply=apply)
    if email:
        try:
            import tk_email
            if tk_email.enabled():
                tk_email.send(
                    os.environ.get("AGREEMENTS_TO", "peterm@yochi.com.au"),
                    "Signed agreements check - %d added, %d amended, %d flagged" % (
                        len(res["added"]) + len(res["linked"]),
                        len(res["amended"]), len(res["flags"])),
                    _html(res))
                res["emailed"] = True
        except Exception:
            LOG.exception("agreements email failed (check itself succeeded)")
    LOG.info("agreements: %d added, %d linked, %d amended, %d flags",
             len(res["added"]), len(res["linked"]), len(res["amended"]),
             len(res["flags"]))
    return res
