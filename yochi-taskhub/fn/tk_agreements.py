# -*- coding: utf-8 -*-
"""Signed agreements check - Yochi_Signed_Agreements_Tracker_v3.xlsx.

The tracker is the SOURCE OF TRUTH for the international deals, so every
morning we read it LIVE from SharePoint and make TaskHub agree with it:

  ADDED    an obligation the tracker implies but TaskHub has no task for
  LINKED   an obligation already covered by work on the board - adopted, not
           duplicated
  AMENDED  a task whose due date or wording no longer matches the tracker
  CLOSED   the tracker says done, so the task is done
  FLAGGED  gaps in the tracker itself (TBC/blank), fields that CHANGED since
           yesterday, and open tasks whose obligation has left the tracker

ALL SHEETS are read, because each carries different obligations:
  Deal Summary          the territory register - signing dates, structure, %
  Funding & Fees        capital still to be called, fees, payment terms
  Support & Costs       the no-cost support window and when it ends
  Annual Business Plan  delivery deadlines and the 60-day partner reminder
  D&O Insurance         cover confirmation and renewal dates
  Compliance Register   75 standing obligations (15 per market) with owners,
                        frequency and status - the bulk of the work
  Global Policy         principles that apply everywhere; quoted into task
                        descriptions rather than turned into tasks of its own

The tracker is never written to - it is the source of truth, not a workspace.
Asana is not touched. Dedup/idempotency = tasks.external_ref 'agmt:<slug>:<kind>'.
Runs daily 07:45 AEST + run_agreements."""
import calendar
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
FILE_NAME = "Yochi_Signed_Agreements_Tracker_v3.xlsx"
# located by name each run, so the check survives the file being moved or
# renamed forward (v2 -> v3 broke the old hard-coded path silently)
FALLBACK_ITEM_ID = "01Q7M2EZ3NPQ477PKYYZF3WCDZQAWPR77K"
PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"
LOOK_AHEAD = 180             # obligations this far out become tasks

# territory -> (slug, label, project). The sheets name the same market
# differently ("Texas" on the register, "Texas, USA" on Deal Summary), so
# match on a keyword and normalise.
TERRITORIES = [
    ("sea", "sea", "SEA ex-Singapore", "aaaaaaaa-0000-0000-0000-000000000014"),
    ("texas", "texas", "Texas (USA)", "aaaaaaaa-0000-0000-0000-000000000012"),
    ("florida", "florida", "Florida (USA)", "aaaaaaaa-0000-0000-0000-000000000011"),
    ("united kingdom", "uk", "United Kingdom", "aaaaaaaa-0000-0000-0000-000000000013"),
    ("uk", "uk", "United Kingdom", "aaaaaaaa-0000-0000-0000-000000000013"),
    ("singapore", "singapore", "Singapore", "aaaaaaaa-0000-0000-0000-000000000014"),
    ("thailand", "thailand", "Thailand", "aaaaaaaa-0000-0000-0000-000000000015"),
    ("malaysia", "malaysia", "Malaysia", "aaaaaaaa-0000-0000-0000-000000000015"),
    ("philippines", "philippines", "Philippines", "aaaaaaaa-0000-0000-0000-000000000015"),
    ("indonesia", "indonesia", "Indonesia", "aaaaaaaa-0000-0000-0000-000000000015"),
]
PIPELINE = "aaaaaaaa-0000-0000-0000-000000000015"

_BLANK = ("", "tbc", "tba", "n/a", "na", "none", "-", "—", "?", "unknown",
          "not confirmed", "nil")
_DONE_WORDS = ("complete", "done", "received", "filed", "confirmed", "closed",
               "exercised", "waived", "paid", "n/a", "not applicable")
_DATE_FMTS = ("%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y",
              "%d-%b-%y", "%d %b %y")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}


# ------------------------------------------------------------------ SharePoint
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


def _graph(url, tok):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def download():
    """The workbook, found by NAME rather than a fixed path.

    v2 lived in a different folder; hard-coding the path meant the move to v3
    would have failed quietly. Searching by name survives a move or a rename
    forward, and the item id is only a fallback."""
    tok = _token()
    site = json.loads(_graph("https://graph.microsoft.com/v1.0/sites/%s"
                             % SP_HOST_PATH, tok))["id"]
    drive = json.loads(_graph("https://graph.microsoft.com/v1.0/sites/%s/drive"
                              % site, tok))["id"]
    item_id = None
    try:
        hits = json.loads(_graph(
            "https://graph.microsoft.com/v1.0/drives/%s/root/search(q='%s')"
            % (drive, urllib.parse.quote(FILE_NAME.replace(".xlsx", ""))), tok))
        for it in hits.get("value", []):
            if it.get("name", "").lower() == FILE_NAME.lower():
                item_id = it["id"]
                break
    except Exception:
        LOG.exception("tracker search failed - falling back to the stored id")
    item_id = item_id or FALLBACK_ITEM_ID
    return _graph("https://graph.microsoft.com/v1.0/drives/%s/items/%s/content"
                  % (drive, item_id), tok)


# ------------------------------------------------------------------ helpers
def _txt(v):
    if v is None:
        return ""
    if isinstance(v, dt.datetime):
        return v.date().isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    return re.sub(r"\s+", " ", str(v)).strip()


def _missing(s):
    return _txt(s).lower() in _BLANK


def _is_done(s):
    t = _txt(s).lower()
    return bool(t) and any(t.startswith(w) for w in _DONE_WORDS)


def _date(v):
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    t = _txt(v)
    if not t or _missing(t):
        return None
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t, flags=re.I)
    m = re.search(r"\d{4}-\d{2}-\d{2}", t)
    if m:
        try:
            return dt.date.fromisoformat(m.group(0))
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})", t)
    if m and m.group(2)[:3].lower() in MONTHS:
        try:
            return dt.date(int(m.group(3)), MONTHS[m.group(2)[:3].lower()],
                           int(m.group(1)))
        except ValueError:
            pass
    for fmt in _DATE_FMTS:
        try:
            return dt.datetime.strptime(t, fmt).date()
        except ValueError:
            pass
    return None


def _territory(name):
    low = _txt(name).lower()
    if not low:
        return None
    for key, slug, label, proj in TERRITORIES:
        if key in low:
            return {"slug": slug, "label": label, "project_id": proj,
                    "raw": _txt(name)}
    return None


def _eoq(today):
    q_end_month = ((today.month - 1) // 3 + 1) * 3
    return dt.date(today.year, q_end_month,
                   calendar.monthrange(today.year, q_end_month)[1])


def _eom(today):
    return dt.date(today.year, today.month,
                   calendar.monthrange(today.year, today.month)[1])


def _header_row(rows, must_have="territory", limit=10):
    """Each sheet carries a title and a note above its header."""
    for i, r in enumerate(rows[:limit]):
        vals = [_txt(c).lower() for c in (r or [])]
        if any(v.startswith(must_have) for v in vals):
            return i
    return 3


def _sheet(wb, name):
    if name not in wb.sheetnames:
        return [], {}, 0
    rows = list(wb[name].iter_rows(values_only=True))
    hi = _header_row(rows)
    hdr = {}
    for j, c in enumerate(rows[hi] if hi < len(rows) else []):
        key = _txt(c).lower()
        if key:
            hdr[key] = j
    return rows, hdr, hi


def _col(hdr, *starts):
    for s in starts:
        for k, j in hdr.items():
            if k.startswith(s):
                return j
    return None


def _get(row, idx):
    return _txt(row[idx]) if idx is not None and idx < len(row) else ""


# ------------------------------------------------------------------ parsing

WATCH_SHEETS = ("Deal Summary", "Funding & Fees", "Support & Costs",
                "Annual Business Plan", "D&O Insurance", "Compliance Register")


def parse(data):
    """Every sheet, as one structure keyed by territory plus the register rows."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    out = {"markets": {}, "register": [], "policy": {}, "sheets": wb.sheetnames}

    def market(name):
        t = _territory(name)
        if not t:
            return None
        return out["markets"].setdefault(t["slug"], dict(t))

    def merge(m, updates):
        """Fill in what a sheet knows WITHOUT erasing what another sheet said.

        Several rows can map to one market - "SEA ex Singapore UMBRELLA DEAL"
        and "other SEA territories TBC" are both SEA - and a plain update let
        the emptier row win, silently blanking the signing date."""
        if not m:
            return
        for k, v in updates.items():
            if v or not m.get(k):
                m[k] = v

    # --- Deal Summary: the territory register
    rows, hdr, hi = _sheet(wb, "Deal Summary")
    ci = {k: _col(hdr, k) for k in ("territory", "structure", "signed",
                                    "venue opening", "yo-chi %", "royalty",
                                    "jv entity", "partner entity")}
    for r in rows[hi + 1:]:
        m = market(_get(r, ci["territory"]))
        if not m:
            continue
        merge(m, {"structure": _get(r, ci["structure"]),
                  "signed": _get(r, ci["signed"]),
                  "venue_opening": _get(r, ci["venue opening"]),
                  "ownership": _get(r, ci["yo-chi %"]),
                  "royalty": _get(r, ci["royalty"]),
                  "jv_entity": _get(r, ci["jv entity"]),
                  "partner": _get(r, ci["partner entity"])})

    # --- Funding & Fees
    rows, hdr, hi = _sheet(wb, "Funding & Fees")
    ci = {k: _col(hdr, k) for k in ("territory", "initial capital", "funded to date",
                                    "future funding", "initial mff", "payment terms",
                                    "payments received", "design fee")}
    for r in rows[hi + 1:]:
        m = market(_get(r, ci["territory"]))
        if not m:
            continue
        merge(m, {"initial_capital": _get(r, ci["initial capital"]),
                  "funded": _get(r, ci["funded to date"]),
                  "future_funding": _get(r, ci["future funding"]),
                  "mff": _get(r, ci["initial mff"]),
                  "payment_terms": _get(r, ci["payment terms"]),
                  "payments_received": _get(r, ci["payments received"]),
                  "design_fee": _get(r, ci["design fee"])})

    # --- Support & Costs
    rows, hdr, hi = _sheet(wb, "Support & Costs")
    ci = {k: _col(hdr, k) for k in ("territory", "no-cost support", "window ends",
                                    "travel & accom", "travel approvals",
                                    "control rights")}
    for r in rows[hi + 1:]:
        m = market(_get(r, ci["territory"]))
        if not m:
            continue
        merge(m, {"support_window": _get(r, ci["no-cost support"]),
                  "window_ends": _get(r, ci["window ends"]),
                  "travel": _get(r, ci["travel & accom"]),
                  "travel_approvals": _get(r, ci["travel approvals"]),
                  "control_rights": _get(r, ci["control rights"])})

    # --- Annual Business Plan
    rows, hdr, hi = _sheet(wb, "Annual Business Plan")
    ci = {k: _col(hdr, k) for k in ("territory", "defined term", "clause",
                                    "delivery deadline", "delivered to",
                                    "when board", "have we previously",
                                    "reminder to partner")}
    for r in rows[hi + 1:]:
        m = market(_get(r, ci["territory"]))
        if not m:
            continue
        merge(m, {"abp_year": _get(r, ci["defined term"]),
                  "abp_clause": _get(r, ci["clause"]),
                  "abp_deadline": _get(r, ci["delivery deadline"]),
                  "abp_to": _get(r, ci["delivered to"]),
                  "abp_board": _get(r, ci["when board"]),
                  "abp_received": _get(r, ci["have we previously"]),
                  "abp_reminder": _get(r, ci["reminder to partner"])})

    # --- D&O Insurance
    rows, hdr, hi = _sheet(wb, "D&O Insurance")
    ci = {k: _col(hdr, k) for k in ("territory", "cover confirmed", "yo-chi holdco",
                                    "policy", "adequacy", "key exclusions",
                                    "limit", "renewal")}
    for r in rows[hi + 1:]:
        m = market(_get(r, ci["territory"]))
        if not m:
            continue
        merge(m, {"do_cover": _get(r, ci["cover confirmed"]),
                  "do_holdco": _get(r, ci["yo-chi holdco"]),
                  "do_policy": _get(r, ci["policy"]),
                  "do_adequacy": _get(r, ci["adequacy"]),
                  "do_exclusions": _get(r, ci["key exclusions"]),
                  "do_limit": _get(r, ci["limit"]),
                  "do_renewal": _get(r, ci["renewal"])})

    # --- Compliance Register: the standing obligations, 15 per market
    rows, hdr, hi = _sheet(wb, "Compliance Register")
    ci = {k: _col(hdr, k) for k in ("territory", "obligation", "frequency",
                                    "what 'done'", "owner", "last completed",
                                    "status", "notes")}
    for r in rows[hi + 1:]:
        t = _territory(_get(r, ci["territory"]))
        obligation = _get(r, ci["obligation"])
        if not t or not obligation:
            continue
        market(_get(r, ci["territory"]))
        out["register"].append({
            "slug": t["slug"], "label": t["label"], "project_id": t["project_id"],
            "obligation": obligation,
            "frequency": _get(r, ci["frequency"]),
            "done_looks_like": _get(r, ci["what 'done'"]),
            "owner": _get(r, ci["owner"]),
            "last_completed": _get(r, ci["last completed"]),
            "status": _get(r, ci["status"]),
            "notes": _get(r, ci["notes"])})

    # --- Global Policy: principles, quoted into descriptions
    if "Global Policy" in wb.sheetnames:
        for r in wb["Global Policy"].iter_rows(values_only=True):
            k, v = _txt(r[0] if r else ""), _txt(r[1] if r and len(r) > 1 else "")
            if k and v and not k.lower().startswith("global policy"):
                out["policy"][k[:120]] = v[:600]
    return out


# ------------------------------------------------------------------ deriving

TASK_COLS = ("project_id", "title", "description", "priority", "due_date",
             "source", "assignee_id", "external_ref")
SRC = ("Source of truth: %s on the team SharePoint, '%%s' sheet. Kept in step "
       "by the daily agreements check - fix the tracker and this task follows "
       "the next morning." % FILE_NAME)


def _row(spec):
    return {k: v for k, v in spec.items() if k in TASK_COLS}


def _slugify(text, n=44):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:n]


def _next_annual(anchor, today):
    """The next occurrence of a day/month that recurs every year."""
    if not anchor:
        return None
    try:
        nxt = anchor.replace(year=today.year)
    except ValueError:
        nxt = anchor.replace(year=today.year, day=28)
    if nxt < today:
        try:
            nxt = nxt.replace(year=today.year + 1)
        except ValueError:
            nxt = nxt.replace(year=today.year + 1, day=28)
    return nxt


def _register_due(freq, market, today):
    """When a standing obligation next falls due, and why.

    Frequencies are words, not dates: Monthly and Quarterly land on the period
    end, Annual on the next anniversary of signing (the agreement's own clock),
    and 'Per SHA'/'As arising' have no schedule at all - those are reviewed at
    quarter end rather than pretending to a deadline."""
    f = (freq or "").strip().lower()
    if f.startswith("month"):
        return _eom(today), "month end (monthly obligation)"
    if f.startswith("quarter"):
        return _eoq(today), "quarter end (quarterly obligation)"
    if f.startswith("annual"):
        signed = _date(market.get("signed"))
        nxt = _next_annual(signed, today)
        if nxt:
            return nxt, "next anniversary of signing (%s)" % signed
        return dt.date(today.year, 6, 30), "financial year end (no signing date on file)"
    return _eoq(today), "quarter end - '%s' has no schedule of its own" % (freq or "unspecified")


def derive(parsed, today):
    """Every obligation the tracker implies, as task specs."""
    out = []
    policy = parsed.get("policy") or {}
    # A market is LIVE once it appears on the Compliance Register. Thailand,
    # Malaysia, Indonesia and the Philippines sit on Deal Summary as sub-
    # franchise territories with "TBC" against everything: that is a deal not
    # yet done, not a gap in the record, and flagging it every morning would
    # be noise that teaches people to ignore the flags.
    live = {r["slug"] for r in parsed.get("register", [])}
    # A market is LIVE once it appears on the Compliance Register. Thailand,
    # Malaysia, Indonesia and the Philippines sit on Deal Summary as sub-
    # franchise territories with "TBC" against everything: that is a deal not
    # yet done, not a gap in the record, and flagging it every morning would
    # be noise that teaches people to ignore the flags.
    live = {r["slug"] for r in parsed.get("register", [])}

    def add(market, kind, title, body, due, priority="medium", anchored=True,
            sheet="Deal Summary"):
        out.append({
            "kind": kind, "slug": market["slug"],
            "project_id": market.get("project_id") or PIPELINE,
            "title": title[:200],
            "description": body + "\n\n" + (SRC % sheet),
            "priority": priority,
            "due_date": due.isoformat() if hasattr(due, "isoformat") else due,
            "anchored": anchored, "source": "watcher",
            "assignee_id": PETER,
            "external_ref": "agmt:%s:%s" % (market["slug"], kind)})

    for slug, m in sorted(parsed.get("markets", {}).items()):
        label = m.get("label") or slug

        # --- the no-cost support window
        end = _date(m.get("window_ends"))
        if end:
            overdue = end < today
            add(m, "support-window",
                ("Support window ENDED %s - confirm %s support is now charged"
                 % (end.strftime("%d %b %Y"), label)) if overdue else
                ("Support window ends %s - move %s to chargeable support"
                 % (end.strftime("%d %b %Y"), label)),
                "No-cost support window: %s\nWindow ends: %s\n\n%s\n\n%s"
                % (m.get("support_window") or "(not stated)", end.isoformat(),
                   "This window has ALREADY ENDED - any support given since is "
                   "likely unbilled." if overdue else
                   "Raise the fee schedule before the window closes.",
                   policy.get("No-cost support window: principles", "")[:400]),
                end if overdue else end - dt.timedelta(days=30),
                "high" if overdue else "medium", sheet="Support & Costs")
        elif (slug in live and _missing(m.get("window_ends"))
              and m.get("support_window")):
            add(m, "window-tbc",
                "Tracker gap - %s: the support window end date is %s" % (
                    label, m.get("window_ends") or "blank"),
                "The window is defined as '%s' but its end date is not set, so "
                "nobody can tell when support becomes chargeable.\n\nFill it in "
                "the tracker (not here) and this closes itself."
                % (m.get("support_window") or "(blank)"),
                today + dt.timedelta(days=7), "medium", anchored=False,
                sheet="Support & Costs")

        # --- funding still to be called
        fund = m.get("future_funding") or ""
        if fund and not _missing(fund) and not fund.lower().startswith("none"):
            add(m, "funding",
                "Future funding required - %s: %s" % (label, fund[:70]),
                "Future funding required: %s\nInitial capital: %s\nFunded to "
                "date: %s\nPayments received: %s\n\nConfirm timing, entity and "
                "approval path for the call."
                % (fund, m.get("initial_capital") or "(blank)",
                   m.get("funded") or "(blank)",
                   m.get("payments_received") or "(blank)"),
                today + dt.timedelta(days=30), "medium", anchored=False,
                sheet="Funding & Fees")

        # --- annual business plan: the deadline, and the 60-day reminder
        abp = _date(m.get("abp_deadline")) or _next_annual(
            _date(m.get("abp_deadline")), today)
        if abp:
            add(m, "abp",
                "Annual business plan due %s - %s" % (abp.strftime("%d %b %Y"), label),
                "Deadline: %s\nClause: %s\nDelivered to: %s\nBoard must consider "
                "by: %s\nPreviously received: %s"
                % (m.get("abp_deadline") or "(blank)", m.get("abp_clause") or "(blank)",
                   m.get("abp_to") or "(blank)", m.get("abp_board") or "(blank)",
                   m.get("abp_received") or "(blank)"),
                abp, "medium", sheet="Annual Business Plan")
            remind = abp - dt.timedelta(days=60)
            if remind >= today - dt.timedelta(days=30):
                add(m, "abp-reminder",
                    "Remind %s partner: business plan due %s (60 days)"
                    % (label, abp.strftime("%d %b")),
                    "The tracker sets a 60-day reminder to the partner ahead of "
                    "the business plan deadline of %s.\n\nReminder note in the "
                    "tracker: %s" % (abp.isoformat(), m.get("abp_reminder") or "(blank)"),
                    remind, "medium", sheet="Annual Business Plan")
        elif (slug in live and m.get("abp_deadline")
              and _missing(m.get("abp_deadline"))):
            add(m, "abp-tbc",
                "Tracker gap - %s: no annual business plan deadline" % label,
                "The delivery deadline reads '%s'. Until it is a date, the "
                "obligation cannot be diarised and the 60-day partner reminder "
                "cannot fire." % (m.get("abp_deadline") or "(blank)"),
                today + dt.timedelta(days=14), "medium", anchored=False,
                sheet="Annual Business Plan")

        # --- D&O insurance
        cover = (m.get("do_cover") or "") if slug in live else ""
        renewal = _date(m.get("do_renewal"))
        if cover and not _is_done(cover):
            add(m, "do-cover",
                "D&O cover not confirmed - %s" % label,
                "Cover confirmed: %s\nYo-Chi holdco covered: %s\nPolicy: %s\n"
                "Limit: %s\nAdequacy assessed: %s\nKnown gaps: %s\n\nDirectors "
                "are exposed until this is confirmed in writing."
                % (cover or "(blank)", m.get("do_holdco") or "(blank)",
                   m.get("do_policy") or "(blank)", m.get("do_limit") or "(blank)",
                   m.get("do_adequacy") or "(blank)", m.get("do_exclusions") or "(blank)"),
                today + dt.timedelta(days=21), "high", anchored=False,
                sheet="D&O Insurance")
        if renewal:
            add(m, "do-renewal",
                "D&O renewal %s - %s" % (renewal.strftime("%d %b %Y"), label),
                "Policy: %s\nLimit: %s\nRenewal/expiry: %s\n\nConfirm renewal "
                "before expiry, and reassess adequacy at the same time."
                % (m.get("do_policy") or "(blank)", m.get("do_limit") or "(blank)",
                   renewal.isoformat()),
                renewal - dt.timedelta(days=30), "medium", sheet="D&O Insurance")

        # --- a signed deal with no signing date is a register gap
        if slug in live and m.get("structure") and _missing(m.get("signed")):
            add(m, "signed-tbc",
                "Tracker gap - %s: no signing date" % label,
                "Structure is '%s' but the signing date reads '%s'. Anniversary "
                "obligations - business plan, store targets, annual reviews - "
                "all count from that date, so none of them can be scheduled."
                % (m.get("structure"), m.get("signed") or "(blank)"),
                today + dt.timedelta(days=14), "medium", anchored=False)

    # --- the standing register: 15 obligations per market
    for reg in parsed.get("register", []):
        m = parsed["markets"].get(reg["slug"]) or reg
        if _is_done(reg.get("status")):
            continue
        due, basis = _register_due(reg.get("frequency"), m, today)
        if due > today + dt.timedelta(days=LOOK_AHEAD):
            continue
        behind = "behind" in (reg.get("status") or "").lower()
        kind = "reg-" + _slugify(reg["obligation"])
        out.append({
            "kind": kind, "slug": reg["slug"],
            "project_id": reg.get("project_id") or PIPELINE,
            "title": "%s - %s (%s)" % (reg["obligation"][:90], reg["label"],
                                       reg.get("frequency") or "recurring"),
            "description": (
                "Obligation: %s\nFrequency: %s\nWhat 'done' looks like: %s\n"
                "Owner: %s\nLast completed: %s\nStatus in the tracker: %s\n"
                "Notes / next action: %s\n\nDue date derived: %s\n\n%s"
                % (reg["obligation"], reg.get("frequency") or "(blank)",
                   reg.get("done_looks_like") or "(blank)",
                   reg.get("owner") or "(unassigned)",
                   reg.get("last_completed") or "never recorded",
                   reg.get("status") or "(blank)",
                   reg.get("notes") or "(none)", basis,
                   SRC % "Compliance Register")),
            "priority": "high" if behind else "medium",
            "due_date": due.isoformat(),
            "anchored": True, "source": "watcher", "assignee_id": PETER,
            "external_ref": "agmt:%s:%s" % (reg["slug"], kind)})
    return out


# --------------------------------------------------- reconcile against tasks

_STOP = {"the", "and", "for", "with", "per", "from", "that", "this", "are",
         "was", "not", "yo", "chi", "yochi", "usa", "ex", "annual", "monthly",
         "quarterly"}


def _tokens(title):
    return {w for w in re.findall(r"[a-z0-9]+", (title or "").lower())
            if len(w) > 2 and w not in _STOP}


def _adopt_candidate(spec, open_tasks):
    """An existing unmanaged task that already covers this obligation.

    Without this the first v3 run would raise a second copy of work already on
    the board. Tasks owned by another sync - Asana, the compliance calendar,
    the previous tracker - are never taken over."""
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
    return best if best_score >= 0.62 else None


def _variances(parsed, today):
    """Where the two registers disagree.

    The FY27 compliance calendar and this tracker are both meant to describe
    what each market owes. tk_compliance already loads the calendar into
    compliance_items, so the comparison is a read, not a second download.
    We report ONE direction - obligations the agreements tracker carries that
    the calendar does not mention for that market - because the calendar also
    covers ASIC, payroll and tax matters that have no business being in a
    JV tracker, and flagging those would be noise."""
    out = []
    try:
        cal = tk_db.get("compliance_items",
                        {"active": "eq.true", "stream": "eq.calendar",
                         "select": "ref,obligation,entities,category", "limit": "2000"})
    except Exception:
        LOG.exception("compliance calendar unavailable - variance check skipped")
        return out
    if not cal:
        return out
    by_market = {}
    for c in cal:
        blob = ("%s %s" % (c.get("entities") or "", c.get("category") or "")).lower()
        for _key, slug, label, _p in TERRITORIES:
            if slug in blob or label.split(" (")[0].lower() in blob:
                by_market.setdefault(slug, []).append(c)
    seen = set()
    for reg in parsed.get("register", []):
        key = (reg["slug"], reg["obligation"].lower())
        if key in seen:
            continue
        seen.add(key)
        want = _tokens(reg["obligation"])
        hit = None
        for c in by_market.get(reg["slug"], []):
            have = _tokens(c.get("obligation"))
            if have and want and len(want & have) / float(min(len(want), len(have))) >= 0.5:
                hit = c
                break
        if not hit:
            out.append({"kind": "variance", "market": reg["label"],
                        "obligation": reg["obligation"],
                        "detail": "'%s' is in the agreements tracker for %s but "
                                  "nothing matching it is in the FY27 compliance "
                                  "calendar" % (reg["obligation"], reg["label"])})
    return out


def _snapshot_diff(parsed):
    """What changed in the tracker since the last run."""
    changes = []
    prev = {}
    for r in tk_db.get("agreement_snapshots",
                       {"select": "slug,data,captured_at",
                        "order": "captured_at.desc", "limit": "500"}):
        prev.setdefault(r["slug"], r)
    for slug, m in (parsed.get("markets") or {}).items():
        old = (prev.get(slug) or {}).get("data") or {}
        if not old:
            continue
        # A new tracker version renames every field, so comparing v2's snapshot
        # against v3 reports the whole workbook as "changed" on day one. That is
        # a schema change, not news: skip it and let tomorrow compare like with
        # like.
        fields = {k for k in m if k not in ("slug", "label", "project_id", "raw")}
        if len(fields & set(old)) < max(2, len(fields) // 3):
            LOG.info("snapshot for %s predates this tracker version - skipping "
                     "change detection for one run", slug)
            continue
        for k, v in m.items():
            if k in ("slug", "label", "project_id", "raw"):
                continue
            if k not in old:
                # a field the previous snapshot never captured is new coverage,
                # not a change - reporting it says the tracker was edited when
                # only our reading of it grew
                continue
            a, b = _txt(old.get(k)), _txt(v)
            if a != b:
                changes.append({"territory": m.get("label") or slug, "field": k,
                                "old": a or "(blank)", "new": b or "(blank)"})
    return changes


def _save_snapshot(parsed):
    rows = []
    for slug, m in (parsed.get("markets") or {}).items():
        data = {k: v for k, v in m.items()
                if k not in ("slug", "label", "project_id", "raw")}
        rows.append({"slug": slug, "territory": m.get("label") or slug,
                     "data": data,
                     "hash": hashlib.sha256(json.dumps(data, sort_keys=True)
                                            .encode()).hexdigest()[:16]})
    if rows:
        tk_db.insert("agreement_snapshots", rows)


def check(apply=True, today=None):
    """Reconcile TaskHub against the tracker. apply=False is a dry run."""
    today = today or dt.datetime.now(AEST).date()
    parsed = parse(download())
    specs = derive(parsed, today)
    by_ref = {s["external_ref"]: s for s in specs}

    existing = {}
    refs = list(by_ref)
    for i in range(0, len(refs), 80):
        chunk = refs[i:i + 80]
        for t in tk_db.get("tasks", {
                "external_ref": "in.(%s)" % ",".join('"%s"' % r for r in chunk),
                "select": "id,external_ref,title,due_date,status,priority,project_id"}):
            existing[t["external_ref"]] = t

    pids = sorted({s["project_id"] for s in specs if s.get("project_id")})
    open_tasks = tk_db.get("tasks", {
        "project_id": "in.(%s)" % ",".join(pids), "status": "neq.done",
        "external_ref": "is.null",
        "select": "id,title,project_id,external_ref,status"}) if pids else []

    added, linked, amended, flags = [], [], [], []
    claimed = set()
    new_rows = []
    for ref, spec in by_ref.items():
        cur = existing.get(ref)
        if not cur:
            twin = _adopt_candidate(spec, [t for t in open_tasks
                                           if t["id"] not in claimed])
            if twin:
                claimed.add(twin["id"])
                if apply:
                    patch = {"external_ref": ref, "description": spec["description"]}
                    if spec.get("anchored"):
                        patch["due_date"] = spec["due_date"]
                    tk_db.patch("tasks", {"id": "eq." + twin["id"]}, patch)
                linked.append({"ref": ref, "title": twin["title"], "now": spec["title"]})
                continue
            new_rows.append(_row(spec))
            added.append({"ref": ref, "title": spec["title"], "due": spec["due_date"]})
            continue
        if cur["status"] == "done":
            continue
        diff = []
        if spec.get("anchored") and (cur.get("due_date") or "") != spec["due_date"]:
            diff.append("due %s -> %s" % (cur.get("due_date") or "(none)", spec["due_date"]))
        if (cur.get("title") or "") != spec["title"]:
            diff.append("title updated")
        if (cur.get("priority") or "") != spec["priority"]:
            diff.append("priority %s -> %s" % (cur.get("priority"), spec["priority"]))
        if diff:
            if apply:
                patch = {"title": spec["title"], "priority": spec["priority"],
                         "description": spec["description"]}
                if spec.get("anchored"):
                    patch["due_date"] = spec["due_date"]
                tk_db.patch("tasks", {"id": "eq." + cur["id"]}, patch)
            amended.append({"ref": ref, "title": spec["title"], "changes": "; ".join(diff)})

    if apply and new_rows:
        for i in range(0, len(new_rows), 100):
            tk_db.insert("tasks", new_rows[i:i + 100], on_conflict="external_ref",
                         ignore_duplicates=True)

    # the tracker says done -> close the task
    closed = []
    done_refs = [("agmt:%s:reg-%s" % (r["slug"], _slugify(r["obligation"])))
                 for r in parsed.get("register", []) if _is_done(r.get("status"))]
    for i in range(0, len(done_refs), 80):
        chunk = done_refs[i:i + 80]
        if not chunk:
            continue
        for t in tk_db.get("tasks", {
                "external_ref": "in.(%s)" % ",".join('"%s"' % r for r in chunk),
                "status": "neq.done", "select": "id,external_ref,title"}):
            if apply:
                tk_db.patch("tasks", {"id": "eq." + t["id"]},
                            {"status": "done",
                             "completed_at": dt.datetime.utcnow().isoformat() + "Z"})
            closed.append({"ref": t["external_ref"], "title": t["title"][:80]})

    # open agmt: tasks that no longer match anything the tracker says
    for t in tk_db.get("tasks", {"external_ref": "like.agmt:*", "status": "neq.done",
                                 "select": "external_ref,title", "limit": "500"}):
        # the summary tasks this check raises are not obligations, so they are
        # not orphans when they do not appear in the derived set
        if (t["external_ref"] not in by_ref
                and not t["external_ref"].startswith(("agmt:changes", "agmt:variance"))):
            flags.append({"kind": "orphan-task", "detail":
                          "%s: '%s' no longer matches the tracker - close it or "
                          "restore the row" % (t["external_ref"],
                                               (t.get("title") or "")[:70])})

    changes = _snapshot_diff(parsed)
    for c in changes:
        flags.append({"kind": "changed", "detail": "%s | %s: %s -> %s"
                      % (c["territory"], c["field"], c["old"][:60], c["new"][:60])})

    variances = _variances(parsed, today)
    for v in variances[:40]:
        flags.append({"kind": "variance", "detail": v["detail"]})

    if apply:
        try:
            _save_snapshot(parsed)
        except Exception:
            LOG.exception("snapshot save failed - change detection resumes next run")
        if variances:
            body = "\n".join("- %s | %s" % (v["market"], v["obligation"])
                             for v in variances[:40])
            try:
                tk_db.insert("tasks", [{
                    "project_id": "aaaaaaaa-0000-0000-0000-000000000021",
                    "title": "[Registers] %d obligation%s in the agreements tracker "
                             "are not in the FY27 compliance calendar"
                             % (len(variances), "" if len(variances) == 1 else "s"),
                    "description": (
                        "The two registers disagree. These are in "
                        "Yochi_Signed_Agreements_Tracker_v3.xlsx but nothing "
                        "matching them is in Yochi_Group_Compliance_Calendar_FY27"
                        ".xlsx for that market:\n\n%s\n\nEither add them to the "
                        "calendar so the compliance cycle picks them up, or "
                        "record why they sit only in the tracker. Note the "
                        "calendar cannot be written to from here - SharePoint "
                        "write consent is not granted - so this is raised for a "
                        "person to action." % body),
                    "priority": "medium", "source": "watcher",
                    "due_date": (today + dt.timedelta(days=7)).isoformat(),
                    "assignee_id": PETER,
                    "external_ref": "agmt:variance:" + today.isoformat(),
                }], on_conflict="external_ref", ignore_duplicates=True)
            except Exception:
                LOG.exception("could not raise the register variance task")

    return {"date": today.isoformat(), "sheets": parsed.get("sheets"),
            "markets": len(parsed.get("markets") or {}),
            "register_rows": len(parsed.get("register") or []),
            "obligations": len(specs), "added": added, "linked": linked,
            "amended": amended, "closed": closed, "flags": flags,
            "variances": variances, "changes": changes, "applied": bool(apply)}


def _html(res):
    def table(items, cols, empty="none"):
        if not items:
            return "<tr><td colspan='%d' style='padding:8px;color:#888'>%s</td></tr>" % (
                len(cols), empty)
        return "".join("<tr>" + "".join(
            "<td style='padding:6px 10px;border-top:1px solid #eee;vertical-align:top'>%s</td>"
            % str(it.get(c, "") or "")[:200] for c in cols) + "</tr>"
            for it in items[:40])

    h = ["<div style=\"font-family:-apple-system,Segoe UI,Arial,sans-serif;"
         "max-width:900px;color:#222\">",
         "<h2 style='margin:0 0 4px'>Signed agreements check</h2>",
         "<div style='color:#666;font-size:13px;margin-bottom:14px'>%s &middot; %d "
         "markets &middot; %d register obligations &middot; %d derived across %d "
         "sheets</div>" % (res["date"], res["markets"], res["register_rows"],
                           res["obligations"], len(res.get("sheets") or [])),
         "<div style='font-size:15px;margin-bottom:16px'><b>%d</b> added &nbsp;|&nbsp; "
         "<b>%d</b> linked &nbsp;|&nbsp; <b>%d</b> amended &nbsp;|&nbsp; <b>%d</b> "
         "closed &nbsp;|&nbsp; <b>%d</b> flagged</div>"
         % (len(res["added"]), len(res["linked"]), len(res["amended"]),
            len(res["closed"]), len(res["flags"]))]
    v = res.get("variances") or []
    if v:
        h.append(
            "<div style='border-left:4px solid #d97706;background:#fffbeb;"
            "padding:12px 14px;margin:6px 0 18px'>"
            "<div style='font-weight:700;margin-bottom:4px'>%d obligation%s in the "
            "agreements tracker are not in the FY27 compliance calendar</div>"
            "<div style='font-size:12px;color:#92400e'>Neither register can be "
            "written to from here - this app holds Graph read only - so these "
            "need a person to add them to the calendar, or to record why they "
            "belong only in the tracker.</div></div>"
            % (len(v), "" if len(v) == 1 else "s"))
    for title, items, cols in (
            ("Where the two registers differ", v, ("market", "obligation")),
            ("Added", res["added"], ("title", "due")),
            ("Linked to work already on the board", res["linked"], ("title", "now")),
            ("Amended", res["amended"], ("title", "changes")),
            ("Closed - the tracker says done", res["closed"], ("ref", "title")),
            ("Flags", res["flags"], ("kind", "detail"))):
        h.append("<h3 style='margin:18px 0 6px;font-size:14px'>%s</h3>"
                 "<table style='border-collapse:collapse;width:100%%;font-size:13px'>"
                 "%s</table>" % (title, table(items, cols)))
    h.append("<p style='color:#888;font-size:12px;margin-top:18px'>Source of truth: "
             "%s on the team SharePoint - read only, never written to.</p></div>"
             % FILE_NAME)
    return "".join(h)


def run(apply=True, email=True):
    """The daily pass: amend the tasks silently, and write only when there is
    something a person needs to see.

    The email exists mainly to report where the two registers DIFFER - the
    signed agreements tracker against the FY27 compliance calendar - because
    neither file can be corrected from here (the app holds Graph read only), so
    a difference is work for a person rather than something we can fix.
    Nothing to report means no email: a daily message that usually says
    "no change" is a daily message people stop opening."""
    res = check(apply=apply)
    variances = res.get("variances") or []
    worth_sending = bool(variances or res["added"] or res["linked"]
                         or res["amended"] or res["closed"]
                         or [f for f in res["flags"] if f["kind"] != "variance"])
    if email and worth_sending:
        try:
            import tk_email
            if tk_email.enabled():
                if variances:
                    subject = ("Signed agreements: %d obligation%s differ from the "
                               "FY27 compliance calendar" %
                               (len(variances), "" if len(variances) == 1 else "s"))
                else:
                    subject = ("Signed agreements - %d added, %d amended"
                               % (len(res["added"]) + len(res["linked"]),
                                  len(res["amended"])))
                tk_email.send(
                    os.environ.get("AGREEMENTS_TO", "peterm@yochi.com.au"),
                    subject, _html(res))
                res["emailed"] = True
        except Exception:
            LOG.exception("agreements email failed (the check itself succeeded)")
    else:
        res["emailed"] = False
    LOG.info("agreements v3: %d obligations, %d added, %d linked, %d amended, "
             "%d closed, %d variances", res["obligations"], len(res["added"]),
             len(res["linked"]), len(res["amended"]), len(res["closed"]),
             len(res.get("variances") or []))
    return res


