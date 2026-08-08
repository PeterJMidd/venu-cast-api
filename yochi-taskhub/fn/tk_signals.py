# -*- coding: utf-8 -*-
"""Cloud signals refresh (replaces the desktop yochi-cfo-signals task):
web-search-grounded economy/industry/statutory context for the cockpit,
written into taskapp.signals. Never fabricates - if research is thin,
fewer rows are written."""
import datetime as dt
import logging
import os
import urllib.parse
import urllib.request

import tk_ai
import tk_db

LOG = logging.getLogger("tk_signals")

ROWS_SCHEMA = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array", "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["economy", "industry", "notice"]},
                    "headline": {"type": "string", "description": "<=80 chars, factual, with numbers"},
                    "detail": {"type": "string", "description": "1-2 sentences: specifics + why it matters to Yo-Chi"},
                    "source": {"type": "string", "description": "publication name"},
                },
                "required": ["kind", "headline", "detail", "source"],
            },
        }
    },
    "required": ["signals"],
}


def run():
    research = tk_ai.searched_text(
        "You are the economic research assistant for the CFO of Yo-Chi (74 frozen-"
        "yoghurt venues across Australia, ~$4.5m/week sales). Use web search and "
        "report ONLY facts you found, with numbers and dates. Today is %s."
        % dt.date.today().isoformat(),
        "Research three things for this week's CFO briefing:\n"
        "1. RBA cash rate: current setting, any decision or meeting this week, and "
        "the latest CPI print.\n"
        "2. Australian retail/hospitality/QSR industry: 1-2 genuinely notable items "
        "this week (consumer spending data, award wage decisions, major food input "
        "cost moves). Skip generic filler.\n"
        "3. Australian statutory/tax dates in the next 3 weeks relevant to a food "
        "retail group (BAS/IAS, super guarantee, payroll tax, ASIC).\n"
        "Summarise findings with sources.")
    out = tk_ai.structured(
        "Convert the research into cockpit signal rows. Only include items with "
        "concrete facts; drop anything vague. headline <=80 chars with the key "
        "number; detail explains why it matters to Yo-Chi.",
        research, "write_signals", ROWS_SCHEMA, max_tokens=1500)
    rows = out.get("signals", [])[:5]
    if rows:
        tk_db.insert("signals", [{**r} for r in rows])
    # prune anything older than 28 days (the anomaly layer prunes its own at 7)
    try:
        cutoff = (dt.datetime.utcnow() - dt.timedelta(days=28)).isoformat() + "Z"
        qs = urllib.parse.urlencode({"created_at": "lt." + cutoff})
        req = urllib.request.Request(
            "%s/rest/v1/signals?%s" % (os.environ["SUPABASE_URL"].rstrip("/"), qs),
            headers={"apikey": os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                     "Authorization": "Bearer " + os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                     "Accept-Profile": "taskapp", "Content-Profile": "taskapp"},
            method="DELETE")
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        LOG.exception("signal prune failed (non-fatal)")
    return {"written": len(rows),
            "headlines": [r["headline"] for r in rows]}
