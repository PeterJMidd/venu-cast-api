# -*- coding: utf-8 -*-
"""The living position: a versioned cumulative state document per project.
Individual agent runs append lightweight events; every batch/steer run
consolidates events + new results into the next version via an AI merge, so
each run CHANGES the incremental position instead of starting fresh."""
import datetime as dt
import json
import logging

import tk_db

LOG = logging.getLogger("tk_position")

MERGE_GUIDE = (
    "POSITION DOCUMENT RULES: maintain a living state document (plain text) with these "
    "sections: STATUS (2-3 sentences, overall state as of today); RESOLVED THIS RUN "
    "(what moved to done/confirmed, with the evidence); OPEN ITEMS (carried forward - "
    "every unresolved item from the previous position MUST appear here unless explicitly "
    "resolved, each with [first seen: date]); NEW THIS RUN; WATCHLIST (things to keep an "
    "eye on). Date-stamp today as %s. Never silently drop an open item - resolve it or "
    "carry it. Keep under 600 words: consolidate duplicates, tighten wording, but the "
    "position must remain complete. Fast Food Award items are 'areas to review', never "
    "'breaches'."
)


def latest(project_id):
    rows = tk_db.get("positions", {
        "project_id": "eq." + project_id,
        "order": "version.desc", "limit": "1", "select": "*"})
    return rows[0] if rows else None


def append_event(project_id, text):
    """Cheap delta recorded between consolidations (no AI call)."""
    try:
        cur = latest(project_id)
        event = {"at": dt.datetime.utcnow().isoformat() + "Z", "text": text[:400]}
        if cur:
            events = (cur.get("events") or []) + [event]
            tk_db.patch("positions", {"id": "eq." + cur["id"]}, {"events": events[-40:]})
        else:
            tk_db.insert("positions", [{
                "project_id": project_id, "version": 1,
                "content": "STATUS\nPosition tracking started %s. First consolidation "
                           "will occur on the next batch run." % dt.date.today().isoformat(),
                "events": [event], "source": "bootstrap"}])
    except Exception:
        LOG.exception("append_event failed (non-fatal)")


def merge_guide():
    return MERGE_GUIDE % dt.date.today().isoformat()


def save_version(project_id, content, source):
    cur = latest(project_id)
    version = (cur["version"] + 1) if cur else 1
    tk_db.insert("positions", [{
        "project_id": project_id, "version": version,
        "content": content, "events": [], "source": source}])
    return version
