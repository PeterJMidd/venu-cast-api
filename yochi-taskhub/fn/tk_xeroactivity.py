# -*- coding: utf-8 -*-
"""Xero per-user activity ingest. Xero's API carries no user attribution -
only the UI's History & Notes / Assurance dashboard does - so a scheduled
desktop task drives the logged-in Xero tab each morning, reads the prior day's
History & Notes for BOTH entities, aggregates user x type x action counts, and
POSTs them to ops_xero_activity. Rows land in the lake as
feed_xero_user_activity (tk_feeds parquet append + catalog registration), and
tk_workreport folds them into the daily work-activity report."""
import logging

import tk_feeds

LOG = logging.getLogger("tk_xeroactivity")

COLUMNS = [
    {"name": "activity_date", "description": "AEST date the actions happened (YYYY-MM-DD)"},
    {"name": "entity", "description": "Xero organisation name"},
    {"name": "xero_user", "description": "Xero user who performed the actions"},
    {"name": "item_type", "description": "document type: Manual Journal / Invoice / Bill / Bank Transaction / Contact / ..."},
    {"name": "action", "description": "what was done: Created / Edited / Approved / Deleted / Voided / ..."},
    {"name": "items", "description": "number of documents"},
    {"name": "detail", "description": "optional note or sample references"},
]
REQUIRED = ("activity_date", "entity", "xero_user", "item_type", "action", "items")


def ingest(body):
    rows = (body or {}).get("rows") or []
    clean = []
    for r in rows[:500]:
        if not all(str(r.get(k, "")).strip() for k in REQUIRED):
            continue
        clean.append({c["name"]: str(r.get(c["name"], "")) for c in COLUMNS})
    if not clean:
        return {"rows": 0, "error": "no valid rows (need %s)" % ", ".join(REQUIRED)}
    total = tk_feeds._append_parquet("xero_user_activity", COLUMNS, clean)
    return {"rows": len(clean), "table_total": total,
            "table": "feed_xero_user_activity"}
