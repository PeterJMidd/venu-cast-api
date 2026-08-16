# -*- coding: utf-8 -*-
"""Decision packs: every open CRITICAL task should arrive with an agent-
prepared recommendation (the way the surcharge war-game did). Daily 07:35:
find open critical tasks with no successful agent run in the last 7 days and
queue up to 2 for agent pre-work via the triage machinery."""
import datetime as dt
import logging

import tk_db

LOG = logging.getLogger("tk_decisions")

MAX_PER_DAY = 2


def pick():
    crit = tk_db.get("tasks", {"status": "neq.done", "priority": "eq.critical",
                               "select": "id,title", "order": "due_date.asc.nullslast",
                               "limit": "50"})
    if not crit:
        return []
    cutoff = (dt.datetime.utcnow() - dt.timedelta(days=7)).isoformat()
    recent = {r["task_id"] for r in tk_db.get(
        "agent_runs", {"outcome": "eq.success", "created_at": "gte." + cutoff,
                       "select": "task_id", "limit": "500"})}
    todo = [t for t in crit if t["id"] not in recent]
    return [t["id"] for t in todo[:MAX_PER_DAY]]
