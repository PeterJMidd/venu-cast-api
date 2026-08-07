# -*- coding: utf-8 -*-
"""Natural-language task entry: free text in, structured task out (cheap model,
forced tool-use). The SPA shows the parsed task for confirmation before insert."""
import datetime as dt
import json
import logging

import tk_ai
import tk_db

LOG = logging.getLogger("tk_nl")

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short imperative task title"},
        "description": {"type": ["string", "null"]},
        "project_id": {"type": "string", "description": "id of the best-matching project"},
        "assignee_email": {"type": ["string", "null"], "description": "email of the assignee, if named"},
        "reviewer_email": {"type": ["string", "null"]},
        "due_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "checklist": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "project_id", "priority"],
}

SYSTEM = (
    "You turn a finance team member's free-text note into a structured task for the "
    "Yo-Chi TaskHub. Pick the best-matching project from the list (fall back to the most "
    "general one in the right category). Resolve people by fuzzy first-name match against "
    "the user list. Dates: resolve relative phrases ('by the 21st', 'next Friday', 'end of "
    "month') against today's date given. If no assignee is named, leave assignee_email null "
    "(the requester will own it). Keep titles short and imperative. Only extract a checklist "
    "when the note clearly lists sub-steps."
)


def parse(text, requester_email):
    profiles = tk_db.get("profiles", {"active": "eq.true", "select": "email,full_name,role"})
    projects = tk_db.get("projects", {"archived": "eq.false",
                                      "select": "id,name,description,category_id"})
    cats = tk_db.get("categories", {"select": "id,name"})
    user = (
        "Today (AEST): %s\nRequester: %s\n\nUsers:\n%s\n\nCategories:\n%s\n\nProjects:\n%s\n\n"
        "Note to convert:\n%s"
    ) % (
        dt.date.today().isoformat(), requester_email,
        json.dumps(profiles, separators=(",", ":")),
        json.dumps(cats, separators=(",", ":")),
        json.dumps(projects, separators=(",", ":")),
        text,
    )
    out = tk_ai.structured(SYSTEM, user, "create_task", TOOL_SCHEMA, cheap=True)

    # resolve emails -> profile ids so the SPA can insert directly
    by_email = {p["email"].lower(): p for p in profiles}

    def resolve(email):
        if not email:
            return None
        p = by_email.get(email.lower())
        return p and email

    out["assignee_email"] = resolve(out.get("assignee_email"))
    out["reviewer_email"] = resolve(out.get("reviewer_email"))
    return out
