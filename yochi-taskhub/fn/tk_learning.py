# -*- coding: utf-8 -*-
"""Monthly learning loop: analyse how templated work actually ran over the last
3 complete months, ask Claude for template/deadline improvements, store them in
ai_suggestions and email admins a summary."""
import datetime as dt
import json
import logging
import os

import tk_ai
import tk_db
import tk_email

LOG = logging.getLogger("tk_learning")

SYSTEM = (
    "You are the continuous-improvement analyst for a finance team's task system. You "
    "get per-template statistics for the last 3 complete months (days_late>0 = missed "
    "deadline; reassignments = how often the task changed hands after creation). Suggest "
    "changes ONLY where the data supports them: consistently late -> propose a later "
    "due_rule or a different default assignee; never late and low-effort -> maybe an "
    "earlier due date is fine but do NOT churn for its own sake. Return 0-5 suggestions."
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["template_change", "deadline_change", "other"]},
                    "template_id": {"type": ["string", "null"]},
                    "template_title": {"type": "string"},
                    "change": {"type": "string", "description": "the concrete proposed change"},
                    "rationale": {"type": "string"},
                },
                "required": ["kind", "template_title", "change", "rationale"],
            },
        }
    },
    "required": ["suggestions"],
}


def _stats():
    """Per-template outcomes over the last 3 complete months."""
    cutoff = (dt.date.today().replace(day=1) - dt.timedelta(days=92)).isoformat()
    tasks = tk_db.get("tasks", {
        "template_id": "not.is.null",
        "created_at": "gte." + cutoff,
        "select": "id,template_id,title,status,due_date,completed_at",
    })
    audit = tk_db.get("audit_log", {
        "table_name": "eq.tasks",
        "action": "eq.UPDATE",
        "at": "gte." + cutoff,
        "select": "row_id,old_row,new_row",
    })
    reassigned = {}
    for a in audit:
        old_a = (a.get("old_row") or {}).get("assignee_id")
        new_a = (a.get("new_row") or {}).get("assignee_id")
        if old_a != new_a:
            rid = a["row_id"]
            reassigned[rid] = reassigned.get(rid, 0) + 1

    by_tpl = {}
    for t in tasks:
        s = by_tpl.setdefault(t["template_id"], {
            "template_id": t["template_id"], "title": t["title"], "instances": 0,
            "completed": 0, "late": 0, "total_days_late": 0, "reassignments": 0})
        s["instances"] += 1
        s["reassignments"] += reassigned.get(t["id"], 0)
        if t["status"] == "done" and t["completed_at"]:
            s["completed"] += 1
            if t["due_date"]:
                days_late = (dt.date.fromisoformat(t["completed_at"][:10])
                             - dt.date.fromisoformat(t["due_date"])).days
                if days_late > 0:
                    s["late"] += 1
                    s["total_days_late"] += days_late
    return list(by_tpl.values())


def run():
    stats = _stats()
    if not stats:
        return {"suggestions": 0, "note": "no templated task history yet"}
    out = tk_ai.structured(SYSTEM, json.dumps(stats, separators=(",", ":")),
                           "suggest_changes", TOOL_SCHEMA, max_tokens=2000)
    suggestions = out.get("suggestions", [])
    for s in suggestions:
        tk_db.insert("ai_suggestions", [{
            "kind": s["kind"],
            "payload": {"template_id": s.get("template_id"),
                        "template_title": s["template_title"], "change": s["change"]},
            "rationale": s["rationale"],
        }])
    if suggestions:
        admins = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                        "select": "email"})
        items = "".join("<li><b>%s</b>: %s<br><i>%s</i></li>"
                        % (s["template_title"], s["change"], s["rationale"])
                        for s in suggestions)
        html = ("<p>The monthly TaskHub learning pass produced %d suggestion(s):</p>"
                "<ul>%s</ul><p>Review them under Admin &rarr; Suggestions in "
                "<a href='%s'>TaskHub</a>.</p>"
                % (len(suggestions), items, os.environ.get("APP_URL", "")))
        for a in admins:
            tk_email.send(a["email"], "TaskHub learning loop - %d suggestion(s)"
                          % len(suggestions), html)
    return {"suggestions": len(suggestions), "templates_analysed": len(stats)}
