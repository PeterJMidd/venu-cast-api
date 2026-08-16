# -*- coding: utf-8 -*-
"""Smart task setup: the user describes exactly what a task should DO
('check payroll tax was paid on time for each state...'). The planner maps it
to the lake + feed tables, writes a DATA PLAN into the task, proposes any
missing external feeds into the registry (status 'recommended' - the user
activates them), creates the task, and hands back everything needed for the
agent to execute it immediately."""
import datetime as dt
import json
import logging

import tk_ai
import tk_db
import tk_skillbuilder

LOG = logging.getLogger("tk_smart")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "sharp task title, <=100 chars"},
        "project_id": {"type": "string", "description": "uuid of the best-fit project from the list"},
        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "data_plan": {"type": "string",
                      "description": "numbered method: which tables/feeds to query, what to check, what evidence closes the task"},
        "lake_tables": {"type": "array", "items": {"type": "string"}},
        "feeds_used": {"type": "array", "items": {"type": "string"},
                       "description": "slugs of existing feeds the plan relies on"},
        "feeds_missing": {"type": "array", "maxItems": 3,
            "items": {"type": "object", "properties": {
                "slug": {"type": "string"}, "name": {"type": "string"},
                "kind": {"type": "string", "enum": ["award", "tax_rates", "due_dates", "economy", "industry", "other"]},
                "why": {"type": "string"},
                "research_prompt": {"type": "string", "description": "exact web research instruction for the weekly run"},
                "columns": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["name", "description"]}}},
                "required": ["slug", "name", "kind", "why", "research_prompt", "columns"]}},
        "recurrence": {"type": "string", "description": "suggested cadence in plain words, or 'one-off'"},
    },
    "required": ["title", "project_id", "priority", "data_plan", "lake_tables",
                 "feeds_used", "feeds_missing", "recurrence"],
}

SYSTEM = (
    "You design executable data-checking tasks for Yo-Chi's finance TaskHub. Given the "
    "user's specification, the lake schema, the external feed registry (feed_<slug> lake "
    "tables) and the project list, produce the plan. data_plan must be concrete enough "
    "for the task agent to run: name the exact tables/feeds and the comparison logic "
    "(e.g. 'join feed_state_payroll_tax due dates to Invoices/GL payments to each state "
    "revenue office; flag any month paid after due'). If required external data has no "
    "feed yet, propose it in feeds_missing (tight research_prompt, minimal columns). "
    "Fast Food Award items are 'areas to review', never 'breaches'. Money is AUD net of GST."
)

PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"


def plan_and_create(description, requested_by=None, due_date=None):
    feeds = tk_db.get("feeds", {"select": "slug,name,status,description,cadence"})
    projects = tk_db.get("projects", {"select": "id,name,category_id",
                                      "archived": "eq.false"})
    user = ("TASK SPECIFICATION:\n%s\n\nFEED REGISTRY:\n%s\n\nPROJECTS:\n%s\n\n"
            "LAKE SCHEMA:\n%s") % (
        description[:3000],
        json.dumps(feeds, separators=(",", ":")),
        json.dumps(projects, separators=(",", ":")),
        tk_skillbuilder._schema_text())
    plan = tk_ai.structured(SYSTEM, user, "plan_smart_task", PLAN_SCHEMA, max_tokens=3000)

    valid_projects = {p["id"] for p in projects}
    if plan["project_id"] not in valid_projects:
        plan["project_id"] = "aaaaaaaa-0000-0000-0000-000000000019"  # Finance operations

    # register missing feeds as recommendations (user activates in Admin -> Data feeds)
    new_feeds = []
    known = {f["slug"] for f in feeds}
    for nf in plan.get("feeds_missing", [])[:3]:
        if nf["slug"] in known:
            continue
        try:
            tk_db.insert("feeds", [{
                "slug": nf["slug"][:60], "name": nf["name"][:120], "kind": nf["kind"],
                "description": nf["why"][:400],
                "research_prompt": nf["research_prompt"][:1500],
                "columns": nf["columns"][:8],
                "cadence": "weekly", "status": "recommended",
            }], on_conflict="slug", ignore_duplicates=True)
            new_feeds.append(nf["slug"])
        except Exception:
            LOG.exception("feed proposal %s failed", nf.get("slug"))

    inactive = [f["slug"] for f in feeds
                if f["slug"] in plan.get("feeds_used", []) and f["status"] != "active"]
    notes = []
    if inactive:
        notes.append("Feeds to ACTIVATE in Admin → Data feeds: " + ", ".join(inactive))
    if new_feeds:
        notes.append("New feeds proposed (activate to enable): " + ", ".join(new_feeds))

    desc = ("%s\n\nDATA PLAN (agent-executable)\n%s\n\nSources: lake [%s]%s\n"
            "Recurrence: %s%s") % (
        description.strip(), plan["data_plan"],
        ", ".join(plan.get("lake_tables", [])),
        (" + feeds [%s]" % ", ".join(plan["feeds_used"])) if plan.get("feeds_used") else "",
        plan.get("recurrence", "one-off"),
        ("\n\n" + "\n".join(notes)) if notes else "")

    rows = tk_db.insert("tasks", [{
        "project_id": plan["project_id"],
        "title": plan["title"][:200],
        "description": desc[:4000],
        "priority": plan["priority"],
        "assignee_id": requested_by or PETER,
        "due_date": due_date,
        "source": "manual",
    }], returning=True)
    task_id = rows[0]["id"]
    return {"task_id": task_id, "title": plan["title"],
            "project_id": plan["project_id"], "data_plan": plan["data_plan"],
            "feeds_used": plan.get("feeds_used", []),
            "feeds_to_activate": inactive, "feeds_proposed": new_feeds,
            "recurrence": plan.get("recurrence")}
