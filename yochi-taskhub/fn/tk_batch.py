# -*- coding: utf-8 -*-
"""Project batch runs: execute the task agent across every open task in a
project (queue-triggered worker, no HTTP timeout), write live progress to
batch_runs, finish with an AI batch summary, and support 'steer' follow-ups
that re-run chosen tasks with the user's direction."""
import json
import logging

import tk_agent
import tk_ai
import tk_db
import tk_email
import tk_position

LOG = logging.getLogger("tk_batch")

MAX_TASKS_PER_BATCH = 30

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "debrief": {"type": "string", "description": "the batch debrief"},
        "updated_position": {"type": "string", "description": "the full updated position document"},
    },
    "required": ["debrief", "updated_position"],
}

SUMMARY_SYSTEM = (
    "You are summarising an automated run of the task agent across a finance project's open "
    "tasks, AND updating the project's living position document. You get: the PREVIOUS "
    "position (with any interim events), and this run's per-task outcomes.\n\n"
    "Return debrief: 1) OVERALL - one paragraph, INCLUDING what changed vs the previous "
    "position (improved/worsened/unchanged). 2) NEEDS A HUMAN - worst first, one key number "
    "each; mark items that are repeats from the previous position as '(carried, first seen "
    "<date>)'. 3) CLEAN. 4) SUGGESTED STEER - 2-3 concrete directions. Under 300 words.\n\n"
    "Return updated_position: the NEW full position document following the position rules "
    "given. LANGUAGE RULE everywhere: Fast Food Award items are a heuristic screen, NEVER "
    "'breaches'/'violations' - always 'areas to review'."
)

STEER_SCHEMA = {
    "type": "object",
    "properties": {
        "reruns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "feedback": {"type": "string", "description": "instruction for this task's re-run"},
                },
                "required": ["task_id", "feedback"],
            },
        },
        "note": {"type": "string", "description": "one line on how the steer was interpreted"},
    },
    "required": ["reruns"],
}

STEER_SYSTEM = (
    "The user is steering an automated batch run over project tasks. Given their direction, "
    "the task list and each task's last outcome, decide which tasks to re-run and with what "
    "specific feedback instruction each. Only include tasks the direction actually applies "
    "to. If the direction names work that matches no task, leave reruns empty and explain "
    "in note."
)


def _patch(batch_id, fields):
    tk_db.patch("batch_runs", {"id": "eq." + batch_id}, fields)


def enqueue_triage(task_ids, requested_by):
    """Create queued 'triage' batches (one per project) for freshly created
    watcher/GL-sweep tasks. Returns the batch ids - the caller pushes them to
    the taskhub-batch queue."""
    if not task_ids:
        return []
    tasks = tk_db.get("tasks", {"id": "in.(%s)" % ",".join(task_ids),
                                "select": "id,project_id"})
    by_project = {}
    for t in tasks:
        by_project.setdefault(t["project_id"], []).append(t["id"])
    batch_ids = []
    for project_id, ids in by_project.items():
        rows = tk_db.insert("batch_runs", [{
            "project_id": project_id,
            "kind": "triage",
            "task_ids": ids,
            "requested_by": requested_by,
        }], returning=True)
        batch_ids.append(rows[0]["id"])
    return batch_ids


def process(batch_id):
    rows = tk_db.get("batch_runs", {"id": "eq." + batch_id, "select": "*"})
    if not rows:
        LOG.error("batch %s not found", batch_id)
        return
    batch = rows[0]
    if batch["status"] not in ("queued",):
        LOG.info("batch %s already %s - skipping", batch_id, batch["status"])
        return
    _patch(batch_id, {"status": "running"})
    try:
        if batch["kind"] == "steer":
            targets = _steer_targets(batch)
        elif batch["kind"] == "triage":
            ids = (batch.get("task_ids") or [])[:MAX_TASKS_PER_BATCH]
            tasks = tk_db.get("tasks", {"id": "in.(%s)" % ",".join(ids),
                                        "status": "neq.done",
                                        "select": "id,title"}) if ids else []
            targets = [{"task_id": t["id"], "title": t["title"], "feedback": None}
                       for t in tasks]
        else:
            tasks = tk_db.get("tasks", {
                "project_id": "eq." + batch["project_id"],
                "status": "neq.done",
                "select": "id,title",
                "order": "priority.desc,due_date.asc",
                "limit": str(MAX_TASKS_PER_BATCH),
            })
            targets = [{"task_id": t["id"], "title": t["title"], "feedback": None}
                       for t in tasks]

        results = []
        total = len(targets)
        for i, t in enumerate(targets):
            _patch(batch_id, {"progress": {
                "done": i, "total": total, "current": t.get("title") or t["task_id"],
                "results": results[-10:]}})
            try:
                plan = tk_agent.propose(t["task_id"])
                if not plan.get("all_valid"):
                    results.append({"task_id": t["task_id"], "title": t.get("title"),
                                    "ok": False, "note": "plan queries failed validation"})
                    continue
                r = tk_agent.execute(t["task_id"], plan, batch["requested_by"],
                                     feedback=t.get("feedback"))
                results.append({"task_id": t["task_id"], "title": t.get("title"),
                                "ok": True, "summary": r["summary"][:500]})
            except Exception as e:
                LOG.exception("batch task %s failed", t["task_id"])
                results.append({"task_id": t["task_id"], "title": t.get("title"),
                                "ok": False, "note": str(e)[:200]})

        prev = tk_position.latest(batch["project_id"])
        prev_txt = "(none - this is the first consolidated position)"
        if prev:
            prev_txt = prev["content"]
            if prev.get("events"):
                prev_txt += "\n\nInterim events since that position:\n" + "\n".join(
                    "- %s %s" % (e.get("at", "")[:10], e.get("text", ""))
                    for e in prev["events"])
        out = tk_ai.structured(
            SUMMARY_SYSTEM,
            "%s\n\nPREVIOUS POSITION:\n%s\n\nTHIS RUN'S RESULTS (%d tasks):\n%s%s" % (
                tk_position.merge_guide(), prev_txt, total,
                json.dumps(results, separators=(",", ":"))[:55000],
                ("\n\nUser steer for this batch: " + batch["steer_text"])
                if batch.get("steer_text") else ""),
            "debrief_and_position", SUMMARY_SCHEMA, max_tokens=4000)
        summary = out["debrief"]
        version = tk_position.save_version(
            batch["project_id"], out["updated_position"],
            "%s:%s" % (batch["kind"], batch_id))
        summary += "\n\n(Position updated to v%d — see the 📍 Position panel.)" % version
        _patch(batch_id, {
            "status": "done",
            "summary": summary,
            "progress": {"done": total, "total": total, "results": results},
        })
        req = tk_db.get("profiles", {"id": "eq." + batch["requested_by"],
                                     "select": "email"})
        if req:
            subject = ("Auto-triage complete — %d task(s) pre-worked"
                       if batch["kind"] == "triage"
                       else "Batch run complete — %d task(s)") % total
            tk_email.send(req[0]["email"], subject,
                          "<pre style='font-family:inherit;white-space:pre-wrap'>%s</pre>"
                          % summary.replace("&", "&amp;").replace("<", "&lt;"))
    except Exception as e:
        LOG.exception("batch %s failed", batch_id)
        _patch(batch_id, {"status": "error", "error": str(e)[:300]})


def _steer_targets(batch):
    """AI interprets the steer text against the parent batch's results."""
    parent = tk_db.get("batch_runs", {"id": "eq." + batch["parent_batch_id"],
                                      "select": "progress"})
    results = (parent[0]["progress"] or {}).get("results", []) if parent else []
    out = tk_ai.structured(
        STEER_SYSTEM,
        "User direction:\n%s\n\nTasks and last outcomes:\n%s" % (
            batch["steer_text"], json.dumps(results, separators=(",", ":"))[:40000]),
        "plan_steer", STEER_SCHEMA, max_tokens=2000)
    by_id = {r["task_id"]: r.get("title") for r in results}
    return [{"task_id": r["task_id"], "title": by_id.get(r["task_id"]),
             "feedback": r["feedback"]}
            for r in out.get("reruns", []) if r.get("task_id") in by_id]
