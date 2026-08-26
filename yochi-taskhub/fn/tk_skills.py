# -*- coding: utf-8 -*-
"""AI skills: reusable scheduled review routines. Each skill = a review brief
(prompt) + data pulls (DuckDB SQL over the lake) + a cadence. When due, the
runner executes the queries, has Claude write the review, and raises a task
(optionally emailing the review to the assignee)."""
import datetime as dt
import json
import logging
import re

import tk_ai
import tk_calendar
import tk_db
import tk_email
import tk_templates
import lake_reader

LOG = logging.getLogger("tk_skills")

SYSTEM = (
    "You are the finance review analyst for Yo-Chi, an Australian frozen-yoghurt chain. "
    "You are running a recurring review 'skill'. You get the skill's brief and fresh data "
    "pulls (JSON; money is AUD net of GST unless stated otherwise). Write the review as "
    "PLAIN TEXT (no markdown symbols): a one-line overall assessment (Good / Watch / Action "
    "needed) with a one-sentence reason, then short titled sections per the brief, each with "
    "specific venues/numbers, then 'Actions:' with 2-4 concrete next steps. Only use the data "
    "given - if a pull is empty or errored, say so plainly. Never call award items 'breaches'; "
    "use 'areas to review'. Keep it under ~350 words - this is a working review, not a report."
)


def _due(skill, today):
    last = skill.get("last_run_at")
    last_d = dt.date.fromisoformat(last[:10]) if last else None
    if last_d == today:
        return False
    cadence = skill["cadence"]
    if cadence == "daily":
        return True
    if cadence == "weekly":
        wd = skill.get("weekday")
        return today.weekday() == (wd if wd is not None else 0)
    if cadence == "monthly":
        # first business day of the month
        return today == tk_calendar.business_day_of_month(today.year, today.month, 1)
    return False


def run(force_skill_id=None):
    today = dt.date.today()
    period_id, _ = tk_templates.ensure_period(today)
    skills = tk_db.get("ai_skills", {"active": "eq.true", "select": "*"})
    if force_skill_id:
        skills = [s for s in skills if s["id"] == force_skill_id]
    due = [s for s in skills if force_skill_id or _due(s, today)]
    if not due:
        return {"skills": len(skills), "run": 0}

    # sync every table the due skills reference
    referenced = set()
    for s in due:
        for q in s["data_queries"]:
            referenced.update(re.findall(r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_][A-Za-z0-9_]*)\"?",
                                         q.get("sql", ""), re.I))
    lake_reader.sync(extra_tables=referenced, log=LOG.info)

    ran, errors = 0, []
    for s in due:
        try:
            if s.get("task_id"):
                # a task routine: run through the routine engine, which posts
                # to its task, emails its saved recipients in their chosen
                # formats, and records the learning - raising a brand-new task
                # each run is the standalone skills' behaviour, not this one's
                import tk_taskskill
                tk_taskskill.execute(s)
                ran += 1
                continue
            pulls = {}
            for q in s["data_queries"]:
                label = q.get("label", "data")
                try:
                    pulls[label] = lake_reader.query(q["sql"], max_rows=200)
                except Exception as e:
                    pulls[label] = {"error": str(e)[:200]}
            user = "Skill: %s\nToday: %s\n\nBrief:\n%s\n\nData pulls:\n%s" % (
                s["name"], today.isoformat(), s["prompt"],
                json.dumps(pulls, separators=(",", ":"))[:60000])
            review = tk_ai.text(SYSTEM, user, max_tokens=1500)

            due_date = tk_calendar.roll_forward(today + dt.timedelta(days=2))
            tk_db.insert("tasks", [{
                "project_id": s["project_id"],
                "period_id": period_id,
                "title": "[Review] %s — %s" % (s["name"], today.strftime("%d %b")),
                "description": review,
                "priority": "medium",
                "assignee_id": s.get("assignee_id"),
                "due_date": due_date.isoformat(),
                "source": "watcher",
            }])
            if s.get("email_review") and s.get("assignee_id"):
                prof = tk_db.get("profiles", {"id": "eq." + s["assignee_id"],
                                              "select": "email,full_name"})
                if prof:
                    html = "<pre style='font-family:inherit;white-space:pre-wrap'>%s</pre>" % (
                        review.replace("&", "&amp;").replace("<", "&lt;"))
                    tk_email.send(prof[0]["email"], "AI review: %s" % s["name"], html)
            tk_db.patch("ai_skills", {"id": "eq." + s["id"]},
                        {"last_run_at": dt.datetime.utcnow().isoformat() + "Z"})
            ran += 1
        except Exception as e:
            LOG.exception("skill %s failed", s.get("name"))
            errors.append({"skill": s.get("name"), "error": str(e)[:200]})
    return {"skills": len(skills), "run": ran, "errors": errors}
