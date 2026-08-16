# -*- coding: utf-8 -*-
"""Escalation & nudge engine + close auto-assignment. Deterministic (no AI).

Daily on business days:
  1. auto_assign - open unassigned template/register tasks whose description
     starts "Owner: <name>" are matched against active profiles by name and
     assigned (the template's default_assignee_id is back-filled too, so future
     months assign at creation). New assignees get one welcome email listing
     their tasks.
  2. Nudge rules over open tasks (dedup: skip a (task, rule) that was nudged
     in the last RENUDGE_DAYS via taskapp.escalations):
       critical_overdue  - critical, due > 2 days ago        -> assignee
       review_stall      - waiting_review, idle > 3 days     -> reviewer
       stale_inprogress  - in_progress, idle > 7 days        -> assignee
       long_overdue      - any open task > 14 days past due  -> admin digest
     One email per person listing all their nudges; one digest to admins
     covering everything triggered today.
"""
import datetime as dt
import logging
import os
import re

import tk_calendar
import tk_db
import tk_email

LOG = logging.getLogger("tk_escalate")

RENUDGE_DAYS = 3
CRIT_OVERDUE_DAYS = 2
REVIEW_STALL_DAYS = 3
STALE_DAYS = 7
LONG_OVERDUE_DAYS = 14

RULE_LABELS = {
    "critical_overdue": "Critical and overdue",
    "review_stall": "Waiting on your review",
    "stale_inprogress": "In progress but idle a week",
    "long_overdue": "Overdue 14+ days",
}


def _app_url():
    return os.environ.get("APP_URL", "").rstrip("/")


def _task_link(t):
    return "%s/my-tasks?task=%s" % (_app_url(), t["id"])


def _owner_from_description(desc):
    """'Owner: Sally Bourchier. ...' -> 'sally bourchier'; None for multi-owner
    lines ('Owners: state leads (...)') or anything ambiguous."""
    m = re.match(r"^Owner:\s*([^.\n(]+)", desc or "")
    if not m:
        return None
    name = m.group(1).strip().lower()
    if not name or "," in name or " and " in name or "lead" in name:
        return None
    return name


def _match_profile(owner, profiles):
    """Exact full-name match first, else unique first-name match."""
    exact = [p for p in profiles if (p.get("full_name") or "").strip().lower() == owner]
    if len(exact) == 1:
        return exact[0]
    first = owner.split()[0]
    firsts = [p for p in profiles
              if (p.get("full_name") or "").strip().lower().split(" ")[0] == first]
    return firsts[0] if len(firsts) == 1 else None


def auto_assign():
    profiles = tk_db.get("profiles", {"active": "eq.true",
                                      "select": "id,email,full_name"})
    open_unassigned = tk_db.get("tasks", {
        "status": "neq.done", "assignee_id": "is.null",
        "description": "not.is.null",
        "select": "id,title,description,due_date,template_id,project_id"})
    assigned = {}  # profile id -> [task]
    templates_patched = set()
    for t in open_unassigned:
        owner = _owner_from_description(t["description"])
        if not owner:
            continue
        p = _match_profile(owner, profiles)
        if not p:
            continue
        tk_db.patch("tasks", {"id": "eq." + t["id"]}, {"assignee_id": p["id"]})
        assigned.setdefault(p["id"], []).append(t)
        # future months: assign at creation
        if t.get("template_id") and t["template_id"] not in templates_patched:
            tk_db.patch("task_templates",
                        {"id": "eq." + t["template_id"],
                         "default_assignee_id": "is.null"},
                        {"default_assignee_id": p["id"]})
            templates_patched.add(t["template_id"])
    people = {p["id"]: p for p in profiles}
    for pid, tasks in assigned.items():
        p = people[pid]
        items = "".join(
            "<li><a href='%s'>%s</a>%s</li>" % (
                _task_link(t), t["title"],
                " — due %s" % t["due_date"] if t.get("due_date") else "")
            for t in tasks)
        try:
            tk_email.send(
                p["email"],
                "TaskHub: %d task(s) assigned to you" % len(tasks),
                "<p>Hi %s,</p><p>These recurring tasks name you as owner and have "
                "been assigned to you in TaskHub:</p><ul>%s</ul>"
                "<p><a href='%s/my-tasks'>Open my tasks</a></p>"
                % ((p.get("full_name") or "").split(" ")[0] or "there",
                   items, _app_url()))
        except Exception:
            LOG.exception("assign email failed for %s", p["email"])
    return {"assigned": sum(len(v) for v in assigned.values()),
            "people": len(assigned), "templates_backfilled": len(templates_patched)}


def _recent_escalations():
    since = (dt.datetime.utcnow() - dt.timedelta(days=RENUDGE_DAYS)).isoformat() + "Z"
    rows = tk_db.get("escalations", {"sent_at": "gte." + since,
                                     "select": "task_id,rule"})
    return {(r["task_id"], r["rule"]) for r in rows}


def _trigger_rules(t, today, now):
    out = []
    due = t.get("due_date")
    upd = (t.get("updated_at") or "")[:19]
    idle_days = None
    if upd:
        try:
            idle_days = (now - dt.datetime.fromisoformat(upd)).days
        except ValueError:
            pass
    if due:
        over = (today - dt.date.fromisoformat(due)).days
        if t["priority"] == "critical" and over > CRIT_OVERDUE_DAYS:
            out.append("critical_overdue")
        if over > LONG_OVERDUE_DAYS:
            out.append("long_overdue")
    if t["status"] == "waiting_review" and t.get("reviewer_id") \
            and idle_days is not None and idle_days > REVIEW_STALL_DAYS:
        out.append("review_stall")
    if t["status"] == "in_progress" and t.get("assignee_id") \
            and idle_days is not None and idle_days > STALE_DAYS:
        out.append("stale_inprogress")
    return out


def _nudge_email(name, groups):
    parts = ["<p>Hi %s,</p><p>The following need a push along:</p>" % name]
    for rule, tasks in groups.items():
        parts.append("<h3 style='margin:14px 0 4px'>%s</h3><ul>" % RULE_LABELS[rule])
        for t in tasks:
            parts.append("<li><a href='%s'>%s</a>%s</li>" % (
                _task_link(t), t["title"],
                " — due %s" % t["due_date"] if t.get("due_date") else ""))
        parts.append("</ul>")
    parts.append("<p style='color:#6b7280;font-size:12px'>You'll only be re-nudged "
                 "about a task every %d days. Close it out or update it to stop the "
                 "chase.</p>" % RENUDGE_DAYS)
    return "".join(parts)


def run(force=False):
    today = dt.date.today()
    if not force and not tk_calendar.is_business_day(today):
        return {"skipped": "not a business day"}
    assign_out = auto_assign()
    now = dt.datetime.utcnow()
    tasks = tk_db.get("tasks", {
        "status": "neq.done",
        "select": "id,title,status,priority,due_date,updated_at,"
                  "assignee_id,reviewer_id,project_id"})
    recent = _recent_escalations()
    profiles = {p["id"]: p for p in tk_db.get(
        "profiles", {"active": "eq.true", "select": "id,email,full_name,role"})}

    per_person = {}   # uid -> {rule: [task]}
    digest_rows = []  # (rule, task, who)
    to_record = []
    for t in tasks:
        for rule in _trigger_rules(t, today, now):
            if (t["id"], rule) in recent:
                continue
            target = t.get("reviewer_id") if rule == "review_stall" else t.get("assignee_id")
            who = profiles.get(target)
            if rule != "long_overdue" and who:
                per_person.setdefault(target, {}).setdefault(rule, []).append(t)
            digest_rows.append((rule, t, (who or {}).get("full_name") or "unassigned"))
            to_record.append({"task_id": t["id"], "rule": rule})

    sent = 0
    for uid, groups in per_person.items():
        p = profiles[uid]
        try:
            if tk_email.send(p["email"], "TaskHub nudges — %d item(s) need you"
                             % sum(len(v) for v in groups.values()),
                             _nudge_email((p.get("full_name") or "").split(" ")[0]
                                          or "there", groups)):
                sent += 1
        except Exception:
            LOG.exception("nudge email failed for %s", p["email"])

    if digest_rows:
        admins = [p for p in profiles.values() if p["role"] == "admin"]
        rows = "".join(
            "<tr><td style='padding:3px 10px 3px 0;white-space:nowrap'>%s</td>"
            "<td style='padding:3px 10px 3px 0'><a href='%s'>%s</a></td>"
            "<td style='padding:3px 0'>%s</td></tr>"
            % (RULE_LABELS[r], _task_link(t), t["title"], w)
            for r, t, w in digest_rows)
        html = ("<p>Escalations triggered this morning (%d):</p>"
                "<table style='font-size:13px'>%s</table>"
                "<p>Assignment sweep: %d task(s) auto-assigned to %d person/people.</p>"
                % (len(digest_rows), rows, assign_out["assigned"], assign_out["people"]))
        for a in admins:
            try:
                tk_email.send(a["email"], "TaskHub escalation digest — %d item(s)"
                              % len(digest_rows), html)
            except Exception:
                LOG.exception("digest email failed for %s", a["email"])

    if to_record:
        tk_db.insert("escalations", to_record)
    return {"auto_assign": assign_out, "triggered": len(digest_rows),
            "nudge_emails": sent, "email_enabled": tk_email.enabled()}
