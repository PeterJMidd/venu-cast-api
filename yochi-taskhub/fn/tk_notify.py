# -*- coding: utf-8 -*-
"""Notification drainer: assignments, reviewers, comments and @/email tags.

Every assignment now emails the person, no matter what did the assigning - the
web UI, a watcher rule, the compliance register, the email-drop pipeline, the
escalation engine's owner sweep, or an AI agent. A DB trigger writes one row to
taskapp.notify_outbox whenever assignee_id/reviewer_id changes, and another
does the same for every comment (tagging by email address or @name, plus the
task assignee/reviewer); this drains them
on a one-minute timer and sends via Graph. Self-assignment never emails (the trigger
compares against auth.uid()), and notification_prefs.email_on_assign still
opts a person out."""
import datetime as dt
import logging
import os

import tk_db
import tk_email

LOG = logging.getLogger("tk_notify")

BATCH = 50
MAX_ATTEMPTS = 3
SUBJECTS = {"assigned": "You've been assigned: %s",
            "reviewer": "You're the reviewer on: %s",
            "mention": "You were tagged on: %s",
            "comment": "New comment on: %s"}


def _esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html(task, kind, actor_name, project_name, comment_body=None):
    app_url = os.environ.get("APP_URL", "").rstrip("/")
    link = "%s/my-tasks?task=%s" % (app_url, task["id"])
    bits = []
    if task.get("due_date"):
        bits.append("Due <b>%s</b>" % task["due_date"])
    if task.get("priority"):
        bits.append("Priority <b>%s</b>" % task["priority"])
    if project_name:
        bits.append("Project <b>%s</b>" % _esc(project_name))
    meta = " &middot; ".join(bits)
    desc = (task.get("description") or "").strip()
    if len(desc) > 600:
        desc = desc[:600] + "..."
    role = {"assigned": "You've been assigned this task",
            "reviewer": "You've been set as reviewer on this task",
            "mention": "You were tagged in a comment",
            "comment": "There's a new comment on a task you're on"}.get(
        kind, "Update on this task")
    if comment_body is not None:
        # the comment IS the message - show it instead of the task description
        desc = comment_body.strip()
        if len(desc) > 900:
            desc = desc[:900] + "..."
    return (
        "<div style='font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
        "color:#1f2937;max-width:620px'>"
        "<p style='color:#6b7280;font-size:13px;margin:0 0 4px'>%s%s.</p>"
        "<h2 style='margin:0 0 6px;font-size:17px;color:#1d683d'>%s</h2>"
        "<p style='color:#6b7280;font-size:12px;margin:0 0 12px'>%s</p>"
        "%s"
        "<p style='margin:18px 0 6px'><a href='%s' style='display:inline-block;"
        "background:#23824a;color:#fff;text-decoration:none;font-weight:700;"
        "font-size:14px;padding:10px 20px;border-radius:9px'>Open the task &rarr;</a></p>"
        "<p style='color:#9ca3af;font-size:11px'>Yo-Chi TaskHub</p></div>"
    ) % (role, (" by %s" % _esc(actor_name)) if actor_name else "",
         _esc(task.get("title")), meta,
         ("<div style='white-space:pre-wrap;font-size:13px;color:#374151;"
          "border-left:3px solid #e5e7eb;padding-left:10px'>%s</div>" % _esc(desc))
         if desc else "", link)


def run():
    pending = tk_db.get("notify_outbox", {
        "sent_at": "is.null", "attempts": "lt.%d" % MAX_ATTEMPTS,
        "select": "id,task_id,recipient,kind,actor,attempts,comment_id",
        "order": "created_at", "limit": str(BATCH)})
    if not pending:
        return {"pending": 0, "sent": 0}
    profiles = {p["id"]: p for p in tk_db.get(
        "profiles", {"select": "id,email,full_name,active"})}
    prefs = {p["user_id"]: p for p in tk_db.get(
        "notification_prefs", {"select": "user_id,email_on_assign"})}
    projects = {p["id"]: p["name"] for p in tk_db.get(
        "projects", {"select": "id,name"})}

    sent = skipped = failed = 0
    for row in pending:
        stamp = {"sent_at": dt.datetime.utcnow().isoformat() + "Z"}
        try:
            who = profiles.get(row["recipient"])
            pref = prefs.get(row["recipient"])
            if not who or not who.get("active"):
                tk_db.patch("notify_outbox", {"id": "eq." + row["id"]},
                            dict(stamp, error="recipient inactive/unknown"))
                skipped += 1
                continue
            if row["kind"] == "assigned" and pref \
                    and not pref.get("email_on_assign", True):
                tk_db.patch("notify_outbox", {"id": "eq." + row["id"]},
                            dict(stamp, error="opted out"))
                skipped += 1
                continue
            tasks = tk_db.get("tasks", {
                "id": "eq." + row["task_id"],
                "select": "id,title,description,due_date,priority,project_id,status"})
            if not tasks:
                tk_db.patch("notify_outbox", {"id": "eq." + row["id"]},
                            dict(stamp, error="task gone"))
                skipped += 1
                continue
            task = tasks[0]
            if task.get("status") == "done" and row["kind"] in ("assigned", "reviewer"):
                tk_db.patch("notify_outbox", {"id": "eq." + row["id"]},
                            dict(stamp, error="task already done"))
                skipped += 1
                continue
            comment_body = None
            if row.get("comment_id"):
                crows = tk_db.get("comments", {"id": "eq." + row["comment_id"],
                                               "select": "body"})
                if not crows:
                    tk_db.patch("notify_outbox", {"id": "eq." + row["id"]},
                                dict(stamp, error="comment gone"))
                    skipped += 1
                    continue
                comment_body = crows[0]["body"]
            actor = profiles.get(row.get("actor")) or {}
            actor_name = actor.get("full_name") or actor.get("email") or ""
            subject = SUBJECTS.get(row["kind"], "Update on: %s") % task["title"]
            ok = tk_email.send(who["email"], subject[:200],
                               _html(task, row["kind"], actor_name,
                                     projects.get(task.get("project_id")),
                                     comment_body))
            if ok:
                tk_db.patch("notify_outbox", {"id": "eq." + row["id"]}, stamp)
                sent += 1
            else:
                # email disabled globally - don't retry forever
                tk_db.patch("notify_outbox", {"id": "eq." + row["id"]},
                            dict(stamp, error="email disabled"))
                skipped += 1
        except Exception as e:
            failed += 1
            LOG.exception("notify_outbox row %s failed", row["id"])
            try:
                tk_db.patch("notify_outbox", {"id": "eq." + row["id"]},
                            {"attempts": (row.get("attempts") or 0) + 1,
                             "error": str(e)[:300]})
            except Exception:
                LOG.exception("could not record notify failure")
    return {"pending": len(pending), "sent": sent, "skipped": skipped,
            "failed": failed, "email_enabled": tk_email.enabled()}
