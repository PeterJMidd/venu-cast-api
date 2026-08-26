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

    # a burst of assignments to one person becomes a single digest
    digested = set()
    try:
        digested = _digest(pending, profiles, projects)
    except Exception:
        LOG.exception("digest pass failed - sending individually instead")

    sent = skipped = failed = 0
    for row in pending:
        if row["id"] in digested:
            sent += 1
            continue
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


# --------------------------------------------------------------- digesting

# Above this many assignment notifications waiting for one person, send a
# single digest instead of one email each. A nightly sync that hands somebody
# 71 obligations should arrive as one message they will read, not 71 they will
# filter - and a mailbox rule built to survive that flood would also hide the
# one assignment that mattered.
DIGEST_THRESHOLD = int(os.environ.get("NOTIFY_DIGEST_THRESHOLD", "4"))


def _digest_html(rows, tasks_by_id, projects, name):
    items = []
    for r in rows:
        t = tasks_by_id.get(r["task_id"]) or {}
        items.append(
            "<tr><td style='padding:6px 10px;border-top:1px solid #eee'>"
            "<div style='font-weight:600'>%s</div>"
            "<div style='font-size:11px;color:#666'>%s%s%s</div></td></tr>"
            % ((t.get("title") or "(task)")[:140],
               projects.get(t.get("project_id")) or "",
               " &middot; due %s" % t["due_date"] if t.get("due_date") else "",
               " &middot; %s" % t["priority"] if t.get("priority") else ""))
    return (
        "<div style=\"font-family:-apple-system,Segoe UI,Arial,sans-serif;"
        "max-width:760px;color:#222\">"
        "<h2 style='margin:0 0 4px'>%d task%s assigned to you</h2>"
        "<div style='color:#666;font-size:13px;margin-bottom:14px'>%s - these "
        "arrived together, so here they are in one message rather than %d.</div>"
        "<table style='border-collapse:collapse;width:100%%;font-size:13px'>%s"
        "</table>"
        "<p style='margin-top:16px'><a href='%s/my-tasks' style='background:"
        "#0f766e;color:#fff;padding:9px 16px;border-radius:8px;text-decoration:"
        "none;font-size:13px'>Open my tasks</a></p></div>"
        % (len(rows), "" if len(rows) == 1 else "s", name or "TaskHub",
           len(rows), "".join(items),
           os.environ.get("APP_URL", "").rstrip("/")))


def _digest(rows, profiles, projects):
    """One email per recipient for a burst of assignments. Returns the set of
    outbox ids it handled."""
    handled = set()
    by_person = {}
    for r in rows:
        if r["kind"] == "assigned":
            by_person.setdefault(r["recipient"], []).append(r)
    for uid, rs in by_person.items():
        if len(rs) < DIGEST_THRESHOLD:
            continue
        who = profiles.get(uid)
        if not who or not who.get("active"):
            continue
        ids = [r["task_id"] for r in rs]
        tasks_by_id = {}
        for i in range(0, len(ids), 80):
            chunk = ids[i:i + 80]
            for t in tk_db.get("tasks", {
                    "id": "in.(%s)" % ",".join(chunk),
                    "select": "id,title,due_date,priority,project_id,status"}):
                tasks_by_id[t["id"]] = t
        live = [r for r in rs
                if (tasks_by_id.get(r["task_id"]) or {}).get("status") not in
                (None, "done")]
        if not live:
            continue
        stamp = {"sent_at": dt.datetime.utcnow().isoformat() + "Z"}
        ok = False
        try:
            ok = tk_email.send(
                who["email"],
                "%d tasks assigned to you in TaskHub" % len(live),
                _digest_html(live, tasks_by_id, projects,
                             who.get("full_name") or who.get("email")))
        except Exception:
            LOG.exception("digest send failed for %s - falling back to one "
                          "email per task", who.get("email"))
        if not ok:
            continue
        for r in rs:
            tk_db.patch("notify_outbox", {"id": "eq." + r["id"]},
                        dict(stamp, error=None if r in live else "rolled into digest"))
            handled.add(r["id"])
        LOG.info("digest: %d assignment(s) to %s in one email",
                 len(live), who.get("email"))
    return handled
