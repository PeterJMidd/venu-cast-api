# -*- coding: utf-8 -*-
"""Daily per-user AI briefing: gather each active user's open work, ask Claude
for a tight morning summary, email it (gated by EMAIL_ENABLED / EMAIL_OVERRIDE_TO)."""
import datetime as dt
import json
import logging
import os

import tk_ai
import tk_calendar
import tk_db
import tk_email

LOG = logging.getLogger("tk_briefing")

SYSTEM = (
    "You write a short morning task briefing for a Yo-Chi finance team member. You get "
    "their open tasks in JSON buckets. Return ONLY an HTML fragment (no <html>/<body>, "
    "no markdown): <h3>Today</h3> 1-2 sentences naming the single most important thing; "
    "then <h3>Overdue</h3>, <h3>Due soon</h3>, <h3>Waiting on you to review</h3>, "
    "<h3>Blocked</h3>, <h3>You were mentioned</h3> as short <ul> lists (task title + "
    "due date + project; for mentions: who said what, briefly). Omit any "
    "empty section entirely. If they recently completed things, close with one positive "
    "sentence under <h3>Done this week</h3>. Be specific, warm but brief - this is a "
    "30-second read."
)


def _next_bds(n):
    d = dt.date.today()
    out = []
    while len(out) < n:
        d += dt.timedelta(days=1)
        if tk_calendar.is_business_day(d):
            out.append(d.isoformat())
    return out


def _mentions(uid, profiles_by_id):
    """Comments from the last 24h that @mention this user (by first or full
    name), excluding their own comments."""
    me = profiles_by_id.get(uid) or {}
    name = (me.get("full_name") or me.get("email", "").split("@")[0]).strip().lower()
    if not name:
        return []
    since = (dt.datetime.utcnow() - dt.timedelta(days=1)).isoformat() + "Z"
    comments = tk_db.get("comments", {
        "created_at": "gte." + since, "author_id": "neq." + uid,
        "select": "task_id,author_id,body,created_at", "limit": "200"})
    first = name.split(" ")[0]
    hits = [c for c in comments
            if "@" + name in c["body"].lower() or "@" + first in c["body"].lower()]
    if not hits:
        return []
    task_ids = list({c["task_id"] for c in hits})
    titles = {t["id"]: t["title"] for t in tk_db.get(
        "tasks", {"id": "in.(%s)" % ",".join(task_ids), "select": "id,title"})}
    return [{"task": titles.get(c["task_id"], ""),
             "by": (profiles_by_id.get(c["author_id"]) or {}).get("full_name", ""),
             "said": c["body"][:160]} for c in hits[:10]]


def _user_buckets(uid, profiles_by_id=None):
    tasks = tk_db.get("tasks", {
        "or": "(assignee_id.eq.%s,reviewer_id.eq.%s)" % (uid, uid),
        "select": "id,title,status,priority,due_date,project_id,assignee_id,reviewer_id,completed_at",
    })
    projects = {p["id"]: p["name"] for p in tk_db.get("projects", {"select": "id,name"})}
    today = dt.date.today().isoformat()
    soon = set(_next_bds(3))
    week_ago = (dt.date.today() - dt.timedelta(days=7)).isoformat()

    def slim(t):
        return {"title": t["title"], "due": t["due_date"], "priority": t["priority"],
                "project": projects.get(t["project_id"], "")}

    open_t = [t for t in tasks if t["status"] != "done"]
    return {
        "overdue": [slim(t) for t in open_t if t["due_date"] and t["due_date"] < today],
        "due_soon": [slim(t) for t in open_t if t["due_date"] in soon or t["due_date"] == today],
        "waiting_review": [slim(t) for t in open_t
                           if t["status"] == "waiting_review" and t["reviewer_id"] == uid],
        "blocked": [slim(t) for t in open_t if t["status"] == "blocked"],
        "done_this_week": [slim(t) for t in tasks
                           if t["status"] == "done" and (t["completed_at"] or "") >= week_ago],
        "mentions_last_24h": _mentions(uid, profiles_by_id or {}),
    }


def _wrap(name, ai_html):
    app_url = os.environ.get("APP_URL", "")
    return """<!DOCTYPE html><html><body style="margin:0;background:#eef1f5;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1f2937">
<div style="max-width:640px;margin:0 auto;padding:24px">
  <div style="background:#1d683d;color:#fff;border-radius:12px 12px 0 0;padding:18px 24px">
    <div style="font-size:18px;font-weight:750">TaskHub morning briefing</div>
    <div style="font-size:13px;color:#c7e5d2;margin-top:2px">%s &middot; %s</div>
  </div>
  <div style="background:#fff;padding:6px 24px 20px;border:1px solid #e3e7ee;border-top:0">%s</div>
  <div style="background:#fff;border:1px solid #e3e7ee;border-top:0;border-radius:0 0 12px 12px;padding:16px 24px;text-align:center">
    <a href="%s/my-tasks" style="display:inline-block;background:#23824a;color:#fff;text-decoration:none;font-weight:700;font-size:14px;padding:10px 20px;border-radius:9px">Open my tasks &rarr;</a>
  </div>
</div></body></html>""" % (name, dt.date.today().strftime("%A %d %B"), ai_html, app_url)


def run(force=False):
    if not force and not tk_calendar.is_business_day(dt.date.today()):
        return {"skipped": "not a business day"}
    profiles = tk_db.get("profiles", {"active": "eq.true", "select": "id,email,full_name"})
    profiles_by_id = {p["id"]: p for p in profiles}
    prefs = {p["user_id"]: p for p in tk_db.get("notification_prefs", {"select": "*"})}
    sent = skipped = 0
    for p in profiles:
        pref = prefs.get(p["id"])
        if pref and not pref.get("daily_briefing", True):
            skipped += 1
            continue
        buckets = _user_buckets(p["id"], profiles_by_id)
        if not any(buckets.values()):
            skipped += 1
            continue
        name = p.get("full_name") or p["email"]
        try:
            ai_html = tk_ai.text(SYSTEM, "User: %s\nToday: %s\n\n%s" % (
                name, dt.date.today().isoformat(),
                json.dumps(buckets, separators=(",", ":"))), max_tokens=1200)
            if tk_email.send(p["email"], "Your TaskHub briefing - %s"
                             % dt.date.today().strftime("%d %b"), _wrap(name, ai_html)):
                sent += 1
        except Exception:
            LOG.exception("briefing failed for %s", p["email"])
    return {"sent": sent, "skipped": skipped, "email_enabled": tk_email.enabled()}
