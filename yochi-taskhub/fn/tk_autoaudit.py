# -*- coding: utf-8 -*-
"""Monthly automation audit (1st, 08:15): the machine reports on itself so
alert fatigue and silent failures get caught. Deterministic - no AI."""
import datetime as dt
import logging

import tk_db
import tk_email

LOG = logging.getLogger("tk_autoaudit")


def run(email=True):
    cutoff = (dt.datetime.utcnow() - dt.timedelta(days=30)).isoformat()
    runs = tk_db.get("agent_runs", {"created_at": "gte." + cutoff,
                                    "select": "outcome", "limit": "1000"})
    ok = sum(1 for r in runs if r["outcome"] == "success")
    err = len(runs) - ok
    batches = tk_db.get("batch_runs", {"created_at": "gte." + cutoff,
                                       "select": "status,kind", "limit": "500"})
    watch = tk_db.get("tasks", {"source": "eq.watcher", "created_at": "gte." + cutoff,
                                "select": "status,title", "limit": "1000"})
    watch_open = sum(1 for t in watch if t["status"] != "done")
    prefix_counts = {}
    for t in watch:
        p = t["title"].split("]")[0] + "]" if t["title"].startswith("[") else "(rule)"
        prefix_counts[p] = prefix_counts.get(p, 0) + 1
    feeds = tk_db.get("feeds", {"select": "slug,status,cadence,last_run_at"})
    stale_feeds = [f["slug"] for f in feeds if f["status"] == "active" and (
        not f.get("last_run_at") or f["last_run_at"] < (
            dt.datetime.utcnow() - dt.timedelta(days=9)).isoformat())]
    summary = {
        "agent_runs": {"total": len(runs), "success": ok, "error": err},
        "batches": {"total": len(batches),
                    "errors": sum(1 for b in batches if b["status"] == "error")},
        "watcher_tasks_30d": {"created": len(watch), "still_open": watch_open,
                              "by_screen": prefix_counts},
        "active_feeds_stale": stale_feeds,
    }
    if email:
        rows = "".join("<li><b>%s</b>: %s created, still open: see list</li>" % (k, v)
                       for k, v in sorted(prefix_counts.items()))
        html = ("<p><b>Automation audit — last 30 days</b></p>"
                "<ul><li>Agent runs: %d (%d ok, %d errors)</li>"
                "<li>Batch runs: %d (%d errored)</li>"
                "<li>Automation-raised tasks: %d created, <b>%d still open</b> — "
                "open items rot the signal; close or re-tune the screens</li></ul>"
                "<p>Tasks by screen:</p><ul>%s</ul>"
                "<p>%s</p>"
                "<p style='color:#6b7280;font-size:11px'>If a screen generates items "
                "nobody actions, tune its threshold (Admin → Watch rules / tk_* "
                "constants) rather than ignoring it.</p>") % (
            len(runs), ok, err, len(batches),
            sum(1 for b in batches if b["status"] == "error"),
            len(watch), watch_open, rows,
            ("⚠ Active feeds not running: " + ", ".join(stale_feeds))
            if stale_feeds else "All active feeds ran within 9 days.")
        for a in tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                        "select": "email"}):
            tk_email.send(a["email"], "Automation audit — %d runs, %d screens, %d open flags"
                          % (len(runs), len(prefix_counts), watch_open), html)
    return summary
