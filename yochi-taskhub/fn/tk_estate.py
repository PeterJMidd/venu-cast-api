# -*- coding: utf-8 -*-
"""Estate health: plumbing checks the data watch rules can't express in lake SQL
(blob feed freshness, stuck batch runs, agent error spikes, skills runner
heartbeat) -> tasks in the Data watch project. Deterministic - no AI calls.
Data-level staleness (mart dates, venue counts) lives in watcher_rules so it
stays visible/editable under Admin -> Watch rules."""
import datetime as dt
import logging
import os

import tk_calendar
import tk_db
import tk_templates

LOG = logging.getLogger("tk_estate")

DATA_WATCH_PROJECT = "aaaaaaaa-0000-0000-0000-000000000005"

CATALOG_MAX_AGE_H = 30   # nightly lake sync runs 04:45; >30h means it missed a run
BATCH_STUCK_H = 3        # host functionTimeout is 2h; older queued/running = dead
AGENT_ERROR_THRESHOLD = 3  # errors in the last 24h
SKILLS_HEARTBEAT_H = 48  # skills timer runs daily 06:30


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)


def _check_lake_sync():
    """The data-lake function writes catalog/catalog.json every night."""
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"], connection_timeout=60)
    props = svc.get_blob_client("datasights-lake", "catalog/catalog.json") \
               .get_blob_properties()
    age_h = (_utcnow() - props.last_modified).total_seconds() / 3600
    if age_h > CATALOG_MAX_AGE_H:
        return ("Data lake nightly sync stale",
                "The datasights-lake catalog was last written %.0f hours ago "
                "(threshold %dh). The yochi-data-lake function's nightly 04:45 run "
                "has likely failed - every downstream consumer (watch rules, AI "
                "skills, agent runs, dashboards) is reading stale data.\n"
                "First steps:\n- Check the yochi-data-lake function app logs\n"
                "- Trigger its backfill_now route once fixed" % (age_h, CATALOG_MAX_AGE_H))
    return None


def _check_stuck_batches():
    cutoff = (_utcnow() - dt.timedelta(hours=BATCH_STUCK_H)).isoformat()
    stuck = tk_db.get("batch_runs", {
        "status": "in.(queued,running)",
        "created_at": "lt." + cutoff,
        "select": "id,kind,status,created_at", "limit": "10"})
    if stuck:
        lines = "\n".join("- %s batch %s: %s since %s" % (
            b["kind"], b["id"][:8], b["status"], b["created_at"][:16]) for b in stuck)
        return ("Batch runs stuck in the queue",
                "%d batch run(s) have been queued/running for over %d hours (the "
                "worker times out at 2h, so these are dead):\n%s\n"
                "First steps:\n- Check yochi-taskhub-fn logs for the batch_worker\n"
                "- Mark the dead rows status='error' and re-queue if still needed"
                % (len(stuck), BATCH_STUCK_H, lines))
    return None


def _check_agent_errors():
    cutoff = (_utcnow() - dt.timedelta(hours=24)).isoformat()
    errs = tk_db.get("agent_runs", {
        "outcome": "eq.error",
        "created_at": "gte." + cutoff,
        "select": "task_title,error", "limit": "20"})
    if len(errs) >= AGENT_ERROR_THRESHOLD:
        lines = "\n".join("- %s: %s" % (e["task_title"][:60], (e.get("error") or "")[:100])
                          for e in errs[:8])
        return ("Agent error spike (last 24h)",
                "%d agent runs failed in the last 24 hours (threshold %d):\n%s\n"
                "First steps:\n- Check whether the errors share a cause (AI timeout, "
                "lake sync, storage)\n- Check yochi-taskhub-fn logs" %
                (len(errs), AGENT_ERROR_THRESHOLD, lines))
    return None


def _check_skills_heartbeat():
    skills = tk_db.get("ai_skills", {"active": "eq.true", "select": "last_run_at"})
    if not skills:
        return None
    runs = [s["last_run_at"] for s in skills if s.get("last_run_at")]
    latest = max(runs) if runs else None
    cutoff = (_utcnow() - dt.timedelta(hours=SKILLS_HEARTBEAT_H)).isoformat()
    if latest is None or latest < cutoff:
        return ("AI skills runner silent",
                "No active AI skill has run since %s (threshold %dh) - the 06:30 "
                "skills timer is likely failing.\nFirst steps:\n- Check "
                "yochi-taskhub-fn logs for skills_timer\n- Force one with "
                "run_skills?skill=<id>" % (latest or "(never)", SKILLS_HEARTBEAT_H))
    return None


def run():
    checks = [_check_lake_sync, _check_stuck_batches, _check_agent_errors,
              _check_skills_heartbeat]
    period_id, _ = tk_templates.ensure_period()
    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id", "limit": "1"})
    assignee = admin[0]["id"] if admin else None
    issues, created, errors = [], 0, []
    for check in checks:
        try:
            issue = check()
        except Exception as e:
            LOG.exception("estate check %s failed", check.__name__)
            errors.append({"check": check.__name__, "error": str(e)[:200]})
            continue
        if not issue:
            continue
        title = "[Estate] " + issue[0]
        issues.append(issue[0])
        # dedup: never a second task while one with the same title is still open
        if tk_db.get("tasks", {"title": "eq." + title, "status": "neq.done",
                               "select": "id", "limit": "1"}):
            continue
        due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=1))
        tk_db.insert("tasks", [{
            "project_id": DATA_WATCH_PROJECT,
            "period_id": period_id,
            "title": title,
            "description": issue[1],
            "priority": "high",
            "assignee_id": assignee,
            "due_date": due.isoformat(),
            "source": "watcher",
        }])
        created += 1
    return {"checks": len(checks), "issues": issues, "tasks_created": created,
            "errors": errors}
