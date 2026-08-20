# -*- coding: utf-8 -*-
"""TaskHub as a read-only consumer of the Yo-Chi Lake Brain.

The data lake verifies its own overnight refresh at 07:00 AEST and publishes
the verdict to blob. This reads that verdict at 07:15 and turns it into work:
a failed refresh becomes a task, so "the numbers looked odd this morning"
stops being something someone notices three weeks later.

READ-ONLY, deliberately. We read the published artifacts in `lake-brain` and
never write to that container; the brain owns its own state. We also do NOT
call /api/brain_now - that endpoint RECOMPUTES the brain, and calling it near
07:00 could interleave with the brain's own run. The blob is the cheap, safe
read, and it is already what the brain intends consumers to use.

Artifacts (stable, plain JSON, no auth beyond the storage key the app holds):
  integrity/latest.json  status GREEN|AMBER|RED, stages{last_run, age_h},
                         anomalies[], module_errors[], changed_count,
                         stale_7d_count
  knowledge.json         questions[] {id, question, status, asked_at, ...}
"""
import datetime as dt
import json
import logging
import os

import tk_db

LOG = logging.getLogger("tk_lakebrain")

AEST = dt.timezone(dt.timedelta(hours=10))
CONTAINER = "lake-brain"
INTEGRITY = "integrity/latest.json"
KNOWLEDGE = "knowledge.json"
DATA_WATCH = "aaaaaaaa-0000-0000-0000-000000000005"
PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"
ANSWER_URL = "https://yochi-data-lake.azurewebsites.net/api/brain_answer"
QUESTION_AGE_DAYS = 3


def _cc():
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"], read_timeout=120,
        connection_timeout=30)
    return svc.get_container_client(CONTAINER)


def _read(name):
    return json.loads(_cc().download_blob(name).readall())


def integrity():
    return _read(INTEGRITY)


def knowledge():
    k = _read(KNOWLEDGE)
    return (k.get("questions") if isinstance(k, dict) else k) or []


def _age_days(iso):
    """Days since an ISO timestamp. None when unparseable; negative ages are
    clamped to 0 - a clock ahead of ours must not read as ancient."""
    if not iso:
        return None
    try:
        t = dt.datetime.fromisoformat(str(iso).replace("Z", "").split("+")[0])
    except ValueError:
        return None
    return max(0.0, (dt.datetime.utcnow() - t).total_seconds() / 86400.0)


def _open_refs():
    """Idempotency keys already on the board and not yet done.

    Checked before every insert: the brain republishes the same verdict all
    day, and a daily timer that re-raises it would bury the board."""
    rows = tk_db.get("tasks", {"external_ref": "like.lakebrain-*",
                               "select": "external_ref,status", "limit": "500"})
    return {r["external_ref"] for r in rows if r.get("status") != "done"}


def _stage_lines(stages):
    out = []
    for name, s in sorted((stages or {}).items()):
        if not isinstance(s, dict):
            continue
        age = s.get("age_h")
        out.append("- %-28s last run %s%s" % (
            name, str(s.get("last_run") or "never")[:19],
            "  (%.1f h ago)" % age if isinstance(age, (int, float)) else ""))
    return "\n".join(out)


def _raise(ref, title, body, priority, today, existing, created):
    if ref in existing:
        return False
    try:
        tk_db.insert("tasks", [{
            "project_id": DATA_WATCH, "title": title[:200], "description": body,
            "priority": priority, "source": "watcher",
            "due_date": today.isoformat(), "assignee_id": PETER,
            "external_ref": ref,
        }], on_conflict="external_ref", ignore_duplicates=True)
        created.append({"ref": ref, "title": title[:120]})
        return True
    except Exception:
        LOG.exception("could not raise lake brain task %s", ref)
        return False


def check(raise_tasks=True, today=None):
    today = today or dt.datetime.now(AEST).date()
    created, skipped = [], []
    try:
        rep = integrity()
    except Exception as e:
        LOG.exception("lake brain integrity unreadable")
        return {"error": str(e)[:300], "created": [], "status": None}

    status = (rep.get("status") or "").upper()
    date = rep.get("date") or today.isoformat()
    anomalies = rep.get("anomalies") or []
    module_errors = rep.get("module_errors") or []
    existing = _open_refs() if raise_tasks else set()

    if status == "RED":
        ref = "lakebrain-red-%s" % date
        body = ("The lake brain's %s check says the overnight refresh FAILED.\n\n"
                "PIPELINE STAGES\n%s\n\nANOMALIES\n%s\n%s\n"
                "Until this clears, every report and email built on the lake is "
                "working from whatever survived - which may look complete and "
                "still be wrong.\n\nFull report: the lake-brain container, "
                "integrity/%s.json" % (
                    date, _stage_lines(rep.get("stages")),
                    "\n".join("- " + str(a) for a in anomalies) or "- (none listed)",
                    ("\nMODULE ERRORS\n" + "\n".join("- " + str(m) for m in module_errors))
                    if module_errors else "", date))
        if raise_tasks:
            _raise(ref, "Data lake refresh failed - investigate", body,
                   "critical", today, existing, created) or skipped.append(ref)

    elif status == "AMBER":
        # stale_7d alone is not worth a task: 72 tables are static reference
        # data that legitimately never change, and the brain itself has an
        # open question about excluding them
        real = [a for a in anomalies if "stale" not in str(a).lower()]
        stale_stages = [n for n, s in (rep.get("stages") or {}).items()
                        if isinstance(s, dict)
                        and isinstance(s.get("age_h"), (int, float))
                        and s["age_h"] > 26]
        if real or stale_stages or module_errors:
            ref = "lakebrain-amber-%s" % date
            body = ("The lake brain's %s check is AMBER - the refresh ran but "
                    "something is off.\n\nANOMALIES\n%s\n\nSTAGES OVER 26 HOURS "
                    "OLD\n%s\n%s\nPIPELINE STAGES\n%s" % (
                        date,
                        "\n".join("- " + str(a) for a in real) or "- (none)",
                        "\n".join("- " + n for n in stale_stages) or "- (none)",
                        ("\nMODULE ERRORS\n" + "\n".join("- " + str(m) for m in module_errors) + "\n")
                        if module_errors else "",
                        _stage_lines(rep.get("stages"))))
            if raise_tasks:
                _raise(ref, "Data lake refresh is amber - %d anomal%s"
                       % (len(real), "y" if len(real) == 1 else "ies"),
                       body, "medium", today, existing, created) or skipped.append(ref)
        else:
            skipped.append("amber-driven-only-by-stale-tables")

    # questions the brain has been waiting on
    questions = []
    try:
        questions = knowledge()
    except Exception:
        LOG.exception("lake brain knowledge unreadable (integrity still checked)")
    key = os.environ.get("LAKE_FN_KEY", "")
    waiting = []
    for q in questions:
        if (q.get("status") or "").lower() != "open":
            continue
        age = _age_days(q.get("asked_at"))
        if age is None or age < QUESTION_AGE_DAYS:
            continue
        waiting.append(dict(q, age_days=round(age, 1)))
        ref = "lakebrain-q-%s" % q.get("id")
        links = "\n".join(
            "%s: %s?code=%s&id=%s&a=%s" % (label, ANSWER_URL, key or "<key>",
                                           q.get("id"), a)
            for label, a in (("Yes", "yes"), ("No", "no"), ("Ignore", "ignore")))
        body = ("The lake brain has been waiting %.0f days for an answer.\n\n"
                "QUESTION\n%s\n\n%s%s%s\nANSWER BY CLICKING ONE\n%s\n\n"
                "Answering teaches the brain - it stops asking, and applies the "
                "learning to future checks." % (
                    age, q.get("question") or "(no text)",
                    ("CONTEXT\n%s\n\n" % q["context"]) if q.get("context") else "",
                    ("If yes: %s\n" % q["learn_yes"]) if q.get("learn_yes") else "",
                    ("If no: %s\n" % q["learn_no"]) if q.get("learn_no") else "",
                    links))
        if raise_tasks:
            _raise(ref, "Answer the Lake Brain's question: %s"
                   % (q.get("question") or "")[:110],
                   body, "medium", today, existing, created) or skipped.append(ref)

    out = {"status": status, "date": date, "anomalies": len(anomalies),
           "module_errors": len(module_errors),
           "changed_count": rep.get("changed_count"),
           "stale_7d_count": rep.get("stale_7d_count"),
           "stages": {n: (s or {}).get("age_h") for n, s in
                      (rep.get("stages") or {}).items()},
           "open_questions": len([q for q in questions
                                  if (q.get("status") or "").lower() == "open"]),
           "questions_overdue": len(waiting),
           "created": created, "skipped_existing": skipped}
    LOG.info("lake brain: %s, %d anomalies, %d task(s) raised",
             status or "?", len(anomalies), len(created))
    return out


def run():
    return check(raise_tasks=True)
