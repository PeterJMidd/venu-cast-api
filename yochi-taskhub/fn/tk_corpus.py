# -*- coding: utf-8 -*-
"""Health of the document corpus - the mirror, the index, and their coverage.

Written because the SharePoint mirror died on 17 Jul 2026 and nobody noticed
for 34 days: the app had no Application Insights, and nothing downstream ever
asserted "the corpus should be fresh". Every answer the assistant gave in that
window was drawn from a month-stale corpus and looked perfectly confident.

So the corpus now has to prove it is alive every morning, on the same footing
as the data contracts: stale mirror, stalled indexing or collapsing coverage
raise a task in TaskHub rather than waiting to be stumbled upon."""
import datetime as dt
import gzip
import json
import logging
import os

import tk_db

LOG = logging.getLogger("tk_corpus")

AEST = dt.timezone(dt.timedelta(hours=10))
CONTAINER = "docs-lake"
MIRROR_STATE = "sharepoint/_state.json"
INDEX_STATE = "index/_state.json.gz"
DATA_WATCH = "aaaaaaaa-0000-0000-0000-000000000005"
PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"

MIRROR_STALE_DAYS = 3        # nightly job; 3 days is a real fault, not a blip
INDEX_STALE_DAYS = 3
MIN_COVERAGE_GAIN = 50       # documents the indexer should add on a normal night


def _cc():
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"], read_timeout=300, connection_timeout=60)
    return svc.get_container_client(CONTAINER)


def _age_days(iso, now=None):
    if not iso:
        return None
    try:
        t = dt.datetime.fromisoformat(str(iso).replace("Z", "").split("+")[0])
    except ValueError:
        return None
    return round(((now or dt.datetime.utcnow()) - t).total_seconds() / 86400.0, 1)


def snapshot():
    """What the corpus looks like right now - counts, ages, coverage."""
    cc = _cc()
    out = {"checked_at": dt.datetime.utcnow().isoformat() + "Z"}
    try:
        mirror = json.loads(cc.download_blob(MIRROR_STATE).readall())
        out["mirror_phase"] = mirror.get("phase")
        out["mirror_files"] = int(mirror.get("files_done") or 0)
        out["mirror_updated"] = mirror.get("updated_at")
        out["mirror_age_days"] = _age_days(mirror.get("updated_at"))
    except Exception as e:
        out["mirror_error"] = str(e)[:200]
    try:
        raw = cc.download_blob(INDEX_STATE).readall()
        idx = json.loads(gzip.decompress(raw).decode())
        out["indexed_docs"] = len(idx.get("done") or {})
        out["index_parts"] = idx.get("next_part")
    except Exception as e:
        out["index_error"] = str(e)[:200]
    try:
        props = cc.get_blob_client(INDEX_STATE).get_blob_properties()
        lm = props.last_modified
        out["index_updated"] = lm.isoformat()
        out["index_age_days"] = _age_days(lm.replace(tzinfo=None).isoformat())
    except Exception:
        pass
    mirrored = out.get("mirror_files") or 0
    indexed = out.get("indexed_docs") or 0
    out["coverage_pct"] = round(100.0 * indexed / mirrored, 1) if mirrored else None
    return out


def check(raise_task=True, today=None):
    """Compare against the previous snapshot and complain about what stopped."""
    today = today or dt.datetime.now(AEST).date()
    now = snapshot()
    prev = (tk_db.get("corpus_health", {"select": "*", "order": "checked_at.desc",
                                        "limit": "1"}) or [None])[0]
    problems = []

    if now.get("mirror_error"):
        problems.append("cannot read the mirror state: %s" % now["mirror_error"])
    elif (now.get("mirror_age_days") or 0) > MIRROR_STALE_DAYS:
        problems.append(
            "the SharePoint mirror has not run for %.1f days (last %s) - nothing "
            "created or changed in SharePoint since then is searchable"
            % (now["mirror_age_days"], str(now.get("mirror_updated"))[:16]))

    if now.get("index_error"):
        problems.append("cannot read the index state: %s" % now["index_error"])
    elif (now.get("index_age_days") or 0) > INDEX_STALE_DAYS:
        problems.append("the document indexer has not run for %.1f days"
                        % now["index_age_days"])

    if prev:
        gained = (now.get("indexed_docs") or 0) - (prev.get("indexed_docs") or 0)
        now["indexed_gain"] = gained
        if gained < 0:
            problems.append("the index SHRANK by %d documents - parts may have "
                            "been lost" % abs(gained))
        elif gained == 0 and (now.get("coverage_pct") or 100) < 95:
            problems.append("indexing added nothing since the last check while "
                            "coverage is only %.1f%% - the backlog is not moving"
                            % (now.get("coverage_pct") or 0))
        lost = (prev.get("mirror_files") or 0) - (now.get("mirror_files") or 0)
        if lost > 0:
            problems.append("the mirror reports %d FEWER files than last check" % lost)

    now["problems"] = problems
    now["status"] = "ok" if not problems else "breach"
    try:
        tk_db.insert("corpus_health", [{k: now.get(k) for k in (
            "mirror_phase", "mirror_files", "mirror_updated", "mirror_age_days",
            "indexed_docs", "index_parts", "index_updated", "index_age_days",
            "coverage_pct", "indexed_gain", "status")} | {"problems": problems}])
    except Exception:
        LOG.exception("could not record the corpus health snapshot")

    task_id = None
    if raise_task and problems:
        body = "\n".join("- " + p for p in problems)
        try:
            rows = tk_db.insert("tasks", [{
                "project_id": DATA_WATCH,
                "title": "[Data] The document corpus is not keeping up",
                "description": (
                    "The daily corpus check found problems:\n\n%s\n\n"
                    "State: mirror %s (%s files, %s days old), index %s documents "
                    "(%s%% of the mirror).\n\nUntil this clears, document search "
                    "and any answer drawn from agreements or board papers is "
                    "working from stale material." % (
                        body, now.get("mirror_phase"), now.get("mirror_files"),
                        now.get("mirror_age_days"), now.get("indexed_docs"),
                        now.get("coverage_pct"))),
                "priority": "high", "source": "watcher",
                "due_date": today.isoformat(), "assignee_id": PETER,
                "external_ref": "corpus:" + today.isoformat(),
            }], on_conflict="external_ref", ignore_duplicates=True, returning=True)
            task_id = rows[0]["id"] if rows else None
        except Exception:
            LOG.exception("could not raise the corpus health task")
    now["task_id"] = task_id
    LOG.info("corpus health: %s (%s docs indexed, %s%% coverage, %d problem(s))",
             now["status"], now.get("indexed_docs"), now.get("coverage_pct"),
             len(problems))
    return now


def run():
    """The daily corpus pass: prove it is alive, learn from how it was used,
    and surface what it could not answer."""
    res = check(raise_task=True)
    try:
        import tk_docs
        res["learned"] = tk_docs.learn()
        res["unanswered"] = tk_docs.unanswered(days=14, limit=10)
    except Exception:
        LOG.exception("search learning pass failed (health check still stands)")
    # Questions the corpus repeatedly fails to answer are a work list, not
    # noise: each one names something we have not indexed, or do not hold.
    repeated = [u for u in (res.get("unanswered") or []) if u["asked"] >= 2]
    if repeated:
        body = "\n".join("- asked %dx: %s" % (u["asked"], u["question"])
                         for u in repeated[:10])
        try:
            tk_db.insert("tasks", [{
                "project_id": DATA_WATCH,
                "title": "[Data] %d question%s our documents keep failing to answer"
                         % (len(repeated), "" if len(repeated) == 1 else "s"),
                "description": (
                    "These were asked more than once in the last fortnight and "
                    "nothing in the corpus answered them:\n\n%s\n\nEach one "
                    "means the document is not indexed yet, or we do not hold "
                    "it. Worth checking before someone concludes the answer "
                    "does not exist." % body),
                "priority": "medium", "source": "watcher",
                "due_date": dt.datetime.now(AEST).date().isoformat(),
                "assignee_id": PETER,
                "external_ref": "corpusgap:" + dt.datetime.now(AEST).date().isoformat(),
            }], on_conflict="external_ref", ignore_duplicates=True)
        except Exception:
            LOG.exception("could not raise the unanswered-questions task")
    return res
