# -*- coding: utf-8 -*-
"""Agent playbook: a growing, lake-resident record of HOW to do each kind of
task well.

The agent already learns from precedents in the database (recent successful
plans and the user's refinements). That is per-run memory. This is the
distilled layer on top: after every successful run, the approach that worked is
condensed into a durable entry - the method, the queries that proved useful,
the pitfalls, and what to do better next time - and written to the data lake as
`agent_playbook`. Before planning, the agent recalls entries for the same KIND
of task and is told to build on them rather than start from scratch.

Living in the lake (not just Postgres) means the playbook is queryable by the
voice assistant, the ask-the-lake page and any skill, alongside every other
table."""
import datetime as dt
import json
import logging
import os
import re

import tk_ai

LOG = logging.getLogger("tk_playbook")

TABLE = "agent_playbook"
CONTAINER = "datasights-lake"
COLUMNS = ["approach_key", "task_title", "project", "method", "queries_that_worked",
           "pitfalls", "improve_next_time", "outcome", "version",
           "changed_this_run", "run_at"]
STOP = {"the", "and", "for", "with", "from", "into", "this", "that", "review",
        "check", "confirm", "update", "prepare", "monthly", "weekly", "daily"}

DISTIL_SCHEMA = {
    "type": "object",
    "properties": {
        "approach_key": {"type": "string",
                         "description": "2-4 lowercase words naming the KIND of "
                                        "task, e.g. 'payroll tax reconciliation' "
                                        "or 'venue sales completeness'"},
        "method": {"type": "string",
                   "description": "the approach that worked, as instructions to "
                                  "a future agent; 3-6 short sentences"},
        "queries_that_worked": {"type": "string",
                                "description": "which lake tables/joins mattered "
                                               "and why; name the tables"},
        "pitfalls": {"type": "string",
                     "description": "traps hit or narrowly avoided (naming "
                                    "mismatches, date/type quirks, empty results)"},
        "improve_next_time": {"type": "string",
                              "description": "the single most useful improvement "
                                             "for the next run of this kind"}},
    "required": ["approach_key", "method", "queries_that_worked", "pitfalls",
                 "improve_next_time"]}
DISTIL_SCHEMA["properties"].update({
    "changed_this_run": {"type": "string",
                         "description": "what this run changed about the "
                                        "approach, one or two sentences"}})

DISTIL_SYSTEM = (
    "You maintain the standing RECOMMENDED APPROACH for one KIND of finance "
    "task. You get the current recommended approach (may be empty) and what "
    "just happened on a new run. Return the UPDATED approach - the whole "
    "thing, ready to replace what was there.\n\n"
    "Write instructions to the next agent, not a summary of findings: the "
    "numbers will differ next time, the method should not have to be "
    "rediscovered. Be concrete - name lake tables, columns, filters and "
    "gotchas. Fold the new run in: keep what still holds, correct anything the "
    "new run showed to be wrong, and add what was learned. If something "
    "produced an empty or misleading result, say what to do instead. Never "
    "invent a table or column you were not shown. Also state briefly, in "
    "changed_this_run, what this run actually changed about the approach (or "
    "'first version' / 'no change - confirmed the existing approach').")

DISTIL_SCHEMA_EXTRA = {
    "changed_this_run": {"type": "string",
                         "description": "what this run changed about the "
                                        "approach, one or two sentences"}}


def key_for(title):
    """Normalise a task title to the KIND of task it is."""
    t = re.sub(r"\[[^\]]*\]", " ", title or "")          # drop [Revenue] style tags
    t = re.sub(r"[^A-Za-z ]+", " ", t)                    # drop dates/numbers
    words = [w.lower() for w in t.split()
             if len(w) > 3 and w.lower() not in STOP]
    return " ".join(words[:4])


def _container():
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"])
    return svc.get_container_client(CONTAINER)


def _blob():
    return "tables/%s/data.parquet" % TABLE


def _append(rows):
    """Purely additive: read what is there, add the new rows, write back. The
    playbook is history - nothing is ever dropped."""
    import duckdb
    cc = _container()
    old = "/tmp/%s_old.parquet" % TABLE
    new = "/tmp/%s.parquet" % TABLE
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET enable_progress_bar=false")
    except Exception:
        pass
    con.execute("CREATE TABLE t (%s)" % ", ".join('"%s" VARCHAR' % c for c in COLUMNS))
    con.executemany("INSERT INTO t VALUES (%s)" % ",".join("?" * len(COLUMNS)),
                    [[str(r.get(c, "") or "") for c in COLUMNS] for r in rows])
    has_old = False
    try:
        with open(old, "wb") as f:
            f.write(cc.download_blob(_blob()).readall())
        has_old = True
    except Exception:
        pass
    if has_old:
        con.execute("CREATE TABLE hist AS SELECT * FROM read_parquet('%s', "
                    "union_by_name=true)" % old)
        # tolerate schema evolution: older files predate newer columns
        have = {r[0] for r in con.execute("DESCRIBE hist").fetchall()}
        sel = ", ".join(('"%s"' % c) if c in have else ("'' AS \"%s\"" % c)
                        for c in COLUMNS)
        con.execute("INSERT INTO t SELECT %s FROM hist" % sel)
    con.execute("COPY (SELECT * FROM t ORDER BY run_at DESC) TO '%s' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)" % new)
    total = con.execute("SELECT count(*) FROM t").fetchone()[0]
    con.close()
    with open(new, "rb") as f:
        cc.upload_blob(_blob(), f.read(), overwrite=True)
    cc.upload_blob("tables/%s/_meta.json" % TABLE, json.dumps({
        "table": TABLE, "rows": int(total), "columns": COLUMNS,
        "source": "tk_playbook - distilled agent approaches, appended per run",
        "exported_at": dt.datetime.utcnow().isoformat()}), overwrite=True)
    try:   # make it visible to agents/voice immediately, not at the next rebuild
        cat = json.loads(cc.download_blob("catalog/catalog.json").readall())
        entries = [t for t in cat.get("tables", [])
                   if (t.get("name") or t.get("table")) != TABLE]
        entries.append({"name": TABLE, "table": TABLE, "rows": int(total),
                        "columns": COLUMNS,
                        "note": "how past agent runs of each kind of task were done"})
        cat["tables"] = entries
        cc.upload_blob("catalog/catalog.json", json.dumps(cat), overwrite=True)
    except Exception:
        LOG.exception("playbook catalog update failed (table still queryable)")
    return int(total)


def recall(title, limit=3):
    """Playbook versions for this KIND of task, newest first. The first row is
    the CURRENT recommended approach; the rest are how it got there."""
    import lake_reader
    key = key_for(title)
    if not key:
        return []
    words = [w for w in key.split() if len(w) > 3][:4]
    if not words:
        return []
    where = " OR ".join("lower(approach_key) LIKE '%%%s%%'" % w.replace("'", "")
                        for w in words)
    sql = ("SELECT approach_key, task_title, project, method, "
           "queries_that_worked, pitfalls, improve_next_time, outcome, "
           "version, changed_this_run, run_at FROM %s WHERE %s "
           "ORDER BY run_at DESC LIMIT %d" % (TABLE, where, int(limit)))
    try:
        out = lake_reader.query(sql, max_rows=limit)
        return [dict(zip(out["columns"], r)) for r in out["rows"]]
    except Exception as e:
        msg = str(e)
        if "does not exist" in msg or "not found" in msg.lower():
            return []          # no playbook yet - first run of anything
        LOG.exception("playbook recall failed")
        return []


def current(title):
    """The standing recommended approach for this kind of task, or None."""
    rows = recall(title, limit=1)
    return rows[0] if rows else None


def prompt_block(title, limit=3):
    """The recalled playbook, formatted for the planning prompt."""
    rows = recall(title, limit=limit)
    if not rows:
        return ""
    bits = ["\n\nPLAYBOOK - how tasks of this kind were done before. Build on "
            "this and improve it; do not rediscover it:"]
    for r in rows:
        bits.append(
            "- [%s | %s]\n  method: %s\n  data: %s\n  pitfalls: %s\n  "
            "do better: %s" % (
                r.get("approach_key", ""), (r.get("run_at") or "")[:10],
                r.get("method", ""), r.get("queries_that_worked", ""),
                r.get("pitfalls", ""), r.get("improve_next_time", "")))
    return "\n".join(bits)[:6000]


def record(task_title, project, plan, summary=None, outcome="success"):
    """Fold one run into the standing recommended approach and store the new
    version. Called on PROPOSE (so an approach is captured even if you never
    execute) and again on a successful EXECUTE (so results sharpen it).

    Never raises: losing a playbook version must not fail the run that made it."""
    try:
        prior = current(task_title)
        queries = plan.get("data_queries") or []
        parts = ["TASK: %s" % task_title, "PROJECT: %s" % (project or "")]
        if prior:
            parts.append(
                "CURRENT RECOMMENDED APPROACH (version %s, %s):\n%s\n\nData that "
                "worked: %s\nKnown pitfalls: %s\nWanted improvement: %s" % (
                    prior.get("version") or "1", (prior.get("run_at") or "")[:10],
                    prior.get("method", ""), prior.get("queries_that_worked", ""),
                    prior.get("pitfalls", ""), prior.get("improve_next_time", "")))
        else:
            parts.append("CURRENT RECOMMENDED APPROACH: (none yet - this is the "
                         "first version)")
        parts.append("THIS RUN (%s)\nApproach planned:\n%s\n\nQueries:\n%s" % (
            outcome, plan.get("approach") or "", json.dumps(queries)[:4000]))
        if summary:
            parts.append("What the run produced:\n" + summary[:3000])
        else:
            parts.append("This run was planned but not executed, so there are no "
                         "results yet - capture the intended approach.")
        d = tk_ai.structured(DISTIL_SYSTEM, "\n\n".join(parts),
                             "distil_playbook", DISTIL_SCHEMA, max_tokens=1400)
        try:
            version = int(float(prior.get("version") or 0)) + 1 if prior else 1
        except (TypeError, ValueError):
            version = 1
        row = {
            "approach_key": (d.get("approach_key") or key_for(task_title))[:80],
            "task_title": (task_title or "")[:200],
            "project": (project or "")[:120],
            "method": d.get("method", ""),
            "queries_that_worked": d.get("queries_that_worked", ""),
            "pitfalls": d.get("pitfalls", ""),
            "improve_next_time": d.get("improve_next_time", ""),
            "outcome": outcome,
            "version": str(version),
            "changed_this_run": d.get("changed_this_run", ""),
            "run_at": dt.datetime.utcnow().isoformat(),
        }
        total = _append([row])
        LOG.info("playbook v%d recorded (%s), %d rows total",
                 version, row["approach_key"], total)
        return {"recorded": True, "approach_key": row["approach_key"],
                "version": version, "entries": total,
                "changed_this_run": row["changed_this_run"]}
    except Exception as e:
        LOG.exception("playbook record failed")
        return {"recorded": False, "error": str(e)[:200]}
