# -*- coding: utf-8 -*-
"""External data feeds - the learning data centre. Each ACTIVE feed in
taskapp.feeds is researched on its cadence (Anthropic web search, never
invented), shaped to the feed's declared columns, and APPENDED (with
captured_at) to datasights-lake/tables/feed_<slug>/ - so award rates, tax
rates, due dates and market indicators become ordinary lake tables that the
voice assistant, task agent and watch rules can query alongside internal data.
Daily 05:45 timer runs daily feeds every day, weekly feeds on Mondays."""
import datetime as dt
import json
import logging
import os

import tk_ai
import tk_db

LOG = logging.getLogger("tk_feeds")

MAX_ROWS_PER_RUN = 60


def _rows_schema(columns):
    props = {c["name"]: {"type": "string", "description": c.get("description", "")}
             for c in columns}
    return {"type": "object",
            "properties": {"rows": {"type": "array", "maxItems": MAX_ROWS_PER_RUN,
                                    "items": {"type": "object", "properties": props,
                                              "required": list(props)}},
                           "summary": {"type": "string",
                                       "description": "one line on what was captured/changed"}},
            "required": ["rows", "summary"]}


def _append_parquet(slug, columns, rows):
    import duckdb
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])
    cc = svc.get_container_client("datasights-lake")
    table = "feed_" + slug
    blob = "tables/%s/data.parquet" % table
    old = "/tmp/%s_old.parquet" % slug
    new = "/tmp/%s.parquet" % slug
    cols = [c["name"] for c in columns] + ["captured_at"]
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t (%s)" % ", ".join('"%s" VARCHAR' % c for c in cols))
    today = dt.date.today().isoformat()
    con.executemany("INSERT INTO t VALUES (%s)" % ",".join("?" * len(cols)),
                    [[str(r.get(c["name"], "")) for c in columns] + [today] for r in rows])
    has_old = False
    try:
        with open(old, "wb") as f:
            f.write(cc.download_blob(blob).readall())
        has_old = True
    except Exception:
        pass
    if has_old:
        # union with history; drop exact repeats of today's rows
        con.execute("""CREATE TABLE hist AS SELECT * FROM read_parquet('%s', union_by_name=true)
                       WHERE captured_at <> '%s'""" % (old, today))
        con.execute("INSERT INTO t SELECT * FROM hist")
    con.execute("COPY (SELECT * FROM t ORDER BY captured_at DESC) TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)" % new)
    total = con.execute("SELECT count(*) FROM t").fetchone()[0]
    con.close()
    with open(new, "rb") as f:
        cc.upload_blob(blob, f.read(), overwrite=True)
    cc.upload_blob("tables/%s/_meta.json" % table, json.dumps({
        "table": table, "rows": int(total),
        "source": "tk_feeds web research (learning data centre)",
        "columns": cols, "exported_at": dt.datetime.utcnow().isoformat()}),
        overwrite=True)
    # register in the lake catalog so agents see the table in the schema NOW
    # (rather than after the next nightly rebuild)
    try:
        cat = json.loads(cc.download_blob("catalog/catalog.json").readall())
        entries = [t for t in cat.get("tables", []) if (t.get("name") or t.get("table")) != table]
        entries.append({"name": table, "table": table, "rows": int(total),
                        "columns": cols,
                        "note": "external feed (learning data centre): refreshed by tk_feeds"})
        cat["tables"] = entries
        cc.upload_blob("catalog/catalog.json", json.dumps(cat), overwrite=True)
    except Exception:
        LOG.exception("catalog update failed (feed still queryable; schema listing lags)")
    return int(total)


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "already_available": {"type": "array", "maxItems": 6, "items": {
            "type": "object",
            "properties": {"table": {"type": "string",
                                     "description": "exact lake table or feed name from the provided lists"},
                           "why": {"type": "string",
                                   "description": "one line on how it covers the request"}},
            "required": ["table", "why"]}},
        "proposals": {"type": "array", "maxItems": 4, "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "kind": {"type": "string",
                         "enum": ["award", "tax_rates", "due_dates", "economy",
                                  "industry", "other"]},
                "cadence": {"type": "string", "enum": ["daily", "weekly"]},
                "description": {"type": "string",
                                "description": "one line: what lands in the lake and from where"},
                "research_prompt": {"type": "string",
                                    "description": "exact per-run research instructions: named "
                                                   "sources/sites and precisely which facts to capture"},
                "columns": {"type": "array", "minItems": 2, "maxItems": 6, "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string",
                                            "description": "lowercase snake_case"},
                                   "description": {"type": "string"}},
                    "required": ["name", "description"]}},
                "sources": {"type": "string",
                            "description": "the authoritative sites found, comma separated"}},
            "required": ["name", "kind", "cadence", "description",
                         "research_prompt", "columns", "sources"]}},
        "note": {"type": "string",
                 "description": "one line for the user: coverage found / caveats"}},
    "required": ["already_available", "proposals", "note"]}


def search(query):
    """'Search for data to load': check what the lake + feed registry already
    cover, then web-research authoritative recurring sources for the topic and
    return ready-to-add feed proposals (name/cadence/schema/research prompt)."""
    feeds = tk_db.get("feeds", {"select": "slug,name,status,description"})
    tables = []
    try:
        from azure.storage.blob import BlobServiceClient
        svc = BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])
        cc = svc.get_container_client("datasights-lake")
        cat = json.loads(cc.download_blob("catalog/catalog.json").readall())
        tables = [(t.get("name") or t.get("table") or "",
                   (t.get("note") or "")[:100]) for t in cat.get("tables", [])]
    except Exception:
        LOG.exception("catalog read failed (search continues without it)")
    research = tk_ai.searched_text(
        "You research public data sources for the finance learning data centre of "
        "Yo-Chi, an Australian frozen-yoghurt chain (74 venues, also UK/US/SG). "
        "Find AUTHORITATIVE, regularly-updated sources (government, regulator, "
        "statistics bureau, industry body) for the topic. For each source note the "
        "site, what facts it publishes, and how often it updates. Facts only.",
        "Topic to find recurring data sources for: %s" % query,
        max_searches=5, max_tokens=2500)
    existing_desc = (
        "EXISTING FEEDS (slug | name | status):\n"
        + "\n".join("%s | %s | %s" % (f["slug"], f["name"], f["status"]) for f in feeds)
        + "\n\nEXISTING LAKE TABLES (name | note):\n"
        + "\n".join("%s | %s" % t for t in tables))
    return tk_ai.structured(
        "Design data feeds for a finance learning data centre. From the research, "
        "propose up to 4 NEW recurring feeds for the user's topic - each with a "
        "sharp research_prompt naming its sources and the exact facts to capture "
        "per run, and a small snake_case column schema. Also list any EXISTING "
        "tables/feeds (exact names from the lists) that already cover the topic - "
        "do not propose a new feed that duplicates an existing one. If the topic "
        "has no reliable public source, say so in note and propose fewer/none.",
        "USER TOPIC: %s\n\n%s\n\nWEB RESEARCH:\n%s" % (query, existing_desc, research),
        "propose_feeds", SEARCH_SCHEMA, max_tokens=3000)


def run(force_slug=None):
    today = dt.date.today()
    feeds = tk_db.get("feeds", {"select": "*", "order": "slug"})
    results = []
    for f in feeds:
        due = (f["slug"] == force_slug) or (
            f["status"] == "active" and (
                f["cadence"] == "daily" or today.weekday() == 0 or not f.get("last_run_at")))
        if not due:
            continue
        try:
            columns = f["columns"] if isinstance(f["columns"], list) else json.loads(f["columns"])
            research = tk_ai.searched_text(
                "You are the data researcher for Yo-Chi's (AU frozen-yoghurt chain, 74 "
                "venues) finance learning data centre. Report ONLY facts found via web "
                "search, with numbers, dates and named sources. Today is %s." % today,
                f["research_prompt"], max_searches=6, max_tokens=3000)
            out = tk_ai.structured(
                "Convert the research into rows matching the schema exactly. Only rows "
                "with concrete sourced facts - omit anything vague. All values as strings.",
                research, "write_rows", _rows_schema(columns), max_tokens=3000)
            rows = out.get("rows", [])[:MAX_ROWS_PER_RUN]
            total = _append_parquet(f["slug"], columns, rows) if rows else 0
            tk_db.patch("feeds", {"id": "eq." + f["id"]}, {
                "last_run_at": dt.datetime.utcnow().isoformat() + "Z",
                "last_summary": (out.get("summary") or "")[:300]})
            results.append({"feed": f["slug"], "rows": len(rows), "table_total": total})
        except Exception as e:
            LOG.exception("feed %s failed", f["slug"])
            results.append({"feed": f["slug"], "error": str(e)[:200]})
    return {"ran": len(results), "results": results}
