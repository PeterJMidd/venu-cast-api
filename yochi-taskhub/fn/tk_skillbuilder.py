# -*- coding: utf-8 -*-
"""AI skill builder: turn a plain-English description of a recurring review into
a complete, validated skill (brief + DuckDB data queries + schedule). Every
generated query is executed against the lake before the draft is returned; on
error the AI gets one repair round."""
import json
import logging
import os
import re

import tk_ai
import lake_reader

LOG = logging.getLogger("tk_skillbuilder")

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "short skill name"},
        "prompt": {"type": "string", "description": "the review brief the reviewing AI will follow"},
        "cadence": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
        "weekday": {"type": ["integer", "null"], "description": "0=Monday..6=Sunday, weekly only"},
        "email_review": {"type": "boolean"},
        "data_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "sql": {"type": "string", "description": "DuckDB SELECT/WITH over the lake views"},
                },
                "required": ["label", "sql"],
            },
            "minItems": 1,
        },
    },
    "required": ["name", "prompt", "cadence", "data_queries"],
}

SYSTEM = (
    "You design a recurring data-review 'skill' for Yo-Chi's finance TaskHub from a plain-"
    "English description. You get the data lake schema (DuckDB views over parquet). Produce:\n"
    "- name: short.\n"
    "- prompt: the brief a reviewing AI will follow each run (what to analyse, what to flag, "
    "how to structure the review). Include domain caveats the description implies. NEVER use "
    "the word 'breach' for Fast Food Award items - always 'areas to review'.\n"
    "- cadence/weekday from the description (weekday 0=Monday; e.g. 'each Tuesday' -> weekly, 1).\n"
    "- email_review true only if the description asks for an email.\n"
    "- data_queries: 1-3 DuckDB queries giving the reviewer the evidence it needs.\n"
    "SQL rules (STRICT): SELECT/WITH only. Views are queried by bare name (never quote table "
    "names with single quotes - single-quoted FROM is blocked). Many date columns are VARCHAR: "
    "always CAST(col AS DATE) before date maths. current_date works. Aggregate to useful grain "
    "(per venue/supplier/week) rather than dumping raw rows; always ORDER BY the interesting "
    "metric; keep result sets under ~200 rows (the runner truncates at 200). Known data quirk: "
    "labour_cost in mart_venue_daily has pay-run artefacts >100%% of sales - exclude ratios "
    "over 1.0. Prefer mart_* views where they cover the need. Xero view quirks "
    "(learned the hard way): Xero*View rows repeat PER LINE ITEM - an invoice "
    "appears ~6 times, so ALWAYS dedup (GROUP BY invoiceid with MAX(amountdue), "
    "or COUNT(DISTINCT invoiceid)) before counting or summing, or totals come "
    "out ~6x reality. The status column is a NUMERIC code, never text like "
    "'AUTHORISED' - do not filter on status names; for outstanding/overdue "
    "bills filter TRY_CAST(amountdue AS DOUBLE) > 0 instead, which is what "
    "actually means money is still owed. XeroBillsView is ALREADY bills-only "
    "(invoicetypedescribed = 'AccountsPayable', never 'ACCPAY') - no type "
    "filter needed there. Amount columns may be VARCHAR: "
    "TRY_CAST(... AS DOUBLE) before maths."
)

REPAIR_SYSTEM = (
    "You wrote DuckDB queries for a data-review skill; some failed validation. Fix ONLY the "
    "failing queries (same labels), keeping the working ones unchanged. Same SQL rules: "
    "SELECT/WITH only, no single-quoted FROM, CAST varchar dates, <=200 rows."
)


def _schema_text():
    lake_reader.sync(log=LOG.info)
    cat_path = os.path.join(lake_reader.CACHE_DIR, "catalog", "catalog.json")
    try:
        catalog = json.load(open(cat_path))
    except Exception:
        catalog = {"tables": []}
    lines = []
    for t in catalog.get("tables", []):
        name = t.get("view") or t.get("name")
        if not name:
            continue
        cols = ", ".join((t.get("columns") or [])[:35])
        extra = ""
        if t.get("date_col"):
            extra = " [date_col %s %s..%s]" % (t["date_col"],
                                               str(t.get("date_min", ""))[:10],
                                               str(t.get("date_max", ""))[:10])
        lines.append("- %s (%s rows)%s: %s" % (name, t.get("rows", "?"), extra, cols))
    metrics = ""
    mpath = os.path.join(lake_reader.CACHE_DIR, "catalog", "metrics.md")
    if os.path.exists(mpath):
        metrics = "\n\nSanctioned metric definitions:\n" + open(mpath, encoding="utf-8").read()[:3000]
    return "\n".join(lines) + metrics


def _validate(queries):
    """Run each query (sample). Returns list of {label, ok, rows|error}."""
    referenced = set()
    for q in queries:
        referenced.update(re.findall(r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_][A-Za-z0-9_]*)\"?",
                                     q.get("sql", ""), re.I))
    lake_reader.sync(extra_tables=referenced, log=LOG.info)
    results = []
    for q in queries:
        try:
            r = lake_reader.query(q["sql"], max_rows=5)
            results.append({"label": q.get("label", "data"), "ok": True,
                            "sample_rows": len(r["rows"]), "columns": r["columns"]})
        except Exception as e:
            results.append({"label": q.get("label", "data"), "ok": False,
                            "error": str(e)[:300]})
    return results


def build(description):
    schema = _schema_text()
    user = "Lake schema:\n%s\n\nSkill description from the user:\n%s" % (schema, description)
    draft = tk_ai.structured(SYSTEM, user, "build_skill", TOOL_SCHEMA, max_tokens=3000)
    validation = _validate(draft["data_queries"])

    if any(not v["ok"] for v in validation):
        repair_user = (
            "Lake schema:\n%s\n\nSkill description:\n%s\n\nYour queries:\n%s\n\n"
            "Validation results:\n%s\n\nReturn the full corrected skill."
        ) % (schema, description,
             json.dumps(draft["data_queries"], separators=(",", ":")),
             json.dumps(validation, separators=(",", ":")))
        draft = tk_ai.structured(REPAIR_SYSTEM, repair_user, "build_skill", TOOL_SCHEMA,
                                 max_tokens=3000)
        validation = _validate(draft["data_queries"])

    draft["validation"] = validation
    draft["all_valid"] = all(v["ok"] for v in validation)
    return draft
