# -*- coding: utf-8 -*-
"""Voice/chat assistant for the cockpit: answers a question with an agentic
tool loop - Claude can run live DuckDB queries over the data lake (run_sql),
and the prompt is pre-loaded with current TaskHub state (tasks, cash, health,
meetings, positions, signals) so most questions ground in both worlds.
Answers are written to be speakable: lead with the number, keep it tight."""
import datetime as dt
import json
import logging
import os
import re
import urllib.request

import tk_ai
import tk_db
import tk_skillbuilder
import lake_reader

LOG = logging.getLogger("tk_ask")

MAX_STEPS = 6

TOOLS = [{
    "name": "run_sql",
    "description": ("Run a read-only DuckDB query over the Yo-Chi data lake. "
                    "SELECT/WITH only; bare table names; CAST text dates AS DATE; "
                    "aggregate to a useful grain; <=100 rows."),
    "input_schema": {"type": "object",
                     "properties": {"sql": {"type": "string"}},
                     "required": ["sql"]},
}]

SYSTEM_TMPL = (
    "You are the voice assistant on Yo-Chi TaskHub's CFO cockpit, answering Peter "
    "(CFO of Yo-Chi, 74 frozen-yoghurt venues across Australia, ~$4.5m/week sales). "
    "Today is %s.\n\n"
    "ANSWER STYLE: spoken-friendly - lead with the answer and the key number, then at "
    "most 3 short supporting points. Plain text, no markdown tables unless listing >4 "
    "items (then a compact list). Under 150 words unless the question demands detail. "
    "Money is AUD net of GST. Fast Food Award items are 'areas to review', never "
    "'breaches'. If the data can't answer, say so plainly - never invent numbers.\n\n"
    "You can query the data lake with run_sql - use AT MOST 3 queries, then answer "
    "with what you have (state gaps briefly rather than exploring further). Known "
    "quirks: numeric fields may be VARCHAR (TRY_CAST AS DOUBLE); labour_cost in "
    "mart_venue_daily has pay-run artefacts >100%% of sales - exclude ratios above "
    "1.0; prefer mart_* tables. Also queryable: forecast_venue_daily (venue, d, "
    "forecast_sales) = the Venu Cast base case per venue for the trailing 14 days - "
    "use it for venue vs forecast/base-case questions. BUDGET is group-level only "
    "(daily $ in the context below, when relevant) - there is NO venue-level budget "
    "table; for venue-level comparisons use forecast_venue_daily or LY. When "
    "comparing actual vs forecast/budget/LY periods, MATCH THE SAME DAYS - actuals "
    "end at max(date) in the mart, so join day-by-day or bound both sides by the "
    "same date range, never a part-week vs a full week.\n\n"
    "CURRENT TASKHUB STATE (live, use before querying):\n%s\n\n"
    "LAKE SCHEMA:\n%s"
)


def _context():
    """Compact live snapshot of TaskHub for the system prompt."""
    parts = []
    try:
        tasks = tk_db.get("tasks", {"status": "neq.done",
                                    "select": "title,priority,due_date", "limit": "1000"})
        today = dt.date.today().isoformat()
        overdue = [t for t in tasks if t.get("due_date") and t["due_date"] < today]
        crit = [t for t in tasks if t.get("priority") == "critical"]
        parts.append("Tasks: %d open, %d overdue, %d critical. Top critical: %s" % (
            len(tasks), len(overdue), len(crit),
            "; ".join(t["title"][:60] for t in crit[:5]) or "none"))
    except Exception:
        LOG.exception("ctx tasks")
    try:
        cash = tk_db.get("cash_forecast", {"order": "generated_at.desc,week_start",
                                           "limit": "13",
                                           "select": "week_start,closing,assumptions"})
        if cash:
            a = next((r["assumptions"] for r in cash if r.get("assumptions")), {})
            trough = min(cash, key=lambda w: float(w["closing"]))
            parts.append("Cash 13wk: opening $%s (bank as at %s), trough $%s wk %s, "
                         "DSO %s d, DPO %s d." % (
                a.get("opening_cash"), a.get("bank_as_at"), trough["closing"],
                trough["week_start"], a.get("dso_days"), a.get("dpo_days")))
    except Exception:
        LOG.exception("ctx cash")
    try:
        vh = tk_db.get("venue_health", {"order": "week_start.desc,score",
                                        "limit": "8", "select": "venue,score,week_start"})
        if vh:
            parts.append("Venue health (wk %s) weakest: %s" % (
                vh[0]["week_start"],
                ", ".join("%s %s" % (r["venue"], r["score"]) for r in vh[:5])))
    except Exception:
        LOG.exception("ctx health")
    try:
        now = dt.datetime.utcnow().isoformat() + "Z"
        mts = tk_db.get("calendar_events", {"starts_at": "gte." + now,
                                            "order": "starts_at", "limit": "5",
                                            "select": "subject,starts_at"})
        if mts:
            parts.append("Next meetings: " + "; ".join(
                "%s (%s)" % (m["subject"], m["starts_at"][:16]) for m in mts))
    except Exception:
        LOG.exception("ctx meetings")
    try:
        sigs = tk_db.get("signals", {"order": "created_at.desc", "limit": "5",
                                     "select": "kind,headline"})
        if sigs:
            parts.append("Signals: " + "; ".join(
                "[%s] %s" % (s["kind"], s["headline"]) for s in sigs))
    except Exception:
        LOG.exception("ctx signals")
    try:
        pos = tk_db.get("positions", {"order": "created_at.desc", "limit": "3",
                                      "select": "project_id,content,created_at"})
        for p in pos[:2]:
            parts.append("Recent position (%s): %s" % (
                p["created_at"][:10], re.sub(r"\s+", " ", p["content"])[:400]))
    except Exception:
        LOG.exception("ctx positions")
    return "\n".join(parts) or "(context unavailable)"


def _run_sql(sql):
    referenced = set(re.findall(
        r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_][A-Za-z0-9_]*)\"?", sql, re.I))
    lake_reader.sync(extra_tables=referenced, log=LOG.info)
    result = lake_reader.query(sql, max_rows=100)
    return json.dumps(result, separators=(",", ":"))[:20000]


def answer(question, history=None):
    lake_reader.sync(log=LOG.info)
    try:
        import tk_watcher
        tk_watcher._sync_forecast(log=LOG.info)
    except Exception:
        LOG.exception("forecast sync for ask failed (non-fatal)")
    ctx = _context()
    try:
        import tk_budget
        bud = tk_budget.daily_group()
        month = dt.date.today().isoformat()[:7]
        mdays = sorted((d, v) for d, v in bud.items() if d.startswith(month))
        if mdays:
            ctx += "\nGroup daily budget this month: " + ", ".join(
                "%s $%dk" % (d[8:], round(v / 1000)) for d, v in mdays[:31])
    except Exception:
        LOG.exception("budget ctx failed (non-fatal)")
    system = SYSTEM_TMPL % (dt.date.today().isoformat(), ctx,
                            tk_skillbuilder._schema_text())
    messages = []
    for h in (history or [])[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:2000]})
    messages.append({"role": "user", "content": question[:2000]})

    queries_run = 0
    for step in range(MAX_STEPS):
        final_round = step == MAX_STEPS - 1
        if final_round:
            nudge = ("Answer now with the data you already have; note any gaps "
                     "in one short sentence.")
            last = messages[-1]
            if last["role"] == "user" and isinstance(last["content"], list):
                last["content"].append({"type": "text", "text": nudge})
            else:
                messages.append({"role": "user", "content": nudge})
        body = {
            "model": tk_ai._model(),
            "max_tokens": 1500,
            "system": system,
            "messages": messages,
        }
        if not final_round:
            body["tools"] = TOOLS
        data = tk_ai._call(body)
        content = data.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if not tool_uses or data.get("stop_reason") != "tool_use":
            text = "".join(b.get("text", "") for b in content
                           if b.get("type") == "text").strip()
            return {"answer": text or "I couldn't produce an answer - try rephrasing.",
                    "queries_run": queries_run}
        results = []
        for tu in tool_uses:
            queries_run += 1
            try:
                out = _run_sql(tu["input"].get("sql", ""))
            except Exception as e:
                out = "QUERY ERROR: %s" % str(e)[:300]
            results.append({"type": "tool_result", "tool_use_id": tu["id"],
                            "content": out})
        messages.append({"role": "user", "content": results})
    return {"answer": "That took too many data pulls to answer - try a narrower question.",
            "queries_run": queries_run}
