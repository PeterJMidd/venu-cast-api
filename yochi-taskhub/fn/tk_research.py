# -*- coding: utf-8 -*-
"""Ask-a-question research on a task: live web search, grounded in what the
task is actually about, with the answer written back onto the task.

Two engines, because the value of a second one is a DIFFERENT retrieval stack,
not a different model:
  claude     - Anthropic web search (default; fast, good at dense legal text)
  perplexity - Sonar (different index, recency filter, strong on "what changed")
  compare    - run both, then reconcile: what they AGREE on is corroborated,
               where they DIFFER is the signal to dig or ask an adviser.

Deliberately separate from the Task agent, which analyses OUR data in the lake.
When a question really needs Yo-Chi's own numbers the answer says so and points
there rather than guessing. The answer is posted as a comment, so it is
durable, searchable and notifies whoever else is on the task."""
import datetime as dt
import logging

import tk_ai
import tk_db
import tk_knowledge
import tk_perplexity

LOG = logging.getLogger("tk_research")

DEPTHS = {
    "quick":    {"searches": 3,  "tokens": 1500},
    "standard": {"searches": 6,  "tokens": 3000},
    "deep":     {"searches": 12, "tokens": 6000},
}
ENGINES = ("claude", "perplexity", "compare", "lake")
RECENCY = ("day", "week", "month", "year")

SYSTEM = """You are the research analyst for the CFO of Yo-Chi, an Australian
frozen-yoghurt chain (74 AU venues; also UK, USA (FL/TX), Singapore/SEA).
Answer the question using LIVE WEB SEARCH and report only what you find, with
the source named inline and every URL listed at the end under 'Sources'.
Australian rules, dates and dollars unless the question is about another market.

Rules:
- Lead with the direct answer in 1-2 sentences, then the detail.
- Be specific: rates, thresholds, dates, section numbers, dollar figures.
- Date everything, and say when a rule changes or has recently changed.
- If sources disagree or the position is unsettled, say so plainly.
- If you cannot verify something, say you could not verify it. Never invent a
  figure, a citation, or a URL.
- If the question actually needs Yo-Chi's own data (our sales, our GL, our
  contracts), answer what you can from the web and then say which internal
  source would settle it - suggest the Task agent for that.
- This is research to brief a decision, not tax or legal advice; say so when
  the question is one a licensed adviser should sign off.
- Fast Food Award items are never 'breaches'; call them areas to review.
- Plain text. No markdown headers or asterisks - it is posted as a task
  comment. Short paragraphs and '- ' bullets only."""

RECONCILE = """Two INDEPENDENT research engines answered the same question
using different search indexes. Write a short reconciliation in plain text, no
markdown headers, in exactly these three labelled parts:

AGREED - what both support. This is corroborated: state it as fact, with the
figure and date.
DIFFERS - any point where they conflict, or where one gives a material detail
the other misses. Say which engine said what.
VERIFY - what to confirm against a primary source or with an adviser before
relying on it.

Use ONLY what is in the two answers. Never add a fact, figure or citation of
your own. If they agree on everything material, say so in one line and keep
DIFFERS short."""


def _context(task_id):
    rows = tk_db.get("tasks", {
        "id": "eq." + task_id,
        "select": "id,title,description,due_date,priority,project_id"})
    if not rows:
        raise ValueError("task not found")
    t = rows[0]
    proj = tk_db.get("projects", {"id": "eq." + (t.get("project_id") or ""),
                                  "select": "name"})
    bits = ["TASK: " + (t.get("title") or "")]
    if proj:
        bits.append("PROJECT: " + proj[0]["name"])
    if t.get("due_date"):
        bits.append("DUE: " + t["due_date"])
    desc = (t.get("description") or "").strip()
    if desc:
        bits.append("TASK DETAIL:\n" + desc[:1500])
    recent = tk_db.get("comments", {
        "task_id": "eq." + task_id, "select": "body",
        "order": "created_at.desc", "limit": "3"})
    notes = [c["body"][:400] for c in recent
             if not c["body"].startswith("Research")]
    if notes:
        bits.append("RECENT NOTES ON THE TASK:\n" + "\n---\n".join(notes))
    known = tk_knowledge.summary_text(task_id)
    if known:
        bits.append(
            "WHAT WE HAVE ALREADY ESTABLISHED ON THIS TASK (build on it, do "
            "not repeat it back. If newer sources contradict any of it, say so "
            "explicitly and give the newer position with its date):\n" + known)
    return t, "\n\n".join(bits)


def _prompt(context, question, recency):
    p = "Today is %s.\n\n%s\n\nQUESTION: %s" % (
        dt.date.today().isoformat(), context, question)
    if recency:
        p += ("\n\nPrioritise sources published in the last %s, and give the "
              "date of anything you rely on." % recency)
    return p


def _ask_claude(context, question, cfg, recency):
    return tk_ai.searched_text(SYSTEM, _prompt(context, question, recency),
                               max_searches=cfg["searches"],
                               max_tokens=cfg["tokens"])


def _ask_perplexity(context, question, depth, cfg, recency):
    return tk_perplexity.ask_formatted(
        SYSTEM, _prompt(context, question, recency),
        depth=depth, recency=recency, max_tokens=cfg["tokens"])


LAKE_FRAMING = (
    "You are answering a question asked from inside a specific finance task, "
    "against Yo-Chi's OWN data lake - not the web. Use the task context to "
    "decide what to query and how to frame the answer.\n\n"
    "Answer with figures, not description: the numbers you found, the period "
    "they cover, and the entities or venues involved. Name the tables you "
    "used. If the lake cannot answer part of the question, say which part and "
    "what data would be needed - never estimate a number a query did not "
    "return.")


def _ask_lake(context, question):
    """Analysis over OUR data rather than the web: the same agentic SQL loop
    the cockpit assistant uses, pointed at this task's question."""
    import tk_ask
    out = tk_ask.answer("%s\n\n%s\n\nQUESTION: %s"
                        % (LAKE_FRAMING, context, question))
    answer = (out.get("answer") or "").strip()
    sql = [q for q in (out.get("sql") or []) if q]
    if sql:
        # provenance, kept with the finding: a later question - or the agent
        # playbook - can reuse a query that already worked
        answer += "\n\n---\nQueries run against the lake (%d):\n\n%s" % (
            len(sql), "\n\n".join("```sql\n%s\n```" % q[:1200] for q in sql[:6]))
    return answer


def _compare(context, question, depth, cfg, recency):
    """Both engines, then a reconciliation. One engine failing degrades to the
    other's answer rather than losing the question."""
    a = b = None
    errs = []
    try:
        a = (_ask_claude(context, question, cfg, recency) or "").strip()
    except Exception as e:
        LOG.exception("claude engine failed")
        errs.append("Claude web search failed: " + str(e)[:150])
    try:
        b = (_ask_perplexity(context, question, depth, cfg, recency) or "").strip()
    except Exception as e:
        LOG.exception("perplexity engine failed")
        errs.append("Perplexity failed: " + str(e)[:150])
    if not (a and b):
        answer = a or b or ""
        if errs:
            answer = (answer + "\n\n(" + "; ".join(errs) + ")").strip()
        return answer
    try:
        verdict = tk_ai.text(
            RECONCILE,
            "QUESTION: %s\n\n=== ENGINE A (Claude web search) ===\n%s"
            "\n\n=== ENGINE B (Perplexity) ===\n%s" % (question, a, b),
            max_tokens=1200).strip()
    except Exception:
        LOG.exception("reconciliation failed")
        verdict = "(could not reconcile the two answers automatically)"
    return ("%s\n\n--- Engine A: Claude web search ---\n%s"
            "\n\n--- Engine B: Perplexity ---\n%s" % (verdict, a, b))


def run(task_id, question, uid=None, depth="standard", save=True,
        engine="claude", recency=None):
    question = (question or "").strip()
    if not question:
        raise ValueError("ask a question")
    if engine not in ENGINES:
        engine = "claude"
    if recency not in RECENCY:
        recency = None
    cfg = DEPTHS.get(depth, DEPTHS["standard"])
    task, context = _context(task_id)

    if engine in ("perplexity", "compare") and not tk_perplexity.enabled():
        if engine == "perplexity":
            raise ValueError("Perplexity is not configured (no API key set)")
        engine = "claude"      # compare degrades to one engine, not an error

    if engine == "lake":
        answer = _ask_lake(context, question)
    elif engine == "claude":
        answer = _ask_claude(context, question, cfg, recency)
    elif engine == "perplexity":
        answer = _ask_perplexity(context, question, depth, cfg, recency)
    else:
        answer = _compare(context, question, depth, cfg, recency)

    answer = (answer or "").strip()
    if not answer:
        raise RuntimeError("no answer came back - try rewording the question")

    knowledge = None
    comment_id = None
    if save:
        # the full finding lands in the task's Knowledge tab and refreshes the
        # living summary, so the next question starts from what we already know
        knowledge = tk_knowledge.add(task_id, question, answer, engine=engine,
                                     depth=depth, recency=recency, uid=uid)
        if uid:
            # and a short pointer stays in the conversation, so the team sees a
            # question was answered (and gets notified) without the comment
            # stream carrying the whole finding twice
            head = answer.strip().split("\n\n")[0].strip()
            if len(head) > 600:
                head = head[:600].rstrip() + "..."
            note = ("%s: %s\n\n%s\n\nFull finding and the updated "
                    "knowledge for this task are on the Knowledge tab."
                    % ("Analysed the data lake" if engine == "lake"
                       else "Researched", question[:200], head))
            try:
                rows = tk_db.insert("comments", [{
                    "task_id": task_id, "author_id": uid,
                    "body": note[:4000]}], returning=True)
                comment_id = rows[0]["id"] if rows else None
            except Exception:
                LOG.exception("could not post the research pointer comment")
    return {"question": question, "depth": depth, "engine": engine,
            "recency": recency, "answer": answer, "knowledge": knowledge,
            "saved_comment_id": comment_id, "task_title": task.get("title")}
