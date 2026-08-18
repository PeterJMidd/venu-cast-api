# -*- coding: utf-8 -*-
"""Ask-a-question research on a task: live web search, grounded in what the
task is actually about, with the answer written back onto the task.

Deliberately separate from the Task agent: the agent analyses OUR data (the
lake) and produces a deliverable; this answers "what is the rule / what does
the market do / what changed" from the open web, always cited. When a question
really needs Yo-Chi's own numbers, the answer says so and points at the agent
rather than guessing.

The answer is posted as a comment, so it is durable, searchable, visible to
whoever else is on the task, and notifies them through the usual path."""
import datetime as dt
import logging

import tk_ai
import tk_db

LOG = logging.getLogger("tk_research")

DEPTHS = {
    "quick":    {"searches": 3,  "tokens": 1500},
    "standard": {"searches": 6,  "tokens": 3000},
    "deep":     {"searches": 12, "tokens": 6000},
}

SYSTEM = (
    "You are the research analyst for the CFO of Yo-Chi, an Australian "
    "frozen-yoghurt chain (74 AU venues; also UK, USA (FL/TX), Singapore/SEA). "
    "Answer the question using LIVE WEB SEARCH and report only what you find, "
    "with the source named inline and every URL listed at the end under "
    "'Sources'. Australian rules, dates and dollars unless the question is "
    "about another market.\n\n"
    "Rules:\n"
    "- Lead with the direct answer in 1-2 sentences, then the detail.\n"
    "- Be specific: rates, thresholds, dates, section numbers, dollar figures.\n"
    "- Date everything, and say when a rule changes or has recently changed.\n"
    "- If sources disagree or the position is unsettled, say so plainly.\n"
    "- If you cannot verify something, say you could not verify it. Never "
    "invent a figure, a citation, or a URL.\n"
    "- If the question actually needs Yo-Chi's own data (our sales, our GL, "
    "our contracts), answer what you can from the web and then say which "
    "internal source would settle it - suggest the Task agent for that.\n"
    "- Fast Food Award items are never 'breaches'; call them areas to review.\n"
    "- Plain text, no markdown headers or asterisks - it is posted as a task "
    "comment. Short paragraphs and '- ' bullets only."
)


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
    notes = [c["body"][:400] for c in recent if not c["body"].startswith("Research:")]
    if notes:
        bits.append("RECENT NOTES ON THE TASK:\n" + "\n---\n".join(notes))
    return t, "\n\n".join(bits)


def run(task_id, question, uid=None, depth="standard", save=True):
    question = (question or "").strip()
    if not question:
        raise ValueError("ask a question")
    cfg = DEPTHS.get(depth, DEPTHS["standard"])
    task, context = _context(task_id)
    answer = tk_ai.searched_text(
        SYSTEM,
        "Today is %s.\n\n%s\n\nQUESTION: %s" % (
            dt.date.today().isoformat(), context, question),
        max_searches=cfg["searches"], max_tokens=cfg["tokens"])
    answer = (answer or "").strip()
    if not answer:
        raise RuntimeError("no answer came back - try rewording the question")
    comment_id = None
    if save and uid:
        body = "Research: %s\n\n%s" % (question[:200], answer)
        rows = tk_db.insert("comments", [{
            "task_id": task_id, "author_id": uid, "body": body[:9000]}],
            returning=True)
        comment_id = rows[0]["id"] if rows else None
    return {"question": question, "depth": depth, "answer": answer,
            "saved_comment_id": comment_id, "task_title": task.get("title")}
