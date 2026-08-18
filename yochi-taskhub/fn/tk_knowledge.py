# -*- coding: utf-8 -*-
"""Cumulative task knowledge.

Every research answer is kept as an entry, and on top of them a single rolling
"what we now know" summary is maintained. Two consequences that make the task
actually get smarter:

  1. the summary is fed back INTO the next question, so research builds on what
     was already established instead of re-deriving it; and
  2. when a newer source contradicts an older finding, the consolidation marks
     the old one superseded rather than leaving both to be believed.

Nothing is deleted - the entry history stays, so you can always see how the
understanding got to where it is."""
import datetime as dt
import logging

import tk_ai
import tk_db

LOG = logging.getLogger("tk_knowledge")

MAX_SUMMARY = 7000        # keep the living summary tight enough to re-prompt with
MAX_ENTRY_IN_PROMPT = 4000

CONSOLIDATE = """You maintain the living knowledge base for one finance task.
You get the CURRENT UNDERSTANDING (may be empty) and a NEW FINDING. Return the
updated understanding - the whole thing, ready to replace what was there.

Rules:
- Merge the new finding in. Do not simply append; integrate it where it belongs.
- If the new finding CONTRADICTS something already there, keep the newer
  position and mark the old one, e.g. "(superseded DD Mon YYYY: previously X)".
- Keep every source URL that supports a fact you retain.
- Organise under short ALL-CAPS headings that suit the subject (e.g. RATES,
  DEADLINES, MECHANICS, RISKS). Invent headings that fit; do not force a
  template.
- End with an "OPEN QUESTIONS" section: what is still unresolved or needs a
  primary source or an adviser. Remove questions the new finding has answered.
- Plain text only. No markdown headers, no asterisks. Bullets as "- ".
- Use ONLY what is in the current understanding and the new finding. Never add
  a fact, figure or citation of your own.
- Be concise. Aim under 500 words; drop restatement, keep specifics (rates,
  dates, section numbers, URLs)."""


def get(task_id):
    rows = tk_db.get("task_knowledge", {
        "task_id": "eq." + task_id,
        "select": "task_id,summary,entry_count,updated_at"})
    return rows[0] if rows else None


def summary_text(task_id):
    row = get(task_id)
    return (row or {}).get("summary") or ""


def entries(task_id, limit=50):
    return tk_db.get("knowledge", {
        "task_id": "eq." + task_id,
        "select": "id,question,answer,engine,depth,recency,created_by,created_at",
        "order": "created_at.desc", "limit": str(limit)})


def _consolidate(task_id, question, answer, current):
    """Merge one finding into the living summary. Failure here must not lose
    the finding - the entry is already stored."""
    try:
        new = tk_ai.text(
            CONSOLIDATE,
            "CURRENT UNDERSTANDING:\n%s\n\nNEW FINDING\nQuestion asked: %s\n"
            "Answer:\n%s" % (current or "(nothing recorded yet)",
                             question, (answer or "")[:MAX_ENTRY_IN_PROMPT]),
            max_tokens=2000).strip()
        return new[:MAX_SUMMARY] if new else current
    except Exception:
        LOG.exception("knowledge consolidation failed for task %s", task_id)
        return current


def add(task_id, question, answer, engine=None, depth=None, recency=None,
        uid=None, consolidate=True):
    """Store a finding and refresh the living summary. Returns the new summary."""
    tk_db.insert("knowledge", [{
        "task_id": task_id, "question": question[:1000], "answer": answer,
        "engine": engine, "depth": depth, "recency": recency,
        "created_by": uid}])
    current = summary_text(task_id)
    updated = _consolidate(task_id, question, answer, current) if consolidate \
        else current
    count = len(tk_db.get("knowledge", {"task_id": "eq." + task_id,
                                        "select": "id", "limit": "500"}))
    row = {"summary": updated or "", "entry_count": count,
           "updated_at": dt.datetime.utcnow().isoformat() + "Z"}
    if get(task_id):
        tk_db.patch("task_knowledge", {"task_id": "eq." + task_id}, row)
    else:
        tk_db.insert("task_knowledge", [dict(row, task_id=task_id)])
    return updated


def rebuild(task_id):
    """Re-derive the summary from every stored entry, oldest first. Use after
    deleting an entry, or if a consolidation went wrong."""
    rows = sorted(entries(task_id, limit=200),
                  key=lambda r: r.get("created_at") or "")
    summary = ""
    for r in rows:
        summary = _consolidate(task_id, r["question"], r["answer"], summary)
    if get(task_id):
        tk_db.patch("task_knowledge", {"task_id": "eq." + task_id},
                    {"summary": summary or "", "entry_count": len(rows),
                     "updated_at": dt.datetime.utcnow().isoformat() + "Z"})
    else:
        tk_db.insert("task_knowledge", [{
            "task_id": task_id, "summary": summary or "",
            "entry_count": len(rows)}])
    return {"task_id": task_id, "entries": len(rows),
            "summary_chars": len(summary or "")}
