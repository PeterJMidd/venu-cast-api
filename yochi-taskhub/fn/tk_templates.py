# -*- coding: utf-8 -*-
"""Template instantiation: on the 1st (and safely re-runnable any day), make sure
the current month's period exists and every active template due this period has
its task created. Dedup via unique(template_id, period_id)."""
import datetime as dt
import logging

import tk_db
import tk_calendar

LOG = logging.getLogger("tk_templates")

QUARTER_MONTHS = {1, 4, 7, 10}
ANNUAL_MONTH = 7  # AU financial-year start


def ensure_period(today=None):
    today = today or dt.date.today()
    first = today.replace(day=1)
    label = first.strftime("%b %Y")
    tk_db.insert("periods", [{"period_month": first.isoformat(), "label": label}],
                 on_conflict="period_month", ignore_duplicates=True)
    rows = tk_db.get("periods", {"period_month": "eq." + first.isoformat(), "select": "id"})
    return rows[0]["id"], first


def cadence_due_this_month(cadence, month):
    if cadence == "monthly":
        return True
    if cadence == "quarterly":
        return month in QUARTER_MONTHS
    if cadence == "annual":
        return month == ANNUAL_MONTH
    return False


def run(today=None):
    today = today or dt.date.today()
    period_id, first = ensure_period(today)
    templates = tk_db.get("task_templates", {"active": "eq.true", "select": "*"})
    created = 0
    errors = []
    for t in templates:
        try:
            if not cadence_due_this_month(t["cadence"], first.month):
                continue
            due = tk_calendar.due_date_for(t["due_rule"], first.year, first.month)
            # Grace guard: never retro-create a task more than 3 days past its due
            # date (e.g. templates adopted mid-month, or a long engine outage).
            if due < today - dt.timedelta(days=3):
                continue
            row = {
                "project_id": t["project_id"],
                "template_id": t["id"],
                "period_id": period_id,
                "title": t["title"],
                "description": t.get("description"),
                "priority": t.get("priority", "medium"),
                "assignee_id": t.get("default_assignee_id"),
                "reviewer_id": t.get("default_reviewer_id"),
                "due_date": due.isoformat(),
                "source": "template",
            }
            tk_db.insert("tasks", [row], on_conflict="template_id,period_id",
                         ignore_duplicates=True)
            created += 1
        except Exception as e:
            # one broken template must not sink the whole nightly run
            LOG.exception("template %r failed", t.get("title"))
            errors.append({"template": t.get("title"), "error": str(e)[:200]})
    LOG.info("template run for %s: %d template(s) processed", first, created)
    out = {"period": first.isoformat(), "templates_processed": created}
    if errors:
        out["errors"] = errors
    return out
