# -*- coding: utf-8 -*-
"""Per-task recurrence: tick "Repeat" on any task and the next instance is
created automatically.

Distinct from task_templates (the structured close/statutory calendar, which
generates from bd/dom rules per period). This is the lightweight "check this
every week" recurrence you set from the task itself.

Two modes:
  schedule   - next instance is due one interval after THIS instance's due date
               (BAS is due monthly whether or not you finished last month's)
  completion - next instance appears one interval after you tick this one off
               (good for "review X every 2 weeks from when I last did it")

Generated up to LEAD_DAYS ahead, so exactly one future instance exists at a
time. Dedup is a unique index on (recurrence_parent_id, due_date), so re-runs
and overlapping timers can never double-create."""
import calendar
import datetime as dt
import logging

import tk_db

LOG = logging.getLogger("tk_recurring")

AEST = dt.timezone(dt.timedelta(hours=10))
LEAD_DAYS = 7
MAX_PER_RUN = 200
COPY_FIELDS = ("project_id", "title", "description", "priority", "assignee_id",
               "reviewer_id", "checklist", "recurrence", "recurrence_mode",
               "recurrence_until", "source")


def add_months(d, months):
    """Same day-of-month N months on, clamped to the end of short months."""
    y, m = divmod((d.year * 12 + d.month - 1) + months, 12)
    m += 1
    return d.replace(year=y, month=m, day=min(d.day, calendar.monthrange(y, m)[1]))


def next_due(recurrence, from_date):
    if recurrence == "daily":
        return from_date + dt.timedelta(days=1)
    if recurrence == "weekly":
        return from_date + dt.timedelta(days=7)
    if recurrence == "fortnightly":
        return from_date + dt.timedelta(days=14)
    if recurrence == "monthly":
        return add_months(from_date, 1)
    if recurrence == "quarterly":
        return add_months(from_date, 3)
    if recurrence == "annual":
        return add_months(from_date, 12)
    return None


def _as_date(v):
    if not v:
        return None
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _completed_date(v):
    """completed_at is stored in UTC; the business day is AEST. Without this a
    task ticked off at 9am Tuesday reads as Monday and the next instance lands
    a day early."""
    if not v:
        return None
    try:
        d = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return _as_date(v)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(AEST).date()


def run(today=None):
    today = today or dt.datetime.now(AEST).date()
    horizon = today + dt.timedelta(days=LEAD_DAYS)
    rows = tk_db.get("tasks", {
        "recurrence": "not.is.null",
        "select": "id,project_id,title,description,priority,assignee_id,reviewer_id,"
                  "checklist,status,due_date,completed_at,source,recurrence,"
                  "recurrence_mode,recurrence_until,recurrence_parent_id",
        "limit": "2000"})
    series = {}
    for t in rows:
        series.setdefault(t.get("recurrence_parent_id") or t["id"], []).append(t)

    created, skipped, errors = [], 0, []
    for root_id, instances in list(series.items())[:MAX_PER_RUN]:
        try:
            # the newest instance drives the series
            latest = max(instances, key=lambda t: (
                _as_date(t.get("due_date")) or dt.date.min, t["id"]))
            rec = latest.get("recurrence")
            if not rec:
                continue
            mode = latest.get("recurrence_mode") or "schedule"
            if mode == "completion":
                if latest.get("status") != "done":
                    skipped += 1
                    continue
                anchor = _completed_date(latest.get("completed_at")) or today
            else:
                anchor = _as_date(latest.get("due_date"))
                if not anchor:
                    skipped += 1  # schedule mode needs a due date to count from
                    continue
            nxt = next_due(rec, anchor)
            if not nxt:
                continue
            # catch up a series that has been dormant, without backfilling history
            guard = 0
            while nxt < today and guard < 60:
                nxt = next_due(rec, nxt)
                guard += 1
            until = _as_date(latest.get("recurrence_until"))
            if until and nxt > until:
                skipped += 1
                continue
            # The lead window only applies to schedule mode (don't run months
            # ahead of the calendar). In completion mode the trigger IS the
            # completion, so the next one must appear straight away even though
            # its due date is a full interval out.
            if mode == "schedule" and nxt > horizon:
                skipped += 1
                continue
            row = {f: latest.get(f) for f in COPY_FIELDS}
            row["due_date"] = nxt.isoformat()
            row["status"] = "todo"
            row["recurrence_parent_id"] = root_id
            tk_db.insert("tasks", [row],
                         on_conflict="recurrence_parent_id,due_date",
                         ignore_duplicates=True)
            created.append({"title": latest["title"], "due": nxt.isoformat(),
                            "every": rec, "mode": mode})
        except Exception as e:
            LOG.exception("recurrence failed for series %s", root_id)
            errors.append({"series": root_id, "error": str(e)[:200]})
    out = {"series": len(series), "created": len(created),
           "not_due_yet": skipped, "instances": created[:25]}
    if errors:
        out["errors"] = errors
    return out
