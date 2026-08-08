# -*- coding: utf-8 -*-
"""Weekly venue health score - LEADING indicators blended into one 0-100 score
per venue, so trouble is visible before it reaches the sales line.

Components (each scored 0-100 vs the network, equally weighted, only when the
underlying feed has data for the venue):
  reviews    - last-4-week avg rating and negative-review share (mart_reviews_weekly)
  procedures - last-week completion % (mart_procedure_daily)
  mystery    - latest month score (MysteryShoppingSiteComparisons, dynamic month cols)
  labour     - last-4-week labour% distance from the network median (only healthy rows)
  lfl        - last-4-week average LFL

Writes taskapp.venue_health, emails the bottom-10 + biggest droppers, and
raises ONE summary task when venues fall below 40 or drop >15 points."""
import datetime as dt
import json
import logging
import re

import tk_calendar
import tk_db
import tk_email
import tk_flash
import lake_reader


def _n(name):
    """Canonical venue key across the three naming styles: 'Harbour Town'
    (reviews), 'Yo-Chi Burleigh' (procedures/POS), 'Broadbeach Yo-Chi' (mystery)."""
    n = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    n = re.sub(r"^yochi", "", n)
    n = re.sub(r"yochi$", "", n)
    n = re.sub(r"street$", "st", n)
    return tk_flash.ALIASES.get(n, n)

LOG = logging.getLogger("tk_health")

DATA_WATCH_PROJECT = "aaaaaaaa-0000-0000-0000-000000000005"
ALERT_SCORE = 40
ALERT_DROP = 15


def _monday(d):
    return d - dt.timedelta(days=d.weekday())


def _pct_rank(values, v, invert=False):
    """0-100 percentile of v among values (higher = healthier)."""
    vals = sorted(x for x in values if x is not None)
    if not vals or v is None:
        return None
    below = sum(1 for x in vals if x < v)
    pr = 100.0 * below / len(vals)
    return round(100 - pr if invert else pr, 1)


def build():
    lake_reader.sync(extra_tables=["MysteryShoppingSiteComparisons"], log=LOG.info)
    wk = _monday(dt.date.today()) - dt.timedelta(days=7)
    w4 = wk - dt.timedelta(days=21)

    rev = lake_reader.query("""
        SELECT venue, sum(reviews*avg_rating)/nullif(sum(reviews),0) rating,
               sum(negative_reviews)*1.0/nullif(sum(reviews),0) neg
        FROM mart_reviews_weekly
        WHERE CAST(week_start AS DATE) >= DATE '%s' GROUP BY 1""" % w4, max_rows=120)
    proc = lake_reader.query("""
        SELECT venue, avg(completion_pct) FROM mart_procedure_daily
        WHERE CAST(day AS DATE) >= DATE '%s' GROUP BY 1""" % wk, max_rows=120)
    lab = lake_reader.query("""
        SELECT venue,
               sum(CASE WHEN labour_cost > 0 AND labour_cost < net_sales THEN labour_cost END)
                 / nullif(sum(CASE WHEN labour_cost > 0 AND labour_cost < net_sales
                                   THEN net_sales END),0) lp,
               avg(lfl_pct) lfl
        FROM mart_venue_daily
        WHERE CAST("date" AS DATE) >= DATE '%s' GROUP BY 1""" % w4, max_rows=120)
    myst = lake_reader.query(
        'SELECT * FROM MysteryShoppingSiteComparisons', max_rows=120)

    # mystery: the month columns are literal 'YYYY-MM' names - use the latest with data
    mcols = sorted(c for c in myst["columns"] if re.fullmatch(r"\d{4}-\d{2}", c))
    vidx = myst["columns"].index("Venue")
    mystery = {}
    for row in myst["rows"]:
        for c in reversed(mcols):
            v = row[myst["columns"].index(c)]
            if v is not None and str(v).strip() not in ("", "0"):
                try:
                    mystery[_n(row[vidx])] = float(v)
                except (ValueError, TypeError):
                    pass
                break

    ratings = {_n(r[0]): r[1] for r in rev["rows"]}
    negs = {_n(r[0]): r[2] for r in rev["rows"]}
    procs = {_n(r[0]): r[1] for r in proc["rows"]}
    labs = {_n(r[0]): r[1] for r in lab["rows"]}
    lfls = {_n(r[0]): r[2] for r in lab["rows"]}
    display = {_n(r[0]): r[0] for r in lab["rows"]}  # POS names win for display
    venues = set(labs) | set(ratings) | set(procs)

    scores = []
    for v in venues:
        comp = {
            "rating": _pct_rank(ratings.values(), ratings.get(v)),
            "negative_reviews": _pct_rank(negs.values(), negs.get(v), invert=True),
            "procedures": _pct_rank(procs.values(), procs.get(v)),
            "labour": _pct_rank(labs.values(), labs.get(v), invert=True),
            "lfl": _pct_rank(lfls.values(), lfls.get(v)),
            "mystery": _pct_rank(mystery.values(), mystery.get(v)),
        }
        used = {k: s for k, s in comp.items() if s is not None}
        if len(used) < 3:
            continue
        scores.append({"venue": display.get(v, v), "week_start": wk.isoformat(),
                       "score": round(sum(used.values()) / len(used), 1),
                       "components": comp})
    return {"week": wk.isoformat(), "scores": scores}


def run(email=True):
    out = build()
    if not out["scores"]:
        return {"week": out["week"], "venues": 0}
    prev = {r["venue"]: float(r["score"]) for r in tk_db.get(
        "venue_health", {"week_start": "eq." + (
            dt.date.fromisoformat(out["week"]) - dt.timedelta(days=7)).isoformat(),
            "select": "venue,score"})}
    tk_db.insert("venue_health",
                 [{**s, "components": s["components"]} for s in out["scores"]],
                 on_conflict="venue,week_start", ignore_duplicates=True)

    ranked = sorted(out["scores"], key=lambda s: s["score"])
    droppers = sorted(
        (s for s in out["scores"] if s["venue"] in prev
         and prev[s["venue"]] - s["score"] >= ALERT_DROP),
        key=lambda s: s["score"] - prev[s["venue"]])
    alerts = [s for s in ranked if s["score"] < ALERT_SCORE][:10]

    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id,email"})
    created = 0
    if alerts or droppers:
        title = "[Health] Venue health alerts — week of %s" % out["week"]
        if not tk_db.get("tasks", {"title": "eq." + title, "select": "id", "limit": "1"}):
            lines = ["Score < %d:" % ALERT_SCORE] + [
                "- %s: %s (%s)" % (s["venue"], s["score"], ", ".join(
                    "%s %s" % (k, v) for k, v in s["components"].items()
                    if v is not None and v < 30)) for s in alerts]
            if droppers:
                lines += ["Dropped >%d pts week-on-week:" % ALERT_DROP] + [
                    "- %s: %s -> %s" % (s["venue"], prev[s["venue"]], s["score"])
                    for s in droppers]
            due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=3))
            tk_db.insert("tasks", [{
                "project_id": DATA_WATCH_PROJECT, "title": title,
                "description": ("Leading-indicator health score (reviews, procedures, "
                                "mystery, labour, LFL — percentile vs network).\n%s\n"
                                "First steps:\n- Look at the weakest component per venue\n"
                                "- Loop in the area manager before it reaches sales") %
                               "\n".join(lines),
                "priority": "high",
                "assignee_id": admin[0]["id"] if admin else None,
                "due_date": due.isoformat(), "source": "watcher"}])
            created = 1

    if email:
        rows = "".join(
            "<tr><td>%s</td><td align=right><b%s>%s</b></td><td align=right>%s</td></tr>" % (
                s["venue"],
                " style='color:#b91c1c'" if s["score"] < ALERT_SCORE else "",
                s["score"],
                ("%+.1f" % (s["score"] - prev[s["venue"]])) if s["venue"] in prev else "")
            for s in ranked[:15])
        html = ("<p><b>Venue health — week of %s</b> (0-100, leading indicators; "
                "network percentile blend). Bottom 15:</p>"
                "<table cellpadding=4 border=1 style='border-collapse:collapse;font-size:12px'>"
                "<tr><th>Venue</th><th>Score</th><th>vs last wk</th></tr>%s</table>"
                "<p style='color:#6b7280;font-size:11px'>Components: reviews rating, "
                "negative share, procedure completion, labour%%, LFL, mystery shopping. "
                "%d venue(s) scored.</p>") % (out["week"], rows, len(out["scores"]))
        for a in admin:
            tk_email.send(a["email"], "Venue health — wk %s: %d alert(s), %d dropper(s)"
                          % (out["week"], len(alerts), len(droppers)), html)
    return {"week": out["week"], "venues": len(out["scores"]),
            "alerts": len(alerts), "droppers": len(droppers),
            "task_created": bool(created)}
