# -*- coding: utf-8 -*-
"""Weekly prime-cost flash: per-venue food cost (Restoke purchases received in
the week - the purchases-based metric; Yo-Chi has no stocktake-based COGS) +
labour (mart, where the feed is healthy) against net sales, vs each venue's
trailing 8-week average. Exceptions raise tasks; the flash table is emailed to
admins Monday morning. Names: Restoke venue names differ slightly from POS -
normalised join + explicit aliases; unmatched venues are LISTED, never dropped
silently."""
import datetime as dt
import json
import logging
import re

import tk_ai
import tk_calendar
import tk_db
import tk_email
import lake_reader

LOG = logging.getLogger("tk_flash")

DATA_WATCH_PROJECT = "aaaaaaaa-0000-0000-0000-000000000005"
FOOD_JUMP_PTS = 3.0     # flag when food% > trailing avg + 3pts
PRIME_CEILING = 70.0    # flag when prime% (food+labour) above this
MIN_WEEK_SALES = 20000  # ignore tiny/partial weeks
MAX_TASKS = 8

# normalised Restoke name -> normalised POS name (both lower/alnum, 'yochi' stripped)
ALIASES = {
    "albertstreet": "albertst",
    "charlestownsquare": "charlestown",
    "mountgravatt": "mtgravatt",
    "queenvictoriabuilding": "qv",
    "macquariepark": "macquarie",
    "victoriapark": "vicpark",
    "cockburn": "cockburncentral",
    "chevronsurfers": "chevronsurfersparadise",
    "esplanadesurfers": "theesplanadesurfersparadise",
    "kawana": "kawanawaters",
    "gougerstreet": "gougerst",
    "henley": "henleybeach",
}


def _norm(name):
    n = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    n = re.sub(r"^yochi", "", n)
    return ALIASES.get(n, n)


def build():
    lake_reader.sync(extra_tables=["restoke_purchasing"], log=LOG.info)
    today = dt.date.today()
    this_mon = today - dt.timedelta(days=today.weekday())
    wk_start = this_mon - dt.timedelta(days=7)     # last complete week
    hist_start = wk_start - dt.timedelta(weeks=8)

    food = lake_reader.query("""
        SELECT venue, date_trunc('week', CAST(COALESCE(received_date, date) AS DATE)) wk,
               round(sum(TRY_CAST(total AS DOUBLE)),0) purchases
        FROM restoke_purchasing
        WHERE CAST(COALESCE(received_date, date) AS DATE) >= DATE '%s'
          AND CAST(COALESCE(received_date, date) AS DATE) < DATE '%s'
        GROUP BY 1,2""" % (hist_start, this_mon), max_rows=1000)
    sales = lake_reader.query("""
        SELECT venue, date_trunc('week', CAST("date" AS DATE)) wk,
               round(sum(net_sales),0) net_sales,
               round(sum(CASE WHEN labour_cost > 0 AND labour_cost < net_sales
                              THEN labour_cost END),0) labour
        FROM mart_venue_daily
        WHERE CAST("date" AS DATE) >= DATE '%s' AND CAST("date" AS DATE) < DATE '%s'
        GROUP BY 1,2""" % (hist_start, this_mon), max_rows=1000)

    s_by = {}
    for v, wk, ns, lab in sales["rows"]:
        s_by[(_norm(v), str(wk)[:10])] = {"venue": v, "net_sales": ns or 0,
                                          "labour": lab}
    unmatched = set()
    f_by = {}
    for v, wk, amt in food["rows"]:
        key = (_norm(v), str(wk)[:10])
        f_by[key] = (f_by.get(key, (0,))[0] + (amt or 0),)
        if not any(k[0] == _norm(v) for k in s_by):
            unmatched.add(v)

    # per-venue: last week + trailing avg food%
    venues = {}
    for (nv, wk), s in s_by.items():
        f = f_by.get((nv, wk), (None,))[0]
        if s["net_sales"] <= 0:
            continue
        rec = venues.setdefault(nv, {"venue": s["venue"], "hist": []})
        row = {"wk": wk, "sales": s["net_sales"], "food": f,
               "labour": s["labour"],
               "food_pct": round(100.0 * f / s["net_sales"], 1) if f else None}
        if wk == wk_start.isoformat():
            rec["last"] = row
        else:
            rec["hist"].append(row)

    flash, exceptions = [], []
    for nv, rec in venues.items():
        last = rec.get("last")
        if not last or last["sales"] < MIN_WEEK_SALES:
            continue
        hist_pcts = [h["food_pct"] for h in rec["hist"] if h["food_pct"]]
        avg_food = round(sum(hist_pcts) / len(hist_pcts), 1) if hist_pcts else None
        labour_pct = (round(100.0 * last["labour"] / last["sales"], 1)
                      if last.get("labour") else None)
        prime = (round(last["food_pct"] + labour_pct, 1)
                 if last["food_pct"] and labour_pct else None)
        row = {"venue": rec["venue"], "sales": last["sales"],
               "food_pct": last["food_pct"], "avg_food_pct": avg_food,
               "labour_pct": labour_pct, "prime_pct": prime}
        flash.append(row)
        if last["food_pct"] and avg_food and last["food_pct"] > avg_food + FOOD_JUMP_PTS:
            exceptions.append({**row, "why": "food %.1f%% vs %.1f%% trailing avg (+%.1fpts)"
                               % (last["food_pct"], avg_food, last["food_pct"] - avg_food)})
        elif prime and prime > PRIME_CEILING:
            exceptions.append({**row, "why": "prime cost %.1f%% above %.0f%% ceiling"
                               % (prime, PRIME_CEILING)})
    flash.sort(key=lambda r: -(r["food_pct"] or 0))
    exceptions.sort(key=lambda r: -(r["food_pct"] or 0))
    return {"week": wk_start.isoformat(), "flash": flash, "exceptions": exceptions,
            "unmatched_restoke_venues": sorted(unmatched)}


def run(email=True):
    out = build()
    admin = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                   "select": "id,email"})
    created = 0
    due = tk_calendar.roll_forward(dt.date.today() + dt.timedelta(days=2))
    for ex in out["exceptions"][:MAX_TASKS]:
        title = "[Flash] %s prime cost — week of %s" % (ex["venue"], out["week"])
        if tk_db.get("tasks", {"title": "eq." + title, "select": "id", "limit": "1"}):
            continue
        tk_db.insert("tasks", [{
            "project_id": DATA_WATCH_PROJECT,
            "title": title,
            "description": ("Weekly prime-cost flash flagged %s: %s.\nWeek sales $%s; "
                            "food %s%% (trailing 8-wk avg %s%%); labour %s%%; prime %s%%.\n"
                            "First steps:\n- Check large/irregular deliveries in Restoke "
                            "for the week\n- Confirm invoice timing (a big order landing "
                            "in one week skews the %%)\n- Ask the venue about waste/prep") % (
                ex["venue"], ex["why"], "{:,.0f}".format(ex["sales"]),
                ex["food_pct"], ex["avg_food_pct"], ex["labour_pct"] or "n/a",
                ex["prime_pct"] or "n/a"),
            "priority": "high",
            "assignee_id": admin[0]["id"] if admin else None,
            "due_date": due.isoformat(),
            "source": "watcher",
        }])
        created += 1

    if email and out["flash"]:
        rows = "".join(
            "<tr><td>%s</td><td align=right>%s</td><td align=right%s>%s</td>"
            "<td align=right>%s</td><td align=right>%s</td><td align=right>%s</td></tr>" % (
                r["venue"], "{:,.0f}".format(r["sales"]),
                " style='color:#b91c1c;font-weight:700'" if any(
                    e["venue"] == r["venue"] for e in out["exceptions"]) else "",
                r["food_pct"] or "-", r["avg_food_pct"] or "-",
                r["labour_pct"] or "-", r["prime_pct"] or "-")
            for r in out["flash"])
        note = ""
        if out["unmatched_restoke_venues"]:
            note = ("<p style='color:#b45309;font-size:11.5px'>Restoke venues with no POS "
                    "match (excluded): %s</p>") % ", ".join(out["unmatched_restoke_venues"])
        html = ("<p><b>Prime-cost flash</b> — week of %s. %d exception(s) raised as tasks."
                "</p><table cellpadding=5 border=1 style='border-collapse:collapse;"
                "font-size:12px'><tr><th>Venue</th><th>Sales</th><th>Food%%</th>"
                "<th>8wk avg</th><th>Labour%%</th><th>Prime%%</th></tr>%s</table>%s"
                "<p style='color:#6b7280;font-size:11.5px'>Food%% = Restoke purchases "
                "received in week / net sales (purchases-based; no stocktake COGS exists). "
                "Labour shown only where the feed is healthy.</p>") % (
            out["week"], len(out["exceptions"]), rows, note)
        for a in admin:
            tk_email.send(a["email"], "Prime-cost flash — wk %s: %d exception(s)" % (
                out["week"], len(out["exceptions"])), html)

    return {"week": out["week"], "venues": len(out["flash"]),
            "exceptions": len(out["exceptions"]), "tasks_created": created,
            "unmatched": out["unmatched_restoke_venues"]}
