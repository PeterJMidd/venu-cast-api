# -*- coding: utf-8 -*-
"""Daily work-activity intelligence report: every morning, scan the prior day
across (a) the whole Azure blob estate (what was produced/changed), (b) the
data lake (Xero transactions & journals by type+entity, invoices touched,
Restoke receipting by venue, procedures completed BY PERSON, Asana completions
BY PERSON, sales context) and (c) TaskHub itself (actions/comments/completions
by person, agent+batch runs). AI narrative on top. Stored in
taskapp.work_reports (the /activity tab reads it) and emailed to admins.
Xero caveat: the Xero API carries no per-user attribution, so Xero activity is
split by transaction type + entity, not by person."""
import datetime as dt
import json
import logging
import os

import lake_reader
import tk_ai
import tk_db
import tk_email

LOG = logging.getLogger("tk_workreport")

AEST = dt.timezone(dt.timedelta(hours=10))
EXTRA_TABLES = ["XeroAccountTransactionsMasterView", "Invoices",
                "restoke_purchasing", "procedure_report", "AsanaTasks"]
BLOB_CAP_PER_CONTAINER = 25000


def _windows(report_date):
    """AEST calendar day -> (aest date iso, utc start iso, utc end iso)."""
    start_aest = dt.datetime.combine(report_date, dt.time(0), AEST)
    end_aest = start_aest + dt.timedelta(days=1)
    fmt = "%Y-%m-%d %H:%M:%S"
    return (start_aest.astimezone(dt.timezone.utc).strftime(fmt),
            end_aest.astimezone(dt.timezone.utc).strftime(fmt))


def _q(sql):
    """lake_reader.query returns {columns, rows-as-lists}; flatten to dicts."""
    try:
        out = lake_reader.query(sql, max_rows=100)
        return [dict(zip(out["columns"], r)) for r in out["rows"]]
    except Exception as e:
        LOG.exception("workreport query failed")
        return [{"error": str(e)[:150]}]


def _lake_stats(day, utc0, utc1):
    d = day.isoformat()
    return {
        "xero_by_source": _q("""
            SELECT "Source" AS source, "XeroOrganisationName" AS org,
                   count(*) AS lines,
                   round(sum(TRY_CAST("Net Amount" AS DOUBLE)), 0) AS net
            FROM XeroAccountTransactionsMasterView
            WHERE TRY_CAST("Date" AS DATE) = DATE '%s'
            GROUP BY 1, 2 ORDER BY lines DESC LIMIT 25""" % d),
        "invoices_touched": _q("""
            SELECT invoicetypedescribed AS typ, count(*) AS n,
                   round(sum(TRY_CAST(total AS DOUBLE)), 0) AS total
            FROM Invoices
            WHERE TRY_CAST(updateddateutc AS TIMESTAMP) >= TIMESTAMP '%s'
              AND TRY_CAST(updateddateutc AS TIMESTAMP) <  TIMESTAMP '%s'
            GROUP BY 1 ORDER BY n DESC""" % (utc0, utc1)),
        "restoke_receipting": _q("""
            SELECT venue, count(DISTINCT order_id) AS orders, count(*) AS lines,
                   round(sum(TRY_CAST(total AS DOUBLE)), 0) AS value
            FROM restoke_purchasing
            WHERE TRY_CAST(received_date AS DATE) = DATE '%s'
            GROUP BY 1 ORDER BY value DESC LIMIT 15""" % d),
        "procedures_by_person": _q("""
            SELECT completed_by AS person, count(*) AS procedures,
                   count(DISTINCT venue) AS venues
            FROM procedure_report
            WHERE TRY_CAST(completion_date AS DATE) = DATE '%s'
              AND status LIKE 'Completed%%'
            GROUP BY 1 ORDER BY procedures DESC LIMIT 20""" % d),
        "asana_completed_by_person": _q("""
            SELECT COALESCE(NULLIF(AssigneeName, ''), '(unassigned)') AS person,
                   count(*) AS completed
            FROM AsanaTasks
            WHERE TRY_CAST(CompletedAt AS TIMESTAMP) >= TIMESTAMP '%s'
              AND TRY_CAST(CompletedAt AS TIMESTAMP) <  TIMESTAMP '%s'
            GROUP BY 1 ORDER BY 2 DESC LIMIT 20""" % (utc0, utc1)),
        "sales_context": _q("""
            SELECT count(*) AS venues_traded,
                   round(sum(TRY_CAST(net_sales AS DOUBLE)), 0) AS net_sales
            FROM mart_venue_daily WHERE CAST("date" AS DATE) = DATE '%s'""" % d),
    }


def _taskhub_stats(utc0, utc1):
    iso0 = utc0.replace(" ", "T") + "Z"
    iso1 = utc1.replace(" ", "T") + "Z"
    profiles = {p["id"]: (p.get("full_name") or p["email"]) for p in tk_db.get(
        "profiles", {"select": "id,email,full_name"})}
    audits = tk_db.get("audit_log", {
        "at": ["gte." + iso0, "lt." + iso1],
        "select": "actor,action,table_name,old_row,new_row", "limit": "5000"})
    per = {}
    for a in audits:
        who = profiles.get(a["actor"], "system/automation")
        p = per.setdefault(who, {"created": 0, "updated": 0, "completed": 0,
                                 "comments": 0})
        if a["action"] == "INSERT":
            p["created"] += 1
        elif a["action"] == "UPDATE":
            new_s = (a.get("new_row") or {}).get("status")
            old_s = (a.get("old_row") or {}).get("status")
            if new_s == "done" and old_s != "done":
                p["completed"] += 1
            else:
                p["updated"] += 1
    comments = tk_db.get("comments", {
        "created_at": ["gte." + iso0, "lt." + iso1],
        "select": "author_id", "limit": "2000"})
    for c in comments:
        who = profiles.get(c["author_id"], "system/automation")
        per.setdefault(who, {"created": 0, "updated": 0, "completed": 0,
                             "comments": 0})["comments"] += 1
    agent_runs = tk_db.get("agent_runs", {
        "created_at": ["gte." + iso0, "lt." + iso1],
        "select": "outcome", "limit": "1000"})
    batches = tk_db.get("batch_runs", {
        "created_at": ["gte." + iso0, "lt." + iso1],
        "select": "kind,status", "limit": "200"})
    return {
        "people": [dict(person=k, **v) for k, v in
                   sorted(per.items(), key=lambda kv: -sum(kv[1].values()))],
        "agent_runs": {"total": len(agent_runs),
                       "success": sum(1 for r in agent_runs
                                      if r.get("outcome") == "success")},
        "batch_runs": [dict(kind=b["kind"], status=b.get("status"))
                       for b in batches],
    }


def _blob_scan(report_date):
    """Everything that changed in the blob estate on the report day (AEST)."""
    from azure.storage.blob import BlobServiceClient
    svc = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"])
    start = dt.datetime.combine(report_date, dt.time(0), AEST)
    end = start + dt.timedelta(days=1)
    out = []
    for cont in svc.list_containers():
        try:
            cc = svc.get_container_client(cont.name)
            changed = 0
            size = 0
            prefixes = {}
            samples = []
            seen = 0
            for b in cc.list_blobs():
                seen += 1
                if seen > BLOB_CAP_PER_CONTAINER:
                    break
                lm = b.last_modified
                if lm is None or not (start <= lm.astimezone(AEST) < end):
                    continue
                changed += 1
                size += b.size or 0
                top = b.name.split("/")[0]
                prefixes[top] = prefixes.get(top, 0) + 1
                if len(samples) < 6:
                    samples.append(b.name)
            if changed:
                out.append({"container": cont.name, "files_changed": changed,
                            "mb": round(size / 1048576.0, 1),
                            "areas": dict(sorted(prefixes.items(),
                                                 key=lambda kv: -kv[1])[:8]),
                            "samples": samples})
        except Exception:
            LOG.exception("blob scan failed for %s", cont.name)
    return sorted(out, key=lambda c: -c["files_changed"])


def _clean(obj):
    """NaN/Inf floats (DuckDB aggregates over NULLs) are invalid JSON and make
    PostgREST reject the whole stats payload - null them out recursively."""
    import math
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _table(rows, cols):
    if not rows:
        return "<p style='color:#9ca3af;font-size:12px'>none</p>"
    head = "".join("<th style='text-align:left;padding:3px 10px 3px 0;"
                   "font-size:11px;color:#6b7280'>%s</th>" % c for c in cols)
    body = "".join(
        "<tr>" + "".join("<td style='padding:2px 10px 2px 0;font-size:12px'>%s</td>"
                         % (r.get(c, "") if r.get(c) is not None else "")
                         for c in cols) + "</tr>"
        for r in rows[:20])
    return "<table><tr>%s</tr>%s</table>" % (head, body)


def _email_html(day, stats, narrative):
    s = stats
    parts = ["<div style='font-family:-apple-system,Segoe UI,Roboto,Arial,"
             "sans-serif;color:#1f2937;max-width:680px'>",
             "<h2 style='color:#1d683d'>Daily work activity — %s</h2>"
             % day.strftime("%A %d %B %Y"),
             narrative or "",
             "<h3>TaskHub — by person</h3>",
             _table(s["taskhub"]["people"],
                    ["person", "created", "updated", "completed", "comments"]),
             "<h3>Venue procedures completed — by person</h3>",
             _table(s["lake"]["procedures_by_person"],
                    ["person", "procedures", "venues"]),
             "<h3>Asana tasks completed — by person</h3>",
             _table(s["lake"]["asana_completed_by_person"],
                    ["person", "completed"]),
             "<h3>Xero activity (by type & entity — Xero has no per-user data)</h3>",
             _table(s["lake"]["xero_by_source"],
                    ["source", "org", "lines", "net"]),
             "<h3>Invoices touched</h3>",
             _table(s["lake"]["invoices_touched"], ["typ", "n", "total"]),
             "<h3>Restoke receipting — by venue</h3>",
             _table(s["lake"]["restoke_receipting"],
                    ["venue", "orders", "lines", "value"]),
             "<h3>Blob estate — what the systems produced</h3>",
             _table(s["blob"], ["container", "files_changed", "mb"]),
             "<p style='font-size:11px;color:#9ca3af'>Full detail on the "
             "Daily activity tab: %s/activity</p></div>"
             % os.environ.get("APP_URL", "")]
    return "".join(parts)


def run(report_date=None):
    today = dt.datetime.now(AEST).date()
    day = dt.date.fromisoformat(report_date) if report_date \
        else today - dt.timedelta(days=1)
    utc0, utc1 = _windows(day)
    lake_reader.sync(extra_tables=EXTRA_TABLES, log=LOG.info)
    stats = _clean({"lake": _lake_stats(day, utc0, utc1),
                    "taskhub": _taskhub_stats(utc0, utc1),
                    "blob": _blob_scan(day)})
    try:
        narrative = tk_ai.text(
            "You write the CFO's morning work-activity intelligence brief for "
            "Yo-Chi (AU frozen-yoghurt chain). From the JSON stats of "
            "EVERYTHING done yesterday (people's TaskHub actions, venue "
            "procedures and Asana completions by person, Xero postings by type/"
            "entity, invoice activity, Restoke receipting, and what the "
            "automated systems produced in the blob estate), return ONLY an "
            "HTML fragment: <h3>Headline</h3> 2-3 sentences on the shape of the "
            "day; <h3>People</h3> short <ul> naming the most active people and "
            "what they did; <h3>Watch</h3> short <ul> of anything unusual - "
            "quiet areas, missing activity, spikes. Specific and numeric, no "
            "padding. Never call award items 'breaches'.",
            json.dumps(stats, default=str)[:14000], max_tokens=1200)
    except Exception:
        LOG.exception("narrative failed")
        narrative = None
    row = {"report_date": day.isoformat(), "stats": stats, "narrative": narrative}
    if tk_db.get("work_reports", {"report_date": "eq." + day.isoformat(),
                                  "select": "report_date"}):
        tk_db.patch("work_reports",
                    {"report_date": "eq." + day.isoformat()},
                    {"stats": stats, "narrative": narrative})
    else:
        tk_db.insert("work_reports", [row])
    sent = 0
    admins = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                    "select": "email"})
    html = _email_html(day, stats, narrative)
    for a in admins:
        try:
            if tk_email.send(a["email"],
                             "Daily work activity — %s" % day.strftime("%d %b"),
                             html):
                sent += 1
        except Exception:
            LOG.exception("workreport email failed for %s", a["email"])
    return {"report_date": day.isoformat(), "emails": sent,
            "people": len(stats["taskhub"]["people"]),
            "blob_containers_changed": len(stats["blob"]),
            "procedures_people": len(stats["lake"]["procedures_by_person"])}
