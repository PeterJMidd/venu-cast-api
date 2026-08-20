# -*- coding: utf-8 -*-
"""Lake Brain - additive intelligence layer over the Yo-Chi data lake.

Does four things every morning (07:00 AEST, after the nightly chain ends 06:20):
  1. INTEGRITY  - verifies the overnight refresh: per-stage heartbeats (lake sync, restoke,
                  opcentral, doc index), per-table freshness + row deltas vs yesterday,
                  module error rollups. Status GREEN / AMBER / RED.
  2. CONTENTS   - scans the catalog and groups all tables into business THEMES, writing a
                  "map of the lake" (lake-brain/contents.md + .json) and registering it as
                  a queryable lake table `lake_contents` so the agent can answer
                  "what data do we have about X?" natively.
  3. LEARNING   - keeps lake-brain/knowledge.json: signals become QUESTIONS for the owner
                  (max 3 open). The morning email carries one-click answer links
                  (yes / no / ignore -> /api/brain_answer). Answers become LEARNINGS that
                  feed back into the contents map - so the lake gets smarter with each reply.
  4. REPORT     - emails the morning brief (Graph sendMail, same app registration the
                  SharePoint mirror already uses).

STRICTLY ADDITIVE: writes only to the new `lake-brain` container + one new lake table
(tables/lake_contents/). Never modifies existing tables, modules or data.

Config (app settings): BLOB_CONNECTION_STRING, TENANT_ID/CLIENT_ID/CLIENT_SECRET (exist),
  BRAIN_RECIPIENT, BRAIN_SENDER (default peterm@yochi.com.au), BRAIN_ANSWER_BASE
  (public brain_answer URL incl ?code=), LAKE_BRAIN_CRON.
"""
import datetime as dt
import hashlib
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request

from azure.storage.blob import BlobServiceClient, ContentSettings

LOG = logging.getLogger("lake_brain")
BRAIN = "lake-brain"
RECIPIENT = os.environ.get("BRAIN_RECIPIENT", "peterm@yochi.com.au")
SENDER = os.environ.get("BRAIN_SENDER", "peterm@yochi.com.au")

THEMES = [
    ("Sales & Trading",        ("sales", "netsales", "transaction", "member", "redcat", "servings", "discount", "surcharge")),
    ("Purchasing & Supply",    ("restoke_", "invoice", "purchas", "order", "supplier", "recipe", "unmatched")),
    ("Labour & Rostering",     ("labor", "labour", "roster", "tanda", "shift", "timesheet")),
    ("Financial (Xero)",       ("xero", "budget", "pl_", "profit", "gl_", "account", "trial")),
    ("Compliance & Food Safety", ("compliance", "procedure", "audit", "foodsafety", "food_safety", "safety", "temperature")),
    ("People & Training",      ("opc_", "opcentral", "training", "turnover", "staff", "employee", "user")),
    ("Reviews & Guest",        ("review", "mystery", "guest", "feedback", "nps")),
    ("Curated Marts",          ("mart_",)),
    ("External Feeds & Economy", ("feed_",)),
    ("Workflow & Systems",     ("asana", "stripe", "celsi", "sharepoint", "organisation",
                                "venue_master", "lineitems", "agent_playbook", "taskhub")),
    ("Knowledge & Meta",       ("lake_contents", "doc", "catalog")),
]


# ---------------------------------------------------------------- blob helpers

def _svc():
    return BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"],
        max_single_put_size=8 * 1024 * 1024, max_block_size=8 * 1024 * 1024,
        connection_timeout=600, read_timeout=600)


def _upload(cc, path, data, ctype="application/json"):
    last = None
    for a in range(4):
        try:
            cc.upload_blob(path, data, overwrite=True,
                           content_settings=ContentSettings(content_type=ctype))
            return
        except Exception as e:
            last = e; time.sleep(3 * (a + 1))
    raise last


def _read_json(cc, path, default=None):
    try:
        return json.loads(cc.download_blob(path).readall())
    except Exception:
        return default


def _last_modified(cc, path):
    try:
        return cc.get_blob_client(path).get_blob_properties().last_modified
    except Exception:
        return None


# ---------------------------------------------------------------- graph email

def _token():
    data = urllib.parse.urlencode({
        "client_id": os.environ["CLIENT_ID"], "client_secret": os.environ["CLIENT_SECRET"],
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        "https://login.microsoftonline.com/%s/oauth2/v2.0/token" % os.environ["TENANT_ID"],
        data=data)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def _send_email(subject, html):
    payload = json.dumps({"message": {
        "subject": subject,
        "body": {"contentType": "HTML", "content": html},
        "toRecipients": [{"emailAddress": {"address": RECIPIENT}}]},
        "saveToSentItems": False}).encode()
    req = urllib.request.Request(
        "https://graph.microsoft.com/v1.0/users/%s/sendMail" % SENDER, data=payload,
        headers={"Authorization": "Bearer " + _token(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status in (200, 202)


# ---------------------------------------------------------------- 1. integrity

def check_integrity(svc):
    now = dt.datetime.now(dt.timezone.utc)
    dl = svc.get_container_client("datasights-lake")
    br = svc.get_container_client(BRAIN)
    cat = _read_json(dl, "catalog/catalog.json", {}) or {}
    tables = [t for t in cat.get("tables", [])
              if isinstance(t, dict) and (t.get("table") or t.get("view"))]
    for t in tables:  # some entries use 'view' instead of 'table'
        t.setdefault("table", t.get("view"))

    def age_h(ts):
        if not ts:
            return None
        try:
            t = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=dt.timezone.utc)
            return round((now - t).total_seconds() / 3600, 1)
        except Exception:
            return None

    # stage heartbeats = last write time of each pipeline's output.
    # NB: lake-sync uses max per-table exported_at, NOT catalog.json's timestamp -
    # the brain itself rebuilds the catalog daily, which would mask a failed sync.
    stages = {}
    # anchor on the big fact views (>100k rows) - they are refreshed by lake_export
    # every night, so their newest exported_at proves the dataSights sync itself ran
    sync_ts = [str(t.get("exported_at", "")) for t in tables
               if (t.get("rows") or 0) > 100000 and t.get("exported_at")]
    latest_sync = max(sync_ts) if sync_ts else None
    stages["Lake sync (dataSights)"] = {"last_run": latest_sync,
                                        "age_h": age_h(latest_sync)}
    for label, cont, path in [
        ("Restoke export", "restoke-export", "catalog.json"),
        ("OpCentral export", "opcentral-export", "catalog.json"),
    ]:
        lm = _last_modified(svc.get_container_client(cont), path)
        stages[label] = {"last_run": lm.isoformat() if lm else None,
                         "age_h": age_h(lm.isoformat()) if lm else None}
    # doc index heartbeat = newest index file (generous listing bound)
    try:
        docs = svc.get_container_client("docs-lake")
        newest = None
        for i, b in enumerate(docs.list_blobs(name_starts_with="index/")):
            if newest is None or b.last_modified > newest:
                newest = b.last_modified
            if i > 20000:
                break
        stages["Doc index"] = {"last_run": newest.isoformat() if newest else None,
                               "age_h": age_h(newest.isoformat()) if newest else None}
    except Exception:
        stages["Doc index"] = {"last_run": None, "age_h": None}

    # per-table freshness + deltas vs yesterday's snapshot
    prev = _read_json(br, "integrity/latest.json", {}) or {}
    prev_rows = {t["table"]: t.get("rows") for t in prev.get("tables", [])}
    fresh24, changed, anomalies, stale7 = 0, [], [], []
    for t in tables:
        name, rows = t.get("table"), t.get("rows")
        a = age_h(t.get("exported_at"))
        if a is not None and a <= 36:
            fresh24 += 1
        if a is not None and a > 168:
            stale7.append(name)
        p = prev_rows.get(name)
        if p is not None and rows is not None and p != rows:
            changed.append({"table": name, "from": p, "to": rows})
            if p > 200 and rows < p * 0.8:
                anomalies.append("%s shrank %d -> %d rows" % (name, p, rows))
    gone = [n for n in prev_rows if n not in {t.get("table") for t in tables}]
    for n in gone:
        anomalies.append("table disappeared: %s" % n)

    # module error rollups - only from catalogs written in the last 26h (fresh runs);
    # an old catalog's errors were already reported the day they happened
    errors = []
    for label, cont in [("restoke", "restoke-export"), ("opcentral", "opcentral-export")]:
        c = _read_json(svc.get_container_client(cont), "catalog.json", {}) or {}
        gen_age = age_h(c.get("generated_at"))
        if gen_age is not None and gen_age <= 26:
            for e in (c.get("errors") or [])[:5]:
                errors.append("%s: %s" % (label, str(e)[:140]))

    lake_age = stages["Lake sync (dataSights)"]["age_h"]
    if lake_age is None or lake_age > 30 or len(anomalies) > 10:
        status = "RED"
    elif anomalies or errors or any(
            s["age_h"] and s["age_h"] > 30 for s in stages.values()):
        status = "AMBER"
    else:
        status = "GREEN"

    return {"date": now.date().isoformat(), "generated_at": now.isoformat(),
            "status": status, "stages": stages, "table_count": len(tables),
            "fresh_24h": fresh24, "changed": changed[:40], "changed_count": len(changed),
            "anomalies": anomalies[:15], "stale_7d": stale7[:25],
            "stale_7d_count": len(stale7), "module_errors": errors[:10],
            "tables": [{"table": t.get("table"), "rows": t.get("rows")} for t in tables]}


# ---------------------------------------------------------------- 2. contents

def _theme_of(name):
    n = (name or "").lower()
    for theme, keys in THEMES:
        if any(k in n for k in keys):
            return theme
    return "Reference & Other"


def build_contents(svc, integrity, learnings):
    dl = svc.get_container_client("datasights-lake")
    br = svc.get_container_client(BRAIN)
    cat = _read_json(dl, "catalog/catalog.json", {}) or {}
    rows_out, themes = [], {}
    for t in cat.get("tables", []):
        if not isinstance(t, dict):
            continue
        t["table"] = t.get("table") or t.get("view")
        if not t["table"]:
            continue
        theme = _theme_of(t.get("table"))
        themes.setdefault(theme, []).append(t)
        rows_out.append({"table": t.get("table"), "theme": theme,
                         "rows": t.get("rows"), "exported_at": str(t.get("exported_at", ""))[:19],
                         "source": str(t.get("source", ""))[:120]})

    md = ["# Yo-Chi Data Lake - Contents Map",
          "_Auto-built daily by Lake Brain. %d tables. Generated %s._\n"
          % (len(rows_out), dt.date.today().isoformat())]
    for theme, ts in sorted(themes.items(), key=lambda kv: -len(kv[1])):
        total = sum(x.get("rows") or 0 for x in ts)
        md.append("## %s  (%d tables, %s rows)" % (theme, len(ts), f"{total:,}"))
        for x in sorted(ts, key=lambda z: -(z.get("rows") or 0))[:15]:
            md.append("- `%s` - %s rows" % (x.get("table"), f"{(x.get('rows') or 0):,}"))
        if len(ts) > 15:
            md.append("- ... and %d more" % (len(ts) - 15))
        md.append("")
    if learnings:
        md.append("## Owner guidance (learned)")
        for l in learnings[-20:]:
            md.append("- " + l)
    contents_md = "\n".join(md)
    _upload(br, "contents.md", contents_md, "text/markdown")
    _upload(br, "contents.json", json.dumps(rows_out, indent=1))

    # register as a queryable lake table (additive)
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        df = pd.DataFrame(rows_out)
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
        _upload(dl, "tables/lake_contents/data.parquet", buf.getvalue(),
                "application/octet-stream")
        _upload(dl, "tables/lake_contents/_meta.json", json.dumps({
            "view": "lake_contents", "mode": "snapshot", "files": ["data.parquet"],
            "columns": list(df.columns), "rows": len(df), "source_rows": len(df),
            "date_col": None,
            "note": "Map of the lake: every table classified into a business theme with "
                    "row counts and freshness. Use to answer 'what data do we have about X'. "
                    "Rebuilt daily by Lake Brain.",
            "exported_at": dt.datetime.now().isoformat()}, indent=1))
        import lake_export
        lake_export.rebuild_catalog(dl)
    except Exception:
        LOG.exception("lake_contents registration failed (non-fatal)")
    return {"themes": {k: len(v) for k, v in themes.items()}, "md_bytes": len(contents_md)}


# ---------------------------------------------------------------- 3. learning

def _qid(text):
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def manage_questions(svc, integrity):
    br = svc.get_container_client(BRAIN)
    kn = _read_json(br, "knowledge.json", {"questions": [], "learnings": []})
    seen = {q["id"] for q in kn["questions"]}
    open_qs = [q for q in kn["questions"] if q["status"] == "open"]

    def propose(text, context, learn_yes, learn_no):
        qid = _qid(text)
        if qid in seen or len(open_qs) >= 3:
            return
        q = {"id": qid, "question": text, "context": context, "status": "open",
             "asked_at": dt.datetime.now().isoformat(),
             "learn_yes": learn_yes, "learn_no": learn_no}
        kn["questions"].append(q); open_qs.append(q); seen.add(qid)

    # signal-driven questions
    for a in integrity.get("anomalies", [])[:2]:
        propose("Anomaly: %s. Is this expected?" % a,
                "Row-count anomaly detected by the integrity scan.",
                "Anomaly '%s' is expected behaviour - don't flag similar shrinks for this table." % a[:60],
                "Anomaly '%s' was a real problem - keep flagging aggressively." % a[:60])
    if integrity.get("stale_7d_count", 0) > 0:
        sample = ", ".join(integrity["stale_7d"][:5])
        propose("%d tables haven't refreshed in over a week (e.g. %s). Are these static/one-off tables I should stop watching?"
                % (integrity["stale_7d_count"], sample),
                "Stale tables dilute the daily freshness signal.",
                "Tables stale >7d are mostly static reference/one-off tables - exclude from freshness alerts.",
                "Stale tables may indicate broken feeds - keep alerting on them.")
    # curated improvement seeds (asked once, ever)
    propose("opc_users carries staff DOB/home addresses readable by all agent logins. Mask those columns in the agent-facing table?",
            "OPCENTRAL_MASK_USER_PII=1 masks the lake copy; raw JSON stays complete in opcentral-export.",
            "Owner wants staff PII masked in the lake - set OPCENTRAL_MASK_USER_PII=1 on the next run.",
            "Owner accepts unmasked PII in the lake for head-office users.")
    propose("Want a weekly 'lake usage & value' section in this report (which themes/tables the agent actually queries)?",
            "Would require logging agent query table-names (additive) to measure what's used.",
            "Owner wants usage analytics - build additive agent query logging next.",
            "Owner doesn't need usage analytics for now.")

    _upload(br, "knowledge.json", json.dumps(kn, indent=1))
    return kn


def record_answer(qid, answer):
    """Called by /api/brain_answer. Records the answer + converts it into a learning."""
    svc = _svc()
    br = svc.get_container_client(BRAIN)
    kn = _read_json(br, "knowledge.json", {"questions": [], "learnings": []})
    for q in kn["questions"]:
        if q["id"] == qid and q["status"] == "open":
            q["status"] = "answered" if answer in ("yes", "no") else "ignored"
            q["answer"] = answer
            q["answered_at"] = dt.datetime.now().isoformat()
            if answer == "yes":
                kn["learnings"].append(q.get("learn_yes") or ("YES: " + q["question"]))
            elif answer == "no":
                kn["learnings"].append(q.get("learn_no") or ("NO: " + q["question"]))
            _upload(br, "knowledge.json", json.dumps(kn, indent=1))
            return {"ok": True, "question": q["question"], "recorded": answer}
    return {"ok": False, "error": "question not found or already answered"}


# ---------------------------------------------------------------- 4. report

def _fmt_age(h):
    if h is None:
        return "no data"
    return "%.0f min ago" % (h * 60) if h < 1 else "%.1f h ago" % h


def build_email(integrity, contents, kn):
    colors = {"GREEN": "#1C8F58", "AMBER": "#B77C16", "RED": "#C0392B"}
    c = colors[integrity["status"]]
    base = os.environ.get("BRAIN_ANSWER_BASE", "")
    h = ["<div style='font-family:Segoe UI,Arial,sans-serif;max-width:680px;margin:auto;color:#172029'>",
         "<div style='background:%s;color:#fff;padding:14px 18px;border-radius:10px 10px 0 0'>" % c,
         "<div style='font-size:12px;opacity:.85'>YO-CHI LAKE BRAIN &middot; %s</div>" % integrity["date"],
         "<div style='font-size:22px;font-weight:700'>Overnight refresh: %s</div>" % integrity["status"],
         "<div style='font-size:13px;opacity:.9'>%d tables &middot; %d refreshed in 24h &middot; %d changed &middot; %d anomalies</div>"
         % (integrity["table_count"], integrity["fresh_24h"], integrity["changed_count"],
            len(integrity["anomalies"])),
         "</div><div style='border:1px solid #DBE2E9;border-top:none;padding:16px 18px;border-radius:0 0 10px 10px'>"]
    h.append("<h3 style='margin:2px 0 8px;font-size:15px'>Pipeline stages</h3><table style='width:100%;border-collapse:collapse;font-size:13px'>")
    for k, v in integrity["stages"].items():
        ok = v["age_h"] is not None and v["age_h"] <= 30
        h.append("<tr><td style='padding:4px 0;border-bottom:1px solid #EEF2F6'>%s</td>"
                 "<td style='text-align:right;color:%s;font-weight:600'>%s %s</td></tr>"
                 % (k, "#1C8F58" if ok else "#B77C16", "&#10003;" if ok else "&#9888;",
                    _fmt_age(v["age_h"])))
    h.append("</table>")
    sh = integrity.get("self_heal", {}).get("doc_extract")
    if sh:
        h.append("<div style='background:#E1F1E9;border:1px solid #B7DCC8;border-radius:8px;"
                 "padding:9px 12px;margin:10px 0;font-size:12.5px'><b>Self-heal:</b> doc index "
                 "was stale, so I ran an extract pass now - %s files indexed, %s skipped, "
                 "%s errors (%ss).</div>"
                 % (sh.get("files_indexed"), sh.get("skipped_unchanged"),
                    sh.get("errors"), sh.get("secs")))
    if integrity["anomalies"]:
        h.append("<h3 style='margin:14px 0 6px;font-size:15px;color:#C0392B'>Anomalies</h3><ul style='margin:4px 0;font-size:13px'>")
        h += ["<li>%s</li>" % a for a in integrity["anomalies"]]
        h.append("</ul>")
    if integrity["module_errors"]:
        h.append("<h3 style='margin:14px 0 6px;font-size:15px;color:#B77C16'>Module errors</h3><ul style='margin:4px 0;font-size:12.5px'>")
        h += ["<li>%s</li>" % e for e in integrity["module_errors"]]
        h.append("</ul>")
    big = sorted(integrity["changed"], key=lambda x: -abs(x["to"] - x["from"]))[:6]
    if big:
        h.append("<h3 style='margin:14px 0 6px;font-size:15px'>Biggest movements</h3><table style='width:100%;border-collapse:collapse;font-size:12.5px'>")
        for m in big:
            d = m["to"] - m["from"]
            h.append("<tr><td style='padding:3px 0;border-bottom:1px solid #EEF2F6'><code>%s</code></td>"
                     "<td style='text-align:right'>%+d rows (%s &rarr; %s)</td></tr>"
                     % (m["table"], d, f"{m['from']:,}", f"{m['to']:,}"))
        h.append("</table>")
    th = contents.get("themes", {})
    h.append("<h3 style='margin:14px 0 6px;font-size:15px'>Lake by theme</h3><div style='font-size:12.5px;line-height:1.9'>")
    h.append(" &middot; ".join("<b>%s</b> %d" % (k, v) for k, v in
                               sorted(th.items(), key=lambda kv: -kv[1])))
    h.append("</div>")
    open_qs = [q for q in kn["questions"] if q["status"] == "open"]
    if open_qs and base:
        h.append("<h3 style='margin:16px 0 6px;font-size:15px'>Help me learn (one click)</h3>")
        for q in open_qs:
            link = lambda a: "%s&id=%s&a=%s" % (base, q["id"], a)
            h.append("<div style='background:#F6F8FB;border:1px solid #DBE2E9;border-radius:8px;padding:10px 12px;margin:7px 0;font-size:13px'>"
                     "%s<br><span style='font-size:11.5px;color:#5B6774'>%s</span><br>"
                     "<a href='%s' style='color:#0C7C8C;font-weight:700'>Yes</a> &nbsp;|&nbsp; "
                     "<a href='%s' style='color:#C94E28;font-weight:700'>No</a> &nbsp;|&nbsp; "
                     "<a href='%s' style='color:#7C8996'>Ignore</a></div>"
                     % (q["question"], q.get("context", ""), link("yes"), link("no"), link("ignore")))
    n_learn = len(kn.get("learnings", []))
    h.append("<div style='margin-top:14px;font-size:11.5px;color:#7C8996'>Learnings accumulated: %d "
             "&middot; contents map: lake_contents table &middot; ask the agent anything at "
             "<a href='https://yochi-lake-agent.azurewebsites.net'>yochi-lake-agent</a></div>" % n_learn)
    h.append("</div></div>")
    return "".join(h)


# ---------------------------------------------------------------- orchestrate

def run(send=True):
    svc = _svc()
    try:
        svc.create_container(BRAIN)
    except Exception:
        pass
    br = svc.get_container_client(BRAIN)
    t0 = time.time()
    integrity = check_integrity(svc)

    # SELF-HEAL: if the doc index is stale (>26h) while the mirror is alive, run a
    # bounded extract right now instead of just reporting it. The stall of Jul-17
    # (nightly_doc_extract producing nothing for a month) is exactly this case.
    healed = None
    doc_age = (integrity["stages"].get("Doc index") or {}).get("age_h")
    if doc_age is not None and doc_age > 26:
        try:
            import doc_extract
            healed = doc_extract.run_extract(
                minutes=int(os.environ.get("BRAIN_HEAL_EXTRACT_MIN", "25")), prefix=None)
            integrity["self_heal"] = {"doc_extract": healed}
            LOG.info("brain self-heal doc_extract: %s", healed)
        except Exception:
            LOG.exception("brain self-heal doc_extract failed")
    kn = manage_questions(svc, integrity)
    contents = build_contents(svc, integrity, kn.get("learnings", []))
    _upload(br, "integrity/%s.json" % integrity["date"], json.dumps(integrity, indent=1))
    _upload(br, "integrity/latest.json", json.dumps(integrity, indent=1))
    html = build_email(integrity, contents, kn)
    _upload(br, "report/%s.html" % integrity["date"], html, "text/html")
    sent = False
    if send:
        try:
            sent = _send_email("Yo-Chi Lake Brain - %s - %s" %
                               (integrity["status"], integrity["date"]), html)
        except Exception:
            LOG.exception("brain email failed")
    return {"status": integrity["status"], "tables": integrity["table_count"],
            "fresh_24h": integrity["fresh_24h"], "anomalies": len(integrity["anomalies"]),
            "themes": contents.get("themes"), "open_questions":
            len([q for q in kn["questions"] if q["status"] == "open"]),
            "learnings": len(kn.get("learnings", [])), "email_sent": sent,
            "secs": round(time.time() - t0)}
