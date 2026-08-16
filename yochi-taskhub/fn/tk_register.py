# -*- coding: utf-8 -*-
"""FY27 compliance register sync: reads Yochi_Group_Compliance_Calendar_FY27.xlsx
(Master Obligations, 155 rows) LIVE from SharePoint via Graph, refreshes the
lake copy (compliance_register - agents query it), and tops up dated TaskHub
tasks for the rolling window. Dedup = tasks.external_ref 'creg:REF:YYYY-MM'.
Monthly cat-4/5 obligations are skipped (TaskHub monthly templates cover them).
Runs monthly (1st, 06:05) + run_register."""
import calendar
import datetime as dt
import io
import json
import logging
import os
import re
import urllib.parse
import urllib.request

import tk_db

LOG = logging.getLogger("tk_register")

SP_HOST_PATH = "embraceyochi.sharepoint.com:/sites/YochiTeamSharepoint"
FILE_PATH = ("ALL-SHARES/Finance/GROUP STRUCTURE & CORPORATE/"
             "Entity Register - Aug 2026/00. Group/"
             "Yochi_Group_Compliance_Calendar_FY27.xlsx")
PETER = "818a4f97-84f7-41be-8c25-63126e4e411a"
P = lambda n: "aaaaaaaa-0000-0000-0000-0000000000%02d" % n
WINDOW_DAYS = 120
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _token():
    body = urllib.parse.urlencode({
        "client_id": os.environ["CLIENT_ID"],
        "client_secret": os.environ["CLIENT_SECRET"],
        "grant_type": "client_credentials",
        "scope": "https://graph.microsoft.com/.default"}).encode()
    req = urllib.request.Request(
        "https://login.microsoftonline.com/%s/oauth2/v2.0/token" % os.environ["TENANT_ID"],
        data=body)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["access_token"]


def _download():
    tok = _token()
    h = {"Authorization": "Bearer " + tok}
    req = urllib.request.Request(
        "https://graph.microsoft.com/v1.0/sites/%s" % SP_HOST_PATH, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        site_id = json.loads(r.read().decode())["id"]
    url = "https://graph.microsoft.com/v1.0/sites/%s/drive/root:/%s:/content" % (
        site_id, urllib.parse.quote(FILE_PATH))
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _parse(data):
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    ws = wb["Master Obligations"]
    rows = []
    for r in ws.iter_rows(min_row=5, values_only=True):
        if not r[0]:
            continue
        rows.append({k: str(v or "").strip() for k, v in zip(
            ("ref", "category", "obligation", "what", "authority", "frequency",
             "months_due", "applies_to", "contact", "owner", "risk", "notes"), r)})
    return rows


def _cat_to_project(catnum, applies):
    a = (applies or "").lower()
    return {1: P(17), 2: P(17), 3: P(2), 4: P(2), 5: P(2), 6: P(2),
            7: P(21), 8: P(22), 9: P(22), 10: P(14),
            11: P(12) if "texas" in a else P(11),
            12: P(13), 13: P(16), 14: P(16), 15: P(21), 16: P(21),
            17: P(21)}[catnum]


def _last_weekday(y, m):
    d = calendar.monthrange(y, m)[1]
    date = dt.date(y, m, d)
    while date.weekday() >= 5:
        date -= dt.timedelta(days=1)
    return date.day


def _due_day(catnum, text, y, m):
    t = text.lower()
    mm = re.search(r"~\s*(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", t)
    if mm:
        return int(mm.group(1))
    if "super" in t:
        return 28
    if catnum == 4 or "bas" in t or " ias" in t:
        return 21
    return _last_weekday(y, m)


def _parse_months(s):
    out = set()
    for m in re.finditer(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*[-']?\s*(\d{2,4})",
                         (s or "").lower()):
        y = int(m.group(2))
        out.add((2000 + y if y < 100 else y, MONTHS[m.group(1)]))
    return sorted(out)


def _lake_upload(rows):
    import duckdb
    from azure.storage.blob import BlobServiceClient
    tmp = "/tmp/compliance_register.parquet"
    con = duckdb.connect(":memory:")
    cols = list(rows[0].keys())
    con.execute("CREATE TABLE t (%s)" % ", ".join("%s VARCHAR" % c for c in cols))
    con.executemany("INSERT INTO t VALUES (%s)" % ",".join("?" * len(cols)),
                    [[x[c] for c in cols] for x in rows])
    con.execute("COPY t TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)" % tmp)
    con.close()
    svc = BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])
    cc = svc.get_container_client("datasights-lake")
    with open(tmp, "rb") as f:
        cc.upload_blob("tables/compliance_register/data.parquet", f.read(), overwrite=True)
    cc.upload_blob("tables/compliance_register/_meta.json", json.dumps({
        "table": "compliance_register", "rows": len(rows),
        "source": "Yochi_Group_Compliance_Calendar_FY27.xlsx / Master Obligations",
        "exported_at": dt.datetime.utcnow().isoformat()}), overwrite=True)


def run():
    rows = _parse(_download())
    try:
        _lake_upload(rows)
    except Exception:
        LOG.exception("lake upload failed (non-fatal)")
    today = dt.date.today()
    horizon = today + dt.timedelta(days=WINDOW_DAYS)
    tasks, skipped = [], 0
    for x in rows:
        m = re.match(r"(\d+)", x["category"])
        if not m:
            continue
        catnum = int(m.group(1))
        freq = x["frequency"].lower()
        months = _parse_months(x["months_due"])
        if "monthly" in freq or "weekly" in freq:
            if catnum in (4, 5) and "monthly" in freq:
                skipped += 1
                continue
            months = [((today + dt.timedelta(days=31 * i)).year,
                       (today + dt.timedelta(days=31 * i)).month) for i in range(5)]
        due_list = []
        for (y, mo) in sorted(set(months)):
            day = _due_day(catnum, x["obligation"] + " " + x["what"], y, mo)
            try:
                d = dt.date(y, mo, min(day, calendar.monthrange(y, mo)[1]))
            except ValueError:
                d = dt.date(y, mo, 28)
            if today <= d <= horizon:
                due_list.append(d)
        for d in due_list[:4]:
            desc = ("%s\n\nWhat: %s\nAuthority: %s\nFrequency: %s | Months due: %s\n"
                    "Applies to: %s\nOwner: %s\nRisk: %s\n\nCal ref %s — Master "
                    "Obligations tab, Yochi_Group_Compliance_Calendar_FY27.xlsx") % (
                x["obligation"], x["what"][:400], x["authority"], x["frequency"],
                x["months_due"], x["applies_to"], x["owner"], x["risk"], x["ref"])
            tasks.append({
                "project_id": _cat_to_project(catnum, x["applies_to"]),
                "title": "%s — %s (%s)" % (x["ref"], x["obligation"][:120],
                                           d.strftime("%b %y")),
                "description": desc,
                "priority": "high" if x["risk"].lower().startswith("high") else "medium",
                "assignee_id": PETER if ("peter" in x["owner"].lower()
                                         or "cfo" in x["owner"].lower()) else None,
                "due_date": d.isoformat(),
                "source": "manual",
                "external_ref": "creg:%s:%s" % (x["ref"], d.strftime("%Y-%m")),
            })
    inserted = tk_db.insert("tasks", tasks, on_conflict="external_ref",
                            ignore_duplicates=True, returning=True) if tasks else []
    return {"register_rows": len(rows), "window_tasks": len(tasks),
            "new_tasks": len(inserted or []), "skipped_template_covered": skipped}
