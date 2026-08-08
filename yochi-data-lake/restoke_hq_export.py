# -*- coding: utf-8 -*-
"""Extract the Restoke HQ **Procedure Report** (compliance) to Azure Blob + lake.

The Procedure Report is NOT in the Restoke Analytics API - it only exists in the HQ web
app (app.restoke.ai/hq) behind an interactive Django login, and its data comes from an
ASYNC job endpoint that returns a rendered HTML table (not JSON):

  1) GET /summary_grouping_report?method=get&restaurant_ids=<vid>&is_hq=<bool>
        &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&period=<days>
        &report_type=procedure&url=%2Fsummary_grouping_report&csrfmiddlewaretoken=<csrf>
     -> kicks off a job, returns a job id (fresh UUID each call)
  2) GET /get_job_result/<job_id>   (poll)
     -> {"status": <int>, "result": {"content": "<html table>"}}

Auth = Django session cookie + CSRF token, refreshed every run by _login() (the CSRF form
value equals the csrftoken cookie, which we reuse as the csrfmiddlewaretoken query param).

This module logs in, discovers venue ids from the HQ venue switcher, runs the procedure
report per venue for a rolling window, parses the HTML summary, and writes a venue-level
compliance snapshot to blob + a lake table/mart. History at venue x day is seeded from the
manual xlsx export (parse_xlsx) when present.

NB: the live endpoint is SUMMARY grain (per venue/department), not procedure-level detail.
Detail (~50k rows) only exists in the client-built xlsx -> use parse_xlsx / drop-and-ingest.

Config (app settings): RESTOKE_HQ_EMAIL, RESTOKE_HQ_PASSWORD (secrets),
  BLOB_CONNECTION_STRING, RESTOKE_HQ_START (default 2026-07-01),
  RESTOKE_HQ_WINDOW_DAYS (rolling window, default 7), RESTOKE_HQ_BASE, RESTOKE_HQ_IS_HQ.
"""
import datetime as dt
import io
import json
import logging
import os
import re
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from azure.storage.blob import BlobServiceClient, ContentSettings

LOG = logging.getLogger("restoke_hq_export")
BASE = os.environ.get("RESTOKE_HQ_BASE", "https://app.restoke.ai")
CONTAINER = os.environ.get("RESTOKE_HQ_CONTAINER", "restoke-hq-export")
START = os.environ.get("RESTOKE_HQ_START", "2026-07-01")
WINDOW_DAYS = int(os.environ.get("RESTOKE_HQ_WINDOW_DAYS", "7"))
POLL_TRIES = int(os.environ.get("RESTOKE_HQ_POLL_TRIES", "40"))
POLL_SLEEP = float(os.environ.get("RESTOKE_HQ_POLL_SLEEP", "1.5"))
HTTP_TIMEOUT = 120
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
XHR = {"accept": "application/json, text/javascript, */*; q=0.01",
       "x-requested-with": "XMLHttpRequest", "referer": BASE + "/hq/"}


# ------------------------------- auth -------------------------------

def _login():
    email = os.environ.get("RESTOKE_HQ_EMAIL")
    pw = os.environ.get("RESTOKE_HQ_PASSWORD")
    if not (email and pw):
        raise RuntimeError("RESTOKE_HQ_EMAIL / RESTOKE_HQ_PASSWORD not configured")
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    r = s.get(BASE + "/login/", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', r.text)
    if not m:
        raise RuntimeError("no csrfmiddlewaretoken on login page")
    s.post(BASE + "/login?next=/", data={"csrfmiddlewaretoken": m.group(1),
           "email": email, "password": pw}, headers={"Referer": BASE + "/login/"},
           timeout=HTTP_TIMEOUT, allow_redirects=True)
    chk = s.get(BASE + "/hq/", timeout=HTTP_TIMEOUT, allow_redirects=True)
    if "/login" in chk.url or 'name="csrfmiddlewaretoken"' in chk.text[:4000]:
        raise RuntimeError("Restoke HQ login failed (credentials / MFA / captcha)")
    s._hq_html = chk.text
    s._csrf = s.cookies.get("csrftoken", "")
    return s


def discover_venues(s=None):
    """Return {restaurant_id: name}. The HQ SPA renders the venue switcher client-side, so
    the raw HTML has no ids - instead use the Restoke Analytics API venue list (same
    restaurant_ids as the app; verified HQ=16303). Falls back to scraping switch_restaurant
    links if the analytics list is unavailable."""
    try:
        import restoke_export
        return {str(v["id"]): v["name"] for v in restoke_export.list_venues(include_test=False)}
    except Exception as e:
        LOG.warning("analytics venue list failed (%s); trying HQ html scrape", str(e)[:120])
    out = {}
    html = (getattr(s, "_hq_html", "") if s else "")
    for m in re.finditer(r'/switch_restaurant/(\d+)"[^>]*>([^<]{0,60})', html):
        vid, name = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
        if name.upper().startswith("YO-CHI") and not name.upper().startswith("TEST"):
            out[vid] = name
    return out


# ------------------------------- async job -------------------------------

def _kick(s, venue_id, start, end, is_hq):
    params = {"method": "get", "restaurant_ids": venue_id, "is_hq": str(bool(is_hq)),
              "start_date": start, "end_date": end, "period": (dt.date.fromisoformat(end)
              - dt.date.fromisoformat(start)).days + 1, "report_type": "procedure",
              "url": "/summary_grouping_report", "csrfmiddlewaretoken": s._csrf}
    r = s.get(BASE + "/summary_grouping_report", params=params, headers=XHR, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    body = r.text
    try:
        j = r.json()
        for k in ("job_id", "task_id", "id", "job", "result"):
            v = j.get(k) if isinstance(j, dict) else None
            if isinstance(v, str) and UUID_RE.search(v):
                return UUID_RE.search(v).group(0)
    except Exception:
        pass
    m = UUID_RE.search(body)
    if not m:
        raise RuntimeError("no job id in summary_grouping_report response: %s" % body[:300])
    return m.group(0)


def _poll_once(s, job_id):
    """One poll; return the HTML content if ready, else None."""
    r = s.get("%s/get_job_result/%s" % (BASE, job_id), headers=XHR, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        return None
    try:
        j = r.json()
    except Exception:
        return None
    res = j.get("result") if isinstance(j, dict) else None
    return res.get("content") if isinstance(res, dict) and res.get("content") else None


def _poll(s, job_id):
    for _ in range(POLL_TRIES):
        c = _poll_once(s, job_id)
        if c:
            return c
        time.sleep(POLL_SLEEP)
    raise RuntimeError("job %s did not complete after %d polls" % (job_id, POLL_TRIES))


def _parse_summary(content):
    """Parse the report response. It embeds a JSON string field:
      "summary": "completed,canceled,issues,completion_percent,non_completion_percent,issues_percent\\r\\n
                  <per-department rows>\\r\\n<GRAND TOTAL row>"
    The LAST data row is the venue grand total (the department rows sum to it), so we take
    that row directly - Restoke's own completion_percent is authoritative (avoids the
    double-count you get from summing dept rows + total)."""
    m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
    if not m:
        raise RuntimeError("no summary CSV in response; head=%s" % content[:250])
    csv = m.group(1).encode("utf-8").decode("unicode_escape")
    lines = [ln.strip() for ln in csv.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("summary CSV too short: %r" % csv[:200])
    header = [h.strip().lower() for h in lines[0].split(",")]
    total = [c.strip() for c in lines[-1].split(",")]  # grand-total row (last)

    def val(name, idx):
        for i, h in enumerate(header):
            if name in h and i < len(total):
                try:
                    return float(total[i])
                except Exception:
                    return 0.0
        return float(total[idx]) if idx < len(total) else 0.0

    return {"completed": int(val("completed", 0)), "cancelled": int(val("cancel", 1)),
            "with_issues": int(val("issue", 2)),
            "completion_pct": round(val("completion_percent", 3), 2),
            "departments": max(0, len(lines) - 2)}


# ------------------------------- blob io -------------------------------

def _svc():
    return BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"], max_single_put_size=8 * 1024 * 1024,
        max_block_size=8 * 1024 * 1024, connection_timeout=600, read_timeout=600)


def _upload(cc, path, data, ctype):
    last = None
    for a in range(4):
        try:
            cc.upload_blob(path, data, overwrite=True, max_concurrency=4,
                           content_settings=ContentSettings(content_type=ctype))
            return
        except Exception as e:
            last = e; time.sleep(4 * (a + 1))
    raise last


# ------------------------------- xlsx detail parser (seed / drop-ingest) -------------------------------

_DT_RE = re.compile(r"^\s*(\d{1,2}\s+\w{3}\s+\d{4})")


def _pdate(v):
    if v is None:
        return None
    m = _DT_RE.match(str(v))
    try:
        return dt.datetime.strptime(m.group(1), "%d %b %Y").date() if m else None
    except Exception:
        return None


def parse_xlsx(data):
    """Parse the manual Procedure Report xlsx detail sheet -> normalised rows (venue x day)."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = max(wb.worksheets, key=lambda w: (w.max_row or 0))
    out = []
    for r in ws.iter_rows(min_row=1, values_only=True):
        if not r or len(r) < 9 or not r[1]:
            continue
        venue = str(r[8] or "").strip(); status = str(r[2] or "").strip()
        if not venue.startswith("Yo-Chi") or status not in (
                "Completed", "Cancelled", "Completed with issues", "Created"):
            continue
        sd = _pdate(r[0])
        try:
            issues = int(float(r[6])) if r[6] not in (None, "") else 0
        except Exception:
            issues = 0
        out.append({"start_day": sd.isoformat() if sd else None, "venue": venue,
                    "procedure_name": str(r[1]).strip(), "status": status,
                    "completed_by": str(r[4] or "").strip(), "num_issues": issues,
                    "is_completed": 1 if status.startswith("Completed") else 0,
                    "is_cancelled": 1 if status == "Cancelled" else 0,
                    "has_issue": 1 if (issues > 0 or status == "Completed with issues") else 0})
    return out


# ------------------------------- run -------------------------------

def _windows(today, backfill=False, window_days=None):
    """Return [(label, start_iso, end_iso, day_iso)] - one **single-day** window per day.
    Restoke's report engine 500s on ranges beyond ~a week, so we can only pull a day at a
    time; D/WTD/MTD/YTD are then derived by summing days (in _register). Nightly refreshes
    the trailing RESTOKE_HQ_REFRESH_DAYS days (default 8, to catch late completions);
    backfill=True runs every day from RESTOKE_HQ_START (default 2026-07-01) to today."""
    start0 = dt.date.fromisoformat(os.environ.get("RESTOKE_HQ_START", "2026-07-01"))
    refresh = int(os.environ.get("RESTOKE_HQ_REFRESH_DAYS", "8"))
    start = start0 if backfill else max(start0, today - dt.timedelta(days=refresh))
    out, d = [], start
    while d <= today:
        out.append(("DAY", d.isoformat(), d.isoformat(), d.isoformat()))
        d += dt.timedelta(days=1)
    return out


def run(window_days=None, limit=None, raw=False, backfill=False):
    """Live: login -> per-venue procedure summary over short (D/WTD) + monthly windows ->
    compliance snapshot -> blob + lake mart. limit caps venues (testing). backfill=True runs
    every month of the year (one-time YTD history). raw=True returns first venue's HTML."""
    if raw:
        s = _login()
        venues = discover_venues(s)
        vid, name = next(iter(venues.items()))
        today = dt.date.today()
        is_hq = os.environ.get("RESTOKE_HQ_IS_HQ", "0") == "1"
        # compare two same-length windows at different dates to see if dates are honoured
        probe = os.environ.get("RESTOKE_HQ_PROBE", "")
        out = {"debug_venue": name, "vid": vid}
        if probe:
            for label, (a, b) in {"pastweek": ("2026-07-01", "2026-07-07"),
                                  "curweek": ("2026-07-29", "2026-08-04")}.items():
                try:
                    tot = _parse_summary(_poll(s, _kick(s, vid, a, b, is_hq)))
                    out[label] = {"window": [a, b], **tot}
                except Exception as e:
                    out[label] = {"error": str(e)[:150]}
            return out
        win = window_days or WINDOW_DAYS
        start = (today - dt.timedelta(days=win)).isoformat(); end = today.isoformat()
        html = _poll(s, _kick(s, vid, start, end, is_hq))
        return {**out, "window": [start, end], "html_len": len(html), "html": html[:6000]}
    svc = _svc()
    try:
        svc.create_container(CONTAINER)
    except Exception:
        pass
    cc = svc.get_container_client(CONTAINER)

    today = dt.date.today()
    win = window_days or WINDOW_DAYS
    start = (today - dt.timedelta(days=win)).isoformat()
    end = today.isoformat()
    is_hq = os.environ.get("RESTOKE_HQ_IS_HQ", "0") == "1"

    s = _login()
    diag = {}
    try:
        import restoke_export
        alist = restoke_export.list_venues(include_test=False)
        diag["analytics_venues"] = len(alist)
    except Exception as e:
        diag["analytics_error"] = str(e)[:300]
    venues = discover_venues(s)
    diag["discovered"] = len(venues)
    LOG.info("restoke-hq: logged in, %d venues discovered %s", len(venues), diag)
    if limit:
        venues = dict(list(venues.items())[:int(limit)])
    rows, errors = [], []

    def run_window(pstart, pend, label, day):
        # Phase 1 - kick every venue's job (they generate in parallel server-side)
        pending = {}
        for vid, name in venues.items():
            try:
                pending[vid] = (name, _kick(s, vid, pstart, pend, is_hq))
            except Exception as e:
                errors.append("%s %s kick: %s" % (day, name, str(e)[:120]))
        # Phase 2 - poll all pending round-robin until done or budget hit
        deadline = time.time() + POLL_TRIES * POLL_SLEEP + 30
        while pending and time.time() < deadline:
            for vid in list(pending):
                name, job_id = pending[vid]
                try:
                    c = _poll_once(s, job_id)
                except Exception:
                    c = None
                if c:
                    try:
                        tot = _parse_summary(c)
                        rows.append({"venue_id": vid, "venue": name, "day": day,
                                     "snapshot_date": today.isoformat(), **tot})
                    except Exception as e:
                        errors.append("%s %s parse: %s" % (day, name, str(e)[:120]))
                    del pending[vid]
            if pending:
                time.sleep(POLL_SLEEP)
        for vid, (name, job_id) in pending.items():
            errors.append("%s %s(%s): job did not complete in budget" % (day, name, vid))

    windows = _windows(today, backfill=backfill, window_days=window_days)
    LOG.info("restoke-hq: %d day-windows (%s..%s)", len(windows),
             windows[0][3] if windows else "-", windows[-1][3] if windows else "-")
    for label, ps, pe, day in windows:
        run_window(ps, pe, label, day)

    snap = ("backfill_" if backfill else "") + today.isoformat()
    summary = {"source": "Restoke HQ Procedure Report (live async)", "backfill": backfill,
               "windows": ["%s%s" % (w[0], "/" + w[3] if w[3] else "") for w in windows],
               "venues": len(venues), "ok": len(rows), "diag": diag, "errors": errors[:20]}
    if rows:
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
        _upload(cc, "snapshot/%s.parquet" % snap, buf.getvalue(), "application/octet-stream")
        _upload(cc, "snapshot/latest.parquet", buf.getvalue(), "application/octet-stream")
        try:
            summary["lake"] = _register(svc, cc)
        except Exception as e:
            errors.append("register: %s" % str(e)[:200])
    _upload(cc, "catalog.json", json.dumps({**summary, "generated_at":
            dt.datetime.now().isoformat()}, indent=1), "application/json")
    summary["finished_at"] = dt.datetime.now().isoformat()
    return summary


def _register(svc, cc):
    """Build the daily backbone (mart_procedure_daily, venue x day) from all snapshots, then
    derive D/WTD/MTD/YTD (mart_procedure_compliance) by summing days. Daily rows are deduped
    to the latest snapshot_date per (venue, day) so late-completion refreshes win."""
    import glob
    import tempfile

    import duckdb
    dl = svc.get_container_client("datasights-lake")
    with tempfile.TemporaryDirectory() as tmp:
        sdir = os.path.join(tmp, "snap"); os.makedirs(sdir)
        for b in cc.list_blobs(name_starts_with="snapshot/"):
            if b.name.endswith(".parquet") and "latest" not in b.name:
                with open(os.path.join(sdir, os.path.basename(b.name)), "wb") as f:
                    f.write(cc.download_blob(b.name).readall())
        if not glob.glob(os.path.join(sdir, "*.parquet")):
            return {}
        con = duckdb.connect(":memory:")
        gp = os.path.join(sdir, "*.parquet").replace("\\", "/")
        # daily backbone: latest refresh per (venue, day)
        con.execute("""CREATE TABLE daily AS SELECT * EXCLUDE(_rn) FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY venue, day
                ORDER BY snapshot_date DESC) AS _rn
            FROM read_parquet('%s', union_by_name=true)) WHERE _rn = 1""" % gp)
        # D/WTD/MTD/YTD derived by summing days, relative to the latest day in the data
        con.execute("""CREATE TABLE comp AS
            WITH ref AS (SELECT MAX(day::DATE) md FROM daily),
                 x AS (
                   SELECT d.venue_id, d.venue, p.period, d.completed, d.cancelled, d.with_issues
                   FROM daily d, ref, LATERAL (VALUES
                       ('D',   d.day::DATE = ref.md),
                       ('WTD', d.day::DATE >= date_trunc('week',  ref.md) AND d.day::DATE <= ref.md),
                       ('MTD', d.day::DATE >= date_trunc('month', ref.md) AND d.day::DATE <= ref.md),
                       ('YTD', d.day::DATE >= date_trunc('year',  ref.md) AND d.day::DATE <= ref.md)
                   ) p(period, incl) WHERE p.incl)
            SELECT venue_id, venue, period,
                   SUM(completed) completed, SUM(cancelled) cancelled, SUM(with_issues) with_issues,
                   ROUND(100.0*SUM(completed)/NULLIF(SUM(completed)+SUM(cancelled),0),2) completion_pct
            FROM x GROUP BY venue_id, venue, period""")
        res = {}
        for tbl, note in [
            ("mart_procedure_daily",
             "Restoke procedure compliance per venue per DAY (live HQ scrape backbone; the "
             "report engine only allows ~1-day ranges). Latest refresh per (venue, day). "
             "completion_pct is Restoke's own figure; completed/cancelled are checklist-item "
             "counts. Sum days for any period. Venues with completed=0 & cancelled>0 aren't "
             "recording in Restoke."),
            ("mart_procedure_compliance",
             "Restoke procedure compliance rolled up to period per venue, derived from "
             "mart_procedure_daily relative to the latest data day. period in "
             "(D=latest day, WTD=Mon-to-date, MTD=month-to-date, YTD=year-to-date). "
             "completion_pct = 100*completed/(completed+cancelled)."),
        ]:
            src = "daily" if tbl.endswith("daily") else "comp"
            out = os.path.join(tmp, tbl + ".parquet")
            con.execute("COPY %s TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)" % (src, out.replace("\\", "/")))
            cols = [r[0] for r in con.execute("DESCRIBE %s" % src).fetchall()]
            nrows = con.execute("SELECT COUNT(*) FROM %s" % src).fetchone()[0]
            with open(out, "rb") as f:
                _upload(dl, "tables/%s/data.parquet" % tbl, f.read(), "application/octet-stream")
            _upload(dl, "tables/%s/_meta.json" % tbl, json.dumps({
                "view": tbl, "mode": "snapshot", "files": ["data.parquet"], "columns": cols,
                "rows": nrows, "source_rows": nrows,
                "date_col": "day" if src == "daily" else None, "note": note,
                "exported_at": dt.datetime.now().isoformat()}, indent=1), "application/json")
            res[tbl] = nrows
        con.close()
    import lake_export
    n = lake_export.rebuild_catalog(dl)
    return {"tables": res, "catalog_tables": n}
