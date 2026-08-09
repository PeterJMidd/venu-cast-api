# -*- coding: utf-8 -*-
"""Group daily net-revenue budget for the cockpit, live from the maintained
DIMENSIONS workbook (artiqbi OneDrive, via Graph client-credentials - same
source as the v2 decks; NET_REVENUE = servings + retail - discounts, ex-GST,
the same basis as mart actuals). Trimmed from yochi-venue-build/budget.py:
group totals only, no venue matching. Never raises - {} on any failure."""
import datetime
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request

LOG = logging.getLogger("tk_budget")

BUDGET_DRIVE_ID = os.environ.get(
    "BUDGET_DRIVE_ID",
    "b!O-ZiHsS0WUqEqYG68HkwOic8B2y3FflPqvJI8FASf5mZ0Tr9c3hPRLVdSaAW-DZz")
BUDGET_PATH = os.environ.get("BUDGET_PATH", "DIMENSIONS/JULY_DAILY_BUDGET.xlsx")
BUDGET_SHEET = os.environ.get("BUDGET_SHEET", "DAILY SALES ")

_GRAPH = "https://graph.microsoft.com/v1.0"
_DATE_FMTS = ("%A, %d %B %Y", "%d %B %Y", "%A, %B %d, %Y", "%B %d, %Y",
              "%d/%m/%Y", "%Y-%m-%d")
_CACHE = {"at": 0.0, "days": {}}


def _token():
    body = urllib.parse.urlencode({
        "client_id": os.environ["CLIENT_ID"],
        "client_secret": os.environ["CLIENT_SECRET"],
        "grant_type": "client_credentials",
        "scope": "https://graph.microsoft.com/.default",
    }).encode()
    req = urllib.request.Request(
        "https://login.microsoftonline.com/%s/oauth2/v2.0/token" % os.environ["TENANT_ID"],
        data=body)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["access_token"]


def _pdate(d):
    if hasattr(d, "date"):
        return d.date().isoformat()
    s = str(d).strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return s[:10]


def daily_group(max_age=21600):
    """{iso_date: group budget net} - cached for 6h per instance."""
    if _CACHE["days"] and time.time() - _CACHE["at"] < max_age:
        return _CACHE["days"]
    import openpyxl
    try:
        token = _token()
        url = "%s/drives/%s/root:/%s:/content" % (
            _GRAPH, BUDGET_DRIVE_ID, urllib.parse.quote(BUDGET_PATH))
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        sheet = BUDGET_SHEET if BUDGET_SHEET in wb.sheetnames else next(
            (s for s in wb.sheetnames if s.strip() == BUDGET_SHEET.strip()),
            wb.sheetnames[0])
        ws = wb[sheet]
        hdr = [str(c.value).strip() if c.value is not None else ""
               for c in next(ws.iter_rows(min_row=1, max_row=1))]
        di, ni = hdr.index("DATE"), hdr.index("NET_REVENUE")
        days = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            d, val = row[di], row[ni]
            if d is None or val is None:
                continue
            ds = _pdate(d)
            days[ds] = days.get(ds, 0.0) + float(val)
        _CACHE["at"] = time.time()
        _CACHE["days"] = days
        LOG.info("budget: %d days loaded, group total sample %s", len(days),
                 sorted(days.items())[:1])
        return days
    except Exception:
        LOG.exception("budget load failed - cockpit renders without budget")
        return _CACHE["days"] or {}
