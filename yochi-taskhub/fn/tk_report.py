# -*- coding: utf-8 -*-
"""Status report exporter: scope = everything, one pillar, and/or one project.
PDF (nice-to-read summary) or CSV (flat task rows). Built server-side from
taskapp so the numbers match the app exactly."""
import csv
import datetime as dt
import io
import logging

import tk_db

LOG = logging.getLogger("tk_report")


def _fetch(category_id=None, project_id=None):
    cats = {c["id"]: c for c in tk_db.get("categories", {"select": "id,name,sort",
                                                         "order": "sort"})}
    pparams = {"select": "id,name,category_id", "archived": "eq.false", "order": "name"}
    if project_id:
        pparams["id"] = "eq." + project_id
    elif category_id:
        pparams["category_id"] = "eq.%s" % category_id
    projects = tk_db.get("projects", pparams)
    pids = [p["id"] for p in projects]
    tasks = []
    for i in range(0, len(pids), 25):
        chunk = pids[i:i + 25]
        tasks += tk_db.get("tasks", {
            "project_id": "in.(%s)" % ",".join(chunk),
            "select": "project_id,title,status,priority,due_date,assignee_id,completed_at,source",
            "limit": "3000"})
    profiles = {p["id"]: (p["full_name"] or p["email"]) for p in tk_db.get(
        "profiles", {"select": "id,full_name,email"})}
    return cats, projects, tasks, profiles


def build_csv(category_id=None, project_id=None):
    cats, projects, tasks, profiles = _fetch(category_id, project_id)
    pmap = {p["id"]: p for p in projects}
    today = dt.date.today().isoformat()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Pillar", "Project", "Task", "Status", "Priority", "Due",
                "Overdue", "Assignee", "Source", "Completed"])
    for t in sorted(tasks, key=lambda x: (pmap.get(x["project_id"], {}).get("name", ""),
                                          x.get("due_date") or "9999", x["title"])):
        p = pmap.get(t["project_id"], {})
        w.writerow([
            cats.get(p.get("category_id"), {}).get("name", ""), p.get("name", ""),
            t["title"], t["status"], t["priority"], t.get("due_date") or "",
            "YES" if (t.get("due_date") and t["due_date"] < today
                      and t["status"] != "done") else "",
            profiles.get(t.get("assignee_id"), ""), t.get("source", ""),
            (t.get("completed_at") or "")[:10]])
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel opens UTF-8 cleanly


def build_pdf(category_id=None, project_id=None):
    cats, projects, tasks, profiles = _fetch(category_id, project_id)
    today = dt.date.today()
    today_iso = today.isoformat()
    esc = lambda s: str(s or "").replace("&", "&amp;").replace("<", "&lt;")

    by_proj = {}
    for t in tasks:
        by_proj.setdefault(t["project_id"], []).append(t)

    def counts(ts):
        open_t = [t for t in ts if t["status"] != "done"]
        return {
            "open": len(open_t),
            "overdue": sum(1 for t in open_t if t.get("due_date") and t["due_date"] < today_iso),
            "critical": sum(1 for t in open_t if t["priority"] == "critical"),
            "done30": sum(1 for t in ts if t["status"] == "done" and (t.get("completed_at") or "") >=
                          (today - dt.timedelta(days=30)).isoformat()),
        }

    total = counts(tasks)
    scope_name = "All pillars & projects"
    if project_id and projects:
        scope_name = projects[0]["name"]
    elif category_id:
        scope_name = cats.get(int(category_id), {}).get("name", "Pillar %s" % category_id)

    sections = []
    for cid in sorted(cats, key=lambda c: cats[c]["sort"]):
        cat_projects = [p for p in projects if p["category_id"] == cid]
        if not cat_projects:
            continue
        sections.append("<h1>%s</h1>" % esc(cats[cid]["name"]))
        for p in cat_projects:
            ts = by_proj.get(p["id"], [])
            c = counts(ts)
            open_sorted = sorted((t for t in ts if t["status"] != "done"),
                                 key=lambda x: (x.get("due_date") or "9999"))
            rows = "".join(
                "<tr><td>%s</td><td>%s</td><td>%s</td><td%s>%s</td><td>%s</td></tr>" % (
                    esc(t["title"][:90]), esc(t["status"].replace("_", " ")),
                    esc(t["priority"]),
                    ' style="color:#b91c1c;font-weight:bold"'
                    if (t.get("due_date") and t["due_date"] < today_iso) else "",
                    esc(t.get("due_date") or "—"),
                    esc(profiles.get(t.get("assignee_id"), "—")))
                for t in open_sorted[:40])
            more = ("<p style='color:#6b7280;font-size:8pt'>… and %d more open task(s)</p>"
                    % (len(open_sorted) - 40)) if len(open_sorted) > 40 else ""
            sections.append(
                "<h2>%s</h2><p class='meta'>%d open · %d overdue · %d critical · "
                "%d completed in the last 30 days</p>"
                "<table><tr><th>Task</th><th>Status</th><th>Priority</th><th>Due</th>"
                "<th>Assignee</th></tr>%s</table>%s" % (
                    esc(p["name"]), c["open"], c["overdue"], c["critical"], c["done30"],
                    rows or "<tr><td colspan='5' style='color:#9ca3af'>No open tasks</td></tr>",
                    more))

    html = """<html><head><style>
      @page { size: A4; margin: 1.6cm; }
      body { font-family: Helvetica, Arial, sans-serif; color: #1f2937; font-size: 9pt; }
      h1 { font-size: 13pt; color: #1d683d; border-bottom: 2px solid #dbf2e4;
           padding-bottom: 3px; margin: 18px 0 6px; }
      h2 { font-size: 10.5pt; margin: 12px 0 2px; color: #374151; }
      .meta { color: #6b7280; font-size: 8pt; margin: 0 0 4px; }
      table { width: 100%%; border-collapse: collapse; font-size: 8pt; margin-bottom: 6px; }
      th { background: #f0faf4; text-align: left; padding: 3px 5px; border: 1px solid #e3e7ee; }
      td { padding: 3px 5px; border: 1px solid #e3e7ee; }
      .cover { background: #1d683d; color: #fff; padding: 16px 20px; border-radius: 8px; }
    </style></head><body>
    <div class="cover">
      <div style="font-size:16pt;font-weight:bold">Yo-Chi TaskHub — Status report</div>
      <div style="font-size:9pt;margin-top:3px">Scope: %s · Generated %s</div>
      <div style="font-size:10pt;margin-top:8px">%d open · <b>%d overdue</b> · %d critical ·
      %d completed in the last 30 days</div>
    </div>
    %s
    <p style="color:#9ca3af;font-size:7pt;margin-top:14px">Generated by TaskHub.
    Fast Food Award items are areas to review, never breaches. Money is AUD net of GST.</p>
    </body></html>""" % (esc(scope_name), today.strftime("%d %b %Y"),
                         total["open"], total["overdue"], total["critical"],
                         total["done30"], "".join(sections))

    from xhtml2pdf import pisa
    buf = io.BytesIO()
    result = pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError("pdf generation failed")
    return buf.getvalue(), scope_name
