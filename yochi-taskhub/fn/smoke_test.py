# -*- coding: utf-8 -*-
"""Pre/post-deploy smoke test - catches the class of bug that otherwise costs a
full deploy-restart-wait cycle to discover:

  * a DB column the code selects but that doesn't exist (agent_runs.status)
  * a lake query with a typo'd column, bad DuckDB syntax, or a missing table
  * a helper whose RETURN SHAPE changed under its callers (lake_reader.query)
  * output that won't serialise to JSON (NaN from a DuckDB aggregate)

Read-only: every probe is a SELECT. Run it from deploy.ps1 (ops_smoke route)
or on demand; it finishes in seconds because lake queries hit blob directly."""
import datetime as dt
import json
import logging
import math
import os
import re

import lake_reader
import tk_db

LOG = logging.getLogger("smoke_test")

# --- every taskapp table + the columns our code actually selects -------------
DB_PROBES = {
    "tasks": "id,project_id,template_id,period_id,title,description,status,priority,"
             "assignee_id,reviewer_id,due_date,completed_at,source,checklist,created_by,"
             "created_at,updated_at,watcher_rule_id,parent_id,sort_order,external_ref,"
             "recurrence,recurrence_mode,recurrence_until,recurrence_parent_id,email_thread",
    "notify_outbox": "id,task_id,recipient,kind,actor,created_at,sent_at,attempts,error,"
                     "comment_id",
    "email_drop_log": "ref,task_id,action,subject,processed_at",
    "knowledge": "id,task_id,question,answer,engine,depth,recency,created_by,"
                 "created_at",
    "task_knowledge": "task_id,summary,entry_count,updated_at",
    "agreement_snapshots": "id,slug,territory,data,hash,captured_at",
    "compliance_items": "id,ref,period,stream,category,obligation,what_to_do,authority,frequency,due_text,due_date,due_basis,owner,risk,entities,sheet_status,completed_date,evidence,notes,severity,action,task_id,task_status,writeback,active,updated_at",
    "compliance_runs": "id,ran_at,ran_by,items,tasks_added,tasks_amended,tasks_linked,writeback_ready,flags,detail",
    "playbook": "approach_key,task_title,project,method,queries_that_worked,pitfalls,improve_next_time,outcome,version,changed_this_run,updated_at",
    "playbook_versions": "id,approach_key,task_title,method,version,changed_this_run,created_at",
    "projects": "id,name,category_id",
    "categories": "id,name,sort",
    "profiles": "id,email,full_name,role,active",
    "comments": "id,task_id,author_id,body,created_at",
    "audit_log": "id,table_name,row_id,action,actor,old_row,new_row,at",
    "agent_runs": "id,task_id,task_title,plan,outcome,error,requested_by,created_at",
    "batch_runs": "id,project_id,kind,status,created_at",
    "approvals": "id,task_id,kind,approver_id,note",
    "attachments": "id,task_id,kind,storage_path,url,filename,size_bytes,mime,uploaded_by",
    "task_dependencies": "task_id,depends_on_task_id",
    "task_templates": "id,project_id,title,description,cadence,due_rule,active,priority,"
                      "default_assignee_id,default_reviewer_id",
    "periods": "id,period_month,label",
    "notification_prefs": "user_id,daily_briefing,email_on_assign",
    "watcher_rules": "id,name,description,check_sql,comparator,threshold,project_id,"
                     "assignee_id,priority,active,last_run_at",
    "ai_skills": "id,name,prompt,data_queries,cadence,weekday,project_id,assignee_id,"
                 "email_review,active,last_run_at",
    "feeds": "id,slug,name,description,kind,cadence,status,last_run_at,last_summary,"
             "research_prompt,columns",
    "signals": "id,kind,headline,detail,source,created_at",
    "vip_messages": "id,external_ref,sender,sender_email,subject,snippet,received_at",
    "calendar_events": "id,external_ref,subject,starts_at,ends_at,organizer,attendees,prep",
    "cash_forecast": "id,generated_at,week_start,receipts,ap,payroll,ato_super,net,"
                     "closing,assumptions",
    "pl_pulse": "day,data,created_at",
    "venue_health": "id,venue,week_start,score,components,created_at",
    "positions": "id,project_id,version,content,events,source,created_at",
    "escalations": "id,task_id,rule,sent_at",
    "work_reports": "report_date,stats,narrative,created_at",
    "data_contracts": "id,table_name,label,date_expr,max_staleness_days,min_rows_recent,"
                      "recent_days,weekdays_only,active,last_status,last_checked_at,"
                      "last_detail",
    "project_members": "project_id,user_id",
    "category_members": "category_id,user_id",
}

# --- representative lake queries per subsystem ------------------------------
LAKE_PROBES = {
    "mart_venue_daily": 'SELECT "date", venue, net_sales FROM mart_venue_daily '
                        'ORDER BY CAST("date" AS DATE) DESC LIMIT 1',
    "xero_transactions": 'SELECT "Date", "Source", "Net Amount", "XeroOrganisationName" '
                         'FROM XeroAccountTransactionsMasterView LIMIT 1',
    "invoices": "SELECT invoiceid, total, status, type, updateddateutc, fullypaidondate "
                "FROM Invoices LIMIT 1",
    "restoke_purchasing": "SELECT venue, product, total, received_date, invoice_number, "
                          "order_id FROM restoke_purchasing LIMIT 1",
    "procedure_report": "SELECT venue, status, completed_by, completion_date "
                        "FROM procedure_report LIMIT 1",
    "asana_tasks": "SELECT TaskId, AssigneeName, CompletedAt FROM AsanaTasks LIMIT 1",
    "aggregate_shape": "SELECT count(*) AS n, round(sum(TRY_CAST(net_sales AS DOUBLE)),0) "
                       "AS total FROM mart_venue_daily WHERE 1=0",
}


def _literal_sql(node, consts):
    import ast
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.Name):
        v = consts.get(node.id)
        return (v, False) if v else (None, False)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        left, _ = _literal_sql(node.left, consts)
        return left, True  # runtime-formatted template
    return None, False


def extract_lake_sql(fn_dir=None):
    """Every lake_reader.query(...) SQL in the codebase, found by parsing the
    source. Self-maintaining: new queries are covered the day they're written."""
    import ast
    import glob as _glob
    fn_dir = fn_dir or os.path.dirname(os.path.abspath(__file__))
    found = []
    paths = sorted(_glob.glob(os.path.join(fn_dir, "tk_*.py")))
    paths.append(os.path.join(fn_dir, "function_app.py"))
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception:
            continue
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = node.value.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and f.attr == "query"
                    and isinstance(f.value, ast.Name) and f.value.id == "lake_reader"):
                continue
            if not node.args:
                continue
            sql, templated = _literal_sql(node.args[0], consts)
            if sql and sql.strip():
                found.append({"file": os.path.basename(path), "line": node.lineno,
                              "sql": sql, "templated": templated})
    return found


def _fill_template(sql):
    """Plug benign values into a runtime-formatted query so it can be bound.
    A %s inside double quotes is an identifier (table name), not a value."""
    sql = sql.replace('"%s"', '"mart_venue_daily"')
    return re.sub(r"%[sdf]", lambda m: "1" if m.group(0) in ("%d", "%f")
                  else "2026-08-15", sql).replace("%%", "%")


def validation_con(all_sql):
    """One connection carrying views for every table the whole codebase reads,
    so each query costs an EXPLAIN rather than a fresh connection + extension
    load. Pass the concatenated SQL - _direct_con derives the table list."""
    combined = "\n".join(all_sql)
    return (lake_reader._direct_con(combined) if lake_reader._direct_ok()
            else lake_reader._local_con(combined))


def validate_sql(sql, con=None):
    """Parse + bind against the real lake schema WITHOUT scanning data - catches
    missing tables, wrong column names and bad syntax in milliseconds."""
    own = con is None
    con = con or validation_con([sql])
    try:
        con.execute("EXPLAIN " + sql)
    finally:
        if own:
            con.close()


def _json_safe(obj):
    """Anything that survives json.dumps also survives PostgREST."""
    try:
        json.dumps(obj)
        return True, None
    except (TypeError, ValueError) as e:
        return False, str(e)[:120]


def _has_nan(obj):
    if isinstance(obj, float):
        return not math.isfinite(obj)
    if isinstance(obj, dict):
        return any(_has_nan(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nan(v) for v in obj)
    return False


def run(level="full", deep=None):
    """level: quick  - DB columns + guard + one light lake read (fits inside
                       Azure's ~230s HTTP limit; used as the deploy gate)
              full   - + every lake query in the codebase bound against the
                       real schema, + per-subsystem probes (the local loop)
              deep   - + the actual report builders and contract engine"""
    if deep is not None:  # back-compat
        level = "deep" if deep else "full"
    do_sql = level in ("full", "deep")
    do_deep = level == "deep"
    results = {"passed": [], "failed": [], "level": level}

    def ok(name, detail=""):
        results["passed"].append({"check": name, "detail": detail})

    def fail(name, err):
        results["failed"].append({"check": name, "error": str(err)[:300]})

    # 1. every table/column the code selects must exist
    for table, cols in DB_PROBES.items():
        try:
            tk_db.get(table, {"select": cols, "limit": "1"})
            ok("db:" + table)
        except Exception as e:
            fail("db:" + table, e)

    # 2. lake reachable, and each subsystem's columns/syntax valid
    try:
        mode = "direct" if lake_reader._direct_ok() else "local-cache"
        ok("lake:mode", mode)
    except Exception as e:
        fail("lake:mode", e)
    probes = LAKE_PROBES if do_sql else {"mart_venue_daily": LAKE_PROBES["mart_venue_daily"]}
    for name, sql in probes.items():
        try:
            out = lake_reader.query(sql, max_rows=1)
            if not isinstance(out, dict) or "columns" not in out or "rows" not in out:
                raise AssertionError("query() shape changed: %r" % type(out))
            ok("lake:" + name, "%d col(s)" % len(out["columns"]))
        except Exception as e:
            fail("lake:" + name, e)

    # 3. EVERY lake query in the codebase binds against the real schema
    items = extract_lake_sql() if do_sql else []
    for it in items:
        it["filled"] = _fill_template(it["sql"]) if it["templated"] else it["sql"]
    vcon = None
    try:
        vcon = validation_con([it["filled"] for it in items]) if items else None
    except Exception as e:
        fail("sql:connection", e)
    for item in items:
        name = "sql:%s:%d" % (item["file"], item["line"])
        sql = item["filled"]
        try:
            validate_sql(sql, con=vcon)
            ok(name)
        except Exception as e:
            if item["templated"]:
                # placeholder substitution can produce nonsense literals; only a
                # binder error (missing table/column) is a genuine defect
                if type(e).__name__ == "BinderException" or "does not exist" in str(e):
                    fail(name, e)
                else:
                    ok(name, "template, not fully checkable")
            else:
                fail(name, e)
    if vcon is not None:
        vcon.close()

    # 4. the guard still refuses what it must
    for bad in ("DROP TABLE tasks", "SELECT 1; DROP TABLE tasks",
                "INSTALL azure", "SELECT * FROM 'secrets.csv'",
                "CREATE SECRET x (TYPE AZURE)"):
        try:
            lake_reader.check_sql(bad)
            fail("guard:" + bad[:28], "guard ALLOWED a forbidden query")
        except ValueError:
            ok("guard:" + bad[:28])
        except Exception as e:
            fail("guard:" + bad[:28], e)

    # 5. the real report builders run and stay JSON-clean
    if do_deep:
        try:
            import tk_workreport
            day = dt.datetime.now(tk_workreport.AEST).date() - dt.timedelta(days=1)
            utc0, utc1 = tk_workreport._windows(day)
            lake = tk_workreport._lake_stats(day, utc0, utc1)
            errs = [k for k, rows in lake.items()
                    if isinstance(rows, list) and rows and isinstance(rows[0], dict)
                    and "error" in rows[0]]
            if errs:
                fail("workreport:lake_stats", "query errors in: " + ", ".join(errs))
            else:
                ok("workreport:lake_stats", "%d sections" % len(lake))
            hub = tk_workreport._taskhub_stats(utc0, utc1)
            ok("workreport:taskhub_stats", "%d people" % len(hub["people"]))
            payload = tk_workreport._clean({"lake": lake, "taskhub": hub})
            good, err = _json_safe(payload)
            if not good:
                fail("workreport:json", err)
            elif _has_nan(payload):
                fail("workreport:json", "NaN/Inf survived _clean()")
            else:
                ok("workreport:json")
        except Exception as e:
            fail("workreport", e)

        try:
            import tk_contracts
            out = tk_contracts.check(raise_task=False)
            ok("contracts:check", "%d contract(s), %d breach(es)"
               % (out["checked"], len(out["breaches"])))
        except Exception as e:
            fail("contracts:check", e)

        try:
            import tk_recurring
            base = dt.date(2026, 1, 31)
            cases = [("daily", dt.date(2026, 2, 1)), ("weekly", dt.date(2026, 2, 7)),
                     ("fortnightly", dt.date(2026, 2, 14)),
                     ("monthly", dt.date(2026, 2, 28)),      # clamps short month
                     ("quarterly", dt.date(2026, 4, 30)),
                     ("annual", dt.date(2027, 1, 31))]
            for rec, want in cases:
                got = tk_recurring.next_due(rec, base)
                if got != want:
                    raise AssertionError("%s from %s -> %s, expected %s"
                                         % (rec, base, got, want))
            ok("recurring:next_due", "%d cadences correct" % len(cases))
        except Exception as e:
            fail("recurring:next_due", e)

    results["ok"] = not results["failed"]
    results["summary"] = "%s: %d passed, %d failed" % (
        level, len(results["passed"]), len(results["failed"]))
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    out = run(level=sys.argv[1] if len(sys.argv) > 1 else "deep")
    print(json.dumps(out, indent=2)[:6000])
    raise SystemExit(0 if out["ok"] else 1)
