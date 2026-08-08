# -*- coding: utf-8 -*-
"""yochi-taskhub-fn: server-side companion to the TaskHub SPA.
- template_timer: 00:15 daily -> instantiate period tasks from templates
- HTTP (SPA, Supabase JWT): admin_invite, admin_update_user, nl_task
- HTTP (ops, function key): run_templates
Timers use local crons via WEBSITE_TIME_ZONE=Australia/Sydney."""
import json
import logging
import typing

import azure.functions as func

app = func.FunctionApp()

CORS_JSON = {"Content-Type": "application/json"}

FALLBACK_ADMIN = "818a4f97-84f7-41be-8c25-63126e4e411a"  # Peter


def _json(status, payload):
    return func.HttpResponse(json.dumps(payload), status_code=status, headers=CORS_JSON)


def _admin_uid():
    import tk_db
    rows = tk_db.get("profiles", {"role": "eq.admin", "active": "eq.true",
                                  "select": "id", "limit": "1"})
    return rows[0]["id"] if rows else FALLBACK_ADMIN


def _auto_triage(result, outmsg, forced=False):
    """Queue agent pre-work batches for tasks a watcher/sweep just created.
    On by default; disable with app setting AUTO_TRIAGE=0 (timers) — manual
    HTTP runs only triage when forced (?triage=1)."""
    import os
    import tk_batch
    ids = (result or {}).get("task_ids") or []
    if not ids:
        return []
    if not forced and os.environ.get("AUTO_TRIAGE", "1") == "0":
        return []
    batch_ids = tk_batch.enqueue_triage(ids, _admin_uid())
    if batch_ids:
        outmsg.set(batch_ids)
    return batch_ids


def _authed(req, admin_only=False):
    """Verify the SPA JWT; returns (uid, role, email) or raises AuthError."""
    import tk_auth
    uid, role, email = tk_auth.verify(req)
    if admin_only and role != "admin":
        raise tk_auth.AuthError("admin only")
    return uid, role, email


# ---------------------------------------------------------------- timers
@app.timer_trigger(schedule="0 15 0 * * *", arg_name="timer", run_on_startup=False)
def template_timer(timer: func.TimerRequest) -> None:
    import tk_templates
    result = tk_templates.run()
    logging.info("template_timer: %s", result)


@app.timer_trigger(schedule="0 0 6 * * *", arg_name="timer", run_on_startup=False)
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def watcher_timer(timer: func.TimerRequest,
                  outmsg: func.Out[typing.List[str]]) -> None:
    import tk_watcher
    result = tk_watcher.run()
    result["triage_batches"] = _auto_triage(result, outmsg)
    logging.info("watcher_timer: %s", result)


@app.timer_trigger(schedule="0 0 7 * * *", arg_name="timer", run_on_startup=False)
def briefing_timer(timer: func.TimerRequest) -> None:
    import tk_briefing
    result = tk_briefing.run()
    logging.info("briefing_timer: %s", result)


@app.timer_trigger(schedule="0 0 8 1 * *", arg_name="timer", run_on_startup=False)
def learning_timer(timer: func.TimerRequest) -> None:
    import tk_learning
    result = tk_learning.run()
    logging.info("learning_timer: %s", result)


@app.timer_trigger(schedule="0 30 6 * * *", arg_name="timer", run_on_startup=False)
def skills_timer(timer: func.TimerRequest) -> None:
    import tk_skills
    result = tk_skills.run()
    logging.info("skills_timer: %s", result)


@app.timer_trigger(schedule="0 15 7 * * *", arg_name="timer", run_on_startup=False)
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def glsweep_timer(timer: func.TimerRequest,
                  outmsg: func.Out[typing.List[str]]) -> None:
    import tk_glsweep
    result = tk_glsweep.run()
    result["triage_batches"] = _auto_triage(result, outmsg)
    logging.info("glsweep_timer: %s", result)


@app.timer_trigger(schedule="0 10 6 * * *", arg_name="timer", run_on_startup=False)
def estate_timer(timer: func.TimerRequest) -> None:
    import tk_estate
    result = tk_estate.run()
    logging.info("estate_timer: %s", result)


@app.timer_trigger(schedule="0 5 7 * * 1", arg_name="timer", run_on_startup=False)
def digest_timer(timer: func.TimerRequest) -> None:
    """Monday 07:05: weekly position-delta digest to admins."""
    import tk_digest
    result = tk_digest.run()
    logging.info("digest_timer: %s", result)


@app.timer_trigger(schedule="0 40 6 * * *", arg_name="timer", run_on_startup=False)
def asana_timer(timer: func.TimerRequest) -> None:
    """Daily 06:40: cloud Asana -> TaskHub sync (no desktop dependency)."""
    import tk_asana
    result = tk_asana.run()
    logging.info("asana_timer: %s", result)


@app.timer_trigger(schedule="0 25 6 * * 1", arg_name="timer", run_on_startup=False)
def signals_timer(timer: func.TimerRequest) -> None:
    """Monday 06:25: web-search-grounded cockpit signals."""
    import tk_signals
    result = tk_signals.run()
    logging.info("signals_timer: %s", result)


@app.timer_trigger(schedule="0 35 6 * * 1", arg_name="timer", run_on_startup=False)
def health_timer(timer: func.TimerRequest) -> None:
    """Monday 06:35: leading-indicator venue health scores."""
    import tk_health
    result = tk_health.run()
    logging.info("health_timer: %s", result)


@app.timer_trigger(schedule="0 20 7 * * 1", arg_name="timer", run_on_startup=False)
def price_timer(timer: func.TimerRequest) -> None:
    """Monday 07:20: spend control screen (price creep/dispersion/intensity/POS)."""
    import tk_price
    result = tk_price.run()
    logging.info("price_timer: %s", result)


@app.timer_trigger(schedule="0 20 6 * * *", arg_name="timer", run_on_startup=False)
def anomaly_timer(timer: func.TimerRequest) -> None:
    """Daily 06:20: z-score anomalies -> cockpit signals."""
    import tk_anomaly
    result = tk_anomaly.run()
    logging.info("anomaly_timer: %s", result)


@app.timer_trigger(schedule="0 55 6 * * 1", arg_name="timer", run_on_startup=False)
def cash_timer(timer: func.TimerRequest) -> None:
    """Monday 06:55: 13-week cash forecast + floor alert + email."""
    import tk_cash
    result = tk_cash.run()
    logging.info("cash_timer: %s", result)


@app.timer_trigger(schedule="0 10 7 * * 1", arg_name="timer", run_on_startup=False)
def flash_timer(timer: func.TimerRequest) -> None:
    """Monday 07:10: per-venue prime-cost flash + exception tasks."""
    import tk_flash
    result = tk_flash.run()
    logging.info("flash_timer: %s", result)


@app.timer_trigger(schedule="0 0 5 * * *", arg_name="timer", run_on_startup=False)
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def close_batch_timer(timer: func.TimerRequest, outmsg: func.Out[str]) -> None:
    """WD+1 05:00: agent pre-works the Month-end close project before the team
    starts (batch run over its open tasks, position + debrief email)."""
    result = _run_close_batch(outmsg, force=False)
    logging.info("close_batch_timer: %s", result)


def _run_close_batch(outmsg, force):
    import datetime as dt
    import tk_calendar
    import tk_db
    import tk_glsweep
    today = dt.date.today()
    if not force and today != tk_calendar.business_day_of_month(
            today.year, today.month, 1):
        return {"skipped": "runs on the first business day of the month"}
    # idempotency: never double-queue the scheduled close run for one day
    existing = tk_db.get("batch_runs", {
        "project_id": "eq." + tk_glsweep.CLOSE_PROJECT,
        "kind": "eq.run_all",
        "created_at": "gte." + today.isoformat(),
        "select": "id", "limit": "1"})
    if existing and not force:
        return {"skipped": "close batch already queued today",
                "batch_id": existing[0]["id"]}
    rows = tk_db.insert("batch_runs", [{
        "project_id": tk_glsweep.CLOSE_PROJECT,
        "kind": "run_all",
        "requested_by": _admin_uid(),
    }], returning=True)
    batch_id = rows[0]["id"]
    outmsg.set(batch_id)
    return {"batch_id": batch_id}


# ---------------------------------------------------------------- ops triggers
@app.route(route="run_templates", auth_level=func.AuthLevel.FUNCTION)
def run_templates(req: func.HttpRequest) -> func.HttpResponse:
    import tk_templates
    return _json(200, tk_templates.run())


@app.route(route="run_watcher", auth_level=func.AuthLevel.FUNCTION)
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def run_watcher(req: func.HttpRequest,
                outmsg: func.Out[typing.List[str]]) -> func.HttpResponse:
    import tk_watcher
    result = tk_watcher.run()
    if req.params.get("triage") == "1":
        result["triage_batches"] = _auto_triage(result, outmsg, forced=True)
    return _json(200, result)


@app.route(route="run_briefing", auth_level=func.AuthLevel.FUNCTION)
def run_briefing(req: func.HttpRequest) -> func.HttpResponse:
    import tk_briefing
    force = req.params.get("force") == "1"
    return _json(200, tk_briefing.run(force=force))


@app.route(route="run_learning", auth_level=func.AuthLevel.FUNCTION)
def run_learning(req: func.HttpRequest) -> func.HttpResponse:
    import tk_learning
    return _json(200, tk_learning.run())


@app.route(route="run_skills", auth_level=func.AuthLevel.FUNCTION)
def run_skills(req: func.HttpRequest) -> func.HttpResponse:
    import tk_skills
    return _json(200, tk_skills.run(force_skill_id=req.params.get("skill")))


@app.route(route="run_glsweep", auth_level=func.AuthLevel.FUNCTION)
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def run_glsweep(req: func.HttpRequest,
                outmsg: func.Out[typing.List[str]]) -> func.HttpResponse:
    import tk_glsweep
    result = tk_glsweep.run(force=req.params.get("force") == "1")
    if req.params.get("triage") == "1":
        result["triage_batches"] = _auto_triage(result, outmsg, forced=True)
    return _json(200, result)


@app.route(route="run_asana", auth_level=func.AuthLevel.FUNCTION)
def run_asana(req: func.HttpRequest) -> func.HttpResponse:
    import tk_asana
    return _json(200, tk_asana.run())


@app.route(route="run_signals", auth_level=func.AuthLevel.FUNCTION)
def run_signals(req: func.HttpRequest) -> func.HttpResponse:
    import tk_signals
    return _json(200, tk_signals.run())


@app.route(route="run_price", auth_level=func.AuthLevel.FUNCTION)
def run_price(req: func.HttpRequest) -> func.HttpResponse:
    import tk_price
    return _json(200, tk_price.run(email=req.params.get("email") != "0"))


@app.route(route="run_health", auth_level=func.AuthLevel.FUNCTION)
def run_health(req: func.HttpRequest) -> func.HttpResponse:
    import tk_health
    return _json(200, tk_health.run(email=req.params.get("email") != "0"))


@app.route(route="run_anomaly", auth_level=func.AuthLevel.FUNCTION)
def run_anomaly(req: func.HttpRequest) -> func.HttpResponse:
    import tk_anomaly
    return _json(200, tk_anomaly.run())


@app.route(route="run_cash", auth_level=func.AuthLevel.FUNCTION)
def run_cash(req: func.HttpRequest) -> func.HttpResponse:
    import tk_cash
    return _json(200, tk_cash.run(email=req.params.get("email") != "0"))


@app.route(route="run_flash", auth_level=func.AuthLevel.FUNCTION)
def run_flash(req: func.HttpRequest) -> func.HttpResponse:
    import tk_flash
    return _json(200, tk_flash.run(email=req.params.get("email") != "0"))


@app.route(route="run_estate", auth_level=func.AuthLevel.FUNCTION)
def run_estate(req: func.HttpRequest) -> func.HttpResponse:
    import tk_estate
    return _json(200, tk_estate.run())


@app.route(route="run_digest", auth_level=func.AuthLevel.FUNCTION)
def run_digest(req: func.HttpRequest) -> func.HttpResponse:
    import tk_digest
    return _json(200, tk_digest.run())


@app.route(route="run_close_batch", auth_level=func.AuthLevel.FUNCTION)
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def run_close_batch(req: func.HttpRequest,
                    outmsg: func.Out[str]) -> func.HttpResponse:
    return _json(200, _run_close_batch(outmsg, force=req.params.get("force") == "1"))


@app.route(route="ops_build_skill", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
def ops_build_skill(req: func.HttpRequest) -> func.HttpResponse:
    import tk_skillbuilder
    body = req.get_json()
    return _json(200, tk_skillbuilder.build(body["text"]))


@app.route(route="lake_catalog", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["GET", "OPTIONS"])
def lake_catalog(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import os
    import tk_auth
    import lake_reader
    try:
        _, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        lake_reader.sync(log=logging.info)
        cat_path = os.path.join(lake_reader.CACHE_DIR, "catalog", "catalog.json")
        with open(cat_path) as f:
            catalog = json.load(f)
        return _json(200, catalog)
    except Exception as e:
        logging.exception("lake_catalog failed")
        return _json(500, {"error": str(e)})


@app.route(route="ask_lake", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def ask_lake(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_ai
    import tk_skillbuilder
    try:
        _, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        question = (body.get("question") or "").strip()
        if not question:
            return _json(400, {"error": "no question"})
        history = body.get("history") or []
        convo = "".join("%s: %s\n\n" % (h.get("role", "user"), h.get("content", ""))
                        for h in history[-8:])
        system = (
            "You are the data guide for Yo-Chi's TaskHub. You know the company data lake "
            "(schema below) and how TaskHub's watch rules and AI skills query it (DuckDB, "
            "SELECT/WITH only, bare table names, CAST text dates AS DATE, aggregate to a "
            "useful grain, <=200 rows). Answer questions about what data exists, what "
            "columns mean, how to query something, and how to phrase a skill description "
            "for the AI skill builder. When helpful, include a ready-to-use example query "
            "or a suggested skill description they can paste into the builder. Money is "
            "AUD net of GST. Known quirk: labour_cost in mart_venue_daily has pay-run "
            "artefacts over 100%% of sales - exclude ratios above 1.0. Fast Food Award "
            "items are 'areas to review', never 'breaches'. Be concise and practical."
        )
        schema = tk_skillbuilder._schema_text()
        user = "Lake schema:\n%s\n\n%sQuestion: %s" % (schema, convo, question)
        return _json(200, {"answer": tk_ai.text(system, user, max_tokens=1500)})
    except Exception as e:
        logging.exception("ask_lake failed")
        return _json(500, {"error": str(e)})


@app.route(route="peek_table", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def peek_table(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import re as _re
    import tk_auth
    import lake_reader
    try:
        _, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        table = (body.get("table") or "").strip()
        if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            return _json(400, {"error": "bad table name"})
        lake_reader.sync(extra_tables=[table], log=logging.info)
        result = lake_reader.query('SELECT * FROM "%s" LIMIT 10' % table, max_rows=10)
        return _json(200, result)
    except Exception as e:
        logging.exception("peek_table failed")
        return _json(500, {"error": str(e)})


@app.route(route="agent_propose", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def agent_propose(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_agent
    try:
        _, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        return _json(200, tk_agent.propose(body["task_id"]))
    except Exception as e:
        logging.exception("agent_propose failed")
        return _json(500, {"error": str(e)})


@app.route(route="agent_execute", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def agent_execute(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_agent
    try:
        uid, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        return _json(200, tk_agent.execute(
            body["task_id"], body["plan"], uid,
            feedback=body.get("feedback"), prior_summary=body.get("prior_summary")))
    except Exception as e:
        logging.exception("agent_execute failed")
        return _json(500, {"error": str(e)})


@app.route(route="send_report", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def send_report(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_agent
    import tk_email
    try:
        _, role, sender_email = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        task_id = body["task_id"]
        to = (body.get("to") or "").strip()
        files = body.get("files") or []
        note = (body.get("note") or "").strip()
        if not to or "@" not in to:
            return _json(400, {"error": "valid recipient email required"})
        if not files:
            return _json(400, {"error": "no files to send"})
        mimes = {".html": "text/html", ".pdf": "application/pdf",
                 ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
        attachments = []
        for f in files[:5]:
            data = tk_agent.download_file(task_id, f)
            ext = "." + f.rsplit(".", 1)[-1].lower()
            attachments.append({"name": f, "data": data,
                                "mime": mimes.get(ext, "application/octet-stream")})
        html = ("<p>%s</p><p style='color:#6b7280;font-size:12px'>Sent from Yo-Chi TaskHub by %s. "
                "Attached: %s.</p>") % (
            note.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>") or
            "Please find the attached task output.", sender_email, ", ".join(files))
        sent = tk_email.send(to, body.get("subject") or "TaskHub report", html,
                             attachments=attachments)
        return _json(200, {"ok": True, "sent": sent})
    except Exception as e:
        logging.exception("send_report failed")
        return _json(500, {"error": str(e)})


@app.route(route="ops_agent", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
def ops_agent(req: func.HttpRequest) -> func.HttpResponse:
    import tk_agent
    body = req.get_json()
    if body.get("phase") == "execute":
        return _json(200, tk_agent.execute(body["task_id"], body["plan"],
                                           body.get("uid", "818a4f97-84f7-41be-8c25-63126e4e411a")))
    return _json(200, tk_agent.propose(body["task_id"]))


@app.queue_trigger(arg_name="msg", queue_name="taskhub-batch",
                   connection="AzureWebJobsStorage")
def batch_worker(msg: func.QueueMessage) -> None:
    import tk_batch
    batch_id = msg.get_body().decode().strip()
    logging.info("batch_worker picking up %s", batch_id)
    tk_batch.process(batch_id)


@app.route(route="project_run", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def project_run(req: func.HttpRequest, outmsg: func.Out[str]) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_db
    try:
        uid, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        rows = tk_db.insert("batch_runs", [{
            "project_id": body["project_id"],
            "kind": "run_all",
            "requested_by": uid,
        }], returning=True)
        batch_id = rows[0]["id"]
        outmsg.set(batch_id)
        return _json(200, {"batch_id": batch_id})
    except Exception as e:
        logging.exception("project_run failed")
        return _json(500, {"error": str(e)})


@app.route(route="project_steer", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def project_steer(req: func.HttpRequest, outmsg: func.Out[str]) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_db
    try:
        uid, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        steer = (body.get("steer") or "").strip()
        if not steer:
            return _json(400, {"error": "no steer text"})
        rows = tk_db.insert("batch_runs", [{
            "project_id": body["project_id"],
            "kind": "steer",
            "parent_batch_id": body["parent_batch_id"],
            "steer_text": steer,
            "requested_by": uid,
        }], returning=True)
        batch_id = rows[0]["id"]
        outmsg.set(batch_id)
        return _json(200, {"batch_id": batch_id})
    except Exception as e:
        logging.exception("project_steer failed")
        return _json(500, {"error": str(e)})


@app.route(route="ops_batch", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
@app.queue_output(arg_name="outmsg", queue_name="taskhub-batch",
                  connection="AzureWebJobsStorage")
def ops_batch(req: func.HttpRequest, outmsg: func.Out[str]) -> func.HttpResponse:
    import tk_db
    body = req.get_json()
    rows = tk_db.insert("batch_runs", [{
        "project_id": body["project_id"],
        "kind": body.get("kind", "run_all"),
        "parent_batch_id": body.get("parent_batch_id"),
        "steer_text": body.get("steer"),
        "requested_by": body.get("uid", "818a4f97-84f7-41be-8c25-63126e4e411a"),
    }], returning=True)
    batch_id = rows[0]["id"]
    outmsg.set(batch_id)
    return _json(200, {"batch_id": batch_id})


_DASH_CACHE = {"at": 0, "data": None}


@app.route(route="dashboard_data", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["GET", "OPTIONS"])
def dashboard_data(req: func.HttpRequest) -> func.HttpResponse:
    """Live lake KPIs for the home dashboard (10-min instance cache)."""
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import time
    import tk_auth
    import lake_reader
    try:
        _, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    if _DASH_CACHE["data"] and time.time() - _DASH_CACHE["at"] < 600:
        return _json(200, _DASH_CACHE["data"])
    try:
        lake_reader.sync(extra_tables=["procedure_report"], log=logging.info)
        trading = lake_reader.query("""
            WITH d AS (SELECT CAST("date" AS DATE) AS d, net_sales, ly_net_sales
                       FROM mart_venue_daily),
            last14 AS (SELECT d, round(sum(net_sales),0) AS sales,
                              round(sum(ly_net_sales),0) AS ly
                       FROM d WHERE d > (SELECT max(d) FROM d) - INTERVAL 14 DAY
                       GROUP BY d ORDER BY d)
            SELECT * FROM last14""", max_rows=14)
        proc = lake_reader.query("""
            WITH p AS (SELECT CAST(start_date AS DATE) AS d, status
                       FROM procedure_report WHERE venue <> 'TEST Yo-Chi ETC'),
            latest AS (SELECT max(d) - INTERVAL 1 DAY AS d0 FROM p)
            SELECT count(*) AS procs,
                   count(*) FILTER (WHERE status LIKE 'Completed%') AS done,
                   count(*) FILTER (WHERE status = 'Cancelled') AS cancelled,
                   max(d) AS day
            FROM p WHERE d = (SELECT d0 FROM latest)""", max_rows=1)
        rows = trading["rows"]
        latest = rows[-1] if rows else None
        import tk_db
        try:
            forecast = tk_db.get("v_forecast_daily", {"order": "d", "select": "d,forecast_sales"})
        except Exception:
            logging.exception("forecast view fetch failed (non-fatal)")
            forecast = []
        data = {
            "trading": {
                "series": [{"d": r[0], "sales": r[1], "ly": r[2]} for r in rows],
                "latest_day": latest[0] if latest else None,
                "latest_sales": latest[1] if latest else None,
                "latest_ly": latest[2] if latest else None,
            },
            "forecast": forecast,
            "procedures": dict(zip(proc["columns"], proc["rows"][0])) if proc["rows"] else None,
            "as_at": time.time(),
        }
        try:
            data["cockpit"] = _cockpit(tk_db, data)
        except Exception:
            logging.exception("cockpit assembly failed (non-fatal)")
            data["cockpit"] = None
        _DASH_CACHE["at"] = time.time()
        _DASH_CACHE["data"] = data
        return _json(200, data)
    except Exception as e:
        logging.exception("dashboard_data failed")
        return _json(500, {"error": str(e)})


def _cockpit(tk_db, data):
    """Mission-control extras: meetings, cash, compliance radar, team workload,
    data-verified checklist, decisions pending, external signals."""
    import datetime as _dt
    today = _dt.date.today()
    now_iso = _dt.datetime.utcnow().isoformat() + "Z"

    meetings = tk_db.get("calendar_events", {
        "starts_at": "gte." + now_iso, "order": "starts_at",
        "select": "subject,starts_at,ends_at,location,organizer,attendees,prep",
        "limit": "8"})

    cash_rows = tk_db.get("cash_forecast", {
        "order": "generated_at.desc,week_start", "limit": str(26),
        "select": "generated_at,week_start,closing,net,assumptions"})
    cash = None
    if cash_rows:
        latest_gen = cash_rows[0]["generated_at"]
        weeks = [r for r in cash_rows if r["generated_at"] == latest_gen]
        assumptions = next((r["assumptions"] for r in weeks if r.get("assumptions")), {})
        trough = min(weeks, key=lambda w: float(w["closing"]))
        cash = {"weeks": [{"week_start": w["week_start"], "closing": w["closing"]}
                          for w in weeks],
                "trough_week": trough["week_start"], "trough": trough["closing"],
                "opening": assumptions.get("opening_cash"),
                "floor": assumptions.get("floor"),
                "dso_days": assumptions.get("dso_days"),
                "dpo_days": assumptions.get("dpo_days"),
                "generated_at": latest_gen}

    horizon = (today + _dt.timedelta(days=60)).isoformat()
    radar_raw = tk_db.get("tasks", {
        "status": "neq.done", "due_date": "lte." + horizon,
        "select": "title,due_date,priority,project_id,projects(name,category_id)",
        "order": "due_date", "limit": "200"})
    radar = [{"title": t["title"], "due": t["due_date"], "priority": t["priority"],
              "project": (t.get("projects") or {}).get("name")}
             for t in radar_raw
             if (t.get("projects") or {}).get("category_id") in (2, 7)][:12]

    open_tasks = tk_db.get("tasks", {
        "status": "neq.done",
        "select": "assignee_id,due_date,priority,title,project_id", "limit": "1000"})
    profiles = {p["id"]: p["full_name"] or p["email"] for p in tk_db.get(
        "profiles", {"active": "eq.true", "select": "id,full_name,email"})}
    team = {}
    today_iso = today.isoformat()
    for t in open_tasks:
        who = profiles.get(t.get("assignee_id"), "Unassigned")
        rec = team.setdefault(who, {"open": 0, "overdue": 0, "critical": 0})
        rec["open"] += 1
        if t.get("due_date") and t["due_date"] < today_iso:
            rec["overdue"] += 1
        if t.get("priority") == "critical":
            rec["critical"] += 1
    team_rows = sorted(
        ({"who": k, **v} for k, v in team.items()),
        key=lambda r: -r["open"])[:10]

    open_estate = sum(1 for t in open_tasks if t["title"].startswith("[Estate]"))
    open_watch = sum(1 for t in open_tasks if t["title"].startswith("[Watch]"))
    close_overdue = sum(1 for t in open_tasks
                        if t.get("project_id") == "aaaaaaaa-0000-0000-0000-000000000001"
                        and t.get("due_date") and t["due_date"] < today_iso)
    crit_overdue = sum(1 for t in open_tasks if t.get("priority") == "critical"
                       and t.get("due_date") and t["due_date"] < today_iso)
    fresh = bool(data["trading"]["latest_day"] and
                 data["trading"]["latest_day"] >= (today - _dt.timedelta(days=2)).isoformat())
    checklist = [
        {"item": "Sales data fresh (≤2 days)", "ok": fresh,
         "detail": "latest day %s" % data["trading"]["latest_day"]},
        {"item": "No estate-health issues open", "ok": open_estate == 0,
         "detail": "%d open" % open_estate},
        {"item": "No unresolved data-watch flags", "ok": open_watch == 0,
         "detail": "%d open" % open_watch},
        {"item": "Close tasks on schedule", "ok": close_overdue == 0,
         "detail": "%d overdue" % close_overdue},
        {"item": "Cash above floor all 13 weeks",
         "ok": bool(cash) and cash["trough"] is not None and cash.get("floor") is not None
               and float(cash["trough"]) >= float(cash["floor"]),
         "detail": ("trough $%s" % "{:,.0f}".format(float(cash["trough"])))
                   if cash else "no forecast yet"},
        {"item": "No critical tasks overdue", "ok": crit_overdue == 0,
         "detail": "%d overdue" % crit_overdue},
    ]

    decisions = [{"title": t["title"], "due": t.get("due_date"),
                  "who": profiles.get(t.get("assignee_id"), "")}
                 for t in open_tasks if t.get("priority") == "critical"][:6]

    signals = tk_db.get("signals", {
        "order": "created_at.desc", "limit": "6",
        "select": "kind,headline,detail,source,created_at"})

    vip = tk_db.get("vip_messages", {
        "order": "received_at.desc", "limit": "6",
        "select": "sender,subject,snippet,received_at,weblink"})

    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    priority = sorted(
        (t for t in open_tasks if t.get("priority") in ("critical", "high")),
        key=lambda t: (rank.get(t.get("priority"), 9), t.get("due_date") or "9999"))
    priority = [{"title": t["title"], "due": t.get("due_date"),
                 "priority": t["priority"],
                 "who": profiles.get(t.get("assignee_id"), "")}
                for t in priority[:7]]

    return {"meetings": meetings, "cash": cash, "radar": radar,
            "team": team_rows, "checklist": checklist,
            "decisions": decisions, "signals": signals,
            "vip": vip, "priority": priority}


@app.route(route="notify", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def notify(req: func.HttpRequest) -> func.HttpResponse:
    """Fire-and-forget notification emails: assignment / comment / @mention.
    body: {kind, task_id, recipient_ids: [uuid], note?}"""
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import os
    import tk_auth
    import tk_db
    import tk_email
    try:
        uid, _, sender_email = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    try:
        body = req.get_json()
        kind = body.get("kind", "assigned")
        task_id = body["task_id"]
        recipients = [r for r in (body.get("recipient_ids") or []) if r and r != uid][:10]
        if not recipients:
            return _json(200, {"sent": 0})
        tasks = tk_db.get("tasks", {"id": "eq." + task_id, "select": "title,project_id"})
        if not tasks:
            return _json(404, {"error": "task not found"})
        title = tasks[0]["title"]
        app_url = os.environ.get("APP_URL", "")
        link = "%s/my-tasks?task=%s" % (app_url, task_id)
        prefs = {p["user_id"]: p for p in tk_db.get(
            "notification_prefs", {"select": "user_id,email_on_assign"})}
        subj = {"assigned": "You've been assigned: %s",
                "comment": "New comment on: %s",
                "mention": "You were mentioned on: %s"}.get(kind, "Update on: %s") % title
        note = (body.get("note") or "").strip()[:500]
        esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;")
        html = ("<p><b>%s</b></p>%s<p><a href='%s' style='display:inline-block;background:#23824a;"
                "color:#fff;text-decoration:none;font-weight:700;font-size:14px;padding:9px 18px;"
                "border-radius:8px'>Open the task &rarr;</a></p>"
                "<p style='color:#6b7280;font-size:12px'>From %s via Yo-Chi TaskHub.</p>") % (
            esc(title), ("<p>%s</p>" % esc(note)) if note else "", link, esc(sender_email))
        sent = 0
        for rid in recipients:
            pref = prefs.get(rid)
            if kind == "assigned" and pref and not pref.get("email_on_assign", True):
                continue
            prof = tk_db.get("profiles", {"id": "eq." + rid, "active": "eq.true",
                                          "select": "email"})
            if prof and tk_email.send(prof[0]["email"], subj, html):
                sent += 1
        return _json(200, {"sent": sent})
    except Exception as e:
        logging.exception("notify failed")
        return _json(500, {"error": str(e)})


@app.route(route="build_skill", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def build_skill(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_skillbuilder
    try:
        _, role, _ = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "finance or admin only"})
    try:
        body = req.get_json()
        text = (body.get("text") or "").strip()
        if not text:
            return _json(400, {"error": "no description"})
        return _json(200, tk_skillbuilder.build(text))
    except Exception as e:
        logging.exception("build_skill failed")
        return _json(500, {"error": str(e)})


# ---------------------------------------------------------------- SPA endpoints
# NOTE: routes may not begin with "admin" — the Functions host reserves that
# prefix and silently 404s such routes.
@app.route(route="invite_user", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def admin_invite(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_db
    try:
        _authed(req, admin_only=True)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    try:
        body = req.get_json()
        email = body["email"].strip().lower()
        role = body.get("role", "stakeholder")
        if role not in ("admin", "finance", "stakeholder"):
            return _json(400, {"error": "bad role"})
        tk_db.invite_user(email, role, body.get("full_name"),
                          redirect_to=body.get("redirect_to"))
        return _json(200, {"ok": True, "invited": email, "role": role})
    except Exception as e:
        logging.exception("admin_invite failed")
        return _json(500, {"error": str(e)})


@app.route(route="update_user", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def admin_update_user(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_db
    try:
        uid, _, _ = _authed(req, admin_only=True)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    try:
        body = req.get_json()
        target = body["user_id"]
        patch = {}
        if "role" in body:
            if body["role"] not in ("admin", "finance", "stakeholder"):
                return _json(400, {"error": "bad role"})
            patch["role"] = body["role"]
        if "active" in body:
            patch["active"] = bool(body["active"])
        if not patch:
            return _json(400, {"error": "nothing to update"})
        if target == uid and patch.get("role") and patch["role"] != "admin":
            return _json(400, {"error": "cannot demote yourself"})
        rows = tk_db.patch("profiles", {"id": "eq." + target}, patch)
        return _json(200, {"ok": True, "profile": rows[0] if rows else None})
    except Exception as e:
        logging.exception("admin_update_user failed")
        return _json(500, {"error": str(e)})


@app.route(route="nl_task", auth_level=func.AuthLevel.ANONYMOUS,
           methods=["POST", "OPTIONS"])
def nl_task(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=204)
    import tk_auth
    import tk_nl
    try:
        _, role, email = _authed(req)
    except tk_auth.AuthError as e:
        return _json(401, {"error": str(e)})
    if role == "stakeholder":
        return _json(403, {"error": "stakeholders cannot create tasks"})
    try:
        body = req.get_json()
        text = (body.get("text") or "").strip()
        if not text:
            return _json(400, {"error": "no text"})
        return _json(200, tk_nl.parse(text, email))
    except Exception as e:
        logging.exception("nl_task failed")
        return _json(500, {"error": str(e)})
