# -*- coding: utf-8 -*-
"""yochi-data-lake: nightly dataSights -> Parquet lake sync.

Timers (AEST via WEBSITE_TIME_ZONE):
  %LAKE_CRON%   default 05:30 - sync before the 06:30/07:00 morning builds
HTTP (function key):
  /api/lake_sync_now[?only=View1,View2]  - manual/preview trigger
"""
import json
import logging

import azure.functions as func

import doc_extract
import lake_export
import marts
import opcentral_export
import restoke_export
import restoke_hq_export
import sp_mirror

app = func.FunctionApp()


@app.timer_trigger(schedule="%RESTOKE_CRON%", arg_name="timer", run_on_startup=False)
def nightly_restoke_export(timer: func.TimerRequest):
    result = restoke_export.run()
    logging.info("restoke export result: %s", json.dumps(result)[:2000])


@app.timer_trigger(schedule="%OPCENTRAL_CRON%", arg_name="timer", run_on_startup=False)
def nightly_opcentral_export(timer: func.TimerRequest):
    result = opcentral_export.run()
    logging.info("opcentral export result: %s", json.dumps(result)[:2000])


@app.timer_trigger(schedule="%RESTOKE_HQ_CRON%", arg_name="timer", run_on_startup=False)
def nightly_restoke_hq_export(timer: func.TimerRequest):
    result = restoke_hq_export.run()
    logging.info("restoke-hq export result: %s", json.dumps(result)[:2000])


@app.route(route="restoke_hq_export_now", auth_level=func.AuthLevel.FUNCTION)
def restoke_hq_export_now(req: func.HttpRequest) -> func.HttpResponse:
    try:
        wd = req.params.get("window_days")
        lim = req.params.get("limit")
        result = restoke_hq_export.run(window_days=int(wd) if wd else None,
                                       limit=int(lim) if lim else None,
                                       raw=req.params.get("raw") == "1",
                                       backfill=req.params.get("backfill") == "1")
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("restoke_hq_export_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.route(route="opcentral_export_now", auth_level=func.AuthLevel.FUNCTION)
def opcentral_export_now(req: func.HttpRequest) -> func.HttpResponse:
    only = req.params.get("only")
    only_set = {s.strip() for s in only.split(",")} if only else None
    try:
        result = opcentral_export.run(only=only_set)
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("opcentral_export_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.route(route="restoke_export_now", auth_level=func.AuthLevel.FUNCTION)
def restoke_export_now(req: func.HttpRequest) -> func.HttpResponse:
    only = req.params.get("only")
    only_set = {s.strip() for s in only.split(",")} if only else None
    try:
        result = restoke_export.run(
            start_month=req.params.get("from"), end_month=req.params.get("to"),
            only=only_set)
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("restoke_export_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.timer_trigger(schedule="%MARTS_CRON%", arg_name="timer", run_on_startup=False)
def nightly_marts(timer: func.TimerRequest):
    result = marts.build_all()
    logging.info("marts result: %s", json.dumps(result))


@app.route(route="marts_now", auth_level=func.AuthLevel.FUNCTION)
def marts_now(req: func.HttpRequest) -> func.HttpResponse:
    try:
        result = marts.build_all()
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("marts_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.timer_trigger(schedule="%EXTRACT_CRON%", arg_name="timer", run_on_startup=False)
def nightly_doc_extract(timer: func.TimerRequest):
    result = doc_extract.run_extract(minutes=100, prefix=None)
    logging.info("doc extract result: %s", json.dumps(result))


@app.route(route="extract_now", auth_level=func.AuthLevel.FUNCTION)
def extract_now(req: func.HttpRequest) -> func.HttpResponse:
    minutes = int(req.params.get("minutes", "90"))
    prefix = req.params.get("prefix")  # e.g. sharepoint/ or canva/ ; empty = both
    try:
        result = doc_extract.run_extract(minutes=minutes, prefix=prefix or None)
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("extract_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.timer_trigger(schedule="%SP_MIRROR_CRON%", arg_name="timer", run_on_startup=False)
def nightly_sp_mirror(timer: func.TimerRequest):
    result = sp_mirror.run_mirror(minutes=100)
    logging.info("sp mirror result: %s", json.dumps(result))


@app.route(route="sp_mirror_now", auth_level=func.AuthLevel.FUNCTION)
def sp_mirror_now(req: func.HttpRequest) -> func.HttpResponse:
    minutes = int(req.params.get("minutes", "100"))
    try:
        result = sp_mirror.run_mirror(minutes=minutes)
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("sp_mirror_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.timer_trigger(schedule="%LAKE_CRON%", arg_name="timer", run_on_startup=False)
def nightly_lake_sync(timer: func.TimerRequest):
    result = lake_export.run()
    logging.info("lake sync result: %s", json.dumps(result)[:2000])
    if result["errors"]:
        logging.error("lake sync had %d errors", len(result["errors"]))


# The by-store sales view (mart source) receives the prior day's data during
# the MORNING, after the 04:45 nightly sync has already run - so the marts
# trailed upstream by an extra day. This top-up re-pulls just the mart source
# views and rebuilds the marts so the lake is current from mid-morning.
TOPUP_VIEWS = {"PolygonRedcatNetSalesByStoreDailyView", "RestokeLaborCost",
               "ReviewTrackersReviews", "ReviewTrackersCompetitorReviews",
               "Venue_Master"}


def _topup():
    sync = lake_export.run(only=TOPUP_VIEWS)
    built = marts.build_all()
    return {"sync": sync, "marts": built}


@app.timer_trigger(schedule="%TOPUP_CRON%", arg_name="timer", run_on_startup=False)
def midmorning_topup(timer: func.TimerRequest):
    result = _topup()
    logging.info("topup result: %s", json.dumps(result)[:2000])
    if result["sync"]["errors"]:
        logging.error("topup sync had %d errors", len(result["sync"]["errors"]))


@app.timer_trigger(schedule="%TOPUP2_CRON%", arg_name="timer", run_on_startup=False)
def early_topup(timer: func.TimerRequest):
    """07:45: catch prior-day data that lands upstream just after the nightly."""
    result = _topup()
    logging.info("early topup result: %s", json.dumps(result)[:1200])


@app.timer_trigger(schedule="%TOPUP3_CRON%", arg_name="timer", run_on_startup=False)
def evening_topup(timer: func.TimerRequest):
    """16:30: same-day upstream late arrivals so evening/next-morning reads are current."""
    result = _topup()
    logging.info("evening topup result: %s", json.dumps(result)[:1200])


@app.route(route="topup_now", auth_level=func.AuthLevel.FUNCTION)
def topup_now(req: func.HttpRequest) -> func.HttpResponse:
    try:
        return func.HttpResponse(json.dumps(_topup(), indent=1),
                                 mimetype="application/json")
    except Exception as e:
        logging.exception("topup_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.route(route="backfill_now", auth_level=func.AuthLevel.FUNCTION)
def backfill_now(req: func.HttpRequest) -> func.HttpResponse:
    view = req.params.get("view")
    m_from = req.params.get("from")
    m_to = req.params.get("to")
    force = req.params.get("force") == "1"
    if not (view and m_from and m_to):
        return func.HttpResponse('{"error":"need view, from=YYYY-MM, to=YYYY-MM"}',
                                 status_code=400, mimetype="application/json")
    try:
        result = lake_export.backfill(view, m_from, m_to, force=force)
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("backfill_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")


@app.route(route="lake_sync_now", auth_level=func.AuthLevel.FUNCTION)
def lake_sync_now(req: func.HttpRequest) -> func.HttpResponse:
    only = req.params.get("only")
    only_set = {s.strip() for s in only.split(",")} if only else None
    try:
        result = lake_export.run(only=only_set)
        return func.HttpResponse(json.dumps(result, indent=1), mimetype="application/json")
    except Exception as e:
        logging.exception("lake_sync_now failed")
        return func.HttpResponse(json.dumps({"error": str(e)[:500]}), status_code=500,
                                 mimetype="application/json")
