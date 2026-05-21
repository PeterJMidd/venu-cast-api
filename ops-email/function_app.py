import os
import logging
from datetime import date, datetime, timedelta, timezone

import azure.functions as func

from sharepoint_client import (
    get_access_token,
    get_site_id,
    get_list_id,
    download_file,
    send_email_graph,
    _fetch_all_items_for_date_range,
    COL_VENUE,
    COL_STAR,
    _normalize_shift,
)
from config_reader import read_venue_configs
from email_builder import build_email_html
from ai_summary import generate_venue_summary
from collections import Counter

app = func.FunctionApp()


def _process_venue(token, venue, sender, recipients, query_date,
                   all_day_items, all_mtd_items, all_week_items):
    # Filter shifts for this venue from pre-fetched data
    venue_shifts = [
        _normalize_shift(item["fields"])
        for item in all_day_items
        if item["fields"].get(COL_VENUE, "").strip() == venue
    ]

    # Calculate MTD stars for this venue from pre-fetched data
    star_names = []
    for item in all_mtd_items:
        if item["fields"].get(COL_VENUE, "").strip() != venue:
            continue
        star = item["fields"].get(COL_STAR, "").strip()
        if star:
            star_names.append(star)

    counts = Counter(star_names)
    ranked = counts.most_common(3)
    medals = ["Gold medal", "Silver", "Bronze"]
    places = ["1st", "2nd", "3rd"]
    champions = []
    for i, (name, count) in enumerate(ranked):
        champions.append({
            "medal": medals[i],
            "place": places[i],
            "name": name,
            "stars": count,
        })

    # Get 7-day shifts for this venue for AI trend analysis
    venue_week_shifts = [
        _normalize_shift(item["fields"])
        for item in all_week_items
        if item["fields"].get(COL_VENUE, "").strip() == venue
    ]

    # Generate AI summary (today's trade + 7-day trends)
    ai_summary = generate_venue_summary(venue, venue_shifts, venue_week_shifts)
    logging.info(f"AI summary for {venue}: {len(ai_summary)} chars")

    html = build_email_html(venue, query_date, champions, venue_shifts, ai_summary)
    subject = f"{venue} - Operations Update {query_date.strftime('%d/%m/%Y')}"

    send_email_graph(token, sender, recipients, subject, html)
    logging.info(f"Sent ops email for {venue} ({len(venue_shifts)} shifts)")


@app.timer_trigger(schedule="0 0 5 * * *", arg_name="timer", run_on_startup=False)
def ops_email_timer(timer: func.TimerRequest) -> None:
    if timer.past_due:
        logging.warning("Timer is past due — running missed execution now")
    logging.info("Operations email function triggered")

    # Use AEST (UTC+10) so "yesterday" is correct for Australian time
    AEST = timezone(timedelta(hours=10))
    today = datetime.now(AEST).date()
    query_date = today - timedelta(days=1)  # query yesterday's shifts
    logging.info(f"AEST today={today}, querying shifts for {query_date}")

    try:
        token = get_access_token()
        site_id = get_site_id(token)
        list_id = get_list_id(token, site_id)

        config_path = os.environ.get("CONFIG_FILE_PATH", "OpsEmailConfig.xlsx")
        excel_bytes = download_file(token, site_id, config_path)
        venue_configs = read_venue_configs(excel_bytes)

        # Fetch all items for the day in one call
        date_str = query_date.strftime("%Y-%m-%dT00:00:00Z")
        date_end = query_date.strftime("%Y-%m-%dT23:59:59Z")
        all_day_items = _fetch_all_items_for_date_range(token, site_id, list_id, date_str, date_end)
        logging.info(f"Fetched {len(all_day_items)} shift items for {query_date}")

        # Fetch all MTD items in one call
        month_start = query_date.replace(day=1).strftime("%Y-%m-%dT00:00:00Z")
        all_mtd_items = _fetch_all_items_for_date_range(token, site_id, list_id, month_start, date_end)
        logging.info(f"Fetched {len(all_mtd_items)} MTD items for month")

        # Fetch last 7 days for AI trend analysis
        week_start = (query_date - timedelta(days=6)).strftime("%Y-%m-%dT00:00:00Z")
        all_week_items = _fetch_all_items_for_date_range(token, site_id, list_id, week_start, date_end)
        logging.info(f"Fetched {len(all_week_items)} items for 7-day trends")

        failed_venues = []
        for config in venue_configs:
            venue = config["venue"]
            sender = config["sender_email"]
            recipients = config["recipients"]

            if not recipients:
                logging.warning(f"No recipients for {venue}, skipping")
                continue

            # Fallback if sender is a broken formula or empty
            if not sender or sender.startswith("=") or "@" not in sender:
                sender = os.environ.get("FALLBACK_SENDER", "peterm@yochi.com.au")
                logging.warning(f"Invalid sender for {venue}, using fallback: {sender}")

            try:
                _process_venue(token, venue, sender, recipients, query_date,
                               all_day_items, all_mtd_items, all_week_items)
            except Exception:
                logging.exception(f"Failed to process venue {venue}, continuing to next")
                failed_venues.append(venue)
                continue

        if failed_venues:
            logging.warning(f"Failed venues: {', '.join(failed_venues)}")

    except Exception:
        logging.exception("Operations email function failed")
        raise
