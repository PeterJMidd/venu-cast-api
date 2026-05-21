import os
import io
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, date
from collections import Counter

import msal
import requests

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# SharePoint list internal column name mapping
COL_DATE = "Date"
COL_VENUE = "Venue"
COL_SHIFT_LEADER = "What_x0020_is_x0020_your_x0020_n"
COL_SHIFT = "Did_x0020_you_x0020_complete"
COL_SHIFT_SALES = "What_x0020_was_x0020_the_x0020_d"
COL_TRADE_FEEDBACK = "Were_x0020_there_x0020_any_x0020"
COL_GUEST_FEEDBACK = "What_x0020_impacted_x0020_guest_"
COL_OPS_FEEDBACK = "What_x0020_operational_x0020_cha"
COL_ROSTER_FEEDBACK = "Team_x0020__x002b__x0020_Rosteri"
COL_STAR = "Star_x0020_of_x0020_the_x0020_sh"


def get_access_token() -> str:
    authority = f"https://login.microsoftonline.com/{os.environ['TENANT_ID']}"
    app = msal.ConfidentialClientApplication(
        os.environ["CLIENT_ID"],
        authority=authority,
        client_credential=os.environ["CLIENT_SECRET"],
    )
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in result:
        raise RuntimeError(f"Auth failed: {result.get('error_description', result)}")
    return result["access_token"]


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Prefer": "HonorNonIndexedQueriesWarningMayFailRandomly",
    }


def get_site_id(token: str) -> str:
    site_path = os.environ["SHAREPOINT_SITE"]
    url = f"{GRAPH_BASE}/sites/{site_path}"
    resp = requests.get(url, headers=_headers(token))
    resp.raise_for_status()
    return resp.json()["id"]


def get_list_id(token: str, site_id: str) -> str:
    list_name = os.environ["SHAREPOINT_LIST_NAME"]
    url = f"{GRAPH_BASE}/sites/{site_id}/lists/{list_name}"
    resp = requests.get(url, headers=_headers(token))
    resp.raise_for_status()
    return resp.json()["id"]


def _normalize_shift(fields: dict) -> dict:
    """Map SharePoint internal column names to friendly names for the email builder."""
    return {
        "ShiftLeader": fields.get(COL_SHIFT_LEADER, ""),
        "Shift": fields.get(COL_SHIFT, ""),
        "ShiftSales": fields.get(COL_SHIFT_SALES),
        "TradeSalesFeedback": fields.get(COL_TRADE_FEEDBACK, ""),
        "GuestExperienceFeedback": fields.get(COL_GUEST_FEEDBACK, ""),
        "OperationsDailyTaskFeedback": fields.get(COL_OPS_FEEDBACK, ""),
        "TeamRosteringFeedback": fields.get(COL_ROSTER_FEEDBACK, ""),
        "StarOfTheShift": fields.get(COL_STAR, ""),
        "Venue": fields.get(COL_VENUE, ""),
        "Date": fields.get(COL_DATE, ""),
    }


FIELDS_SELECT = ",".join([
    COL_VENUE, "VenueLookupId", COL_DATE, COL_SHIFT_LEADER, COL_SHIFT,
    COL_SHIFT_SALES, COL_TRADE_FEEDBACK, COL_GUEST_FEEDBACK,
    COL_OPS_FEEDBACK, COL_ROSTER_FEEDBACK, COL_STAR,
])


def _fetch_all_items_for_date_range(token: str, site_id: str, list_id: str, start_date: str, end_date: str) -> list[dict]:
    """Fetch all list items within a date range. Venue filtering is done in Python
    because the Venue column is a lookup and cannot be filtered via Graph API.
    Must use $select on fields to resolve lookup columns like Venue."""
    filter_query = f"fields/{COL_DATE} ge '{start_date}' and fields/{COL_DATE} le '{end_date}'"
    url = f"{GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items"
    all_items = []
    params = {"$expand": f"fields($select={FIELDS_SELECT})", "$filter": filter_query, "$top": 200}

    while url:
        resp = requests.get(url, headers=_headers(token), params=params)
        resp.raise_for_status()
        data = resp.json()
        all_items.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
        params = None

    return all_items


def get_shifts_for_date(token: str, site_id: str, list_id: str, target_date: date, venue: str) -> list[dict]:
    date_str = target_date.strftime("%Y-%m-%dT00:00:00Z")
    date_end = target_date.strftime("%Y-%m-%dT23:59:59Z")

    all_items = _fetch_all_items_for_date_range(token, site_id, list_id, date_str, date_end)

    # Filter by venue in Python (lookup columns can't be filtered via Graph API)
    venue_items = [
        item for item in all_items
        if item["fields"].get(COL_VENUE, "").strip() == venue
    ]

    return [_normalize_shift(item["fields"]) for item in venue_items]


def get_mtd_stars(token: str, site_id: str, list_id: str, target_date: date, venue: str) -> list[dict]:
    month_start = target_date.replace(day=1).strftime("%Y-%m-%dT00:00:00Z")
    month_end = target_date.strftime("%Y-%m-%dT23:59:59Z")

    all_items = _fetch_all_items_for_date_range(token, site_id, list_id, month_start, month_end)

    # Filter by venue in Python
    star_names = []
    for item in all_items:
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
    return champions


def download_file(token: str, site_id: str, file_path: str) -> bytes:
    url = f"{GRAPH_BASE}/sites/{site_id}/drive/root:/{file_path}:/content"
    resp = requests.get(url, headers=_headers(token))
    resp.raise_for_status()
    return resp.content


def send_email_graph(token: str, sender: str, recipients: list[str], subject: str, html_body: str):
    url = f"{GRAPH_BASE}/users/{sender}/sendMail"
    to_recipients = [{"emailAddress": {"address": r}} for r in recipients]
    payload = {
        "message": {
            "subject": subject,
            "body": {"contentType": "HTML", "content": html_body},
            "toRecipients": to_recipients,
        },
        "saveToSentItems": False,
    }
    resp = requests.post(url, headers=_headers(token), json=payload)
    if resp.status_code >= 400:
        logging.error(f"Email send failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()
    logging.info(f"Email sent via Graph API from {sender} to {len(recipients)} recipients")
