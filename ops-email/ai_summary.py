import os
import logging
import anthropic


def generate_venue_summary(venue: str, todays_shifts: list[dict], week_shifts: list[dict]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.warning("ANTHROPIC_API_KEY not set, skipping AI summary")
        return ""

    today_text = _format_shifts(todays_shifts, f"Latest {venue} daily summary")
    week_text = _format_shifts(week_shifts, f"{venue} last 7 days comments")

    prompt = f"""You are a sharp, concise operations analyst for Yo-Chi frozen yogurt stores.

Using the {venue} daily summary and the {venue} last 7 days comments below, write a short executive summary in EXACTLY this format:

KEY ISSUES + WEEKLY TRENDS

KEY ISSUES:
<daily issues only from the latest {venue} summary>

WEEKLY TRENDS:
<themes, repeated issues, recurring positives or recurring operational patterns across the last 7 days of {venue} comments>

Rules:
- Do NOT write anything under "KEY ISSUES + WEEKLY TRENDS"
- ONLY use {venue} data provided below
- Do NOT reference any other venue data
- KEY ISSUES must only use the latest {venue} daily summary
- WEEKLY TRENDS must only use the {venue} last 7 days comments
- For WEEKLY TRENDS, identify repeated themes across trade, guest experience, operations, rostering, and team comments where relevant
- Only call something a trend if it appears recurring or repeated across the 7 day period
- If no clear repeated pattern exists, say "No clear weekly trend identified"
- If no major daily issues exist, say "No major issues noted"
- Keep it brief, direct and commercial
- Plain text only
- No markdown
- No tables
- Put headings on separate lines

{today_text}

{week_text}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception:
        logging.exception(f"AI summary failed for {venue}")
        return ""


def _format_shifts(shifts: list[dict], label: str) -> str:
    if not shifts:
        return f"{label}: No data available."

    lines = [f"{label}:"]
    for s in shifts:
        date = s.get("Date", "")
        if date:
            date = date[:10]
        shift_type = s.get("Shift", "Unknown")
        leader = s.get("ShiftLeader", "")
        sales = s.get("ShiftSales", "$0")
        trade = s.get("TradeSalesFeedback", "").strip()
        guest = s.get("GuestExperienceFeedback", "").strip()
        ops = s.get("OperationsDailyTaskFeedback", "").strip()
        roster = s.get("TeamRosteringFeedback", "").strip()

        lines.append(f"- Date: {date}, Shift: {shift_type}, Leader: {leader}, Sales: {sales}")
        if trade:
            lines.append(f"  Trade: {trade}")
        if guest:
            lines.append(f"  Guest: {guest}")
        if ops:
            lines.append(f"  Ops: {ops}")
        if roster:
            lines.append(f"  Roster: {roster}")

    return "\n".join(lines)
