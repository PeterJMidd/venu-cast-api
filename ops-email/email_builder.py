from datetime import datetime, date


SHIFT_ORDER = {"Day/Afternoon": 0, "Day": 1, "Night": 2}


def _sort_shifts(shifts: list[dict]) -> list[dict]:
    return sorted(shifts, key=lambda s: SHIFT_ORDER.get(s.get("Shift", ""), 99))


def _format_currency(value) -> str:
    if value is None:
        return "$0.00"
    try:
        num = float(str(value).replace("$", "").replace(",", ""))
        return f"${num:,.2f}"
    except (ValueError, TypeError):
        return "$0.00"


def _medal_emoji(place: str) -> str:
    if "Gold" in place:
        return "&#129351;"
    if "Silver" in place:
        return "&#129352;"
    return "&#129353;"


def _feedback_section(label: str, value: str, color: str) -> str:
    text = value.strip() if value and value.strip() else "n/r"
    return f"""
    <tr>
      <td style="padding:0;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td style="background-color:{color};padding:6px 12px;font-size:13px;font-weight:bold;color:#333;">
              {label}
            </td>
          </tr>
          <tr>
            <td style="padding:8px 12px;font-size:13px;color:#444;border-bottom:1px solid #eee;">
              {text}
            </td>
          </tr>
        </table>
      </td>
    </tr>"""


def build_shift_block(shift: dict) -> str:
    leader = shift.get("ShiftLeader", "")
    shift_type = shift.get("Shift", "")
    sales = _format_currency(shift.get("ShiftSales"))

    trade = shift.get("TradeSalesFeedback", "")
    guest = shift.get("GuestExperienceFeedback", "")
    ops = shift.get("OperationsDailyTaskFeedback", "")
    roster = shift.get("TeamRosteringFeedback", "")
    star = shift.get("StarOfTheShift", "")

    return f"""
    <tr>
      <td style="padding:16px 0 0 0;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="border:1px solid #ddd;border-radius:4px;overflow:hidden;">
          <!-- Shift header -->
          <tr>
            <td style="padding:10px 12px;background-color:#f8f8f8;border-bottom:1px solid #ddd;">
              <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td style="font-size:11px;color:#888;text-transform:uppercase;">Venue Leader</td>
                  <td style="font-size:11px;color:#888;text-transform:uppercase;" align="center">Shift</td>
                  <td style="font-size:11px;color:#888;text-transform:uppercase;" align="right">Shift Sales</td>
                </tr>
                <tr>
                  <td style="font-size:14px;font-weight:bold;color:#333;padding-top:4px;">{leader}</td>
                  <td style="font-size:14px;font-weight:bold;color:#333;padding-top:4px;" align="center">{shift_type}</td>
                  <td style="font-size:14px;font-weight:bold;color:#333;padding-top:4px;" align="right">{sales}</td>
                </tr>
              </table>
            </td>
          </tr>
          {_feedback_section("Trade - Sales Feedback", trade, "#f0e6e6")}
          {_feedback_section("Guest Experience Feedback", guest, "#e6f0e6")}
          {_feedback_section("Operations: Daily task feedback", ops, "#e6e6f0")}
          {_feedback_section("Team Rostering Feedback", roster, "#f0ece6")}
          {_feedback_section("Star Of the Shift", star, "#fdf6e3")}
        </table>
      </td>
    </tr>"""


def build_champions_table(champions: list[dict], month_name: str) -> str:
    if not champions:
        return ""

    rows = ""
    for c in champions:
        emoji = _medal_emoji(c["medal"])
        rows += f"""
            <tr>
              <td style="padding:6px 12px;font-size:14px;font-weight:bold;color:#333;">{c['medal']}</td>
              <td style="padding:6px 12px;font-size:14px;color:#333;">{c['place']}</td>
              <td style="padding:6px 12px;font-size:14px;color:#333;">{c['name']}</td>
              <td style="padding:6px 12px;font-size:14px;font-weight:bold;color:#333;" align="center">{c['stars']}</td>
            </tr>"""

    return f"""
    <tr>
      <td style="padding:8px 0 0 0;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#f9f9f9;border:1px solid #ddd;border-radius:4px;">
          <tr>
            <td colspan="4" style="padding:12px;font-size:16px;font-weight:bold;color:#333;text-align:center;border-bottom:1px solid #ddd;">
              {emoji} Shift Champions for the month of {month_name}
            </td>
          </tr>
          <tr style="background-color:#eee;">
            <td style="padding:6px 12px;font-size:12px;font-weight:bold;color:#666;">Place</td>
            <td style="padding:6px 12px;font-size:12px;font-weight:bold;color:#666;"></td>
            <td style="padding:6px 12px;font-size:12px;font-weight:bold;color:#666;">Team Member</td>
            <td style="padding:6px 12px;font-size:12px;font-weight:bold;color:#666;" align="center">Stars</td>
          </tr>
          {rows}
        </table>
      </td>
    </tr>"""


def _build_ai_summary_section(summary: str) -> str:
    if not summary:
        return ""
    # Style the headings, convert plain text to HTML
    lines = summary.split("\n")
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("KEY ISSUES + WEEKLY TRENDS", "KEY ISSUES:", "WEEKLY TRENDS:"):
            html_lines.append(f'<strong style="font-size:13px;color:#2c3e50;">{stripped}</strong>')
        elif stripped:
            html_lines.append(f'<span style="font-size:13px;color:#444;">{stripped}</span>')
        else:
            html_lines.append('<br>')
    html_summary = "<br>".join(html_lines)
    return f"""
    <tr>
      <td style="padding:16px 0 0 0;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="border:1px solid #ddd;border-radius:4px;overflow:hidden;">
          <tr>
            <td style="background-color:#2c3e50;padding:10px 12px;font-size:14px;font-weight:bold;color:#ffffff;">
              &#129302; AI Insights
            </td>
          </tr>
          <tr>
            <td style="padding:12px;font-size:13px;color:#444;line-height:1.6;background-color:#f8fafe;">
              {html_summary}
            </td>
          </tr>
        </table>
      </td>
    </tr>"""


def build_email_html(venue: str, target_date: date, champions: list[dict], shifts: list[dict], ai_summary: str = "") -> str:
    now_str = target_date.strftime("%d/%m/%Y") + " 05:01 AM"
    report_date_str = target_date.strftime("%-d %B %Y")
    month_name = target_date.strftime("%B")
    sorted_shifts = _sort_shifts(shifts)

    shift_blocks = ""
    for shift in sorted_shifts:
        shift_blocks += build_shift_block(shift)

    champions_html = build_champions_table(champions, month_name)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#f4f4f4;">
    <tr>
      <td align="center" style="padding:20px 0;">
        <table width="600" cellpadding="0" cellspacing="0" border="0" style="background-color:#ffffff;border-radius:6px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
          <!-- Header -->
          <tr>
            <td style="background-color:#5a5a5a;padding:16px 20px;">
              <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td style="font-size:20px;font-weight:bold;color:#ffffff;">{venue}</td>
                  <td style="font-size:13px;color:#cccccc;" align="right">Report for shift comments: {report_date_str}</td>
                </tr>
              </table>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="padding:0 20px 20px 20px;">
              {champions_html}
              {shift_blocks}
              {_build_ai_summary_section(ai_summary)}
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="background-color:#f0f0f0;padding:12px 20px;text-align:center;font-size:11px;color:#999;">
              Yo-Chi Operations Daily Update
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
