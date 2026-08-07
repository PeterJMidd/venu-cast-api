# -*- coding: utf-8 -*-
"""AU business-day maths for template due rules.
Rules: {"type":"bd","n":3}  -> 3rd business day of the month (n=-1: last BD)
       {"type":"dom","day":21,"roll":"forward"} -> the 21st, rolled to the next
       business day if it lands on a weekend/public holiday."""
import datetime as dt
import holidays

_STATE = "VIC"  # head-office calendar; national + VIC observed days


def _holidays(year):
    return holidays.Australia(subdiv=_STATE, years=[year, year + 1])


def is_business_day(d):
    return d.weekday() < 5 and d not in _holidays(d.year)


def business_days_of_month(year, month):
    d = dt.date(year, month, 1)
    out = []
    while d.month == month:
        if is_business_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def business_day_of_month(year, month, n):
    days = business_days_of_month(year, month)
    if n == 0:
        n = 1  # treat a misconfigured 0 as "first business day"
    if n > 0:
        return days[min(n, len(days)) - 1]
    return days[max(len(days) + n, 0)]


def roll_forward(d):
    while not is_business_day(d):
        d += dt.timedelta(days=1)
    return d


def due_date_for(rule, year, month):
    """Evaluate a due_rule for the given period month.
    {"month":"next"} shifts evaluation to the following month - used for
    close tasks (period July's 'WD+2' lands in early August)."""
    if rule.get("month") == "next":
        month += 1
        if month == 13:
            month, year = 1, year + 1
    kind = rule.get("type")
    if kind == "bd":
        return business_day_of_month(year, month, int(rule.get("n", 1)))
    if kind == "dom":
        import calendar as _cal
        day = min(int(rule.get("day", 1)), _cal.monthrange(year, month)[1])
        d = dt.date(year, month, day)
        if rule.get("roll", "forward") == "forward":
            d = roll_forward(d)
        return d
    raise ValueError("unknown due rule type: %r" % kind)
