"""Attendance, holiday, and leave-balance lookups for the signed-in employee.

The caller must pass the employee id from the authenticated session. A number
or name in the chat message is never used as the subject.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.clock import org_timezone_name, org_today
from app.models import Approval, Attendance, Department, Employee, Holiday, LeaveRequest, Project, ProjectAssignment
from app.services.holidays import OFFICE_COUNTRY, holidays_for_employee, list_holidays_for_employee
from app.services.leave_engine import balance_payload, get_balance

OPEN_LEAVE = ("pending_manager", "pending_hr", "approved")
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH = (
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"


class WindowError(ValueError):
    """The message asked for a date range that cannot be queried."""


def _week_bounds(day: date) -> tuple[date, date]:
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def _parse_iso(year: str, month: str, day: str) -> date:
    try:
        return date(int(year), int(month), int(day))
    except ValueError as exc:
        raise WindowError(f"{year}-{month}-{day} is not a real date.") from exc


def _month_number(name: str) -> int:
    return MONTHS[name.lower()]


def _inferred_year(month: int, today: date, *, previous: bool) -> int:
    """A month name without a year uses the organization clock, not the server clock."""
    if previous:
        return today.year if today.month > month else today.year - 1
    return today.year if today.month >= month else today.year - 1


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    start = date(year, 3 * (quarter - 1) + 1, 1)
    _s, end = _month_bounds(year, 3 * quarter)
    return start, end


def _quarter_window(text: str, today: date) -> tuple[date, date, str] | None:
    """Calendar quarters: Q3 2026, third quarter, previous/last/this quarter. Comparisons return None."""
    if re.search(r"\b(compare|comparison|versus|vs\.?)\b", text) and len(re.findall(r"\bq[1-4]\b", text)) >= 2:
        return None
    explicit = re.search(r"\bq([1-4])\b(?:\s+(\d{4}))?", text)
    ordinal = re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b(?:\s+(?:of\s+)?(\d{4}))?", text)
    relative = re.search(r"\b(previous|last|this|current)\s+quarter\b", text)
    if explicit:
        quarter = int(explicit.group(1))
        year = int(explicit.group(2)) if explicit.group(2) else today.year
        if not explicit.group(2) and quarter > (today.month - 1) // 3 + 1:
            year -= 1
        start, end = _quarter_bounds(year, quarter)
        return start, end, f"Q{quarter} {year}"
    if ordinal:
        quarter = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}[ordinal.group(1)]
        year = int(ordinal.group(2)) if ordinal.group(2) else today.year
        if not ordinal.group(2) and quarter > (today.month - 1) // 3 + 1:
            year -= 1
        start, end = _quarter_bounds(year, quarter)
        return start, end, f"Q{quarter} {year}"
    if relative:
        current = (today.month - 1) // 3 + 1
        if relative.group(1) in {"previous", "last"}:
            quarter, year = (current - 1, today.year) if current > 1 else (4, today.year - 1)
            start, end = _quarter_bounds(year, quarter)
            return start, end, f"previous quarter (Q{quarter} {year})"
        start, _end = _quarter_bounds(today.year, current)
        return start, today, f"this quarter (Q{current} {today.year} to date)"
    return None


def _named_day(month_name: str, day_text: str, year_text: str | None, today: date) -> date:
    month = _month_number(month_name)
    year = int(year_text) if year_text else _inferred_year(month, today, previous=False)
    try:
        return date(year, month, int(day_text))
    except ValueError as exc:
        raise WindowError(f"{month_name} {day_text}, {year} is not a real date.") from exc


def _ordered(start: date, end: date, label: str) -> tuple[date, date, str]:
    if end < start:
        raise WindowError("The end date is before the start date.")
    return start, end, label


def parse_asked_window(message: str, today: date | None = None) -> tuple[date, date, str] | None:
    """Return an explicit window, or None when the message does not name one.

    Precedence is explicit dates, then named months, then relative periods.
    A missing window stays None so the caller can inherit conversation context
    before using a default.
    """
    today = today or org_today()
    text = message.lower()
    found = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    dates = [_parse_iso(year, month, day) for year, month, day in found]
    if len(dates) >= 2 and (" to " in text or "between" in text):
        return _ordered(dates[0], dates[1], "requested range")
    if len(dates) == 1:
        return dates[0], dates[0], "that date"

    # Slash month/year forms used in chat: "10/2026", "10/26".
    slash_month = re.search(r"\b(0?[1-9]|1[0-2])/(20\d{2}|\d{2})\b", text)
    if slash_month:
        month = int(slash_month.group(1))
        year_token = slash_month.group(2)
        year = int(year_token) if len(year_token) == 4 else 2000 + int(year_token)
        start, end = _month_bounds(year, month)
        return start, end, start.strftime("%B %Y")

    # Natural single-day forms: "Saturday 3 October 2026", "3 October 2026".
    named_day = re.search(
        rf"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if named_day and not re.search(r"\b(to|through|until|and|between)\b", text):
        day = _named_day(named_day.group(2), named_day.group(1), named_day.group(3), today)
        return day, day, day.isoformat()

    natural = re.search(
        rf"\b(?:between|from)\s+({_MONTH})\s+{_DAY}(?:,?\s*(\d{{4}}))?\s+(?:and|to)\s+({_MONTH})\s+{_DAY}(?:,?\s*(\d{{4}}))?",
        text,
    )
    if natural:
        start = _named_day(natural.group(1), natural.group(2), natural.group(3), today)
        end = _named_day(natural.group(4), natural.group(5), natural.group(6), today)
        return _ordered(start, end, "requested range")

    # Day-first ranges such as "27 Oct to 2 Nov 2026" or "between 1 Sep and 15 Sep 2026".
    day_range = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:,?\s*(\d{{4}}))?\s*(?:to|-|–|—|through|thru|until|and)\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if day_range:
        end_year = day_range.group(6) or day_range.group(3)
        end = _named_day(day_range.group(5), day_range.group(4), end_year, today)
        start_year = day_range.group(3) or (str(end.year) if end_year else None)
        start = _named_day(day_range.group(2), day_range.group(1), start_year, today)
        if start > end and not day_range.group(3):
            start = date(start.year - 1, start.month, start.day)
        return _ordered(start, end, "requested range")

    # Quarters: "Q3 2026", "third quarter", "previous quarter (Jul–Sep 2026)".
    quarter = _quarter_window(text, today)
    if quarter:
        return quarter

    # "final/last two weeks of October 2026" (calendar last 14 days of the month).
    two_weeks = re.search(
        rf"\b(?:final|last)\s+two\s+weeks?\s+of\s+({_MONTH})(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if two_weeks:
        month = _month_number(two_weeks.group(1))
        year = int(two_weeks.group(2)) if two_weeks.group(2) else _inferred_year(month, today, previous=False)
        _start, month_end = _month_bounds(year, month)
        start = month_end - timedelta(days=13)
        return start, month_end, f"final two weeks of {_start.strftime('%B %Y')}"

    # "first week of November 2026" style windows.
    week_of = re.search(rf"\b(first|second|third|fourth|last)\s+week\s+of\s+({_MONTH})(?:,?\s*(\d{{4}}))?\b", text)
    if week_of:
        month = _month_number(week_of.group(2))
        year = int(week_of.group(3)) if week_of.group(3) else _inferred_year(month, today, previous=False)
        month_start, month_end = _month_bounds(year, month)
        ordinal = week_of.group(1)
        if ordinal == "last":
            start, end = month_end - timedelta(days=6), month_end
        else:
            offset = {"first": 0, "second": 7, "third": 14, "fourth": 21}[ordinal]
            start = month_start + timedelta(days=offset)
            end = min(start + timedelta(days=6), month_end)
        return start, end, f"{ordinal} week of {month_start.strftime('%B %Y')}"

    # Month ranges written with a dash or "to" and no "and": "Jul–Sep 2026", "April to June 2026".
    month_range = re.search(
        rf"\b({_MONTH})(?:\s+(\d{{4}}))?\s*(?:–|—|-|to|through|thru|until)\s*({_MONTH})(?:\s+(\d{{4}}))?\b(?!\s+\d{{1,2}}\b)",
        text,
    )
    if month_range and not re.search(rf"\b\d{{1,2}}\s+{month_range.group(1)}\b", text):
        first_month = _month_number(month_range.group(1))
        last_month = _month_number(month_range.group(3))
        if month_range.group(4):
            last_year = int(month_range.group(4))
        elif month_range.group(2):
            last_year = int(month_range.group(2)) + (1 if last_month < first_month else 0)
        else:
            last_year = _inferred_year(last_month, today, previous=False)
        if month_range.group(2):
            first_year = int(month_range.group(2))
        else:
            first_year = last_year - 1 if last_month < first_month else last_year
        start, _end = _month_bounds(first_year, first_month)
        _start, end = _month_bounds(last_year, last_month)
        label = (
            f"{date(first_year, first_month, 1).strftime('%B %Y')} to "
            f"{date(last_year, last_month, 1).strftime('%B %Y')}"
        )
        return _ordered(start, end, label)

    # Explicit start bound ending at the organization / reference date.
    # e.g. "from 1 November 2026 through the organization date"
    org_bound = re.search(
        rf"\b(?:from|between)\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:,?\s*(\d{{4}}))?\s+"
        rf"(?:to|through|thru|until)\s+(?:the\s+)?(?:organi[sz]ation|reference)\s+date\b",
        text,
    )
    if org_bound:
        start = _named_day(org_bound.group(2), org_bound.group(1), org_bound.group(3), today)
        return _ordered(start, today, "requested range through organization date")
    org_bound_md = re.search(
        rf"\b(?:from|between)\s+({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\s+"
        rf"(?:to|through|thru|until)\s+(?:the\s+)?(?:organi[sz]ation|reference)\s+date\b",
        text,
    )
    if org_bound_md:
        start = _named_day(org_bound_md.group(1), org_bound_md.group(2), org_bound_md.group(3), today)
        return _ordered(start, today, "requested range through organization date")

    # Day-Month-Year such as "7 Nov 2026" / "through 7 November 2026" before month-only windows.
    # Do not treat "from <date> through …" as year-to-date; that needs an explicit YTD cue.
    day_first = re.search(rf"\b(\d{{1,2}})\s+({_MONTH})(?:,?\s*(\d{{4}}))?\b", text)
    if (
        day_first
        and re.search(r"\b(through|thru|until|up to|as of)\b", text)
        and re.search(r"\b(year-?to-?date|ytd|year to date)\b", text)
        and not re.search(r"\b(?:from|between)\b", text)
    ):
        end = _named_day(day_first.group(2), day_first.group(1), day_first.group(3), today)
        start = date(end.year, 1, 1)
        return start, end, f"year to date through {end.isoformat()}"
    if re.search(r"\b(year-?to-?date|ytd|year to date)\b", text):
        end = today
        through = re.search(rf"\b(?:through|thru|until|up to|as of)\s+(\d{{4}}-\d{{2}}-\d{{2}})\b", text)
        if through:
            end = date.fromisoformat(through.group(1))
        elif day_first and not re.search(r"\b(?:from|between)\b", text):
            end = _named_day(day_first.group(2), day_first.group(1), day_first.group(3), today)
        return date(end.year, 1, 1), end, f"year to date through {end.isoformat()}"

    # Same-month inclusive day ranges before a single-day match: "November 1–7, 2026",
    # "Nov 1-7 2026", "November 1 to 7, 2026".
    same_month = re.search(
        rf"\b({_MONTH})\s+{_DAY}\s*(?:–|—|-|to|through|thru|until|and)\s*{_DAY}(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if same_month:
        year = same_month.group(4)
        start = _named_day(same_month.group(1), same_month.group(2), year, today)
        end = _named_day(same_month.group(1), same_month.group(3), year, today)
        return _ordered(start, end, "requested range")

    single_day = re.search(rf"\b({_MONTH})\s+{_DAY}(?:,?\s*(\d{{4}}))?\b", text)
    if single_day:
        day = _named_day(single_day.group(1), single_day.group(2), single_day.group(3), today)
        return day, day, "that date"

    bare_months = []
    for match in re.finditer(rf"\b(?:(last|previous)\s+)?({_MONTH})\b(?:\s+(\d{{4}}))?", text):
        if re.match(r"\s+\d{1,2}\b", text[match.end() :]):
            continue
        if match.group(2) == "may" and not match.group(1) and not match.group(3):
            continue
        bare_months.append(match)
    if len(bare_months) >= 2 and re.search(r"\band\b", text):
        first, last = bare_months[0], bare_months[-1]
        first_month = _month_number(first.group(2))
        last_month = _month_number(last.group(2))
        first_year = int(first.group(3)) if first.group(3) else _inferred_year(first_month, today, previous=bool(first.group(1)))
        if last.group(3):
            last_year = int(last.group(3))
        elif last_month < first_month:
            last_year = first_year + 1
        else:
            last_year = first_year
        start, _end = _month_bounds(first_year, first_month)
        _start, end = _month_bounds(last_year, last_month)
        label = (
            f"{date(first_year, first_month, 1).strftime('%B %Y')} to "
            f"{date(last_year, last_month, 1).strftime('%B %Y')}"
        )
        return _ordered(start, end, label)

    named_month = re.search(
        rf"\b(?:(last|previous)\s+)?({_MONTH})\b(?:\s+(\d{{4}}))?(?!\s+\d{{1,2}}\b)",
        text,
    )
    if named_month and not (named_month.group(2) == "may" and not named_month.group(1) and not named_month.group(3)):
        month = _month_number(named_month.group(2))
        if named_month.group(3):
            year = int(named_month.group(3))
        else:
            year = _inferred_year(month, today, previous=bool(named_month.group(1)))
        start, end = _month_bounds(year, month)
        return start, end, f"{date(year, month, 1).strftime('%B %Y')}"

    count = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+days?\b", text)
    if count:
        days = int(count.group(1))
        if days < 1 or days > 366:
            raise WindowError("Ask for a range between 1 and 366 days.")
        return today - timedelta(days=days - 1), today, f"last {days} days"

    named = re.search(
        r"\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
    )
    if named:
        weekday = WEEKDAYS[named.group(2)]
        delta = (today.weekday() - weekday) % 7
        if named.group(1) == "last":
            if delta == 0:
                delta = 7
            day = today - timedelta(days=delta)
        elif named.group(1) == "this":
            day = today - timedelta(days=delta)
        else:
            ahead = (weekday - today.weekday()) % 7
            if ahead == 0:
                ahead = 7
            day = today + timedelta(days=ahead)
        return day, day, f"{named.group(1)} {named.group(2)}"

    if "next week" in text:
        start, end = _week_bounds(today + timedelta(days=7))
        return start, end, "next week"
    if "last week" in text:
        start, end = _week_bounds(today - timedelta(days=7))
        return start, end, "last week"
    if "this week" in text:
        start, end = _week_bounds(today)
        return start, end, "this week"
    if "this month" in text:
        return today.replace(day=1), today, "this month"
    # Explicit calendar-month phrasing must win over "last N days" defaults.
    if re.search(r"\b(last|previous)\s+calendar\s+month\b", text) or "last month" in text or "previous month" in text:
        first_this = today.replace(day=1)
        end = first_this - timedelta(days=1)
        return end.replace(day=1), end, "last month"
    # "last/previous N calendar months" → contiguous full calendar months ending last month.
    calendar_months = re.search(
        r"\b(?:last|past|previous)\s+(\d+|one|two|three|four|five|six)\s+calendar\s+months?\b",
        text,
    )
    if calendar_months:
        count = _month_count(calendar_months.group(1))
        if count is None or count < 1 or count > 24:
            raise WindowError("Ask for between 1 and 24 months.")
        # Full prior calendar months only (exclude the current partial month).
        first_this = today.replace(day=1)
        end = first_this - timedelta(days=1)
        series = []
        cursor = end.replace(day=1)
        for _ in range(count):
            month_start, month_end = _month_bounds(cursor.year, cursor.month)
            series.append((month_start, month_end, month_start.strftime("%B %Y")))
            cursor = month_start - timedelta(days=1)
            cursor = cursor.replace(day=1)
        series.reverse()
        return series[0][0], series[-1][1], f"previous {count} calendar months"
    # "last/past N calendar days" (inclusive of today).
    calendar_days = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+calendar\s+days?\b", text)
    if calendar_days:
        count = int(calendar_days.group(1))
        if count < 1 or count > 366:
            raise WindowError("Ask for between 1 and 366 calendar days.")
        start = today - timedelta(days=count - 1)
        return start, today, f"last {count} calendar days"
    if re.search(r"\b(the )?month before\b", text):
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        month_before_end = last_month_end.replace(day=1) - timedelta(days=1)
        return month_before_end.replace(day=1), month_before_end, "month before last"
    months = re.search(r"\b(?:last|past|previous)\s+(\d+|one|two|three|four|five|six)\s+months?\b", text)
    if months and "calendar" not in text:
        count = _month_count(months.group(1))
        if count is None or count < 1 or count > 24:
            raise WindowError("Ask for between 1 and 24 months.")
        series = month_series(today, count)
        return series[0][0], series[-1][1], f"last {count} months"
    if re.search(r"\btoday\b", text):
        return today, today, "today"
    if re.search(r"\byesterday\b", text):
        day = today - timedelta(days=1)
        return day, day, "yesterday"
    if re.search(r"\btomorrow\b", text):
        day = today + timedelta(days=1)
        return day, day, "tomorrow"
    if re.search(r"\blast year\b", text):
        year = today.year - 1
        return date(year, 1, 1), date(year, 12, 31), str(year)
    if re.search(r"\b(this year|entire year|whole year|full year|the year)\b", text):
        return date(today.year, 1, 1), today, str(today.year)
    return None


def _inherited_window(inherited: dict | None, subject: str) -> tuple[date, date, str] | None:
    if not inherited or inherited.get("subject") != subject:
        return None
    if not inherited.get("start") or not inherited.get("end"):
        return None
    return date.fromisoformat(inherited["start"]), date.fromisoformat(inherited["end"]), inherited.get("label") or "earlier range"


_MONTH_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def _month_count(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _MONTH_COUNTS.get(token)


def month_series(today: date, count: int) -> list[tuple[date, date, str]]:
    """The last `count` calendar months, current month truncated to today."""
    periods: list[tuple[date, date, str]] = []
    year = today.year
    month = today.month
    for _ in range(count):
        start, end = _month_bounds(year, month)
        if year == today.year and month == today.month:
            end = today
            label = f"{start.strftime('%B %Y')} (month to date)"
        else:
            label = start.strftime("%B %Y")
        periods.append((start, end, label))
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1
    periods.reverse()
    return periods


def complete_month_series(today: date, count: int) -> list[tuple[date, date, str]]:
    """The last `count` complete calendar months ending before the current month."""
    count = max(1, min(int(count), 24))
    first_this = today.replace(day=1)
    end = first_this - timedelta(days=1)
    series: list[tuple[date, date, str]] = []
    cursor = end.replace(day=1)
    for _ in range(count):
        month_start, month_end = _month_bounds(cursor.year, cursor.month)
        series.append((month_start, month_end, month_start.strftime("%B %Y")))
        cursor = (month_start - timedelta(days=1)).replace(day=1)
    series.reverse()
    return series


def parse_named_month_periods(message: str, today: date | None = None) -> list[tuple[date, date, str]]:
    """Every named calendar month in the message, in order. Keeps months with zero rows."""
    today = today or org_today()
    text = message.lower()
    periods: list[tuple[date, date, str]] = []
    seen: set[tuple[int, int]] = set()
    for match in re.finditer(rf"\b(?:(last|previous)\s+)?({_MONTH})\b(?:\s+(\d{{4}}))?(?!\s+\d{{1,2}}\b)", text):
        name = match.group(2)
        if name == "may" and not match.group(1) and not match.group(3):
            continue
        month = _month_number(name)
        if match.group(3):
            year = int(match.group(3))
        else:
            year = _inferred_year(month, today, previous=bool(match.group(1)))
        key = (year, month)
        if key in seen:
            continue
        seen.add(key)
        start, end = _month_bounds(year, month)
        # Explicitly named months keep the calendar month. Truncation to today is only
        # for relative series such as "last three months".
        label = start.strftime("%B %Y")
        periods.append((start, end, label))
    return periods


def parse_comparison(
    message: str,
    today: date | None = None,
    inherited: dict | None = None,
    subject: str = "attendance",
) -> list[tuple[date, date, str]] | None:
    """Split a comparison into windows. Listed months and last-N-months become one window each."""
    today = today or org_today()
    text = message.lower()
    if not re.search(r"\b(compare|comparison|versus|vs\.?|compared)\b", text):
        return None
    # "Compare those two …" after a month switch: use prior_window + current window.
    if inherited and re.search(r"\b(those two|the two|both (?:months|periods|windows|counts))\b", text):
        prior = inherited.get("prior_window") or {}
        current = None
        if inherited.get("start") and inherited.get("end"):
            try:
                current = (
                    date.fromisoformat(str(inherited["start"])[:10]),
                    date.fromisoformat(str(inherited["end"])[:10]),
                    inherited.get("label") or "current window",
                )
            except Exception:
                current = None
        prior_window = None
        if prior.get("start") and prior.get("end"):
            try:
                prior_window = (
                    date.fromisoformat(str(prior["start"])[:10]),
                    date.fromisoformat(str(prior["end"])[:10]),
                    prior.get("label") or "earlier window",
                )
            except Exception:
                prior_window = None
        if prior_window and current and (
            prior_window[0] != current[0] or prior_window[1] != current[1]
        ):
            return [prior_window, current]
    # "last month ... month before" is the prior two complete calendar months.
    if re.search(r"\blast month\b", text) and re.search(r"\bmonth before\b", text):
        return complete_month_series(today, 2)
    months = re.search(r"\b(?:last|past|previous)\s+(\d+|one|two|three|four|five|six)\s+months?\b", text)
    if months:
        count = _month_count(months.group(1))
        if count is None or count < 1 or count > 24:
            return None
        # Comparisons use complete prior calendar months (exclude the current partial month).
        return complete_month_series(today, count)
    listed = parse_named_month_periods(message, today)
    if len(listed) >= 2:
        return listed
    splitter = re.search(r"\b(?:compared to|compared with|versus|vs\.?|with)\b", text)
    if splitter:
        sides = (text[: splitter.start()], text[splitter.end() :])
    else:
        sides = tuple(re.split(r"\band\b", text, maxsplit=1))
        if len(sides) != 2:
            return None
    windows = []
    for side in sides:
        parsed = parse_asked_window(side, today)
        if parsed is None and inherited and re.search(r"\b(it|that|those)\b", side):
            parsed = _inherited_window(inherited, subject)
        if parsed is None:
            return None
        windows.append(parsed)
    if windows[0][0] == windows[1][0] and windows[0][1] == windows[1][1]:
        return None
    return windows


def _shift_inherited_month(message: str, inherited: dict | None, subject: str) -> tuple[date, date, str] | None:
    """Move one month from the stored window. A named month stays on the normal parser."""
    window = _inherited_window(inherited, subject)
    if window is None:
        return None
    text = message.lower()
    if re.search(rf"\b({_MONTH})\b", text):
        return None
    start, _end, _label = window
    if re.search(r"\b(previous|last)\s+month\b", text):
        month_end = start.replace(day=1) - timedelta(days=1)
        month_start = month_end.replace(day=1)
        return month_start, month_end, month_start.strftime("%B %Y")
    if re.search(r"\bnext\s+month\b", text):
        year = start.year + 1 if start.month == 12 else start.year
        month = 1 if start.month == 12 else start.month + 1
        month_start, month_end = _month_bounds(year, month)
        return month_start, month_end, month_start.strftime("%B %Y")
    return None


def _upcoming_month(message: str, parsed: tuple[date, date, str], today: date) -> tuple[date, date, str]:
    """A past calendar month in a holiday follow-up means the next time that month occurs."""
    start, end, _label = parsed
    if re.search(r"\b(19|20)\d{2}\b", message.lower()):
        return parsed
    if start.day != 1 or end < today:
        if start.day == 1 and end < today:
            start, end = _month_bounds(start.year + 1, start.month)
            return start, end, start.strftime("%B %Y")
    return parsed


def attendance_window(
    message: str,
    today: date | None = None,
    inherited: dict | None = None,
    subject: str = "attendance",
) -> tuple[date, date, str]:
    today = today or org_today()
    shifted = _shift_inherited_month(message, inherited, subject)
    if shifted is not None:
        return shifted
    parsed = parse_asked_window(message, today)
    if parsed is not None:
        return parsed
    previous = _inherited_window(inherited, subject)
    if previous is not None:
        return previous
    return today - timedelta(days=6), today, "last 7 days"


def holiday_window(message: str, today: date | None = None, inherited: dict | None = None) -> tuple[date, date, str]:
    today = today or org_today()
    text = message.lower()
    # Leave-plan follow-ups ("during that period") must reuse the planned leave window.
    if (
        inherited
        and inherited.get("subject") == "leave_plan"
        and inherited.get("start")
        and inherited.get("end")
        and (
            re.search(r"\b(that period|those dates|that window|same period|same dates|during)\b", text)
            or re.search(r"\bholidays?\b", text)
        )
    ):
        start = date.fromisoformat(inherited["start"])
        end = date.fromisoformat(inherited["end"])
        return start, end, inherited.get("label") or f"{start.isoformat()} to {end.isoformat()}"
    shifted = _shift_inherited_month(message, inherited, "holiday")
    if shifted is not None:
        return shifted
    parsed = parse_asked_window(message, today)
    if parsed is not None:
        return _upcoming_month(message, parsed, today)
    previous = _inherited_window(inherited, "holiday")
    if previous is not None:
        return previous
    previous_plan = _inherited_window(inherited, "leave_plan")
    if previous_plan is not None and re.search(r"\b(that|those|it|period|window|dates)\b", text):
        return previous_plan
    return today, today + timedelta(days=366), "upcoming"


def _scheduled_working_days(db: Session, employee: Employee | None, start: date, end: date) -> dict:
    """Weekdays through today, excluding published holidays when that office calendar exists.

    Weekends are never scheduled. A missing holiday calendar means the scheduled
    denominator is not reliable, so the caller keeps the recorded-row percentage.
    """
    today = org_today()
    window_end = min(end, today)
    empty = {
        "calendar_status": "missing",
        "scheduled_working_days": None,
        "scheduled_attendance_percentage": None,
        "missing_scheduled_dates": [],
        "holiday_dates": [],
    }
    if employee is None or window_end < start:
        return empty
    calendar = holidays_for_employee(db, employee, start, window_end)
    if calendar.status != "configured":
        empty["calendar_status"] = calendar.status
        return empty
    scheduled: list[date] = []
    cursor = start
    while cursor <= window_end:
        if cursor.weekday() < 5 and cursor not in calendar.dates:
            scheduled.append(cursor)
        cursor += timedelta(days=1)
    return {
        "calendar_status": "configured",
        "scheduled_working_days": len(scheduled),
        "scheduled_dates": scheduled,
        "holiday_dates": sorted(day.isoformat() for day in calendar.dates),
        "missing_scheduled_dates": [],
    }


def _attendance_period(
    db: Session,
    employee_id: int,
    start: date,
    end: date,
    label: str,
    employee: Employee | None = None,
) -> dict:
    rows = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee_id,
            Attendance.work_date >= start,
            Attendance.work_date <= end,
        )
        .order_by(Attendance.work_date, Attendance.id)
        .all()
    )
    counts = {"present": 0, "remote": 0, "absent": 0, "sick": 0, "leave": 0, "late": 0, "other": 0}
    records = []
    for row in rows:
        status = row.status if row.status in counts else "other"
        counts[status] = counts.get(status, 0) + 1
        records.append({"date": row.work_date.isoformat(), "status": row.status, "hours": row.hours})
    recorded = len(records)
    attended = counts["present"] + counts["remote"] + counts["late"]
    stored = {row.work_date for row in rows}
    missing = []
    cursor = start
    while cursor <= end:
        if cursor not in stored:
            missing.append(cursor.isoformat())
        cursor += timedelta(days=1)
    hours = sum((row.hours or 0) for row in rows)
    if isinstance(hours, float) and hours.is_integer():
        hours = int(hours)
    schedule = _scheduled_working_days(db, employee, start, end)
    scheduled_dates = schedule.pop("scheduled_dates", [])
    attended_scheduled = sum(
        1
        for row in rows
        if row.work_date in set(scheduled_dates) and row.status in {"present", "remote", "late"}
    )
    scheduled_count = schedule.get("scheduled_working_days")
    schedule["missing_scheduled_dates"] = [day.isoformat() for day in scheduled_dates if day not in stored]
    schedule["scheduled_attendance_percentage"] = (
        round(100 * attended_scheduled / scheduled_count, 1) if scheduled_count else None
    )
    recorded_percentage = round(100 * attended / recorded, 1) if recorded else None
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "range_label": label,
        "month_to_date": label == "this month",
        "records": records,
        "missing_dates": missing,
        "counts": counts,
        "recorded_days": recorded,
        "recorded_hours": hours,
        "attendance_percentage": recorded_percentage,
        "recorded_attendance_percentage": recorded_percentage,
        **schedule,
    }


# remote_zero_hours and hours_vs_scheduled are handled after focus application.
_STATUS_OF_FOCUS = {
    "present": "present",
    "present_count": "present",
    "remote": "remote",
    "remote_hours": "remote",
    "remote_last": "remote",
    "remote_history": "remote",
    "late": "late",
    "late_count": "late",
    "absent": "absent",
    "absent_dates": "absent",
    "sick": "sick",
    "leave_status": "leave",
}
_STORED_SPAN_FOCUSES = set(_STATUS_OF_FOCUS) | {
    "hours",
    "hours_vs_scheduled",
    "remote_zero_hours",
    "missed",
    "remote_share",
    "early_departure",
}
_ATTENDANCE_INTENT = {
    "missing": "missing_attendance",
    "missed": "missed_working_days",
    "reconcile": "attendance_leave_reconciliation",
    "leave_overlap": "leave_attendance_overlap",
    "year_coverage": "annual_attendance_coverage",
    "daily": "daily_attendance",
    "remote": "remote_attendance",
    "remote_hours": "remote_hours",
    "remote_zero_hours": "remote_zero_hours",
    "remote_last": "remote_last_day",
    "remote_history": "remote_history",
    "remote_share": "remote_share",
    "early_departure": "early_departure",
    "present": "present_attendance",
    "present_count": "present_count",
    "late": "late_history",
    "late_count": "late_count",
    "absent": "absence_count",
    "absent_dates": "absence_dates",
    "sick": "sick_attendance",
    "hours": "working_hours",
    "hours_vs_scheduled": "hours_vs_scheduled",
}


def attendance_question_focus(message: str, inherited: dict | None = None) -> str | None:
    """Which attendance rows the question asked for. None means the full report."""
    text = message.lower()
    # Structured metric hints from the query interpreter.
    try:
        from app.ai.query_spec import interpret_query

        spec = interpret_query(message, inherited=inherited or {})
        metric_focus = {
            "remote_weekday_peak": "remote_weekday_peak",
            "longest_hours_dates": "longest_hours",
            "zero_hours": "zero_hours",
            "remote_zero_hours": "remote_zero_hours",
            "remote_vs_present": "remote_vs_present",
            "hours_compare": None,  # comparison path
        }.get(spec.metric or "")
        if metric_focus == "zero_hours" and re.search(r"\b(remote|remotely)\b", text):
            return "remote_zero_hours"
        if metric_focus:
            return metric_focus
        if spec.metric == "hours_compare":
            return None
    except Exception:
        pass
    if re.search(r"\b(compare|comparison|versus|vs\.?)\b", text):
        return None
    if re.search(r"\bweekday\b", text) and re.search(r"\bremote\b", text) and re.search(
        r"\b(most|largest|peak|highest)\b", text
    ):
        return "remote_weekday_peak"
    if re.search(r"\b(longest|most hours|highest .{0,12}hours)\b", text) and re.search(
        r"\b(hours|working)\b", text
    ):
        return "longest_hours"
    if re.search(r"\b(remote|remotely)\b", text) and re.search(r"\bzero\b", text) and re.search(
        r"\b(working )?hours?\b", text
    ):
        return "remote_zero_hours"
    if re.search(r"\bzero\b", text) and re.search(r"\b(working )?hours?\b", text):
        return "zero_hours"
    if re.search(r"\b(remote).{0,40}(present|office)|difference between .{0,30}remote\b", text):
        return "remote_vs_present"
    if re.search(r"\b(discrepanc\w*|reconcil\w*|inconsistenc\w*)\b", text) and re.search(
        r"\b(attendance|leave)\b", text
    ):
        return "reconcile"
    if re.search(r"\b(overlap|overlapping)\b", text) or (
        re.search(r"\b(present|remote)\b", text)
        and re.search(r"\bapproved leave\b", text)
        and re.search(r"\b(while|also|and)\b", text)
    ):
        return "leave_overlap"
    if re.search(r"\b(entire year|whole year|full year|exact attendance)\b", text) or (
        re.search(r"\bwhy\b", text)
        and re.search(r"\b(calculate|exact|attendance)\b", text)
        and re.search(r"\byear\b", text)
    ):
        return "year_coverage"
    if _lifetime_cue(text) and re.search(r"\b(attendance|percentage|percent|rate|coverage)\b", text):
        return "year_coverage"
    if re.search(r"\bmiss(ed|ing)?\b", text) and re.search(r"\b(working days?|work days?|days?)\b", text):
        return "missed"
    if re.search(r"\bmissing\b", text) and re.search(r"\b(attendance|records?)\b", text):
        return "missing"
    if re.search(r"\b(was i|were i)\b", text) and re.search(r"\b(late|present|remote|absent|sick)\b", text):
        try:
            window = parse_asked_window(message)
        except WindowError:
            window = None
        if window is not None and window[0] == window[1]:
            return "daily"
    if re.search(
        r"\b(early departures?|clock(?:ed|ing)?[- ]?out early|early clock-?outs?|leave early|leaving early|"
        r"left early|early[- ](?:exits?|finish(?:es)?|sign-?offs?|departures?|leavers?|outs?))\b",
        text,
    ):
        return "early_departure"
    status_words = re.findall(r"\b(present|remote|late|absent|sick)\b", text)
    if len(set(status_words)) >= 2 and re.search(
        r"\b(how many|counts?|together|breakdown|summary|summar(?:ise|ize)|totals?|report|mix|side by side|each|and)\b",
        text,
    ):
        return "status_summary"
    if re.search(r"\b(remote|remotely)\b", text):
        if re.search(r"\b(share|percent|percentage|proportion|what share)\b", text):
            return "remote_share"
        if re.search(r"\b(last|latest|most recent)\b", text) and re.search(r"\b(day|when)\b", text):
            return "remote_last"
        if re.search(r"\bhours?\b", text) and re.search(r"\b(zero|0)\b", text):
            return "remote_zero_hours"
        if re.search(r"\bhours?\b", text):
            return "remote_hours"
        if re.search(r"\bhistory\b", text):
            return "remote_history"
        return "remote"
    if re.search(r"\bpresent\b", text) and not re.search(r"\b(remote|absent|late|sick)\b", text):
        if re.search(r"\b(how many|count|rows?|records?|times)\b", text):
            return "present_count"
        return "present"
    if re.search(r"\blate\b", text):
        if re.search(r"\b(how many|times)\b", text):
            return "late_count"
        return "late"
    if re.search(r"\b(absent|absences?)\b", text):
        if re.search(r"\b(dates|when)\b", text) and not re.search(r"\bhow many\b", text):
            return "absent_dates"
        return "absent"
    if re.search(r"\bsick\b", text) and re.search(r"\b(attendance|taken)\b", text):
        return "sick"
    # Leave-status attendance days, optionally combined with total hours.
    if re.search(r"\b(leave[-\s]?status|status\s+leave|leave\s+days?|on leave)\b", text) and re.search(
        r"\b(attendance|days?|hours?)\b", text
    ):
        if re.search(r"\bhours?\b", text):
            return "leave_and_hours"
        return "leave_status"
    if re.search(r"\bhours\b", text) and re.search(r"\bscheduled\b", text):
        return "hours_vs_scheduled"
    # Daily calendar-day listing (status + hours) must not collapse to a hours total.
    if re.search(r"\b(?:last|past|previous)\s+\d+\s+calendar\s+days?\b", text) and re.search(
        r"\b(dates?|records?|status|attendance)\b", text
    ):
        return "calendar_day_series"
    if re.search(r"\bhours\b", text) and re.search(r"\b(how many|total|worked|recorded|completed)\b", text):
        return "hours"
    previous = (inherited or {}).get("focus") or ""
    if (inherited or {}).get("subject") == "attendance" and previous:
        if re.search(r"\b(which dates|what dates|those dates|which days|the dates)\b", text):
            if str(previous).startswith("remote"):
                return "remote"
            if str(previous).startswith("late"):
                return "late"
            if str(previous).startswith("absent") or previous == "missed":
                return "absent_dates"
        if re.search(r"\bhours\b", text) and str(previous).startswith("remote"):
            return "remote_hours"
    return None


def _whole_hours(value: float | int) -> float | int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _apply_attendance_focus(period: dict, focus: str | None) -> None:
    """Keep period counts for every status. The returned rows are only the asked status."""
    period["focus"] = focus
    period["attendance_intent"] = _ATTENDANCE_INTENT.get(focus or "", "attendance_report")
    status = _STATUS_OF_FOCUS.get(focus or "")
    records = period.get("records") or []
    if not status:
        period["matched_status"] = None
        period["matched_count"] = len(records)
        period["matched_hours"] = period.get("recorded_hours", 0)
        period["matched_dates"] = [row["date"] for row in records]
        return
    matched = [row for row in records if row.get("status") == status]
    total = len(matched)
    if focus == "remote_last" and matched:
        matched = [matched[-1]]
    hours = _whole_hours(sum((row.get("hours") or 0) for row in matched))
    period["records"] = matched
    period["matched_status"] = status
    period["matched_count"] = len(matched)
    period["status_count"] = total
    period["matched_hours"] = hours
    period["matched_dates"] = [row["date"] for row in matched]


def _stored_span(db: Session, employee_id: int) -> tuple[date, date]:
    today = org_today()
    first = db.query(func.min(Attendance.work_date)).filter(Attendance.employee_id == employee_id).scalar()
    if first is None:
        return today, today
    return first, today


def attendance_report(db: Session, employee_id: int, message: str, inherited: dict | None = None) -> dict:
    employee = db.get(Employee, employee_id)
    # Unsupported metrics must disclose the gap — never substitute another attendance fact.
    from app.ai.query_spec import detect_unsupported_metric

    unsupported = detect_unsupported_metric(message or "")
    focus = attendance_question_focus(message, inherited)
    # Pure unsupported metric: disclose only. Mixed supported+unsupported continues below.
    supported_cue = bool(
        re.search(
            r"\b(present|remote|absent|late|sick|leave|hours|attendance|days?)\b",
            (message or "").lower(),
        )
        and focus not in {None, "unsupported_metric"}
    )
    if unsupported and not supported_cue:
        code, label = unsupported
        return {
            "kind": "attendance",
            "focus": "unsupported_metric",
            "unsupported": code,
            "unsupported_label": label,
            "employee_id": employee_id,
            "timezone": org_timezone_name(),
            "as_of": org_today().isoformat(),
            "note": (
                f"{label} is not stored in Helios attendance or HR records. "
                "Helios cannot invent, estimate, or substitute another metric for that request."
            ),
            "records": [],
            "counts": {},
        }
    if focus == "reconcile":
        return reconciliation_report(db, employee_id, message, inherited)
    if focus == "leave_overlap":
        return leave_overlap_report(db, employee_id, message, inherited)
    if focus == "year_coverage":
        return year_coverage_report(db, employee_id, message, inherited)
    compared = parse_comparison(message, inherited=inherited)
    if compared:
        periods = [
            _attendance_period(db, employee_id, start, end, label, employee) for start, end, label in compared
        ]
        return {
            "kind": "attendance_comparison",
            "employee_id": employee_id,
            "timezone": org_timezone_name(),
            "as_of": org_today().isoformat(),
            "start": periods[0]["start"] if periods else None,
            "end": periods[-1]["end"] if periods else None,
            "range_label": " and ".join(period["range_label"] for period in periods),
            "periods": periods,
            "focus": None,
            "matched_status": None,
            "includes_remote": bool(re.search(r"\b(remote|remotely)\b", message.lower())),
            "integrity": integrity_findings(db, employee_id, periods[0]["start"], periods[-1]["end"]) if periods else [],
        }
    text = message.lower()
    explicit = parse_asked_window(message)
    inherited_window = _inherited_window(inherited, "attendance")
    if focus == "missed" and explicit is None and inherited_window is None:
        start, end, label = attendance_window(message, inherited=inherited)
    elif explicit is None and inherited_window is None and (
        focus in _STORED_SPAN_FOCUSES or re.search(r"\battendance history\b", text)
    ):
        start, end = _stored_span(db, employee_id)
        label = "stored attendance"
    else:
        start, end, label = attendance_window(message, inherited=inherited)
    period = _attendance_period(db, employee_id, start, end, label, employee)
    if focus == "calendar_day_series":
        by_date = {row["date"]: row for row in (period.get("records") or [])}
        series = []
        cursor = start
        while cursor <= end:
            key = cursor.isoformat()
            row = by_date.get(key)
            if row:
                series.append(
                    {
                        "date": key,
                        "status": row.get("status"),
                        "hours": row.get("hours"),
                        "recorded": True,
                    }
                )
            else:
                series.append({"date": key, "status": None, "hours": None, "recorded": False})
            cursor += timedelta(days=1)
        period["focus"] = "calendar_day_series"
        period["attendance_intent"] = "calendar_day_series"
        period["calendar_days"] = series
        period["records"] = [item for item in series if item.get("recorded")]
        period["matched_count"] = len(period["records"])
        period["matched_dates"] = [item["date"] for item in period["records"]]
        return {
            "kind": "attendance",
            "employee_id": employee_id,
            "timezone": org_timezone_name(),
            "as_of": org_today().isoformat(),
            **period,
        }
    if focus == "missing":
        period["missing_weekdays"] = [
            day for day in period["missing_dates"] if date.fromisoformat(day).weekday() < 5
        ]
    if focus == "missed":
        _attach_missed_breakdown(period)
    _apply_attendance_focus(period, focus)
    if focus == "remote_weekday_peak":
        from collections import Counter

        remote_rows = [row for row in (period.get("records") or []) if row.get("status") == "remote"]
        counts = Counter()
        for row in remote_rows:
            try:
                counts[date.fromisoformat(row["date"]).strftime("%A")] += 1
            except Exception:
                continue
        top = counts.most_common(1)[0] if counts else ("", 0)
        period["focus"] = "remote_weekday_peak"
        period["attendance_intent"] = "remote_weekday_peak"
        period["weekday_peak"] = top[0]
        period["weekday_peak_count"] = int(top[1])
        period["weekday_counts"] = dict(counts)
        period["matched_count"] = len(remote_rows)
        period["matched_dates"] = [row["date"] for row in remote_rows]
    if focus == "longest_hours":
        rows = list(period.get("records") or [])
        max_hours = max((float(row.get("hours") or 0) for row in rows), default=0)
        longest = [row for row in rows if float(row.get("hours") or 0) == max_hours and max_hours > 0]
        period["focus"] = "longest_hours"
        period["attendance_intent"] = "longest_hours"
        period["longest_hours"] = _whole_hours(max_hours)
        period["longest_dates"] = [row["date"] for row in longest]
        period["records"] = longest
        period["matched_count"] = len(longest)
        period["matched_hours"] = _whole_hours(max_hours)
        period["matched_dates"] = period["longest_dates"]
    if focus == "zero_hours":
        matched = [row for row in (period.get("records") or []) if float(row.get("hours") or 0) == 0]
        period["focus"] = "zero_hours"
        period["attendance_intent"] = "zero_hours"
        period["records"] = matched
        period["matched_count"] = len(matched)
        period["matched_hours"] = 0
        period["matched_dates"] = [row["date"] for row in matched]
        period["zero_hours_count"] = len(matched)
    if focus == "remote_vs_present":
        records = period.get("records") or []
        remote_n = sum(1 for row in records if row.get("status") == "remote")
        present_n = sum(1 for row in records if row.get("status") == "present")
        period["focus"] = "remote_vs_present"
        period["attendance_intent"] = "remote_vs_present"
        period["remote_days"] = remote_n
        period["present_days"] = present_n
        period["remote_present_diff"] = abs(remote_n - present_n)
    if focus == "remote_zero_hours":
        matched = [
            row
            for row in (period.get("records") or [])
            if row.get("status") == "remote" and float(row.get("hours") or 0) == 0
        ]
        period["focus"] = "remote_zero_hours"
        period["matched_status"] = "remote"
        period["records"] = matched
        period["matched_count"] = len(matched)
        period["matched_hours"] = 0
        period["matched_dates"] = [row["date"] for row in matched]
        period["attendance_intent"] = "remote_zero_hours"
    if focus == "hours_vs_scheduled":
        period["focus"] = "hours_vs_scheduled"
        period["attendance_intent"] = "hours_vs_scheduled"
        period["scheduled_hours_available"] = False
        period["scheduled_hours_note"] = (
            "Helios stores recorded attendance hours per day. It does not store a separate "
            "scheduled-hours field for each employee, so worked-versus-scheduled hours cannot "
            "be calculated from attendance rows alone."
        )
    if focus == "leave_and_hours":
        leave_rows = [row for row in (period.get("records") or []) if row.get("status") == "leave"]
        period["focus"] = "leave_and_hours"
        period["attendance_intent"] = "leave_and_hours"
        period["leave_days"] = len(leave_rows)
        period["leave_dates"] = [row["date"] for row in leave_rows]
        period["matched_status"] = "leave"
        period["matched_count"] = len(leave_rows)
        period["matched_dates"] = period["leave_dates"]
        period["recorded_hours"] = period.get("recorded_hours", 0)
    if focus == "absent":
        absent_rows = [row for row in (period.get("records") or []) if row.get("status") == "absent"]
        period["absent_day_count"] = len(absent_rows)
        period["absent_hours"] = _whole_hours(sum((row.get("hours") or 0) for row in absent_rows))
        period["absent_hours_note"] = (
            "Absent day count is independent of recorded hours. "
            "An absent row may store 0 hours; that does not mean the absence did not occur."
        )
    period["integrity"] = integrity_findings(db, employee_id, period["start"], period["end"])
    if unsupported:
        code, label = unsupported
        period["unsupported"] = code
        period["unsupported_label"] = label
        period["unsupported_note"] = (
            f"{label} is not stored in Helios attendance or HR records and was not substituted."
        )
    return {
        "kind": "attendance",
        "employee_id": employee_id,
        "timezone": org_timezone_name(),
        "as_of": org_today().isoformat(),
        **period,
    }


def _attach_missed_breakdown(period: dict) -> None:
    """Confirmed non-attendance rows versus unverified scheduled gaps."""
    records = period.get("records") or []
    by_status = {
        "absent": [row for row in records if row.get("status") == "absent"],
        "sick": [row for row in records if row.get("status") == "sick"],
        "leave": [row for row in records if row.get("status") == "leave"],
        "other": [row for row in records if row.get("status") not in {"present", "remote", "late", "absent", "sick", "leave"}],
    }
    missing = period.get("missing_scheduled_dates") or []
    confirmed = len(by_status["absent"]) + len(by_status["sick"])
    period["missed"] = {
        "confirmed_missed_days": confirmed,
        "absent_days": len(by_status["absent"]),
        "sick_days": len(by_status["sick"]),
        "leave_days": len(by_status["leave"]),
        "other_absence_days": len(by_status["other"]),
        "absent_dates": [row["date"] for row in by_status["absent"]],
        "sick_dates": [row["date"] for row in by_status["sick"]],
        "leave_dates": [row["date"] for row in by_status["leave"]],
        "missing_scheduled_days": len(missing),
        "missing_scheduled_dates": missing,
        "scheduled_working_days": period.get("scheduled_working_days"),
        "calendar_status": period.get("calendar_status"),
        "include_leave_note": (
            "Approved leave is recorded as status leave. It is planned absence, not an unverified gap. "
            "Whether leave counts in a missed-working-days metric depends on the report definition; "
            "this answer lists leave separately and does not add it into confirmed missed days."
        ),
        "reliable_total": False,
        "reliable_total_reason": (
            "No single reliable missed-day total is available while scheduled days are missing attendance rows. "
            "Confirmed missed days are only rows with status absent or sick. Missing scheduled days are not absences."
        ),
    }


def _lifetime_cue(text: str) -> bool:
    """Whole-employment framing: lifetime, all-time, since/from hire date."""
    return bool(
        re.search(
            r"\b(lifetime|all[- ]time|since (?:my )?(?:hire|start) date|since (?:i was )?hired|since i joined|"
            r"from (?:my )?(?:hire|start) date|from (?:the )?day i (?:joined|started)|"
            r"(?:entire|whole|full) (?:employment|tenure|time (?:here|with the company)))\b",
            text.lower(),
        )
    )


def _year_to_date_window(message: str, inherited: dict | None = None) -> tuple[date, date, str]:
    """Default reconciliation and coverage windows to the current year, not last seven days."""
    explicit = parse_asked_window(message)
    if explicit is not None:
        return explicit
    inherited_window = _inherited_window(inherited, "attendance") or _inherited_window(inherited, "reconciliation")
    if inherited_window is not None and not re.search(
        r"\b(entire year|whole year|full year|this year|discrepanc|reconcil)\b", message.lower()
    ):
        return inherited_window
    today = org_today()
    return date(today.year, 1, 1), today, str(today.year)


def reconciliation_report(
    db: Session, employee_id: int, message: str, inherited: dict | None = None
) -> dict:
    """Cross-record attendance and leave discrepancies for the signed-in employee."""
    start, end, label = _year_to_date_window(message, inherited)
    findings = integrity_findings(db, employee_id, start, end)
    confirmed = [item for item in findings if item.get("finding_class") != "unverified"]
    unverified = [item for item in findings if item.get("finding_class") == "unverified"]
    return {
        "kind": "reconciliation",
        "employee_id": employee_id,
        "timezone": org_timezone_name(),
        "as_of": org_today().isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "range_label": label,
        "focus": "reconcile",
        "attendance_intent": "attendance_leave_reconciliation",
        "findings": findings,
        "confirmed_findings": confirmed,
        "unverified_findings": unverified,
        "integrity": findings,
        "changed": False,
    }


def leave_overlap_report(
    db: Session, employee_id: int, message: str, inherited: dict | None = None
) -> dict:
    """Present/remote/late attendance that overlaps approved leave."""
    explicit = parse_asked_window(message)
    if explicit is not None:
        start, end, label = explicit
    else:
        start, end = _stored_span(db, employee_id)
        first_leave = (
            db.query(func.min(LeaveRequest.start_date))
            .filter(LeaveRequest.employee_id == employee_id, LeaveRequest.status == "approved")
            .scalar()
        )
        if first_leave is not None and first_leave < start:
            start = first_leave
        label = "available leave and attendance span"
    attendance = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee_id,
            Attendance.work_date >= start,
            Attendance.work_date <= end,
            Attendance.status.in_(("present", "remote", "late")),
        )
        .order_by(Attendance.work_date, Attendance.id)
        .all()
    )
    by_date = {row.work_date: row for row in attendance}
    requests = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.status == "approved",
            LeaveRequest.start_date <= end,
            LeaveRequest.end_date >= start,
        )
        .order_by(LeaveRequest.id)
        .all()
    )
    overlaps = []
    for request in requests:
        cursor = max(request.start_date, start)
        last = min(request.end_date, end)
        while cursor <= last:
            row = by_date.get(cursor)
            if cursor.weekday() < 5 and row is not None:
                overlaps.append(
                    {
                        "date": cursor.isoformat(),
                        "attendance_status": row.status,
                        "hours": row.hours,
                        "leave_request_id": request.id,
                        "leave_type": request.leave_type,
                        "leave_status": request.status,
                        "leave_start": request.start_date.isoformat(),
                        "leave_end": request.end_date.isoformat(),
                    }
                )
            cursor += timedelta(days=1)
    return {
        "kind": "leave_overlap",
        "employee_id": employee_id,
        "timezone": org_timezone_name(),
        "as_of": org_today().isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "range_label": label,
        "focus": "leave_overlap",
        "attendance_intent": "leave_attendance_overlap",
        "overlaps": overlaps,
        "changed": False,
    }


def year_coverage_report(
    db: Session, employee_id: int, message: str, inherited: dict | None = None
) -> dict:
    """Why a full-year attendance percentage is not authoritative."""
    employee = db.get(Employee, employee_id)
    lifetime = _lifetime_cue(message) and parse_asked_window(message) is None
    if lifetime and employee is not None and employee.hire_date:
        start, end = employee.hire_date, org_today()
        label = f"since hire date {employee.hire_date.isoformat()}"
    else:
        start, end, label = _year_to_date_window(message, inherited)
    period = _attendance_period(db, employee_id, start, end, label, employee)
    stored_start, stored_end = _stored_span(db, employee_id)
    rows = period.get("records") or []
    earliest = rows[0]["date"] if rows else None
    latest = rows[-1]["date"] if rows else None
    missing = period.get("missing_scheduled_dates") or []
    scheduled = period.get("scheduled_working_days")
    recorded = period.get("recorded_days") or 0
    return {
        "kind": "year_coverage",
        "employee_id": employee_id,
        "timezone": org_timezone_name(),
        "as_of": org_today().isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "range_label": label,
        "focus": "year_coverage",
        "attendance_intent": "annual_attendance_coverage",
        "scheduled_working_days": scheduled,
        "recorded_days": recorded,
        "missing_scheduled_days": len(missing),
        "missing_scheduled_dates": missing,
        "earliest_attendance": earliest,
        "latest_attendance": latest,
        "stored_attendance_start": stored_start.isoformat(),
        "stored_attendance_end": stored_end.isoformat(),
        "calendar_status": period.get("calendar_status"),
        "counts": period.get("counts") or {},
        "authoritative": False,
        "lifetime": lifetime,
        "hire_date": employee.hire_date.isoformat() if employee is not None and employee.hire_date else None,
        "reason": (
            (
                "A lifetime attendance percentage cannot be computed from the hire date because stored attendance "
                f"rows only begin on {stored_start.isoformat()}; the period between hire and the first stored row "
                "has no attendance rows at all, and missing scheduled days are not classified as absences. "
                "Any percentage would have an unknown denominator."
            )
            if lifetime
            else (
                "A full-year attendance percentage is not authoritative while scheduled working days "
                "are missing attendance rows. Missing scheduled days are not classified as absences."
            )
        ),
        "changed": False,
    }


def integrity_findings(db: Session, employee_id: int, start: str | date, end: str | date) -> list[dict]:
    """Read-only inconsistencies for one employee. Does not change any HRIS row."""
    start_day = date.fromisoformat(start) if isinstance(start, str) else start
    end_day = date.fromisoformat(end) if isinstance(end, str) else end
    findings: list[dict] = []
    attendance = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee_id,
            Attendance.work_date >= start_day,
            Attendance.work_date <= end_day,
        )
        .order_by(Attendance.work_date, Attendance.id)
        .all()
    )
    requests = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.start_date <= end_day,
            LeaveRequest.end_date >= start_day,
        )
        .order_by(LeaveRequest.id)
        .all()
    )
    approvals = _approval_details(db, [row.id for row in requests])
    for request in requests:
        if request.status == "approved" and not approvals.get(request.id):
            findings.append(
                {
                    "code": "approved_leave_without_approval_row",
                    "summary": (
                        f"Leave request {request.id} ({request.leave_type}, "
                        f"{request.start_date.isoformat()} to {request.end_date.isoformat()}, "
                        f"status {request.status}) has no approval decision row."
                    ),
                    "leave_request_id": request.id,
                    "leave_type": request.leave_type,
                    "start": request.start_date.isoformat(),
                    "end": request.end_date.isoformat(),
                    "status": request.status,
                    "finding_class": "confirmed",
                }
            )
    sick_attendance = [row for row in attendance if row.status == "sick"]
    sick_requests = [row for row in requests if row.leave_type == "sick"]
    if sick_attendance and not sick_requests:
        findings.append(
            {
                "code": "sick_attendance_without_sick_request",
                "summary": (
                    f"{len(sick_attendance)} sick attendance row(s) exist in this window "
                    "and no overlapping sick leave request is stored."
                ),
                "dates": [row.work_date.isoformat() for row in sick_attendance],
                "finding_class": "confirmed",
            }
        )
    attended = {
        row.work_date: row.status for row in attendance if row.status in {"present", "remote", "late"}
    }
    stored = {row.work_date for row in attendance}
    stored_start = min(stored) if stored else None
    stored_end = max(stored) if stored else None
    for request in requests:
        if request.status != "approved":
            continue
        cursor = max(request.start_date, start_day)
        last = min(request.end_date, end_day)
        overlaps = []
        missing_leave_attendance = []
        while cursor <= last:
            if cursor.weekday() < 5:
                if cursor in attended:
                    overlaps.append(
                        {
                            "date": cursor.isoformat(),
                            "attendance_status": attended[cursor],
                        }
                    )
                elif (
                    stored_start is not None
                    and stored_end is not None
                    and stored_start <= cursor <= stored_end
                    and cursor not in stored
                ):
                    missing_leave_attendance.append(cursor.isoformat())
            cursor += timedelta(days=1)
        if overlaps:
            findings.append(
                {
                    "code": "approved_leave_overlaps_attendance",
                    "summary": (
                        f"Approved leave request {request.id} ({request.leave_type}, "
                        f"{request.start_date.isoformat()} to {request.end_date.isoformat()}) "
                        f"overlaps attendance marked present, remote, or late on {len(overlaps)} weekday(s)."
                    ),
                    "leave_request_id": request.id,
                    "leave_type": request.leave_type,
                    "start": request.start_date.isoformat(),
                    "end": request.end_date.isoformat(),
                    "status": request.status,
                    "dates": [item["date"] for item in overlaps],
                    "overlap_details": overlaps,
                    "finding_class": "confirmed",
                }
            )
        if missing_leave_attendance:
            findings.append(
                {
                    "code": "approved_leave_without_attendance",
                    "summary": (
                        f"Approved leave request {request.id} has {len(missing_leave_attendance)} weekday(s) "
                        "inside the stored attendance span with no attendance row."
                    ),
                    "leave_request_id": request.id,
                    "leave_type": request.leave_type,
                    "start": request.start_date.isoformat(),
                    "end": request.end_date.isoformat(),
                    "status": request.status,
                    "dates": missing_leave_attendance,
                    "finding_class": "confirmed",
                }
            )
    employee = db.get(Employee, employee_id)
    schedule = _scheduled_working_days(db, employee, start_day, end_day)
    missing = [day.isoformat() for day in schedule.get("scheduled_dates") or [] if day not in stored]
    if missing:
        findings.append(
            {
                "code": "missing_scheduled_attendance",
                "summary": (
                    f"{len(missing)} scheduled working day(s) have no attendance row. "
                    "These are unverified gaps, not automatic absences."
                ),
                "count": len(missing),
                "dates": missing[:20],
                "date_count": len(missing),
                "finding_class": "unverified",
            }
        )
    seen: dict[date, int] = {}
    for row in attendance:
        seen[row.work_date] = seen.get(row.work_date, 0) + 1
    duplicates = [day.isoformat() for day, count in seen.items() if count > 1]
    if duplicates:
        findings.append(
            {
                "code": "duplicate_attendance_rows",
                "summary": f"{len(duplicates)} date(s) have more than one attendance row.",
                "dates": duplicates,
                "finding_class": "confirmed",
            }
        )
    year = end_day.year
    year_start = date(year, 1, 1)
    year_end = min(date(year, 12, 31), org_today())
    if start_day <= year_start and end_day >= year_end:
        for leave_type in ("annual", "sick"):
            balance = get_balance(db, employee_id, leave_type, year)
            if balance is None:
                continue
            year_requests = (
                db.query(LeaveRequest)
                .filter(
                    LeaveRequest.employee_id == employee_id,
                    LeaveRequest.leave_type == leave_type,
                    LeaveRequest.status == "approved",
                    LeaveRequest.start_date >= year_start,
                    LeaveRequest.start_date <= date(year, 12, 31),
                )
                .all()
            )
            approved_days = sum(float(row.business_days) for row in year_requests)
            if float(balance.used_days) != approved_days:
                findings.append(
                    {
                        "code": f"{leave_type}_balance_does_not_match_approved_requests",
                        "summary": (
                            f"{leave_type.capitalize()} leave balance used is {balance.used_days} for {year}, "
                            f"while approved {leave_type} requests starting in {year} sum to {approved_days}."
                        ),
                        "balance_used": balance.used_days,
                        "approved_request_days": approved_days,
                        "leave_type": leave_type,
                        "finding_class": "confirmed",
                    }
                )
            if leave_type == "sick":
                sick_days = len([row for row in attendance if row.status == "sick"])
                if float(balance.used_days) != float(sick_days):
                    findings.append(
                        {
                            "code": "sick_balance_does_not_match_attendance",
                            "summary": (
                                f"Sick leave balance used is {balance.used_days} for {year}, "
                                f"while sick attendance rows in this window total {sick_days}. "
                                "Attendance and leave balance are separate sources."
                            ),
                            "balance_used": balance.used_days,
                            "attendance_sick_days": sick_days,
                            "finding_class": "confirmed",
                        }
                    )
    return findings


def holiday_report(db: Session, employee: Employee, message: str, inherited: dict | None = None) -> dict:
    start, end, label = holiday_window(message, inherited=inherited)
    entries, status = list_holidays_for_employee(db, employee, start, end)
    countries = [
        row[0]
        for row in db.query(Holiday.country).filter(Holiday.is_active.is_(True)).distinct().order_by(Holiday.country).all()
    ]
    return {
        "kind": "holiday",
        "employee_id": employee.id,
        "location": employee.location,
        "office_country": OFFICE_COUNTRY.get(employee.location),
        "configured_countries": countries,
        "calendar_status": status,
        "timezone": org_timezone_name(),
        "as_of": org_today().isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "range_label": label,
        "holidays": entries,
    }


def sick_usage_report(db: Session, employee_id: int, message: str) -> dict:
    """Sick days taken from attendance, plus the separate balance and request records.

    The employee id is the authenticated id passed by the caller.
    """
    today = org_today()
    year = today.year - 1 if re.search(r"\blast year\b", message.lower()) else today.year
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), today)
    attendance_rows = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee_id,
            Attendance.work_date >= start,
            Attendance.work_date <= end,
        )
        .order_by(Attendance.work_date, Attendance.id)
        .all()
    )
    sick_rows = [row for row in attendance_rows if row.status == "sick"]
    balance = get_balance(db, employee_id, "sick", year)
    requests = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.leave_type == "sick",
            LeaveRequest.start_date <= date(year, 12, 31),
            LeaveRequest.end_date >= start,
        )
        .order_by(LeaveRequest.start_date, LeaveRequest.id)
        .all()
    )
    used = None if balance is None else balance.used_days
    attendance_count = len(sick_rows)
    approved_sick_days = sum(
        float(row.business_days) for row in requests if row.status == "approved"
    )
    difference = None if used is None else float(used) - float(attendance_count)
    text = message.lower()
    reconcile = bool(re.search(r"\b(match|matches|reconcil|discrepanc|compare)\b", text))
    deduction_audit = bool(
        re.search(r"\b(deduct|responsible|account for|says?\b.*\bused|used.*records)\b", text)
    )
    stated_used = None
    stated = re.search(r"\b(?:says?|states?|shows?)\s+(\d+(?:\.\d+)?)\s+days?\s+used\b", text)
    if stated:
        stated_used = float(stated.group(1))
    return {
        "kind": "sick_taken",
        "employee_id": employee_id,
        "year": year,
        "as_of": today.isoformat(),
        "timezone": org_timezone_name(),
        "attendance_sick_days": attendance_count,
        "attendance_dates": [row.work_date.isoformat() for row in sick_rows],
        "stored_attendance_start": attendance_rows[0].work_date.isoformat() if attendance_rows else "",
        "stored_attendance_end": attendance_rows[-1].work_date.isoformat() if attendance_rows else "",
        "stored_attendance_rows": len(attendance_rows),
        "balance": None if balance is None else balance_payload(balance, "sick"),
        "balance_matches_attendance": None if used is None else float(used) == float(attendance_count),
        "approved_sick_request_days": approved_sick_days,
        "difference": difference,
        "reconcile": reconcile or deduction_audit,
        "deduction_audit": deduction_audit,
        "stated_used": stated_used,
        "has_deduction_ledger": False,
        "integrity": integrity_findings(db, employee_id, start, end),
        "requests": [
            {
                "id": row.id,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
                "status": row.status,
                "business_days": row.business_days,
            }
            for row in requests
        ],
    }


def leave_balance_report(db: Session, employee_id: int, message: str, inherited: dict | None = None) -> dict:
    text = message.lower()
    if "sick" in text and re.search(
        r"\b(taken|match|matches|reconcil|discrepanc|deduct|responsible|account for)\b", text
    ):
        return sick_usage_report(db, employee_id, message)
    types: list[str] = []
    if "sick" in text:
        types.append("sick")
    if "annual" in text or "vacation" in text:
        types.append("annual")
    compare_ask = bool(
        re.search(r"\b(which balance|larger|bigger|difference|compare|versus|vs\.?)\b", text)
        and (inherited or {}).get("subject") == "leave_balance"
    )
    if compare_ask or re.search(r"\b(both|annual and sick|sick and annual)\b", text):
        types = ["annual", "sick"]
    if not types and inherited and inherited.get("subject") == "leave_balance":
        types.extend(inherited.get("leave_types") or ["annual"])
    if not types:
        types.append("annual")
    # Preserve prior leave types for compare follow-ups.
    if compare_ask and inherited and inherited.get("leave_types"):
        for leave_type in inherited.get("leave_types") or []:
            if leave_type not in types:
                types.append(leave_type)
    year = org_today().year
    balances = []
    for leave_type in types:
        balances.append(balance_payload(get_balance(db, employee_id, leave_type, year), leave_type))
    focus = "balance"
    if compare_ask or (
        len(balances) >= 2 and re.search(r"\b(larger|bigger|difference|compare|which balance)\b", text)
    ):
        focus = "compare_balances"
    elif re.search(r"\b(committed to pending|pending requests?|days .{0,24}pending|pending leave days)\b", text):
        focus = "pending_days"
    elif re.search(r"\b(difference between)\b", text) and re.search(
        r"\b(entitled|used|pending)\b", text
    ):
        focus = "breakdown"
    elif (
        re.search(r"\b(entitled|used leave|pending leave)\b", text)
        and re.search(r"\b(entitled|used|pending)\b", text)
        and re.search(r"\b(difference|breakdown|composed|made up)\b", text)
    ):
        focus = "breakdown"
    return {
        "kind": "leave_balance",
        "focus": focus,
        "employee_id": employee_id,
        "timezone": org_timezone_name(),
        "as_of": org_today().isoformat(),
        "balances": balances,
    }


def cancellation_report(db: Session, employee_id: int) -> dict:
    today = org_today()
    rows = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.status.in_(OPEN_LEAVE),
            LeaveRequest.end_date >= today,
        )
        .order_by(LeaveRequest.start_date, LeaveRequest.id)
        .all()
    )
    return {
        "kind": "leave_cancellation",
        "employee_id": employee_id,
        "timezone": org_timezone_name(),
        "as_of": today.isoformat(),
        "changed": False,
        "requests": [
            {
                "id": row.id,
                "leave_type": row.leave_type,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
                "status": row.status,
                "business_days": row.business_days,
            }
            for row in rows
        ],
    }


def _approval_details(db: Session, request_ids: list[int]) -> dict[int, list[dict]]:
    if not request_ids:
        return {}
    rows = (
        db.query(Approval)
        .filter(Approval.leave_request_id.in_(request_ids), Approval.action_type == "leave")
        .order_by(Approval.id.asc())
        .all()
    )
    names: dict[int, str] = {}
    grouped: dict[int, list[dict]] = {request_id: [] for request_id in request_ids}
    for row in rows:
        approver = ""
        if row.approver_employee_id:
            approver = names.get(row.approver_employee_id, "")
            if not approver:
                person = db.get(Employee, row.approver_employee_id)
                approver = person.full_name if person is not None else ""
                names[row.approver_employee_id] = approver
        elif row.approver_role == "hr_admin":
            approver = "People Operations"
        grouped.setdefault(row.leave_request_id, []).append(
            {
                "id": row.id,
                "role": row.approver_role,
                "status": row.status,
                "approver": approver,
                "decided_at": row.decided_at.isoformat() if row.decided_at else "",
            }
        )
    return grouped


def _leave_question(text: str) -> str:
    if re.search(r"\bwho approved\b", text):
        return "who"
    if re.search(r"\b(has|have)\b", text) and re.search(r"\breject", text):
        return "rejected"
    if re.search(r"\b(has|have|been)\b", text) and re.search(r"\bapprov", text):
        return "approved"
    return ""


def leave_status_report(
    db: Session,
    employee_id: int,
    message: str = "",
    inherited: dict | None = None,
) -> dict:
    """Leave requests for one employee. The id comes from the authenticated session.

    The latest request is the one with the latest submission timestamp. Request id
    is only a tie-breaker. Asking whether that request was approved does not switch
    the query to approved-only rows.
    """
    today = org_today()
    text = message.lower()
    base = db.query(LeaveRequest).filter(LeaveRequest.employee_id == employee_id)
    request_count = base.count()
    question = _leave_question(text)
    inherited_id = None
    if inherited and inherited.get("subject") == "leave_history":
        inherited_id = inherited.get("request_id")
    asks_latest_approved = bool(
        (
            re.search(r"\b(latest|most recent|last)\s+approved\b", text)
            or (
                re.search(r"\b(latest|most recent|last)\b", text)
                and re.search(r"\bapproved (leave )?request\b", text)
            )
        )
        and question != "approved"
        and question != "who"
        and not re.search(r"\b(who approved|person who)\b", text)
    )
    follow_same = bool(
        inherited_id
        and re.search(r"\b(it|that|who approved)\b", text)
        and not re.search(r"\b(other|all|pending|history|latest|most recent)\b", text)
    )
    from app.ai.context import delayed_approval_advice_question

    inherited_delayed = bool(
        inherited
        and inherited.get("focus") in {"delayed_approval", "pending", "latest"}
        and re.search(
            r"\b(how long|pending|approve|approver|respond|follow up|next steps?|still|which request|what if)\b",
            text,
        )
    )
    if delayed_approval_advice_question(text) or inherited_delayed or (
        re.search(
            r"\b("
            r"manager has not approved|has(?:n'?t| not) (?:been )?approved|"
            r"still (?:waiting|pending|awaiting)|waiting (?:for|on) (?:my )?(?:manager|approval|decision)|"
            r"still awaiting approval|awaiting approval|"
            r"pending for (?:several |a few |\d+ )?days|"
            r"(?:my )?leave (?:has )?been pending|leave request still|"
            r"approval (?:is|has been) delayed|delayed approval|"
            r"what should i do|can you advise|who needs to approve|"
            r"if (?:my )?manager doesn'?t respond|"
            r"why .{0,40}(?:still )?(?:pending|awaiting)"
            r")\b",
            text,
        )
        and re.search(r"\b(leave|request|manager|approval|pending)\b", text)
        and not re.search(r"\b(submit|book|apply for|prepare|create|draft|file)\b", text)
    ):
        report_filter = "delayed_approval"
        rows = (
            base.filter(LeaveRequest.status.in_(("pending_manager", "pending_hr")))
            .order_by(LeaveRequest.submitted_at.asc(), LeaveRequest.id.asc())
            .limit(100)
            .all()
        )
        approvals = _approval_details(db, [row.id for row in rows])
    elif re.search(r"\bno approval\b", text) or (
        re.search(r"\bapproved\b", text)
        and re.search(r"\b(no|without|missing)\b", text)
        and re.search(r"\b(approval|decision)\b", text)
        and not re.search(r"\bhas not approved\b", text)
    ):
        report_filter = "approved_without_decision"
        approved_rows = (
            base.filter(LeaveRequest.status == "approved")
            .order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc())
            .all()
        )
        approvals = _approval_details(db, [row.id for row in approved_rows])
        rows = [row for row in approved_rows if not approvals.get(row.id)]
    elif (
        re.search(r"\b(approved|rejected)\b", text)
        and re.search(r"\bpending\b", text)
        and re.search(r"\b(summar|separate|history|current year|this year|for the year|break .{0,20}into)\b", text)
    ):
        # Historical year status split — do not collapse to pending-only because "pending" appears.
        # Current calendar year includes future-dated requests still stored for this year.
        report_filter = "year_status_summary"
        year_start = date(today.year, 1, 1)
        year_end = date(today.year, 12, 31)
        rows = (
            base.filter(LeaveRequest.start_date >= year_start, LeaveRequest.start_date <= year_end)
            .order_by(LeaveRequest.start_date.desc(), LeaveRequest.id.desc())
            .limit(200)
            .all()
        )
        approvals = _approval_details(db, [row.id for row in rows])
    elif re.search(r"\bhistory\b", text):
        report_filter = "history"
        rows = base.order_by(LeaveRequest.start_date.desc(), LeaveRequest.id.desc()).limit(100).all()
        approvals = _approval_details(db, [row.id for row in rows])
    elif re.search(r"\bother\b", text) and re.search(r"\b(request|leave)\b", text):
        report_filter = "other"
        query = base
        if inherited_id:
            query = query.filter(LeaveRequest.id != int(inherited_id))
        rows = query.order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc()).limit(100).all()
        approvals = _approval_details(db, [row.id for row in rows])
    elif re.search(
        r"\b("
        r"pending|"
        r"waiting for a decision|"
        r"awaiting (?:a |an )?decision|"
        r"awaiting approval|"
        r"waiting (?:for|on) approval|"
        r"awaiting (?:manager|hr) approval"
        r")\b",
        text,
    ):
        report_filter = "pending"
        rows = (
            base.filter(LeaveRequest.status.in_(("pending_manager", "pending_hr")))
            .order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc())
            .limit(100)
            .all()
        )
        approvals = _approval_details(db, [row.id for row in rows])
    elif re.search(r"\b(ever rejected|any .{0,20}rejected|was .{0,20}rejected)\b", text) or (
        re.search(r"\brejected\b", text) and re.search(r"\b(any|ever|was|were)\b", text)
    ):
        report_filter = "rejected"
        rows = (
            base.filter(LeaveRequest.status == "rejected")
            .order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc())
            .limit(100)
            .all()
        )
        approvals = _approval_details(db, [row.id for row in rows])
    elif asks_latest_approved:
        report_filter = "latest_approved"
        rows = (
            base.filter(LeaveRequest.status == "approved")
            .order_by(LeaveRequest.resolved_at.desc().nullslast(), LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc())
            .limit(1)
            .all()
        )
        approvals = _approval_details(db, [row.id for row in rows])
    elif follow_same or re.search(r"\b(latest|most recent)\b", text):
        report_filter = "latest"
        if follow_same:
            rows = base.filter(LeaveRequest.id == int(inherited_id)).limit(1).all()
        else:
            rows = (
                base.order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc())
                .limit(1)
                .all()
            )
        approvals = _approval_details(db, [row.id for row in rows])
    else:
        report_filter = "all"
        rows = base.order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc()).limit(100).all()
        approvals = _approval_details(db, [row.id for row in rows])
    manager_ids = {row.routed_manager_id for row in rows if row.routed_manager_id}
    managers = {
        row.id: row.full_name
        for row in db.query(Employee).filter(Employee.id.in_(manager_ids)).all()
    } if manager_ids else {}
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
    pending_n = sum(n for status, n in by_status.items() if "pending" in status)
    actor = db.get(Employee, employee_id)
    approval_manager = None
    if actor is not None and actor.approval_manager_id:
        approval_manager = db.get(Employee, actor.approval_manager_id)
    advice = None
    if report_filter == "delayed_approval":
        advice = {
            "process": [
                "Leave requests first wait in pending_manager for the approval manager on your employee record.",
                "If the request is longer than 10 business days or touches a critical project deadline, "
                "People Operations co-approval (pending_hr) is also required after the manager decision.",
                "The assistant cannot approve, reject, or escalate leave on your behalf.",
            ],
            "next_steps": [
                "Follow up politely with your assigned approval manager about the pending request id(s).",
                "Confirm the leave dates and any coverage plan so the manager can decide.",
                "If the request has moved to pending_hr, contact People Operations after your manager has decided.",
                "Do not submit a duplicate leave request unless you intentionally want a new request.",
            ],
            "approval_manager": None if approval_manager is None else approval_manager.full_name,
        }
    return {
        "kind": "leave_request_status",
        "focus": "delayed_approval" if report_filter == "delayed_approval" else report_filter,
        "employee_id": employee_id,
        "timezone": org_timezone_name(),
        "as_of": today.isoformat(),
        "filter": report_filter,
        "question": question,
        "request_count": request_count,
        "changed": False,
        "advice": advice,
        "by_status": {
            "approved": by_status.get("approved", 0),
            "rejected": by_status.get("rejected", 0),
            "pending": pending_n,
            "pending_manager": by_status.get("pending_manager", 0),
            "pending_hr": by_status.get("pending_hr", 0),
            "cancelled": by_status.get("cancelled", 0),
            "raw": by_status,
        },
        "requests": [
            {
                "id": row.id,
                "leave_type": row.leave_type,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
                "status": row.status,
                "business_days": row.business_days,
                "submitted_at": row.submitted_at.isoformat() if row.submitted_at else "",
                "assigned_approver_id": row.routed_manager_id,
                "assigned_approver": managers.get(row.routed_manager_id) if row.routed_manager_id else None,
                "approvals": approvals.get(row.id) or [],
            }
            for row in rows
        ],
    }


def _upcoming(day: date, today: date) -> date:
    """A leave date named without a year means the next time that month and day occur."""
    if day >= today:
        return day
    return date(day.year + 1, day.month, day.day)


def parse_leave_dates(message: str, today: date | None = None) -> dict | None:
    """Dates named in a leave request. An end of None means the caller supplies the length."""
    today = today or org_today()
    text = message.lower()
    leave_type = "sick" if re.search(r"\bsick\b", text) else "annual"

    def _pack(start: date, end: date | None) -> dict:
        if end is not None and end < start:
            start, end = end, start
        return {"start": start, "end": end, "leave_type": leave_type}

    # ISO ranges: 2026-12-07 to 2026-12-18
    iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if len(iso) >= 2:
        start = date(int(iso[0][0]), int(iso[0][1]), int(iso[0][2]))
        end = date(int(iso[1][0]), int(iso[1][1]), int(iso[1][2]))
        return _pack(start, end)
    # Single ISO date: "... for 2026-12-28" / "on 2026-12-28"
    if len(iso) == 1:
        start = date(int(iso[0][0]), int(iso[0][1]), int(iso[0][2]))
        return _pack(start, None)

    # Month-first same-month span: December 7-18, 2026 / December 7 to 18, 2026
    same_month = re.search(
        rf"\b({_MONTH})\s+{_DAY}\s*(?:–|—|-|to|through|thru|until|and)\s*{_DAY}(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if same_month:
        year = same_month.group(4)
        start = _named_day(same_month.group(1), same_month.group(2), year, today)
        end = _named_day(same_month.group(1), same_month.group(3), year, today)
        return _pack(_upcoming(start, today), _upcoming(end, today))

    # Day-first cross/same month: 7 December to 18 December 2026
    day_first_range = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:,?\s*(\d{{4}}))?\s*"
        rf"(?:to|-|–|—|through|thru|until|and)\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if day_first_range:
        end_year = day_first_range.group(6) or day_first_range.group(3)
        start = _named_day(day_first_range.group(2), day_first_range.group(1), day_first_range.group(3), today)
        end = _named_day(day_first_range.group(5), day_first_range.group(4), end_year, today)
        return _pack(_upcoming(start, today), _upcoming(end, today))

    # Month-first cross-month: December 7 to January 5, 2027
    month_first_range = re.search(
        rf"\b(?:from|between)?\s*({_MONTH})\s+{_DAY}(?:,?\s*(\d{{4}}))?\s+"
        rf"(?:to|through|thru|until|and)\s+({_MONTH})\s+{_DAY}(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if month_first_range:
        start = _named_day(
            month_first_range.group(1), month_first_range.group(2), month_first_range.group(3), today
        )
        end = _named_day(
            month_first_range.group(4), month_first_range.group(5), month_first_range.group(6), today
        )
        return _pack(_upcoming(start, today), _upcoming(end, today))

    # "December 7 and 18"
    pair = re.search(rf"\b({_MONTH})\s+{_DAY}(?:,?\s*(\d{{4}}))?\s+and\s+{_DAY}\b", text)
    if pair:
        start = _upcoming(_named_day(pair.group(1), pair.group(2), pair.group(3), today), today)
        end = _upcoming(_named_day(pair.group(1), pair.group(4), pair.group(3), today), today)
        return _pack(start, end)

    # Relative duration with a start anchor: two weeks starting December 7
    relative = re.search(
        rf"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fourteen)\s+"
        rf"(business\s+days?|days?|weeks?)\s+(?:off\s+)?(?:starting|from|beginning)\s+"
        rf"(?:({_MONTH})\s+{_DAY}|(\d{{1,2}})\s+({_MONTH}))(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if relative:
        count_token = relative.group(1)
        unit = relative.group(2)
        if relative.group(3):
            start = _named_day(relative.group(3), relative.group(4), relative.group(7), today)
        else:
            start = _named_day(relative.group(6), relative.group(5), relative.group(7), today)
        start = _upcoming(start, today)
        count = _month_count(count_token) or (int(count_token) if count_token.isdigit() else None)
        if count:
            if "week" in unit:
                # Calendar weeks converted later via business-day planner when end is open;
                # approximate inclusive calendar span for explicit end when asked.
                end = start + timedelta(days=(count * 7) - 1)
                return _pack(start, end)
            # business days / days: leave end open so leave_analysis uses stated duration
            return _pack(start, None)

    starting = re.search(
        rf"\b(?:starting|from|on)\s+(?:({_MONTH})\s+{_DAY}|(\d{{1,2}})\s+({_MONTH}))(?:,?\s*(\d{{4}}))?\b",
        text,
    )
    if starting:
        if starting.group(1):
            start = _named_day(starting.group(1), starting.group(2), starting.group(5), today)
        else:
            start = _named_day(starting.group(4), starting.group(3), starting.group(5), today)
        return _pack(_upcoming(start, today), None)

    if re.search(r"\btomorrow\b", text) and re.search(r"\b(sick|leave|request)\b", text):
        day = today + timedelta(days=1)
        return _pack(day, day)
    return None


def profile_report(
    db: Session,
    employee: Employee,
    *,
    include_salary: bool,
    include_performance: bool,
    include_phone: bool = False,
) -> dict:
    """Profile of the signed-in employee. Phone is included only when the caller's access policy allows it."""
    manager = db.get(Employee, employee.manager_id) if employee.manager_id else None
    approval = db.get(Employee, employee.approval_manager_id) if employee.approval_manager_id else None
    department = db.get(Department, employee.department_id)
    department_name = department.name if department is not None else ""
    assignment_rows = (
        db.query(Project, ProjectAssignment)
        .join(ProjectAssignment, ProjectAssignment.project_id == Project.id)
        .filter(ProjectAssignment.employee_id == employee.id, Project.status == "active")
        .order_by(Project.name)
        .all()
    )
    projects = [project.name for project, _assignment in assignment_rows]
    project_assignments = [
        {
            "name": project.name,
            "code": project.code,
            "allocation_pct": assignment.allocation_pct,
            "role_name": assignment.role_name,
        }
        for project, assignment in assignment_rows
    ]
    populated = []
    empty = []
    field_values = {
        "employee_number": employee.employee_number,
        "full_name": employee.full_name,
        "job_title": employee.job_title,
        "department": department_name,
        "location": employee.location,
        "manager_id": employee.manager_id,
        "approval_manager_id": employee.approval_manager_id,
        "hire_date": employee.hire_date,
        "employment_status": employee.employment_status,
        "grade": employee.grade,
        "phone": employee.phone,
        "prior_department_id": employee.prior_department_id,
        "transfer_effective_date": employee.transfer_effective_date,
    }
    for name, value in field_values.items():
        if value is None or value == "":
            empty.append(name)
        else:
            populated.append(name)
    # Schema-adjacent attributes that are intentionally not employee columns.
    not_in_schema = [
        "home_address",
        "date_of_birth",
        "emergency_contact",
        "scheduled_daily_hours",
    ]
    payload = {
        "kind": "employee_profile",
        "employee_id": employee.id,
        "employee_number": employee.employee_number,
        "full_name": employee.full_name,
        "job_title": employee.job_title,
        "department": department_name,
        "location": employee.location,
        "manager_id": employee.manager_id,
        "manager": None if manager is None else manager.full_name,
        "approval_manager_id": employee.approval_manager_id,
        "approval_manager": None if approval is None else approval.full_name,
        "department_id": employee.department_id,
        "routing_consistent": (
            employee.department_id is not None
            and employee.manager_id is not None
            and employee.approval_manager_id is not None
            and employee.manager_id == employee.approval_manager_id
            and manager is not None
            and approval is not None
            and department is not None
        ),
        "hire_date": employee.hire_date.isoformat() if employee.hire_date else None,
        "employment_status": employee.employment_status,
        "grade": employee.grade,
        "tenure_days": (org_today() - employee.hire_date).days if employee.hire_date else None,
        "tenure_years": round((org_today() - employee.hire_date).days / 365.25, 2) if employee.hire_date else None,
        "projects": projects,
        "project_assignments": project_assignments,
        "timezone": org_timezone_name(),
        "as_of": org_today().isoformat(),
        "profile_completeness": {
            "populated_fields": populated,
            "empty_fields": empty,
            "not_in_schema": not_in_schema + ["blood_type"],
        },
    }
    if include_phone:
        payload["phone"] = employee.phone
    if include_salary:
        payload["salary_annual"] = employee.salary_annual
    if include_performance:
        payload["performance_rating"] = employee.performance_rating
    # Cross-record convenience: attach annual remaining when hire+leave are asked together.
    from app.models import LeaveBalance

    annual = (
        db.query(LeaveBalance)
        .filter(
            LeaveBalance.employee_id == employee.id,
            LeaveBalance.leave_type == "annual",
            LeaveBalance.year == org_today().year,
        )
        .one_or_none()
    )
    if annual is not None:
        remaining = float(annual.entitled_days) - float(annual.used_days) - float(annual.pending_days)
        payload["balance"] = {
            "leave_type": "annual",
            "entitled": float(annual.entitled_days),
            "used": float(annual.used_days),
            "pending": float(annual.pending_days),
            "remaining": remaining,
        }
    return payload
