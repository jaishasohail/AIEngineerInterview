from datetime import date, datetime, timedelta, timezone

from app.config import get_settings

# The simulated organization has one calendar date. Attendance and holidays use
# that date, not the server's local clock.
ORG_TIMEZONE = "UTC"


def org_today() -> date:
    return date.fromisoformat(get_settings().org_clock)


def org_timezone_name() -> str:
    return ORG_TIMEZONE


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_business_day(day: date, holidays: set[date] | None = None) -> bool:
    return day.weekday() < 5 and day not in (holidays or set())


def business_days_inclusive(start: date, end: date, holidays: set[date] | None = None) -> int:
    if end < start:
        return 0
    count = 0
    cursor = start
    while cursor <= end:
        if is_business_day(cursor, holidays):
            count += 1
        cursor += timedelta(days=1)
    return count


def add_business_days(start: date, days: int, holidays: set[date] | None = None) -> date:
    """Return the date that completes `days` business days beginning on start."""
    if days <= 0:
        raise ValueError("days must be positive")
    cursor = start
    remaining = days
    while True:
        if is_business_day(cursor, holidays):
            remaining -= 1
            if remaining == 0:
                return cursor
        cursor += timedelta(days=1)


def notice_business_days(today: date, start: date, holidays: set[date] | None = None) -> int:
    """Business days strictly after today and strictly before start."""
    if start <= today:
        return 0
    cursor = today + timedelta(days=1)
    count = 0
    while cursor < start:
        if is_business_day(cursor, holidays):
            count += 1
        cursor += timedelta(days=1)
    return count


def first_monday(year: int, month: int) -> date:
    cursor = date(year, month, 1)
    while cursor.weekday() != 0:
        cursor += timedelta(days=1)
    return cursor


def next_month_bounds(today: date) -> tuple[date, date]:
    if today.month == 12:
        start = date(today.year + 1, 1, 1)
    else:
        start = date(today.year, today.month + 1, 1)
    if start.month == 12:
        end = date(start.year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(start.year, start.month + 1, 1) - timedelta(days=1)
    return start, end


def shift_months(today: date, months: int) -> date:
    month_index = today.year * 12 + (today.month - 1) - months
    year, month = divmod(month_index, 12)
    return date(year, month + 1, 1)


def format_day(value: date | str) -> str:
    if isinstance(value, str):
        value = date.fromisoformat(value)
    return value.strftime("%d %b %Y")


def parse_day(value: str) -> date:
    return date.fromisoformat(value)
