"""Location-aware holiday lookup. Unknown offices do not inherit another country's dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from app.models import Employee, Holiday

OFFICE_COUNTRY = {
    "Austin": "US",
    "London": "GB",
    "Berlin": "DE",
    "Singapore": "SG",
}


@dataclass
class HolidayCalendar:
    location: str
    country: str | None
    status: str
    dates: set[date]


def holiday_applies(row: Holiday, *, country: str | None, location: str, department_id: int | None) -> bool:
    if row.region and row.region != location:
        return False
    if row.country != country and row.region != location:
        return False
    if not row.region and row.country != country:
        return False
    if row.department_id is not None and row.department_id != department_id:
        return False
    return True


def holidays_for_location(
    db: Session,
    *,
    location: str,
    department_id: int | None,
    start: date,
    end: date,
) -> HolidayCalendar:
    country = OFFICE_COUNTRY.get(location)
    rows = (
        db.query(Holiday)
        .filter(Holiday.is_active.is_(True), Holiday.holiday_date >= start, Holiday.holiday_date <= end)
        .all()
    )
    matched: set[date] = set()
    for row in rows:
        if holiday_applies(row, country=country, location=location, department_id=department_id):
            matched.add(row.holiday_date)
    configured = False
    if country is not None:
        configured = (
            db.query(Holiday.id)
            .filter(Holiday.is_active.is_(True), Holiday.country == country)
            .first()
            is not None
        )
    if not configured:
        configured = (
            db.query(Holiday.id).filter(Holiday.is_active.is_(True), Holiday.region == location).first() is not None
        )
    return HolidayCalendar(
        location=location,
        country=country,
        status="configured" if configured else "missing",
        dates=matched,
    )


def list_holidays_for_employee(db: Session, employee: Employee, start: date, end: date) -> tuple[list[dict], str]:
    """Named holidays for the employee's office. Dates are not copied from another country."""
    if end < start:
        start, end = end, start
    calendar = holidays_for_employee(db, employee, start, end)
    country = OFFICE_COUNTRY.get(employee.location)
    rows = (
        db.query(Holiday)
        .filter(Holiday.is_active.is_(True), Holiday.holiday_date >= start, Holiday.holiday_date <= end)
        .order_by(Holiday.holiday_date, Holiday.name)
        .all()
    )
    entries = []
    for row in rows:
        if not holiday_applies(row, country=country, location=employee.location, department_id=employee.department_id):
            continue
        entries.append({"name": row.name, "date": row.holiday_date.isoformat(), "country": row.country})
    return entries, calendar.status


def holidays_for_employee(db: Session, employee: Employee, start: date, end: date) -> HolidayCalendar:
    if end < start:
        start, end = end, start
    return holidays_for_location(
        db,
        location=employee.location,
        department_id=employee.department_id,
        start=start,
        end=end,
    )
