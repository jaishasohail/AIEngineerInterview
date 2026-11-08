"""Office holiday seed for the four Northstar locations.

These are published public holidays for the United States, England, Germany, and
Singapore in the seeded years. Dates that are not in this table are not inferred.
People Operations can add, disable, or remove rows through the holiday API.
"""

from datetime import date

from sqlalchemy.orm import Session

from app.models import Holiday

# (date, country, region, name). An empty region applies to every office in that country.
SEED_HOLIDAYS: list[tuple[date, str, str, str]] = [
    (date(2026, 1, 1), "US", "", "New Year's Day"),
    (date(2026, 1, 19), "US", "", "Martin Luther King Jr. Day"),
    (date(2026, 5, 25), "US", "", "Memorial Day"),
    (date(2026, 6, 19), "US", "", "Juneteenth"),
    (date(2026, 7, 3), "US", "", "Independence Day (observed)"),
    (date(2026, 7, 4), "US", "", "Independence Day"),
    (date(2026, 9, 7), "US", "", "Labor Day"),
    (date(2026, 11, 26), "US", "", "Thanksgiving Day"),
    (date(2026, 12, 25), "US", "", "Christmas Day"),
    (date(2027, 1, 1), "US", "", "New Year's Day"),
    (date(2026, 1, 1), "GB", "", "New Year's Day"),
    (date(2026, 4, 3), "GB", "", "Good Friday"),
    (date(2026, 4, 6), "GB", "", "Easter Monday"),
    (date(2026, 5, 4), "GB", "", "Early May bank holiday"),
    (date(2026, 5, 25), "GB", "", "Spring bank holiday"),
    (date(2026, 8, 31), "GB", "", "Summer bank holiday"),
    (date(2026, 12, 25), "GB", "", "Christmas Day"),
    (date(2026, 12, 28), "GB", "", "Boxing Day (substitute)"),
    (date(2027, 1, 1), "GB", "", "New Year's Day"),
    (date(2026, 1, 1), "DE", "", "New Year's Day"),
    (date(2026, 4, 3), "DE", "", "Good Friday"),
    (date(2026, 4, 6), "DE", "", "Easter Monday"),
    (date(2026, 5, 1), "DE", "", "Labour Day"),
    (date(2026, 5, 14), "DE", "", "Ascension Day"),
    (date(2026, 5, 25), "DE", "", "Whit Monday"),
    (date(2026, 10, 3), "DE", "", "German Unity Day"),
    (date(2026, 12, 25), "DE", "", "Christmas Day"),
    (date(2026, 12, 26), "DE", "", "Second Christmas Day"),
    (date(2027, 1, 1), "DE", "", "New Year's Day"),
    (date(2026, 1, 1), "SG", "", "New Year's Day"),
    (date(2026, 2, 17), "SG", "", "Chinese New Year"),
    (date(2026, 2, 18), "SG", "", "Chinese New Year (second day)"),
    (date(2026, 4, 3), "SG", "", "Good Friday"),
    (date(2026, 5, 1), "SG", "", "Labour Day"),
    (date(2026, 8, 9), "SG", "", "National Day"),
    (date(2026, 8, 10), "SG", "", "National Day (observed)"),
    (date(2026, 12, 25), "SG", "", "Christmas Day"),
    (date(2027, 1, 1), "SG", "", "New Year's Day"),
]


def seed_holidays(db: Session) -> int:
    added = 0
    for holiday_date, country, region, name in SEED_HOLIDAYS:
        exists = (
            db.query(Holiday)
            .filter(
                Holiday.holiday_date == holiday_date,
                Holiday.country == country,
                Holiday.region == region,
                Holiday.name == name,
                Holiday.department_id.is_(None),
            )
            .one_or_none()
        )
        if exists is not None:
            continue
        db.add(
            Holiday(
                name=name,
                holiday_date=holiday_date,
                country=country,
                region=region,
                year=holiday_date.year,
                department_id=None,
                is_active=True,
                source="seed",
            )
        )
        added += 1
    db.flush()
    return added
