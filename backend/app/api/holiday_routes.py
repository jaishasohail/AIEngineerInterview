from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import client_ip, get_current_user, get_db
from app.audit import write_audit
from app.models import Department, Holiday, User

router = APIRouter()


class HolidayIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    holiday_date: date
    country: str = Field(min_length=2, max_length=8)
    region: str = ""
    department_id: int | None = None
    is_active: bool = True


class HolidayPatch(BaseModel):
    name: str | None = None
    holiday_date: date | None = None
    country: str | None = None
    region: str | None = None
    department_id: int | None = None
    is_active: bool | None = None
    clear_department: bool = False


def _payload(row: Holiday, departments: dict[int, str]) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "holiday_date": row.holiday_date.isoformat(),
        "country": row.country,
        "region": row.region,
        "year": row.year,
        "department_id": row.department_id,
        "department": None if row.department_id is None else departments.get(row.department_id),
        "is_active": row.is_active,
        "source": row.source,
    }


def _require_hr(user: User) -> None:
    if user.role != "hr_admin":
        raise HTTPException(status_code=403, detail="Only People Operations can change holiday calendars")


@router.get("/holidays")
def list_holidays(
    country: str = "",
    region: str = "",
    year: int | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    query = db.query(Holiday)
    if country.strip():
        query = query.filter(Holiday.country == country.strip().upper())
    if region.strip():
        query = query.filter(Holiday.region == region.strip())
    if year is not None:
        query = query.filter(Holiday.year == year)
    if user.role != "hr_admin":
        query = query.filter(Holiday.is_active.is_(True))
    rows = query.order_by(Holiday.holiday_date, Holiday.country, Holiday.name).all()
    departments = {item.id: item.name for item in db.query(Department).all()}
    return {"holidays": [_payload(row, departments) for row in rows]}


@router.post("/holidays")
def create_holiday(
    body: HolidayIn,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _require_hr(user)
    if body.department_id is not None and db.get(Department, body.department_id) is None:
        raise HTTPException(status_code=422, detail="Department not found")
    country = body.country.strip().upper()
    region = body.region.strip()
    exists = (
        db.query(Holiday)
        .filter(
            Holiday.holiday_date == body.holiday_date,
            Holiday.country == country,
            Holiday.region == region,
            Holiday.name == body.name.strip(),
            Holiday.department_id == body.department_id if body.department_id is not None else Holiday.department_id.is_(None),
        )
        .one_or_none()
    )
    if exists is not None:
        raise HTTPException(status_code=409, detail="That holiday is already on the calendar")
    row = Holiday(
        name=body.name.strip(),
        holiday_date=body.holiday_date,
        country=country,
        region=region,
        year=body.holiday_date.year,
        department_id=body.department_id,
        is_active=body.is_active,
        source="admin",
    )
    db.add(row)
    db.commit()
    write_audit(
        db,
        actor_user_id=user.id,
        action="holiday_create",
        resource_type="holiday",
        resource_id=str(row.id),
        outcome="success",
        details={"name": row.name, "date": row.holiday_date.isoformat(), "country": row.country, "region": row.region},
        ip_address=client_ip(request),
    )
    departments = {item.id: item.name for item in db.query(Department).all()}
    return _payload(row, departments)


@router.patch("/holidays/{holiday_id}")
def update_holiday(
    holiday_id: int,
    body: HolidayPatch,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _require_hr(user)
    row = db.get(Holiday, holiday_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Holiday not found")
    if body.name is not None:
        row.name = body.name.strip()
    if body.holiday_date is not None:
        row.holiday_date = body.holiday_date
        row.year = body.holiday_date.year
    if body.country is not None:
        row.country = body.country.strip().upper()
    if body.region is not None:
        row.region = body.region.strip()
    if body.clear_department:
        row.department_id = None
    elif body.department_id is not None:
        if db.get(Department, body.department_id) is None:
            raise HTTPException(status_code=422, detail="Department not found")
        row.department_id = body.department_id
    if body.is_active is not None:
        row.is_active = body.is_active
    db.commit()
    write_audit(
        db,
        actor_user_id=user.id,
        action="holiday_update",
        resource_type="holiday",
        resource_id=str(row.id),
        outcome="success",
        details={"name": row.name, "is_active": row.is_active},
        ip_address=client_ip(request),
    )
    departments = {item.id: item.name for item in db.query(Department).all()}
    return _payload(row, departments)


@router.delete("/holidays/{holiday_id}")
def delete_holiday(
    holiday_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _require_hr(user)
    row = db.get(Holiday, holiday_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Holiday not found")
    db.delete(row)
    db.commit()
    write_audit(
        db,
        actor_user_id=user.id,
        action="holiday_delete",
        resource_type="holiday",
        resource_id=str(holiday_id),
        outcome="success",
        details={"name": row.name, "date": row.holiday_date.isoformat()},
        ip_address=client_ip(request),
    )
    return {"deleted": holiday_id}
