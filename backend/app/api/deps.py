from collections.abc import Generator

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import session_scope
from app.models import User
from app.security import AuthError, decode_token


def get_db() -> Generator[Session, None, None]:
    yield from session_scope()


def get_current_user(
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
) -> User:
    """Authenticate from JWT, then load the live User row.

    Role and employee linkage always come from the database — never from JWT claims —
    so forged role/employee_id fields cannot escalate privileges.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in required")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
        user = db.get(User, int(payload["sub"]))
    except (AuthError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Sign in required") from None
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Sign in required")
    return user


def client_ip(request: Request) -> str | None:
    if request.client is None:
        return None
    return request.client.host
