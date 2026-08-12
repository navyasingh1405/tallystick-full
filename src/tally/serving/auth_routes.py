"""Sign-in, registration, and instant demo access."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from tally.auth.deps import (
    auth_store,
    clear_session_cookie,
    client_ip,
    current_user,
    current_user_optional,
    new_demo_credentials,
    set_session_cookie,
)
from tally.auth.passwords import MIN_LENGTH, PasswordPolicyError, check_policy
from tally.auth.store import (
    AccountDisabled,
    EmailAlreadyRegistered,
    InvalidCredentials,
    RateLimited,
    User,
)
from tally.config import get_config

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=MIN_LENGTH, max_length=1024)
    full_name: str | None = Field(default=None, max_length=120)
    organisation: str | None = Field(default=None, max_length=160)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class PasswordCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: str = Field(max_length=1024)
    email: str = ""


def _session_response(user: User, request: Request, response: Response, hours: int | None = None):
    store = auth_store()
    session = store.start_session(user, user_agent=request.headers.get("user-agent"))
    set_session_cookie(response, session.token, hours=hours)
    return {"user": user.as_dict(), "expires_at": session.expires_at}


@router.post("/register")
def register(payload: RegisterRequest, request: Request, response: Response) -> dict[str, Any]:
    """Create an account.

    New accounts get the `demo` role. Elevation is a deliberate act performed with
    `tally user role`, never something a signup form can grant itself.
    """
    store = auth_store()
    try:
        user = store.create_user(
            str(payload.email),
            payload.password,
            full_name=payload.full_name,
            organisation=payload.organisation,
            role="demo",
        )
    except EmailAlreadyRegistered:
        # Registration necessarily discloses that an address is taken; there is no
        # way around it for a form that must prevent duplicates. Sign-in does not.
        raise HTTPException(status_code=409, detail="that email is already registered") from None
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    return _session_response(user, request, response)


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    store = auth_store()
    try:
        user = store.authenticate(
            str(payload.email), payload.password, client_ip=client_ip(request)
        )
    except RateLimited as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from None
    except AccountDisabled as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except InvalidCredentials:
        # One message for both "no such account" and "wrong password", so the form
        # cannot be used to enumerate who has an account here.
        raise HTTPException(status_code=401, detail="email or password is incorrect") from None

    return _session_response(user, request, response)


@router.post("/demo")
def instant_demo(request: Request, response: Response) -> dict[str, Any]:
    """Provision a scoped, time-limited demo account and sign it in.

    An evaluator should be able to look at a demonstration built on public data
    without inventing a password first. Each click gets its own account rather than
    a shared one, so activity stays attributable and any single session can be
    revoked without affecting anyone else.
    """
    cfg = get_config().auth
    if not bool(cfg["allow_instant_demo"]):
        raise HTTPException(status_code=403, detail="instant demo access is disabled")

    email, password = new_demo_credentials()
    store = auth_store()
    user = store.create_user(
        email,
        password,
        full_name="Demo evaluator",
        organisation="Guest access",
        role="demo",
    )
    hours = int(cfg["demo_session_hours"])
    body = _session_response(user, request, response, hours=hours)
    body["demo"] = True
    body["expires_in_hours"] = hours
    body["note"] = (
        "This is a guest account on a demonstration system operating on public FFIEC "
        "data. It expires by itself, and it is not a route into anything real."
    )
    return body


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, Any]:
    from tally.auth.deps import cookie_name

    auth_store().end_session(request.cookies.get(cookie_name()))
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me")
def me(user: Annotated[User | None, Depends(current_user_optional)]) -> dict[str, Any]:
    """Who is signed in. Returns `authenticated: false` rather than 401, so the
    dashboard can ask without treating a signed-out visitor as an error."""
    if user is None:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": user.as_dict()}


@router.post("/check-password")
def check_password_strength(payload: PasswordCheckRequest) -> dict[str, Any]:
    """Live feedback for the signup form.

    The password is checked and discarded; it is never stored, logged, or echoed.
    """
    result = check_policy(payload.password, email=payload.email)
    return {"ok": result.ok, "problems": list(result.problems), "min_length": MIN_LENGTH}


@router.get("/sessions")
def my_sessions(user: Annotated[User, Depends(current_user)]) -> dict[str, Any]:
    return {"active_sessions": auth_store().active_session_count(user.id)}


@router.post("/sessions/revoke-all")
def revoke_all(
    user: Annotated[User, Depends(current_user)], response: Response
) -> dict[str, Any]:
    auth_store().end_all_sessions(user.id)
    clear_session_cookie(response)
    return {"ok": True, "note": "All sessions for this account were signed out."}
