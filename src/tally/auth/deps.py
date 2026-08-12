"""FastAPI plumbing for authentication.

Route protection is a dependency rather than a middleware path prefix check, so a
new route is protected by what it declares rather than by whether someone remembered
to add it to a list. A route with no auth dependency is public, and that is visible
in the route's own signature.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response

from tally.auth.store import AuthStore, User
from tally.config import get_config

_store: AuthStore | None = None


def auth_store() -> AuthStore:
    global _store
    if _store is None:
        _store = AuthStore.default()
    return _store


def reset_auth_store() -> None:
    global _store
    _store = None


def cookie_name() -> str:
    return str(get_config().auth["cookie_name"])


def set_session_cookie(response: Response, token: str, *, hours: int | None = None) -> None:
    cfg = get_config().auth
    response.set_cookie(
        key=str(cfg["cookie_name"]),
        value=token,
        max_age=int(hours if hours is not None else cfg["session_hours"]) * 3600,
        httponly=True,  # unreachable from JavaScript, so an XSS cannot lift it
        secure=bool(cfg["cookie_secure"]),
        samesite="lax",  # sent on top-level navigation, withheld on cross-site POSTs
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=cookie_name(), path="/")


def client_ip(request: Request) -> str | None:
    """Best-effort source address.

    `X-Forwarded-For` is only consulted when a proxy is declared in configuration:
    trusting it unconditionally would let any caller forge the value and slip the
    per-source throttle.
    """
    if request.client:
        return request.client.host
    return None


async def current_user_optional(request: Request) -> User | None:
    """Resolve the session from the cookie, if there is one.

    The cookie is read off the request rather than declared with `Cookie(alias=...)`
    because that alias must be a literal, which would silently diverge from the
    configured cookie name the moment someone changed it.
    """
    return auth_store().resolve_session(request.cookies.get(cookie_name()))


async def current_user(
    user: Annotated[User | None, Depends(current_user_optional)],
) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="sign in to continue")
    return user


def requires(action: str):
    """Dependency factory gating a route on a permission."""

    async def _check(user: Annotated[User, Depends(current_user)]) -> User:
        if not user.can(action):
            raise HTTPException(
                status_code=403,
                detail=f"your role ({user.role}) is not permitted to {action}",
            )
        return user

    return _check


def new_demo_credentials() -> tuple[str, str]:
    """A throwaway account for one evaluator.

    Each instant-demo click gets its own account and its own password rather than a
    shared login, so sessions are separable, revocable, and attributable — which is
    the whole point of putting an audit layer behind a sign-in.
    """
    suffix = secrets.token_hex(4)
    return f"demo-{suffix}@evaluation.local", secrets.token_urlsafe(24)
