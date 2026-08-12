"""Route protection.

The failure this guards against is quiet: a route added without an auth dependency
looks identical to one that is deliberately public. So the protected surface is
enumerated here, and a new unprotected route has to be justified in this file before
it can ship.
"""

from __future__ import annotations

import pytest

#: Everything that must refuse an anonymous caller. Method and path only.
PROTECTED = [
    ("GET", "/api/model/current"),
    ("POST", "/api/score"),
    ("GET", "/api/decisions/some-id"),
    ("POST", "/api/decisions/some-id/verify"),
    ("POST", "/api/decisions/some-id/override"),
    ("GET", "/api/decisions/some-id/adverse-action.pdf"),
    ("GET", "/api/ledger/recent"),
    ("GET", "/api/ledger/verify"),
    ("GET", "/api/fairness"),
    ("GET", "/api/drift"),
    ("GET", "/api/lda/frontier"),
    ("GET", "/api/lda/frontier.png"),
    ("POST", "/api/lda/adopt"),
    ("GET", "/api/batch/template.csv"),
    ("POST", "/api/batch"),
    ("GET", "/api/batch/some-job"),
    ("GET", "/api/batches/some-batch"),
    ("GET", "/api/batches/some-batch/results.csv"),
    ("POST", "/api/validation-pack"),
    ("GET", "/api/validation-pack/some-job"),
    ("GET", "/api/validation-pack/file/x.pdf"),
]

#: Deliberately open, each for a stated reason.
PUBLIC = [
    ("GET", "/", "the landing page has to be readable before anyone signs in"),
    ("GET", "/login", "the sign-in form cannot require sign-in"),
    ("GET", "/api/auth/me", "the client asks who is signed in before rendering"),
    ("POST", "/api/public/score", "the landing page scorer, rate limited"),
    ("GET", "/api/public/summary", "headline figures shown on the landing page"),
    ("GET", "/api/public/frontier.png", "an aggregate chart already published in the pack"),
    ("GET", "/api/health", "liveness, used by the landing page status strip"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_anonymous_callers_are_refused(client, method, path):
    response = client.request(method, path, json={} if method == "POST" else None)
    assert response.status_code == 401, f"{method} {path} answered {response.status_code}"


@pytest.mark.parametrize(("method", "path", "why"), PUBLIC)
def test_public_routes_stay_public(client, method, path, why):
    response = client.request(method, path, json={} if method == "POST" else None)
    assert response.status_code != 401, f"{method} {path} should be public: {why}"


def test_every_api_route_is_either_protected_or_listed_as_public(client):
    """Catches a new endpoint that nobody remembered to protect.

    Adding a route now forces a decision here, rather than defaulting to open.
    """
    from tally.serving.app import app

    known = {p for _, p in PROTECTED} | {p for _, p, _ in PUBLIC}
    known |= {
        "/api/auth/login",
        "/api/auth/register",
        "/api/auth/logout",
        "/api/auth/demo",
        "/api/auth/check-password",
        "/api/auth/sessions",
        "/api/auth/sessions/revoke-all",
        "/api/docs",
        "/api/openapi.json",
    }

    # The tables above hold concrete paths so they can be requested directly; the
    # app holds templates. Both collapse to the same shape by treating a route
    # template segment, or one of these stand-in values, as a wildcard.
    stand_ins = {"some-id", "some-job", "some-batch", "x.pdf"}

    def normalise(path: str) -> str:
        parts = [
            "{}" if (p.startswith("{") or p in stand_ins) else p for p in path.split("/")
        ]
        return "/".join(parts)

    declared = {normalise(p) for p in known}
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/"):
            continue
        assert normalise(path) in declared, (
            f"{path} is not listed in tests/test_auth_api.py. Decide whether it is "
            "protected or public, and add it to the corresponding table."
        )


def test_the_dashboard_redirects_a_signed_out_visitor(client):
    response = client.get("/app", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login?next=/app"


def test_the_dashboard_opens_for_a_signed_in_visitor(signed_in):
    client, _user = signed_in
    response = client.get("/app", follow_redirects=False)
    assert response.status_code == 200


# --------------------------------------------------------------------------- session


def test_registering_signs_you_in(client):
    response = client.post(
        "/api/auth/register",
        json={
            "email": "new.person@example.com",
            "password": "tessellate-harbour-9182",
            "full_name": "New Person",
            "organisation": "Example Bank",
        },
    )
    assert response.status_code == 200
    assert client.get("/api/auth/me").json()["authenticated"] is True


def test_the_session_cookie_is_not_reachable_from_javascript(client, auth_store):
    """HttpOnly is what stops a cross-site scripting bug from becoming a stolen
    session, and SameSite is what stops another origin from riding the cookie."""
    auth_store.create_user("a@b.com", "tessellate-harbour-9182")
    response = client.post(
        "/api/auth/login", json={"email": "a@b.com", "password": "tessellate-harbour-9182"}
    )
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "path=/" in header


def test_no_response_ever_carries_a_password_or_its_hash(signed_in):
    client, _user = signed_in
    for path in ("/api/auth/me", "/api/auth/sessions"):
        body = client.get(path).text.lower()
        assert "argon2" not in body
        assert "password_hash" not in body
        assert "tessellate" not in body


def test_signing_out_invalidates_the_cookie(signed_in):
    client, _user = signed_in
    assert client.get("/api/auth/me").json()["authenticated"] is True
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").json()["authenticated"] is False
    assert client.get("/api/fairness").status_code == 401


def test_a_forged_cookie_does_not_authenticate(client, auth_store):
    auth_store.create_user("a@b.com", "tessellate-harbour-9182")
    client.cookies.set("tally_session", "a-token-i-made-up")
    assert client.get("/api/auth/me").json()["authenticated"] is False


def test_sign_in_does_not_disclose_whether_an_account_exists(client, auth_store):
    auth_store.create_user("real@example.com", "tessellate-harbour-9182")

    wrong = client.post(
        "/api/auth/login", json={"email": "real@example.com", "password": "wrong-password-x"}
    )
    unknown = client.post(
        "/api/auth/login", json={"email": "ghost@example.com", "password": "wrong-password-x"}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]


def test_repeated_failures_are_throttled_over_http(client, auth_store):
    auth_store.create_user("a@b.com", "tessellate-harbour-9182")
    codes = [
        client.post("/api/auth/login", json={"email": "a@b.com", "password": "nope"}).status_code
        for _ in range(12)
    ]
    assert 429 in codes


def test_new_accounts_cannot_grant_themselves_a_role(client):
    """Role elevation is a command-line act. A signup form that could hand out roles
    would make the role meaningless."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "climber@example.com",
            "password": "tessellate-harbour-9182",
            "role": "admin",
        },
    )
    assert response.status_code == 422  # extra field forbidden


def test_a_weak_password_is_refused_at_registration(client):
    response = client.post(
        "/api/auth/register", json={"email": "a@b.com", "password": "password1234"}
    )
    assert response.status_code == 422


def test_the_password_check_endpoint_does_not_echo_the_password(client):
    response = client.post(
        "/api/auth/check-password", json={"password": "a-very-distinctive-string-here"}
    )
    assert response.status_code == 200
    assert "a-very-distinctive-string-here" not in response.text


# --------------------------------------------------------------------------- demo


def test_instant_demo_provisions_a_working_session(client):
    response = client.post("/api/auth/demo")
    assert response.status_code == 200
    body = response.json()
    assert body["demo"] is True
    assert body["user"]["role"] == "demo"
    assert client.get("/api/auth/me").json()["authenticated"] is True


def test_each_demo_click_gets_its_own_account(client, auth_store):
    """A shared guest login would make every guest action indistinguishable, and one
    revocation would sign out everybody."""
    first = client.post("/api/auth/demo").json()["user"]["email"]
    client.cookies.clear()
    second = client.post("/api/auth/demo").json()["user"]["email"]
    assert first != second
    assert auth_store.count_users() == 2


def test_a_demo_account_can_use_the_product_but_is_not_an_admin(client):
    body = client.post("/api/auth/demo").json()
    permissions = set(body["user"]["permissions"])
    assert {"score", "read", "verify", "pack", "batch"} <= permissions
    assert "admin" not in permissions


# --------------------------------------------------------------------------- public


def test_the_public_scorer_records_its_decisions(client):
    """The claim is that no path scores without recording. A marketing-page exception
    would make that false exactly where it is being advertised."""
    from tally.serving import public_routes

    public_routes.reset_rate_limits()
    response = client.post(
        "/api/public/score",
        json={
            "income_usd": 66000,
            "loan_amount_usd": 345000,
            "cltv": 92,
            "dti_band": ">60%",
            "loan_term": 360,
        },
    )
    if response.status_code == 503:
        pytest.skip("no model registered")
    assert response.status_code == 200
    body = response.json()
    assert body["recorded"]["seq"] >= 1
    assert len(body["recorded"]["row_hash"]) == 64


def test_the_public_scorer_cannot_accept_a_protected_attribute(client):
    from tally.serving import public_routes

    public_routes.reset_rate_limits()
    response = client.post(
        "/api/public/score",
        json={
            "income_usd": 66000,
            "loan_amount_usd": 345000,
            "cltv": 92,
            "dti_band": ">60%",
            "loan_term": 360,
            "derived_race": "White",
        },
    )
    assert response.status_code == 422


def test_the_public_scorer_is_rate_limited(client):
    from tally.serving import public_routes

    public_routes.reset_rate_limits()
    payload = {
        "income_usd": 120000,
        "loan_amount_usd": 300000,
        "cltv": 80,
        "dti_band": "38",
        "loan_term": 360,
    }
    codes = [client.post("/api/public/score", json=payload).status_code for _ in range(40)]
    if 503 in codes:
        pytest.skip("no model registered")
    assert 429 in codes, "an unauthenticated compute endpoint must be throttled"
    public_routes.reset_rate_limits()
