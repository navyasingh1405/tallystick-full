"""API contract.

The service is what the demo runs against, so the shapes the dashboard depends on
are pinned here.
"""

from __future__ import annotations

import pytest

DECLINING = {
    "income_usd": 66000,
    "loan_amount_usd": 345000,
    "property_value_usd": 375000,
    "cltv": 92,
    "dti_band": ">60%",
    "loan_term": 360,
    "tract_to_msa_income_pct": 88,
    "units_band": "1",
    "preapproval": False,
    "interest_only": False,
    "balloon": False,
    "neg_amortization": False,
}


def test_health_reports_model_and_chain(signed_in):
    client, _user = signed_in
    body = client.get("/api/health").json()
    for key in ("ok", "model_version", "ledger_height", "chain_ok", "artifacts"):
        assert key in body


def test_the_dashboard_is_served_from_the_same_origin(signed_in):
    """One process, one port, no CORS, no build step."""
    client, _user = signed_in
    response = client.get("/")
    assert response.status_code == 200
    assert "Tallystick" in response.text


def test_api_routes_win_over_the_static_mount(signed_in):
    """The static mount is at '/', so it must be added last or it would swallow
    every API path."""
    client, _user = signed_in
    assert client.get("/api/health").headers["content-type"].startswith("application/json")


def test_a_protected_attribute_is_rejected_not_ignored(signed_in):
    """Extra fields are forbidden rather than dropped. Silently ignoring a submitted
    race field would mean the caller believes it was used."""
    client, _user = signed_in
    response = client.post("/api/score", json={**DECLINING, "derived_race": "White"})
    assert response.status_code == 422


def test_invalid_input_never_reaches_the_ledger(signed_in):
    client, _user = signed_in
    before = client.get("/api/health").json()["ledger_height"]
    assert client.post("/api/score", json={**DECLINING, "income_usd": -5}).status_code == 422
    assert client.get("/api/health").json()["ledger_height"] == before


@pytest.mark.slow
@pytest.mark.needs_model
def test_score_returns_the_full_decision_record(signed_in):
    client, _user = signed_in
    body = client.post("/api/score", json=DECLINING).json()
    assert body["decision"] == "DECLINED"
    assert 0 <= body["probability"] <= 1
    assert body["reason_codes"]
    assert body["counterfactual"]
    assert body["ledger"]["seq"] >= 1
    assert len(body["model"]["hash"]) == 64

    for code in body["reason_codes"]:
        assert code["text"] and not code["text"].startswith(code["feature"])


@pytest.mark.slow
@pytest.mark.needs_model
def test_scoring_appends_exactly_one_ledger_row(signed_in):
    client, _user = signed_in
    before = client.get("/api/health").json()["ledger_height"]
    client.post("/api/score", json=DECLINING)
    assert client.get("/api/health").json()["ledger_height"] == before + 1


@pytest.mark.slow
@pytest.mark.needs_model
def test_verify_returns_chain_and_replay(signed_in):
    client, _user = signed_in
    decision_id = client.post("/api/score", json=DECLINING).json()["decision_id"]
    body = client.post(f"/api/decisions/{decision_id}/verify").json()
    assert body["chain"]["ok"] is True
    assert body["replay"]["match"] is True
    assert body["replay"]["abs_diff"] <= 1e-9


@pytest.mark.slow
@pytest.mark.needs_model
def test_an_override_creates_a_new_row_and_keeps_the_chain_intact(signed_in):
    client, _user = signed_in
    decision_id = client.post("/api/score", json=DECLINING).json()["decision_id"]
    body = client.post(
        f"/api/decisions/{decision_id}/override",
        json={
            "decision": "APPROVED",
            "reason": "Documented compensating factors reviewed manually.",
        },
    ).json()

    assert body["decision_id"] != decision_id
    assert body["supersedes"] == decision_id
    assert client.get("/api/ledger/verify").json()["ok"] is True

    original = client.get(f"/api/decisions/{decision_id}").json()
    assert original["decision"] == "DECLINED"  # untouched
    assert original["superseded_by"] == body["decision_id"]

    # The override row names the signed-in account, not anything the client sent.
    override_row = client.get(f"/api/decisions/{body['decision_id']}").json()
    assert override_row["override_actor"] == _user.email


def test_an_override_requires_a_documented_reason(signed_in):
    client, _user = signed_in
    response = client.post(
        "/api/decisions/does-not-matter/override",
        json={"decision": "APPROVED", "reason": "no"},
    )
    assert response.status_code == 422


def test_an_override_cannot_name_someone_else_as_the_actor(signed_in):
    """The actor comes from the session. Accepting it from the body would let anyone
    attribute an override to a colleague, which would make the record worthless."""
    client, _user = signed_in
    response = client.post(
        "/api/decisions/does-not-matter/override",
        json={
            "decision": "APPROVED",
            "reason": "A properly documented reason string.",
            "actor": "someone.else@example.com",
        },
    )
    assert response.status_code == 422


def test_unknown_decision_returns_404_with_the_error_envelope(signed_in):
    client, _user = signed_in
    response = client.get("/api/decisions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert "error" in response.json()
    assert set(response.json()["error"]) == {"code", "message", "details"}


@pytest.mark.slow
@pytest.mark.needs_model
def test_analysis_endpoints_return_their_artifacts(signed_in):
    client, _user = signed_in
    for path, key in (
        ("/api/fairness", "raw_disparity"),
        ("/api/lda/frontier", "headline"),
        ("/api/drift", "features"),
    ):
        response = client.get(path)
        if response.status_code == 503:
            pytest.skip(f"{path} artifact not built")
        assert response.status_code == 200
        assert key in response.json()


@pytest.mark.slow
@pytest.mark.needs_model
def test_frontier_points_carry_what_the_plot_needs(signed_in):
    client, _user = signed_in
    response = client.get("/api/lda/frontier")
    if response.status_code == 503:
        pytest.skip("no lda artifact")
    body = response.json()
    assert body["points"]
    assert any(p["is_incumbent"] for p in body["points"])
    assert any(p["on_frontier"] for p in body["points"])
    for p in body["points"]:
        assert {"id", "auc", "disparity_ame", "on_frontier"} <= set(p)


def test_batch_template_lists_the_accepted_columns(signed_in):
    client, _user = signed_in
    response = client.get("/api/batch/template.csv")
    assert response.status_code == 200
    header = response.text.splitlines()[0]
    assert "income_usd" in header
    # Protected columns belong on the template: fairness cannot be measured without
    # them. They are parsed into a separate object that scoring cannot accept.
    assert "derived_race" in header
