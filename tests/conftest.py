"""Shared fixtures.

Tests that need real data or a trained model are marked so `tally test-fast` can skip
them. Everything structural — canonicalisation, Pareto maths, the protected-attribute
boundary — runs without either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tally.config import get_config


@pytest.fixture(scope="session")
def cfg():
    return get_config()


def _have(path) -> bool:
    return path.exists()


@pytest.fixture(scope="session")
def model():
    """The current registered model, or skip."""
    from tally.model.registry import ModelRegistry, load_current

    try:
        ModelRegistry().current_version()
    except FileNotFoundError:
        pytest.skip("no trained model — run: uv run tally train")
    return load_current()


@pytest.fixture(scope="session")
def train_frame(cfg):
    from tally.data.splits import load_split

    if not _have(cfg.paths.data_processed / "train.parquet"):
        pytest.skip("no processed data — run: uv run tally data")
    return load_split("train")


@pytest.fixture(scope="session")
def eval_frame(cfg):
    """The out-of-time frame. Fairness and drift both measure against this."""
    from tally.data.splits import load_split

    if not _have(cfg.paths.data_processed / "oot.parquet"):
        pytest.skip("no processed data — run: uv run tally data")
    return load_split("oot")


@pytest.fixture(scope="session")
def golden_vectors():
    """Frozen input -> hash pairs for the ledger canonical form.

    If these change, the serialisation changed and every chain written under the old
    format is invalidated. Regenerate only as a deliberate act, with a CANON_VERSION
    bump: `uv run python tests/fixtures/make_golden.py`.
    """
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "canonical_golden.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def tmp_ledger(tmp_path):
    """A fresh ledger in a temp directory. Never touches the real one."""
    from tally.ledger.store import LedgerStore

    return LedgerStore(tmp_path / "decisions.db")


@pytest.fixture
def auth_store(tmp_path, monkeypatch):
    """An account store in a temp directory, wired into the service singleton.

    Tests must never create accounts in the developer's real accounts.db.
    """
    from tally.auth import deps
    from tally.auth.store import AuthStore

    store = AuthStore(tmp_path / "accounts.db")
    monkeypatch.setattr(deps, "_store", store)
    return store


@pytest.fixture
def client(auth_store, tmp_path, monkeypatch):
    """An unauthenticated TestClient whose ledger is a temporary one.

    Exercising the API writes decisions. Without this the suite would grow the
    developer's real ledger by a few hundred rows on every run.
    """
    from fastapi.testclient import TestClient

    from tally.ledger.store import LedgerStore
    from tally.serving import deps
    from tally.serving.app import app

    state = deps.ServiceState()
    state.ledger = LedgerStore(tmp_path / "decisions.db")
    monkeypatch.setattr(deps, "_state", state)

    return TestClient(app)


@pytest.fixture
def signed_in(client, auth_store):
    """A TestClient carrying a valid session, plus the user it belongs to."""
    user = auth_store.create_user(
        "tester@example.com", "tessellate-harbour-9182", full_name="Test User", role="analyst"
    )
    response = client.post(
        "/api/auth/login",
        json={"email": "tester@example.com", "password": "tessellate-harbour-9182"},
    )
    assert response.status_code == 200, response.text
    return client, user


@pytest.fixture(autouse=True)
def _guard_the_real_ledger(request, monkeypatch):
    """Fail any test that opens the developer's actual decision ledger.

    Tests write thousands of rows. Left unguarded they silently grow the real ledger,
    which then takes seconds to verify and makes the demo look sluggish. Every test
    that needs a ledger must be handed a temporary one — including the API tests,
    whose `client` fixture points the service at a throwaway database.

    This records which paths the test opened rather than comparing row counts before
    and after. A count comparison looks equivalent and is not: running `tally demo` in
    one terminal while the suite runs in another is an entirely reasonable thing to
    do, and every decision scored in the browser would then fail whichever test
    happened to be in flight. Attributing the write to the test that made it is the
    property actually worth asserting.
    """
    from tally.ledger.store import LedgerStore

    real = get_config().paths.ledger_db.resolve()
    opened: list[Path] = []
    original = LedgerStore.__init__

    def recording_init(self, path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(LedgerStore, "__init__", recording_init)
    yield

    offenders = [p for p in opened if p == real]
    assert not offenders, (
        f"{request.node.name} opened the real decision ledger at {real}. Pass a "
        "temporary LedgerStore instead — see the `tmp_ledger` or `client` fixtures."
    )
