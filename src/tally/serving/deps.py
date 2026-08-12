"""Shared singletons and the in-process job registry.

The ledger is a single-writer SQLite database, so the service must run with one
worker. That is asserted at startup rather than left as a comment nobody reads.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tally.config import get_config
from tally.ledger.store import LedgerStore
from tally.model.registry import LoadedModel, load_current


class ServiceState:
    def __init__(self) -> None:
        self.config = get_config()
        self._ledger: LedgerStore | None = None
        self._model: LoadedModel | None = None
        self._chain_cache: tuple[tuple[int, str], Any] | None = None

    @property
    def ledger(self) -> LedgerStore:
        """Opened on first use, not on construction.

        `LedgerStore.__init__` creates the database file and applies the schema, so
        eagerly opening it here would mean that merely constructing this object
        touches the filesystem — which makes the service impossible to point at a
        different ledger without first creating the default one as a side effect.
        """
        if self._ledger is None:
            self._ledger = LedgerStore.default()
        return self._ledger

    @ledger.setter
    def ledger(self, store: LedgerStore) -> None:
        self._ledger = store
        self._chain_cache = None

    @property
    def model(self) -> LoadedModel:
        if self._model is None:
            self._model = load_current()
        return self._model

    def model_or_none(self) -> LoadedModel | None:
        try:
            return self.model
        except FileNotFoundError:
            return None

    def chain_status(self) -> Any:
        """Chain verification, cached against the ledger's height and head hash.

        The status strip polls this every few seconds, and a full walk over a
        seventy-thousand-row ledger takes seconds — which would make the dashboard
        feel broken and would dominate the server's work. Height and head hash are
        cheap to read, and any append changes both, so the cache is invalidated
        exactly when the ledger changes and a full walk runs then.

        This is a caching decision, not a weakening of the check: nothing is
        reported as verified that was not fully verified against the current
        contents. `/api/ledger/verify` and the Verify & Replay button always force a
        fresh walk.
        """
        from tally.ledger.verify import verify_chain

        height, head = self.ledger.height(), self.ledger.head_hash()
        cached = self._chain_cache

        if cached and cached[0] == (height, head):
            return cached[1]

        if cached and cached[1].ok and height > cached[0][0]:
            # Rows were appended. Verify only the new suffix, linking it back to the
            # head this process already verified. Constant work per decision instead
            # of a full walk on every append.
            result = verify_chain(
                self.ledger,
                after_seq=cached[1].rows_checked,
                expected_prev_hash=cached[1].head_hash,
                rows_already_checked=cached[1].rows_checked,
            )
        else:
            result = verify_chain(self.ledger)

        self._chain_cache = ((height, head), result)
        return result

    def invalidate_chain_cache(self) -> None:
        self._chain_cache = None

    def artifact_status(self) -> dict[str, bool]:
        p = self.config.paths
        return {
            "data": (p.data_processed / "oot.parquet").exists(),
            "model": self.model_or_none() is not None,
            "metrics": p.artifact("metrics", "performance.json").exists(),
            "fairness": p.artifact("fairness", "fairness_report.json").exists(),
            "lda": p.artifact("lda", "lda_summary.json").exists(),
            "drift": p.artifact("drift", "drift_report.json").exists(),
        }


_state: ServiceState | None = None


def state() -> ServiceState:
    global _state
    if _state is None:
        _state = ServiceState()
    return _state


def reset_state() -> None:
    global _state
    _state = None


# --------------------------------------------------------------------------- jobs


@dataclass
class Job:
    """A long-running task. Deliberately in-process: no queue, no broker, no Redis.

    The two slow operations (batch scoring, PDF generation) take seconds to a minute
    and the service runs as a single worker, so a dict and a thread are the honest
    amount of machinery. Polling at 500ms is enough.
    """

    job_id: str
    status: str = "running"
    progress: float = 0.0
    detail: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, *, progress: float | None = None, detail: str | None = None) -> None:
        with self.lock:
            if progress is not None:
                self.progress = max(0.0, min(1.0, progress))
            if detail is not None:
                self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": round(self.progress, 3),
                "detail": self.detail,
                "result": self.result,
                "error": self.error,
            }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[Job], dict[str, Any]]) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job

        def run() -> None:
            try:
                result = fn(job)
                with job.lock:
                    job.result = result
                    job.status = "done"
                    job.progress = 1.0
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                with job.lock:
                    job.status = "error"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.detail = traceback.format_exc(limit=3)

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


jobs = JobRegistry()
