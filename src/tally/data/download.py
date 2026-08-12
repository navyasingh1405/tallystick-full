"""FFIEC HMDA Data Browser client.

Two properties of the API make a naive client dangerous:

1. Unknown query parameters are dropped silently rather than rejected. Sending
   `occupancy_types=1` (a parameter that does not exist) returns the same bytes as
   sending nothing — unfiltered data that looks filtered.
2. More than two filter *categories* is a hard 400. So the five-dimension filter
   in config.yaml cannot be expressed server-side at all.

The defence is a whitelist that raises on anything else, so neither trap is
reachable from this module, plus post-download assertions on the actual contents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

BASE_URL: Final[str] = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

#: The API accepts at most two *data* filter categories alongside the geography and
#: year. There is no occupancy, lien, loan-type, or business-purpose filter — those
#: are applied locally in DuckDB. Sending anything outside this set raises.
ALLOWED_API_PARAMS: Final[frozenset[str]] = frozenset(
    {"states", "msamds", "counties", "leis", "years", "actions_taken", "loan_purposes"}
)

#: Categories that count against the two-filter cap.
FILTER_CATEGORIES: Final[frozenset[str]] = frozenset({"actions_taken", "loan_purposes"})

#: Proves we received the Data Browser header convention rather than a Snapshot
#: file: the Snapshot calls this column `combined_loan_to_value_ratio`.
DATA_BROWSER_MARKER: Final[str] = "loan_to_value_ratio"


class DownloadError(RuntimeError):
    """Raised when the download is wrong in a way that would poison the model."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    rows: int
    bytes_written: int
    final_url: str
    from_cache: bool


def build_params(state: str, year: int) -> dict[str, str]:
    """The only query this project issues. Everything else is filtered locally."""
    params = {
        "states": state,
        "years": str(year),
        "actions_taken": "1,3",
        "loan_purposes": "1",
    }
    _validate_params(params)
    return params


def _validate_params(params: dict[str, str]) -> None:
    unknown = set(params) - ALLOWED_API_PARAMS
    if unknown:
        raise DownloadError(
            f"parameters {sorted(unknown)} are not accepted by the Data Browser API. "
            "Unknown parameters are dropped SILENTLY, yielding unfiltered data that "
            "looks correct. Filter these locally in cal.data.clean instead."
        )
    n_filters = len(set(params) & FILTER_CATEGORIES)
    if n_filters > 2:
        raise DownloadError(
            f"{n_filters} filter categories requested; the API caps at 2 and returns "
            "HTTP 400 'provide-two-or-less-filter-criteria' beyond that."
        )
    if "years" not in params:
        raise DownloadError("'years' is required by the Data Browser API")


def download_year(
    state: str,
    year: int,
    dest: Path,
    *,
    force: bool = False,
    timeout: float = 900.0,
) -> DownloadResult:
    """Fetch one state-year to `dest`, streaming, following the redirect to cache.

    The endpoint answers uncached queries by streaming and cached ones with a 301
    to a pre-materialised object on files.ffiec.cfpb.gov, so redirects must be
    followed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        rows = _count_rows(dest)
        _assert_header(dest)
        return DownloadResult(dest, rows, dest.stat().st_size, "(cached on disk)", True)

    params = build_params(state, year)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    written = 0
    started = time.perf_counter()

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", BASE_URL, params=params) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", "replace")[:400]
                raise DownloadError(
                    f"HTTP {resp.status_code} from {resp.url}\n{body}"
                )
            final_url = str(resp.url)
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
                    if written % (32 << 20) < (1 << 20):
                        print(f"    {written / 1e6:6.1f} MB", end="\r")

    print(f"    {written / 1e6:6.1f} MB in {time.perf_counter() - started:.0f}s")
    tmp.replace(dest)

    _assert_header(dest)
    rows = _count_rows(dest)
    return DownloadResult(dest, rows, written, final_url, False)


def _assert_header(path: Path) -> None:
    """Confirm we got the Data Browser CSV and not something else entirely."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip()
    if not header:
        raise DownloadError(f"{path.name} is empty")
    cols = {c.strip().strip('"').lower() for c in header.split(",")}
    if DATA_BROWSER_MARKER not in cols:
        sample = sorted(cols)[:12]
        raise DownloadError(
            f"{path.name} does not look like a Data Browser CSV: expected a "
            f"'{DATA_BROWSER_MARKER}' column. Got columns starting {sample}. "
            "The Snapshot file names this column 'combined_loan_to_value_ratio'."
        )
    for required in ("action_taken", "loan_purpose", "occupancy_type", "derived_race"):
        if required not in cols:
            raise DownloadError(f"{path.name} is missing required column '{required}'")


def _count_rows(path: Path) -> int:
    """Data rows, excluding the header. Counted on bytes; the file is large."""
    n = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            n += chunk.count(b"\n")
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size - 2))
        trailing = fh.read()
    if trailing and not trailing.endswith(b"\n"):
        n += 1  # final line without a trailing newline
    return max(0, n - 1)
