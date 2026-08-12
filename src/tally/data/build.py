"""Orchestrates download -> clean -> split, and records the lineage.

`data_lineage.json` is the sole source for validation pack section 2. Nothing in
that section is typed by hand.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tally.config import get_config
from tally.data.clean import build_frame
from tally.data.download import DownloadError, download_year
from tally.data.splits import make_splits


def build_all(*, force: bool = False) -> dict[str, Any]:
    cfg = get_config()
    cfg.paths.ensure()
    d = cfg.data
    state = d["state"]
    years = [int(d["train_year"]), int(d["oot_year"])]

    lineage: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": "FFIEC HMDA Data Browser — public Loan/Application Register",
            "endpoint": "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv",
            "state": state,
            "train_year": d["train_year"],
            "oot_year": d["oot_year"],
            "note": (
                "Only actions_taken and loan_purposes can be applied server-side: the "
                "API caps at two filter categories and has no occupancy, lien, loan "
                "type, or business-purpose parameter. The remaining filters are applied "
                "locally and their cost is shown stage by stage below."
            ),
        },
        "filters": d["filters"],
        "years": {},
        "stages": [],
    }

    frames: dict[int, Path] = {}
    for year in years:
        print(f"\n  [{state} {year}] downloading")
        raw = cfg.paths.data_raw / f"hmda_{state}_{year}.csv"
        dl = download_year(state, year, raw, force=force)
        print(f"  [{state} {year}] {dl.rows:,} rows" + ("  (cached)" if dl.from_cache else ""))

        lo, hi = d["expected_api_rows"]
        if not lo <= dl.rows <= hi:
            raise DownloadError(
                f"{state} {year} returned {dl.rows:,} rows; expected between {lo:,} and "
                f"{hi:,}. Either the filter was silently dropped or the vintage changed. "
                "Verify the request before training on this."
            )

        print(f"  [{state} {year}] cleaning")
        out = cfg.paths.data_processed / f"frame_{state}_{year}.parquet"
        clean = build_frame(raw, out, d)
        frames[year] = out

        retention = clean.final_rows / dl.rows if dl.rows else 0.0
        rlo, rhi = d["expected_filter_retention"]
        if not rlo <= retention <= rhi:
            raise DownloadError(
                f"{state} {year} retained {retention:.1%} of API rows after local "
                f"filtering; expected {rlo:.0%}-{rhi:.0%}. A retention far outside this "
                "band means a filter did not apply as intended."
            )

        lineage["years"][str(year)] = {
            "api_rows": dl.rows,
            "final_rows": clean.final_rows,
            "retention": round(retention, 4),
            "bytes": dl.bytes_written,
            "frame": str(out.relative_to(cfg.paths.root)),
        }
        for s in clean.stages:
            lineage["stages"].append(
                {"year": year, "name": s.name, "rows": s.rows, "note": s.note}
            )

    print("\n  building splits")
    split_info = make_splits(frames[years[0]], frames[years[1]])
    lineage["splits"] = split_info
    for name in ("train", "test", "oot"):
        lineage["stages"].append(
            {
                "year": split_info[name]["year"],
                "name": f"split: {name}",
                "rows": split_info[name]["rows"],
                "note": split_info[name]["note"],
            }
        )

    path = cfg.paths.data_processed / "data_lineage.json"
    path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    lineage["path"] = str(path)
    return lineage


def load_lineage() -> dict[str, Any]:
    cfg = get_config()
    path = cfg.paths.data_processed / "data_lineage.json"
    if not path.exists():
        raise FileNotFoundError("data_lineage.json missing — run: uv run tally data")
    return json.loads(path.read_text(encoding="utf-8"))
