"""Export the pinned model for browser-side scoring.

The marketing site has no backend, so the landing page's scorer runs the real model in
the visitor's browser. That is only worth doing if it is genuinely the same model
producing genuinely the same numbers — a lookalike would be worse than a static
screenshot, because it would look live while being wrong.

What ships:
  * the tree ensemble, flattened into typed arrays for fast traversal
  * the threshold-anchored SHAP background, unchanged
  * the fitted transform parameters, so the browser builds the same feature vector

Verified by scripts/verify_browser_model.mjs, which scores a grid of applicants in
both implementations and fails on any disagreement.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def flatten_tree(node: dict[str, Any], nodes: list[dict[str, Any]]) -> int:
    """Depth-first flatten into an index-addressed array.

    LightGBM's dump is a nested object; a flat array of {feature, threshold, left,
    right, value} traverses faster in JavaScript and serialises smaller.
    """
    index = len(nodes)
    nodes.append({})

    if "leaf_value" in node:
        nodes[index] = {"f": -1, "v": float(node["leaf_value"])}
        return index

    # LightGBM uses `<=` for the left branch on numerical splits, and routes missing
    # values by `default_left`. Both have to be reproduced exactly or a small
    # fraction of applicants take the wrong branch.
    entry: dict[str, Any] = {
        "f": int(node["split_feature"]),
        "t": float(node["threshold"]),
        "d": bool(node.get("default_left", True)),
    }
    nodes[index] = entry
    entry["l"] = flatten_tree(node["left_child"], nodes)
    entry["r"] = flatten_tree(node["right_child"], nodes)
    return index


def export(out_path: Path) -> dict[str, Any]:
    from tally.config import get_config
    from tally.model.registry import load_current

    cfg = get_config()
    model = load_current()
    dump = model.booster.dump_model()

    trees = []
    for tree in dump["tree_info"]:
        nodes: list[dict[str, Any]] = []
        flatten_tree(tree["tree_structure"], nodes)
        trees.append({"nodes": nodes, "shrinkage": tree.get("shrinkage", 1.0)})

    background = pd.read_parquet(model.background_path)[model.feature_names]

    payload = {
        "version": model.version,
        "model_hash": model.model_hash,
        "features": list(model.feature_names),
        "monotone": model.monotone_constraints,
        "threshold": model.threshold,
        "raw_threshold": model.raw_threshold,
        "trees": trees,
        "background": background.to_dict(orient="list"),
        "fit_params": model.fit_params.to_json(),
        "reason_codes": _reason_code_table(),
        "dti_bands": _dti_bands(),
        "generated_from": {
            "state": cfg.data["state"],
            "train_year": cfg.data["train_year"],
            "oot_year": cfg.data["oot_year"],
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    out_path.write_text(raw, encoding="utf-8")

    compressed = gzip.compress(raw.encode(), 9)
    print(f"  {out_path.name:<24} {len(raw) / 1e6:5.2f} MB  ({len(compressed) / 1e6:4.2f} MB gzipped)")
    print(f"  trees {len(trees)}  ·  background {background.shape[0]} rows  ·  model {model.version}")
    return payload


def _reason_code_table() -> dict[str, Any]:
    from tally.explain.templates import FEATURE_TO_CODE, TEMPLATES

    return {
        "feature_to_code": dict(FEATURE_TO_CODE),
        "text": {
            code: {"applicant": t.applicant_text, "factor": t.factor}
            for code, t in TEMPLATES.items()
        },
    }


def _dti_bands() -> dict[str, Any]:
    from tally.data import schema

    return {
        "order": list(schema.DTI_BAND_ORDER),
        "label": dict(schema.DTI_BAND_LABEL),
        "units_order": list(schema.UNITS_BAND_ORDER),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "web" / "assets" / "model.json"))
    args = parser.parse_args()
    export(Path(args.out))


if __name__ == "__main__":
    main()
