"""CLI: JSON in → JSON out using a saved SVR artefact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from winequality.config import ARTIFACTS, FEATURE_NAMES, PHYSICAL_BOUNDS, TYPE_COL
from winequality.postprocess import pack_output, physical_violations
from winequality.svr_model import formula_predict


def load_artefact(kind: str, path: Path | None = None):
    path = path or (ARTIFACTS / f"svr_{kind}.joblib")
    return joblib.load(path)


def predict_row(bundle: dict, row: dict, y_expert: float | None = None) -> dict:
    missing = [c for c in FEATURE_NAMES if c not in row]
    if missing:
        raise ValueError(f"missing features: {missing}")
    bad = physical_violations({k: float(row[k]) for k in FEATURE_NAMES})
    if bad:
        raise ValueError(f"values outside physical bounds: {bad} ({PHYSICAL_BOUNDS})")
    x = np.array([[float(row[c]) for c in FEATURE_NAMES]], dtype=float)
    f = float(formula_predict(bundle["pipeline"], x)[0])
    return pack_output(
        f,
        x={k: float(row[k]) for k in FEATURE_NAMES},
        train_min=bundle.get("train_min"),
        train_max=bundle.get("train_max"),
        y_expert=y_expert,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict wine quality from 11 lab measurements.")
    p.add_argument("--kind", choices=("red", "white"), required=True)
    p.add_argument("--json", help="JSON object or path to a JSON file")
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)
    bundle = load_artefact(args.kind, Path(args.model) if args.model else None)
    if args.json is None:
        raw = sys.stdin.read()
    else:
        path = Path(args.json)
        raw = path.read_text(encoding="utf-8") if path.exists() else args.json
    row = json.loads(raw)
    y_expert = row.pop("quality", None)
    row.pop(TYPE_COL, None)
    out = predict_row(bundle, row, y_expert=y_expert)
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
