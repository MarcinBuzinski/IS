"""JSON įvestis → prognozė, tikimybės ir perspėjimai.

Numatytasis artefaktas yra paskutinio pilno eksperimento pasirinktas modelis:
`artifacts/selected_{red|white}.joblib`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from winequality.config import ARTIFACTS, FEATURE_NAMES, PHYSICAL_BOUNDS, TYPE_COL
from winequality.postprocess import pack_output, physical_violations


def load_artefact(kind: str, path: Path | None = None):
    path = path or (ARTIFACTS / f"selected_{kind}.joblib")
    if not path.exists():
        raise FileNotFoundError(
            f"Nėra modelio {path}. Pirmiausia paleiskite python -m winequality.run_experiment "
            "--config configs/main.yaml"
        )
    return joblib.load(path)


def _vector(row: dict, median: dict) -> tuple[np.ndarray, list[str]]:
    missing = []
    values = []
    for name in FEATURE_NAMES:
        if name not in row or row[name] is None or (isinstance(row[name], float) and np.isnan(row[name])):
            missing.append(name)
            values.append(float(median[name]))
        else:
            values.append(float(row[name]))
    return np.array([values], dtype=float), missing


def predict_row(bundle: dict, row: dict, y_expert: float | None = None) -> dict:
    present = {name: row.get(name) for name in FEATURE_NAMES}
    bad = physical_violations(present)
    if bad:
        raise ValueError(f"reikšmės už fizinių ribų: {bad} ({PHYSICAL_BOUNDS})")
    median = bundle["train_median"]
    x, missing = _vector(row, median)
    model = bundle["model"]
    score = float(np.asarray(model.predict(x), dtype=float)[0])
    proba = None
    if bundle.get("probabilistic") and hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=float)[0]
    known = {name: float(row[name]) for name in FEATURE_NAMES if name not in missing}
    return pack_output(
        score,
        model_name=bundle["model_name"],
        wine_type=bundle["kind"],
        proba=proba,
        x=known,
        train_min=bundle.get("train_min"),
        train_max=bundle.get("train_max"),
        y_expert=y_expert,
        missing=missing,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prognozuoti vyno kokybę iš 11 laboratorinių matavimų.")
    parser.add_argument("--kind", choices=("red", "white"), required=True)
    parser.add_argument("--json", help="JSON objektas arba kelias iki JSON failo")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    bundle = load_artefact(args.kind, Path(args.model) if args.model else None)
    raw = sys.stdin.read() if args.json is None else (
        Path(args.json).read_text(encoding="utf-8-sig") if Path(args.json).exists() else args.json
    )
    row = json.loads(raw.lstrip("\ufeff"))
    y_expert = row.pop("quality", None)
    row.pop(TYPE_COL, None)
    row.pop("wine_type", None)
    out = predict_row(bundle, row, y_expert=None if y_expert is None else float(y_expert))
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
