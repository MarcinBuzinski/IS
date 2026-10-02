"""Abliacija ir atsparumas jau parinktais hiperparametrais.

Skaičiai neparenka modelio iš naujo. Parametrai imami iš
results/palyginimas/hpo_{kind}.json, testas lieka tas pats užšaldytas skaidinys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOT))
from sklearn.inspection import permutation_importance

from winequality.ablation import ablation_a1, slim
from winequality.config import ARTIFACTS, FEATURE_NAMES, ROOT, load_config
from winequality.run_experiment import _bundle
from winequality.metrics import mae
from winequality.robustness import cross_type, missingness_curve, noise_curve, random_split_leakage
from winequality.splits import load_split, load_wine, xy
from winequality.tune import _fit, make_model

OUT = ROOT / "results" / "diagnostika" / "diagnostics.json"


def _chosen(kind: str) -> dict:
    payload = json.loads((ROOT / "results" / "palyginimas" / f"hpo_{kind}.json").read_text(encoding="utf-8"))
    return {name: payload[name]["chosen"]["params"] for name in ("xgb_ordinal", "catboost")}


def _perm(model, x_test, y_test, seed: int) -> list[dict]:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    rows = []
    for index in order:
        rows.append(
            {
                "feature": FEATURE_NAMES[int(index)],
                "mae_increase": float(result.importances_mean[int(index)]),
            }
        )
    return rows


def main() -> None:
    cfg = load_config(ROOT / "configs" / "notebook.yaml")
    published = json.loads((ROOT / "results" / "palyginimas" / "metrics.json").read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    frames = {}
    splits = {}
    fitted = {}
    reproduction = {}
    ablation = {}
    permutation = {}
    noise = {}
    missing = {}
    leaky = {}

    for kind in ("red", "white"):
        print(f"=== {kind} ===", flush=True)
        frame = load_wine(kind)
        train_idx, test_idx = load_split(kind)
        x_train, y_train, g_train = xy(frame, train_idx)
        x_test, y_test, _g_test = xy(frame, test_idx)
        frames[kind] = frame
        splits[kind] = (train_idx, test_idx)
        params = _chosen(kind)
        cat = make_model("catboost", params["catboost"], cfg, seed)
        xgb = make_model("xgb_ordinal", params["xgb_ordinal"], cfg, seed)
        print("fit catboost", flush=True)
        _fit(cat, x_train, y_train, g_train)
        print("fit xgb", flush=True)
        _fit(xgb, x_train, y_train, g_train)
        fitted[kind] = {"catboost": cat, "xgb_ordinal": xgb, "x_test": x_test, "y_test": y_test}
        cat_mae = mae(y_test, cat.predict(x_test))
        xgb_mae = mae(y_test, xgb.predict(x_test))
        stored = published["per_wine"][kind]["metrics"]
        reproduction[kind] = {
            "catboost_mae": cat_mae,
            "catboost_published": stored["catboost"]["mae"],
            "xgb_mae": xgb_mae,
            "xgb_published": stored["xgb_ordinal"]["mae"],
        }
        print("reproduction", reproduction[kind], flush=True)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            _bundle(kind, "catboost", cat, frame.iloc[train_idx], params["catboost"], seed),
            ARTIFACTS / f"selected_{kind}.joblib",
        )
        print("ablation A1", flush=True)
        ablation[kind] = slim(ablation_a1(params["xgb_ordinal"], cfg, seed, x_train, y_train, x_test, y_test))
        print("permutation", flush=True)
        permutation[kind] = _perm(cat, x_test, y_test, seed)
        print("noise and missingness", flush=True)
        noise[kind] = noise_curve(cat, x_train, x_test, y_test, [0.05, 0.10, 0.20], seed)
        missing[kind] = missingness_curve(cat, x_train, x_test, y_test, [0.05, 0.10], seed + 1)
        print("random split", flush=True)
        leaky[kind] = random_split_leakage(params["xgb_ordinal"], cfg, seed, frame)
        leaky[kind]["grouped_xgb_mae"] = stored["xgb_ordinal"]["mae"]

    payload = {
        "seed": seed,
        "protocol": "Diagnostika jau parinktais parametrais. Testas naudojamas vertinimui, ne paieškai.",
        "reproduction": reproduction,
        "ablation_a1": ablation,
        "permutation_catboost": permutation,
        "noise_catboost": noise,
        "missingness_catboost": missing,
        "cross_type": {
            "white_model_on_red": cross_type(fitted["white"]["catboost"], fitted["red"]["x_test"], fitted["red"]["y_test"]),
            "red_model_on_white": cross_type(fitted["red"]["catboost"], fitted["white"]["x_test"], fitted["white"]["y_test"]),
        },
        "random_split": leaky,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
