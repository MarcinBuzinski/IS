"""Pagerintų modelių eksperimentas tame pačiame užšaldytame skaidinyje.

    python -m winequality.run_improved --config configs/pagerinta.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from winequality.baselines import MeanRegressor
from winequality.config import FEATURE_NAMES, ROOT, load_config, output_paths
from winequality.errors import top_errors
from winequality.improve import AsymmetricCatBoost, SharedOrdinalBoost, design_matrix
from winequality.metrics import evaluate, mae, over_under_counts, quadratic_weighted_kappa
from winequality.splits import frozen_split, load_split, load_wine, save_split, stratify_label, xy

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_ORDER = ["mean", "linear", "cat_asym", "cat_ordinal", "lgbm_ordinal"]


def _folds(x, y, groups, n_splits: int, seed: int):
    stratum = stratify_label(y)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(x, stratum, groups))
    except ValueError:
        return list(GroupKFold(n_splits=n_splits).split(x, y, groups))


def _fit(model, x, y, groups):
    model.fit(x, y, groups=groups)
    return model


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 180, 420),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0, log=True),
    }


def _make(name: str, params: dict, cfg: dict, seed: int):
    common = dict(params)
    common.update(seed=seed, thread_count=int(cfg.get("n_jobs", 1)), tail_weight=float(cfg.get("tail_weight", 2.0)))
    if name == "cat_asym":
        return AsymmetricCatBoost(alpha=float(cfg.get("asym_alpha", 2.0)), **common)
    if name == "cat_ordinal":
        return SharedOrdinalBoost(backend="catboost", **common)
    if name == "lgbm_ordinal":
        return SharedOrdinalBoost(backend="lightgbm", **common)
    raise KeyError(name)


def _tune(name: str, x, y, groups, cfg: dict, seed: int) -> dict:
    n_trials = int(cfg["trials"][name])
    n_splits = int(cfg["inner_splits"])

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        scores = []
        kappas = []
        for train_idx, valid_idx in _folds(x, y, groups, n_splits, seed):
            model = _make(name, params, cfg, seed)
            _fit(model, x[train_idx], y[train_idx], groups[train_idx])
            pred = model.predict(x[valid_idx])
            scores.append(mae(y[valid_idx], pred))
            kappas.append(quadratic_weighted_kappa(y[valid_idx], pred))
        trial.set_user_attr("kappa", float(np.mean(kappas)))
        trial.set_user_attr("params", params)
        return float(np.mean(scores))

    study_seed = seed + {"cat_asym": 11, "cat_ordinal": 29, "lgbm_ordinal": 47}[name]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=study_seed))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None or trial.value >= 10:
            continue
        records.append(
            {
                "number": trial.number,
                "mae": float(trial.value),
                "kappa": float(trial.user_attrs.get("kappa", 0.0)),
                "params": trial.user_attrs.get("params", {}),
            }
        )
    if not records:
        raise RuntimeError(f"{name} neturi sėkmingų bandymų")
    best_mae = min(row["mae"] for row in records)
    pool = [row for row in records if row["mae"] <= best_mae + 0.01]
    pool.sort(key=lambda row: (-row["kappa"], row["number"]))
    chosen = pool[0]
    model = _make(name, chosen["params"], cfg, seed)
    _fit(model, x, y, groups)
    return {"model": model, "chosen": chosen, "trials": records, "test_used": False}


def _linear(x, y):
    pipe = Pipeline([("scale", StandardScaler()), ("lin", LinearRegression())])
    pipe.fit(design_matrix(x), y)

    class _Wrapped:
        def predict(self, X):
            return pipe.predict(design_matrix(X))

    return _Wrapped()


def run(cfg: dict) -> dict:
    started = time.perf_counter()
    results_dir, _artifacts = output_paths(cfg)
    seed = int(cfg["seed"])
    comparison = []
    per_wine = {}
    for kind in cfg["wines"]:
        print(f"=== {kind} ===", flush=True)
        frame = load_wine(kind)
        try:
            train_idx, test_idx = load_split(kind)
        except FileNotFoundError:
            train_idx, test_idx = frozen_split(frame, seed=seed)
            save_split(kind, train_idx, test_idx, seed)
        x_train, y_train, g_train = xy(frame, train_idx)
        x_test, y_test, _g_test = xy(frame, test_idx)
        models = {
            "mean": MeanRegressor().fit(x_train, y_train),
            "linear": _linear(x_train, y_train),
        }
        hpo = {}
        for name in ("cat_asym", "cat_ordinal", "lgbm_ordinal"):
            print(f"HPO {name} trials={cfg['trials'][name]}", flush=True)
            tuned = _tune(name, x_train, y_train, g_train, cfg, seed)
            models[name] = tuned["model"]
            hpo[name] = {key: value for key, value in tuned.items() if key != "model"}
        metrics = {}
        for name in MODEL_ORDER:
            pred = np.asarray(models[name].predict(x_test), dtype=float)
            proba = models[name].predict_proba(x_test) if hasattr(models[name], "predict_proba") else None
            row = evaluate(y_test, pred, proba)
            row["error_direction"] = over_under_counts(y_test, pred)
            metrics[name] = row
            comparison.append(
                {
                    "wine": kind,
                    "model": name,
                    "mae": row["mae"],
                    "mae_rounded": row["mae_rounded"],
                    "rmse": row["rmse"],
                    "acc_0.5": row["acc_0.5"],
                    "acc_1.0": row["acc_1.0"],
                    "kappa_w": row["kappa_w"],
                    "macro_f1": row["macro_f1"],
                    "log_loss": row.get("log_loss"),
                    "brier": row.get("brier"),
                    "critical_abs_ge_2": row["error_direction"]["critical_abs_ge_2"],
                }
            )
        best = min(MODEL_ORDER, key=lambda item: metrics[item]["mae"])
        pred = np.asarray(models[best].predict(x_test), dtype=float)
        pd.DataFrame(top_errors(frame.iloc[test_idx][FEATURE_NAMES], y_test, pred)).to_csv(
            results_dir / f"error_examples_{kind}.csv", index=False
        )
        (results_dir / f"hpo_{kind}.json").write_text(json.dumps(hpo, indent=2), encoding="utf-8")
        per_wine[kind] = {"n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "metrics": metrics, "best_mae": best}
        print(kind, {name: round(metrics[name]["mae"], 4) for name in MODEL_ORDER}, "best", best, flush=True)

    previous = ROOT / "results" / "palyginimas" / "comparison.csv"
    by_model = {}
    for row in comparison:
        by_model.setdefault(row["model"], []).append(row["mae"])
    selected = min(by_model, key=lambda name: float(np.mean(by_model[name])))
    payload = {
        "protocol": cfg.get("protocol"),
        "seed": seed,
        "selected_model": selected,
        "per_wine": per_wine,
        "changes": [
            "log1p asimetriškiems požymiams",
            "SO2 santykis, rūgščių santykis, alkoholis/tankis",
            "dublikatų svoriai ir kraštinių balų svoriai",
            "asimetrinė MAE žemiems balams",
            "bendras Frank–Hall CatBoost ir LightGBM",
            "monotoniniai apribojimai",
        ],
        "previous_comparison": str(previous) if previous.exists() else None,
        "elapsed_sec": time.perf_counter() - started,
    }
    pd.DataFrame(comparison).to_csv(results_dir / "comparison.csv", index=False)
    (results_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"baigta per {payload['elapsed_sec']:.1f} s", flush=True)
    print(results_dir / "metrics.json", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pagerintų vyno modelių eksperimentas.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg.setdefault("tail_weight", 2.0)
    cfg.setdefault("asym_alpha", 2.0)
    cfg["_config_path"] = str(Path(args.config).resolve())
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
