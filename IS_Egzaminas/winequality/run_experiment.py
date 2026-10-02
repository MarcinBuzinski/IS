"""Vienas eksperimento įėjimo taškas.

    python -m winequality.run_experiment --config configs/main.yaml

Hiperparametrai parenkami tik mokymo bloke. Užšaldytas testas vertinamas vieną kartą.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from winequality.ablation import ablation_a1, ablation_a4, ablation_a5, ablation_a6, ablation_a7, slim
from winequality.baselines import MeanRegressor, linear_pipeline
from winequality.config import ARTIFACTS, FEATURE_NAMES, ROOT, load_config, output_paths
from winequality.data_io import audit_frame, fetch_raw, sha256_of, write_audit
from winequality.errors import permutation_ranks, shap_mean_abs, top_errors
from winequality.hypotheses import compare_pair, decide_deployment, h1_pass, h2_pass, h3_pass, h4_pass, holm
from winequality.metrics import calibration_bins, evaluate, over_under_counts
from winequality.robustness import cross_type, missingness_curve, noise_curve, random_split_leakage
from winequality.splits import assert_no_group_overlap, frozen_split, load_wine, save_split, xy
from winequality.tune import tune_model


def _git_commit() -> str | None:
    candidates = ["git", r"C:\Program Files\Git\cmd\git.exe"]
    for program in candidates:
        try:
            out = subprocess.check_output(
                [program, "rev-parse", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip()
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def _versions() -> dict:
    import importlib.metadata as metadata

    names = ["numpy", "pandas", "scikit-learn", "xgboost", "catboost", "torch", "optuna", "scipy", "shap"]
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def _ranges(frame: pd.DataFrame) -> tuple[dict, dict, dict]:
    low, high, median = {}, {}, {}
    for name in FEATURE_NAMES:
        column = frame[name].astype(float)
        low[name] = float(column.min())
        high[name] = float(column.max())
        median[name] = float(column.median())
    return low, high, median


def _bundle(kind: str, model_name: str, model, train: pd.DataFrame, params: dict | None, seed: int) -> dict:
    low, high, median = _ranges(train)
    return {
        "kind": kind,
        "model_name": model_name,
        "model": model,
        "probabilistic": model_name in {"xgb_ordinal", "corn"},
        "train_min": low,
        "train_max": high,
        "train_median": median,
        "params": params,
        "seed": seed,
    }


def _predict_pair(model, x):
    score = np.asarray(model.predict(x), dtype=float)
    proba = None
    if type(model).__name__ in {"XGBOrdinalRegressor", "CornRegressor"}:
        proba = np.asarray(model.predict_proba(x), dtype=float)
    return score, proba


def _save_figures(path: Path, rows: list[dict], confusion: np.ndarray, wine: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.mkdir(parents=True, exist_ok=True)
    wines = list(dict.fromkeys(row["wine"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(wines), 1)
    for i, wine_name in enumerate(wines):
        heights = []
        for model in models:
            match = [row["mae"] for row in rows if row["wine"] == wine_name and row["model"] == model]
            heights.append(match[0] if match else 0.0)
        positions = np.arange(len(models)) + i * width
        ax.bar(positions, heights, width=width, color=str(0.25 + 0.35 * i), label=wine_name)
    ax.set_xticks(np.arange(len(models)) + width * (len(wines) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "fig_mae.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(confusion, cmap="gray_r")
    ax.set_xlabel("Prognozė")
    ax.set_ylabel("Eksperto balas")
    ax.set_title(f"Balų painiava, {wine}")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path / "fig_confusion.png", dpi=140)
    plt.close(fig)


def run(cfg: dict) -> dict:
    started = time.perf_counter()
    results_dir, artifact_dir = output_paths(cfg)
    seed = int(cfg["seed"])
    np.random.seed(seed)
    raw_paths = fetch_raw()
    audits = {}
    frames = {}
    for kind in ("red", "white"):
        frame = load_wine(kind)
        frames[kind] = frame
        audits[kind] = audit_frame(frame, kind)
        audits[kind]["sha256"] = sha256_of(raw_paths[kind])
    write_audit(results_dir / "data_audit.json", audits)

    per_wine = {}
    comparison = []
    fitted = {}
    splits = {}
    for kind in cfg["wines"]:
        print(f"=== {kind} ===", flush=True)
        frame = frames[kind]
        train_idx, test_idx = frozen_split(frame, test_size=float(cfg["test_size"]), seed=seed)
        assert_no_group_overlap(frame, train_idx, test_idx)
        save_split(kind, train_idx, test_idx, seed)
        splits[kind] = (train_idx, test_idx)
        x_train, y_train, g_train = xy(frame, train_idx)
        x_test, y_test, g_test = xy(frame, test_idx)
        train_frame = frame.iloc[train_idx]
        models = {}
        hpo = {}
        print("baseline", flush=True)
        mean_model = MeanRegressor().fit(x_train, y_train)
        linear = linear_pipeline().fit(x_train, y_train)
        models["mean"] = mean_model
        models["linear"] = linear
        for name in ("xgb_ordinal", "catboost", "corn", "svr"):
            print(f"HPO {name} trials={cfg['trials'][name]}", flush=True)
            tuned = tune_model(name, x_train, y_train, g_train, cfg, seed)
            models[name] = tuned["model"]
            hpo[name] = {key: value for key, value in tuned.items() if key != "model"}
        predictions = {}
        metrics = {}
        for name, model in models.items():
            score, proba = _predict_pair(model, x_test)
            predictions[name] = score
            row = evaluate(y_test, score, proba)
            row["calibration"] = calibration_bins(y_test, score)
            row["error_direction"] = over_under_counts(y_test, score)
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
                }
            )
            joblib.dump(_bundle(kind, name, model, train_frame, hpo.get(name, {}).get("chosen", {}).get("params"), seed), artifact_dir / f"{name}_{kind}.joblib")
        (results_dir / f"hpo_{kind}.json").write_text(
            json.dumps(hpo, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        per_wine[kind] = {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "group_overlap": 0,
            "metrics": metrics,
            "test_used_in_hpo": False,
        }
        fitted[kind] = {
            "models": models,
            "predictions": predictions,
            "x_train": x_train,
            "y_train": y_train,
            "g_train": g_train,
            "x_test": x_test,
            "y_test": y_test,
            "g_test": g_test,
            "frame": frame,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "params": {name: hpo[name]["chosen"]["params"] for name in hpo},
        }

    hypothesis_rows = {}
    intervals = {}
    for kind, pack in fitted.items():
        m = per_wine[kind]["metrics"]
        pred = pack["predictions"]
        y_test = pack["y_test"]
        groups = pack["g_test"]
        n_boot = int(cfg["bootstrap"])
        pair = {
            "xgb_vs_catboost": compare_pair(y_test, pred["xgb_ordinal"], pred["catboost"], groups, n_boot, seed),
            "xgb_vs_corn": compare_pair(y_test, pred["xgb_ordinal"], pred["corn"], groups, n_boot, seed + 2),
            "xgb_vs_svr": compare_pair(y_test, pred["xgb_ordinal"], pred["svr"], groups, n_boot, seed + 3),
            "svr_vs_linear": compare_pair(y_test, pred["svr"], pred["linear"], groups, n_boot, seed + 4),
        }
        intervals[kind] = pair
        hypothesis_rows[kind] = {
            "H1": h1_pass(m["xgb_ordinal"]["mae"], m["linear"]["mae"], m["xgb_ordinal"]["kappa_w"], m["linear"]["kappa_w"]),
            "H2": h2_pass(pair["xgb_vs_catboost"]["mae"], pair["xgb_vs_catboost"]["kappa_w"]),
            "H3": h3_pass(m["xgb_ordinal"]["mae"], m["corn"]["mae"]),
            "H4": h4_pass(m["svr"]["mae"], m["linear"]["mae"], m["svr"]["kappa_w"], m["xgb_ordinal"]["kappa_w"]),
        }
    h1 = all(row["H1"] for row in hypothesis_rows.values())
    h2 = all(row["H2"] for row in hypothesis_rows.values())
    h3 = all(row["H3"] for row in hypothesis_rows.values())
    h4 = all(row["H4"] for row in hypothesis_rows.values())
    chosen_name = decide_deployment(h1, h2)
    secondary_p = {}
    for kind, pair in intervals.items():
        for label, payload in pair.items():
            secondary_p[f"{kind}:{label}"] = payload["mae"]["p_two_sided"]
    decision = {
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "per_wine": hypothesis_rows,
        "selected_model": chosen_name,
        "rule": "Jei H1 nepasitvirtina, lieka tiesinė regresija. Jei H1 pasitvirtina, bet H2 ne, praktinis modelis yra CatBoost. Kitu atveju lieka XGBOrdinal.",
        "holm_secondary_mae": holm(secondary_p),
        "covers_both_wines": list(cfg["wines"]) == ["red", "white"],
    }

    for kind, pack in fitted.items():
        bundle = _bundle(
            kind,
            chosen_name,
            pack["models"][chosen_name],
            pack["frame"].iloc[pack["train_idx"]],
            pack["params"].get(chosen_name),
            seed,
        )
        joblib.dump(bundle, artifact_dir / f"selected_{kind}.joblib")
        if cfg.get("publish_artifacts", cfg.get("protocol") == "full"):
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, ARTIFACTS / f"selected_{kind}.joblib")

    extras = {"ablation": None, "robustness": None, "errors": None}
    if cfg.get("ablation", True):
        print("abliacijos", flush=True)
        try:
            ablation = {}
            for kind, pack in fitted.items():
                params = pack["params"]
                ablation[kind] = {
                    "A1": slim(ablation_a1(params["xgb_ordinal"], cfg, seed, pack["x_train"], pack["y_train"], pack["x_test"], pack["y_test"])),
                    "A2": {
                        "xgb_ordinal": slim(per_wine[kind]["metrics"]["xgb_ordinal"]),
                        "catboost": slim(per_wine[kind]["metrics"]["catboost"]),
                    },
                    "A3": {
                        "xgb_ordinal": slim(per_wine[kind]["metrics"]["xgb_ordinal"]),
                        "corn": slim(per_wine[kind]["metrics"]["corn"]),
                    },
                    "A4": slim(ablation_a4(params["svr"], cfg, seed, pack["x_train"], pack["y_train"], pack["x_test"], pack["y_test"])),
                    "A5": {
                        "xgb_ordinal": slim(ablation_a5("xgb_ordinal", params["xgb_ordinal"], cfg, seed, pack["x_train"], pack["y_train"], pack["x_test"], pack["y_test"])),
                        "catboost": slim(ablation_a5("catboost", params["catboost"], cfg, seed, pack["x_train"], pack["y_train"], pack["x_test"], pack["y_test"])),
                    },
                    "A7": slim(ablation_a7(params["xgb_ordinal"], cfg, seed, pack["x_train"], pack["y_train"], pack["x_test"], pack["y_test"])),
                }
            if set(cfg["wines"]) >= {"red", "white"}:
                ablation["A6"] = slim(
                    ablation_a6(
                        fitted["red"]["params"]["xgb_ordinal"],
                        cfg,
                        seed,
                        {kind: frames[kind] for kind in ("red", "white")},
                        splits,
                    )
                )
            extras["ablation"] = ablation
        except Exception as exc:
            extras["ablation"] = {"error": str(exc)}

    if cfg.get("robustness", True):
        print("atsparumas", flush=True)
        try:
            robust = {}
            for kind, pack in fitted.items():
                model = pack["models"][chosen_name]
                robust[kind] = {
                    "noise": noise_curve(model, pack["x_train"], pack["x_test"], pack["y_test"], [0.05, 0.10, 0.20], seed),
                    "missingness": missingness_curve(model, pack["x_train"], pack["x_test"], pack["y_test"], [0.05, 0.10], seed),
                    "random_split": random_split_leakage(pack["params"]["xgb_ordinal"], cfg, seed, pack["frame"]),
                }
            if "red" in fitted and "white" in fitted:
                robust["cross_type"] = {
                    "white_model_on_red": cross_type(fitted["white"]["models"][chosen_name], fitted["red"]["x_test"], fitted["red"]["y_test"]),
                    "red_model_on_white": cross_type(fitted["red"]["models"][chosen_name], fitted["white"]["x_test"], fitted["white"]["y_test"]),
                }
            extras["robustness"] = robust
        except Exception as exc:
            extras["robustness"] = {"error": str(exc)}

    if cfg.get("errors", True):
        print("klaidų analizė", flush=True)
        try:
            error_report = {}
            for kind, pack in fitted.items():
                model = pack["models"][chosen_name]
                score = pack["predictions"][chosen_name]
                test_frame = pack["frame"].iloc[pack["test_idx"]][FEATURE_NAMES]
                rows = top_errors(test_frame, pack["y_test"], score)
                pd.DataFrame(rows).to_csv(results_dir / f"error_examples_{kind}.csv", index=False)
                item = {
                    "model": chosen_name,
                    "direction": over_under_counts(pack["y_test"], score),
                    "bias_by_score": per_wine[kind]["metrics"][chosen_name]["bias_by_score"],
                    "permutation_importance": permutation_ranks(model, pack["x_test"], pack["y_test"], seed),
                }
                if cfg.get("shap", True):
                    try:
                        item["shap_mean_abs"] = shap_mean_abs(pack["models"]["xgb_ordinal"], pack["x_test"])
                        item["shap_catboost"] = shap_mean_abs(pack["models"]["catboost"], pack["x_test"])
                    except Exception as exc:
                        item["shap_error"] = str(exc)
                error_report[kind] = item
            extras["errors"] = error_report
        except Exception as exc:
            extras["errors"] = {"error": str(exc)}

    figure_wine = cfg["wines"][0]
    confusion = np.asarray(per_wine[figure_wine]["metrics"][chosen_name]["confusion_score"])
    _save_figures(results_dir / "figures", comparison, confusion, figure_wine)
    pd.DataFrame(comparison).to_csv(results_dir / "comparison.csv", index=False)
    shutil.copyfile(Path(cfg["_config_path"]), results_dir / "config.yaml")

    payload = {
        "protocol": cfg.get("protocol"),
        "seed": seed,
        "seeds_recorded": cfg.get("seeds"),
        "selected_model": chosen_name,
        "hypotheses": decision,
        "intervals": intervals,
        "per_wine": per_wine,
        "extras": extras,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "versions": _versions(),
        "elapsed_sec": time.perf_counter() - started,
        "test_evaluated_once": True,
    }
    (results_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"baigta per {payload['elapsed_sec']:.1f} s, modelis={chosen_name}", flush=True)
    print(results_dir / "metrics.json", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vyno kokybės eksperimentas.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_config_path"] = str(Path(args.config).resolve())
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
