"""One-command exam experiment: same split, same metrics, train-only HPO.

    python -m winequality.run_experiment

Writes results/metrics.json, comparison tables, figures, and joblib artefacts.
The frozen test set is used once per model after inner CV on the 80 % train block.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from winequality.ablation import ablation_kernel, sensitivity_curves
from winequality.config import (
    ARTIFACTS,
    DEFAULT_SEED,
    FEATURE_NAMES,
    FIGURES,
    RESULTS,
    SPLIT_SEED,
)
from winequality.errors import mae_by_true_score, residual_table
from winequality.metrics import evaluate
from winequality.robustness import leakage_comparison, noise_stress
from winequality.splits import frozen_split, load_wine, save_split_indices, xy
from winequality.svr_model import formula_predict
from winequality.tune import fit_baselines, tune_mlp, tune_rbf, tune_rf, tune_svr


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _bar_mae(table: dict[str, dict], path: Path, title: str) -> None:
    names = list(table.keys())
    vals = [table[n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    palette = {
        "mean": "#888888",
        "linear": "#888888",
        "svr": "#8b1e3f",
        "mlp": "#2d6a4f",
        "rbf": "#c9a227",
        "rf": "#4a6fa5",
    }
    colors = [palette.get(n, "#555555") for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("MAE (test)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _confusion(mat: list[list[int]], path: Path, labels: list[str], title: str) -> None:
    arr = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(int(arr[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _sensitivity_plot(curves: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for name, c in curves.items():
        ax.plot(c["sd_units"], c["f"], marker="o", label=name)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("feature change (training SD units)")
    ax.set_ylabel("predicted score f(x)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_kind(kind: str, seed: int = DEFAULT_SEED) -> dict:
    df = load_wine(kind)
    train_idx, test_idx = frozen_split(df, seed=SPLIT_SEED)
    save_split_indices(kind, train_idx, test_idx)
    X_tr, y_tr = xy(df, train_idx)
    X_te, y_te = xy(df, test_idx)
    groups_tr = df.iloc[train_idx]["group"].to_numpy()
    df_te = df.iloc[test_idx]

    baselines = fit_baselines(X_tr, y_tr)
    svr, svr_info = tune_svr(X_tr, y_tr, groups_tr, kernel="rbf", seed=seed)
    mlp, mlp_info = tune_mlp(X_tr, y_tr, groups_tr, seed=seed)
    rbf, rbf_info = tune_rbf(X_tr, y_tr, groups_tr, seed=seed)
    rf, rf_info = tune_rf(X_tr, y_tr, groups_tr, seed=seed)

    pred = {
        "mean": baselines["mean"].predict(X_te),
        "linear": baselines["linear"].predict(X_te),
        "svr": formula_predict(svr, X_te),
        "mlp": mlp.predict(X_te),
        "rbf": rbf.predict(X_te),
        "rf": rf.predict(X_te),
    }
    # Guard: formula path must match sklearn.predict for the main model.
    sk_pred = svr.predict(X_te)
    max_abs = float(np.max(np.abs(pred["svr"] - sk_pred)))
    if max_abs >= 1e-6:
        raise RuntimeError(f"SVR formula path disagrees with sklearn: max|Δ|={max_abs}")

    metrics = {name: evaluate(y_te, p) for name, p in pred.items()}
    mae_lin = metrics["linear"]["mae"]
    mae_svr = metrics["svr"]["mae"]
    h1 = {
        "threshold_linear_mae_times_0.95": 0.95 * mae_lin,
        "svr_mae": mae_svr,
        "relative_improvement": (mae_lin - mae_svr) / mae_lin if mae_lin else 0.0,
        "h1_holds": mae_svr <= 0.95 * mae_lin,
    }
    h2 = {
        "mae_svr_minus_mae_rf": mae_svr - metrics["rf"]["mae"],
        "non_inferiority_holds": (mae_svr - metrics["rf"]["mae"]) <= 0.02,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": svr,
        "kind": kind,
        "train_min": {n: float(X_tr[:, i].min()) for i, n in enumerate(FEATURE_NAMES)},
        "train_max": {n: float(X_tr[:, i].max()) for i, n in enumerate(FEATURE_NAMES)},
        "seed": seed,
        "split_seed": SPLIT_SEED,
    }
    joblib.dump(bundle, ARTIFACTS / f"svr_{kind}.joblib")
    joblib.dump(mlp, ARTIFACTS / f"mlp_{kind}.joblib")
    joblib.dump(rbf, ARTIFACTS / f"rbf_{kind}.joblib")
    joblib.dump(rf, ARTIFACTS / f"rf_{kind}.joblib")
    joblib.dump(baselines["linear"], ARTIFACTS / f"linear_{kind}.joblib")

    residuals = residual_table(df_te, pred["svr"])
    residuals.to_csv(RESULTS / f"error_examples_{kind}.csv", index=False)

    sens = sensitivity_curves(svr, X_tr)
    noise = noise_stress(svr, X_te, y_te, seed=seed)
    leak = leakage_comparison(
        df,
        train_idx,
        test_idx,
        seed,
        group_metrics=metrics["svr"],
        group_tune=svr_info,
    )
    a1 = ablation_kernel(
        X_tr,
        y_tr,
        groups_tr,
        X_te,
        y_te,
        seed,
        rbf_metrics=metrics["svr"],
        rbf_tune=svr_info,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    _bar_mae(metrics, FIGURES / f"mae_{kind}.png", f"{kind} wine — test MAE")
    _confusion(
        metrics["svr"]["confusion_class"],
        FIGURES / f"confusion_class_{kind}.png",
        ["low", "medium", "high"],
        f"{kind} SVR — 3-class confusion",
    )
    _sensitivity_plot(sens, FIGURES / f"sensitivity_{kind}.png", f"{kind} SVR sensitivity")

    return {
        "kind": kind,
        "n": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_groups_train": int(df.iloc[train_idx]["group"].nunique()),
        "n_groups_test": int(df.iloc[test_idx]["group"].nunique()),
        "formula_vs_sklearn_max_abs": max_abs,
        "metrics": metrics,
        "h1": h1,
        "h2": h2,
        "svr_tune": svr_info,
        "mlp_tune": mlp_info,
        "rbf_tune": rbf_info,
        "rf_tune": rf_info,
        "mae_by_true_score": mae_by_true_score(y_te, pred["svr"]),
        "error_examples": residuals.head(5).to_dict(orient="records"),
        "sensitivity": sens,
        "robustness_noise": noise,
        "robustness_leakage": leak,
        "ablation_a1_kernel": a1,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = {
        "split_seed": SPLIT_SEED,
        "model_seed": DEFAULT_SEED,
        "note": "Hyperparameters selected by GroupKFold MAE on the training block only.",
        "wines": {},
    }
    for kind in ("red", "white"):
        print(f"=== {kind} ===", flush=True)
        report["wines"][kind] = run_kind(kind)
        print(json.dumps(_jsonable(report["wines"][kind]["metrics"]), indent=2), flush=True)

    out = RESULTS / "metrics.json"
    out.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    rows = []
    for kind, block in report["wines"].items():
        for model, m in block["metrics"].items():
            rows.append(
                {
                    "wine": kind,
                    "model": model,
                    "mae": m["mae"],
                    "mae_rounded": m["mae_rounded"],
                    "acc_0.5": m["acc_0.5"],
                    "acc_1.0": m["acc_1.0"],
                    "kappa_w": m["kappa_w"],
                    "macro_f1": m["macro_f1"],
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / "comparison.csv", index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
