"""Ablation A1 (linear vs RBF kernel) and one-at-a-time feature sensitivity."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from winequality.config import FEATURE_NAMES
from winequality.metrics import evaluate
from winequality.tune import tune_svr


def ablation_kernel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    rbf_metrics: dict | None = None,
    rbf_tune: dict | None = None,
) -> dict:
    """A1: same split, linear kernel SVR vs RBF SVR.

    Pass the already-evaluated RBF metrics/tune so the main model is not fit twice.
    """
    lin_model, lin_info = tune_svr(X_train, y_train, groups_train, kernel="linear", seed=seed)
    out = {
        "linear": {"metrics": evaluate(y_test, lin_model.predict(X_test)), "tune": lin_info},
    }
    if rbf_metrics is not None and rbf_tune is not None:
        out["rbf"] = {"metrics": rbf_metrics, "tune": rbf_tune}
    else:
        model, info = tune_svr(X_train, y_train, groups_train, kernel="rbf", seed=seed)
        out["rbf"] = {"metrics": evaluate(y_test, model.predict(X_test)), "tune": info}
    return out


def sensitivity_curves(
    pipeline: Pipeline,
    X_train: np.ndarray,
    features: tuple[str, ...] = ("alcohol", "volatile acidity"),
    grid: np.ndarray | None = None,
) -> dict[str, dict]:
    """Vary one feature from −1 to +1 training sd; others at the training mean.

    Domain expectation: alcohol ↑ → score ↑; volatile acidity ↑ → score ↓.
    """
    if grid is None:
        grid = np.linspace(-1.0, 1.0, 9)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    curves: dict[str, dict] = {}
    for name in features:
        j = FEATURE_NAMES.index(name)
        xs = []
        fs = []
        for t in grid:
            row = mean.copy()
            row[j] = mean[j] + t * std[j]
            xs.append(float(row[j]))
            fs.append(float(pipeline.predict(row.reshape(1, -1))[0]))
        curves[name] = {
            "sd_units": grid.tolist(),
            "feature_value": xs,
            "f": fs,
            "delta_f": float(fs[-1] - fs[0]),
        }
    return curves
