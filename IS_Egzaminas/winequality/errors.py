"""Kraštinės klaidos, poslinkis ir požymių svarba."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from winequality.config import FEATURE_NAMES
from winequality.metrics import round_score


def top_errors(frame: pd.DataFrame, y_true: np.ndarray, score: np.ndarray, limit: int = 10) -> list[dict]:
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    gap = np.abs(y_true - score)
    order = np.argsort(-gap)[:limit]
    rows = []
    source = frame.reset_index(drop=True)
    for pos in order:
        item = {name: float(source.iloc[pos][name]) for name in FEATURE_NAMES}
        item["y_true"] = float(y_true[pos])
        item["score_cont"] = float(score[pos])
        item["score_int"] = int(round_score(np.array([score[pos]]))[0])
        item["abs_error"] = float(gap[pos])
        rows.append(item)
    return rows


def permutation_ranks(model, x, y, seed: int, n_repeats: int = 5, max_rows: int = 400) -> dict:
    take = min(len(x), max_rows)
    result = permutation_importance(
        model,
        np.asarray(x)[:take],
        np.asarray(y)[:take],
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(-result.importances_mean)
    return {
        FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else str(i): {
            "mean": float(result.importances_mean[i]),
            "std": float(result.importances_std[i]),
        }
        for i in order
    }


def _shap_matrix(values, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim == 2 and values.shape[1] == n_features + 1:
        values = values[:, :-1]
    return values


def shap_mean_abs(model, x, max_rows: int = 200) -> dict | None:
    """Vidutinė |SHAP| CatBoost arba XGBOrdinal medžiams. Klaida grąžinama kvietėjui."""
    import shap

    x = np.asarray(x, dtype=float)[:max_rows]
    name = type(model).__name__
    if name == "Pipeline" and "cat" in getattr(model, "named_steps", {}):
        transformed = model.named_steps["prep"].transform(x)
        explainer = shap.TreeExplainer(model.named_steps["cat"])
        values = _shap_matrix(explainer.shap_values(transformed), transformed.shape[1])
        mean_abs = np.mean(np.abs(values), axis=0)
    elif name == "XGBOrdinalRegressor":
        transformed = model.prep_.transform(x)
        collected = []
        for member in model.models_:
            if not hasattr(member, "get_booster"):
                continue
            explainer = shap.TreeExplainer(member)
            values = _shap_matrix(explainer.shap_values(transformed), transformed.shape[1])
            collected.append(np.mean(np.abs(values), axis=0))
        if not collected:
            return None
        mean_abs = np.mean(collected, axis=0)
    else:
        return None
    names = FEATURE_NAMES if len(mean_abs) == len(FEATURE_NAMES) else [str(i) for i in range(len(mean_abs))]
    return {names[i]: float(mean_abs[i]) for i in np.argsort(-mean_abs)}
