"""Metrikos, vienodos visiems metodams tame pačiame skaidinyje.

    MAE = (1/M) Σ |y_i − ŷ_i|
    κw  = 1 − (Σ w_kl O_kl) / (Σ w_kl E_kl),   w_kl = (k−l)² / (K−1)²
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from winequality.config import CLASS_NAMES, RANKS, SCORE_MAX, SCORE_MIN


def round_score(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, dtype=float)), SCORE_MIN, SCORE_MAX).astype(int)


def to_class(score: np.ndarray) -> np.ndarray:
    """žema ≤4, vidutinė 5–6, aukšta ≥7."""
    scores = np.asarray(score)
    out = np.ones(scores.shape, dtype=int)
    out[scores <= 4] = 0
    out[scores >= 7] = 2
    return out


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(error**2)))


def acc_t(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tolerance))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_i = round_score(y_true)
    y_pred_i = round_score(y_pred)
    labels = list(RANKS)
    observed = confusion_matrix(y_true_i, y_pred_i, labels=labels).astype(float)
    total = observed.sum()
    if total == 0:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    size = len(labels)
    weights = np.array(
        [[((i - j) ** 2) / ((size - 1) ** 2) for j in range(size)] for i in range(size)],
        dtype=float,
    )
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 1.0
    return float(1.0 - (weights * observed).sum() / denominator)


def probability_scores(y_true: np.ndarray, proba: np.ndarray, ranks: np.ndarray | None = None) -> dict:
    ranks = np.asarray(list(RANKS) if ranks is None else ranks, dtype=int)
    y_idx = round_score(y_true) - int(ranks[0])
    if int(ranks[0]) != SCORE_MIN or len(ranks) != proba.shape[1]:
        index = {int(rank): i for i, rank in enumerate(ranks)}
        y_idx = np.array([index[int(value)] for value in round_score(y_true)])
    picked = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-15, 1.0)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_idx)), y_idx] = 1.0
    return {
        "log_loss": float(-np.mean(np.log(picked))),
        "brier": float(np.mean(np.sum((proba - one_hot) ** 2, axis=1))),
    }


def evaluate(y_true: np.ndarray, score: np.ndarray, proba: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    rounded = round_score(score)
    y_int = round_score(y_true)
    true_class = to_class(y_int)
    pred_class = to_class(rounded)
    labels_3 = [0, 1, 2]
    result = {
        "mae": mae(y_true, score),
        "mae_rounded": mae(y_int, rounded),
        "rmse": rmse(y_true, score),
        "acc_0.5": acc_t(y_true, score, 0.5),
        "acc_1.0": acc_t(y_true, score, 1.0),
        "kappa_w": quadratic_weighted_kappa(y_true, score),
        "macro_f1": float(f1_score(true_class, pred_class, average="macro", labels=labels_3, zero_division=0.0)),
        "precision_class": precision_score(
            true_class, pred_class, average=None, labels=labels_3, zero_division=0.0
        ).tolist(),
        "recall_class": recall_score(
            true_class, pred_class, average=None, labels=labels_3, zero_division=0.0
        ).tolist(),
        "class_names": list(CLASS_NAMES),
        "confusion_score": confusion_matrix(y_int, rounded, labels=list(RANKS)).tolist(),
        "confusion_class": confusion_matrix(true_class, pred_class, labels=labels_3).tolist(),
        "bias_by_score": bias_by_score(y_true, score),
    }
    if proba is not None:
        result.update(probability_scores(y_true, proba))
    return result


def bias_by_score(y_true: np.ndarray, score: np.ndarray) -> dict:
    y_int = round_score(y_true)
    pred = np.asarray(score, dtype=float)
    out = {}
    for rank in RANKS:
        mask = y_int == rank
        if not np.any(mask):
            continue
        error = pred[mask] - y_int[mask]
        out[str(rank)] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    return out


def calibration_bins(y_true: np.ndarray, score: np.ndarray, width: float = 0.5) -> list[dict]:
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    edges = np.arange(SCORE_MIN - 0.25, SCORE_MAX + 0.75, width)
    bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (score >= left) & (score < right)
        if not np.any(mask):
            continue
        bins.append(
            {
                "left": float(left),
                "right": float(right),
                "n": int(mask.sum()),
                "mean_pred": float(score[mask].mean()),
                "mean_true": float(y_true[mask].mean()),
            }
        )
    return bins


def over_under_counts(y_true: np.ndarray, score: np.ndarray) -> dict:
    gap = round_score(score) - round_score(y_true)
    return {
        "overestimate": int(np.sum(gap > 0)),
        "underestimate": int(np.sum(gap < 0)),
        "exact": int(np.sum(gap == 0)),
        "critical_abs_ge_2": int(np.sum(np.abs(gap) >= 2)),
    }
