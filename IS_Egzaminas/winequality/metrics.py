"""Metrics used for every method on the same split.

    MAE      = (1/M) Σ |y_i - f_i|
    Acc_T    = (1/M) Σ 1[|y_i - f_i| ≤ T]
    F1_macro = (1/K) Σ 2 P_k R_k / (P_k + R_k)
    κ_w      = 1 - Σ w_kl O_kl / Σ w_kl E_kl ,  w_kl = (k-l)² / (K-1)²
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from winequality.config import CLASS_HIGH_MIN, CLASS_LOW_MAX, SCORE_MAX, SCORE_MIN


def round_score(f: np.ndarray) -> np.ndarray:
    """q̂ = clip(round(f), 3, 9)."""
    return np.clip(np.rint(f), SCORE_MIN, SCORE_MAX).astype(int)


def to_class(score: np.ndarray) -> np.ndarray:
    """low ≤4, medium 5–6, high ≥7."""
    s = np.asarray(score)
    out = np.full(s.shape, 1, dtype=int)  # medium
    out[s <= CLASS_LOW_MAX] = 0
    out[s >= CLASS_HIGH_MIN] = 2
    return out


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def acc_t(y_true: np.ndarray, y_pred: np.ndarray, t: float) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= t))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.clip(np.rint(y_true), SCORE_MIN, SCORE_MAX).astype(int)
    yp = np.clip(np.rint(y_pred), SCORE_MIN, SCORE_MAX).astype(int)
    labels = list(range(SCORE_MIN, SCORE_MAX + 1))
    k = len(labels)
    o = confusion_matrix(yt, yp, labels=labels).astype(float)
    n = o.sum()
    if n == 0:
        return 0.0
    hist_t = o.sum(axis=1)
    hist_p = o.sum(axis=0)
    e = np.outer(hist_t, hist_p) / n
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
    denom = (w * e).sum()
    if denom == 0:
        return 1.0
    return float(1.0 - (w * o).sum() / denom)


def evaluate(y_true: np.ndarray, f: np.ndarray) -> dict:
    """Same metric bundle for every model."""
    y_true = np.asarray(y_true, dtype=float)
    f = np.asarray(f, dtype=float)
    q_hat = round_score(f)
    y_int = np.clip(np.rint(y_true), SCORE_MIN, SCORE_MAX).astype(int)
    cls_true = to_class(y_int)
    cls_pred = to_class(q_hat)
    return {
        "mae": mae(y_true, f),
        "mae_rounded": mae(y_int, q_hat),
        "rmse": float(np.sqrt(np.mean((y_true - f) ** 2))),
        "acc_0.5": acc_t(y_true, f, 0.5),
        "acc_1.0": acc_t(y_true, f, 1.0),
        "kappa_w": quadratic_weighted_kappa(y_true, f),
        "macro_f1": float(
            f1_score(cls_true, cls_pred, average="macro", zero_division=0.0, labels=[0, 1, 2])
        ),
        "confusion_score": confusion_matrix(
            y_int, q_hat, labels=list(range(SCORE_MIN, SCORE_MAX + 1))
        ).tolist(),
        "confusion_class": confusion_matrix(cls_true, cls_pred, labels=[0, 1, 2]).tolist(),
    }
