"""Iš anksto registruotos H1–H4 taisyklės ir grupinis bootstrap.

H1 sprendžiama abiem vyno tipams. H2 reikalauja, kad 95 % intervalo
skirtumas neapimtų ±0,01 zonos ir kryptis būtų XGBOrdinal naudai.
"""

from __future__ import annotations

import numpy as np

from winequality.metrics import mae, quadratic_weighted_kappa


def decide_deployment(h1: bool, h2: bool) -> str:
    if not h1:
        return "linear"
    if not h2:
        return "catboost"
    return "xgb_ordinal"


def h1_pass(mae_model: float, mae_base: float, kappa_model: float, kappa_base: float) -> bool:
    return mae_model <= 0.95 * mae_base and kappa_model >= kappa_base + 0.05


def h2_pass(mae_interval: dict, kappa_interval: dict) -> bool:
    """mae_interval yra XGBOrdinal − CatBoost (mažesnė MAE geriau)."""
    mae_better = mae_interval["hi"] < -0.01
    kappa_better = kappa_interval["lo"] > 0.01
    return bool(mae_better or kappa_better)


def h3_pass(mae_xgb: float, mae_corn: float) -> bool:
    return mae_xgb < mae_corn


def h4_pass(mae_svr: float, mae_linear: float, kappa_svr: float, kappa_xgb: float) -> bool:
    return mae_svr < mae_linear and kappa_svr < kappa_xgb


def _group_buckets(groups: np.ndarray) -> list[np.ndarray]:
    groups = np.asarray(groups)
    _uniq, inverse = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inverse == i) for i in range(len(_uniq))]


def bootstrap_difference(y, pred_a, pred_b, groups, stat, n_boot: int, seed: int) -> dict:
    """Skirtumo stat(a)−stat(b) grupinis bootstrap. Grupė yra persėmimo vienetas."""
    buckets = _group_buckets(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    y = np.asarray(y)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n_groups = len(buckets)
    for i in range(n_boot):
        draw = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([buckets[j] for j in draw])
        diffs[i] = stat(y[idx], pred_a[idx]) - stat(y[idx], pred_b[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_neg = float(np.mean(diffs <= 0.0))
    p_pos = float(np.mean(diffs >= 0.0))
    return {
        "lo": float(lo),
        "hi": float(hi),
        "mean": float(diffs.mean()),
        "p_two_sided": float(min(1.0, 2.0 * min(p_neg, p_pos))),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(pvalue)))
        adjusted[name] = running
    return adjusted


def compare_pair(y, pred_a, pred_b, groups, n_boot: int, seed: int) -> dict:
    return {
        "mae": bootstrap_difference(y, pred_a, pred_b, groups, mae, n_boot, seed),
        "kappa_w": bootstrap_difference(
            y, pred_a, pred_b, groups, quadratic_weighted_kappa, n_boot, seed + 1
        ),
    }
