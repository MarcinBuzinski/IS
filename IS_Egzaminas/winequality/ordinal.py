"""Bendros ordinalinės tikimybių formulės XGBOrdinal ir CORN modeliams.

Frank–Hall slenksčiai:

    g_k(x) = P(y > r_k | x),  k = 1, …, K−1

    P(y = r_1 | x) = 1 − g_1(x)
    P(y = r_k | x) = g_{k−1}(x) − g_k(x)
    P(y = r_K | x) = g_{K−1}(x)

    score_cont = Σ_k r_k P(y = r_k | x)

CORN grandinės taisyklė:

    P(y > r_k | x) = Π_{j=1}^{k} P(y > r_j | y > r_{j−1}, x)
"""

from __future__ import annotations

import numpy as np


def project_monotone(g: np.ndarray) -> np.ndarray:
    """Sutvarko slenksčius taip, kad g_1 ≥ g_2 ≥ … ≥ g_{K−1} ir reikšmės būtų [0, 1]."""
    clipped = np.clip(np.asarray(g, dtype=float), 0.0, 1.0)
    return np.minimum.accumulate(clipped, axis=1)


def chain_rule(conditional: np.ndarray) -> np.ndarray:
    """Iš sąlyginių CORN tikimybių atkuria besąlyginius slenksčius g_k."""
    conditional = np.clip(np.asarray(conditional, dtype=float), 0.0, 1.0)
    return np.cumprod(conditional, axis=1)


def class_proba_from_thresholds(g: np.ndarray) -> np.ndarray:
    """Iš K−1 slenksčių tikimybių sudaro K klasių tikimybes."""
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    n, n_thresholds = g.shape
    proba = np.zeros((n, n_thresholds + 1), dtype=float)
    proba[:, 0] = 1.0 - g[:, 0]
    if n_thresholds > 1:
        proba[:, 1:-1] = g[:, :-1] - g[:, 1:]
    proba[:, -1] = g[:, -1]
    proba = np.clip(proba, 0.0, None)
    totals = proba.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 0.0, 1.0, totals)
    return proba / totals


def expected_score(proba: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """E[y|x] = Σ_k r_k P(y = r_k | x)."""
    return np.asarray(proba, dtype=float) @ np.asarray(ranks, dtype=float)
