"""Išvesties sutartis: balas, klasė, neapibrėžtumas ir perspėjimai."""

from __future__ import annotations

import numpy as np

from winequality.config import CLASS_NAMES, FEATURE_NAMES, PHYSICAL_BOUNDS, RANKS
from winequality.metrics import round_score, to_class


def class_name(score_int: int) -> str:
    return CLASS_NAMES[int(to_class(np.array([score_int]))[0])]


def physical_violations(values: dict) -> list[str]:
    bad = []
    for name, (low, high) in PHYSICAL_BOUNDS.items():
        if name not in values or values[name] is None:
            continue
        value = float(values[name])
        if np.isnan(value) or value < low or value > high:
            bad.append(name)
    return bad


def entropy(proba: np.ndarray) -> float:
    p = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    return float(-np.sum(p * np.log(p)))


def pack_output(
    score: float,
    *,
    model_name: str,
    wine_type: str,
    proba: np.ndarray | None = None,
    x: dict | None = None,
    train_min: dict | None = None,
    train_max: dict | None = None,
    y_expert: float | None = None,
    missing: list[str] | None = None,
) -> dict:
    score_int = int(round_score(np.array([score]))[0])
    distance = abs(float(score) - score_int)
    confidence = float(1.0 - 2.0 * distance)
    flags: list[str] = []
    if missing:
        flags.append("missing_input")
    if x and train_min and train_max:
        for name in FEATURE_NAMES:
            if name in x and x[name] is not None and (float(x[name]) < train_min[name] or float(x[name]) > train_max[name]):
                flags.append("extrapolation")
                break
    uncertainty = confidence
    proba_map = None
    if proba is not None:
        vector = np.asarray(proba, dtype=float).ravel()
        proba_map = {str(rank): float(vector[i]) for i, rank in enumerate(RANKS)}
        uncertainty = entropy(vector)
        if float(vector.max()) < 0.45:
            flags.append("low_confidence")
    elif confidence < 0.3:
        flags.append("low_confidence")
    if y_expert is not None and abs(float(score) - float(y_expert)) > 1.0:
        flags.append("disagreement")
    if score_int <= 4 or score_int >= 8:
        flags.append("extreme")
    return {
        "model": model_name,
        "wine_type": wine_type,
        "score_cont": float(score),
        "score_int": score_int,
        "class": class_name(score_int),
        "proba": proba_map,
        "uncertainty": float(uncertainty),
        "confidence": confidence,
        "flags": flags,
    }
