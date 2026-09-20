"""Post-processing: integer score, 3-class label, confidence, flags.

    q̂    = min(9, max(3, round(f(z))))
    class = low if q̂≤4; medium if 5≤q̂≤6; high if q̂≥7
    conf  = 1 − 2 |f(z) − q̂|          (1 at a whole number, 0 on a .5 boundary)
"""

from __future__ import annotations

import numpy as np

from winequality.config import (
    CLASS_HIGH_MIN,
    CLASS_LOW_MAX,
    CLASS_NAMES,
    FEATURE_NAMES,
    PHYSICAL_BOUNDS,
    SCORE_MAX,
    SCORE_MIN,
)
from winequality.metrics import round_score, to_class


def confidence(f: np.ndarray | float) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    q = round_score(f).astype(float)
    return 1.0 - 2.0 * np.abs(f - q)


def class_name(score_int: int) -> str:
    return CLASS_NAMES[int(to_class(np.array([score_int]))[0])]


def physical_violations(x: dict[str, float]) -> list[str]:
    bad = []
    for name, (lo, hi) in PHYSICAL_BOUNDS.items():
        if name not in x:
            continue
        v = x[name]
        if v is None or (isinstance(v, float) and np.isnan(v)) or v < lo or v > hi:
            bad.append(name)
    return bad


def flags_for(
    x: dict[str, float],
    f: float,
    score_int: int,
    train_min: dict[str, float] | None = None,
    train_max: dict[str, float] | None = None,
    y_expert: float | None = None,
) -> list[str]:
    flags: list[str] = []
    if train_min is not None and train_max is not None:
        for name in FEATURE_NAMES:
            if name in x and (x[name] < train_min[name] or x[name] > train_max[name]):
                flags.append("extrapolation")
                break
    if y_expert is not None and abs(f - y_expert) > 1.0:
        flags.append("disagreement")
    if score_int <= CLASS_LOW_MAX or score_int >= 8:
        flags.append("extreme")
    return flags


def pack_output(
    f: float,
    x: dict[str, float] | None = None,
    train_min: dict[str, float] | None = None,
    train_max: dict[str, float] | None = None,
    y_expert: float | None = None,
) -> dict:
    score_int = int(round_score(np.array([f]))[0])
    return {
        "score_cont": float(f),
        "score_int": score_int,
        "class": class_name(score_int),
        "confidence": float(confidence(np.array([f]))[0]),
        "flags": flags_for(x or {}, f, score_int, train_min, train_max, y_expert),
    }
