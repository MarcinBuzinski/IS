"""Triukšmo, trūkstamumo, atsitiktinio skaidymo ir tipų poslinkio bandymai.

Šie skaičiai yra diagnostika. Jie nekeičia H1–H4 sprendimo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from winequality.config import FEATURE_NAMES, TARGET
from winequality.metrics import evaluate
from winequality.splits import stratify_label
from winequality.tune import _fit, make_model


def _brief(y, pred) -> dict:
    row = evaluate(y, pred)
    return {"mae": row["mae"], "kappa_w": row["kappa_w"]}


def noise_curve(model, x_train, x_test, y_test, levels: list[float], seed: int) -> dict:
    std = np.nanstd(np.asarray(x_train, dtype=float), axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    rng = np.random.default_rng(seed)
    out = {"0": _brief(y_test, model.predict(x_test))}
    for level in levels:
        noisy = np.asarray(x_test, dtype=float) + rng.normal(0.0, level, size=np.shape(x_test)) * std
        out[str(level)] = _brief(y_test, model.predict(noisy))
    return out


def missingness_curve(model, x_train, x_test, y_test, rates: list[float], seed: int) -> dict:
    median = np.nanmedian(np.asarray(x_train, dtype=float), axis=0)
    rng = np.random.default_rng(seed)
    out = {}
    flat_n = int(np.size(x_test))
    for rate in rates:
        corrupted = np.array(x_test, dtype=float, copy=True)
        mask = rng.random(corrupted.shape) < rate
        corrupted[mask] = np.take(median, np.where(mask)[1])
        out[str(rate)] = {
            **_brief(y_test, model.predict(corrupted)),
            "masked_cells": int(mask.sum()),
            "n_cells": flat_n,
        }
    return out


def random_split_leakage(params: dict, cfg: dict, seed: int, frame: pd.DataFrame) -> dict:
    """Parodo dublikatų nutekėjimą, kai skaidymas nepaiso grupių."""
    idx = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_label(frame[TARGET]),
    )
    train_groups = set(frame.iloc[train_idx]["group"])
    test_groups = set(frame.iloc[test_idx]["group"])
    overlap = train_groups & test_groups
    columns = FEATURE_NAMES
    model = make_model("xgb_ordinal", params, cfg, seed)
    x_train = frame.iloc[train_idx][columns].to_numpy(float)
    y_train = frame.iloc[train_idx][TARGET].to_numpy(float)
    x_test = frame.iloc[test_idx][columns].to_numpy(float)
    y_test = frame.iloc[test_idx][TARGET].to_numpy(float)
    _fit(model, x_train, y_train, frame.iloc[train_idx]["group"].to_numpy())
    return {
        "note": "Diagnostika, nenaudojama modelio pasirinkimui.",
        "overlapping_groups": len(overlap),
        "metrics": _brief(y_test, model.predict(x_test)),
    }


def cross_type(model, x_other, y_other) -> dict:
    row = _brief(y_other, model.predict(x_other))
    row["note"] = "Kito vyno tipo testo aibė; tai ne veikimo režimas."
    return row
