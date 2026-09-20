"""Duplicate groups, stratification helper, frozen 80/20 split.

Identical 11-feature rows share a group id and never cross the train/test cut
(Kapoor & Narayanan 2023 leakage). Stratification merges rare scores 3 with 4
and 9 with 8 so StratifiedGroupKFold has enough members per stratum.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from winequality.config import (
    DATA_SPLITS,
    FEATURE_NAMES,
    SPLIT_SEED,
    TARGET,
    TYPE_COL,
)
from winequality.data_io import fetch_raw


def feature_group_id(row: pd.Series) -> str:
    payload = "|".join(f"{row[c]:.10g}" for c in FEATURE_NAMES)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def stratify_label(quality: pd.Series) -> pd.Series:
    y = quality.astype(int)
    return y.replace({3: 4, 9: 8})


def load_wine(kind: str) -> pd.DataFrame:
    """Load one wine type, attach type and duplicate-group ids."""
    paths = fetch_raw()
    if kind not in paths:
        raise ValueError(f"kind must be 'red' or 'white', got {kind!r}")
    df = pd.read_csv(paths[kind], sep=";")
    missing = [c for c in FEATURE_NAMES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected columns, missing {missing}")
    df = df.copy()
    df[TYPE_COL] = 0 if kind == "red" else 1
    df["group"] = df.apply(feature_group_id, axis=1)
    df["stratum"] = stratify_label(df[TARGET])
    return df


def frozen_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test positional indices. Groups never split across the cut.

    Implemented as StratifiedGroupKFold with n_splits = round(1/test_size)
    (5-fold → 20 % test). Fold 0 is the frozen test set.
    """
    n_splits = max(2, int(round(1.0 / test_size)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(df), 1))
    train_idx, test_idx = next(
        cv.split(X_dummy, df["stratum"].to_numpy(), df["group"].to_numpy())
    )
    overlap = set(df.iloc[train_idx]["group"]) & set(df.iloc[test_idx]["group"])
    if overlap:
        raise RuntimeError(f"group leakage: {len(overlap)} groups in both splits")
    return train_idx, test_idx


def save_split_indices(kind: str, train_idx: np.ndarray, test_idx: np.ndarray) -> Path:
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    path = DATA_SPLITS / f"{kind}_split_seed{SPLIT_SEED}.npz"
    np.savez(path, train_idx=train_idx, test_idx=test_idx)
    return path


def xy(df: pd.DataFrame, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    part = df if idx is None else df.iloc[idx]
    X = part[FEATURE_NAMES].to_numpy(dtype=float)
    y = part[TARGET].to_numpy(dtype=float)
    return X, y
