"""Dublikatų grupės ir užšaldytas stratifikuotas grupinis 80/20 skaidymas.

Identiškos 11 požymių eilutės gauna tą patį grupės ID ir nepersidengia
tarp mokymo ir testo. Reti balai 3 ir 9 stratifikacijoje sujungiami su 4 ir 8.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from winequality.config import DATA_SPLITS, FEATURE_NAMES, SPLIT_SEED, TARGET, TYPE_COL
from winequality.data_io import fetch_raw, read_wine_csv


def feature_group_id(row: pd.Series) -> str:
    payload = "|".join(f"{float(row[column]):.10g}" for column in FEATURE_NAMES)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def stratify_label(quality: pd.Series | np.ndarray) -> np.ndarray:
    labels = np.rint(np.asarray(quality)).astype(int)
    labels = np.where(labels == 3, 4, labels)
    labels = np.where(labels == 9, 8, labels)
    return labels


def load_wine(kind: str) -> pd.DataFrame:
    paths = fetch_raw()
    if kind not in paths:
        raise ValueError(f"kind turi būti 'red' arba 'white', gauta {kind!r}")
    frame = read_wine_csv(paths[kind])
    frame[TYPE_COL] = 0 if kind == "red" else 1
    frame["group"] = frame.apply(feature_group_id, axis=1)
    frame["stratum"] = stratify_label(frame[TARGET])
    return frame


def frozen_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Grąžina mokymo ir testo indeksus. 0-oji StratifiedGroupKFold klostė yra testas."""
    n_splits = max(2, int(round(1.0 / test_size)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(frame), 1))
    train_idx, test_idx = next(
        splitter.split(dummy, frame["stratum"].to_numpy(), frame["group"].to_numpy())
    )
    overlap = set(frame.iloc[train_idx]["group"]) & set(frame.iloc[test_idx]["group"])
    if overlap:
        raise RuntimeError(f"grupių nutekėjimas: {len(overlap)} grupės abiejose pusėse")
    return train_idx, test_idx


def assert_no_group_overlap(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> int:
    overlap = set(frame.iloc[train_idx]["group"]) & set(frame.iloc[test_idx]["group"])
    if overlap:
        raise RuntimeError(f"grupių sankirta nėra tuščia: {len(overlap)}")
    return 0


def save_split(kind: str, train_idx: np.ndarray, test_idx: np.ndarray, seed: int = SPLIT_SEED) -> Path:
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    path = DATA_SPLITS / f"{kind}.json"
    payload = {
        "kind": kind,
        "seed": int(seed),
        "train_idx": train_idx.astype(int).tolist(),
        "test_idx": test_idx.astype(int).tolist(),
        "group_overlap": 0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_split(kind: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_SPLITS / f"{kind}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["train_idx"], dtype=int), np.asarray(payload["test_idx"], dtype=int)


def xy(frame: pd.DataFrame, idx: np.ndarray | None = None, columns: list[str] | None = None):
    part = frame if idx is None else frame.iloc[idx]
    columns = columns or FEATURE_NAMES
    x = part[columns].to_numpy(dtype=float)
    y = part[TARGET].to_numpy(dtype=float)
    groups = part["group"].to_numpy()
    return x, y, groups
