"""UCI Wine Quality atsisiuntimas, SHA-256 ir schemos patikra."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from winequality.config import (
    DATA_RAW,
    FEATURE_NAMES,
    RED_SHA256,
    RED_URL,
    TARGET,
    WHITE_SHA256,
    WHITE_URL,
)

SOURCES = {
    "red": (RED_URL, RED_SHA256, "winequality-red.csv"),
    "white": (WHITE_URL, WHITE_SHA256, "winequality-white.csv"),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == expected:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    digest = sha256_of(tmp)
    if digest != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 neatitinka {dest.name}: gauta {digest}, tikėtasi {expected}")
    tmp.replace(dest)
    return dest


def fetch_raw() -> dict[str, Path]:
    """Grąžina raudono ir balto vyno CSV kelius. Trūkstamus failus atsisiunčia."""
    paths = {}
    for kind, (url, digest, name) in SOURCES.items():
        paths[kind] = _download(url, DATA_RAW / name, digest)
    return paths


def read_wine_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    expected = FEATURE_NAMES + [TARGET]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV schema neturi stulpelių {missing}")
    return frame.loc[:, expected].copy()


def audit_frame(frame: pd.DataFrame, kind: str) -> dict:
    """Duomenų kortelės santrauka: apimtis, dublikatai, balų dažniai, diapazonai."""
    features = frame[FEATURE_NAMES]
    duplicate_rows = int(features.duplicated(keep=False).sum())
    quality = frame[TARGET].astype(int)
    ranges = {}
    for name in FEATURE_NAMES:
        column = features[name]
        ranges[name] = {
            "min": float(column.min()),
            "max": float(column.max()),
            "mean": float(column.mean()),
            "missing": int(column.isna().sum()),
        }
    return {
        "kind": kind,
        "n_rows": int(len(frame)),
        "n_features": len(FEATURE_NAMES),
        "missing_cells": int(features.isna().sum().sum()),
        "duplicate_feature_rows": duplicate_rows,
        "quality_counts": {str(k): int(v) for k, v in quality.value_counts().sort_index().items()},
        "feature_ranges": ranges,
        "source": "UCI Wine Quality, Cortez et al. 2009, DOI 10.24432/C56S3T",
    }


def write_audit(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
