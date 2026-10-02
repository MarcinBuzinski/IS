"""Keliai, požymiai, fizinės ribos, sėklos ir numatytasis paieškos biudžetas."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

SPLIT_SEED = 0
MODEL_SEEDS = (0, 1, 2)
DEFAULT_SEED = 0

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
RED_SHA256 = "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
WHITE_SHA256 = "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET = "quality"
TYPE_COL = "type"

LOG1P_FEATURES = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "sulphates",
]

FEATURE_GROUPS = {
    "acidity": ["fixed acidity", "volatile acidity", "citric acid", "pH"],
    "sugar": ["residual sugar"],
    "sulfur": ["free sulfur dioxide", "total sulfur dioxide", "sulphates"],
    "chlorides": ["chlorides"],
    "body": ["density", "alcohol"],
}

PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "fixed acidity": (3.0, 20.0),
    "volatile acidity": (0.0, 2.0),
    "citric acid": (0.0, 2.0),
    "residual sugar": (0.0, 80.0),
    "chlorides": (0.0, 1.0),
    "free sulfur dioxide": (0.0, 300.0),
    "total sulfur dioxide": (0.0, 500.0),
    "density": (0.98, 1.05),
    "pH": (2.5, 4.5),
    "sulphates": (0.0, 2.5),
    "alcohol": (7.0, 16.0),
}

SCORE_MIN = 3
SCORE_MAX = 9
RANKS = tuple(range(SCORE_MIN, SCORE_MAX + 1))
CLASS_LOW_MAX = 4
CLASS_HIGH_MIN = 7
CLASS_NAMES = ("žema", "vidutinė", "aukšta")

# Plano 7 lentelė: bandymai vienam vyno tipui, viduje 5 klostės.
DEFAULT_TRIALS = {
    "xgb_ordinal": 50,
    "catboost": 50,
    "corn": 30,
    "svr": 40,
}
INNER_CV_SPLITS = 5


def load_config(path: str | Path) -> dict:
    """Įkelia YAML ir užpildo numatytąsias plano reikšmes."""
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("seed", DEFAULT_SEED)
    cfg.setdefault("seeds", list(MODEL_SEEDS))
    cfg.setdefault("wines", ["red", "white"])
    cfg.setdefault("test_size", 0.2)
    cfg.setdefault("inner_splits", INNER_CV_SPLITS)
    cfg.setdefault("n_jobs", 1)
    cfg.setdefault("bootstrap", 1000)
    cfg.setdefault("output_dir", "results")
    cfg.setdefault("protocol", "full")
    cfg.setdefault("ablation", True)
    cfg.setdefault("robustness", True)
    cfg.setdefault("errors", True)
    cfg.setdefault("shap", True)
    trials = dict(DEFAULT_TRIALS)
    trials.update(cfg.get("trials") or {})
    cfg["trials"] = trials
    corn = {
        "max_epochs": 40,
        "patience": 6,
    }
    corn.update(cfg.get("corn") or {})
    cfg["corn"] = corn
    return cfg


def output_paths(cfg: dict) -> tuple[Path, Path]:
    results = ROOT / cfg["output_dir"]
    artifacts = results / "artifacts"
    results.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return results, artifacts
