"""Shared constants: seeds, paths, feature names, class rules, physical bounds.

Fixed seeds (exam requirement). Split seed is independent of model RNG so that
every method sees the same frozen 80/20 group-aware hold-out.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

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
TYPE_COL = "type"  # 0 = red, 1 = white

# log1p set from the plan (skew > 1 on at least one wine type, excluding volatile
# acidity / citric acid which are only skewed for white).
LOG1P_FEATURES = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "sulphates",
]

# Conservative physical / laboratory bounds used at serving time.
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
CLASS_LOW_MAX = 4
CLASS_HIGH_MIN = 7
CLASS_NAMES = ("low", "medium", "high")  # žema / vidutinė / aukšta

# Inner HPO is on the training block only (never the frozen test).
INNER_CV_SPLITS = 3
SVR_N_ITER = 12
RF_N_ITER = 8
MLP_N_ITER = 8
RBF_N_ITER = 8
