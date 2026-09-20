"""Bendras kelio ir duomenų paruošimas Jupyter užrašinėms."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "winequality" / "config.py").exists():
            return cand
    raise FileNotFoundError("Nerastas projekto šakninis katalogas (winequality/config.py).")


def ensure_path() -> Path:
    root = repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


ROOT = ensure_path()

from winequality.config import ARTIFACTS, FEATURE_NAMES, RESULTS, SPLIT_SEED  # noqa: E402
from winequality.metrics import evaluate  # noqa: E402
from winequality.splits import frozen_split, load_wine, xy  # noqa: E402


def paruošti(kind: str = "red"):
    """Tas pats užšaldytas 80/20 grupinis skaidymas kaip eksperimento protokole."""
    df = load_wine(kind)
    train_idx, test_idx = frozen_split(df, seed=SPLIT_SEED)
    X_tr, y_tr = xy(df, train_idx)
    X_te, y_te = xy(df, test_idx)
    return {
        "kind": kind,
        "df": df,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te,
        "y_te": y_te,
        "groups_tr": df.iloc[train_idx]["group"].to_numpy(),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def metrikos_lentele(y_true, f, pavadinimas: str) -> pd.DataFrame:
    m = evaluate(y_true, f)
    eil = {
        "modelis": pavadinimas,
        "MAE": m["mae"],
        "MAE (apvalinta)": m["mae_rounded"],
        "Acc_0.5": m["acc_0.5"],
        "Acc_1.0": m["acc_1.0"],
        "κ_w": m["kappa_w"],
        "makro-F1": m["macro_f1"],
    }
    return pd.DataFrame([eil])


def artefaktas(vardas: str, kind: str) -> Path:
    return ARTIFACTS / f"{vardas}_{kind}.joblib"
