"""Error examples: large residuals and regression-to-the-mean by true score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from winequality.config import FEATURE_NAMES, TARGET
from winequality.metrics import round_score
from winequality.postprocess import confidence


def residual_table(df_test: pd.DataFrame, f: np.ndarray, n_examples: int = 8) -> pd.DataFrame:
    out = df_test[FEATURE_NAMES + [TARGET]].copy().reset_index(drop=True)
    out["f"] = f
    out["residual"] = out[TARGET] - out["f"]
    out["abs_residual"] = out["residual"].abs()
    out["q_hat"] = round_score(f)
    out["confidence"] = confidence(f)
    return out.sort_values("abs_residual", ascending=False).head(n_examples)


def mae_by_true_score(y_true: np.ndarray, f: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    f = np.asarray(f)
    out = {}
    for s in sorted(np.unique(np.rint(y_true).astype(int))):
        mask = np.rint(y_true).astype(int) == s
        out[str(int(s))] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(y_true[mask] - f[mask]))),
            "mean_f": float(np.mean(f[mask])),
            "bias": float(np.mean(f[mask] - y_true[mask])),
        }
    return out
