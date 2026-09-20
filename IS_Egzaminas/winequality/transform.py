"""log1p + z-score transformer (fit on the training fold only).

Formulas (plan §5), applied column-wise:

    x'_j = ln(1 + x_j)     for j in S = {residual sugar, chlorides,
                                         free SO2, total SO2, sulphates}
    z_j  = (x'_j - μ_j) / σ_j

μ_j and σ_j are estimated on the current training fold and stored on this
object, so they never leak from the frozen test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from winequality.config import FEATURE_NAMES, LOG1P_FEATURES


class Log1pStandardScaler(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer implementing the plan's preprocessing.

    Code ↔ formula:
      _log1p_mask   selects S
      _apply_log1p  is x'_j = ln(1 + x_j)
      transform     is z_j  = (x'_j - μ_j) / σ_j   (σ_j floored at eps)
    """

    def __init__(self, feature_names: list[str] | None = None, log1p: bool = True) -> None:
        # sklearn clone requires constructor args to be stored unmodified.
        self.feature_names = feature_names
        self.log1p = log1p
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._log1p_mask: np.ndarray | None = None

    def _names(self, n_features: int) -> list[str]:
        names = list(self.feature_names) if self.feature_names is not None else list(FEATURE_NAMES)
        return names[:n_features]

    def _apply_log1p(self, X: np.ndarray) -> np.ndarray:
        # x'_j = ln(1 + x_j)  for j in S; identity otherwise.
        out = np.asarray(X, dtype=float).copy()
        out[:, self._log1p_mask] = np.log1p(np.clip(out[:, self._log1p_mask], a_min=0.0, a_max=None))
        return out

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        names = self._names(X.shape[1])
        self._log1p_mask = np.array(
            [self.log1p and n in LOG1P_FEATURES for n in names],
            dtype=bool,
        )
        X_t = self._apply_log1p(X)
        self.mean_ = X_t.mean(axis=0)
        self.scale_ = X_t.std(axis=0, ddof=1)
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None or self._log1p_mask is None:
            raise RuntimeError("Log1pStandardScaler must be fit before transform.")
        X_t = self._apply_log1p(np.asarray(X, dtype=float))
        return (X_t - self.mean_) / self.scale_
