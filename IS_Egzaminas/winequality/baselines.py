"""Mandatory baselines: mean prediction and linear regression (OLS / ridge)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from winequality.transform import Log1pStandardScaler


def mean_predictor(y_train: np.ndarray):
    """ŷ = ȳ_train  (constant; no features)."""
    mu = float(np.mean(y_train))

    class MeanModel:
        def predict(self, X):
            X = np.asarray(X)
            return np.full(shape=(X.shape[0],), fill_value=mu, dtype=float)

        def get_params(self, deep=True):
            return {"mu": mu}

    return MeanModel()


def linear_pipeline(ridge_alpha: float | None = None) -> Pipeline:
    """ŷ = wᵀ z + b on log1p+standardised features (plan's linear baseline)."""
    model = Ridge(alpha=ridge_alpha) if ridge_alpha is not None else LinearRegression()
    return Pipeline(
        [
            ("prep", Log1pStandardScaler()),
            ("lin", model),
        ]
    )
