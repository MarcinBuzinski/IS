"""Baseline: mokymo vidurkis ir tiesinė regresija ant standartizuotų požymių.

    ŷ_vid = (1/N) Σ_i y_i
    ŷ_lin = wᵀ z + b
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from winequality.transform import TrainPreprocessor


class MeanRegressor(BaseEstimator, RegressorMixin):
    """Visiems mėginiams grąžina mokymo aibės balų vidurkį."""

    def fit(self, X, y):
        self.mu_ = float(np.mean(np.asarray(y, dtype=float)))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.mu_, dtype=float)


def linear_pipeline(feature_names: list[str] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("prep", TrainPreprocessor(feature_names=feature_names, log1p=False, scale=True)),
            ("lin", LinearRegression()),
        ]
    )
