"""Mokymo klostėje apskaičiuojama imputacija, log1p ir standartizavimas.

SVR, CORN ir tiesinė regresija:

    x'_j = ln(1 + x_j)     j ∈ {residual sugar, chlorides, free SO2, total SO2, sulphates}
    z_j  = (x'_j − μ_j) / σ_j

μ ir σ skaičiuojami tik iš tos klostės, kuria modelis mokomas.
XGBOrdinal ir CatBoost gauna tik medianos imputaciją: medžių skaidiniai
nepriklauso nuo monotoninės skalės.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from winequality.config import FEATURE_NAMES, LOG1P_FEATURES


class TrainPreprocessor(BaseEstimator, TransformerMixin):
    """Imputuoja mediana, pasirinktinai taiko log1p ir z-standartizavimą."""

    def __init__(
        self,
        feature_names: list[str] | None = None,
        log1p: bool = True,
        scale: bool = True,
    ) -> None:
        self.feature_names = feature_names
        self.log1p = log1p
        self.scale = scale

    def _names(self, n_features: int) -> list[str]:
        if self.feature_names is not None:
            names = list(self.feature_names)
            if len(names) != n_features:
                raise ValueError(f"feature_names ilgis {len(names)} != {n_features}")
            return names
        if n_features == len(FEATURE_NAMES):
            return list(FEATURE_NAMES)
        return [f"f{i}" for i in range(n_features)]

    def _impute(self, X: np.ndarray) -> np.ndarray:
        out = np.array(X, dtype=float, copy=True)
        missing = np.isnan(out)
        if missing.any():
            out[missing] = np.take(self.median_, np.where(missing)[1])
        return out

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.feature_names_ = self._names(X.shape[1])
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        self._log1p_mask = np.array(
            [self.log1p and name in LOG1P_FEATURES for name in self.feature_names_],
            dtype=bool,
        )
        transformed = self._impute(X)
        if self._log1p_mask.any():
            cols = transformed[:, self._log1p_mask]
            transformed[:, self._log1p_mask] = np.log1p(np.clip(cols, 0.0, None))
        if self.scale:
            self.mean_ = transformed.mean(axis=0)
            scale = transformed.std(axis=0, ddof=1)
            self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.scale_ = np.ones(X.shape[1], dtype=float)
        return self

    def transform(self, X):
        if not hasattr(self, "median_"):
            raise RuntimeError("TrainPreprocessor turi būti apmokytas prieš transform.")
        transformed = self._impute(np.asarray(X, dtype=float))
        if self._log1p_mask.any():
            cols = transformed[:, self._log1p_mask]
            transformed[:, self._log1p_mask] = np.log1p(np.clip(cols, 0.0, None))
        if self.scale:
            transformed = (transformed - self.mean_) / self.scale_
        return transformed
