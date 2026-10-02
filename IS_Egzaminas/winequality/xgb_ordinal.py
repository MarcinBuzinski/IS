"""XGBOrdinal: Frank–Hall slenksčiai ir po vieną XGBClassifier kiekvienam slenksčiui.

    g_k(x) = P(y > r_k | x)
    P(y=r_1|x) = 1−g_1(x)
    P(y=r_k|x) = g_{k−1}(x)−g_k(x)
    P(y=r_K|x) = g_{K−1}(x)
    score_cont = Σ r_k P(y=r_k|x)

Nepriklausomi slenksčiai prieš skirtumą monotoniškai sutvarkomi.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBClassifier

from winequality.config import RANKS
from winequality.ordinal import class_proba_from_thresholds, expected_score, project_monotone
from winequality.transform import TrainPreprocessor


class _ConstantBinary:
    """Slenkstis, kuriam mokymo aibėje yra tik viena klasė."""

    def __init__(self, probability: float) -> None:
        self.constant_ = float(probability)

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        positive = np.full(n, self.constant_)
        return np.column_stack([1.0 - positive, positive])


class XGBOrdinalRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        seed: int = 0,
        n_jobs: int = 1,
        feature_names: list[str] | None = None,
        ranks: tuple[int, ...] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.n_jobs = n_jobs
        self.feature_names = feature_names
        self.ranks = ranks

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.rint(np.asarray(y, dtype=float)).astype(int)
        self.ranks_ = np.asarray(self.ranks if self.ranks is not None else RANKS, dtype=int)
        self.prep_ = TrainPreprocessor(feature_names=self.feature_names, log1p=False, scale=False)
        transformed = self.prep_.fit_transform(X)
        self.models_ = []
        for threshold in self.ranks_[:-1]:
            binary = (y > threshold).astype(int)
            n_pos = int(binary.sum())
            n_neg = int(len(binary) - n_pos)
            if n_pos == 0 or n_neg == 0:
                self.models_.append(_ConstantBinary(1.0 if n_pos == len(binary) else 0.0))
                continue
            # Be scale_pos_weight: g_k turi likti P(y > r_k | x), ne persvertas dažnis.
            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.seed,
                n_jobs=self.n_jobs,
                tree_method="hist",
                verbosity=0,
            )
            clf.fit(transformed, binary)
            self.models_.append(clf)
        return self

    def threshold_proba(self, X) -> np.ndarray:
        transformed = self.prep_.transform(np.asarray(X, dtype=float))
        columns = [model.predict_proba(transformed)[:, 1] for model in self.models_]
        return project_monotone(np.column_stack(columns))

    def predict_proba(self, X) -> np.ndarray:
        return class_proba_from_thresholds(self.threshold_proba(X))

    def expected_score(self, X) -> np.ndarray:
        return expected_score(self.predict_proba(X), self.ranks_.astype(float))

    def predict(self, X) -> np.ndarray:
        return self.expected_score(X)


def suggest_xgb(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 80, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def xgb_complexity(params: dict) -> float:
    return float(params["n_estimators"] * params["max_depth"])
