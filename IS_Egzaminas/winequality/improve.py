"""Požymių paruošimas ir pagerinti modeliai.

Pakeitimai, palyginti su pirmuoju palyginimu:
- log1p asimetriškiems požymiams ir chemijos santykiai
- dublikatų grupės sveria kaip viena eilutė, kraštiniai balai sveria daugiau
- asimetrinė MAE: žemo balo pervertinimas baudžiamas stipriau
- vienas CatBoost arba LightGBM ansamblis visiems Frank–Hall slenksčiams
- monotoniniai apribojimai alkoholiui, lakiosioms rūgštims ir slenksčio numeriui
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, RegressorMixin

from winequality.config import FEATURE_NAMES, LOG1P_FEATURES, RANKS
from winequality.ordinal import class_proba_from_thresholds, expected_score, project_monotone

ENGINEERED_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "log1p residual sugar",
    "log1p chlorides",
    "log1p free sulfur dioxide",
    "log1p total sulfur dioxide",
    "density",
    "pH",
    "log1p sulphates",
    "alcohol",
    "so2 ratio",
    "acid ratio",
    "alcohol density",
]
THRESHOLD_NAME = "threshold"
VOLATILE_INDEX = ENGINEERED_NAMES.index("volatile acidity")
ALCOHOL_INDEX = ENGINEERED_NAMES.index("alcohol")


def design_matrix(X: np.ndarray) -> np.ndarray:
    """11 laboratorinių matavimų → log1p, SO₂ santykis, rūgščių santykis, alkoholis/tankis."""
    raw = np.asarray(X, dtype=float)
    if raw.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"tikėtasi {len(FEATURE_NAMES)} požymių, gauta {raw.shape[1]}")
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out = raw.copy()
    for name in LOG1P_FEATURES:
        column = index[name]
        out[:, column] = np.log1p(np.clip(out[:, column], 0.0, None))
    free = raw[:, index["free sulfur dioxide"]]
    total = raw[:, index["total sulfur dioxide"]]
    volatile = raw[:, index["volatile acidity"]]
    fixed = raw[:, index["fixed acidity"]]
    alcohol = raw[:, index["alcohol"]]
    density = raw[:, index["density"]]
    so2_ratio = free / np.maximum(total, 1e-6)
    acid_ratio = volatile / np.maximum(fixed, 1e-6)
    alcohol_density = alcohol / np.maximum(density, 1e-6)
    return np.column_stack([out, so2_ratio, acid_ratio, alcohol_density])


def training_weights(y: np.ndarray, groups: np.ndarray, tail_weight: float = 2.0) -> np.ndarray:
    """Grupės dublikatai kartu sveria 1. Balai ≤4 ir ≥8 gauna papildomą svorį."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    counts: dict[object, int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    weights = np.array([1.0 / counts[group] for group in groups], dtype=float)
    tails = (y <= 4.0) | (y >= 8.0)
    weights = weights * np.where(tails, tail_weight, 1.0)
    return weights * (len(weights) / weights.sum())


class AsymmetricMaeObjective:
    """MAE, kurioje y≤4 ir prognozė per didelė baudžiama `alpha` kartų stipriau.

    Ženklai sutampa su CatBoost RMSE pavyzdžiu: der1 = kryptis, kuria didinama prognozė,
    der2 neigiama.
    """

    def __init__(self, alpha: float = 2.0, low: float = 4.0) -> None:
        self.alpha = alpha
        self.low = low

    def calc_ders_range(self, approxes, targets, weights):
        alpha = float(self.alpha)
        low = float(self.low)
        result = []
        for i in range(len(targets)):
            gap = targets[i] - approxes[i]
            if gap < 0.0 and targets[i] <= low:
                der1 = alpha * gap / abs(gap)
            elif gap == 0.0:
                der1 = 0.0
            else:
                der1 = gap / abs(gap)
            der2 = -1.0
            if weights is not None:
                der1 *= weights[i]
                der2 *= weights[i]
            result.append((der1, der2))
        return result


def _monotone_flags(n_features: int, with_threshold: bool) -> list[int]:
    flags = [0] * n_features
    flags[VOLATILE_INDEX] = -1
    flags[ALCOHOL_INDEX] = 1
    if with_threshold:
        flags.append(-1)
    return flags


def _monotone_string(n_features: int, with_threshold: bool) -> str:
    flags = _monotone_flags(n_features, with_threshold)
    return "(" + ",".join(str(flag) for flag in flags) + ")"


class AsymmetricCatBoost(BaseEstimator, RegressorMixin):
    """CatBoost regresija su asimetrine MAE ir monotoniniais apribojimais."""

    def __init__(
        self,
        iterations: int = 250,
        depth: int = 6,
        learning_rate: float = 0.08,
        l2_leaf_reg: float = 3.0,
        alpha: float = 2.0,
        tail_weight: float = 2.0,
        seed: int = 0,
        thread_count: int = 1,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.alpha = alpha
        self.tail_weight = tail_weight
        self.seed = seed
        self.thread_count = thread_count

    def fit(self, X, y, groups=None):
        y = np.asarray(y, dtype=float)
        matrix = design_matrix(X)
        if groups is None:
            groups = np.arange(len(y))
        weights = training_weights(y, groups, self.tail_weight)
        # Žemi balai gauna papildomą svorį. CatBoost naudotojo nuostolis lieka Python
        # cikle, todėl čia naudojama įmontuota MAE.
        weights = weights * np.where(y <= 4.0, self.alpha, 1.0)
        weights = weights * (len(weights) / weights.sum())
        pool = Pool(matrix, y, weight=weights, feature_names=ENGINEERED_NAMES)
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="MAE",
            monotone_constraints=_monotone_string(matrix.shape[1], with_threshold=False),
            random_seed=self.seed,
            thread_count=self.thread_count,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(pool)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(design_matrix(X)), dtype=float)


class SharedOrdinalBoost(BaseEstimator, RegressorMixin):
    """Frank–Hall slenksčiai viename ansamblyje.

    Kiekviena eilutė pakartojama kiekvienam slenksčiui, o slenksčio numeris yra požymis
    su apribojimu, kad didesnis slenkstis nemažintų tikimybės P(y > r).
    """

    def __init__(
        self,
        backend: str = "catboost",
        iterations: int = 250,
        depth: int = 6,
        learning_rate: float = 0.08,
        l2_leaf_reg: float = 3.0,
        tail_weight: float = 2.0,
        seed: int = 0,
        thread_count: int = 1,
    ) -> None:
        self.backend = backend
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.tail_weight = tail_weight
        self.seed = seed
        self.thread_count = thread_count

    def fit(self, X, y, groups=None):
        y = np.rint(np.asarray(y, dtype=float)).astype(int)
        if groups is None:
            groups = np.arange(len(y))
        self.ranks_ = np.asarray(RANKS, dtype=int)
        self.thresholds_ = self.ranks_[:-1]
        matrix = design_matrix(X)
        weights = training_weights(y, groups, self.tail_weight)
        n_thresholds = len(self.thresholds_)
        expanded = np.repeat(matrix, n_thresholds, axis=0)
        threshold_id = np.tile(np.arange(n_thresholds, dtype=float), len(y))
        expanded = np.column_stack([expanded, threshold_id])
        labels = (np.repeat(y, n_thresholds) > np.tile(self.thresholds_, len(y))).astype(int)
        expanded_weights = np.repeat(weights, n_thresholds)
        names = ENGINEERED_NAMES + [THRESHOLD_NAME]
        constraints = _monotone_string(matrix.shape[1], with_threshold=True)
        constraint_flags = _monotone_flags(matrix.shape[1], with_threshold=True)
        if self.backend == "catboost":
            pool = Pool(expanded, labels, weight=expanded_weights, feature_names=names)
            self.model_ = CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                loss_function="Logloss",
                monotone_constraints=constraints,
                random_seed=self.seed,
                thread_count=self.thread_count,
                verbose=False,
                allow_writing_files=False,
            )
            self.model_.fit(pool)
        elif self.backend == "lightgbm":
            self.model_ = LGBMClassifier(
                n_estimators=self.iterations,
                max_depth=self.depth,
                learning_rate=self.learning_rate,
                reg_lambda=self.l2_leaf_reg,
                objective="binary",
                monotone_constraints=constraint_flags,
                random_state=self.seed,
                n_jobs=self.thread_count,
                verbosity=-1,
            )
            self.model_.fit(expanded, labels, sample_weight=expanded_weights)
        else:
            raise ValueError(f"nežinomas backend {self.backend!r}")
        return self

    def threshold_proba(self, X) -> np.ndarray:
        matrix = design_matrix(X)
        columns = []
        for index, _threshold in enumerate(self.thresholds_):
            query = np.column_stack([matrix, np.full(len(matrix), float(index))])
            proba = self.model_.predict_proba(query)
            classes = list(getattr(self.model_, "classes_", [0, 1]))
            if 1 in classes:
                columns.append(proba[:, classes.index(1)])
            else:
                columns.append(np.zeros(len(matrix)))
        return project_monotone(np.column_stack(columns))

    def predict_proba(self, X) -> np.ndarray:
        return class_proba_from_thresholds(self.threshold_proba(X))

    def predict(self, X) -> np.ndarray:
        return expected_score(self.predict_proba(X), self.ranks_.astype(float))
