"""Hyperparameter search on the training block only (never the frozen test)."""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from winequality.baselines import linear_pipeline, mean_predictor
from winequality.config import DEFAULT_SEED, INNER_CV_SPLITS, MLP_N_ITER, RBF_N_ITER, RF_N_ITER, SVR_N_ITER
from winequality.forest import forest_pipeline
from winequality.mlp import mlp_pipeline
from winequality.rbf import rbf_pipeline
from winequality.svr_model import data_driven_epsilon, svr_pipeline


def _group_cv(groups: np.ndarray, n_splits: int = INNER_CV_SPLITS) -> GroupKFold:
    n_groups = len(np.unique(groups))
    splits = max(2, min(n_splits, n_groups))
    return GroupKFold(n_splits=splits)


def tune_svr(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kernel: str = "rbf",
    seed: int = DEFAULT_SEED,
    n_iter: int = SVR_N_ITER,
) -> tuple[Pipeline, dict]:
    pipe = svr_pipeline(kernel=kernel, seed=seed)
    eps0 = data_driven_epsilon(y)
    param_dist = {
        "svr__C": loguniform(2**-2, 2**6),
        "svr__epsilon": [0.05, 0.1, 0.2, 0.3, 0.5, round(eps0, 3)],
        "svr__gamma": loguniform(2**-7, 2**1) if kernel == "rbf" else ["scale"],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=_group_cv(groups),
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_mae": float(-search.best_score_),
        "eps0": eps0,
        "n_iter": n_iter,
    }


def tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = DEFAULT_SEED,
    n_iter: int = RF_N_ITER,
) -> tuple[Pipeline, dict]:
    pipe = forest_pipeline(seed=seed)
    param_dist = {
        "rf__n_estimators": [100, 200, 400],
        "rf__max_depth": [None, 8, 12, 16],
        "rf__min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=_group_cv(groups),
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_mae": float(-search.best_score_),
        "n_iter": n_iter,
    }


def tune_mlp(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = DEFAULT_SEED,
    n_iter: int = MLP_N_ITER,
) -> tuple[Pipeline, dict]:
    pipe = mlp_pipeline(seed=seed)
    param_dist = {
        "mlp__hidden_layer_sizes": [(8,), (16,), (32,), (64,), (16, 8), (32, 16)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": loguniform(1e-5, 1e-2),
        "mlp__learning_rate_init": loguniform(3e-4, 3e-3),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=_group_cv(groups),
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, {
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
        "best_cv_mae": float(-search.best_score_),
        "n_iter": n_iter,
    }


def tune_rbf(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = DEFAULT_SEED,
    n_iter: int = RBF_N_ITER,
) -> tuple[Pipeline, dict]:
    pipe = rbf_pipeline(seed=seed)
    param_dist = {
        "rbf__n_centers": [15, 25, 40, 60, 80],
        "rbf__ridge": loguniform(1e-4, 1.0),
        "rbf__n_neighbors": [1, 2, 3],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=_group_cv(groups),
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_mae": float(-search.best_score_),
        "n_iter": n_iter,
    }


def fit_baselines(X: np.ndarray, y: np.ndarray) -> dict[str, object]:
    mean_m = mean_predictor(y)
    lin = linear_pipeline()
    lin.fit(X, y)
    return {"mean": mean_m, "linear": lin}
