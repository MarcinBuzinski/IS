"""SVR su Gauso branduoliu ir ε-nejautria nuostolio funkcija.

    f(z) = Σ_{i ∈ SV} (α_i − α_i*) exp(−γ ‖z_i − z‖²) + b

`predict_from_dual` yra ta formulė. `formula_predict` ją pritaiko po to paties
pirminio apdorojimo, kurį naudoja sklearn pipeline.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from winequality.transform import TrainPreprocessor


def rbf_kernel(support: np.ndarray, z: np.ndarray, gamma: float) -> np.ndarray:
    support = np.asarray(support, dtype=float)
    z = np.asarray(z, dtype=float)
    support_norm = np.sum(support**2, axis=1)
    z_norm = np.sum(z**2, axis=1)
    squared = support_norm[None, :] + z_norm[:, None] - 2.0 * z @ support.T
    np.maximum(squared, 0.0, out=squared)
    return np.exp(-float(gamma) * squared)


def predict_from_dual(
    support_vectors: np.ndarray,
    dual_coef: np.ndarray,
    intercept: float,
    z: np.ndarray,
    gamma: float,
) -> np.ndarray:
    kernel = rbf_kernel(support_vectors, z, gamma)
    return kernel @ np.asarray(dual_coef, dtype=float).ravel() + float(intercept)


def svr_pipeline(
    C: float = 1.0,
    epsilon: float = 0.2,
    gamma: float | str = "scale",
    seed: int = 0,
    feature_names: list[str] | None = None,
    log1p: bool = True,
) -> Pipeline:
    del seed  # SVR sprendimas deterministinis; sėkla palikta bendrai sąsajai.
    return Pipeline(
        [
            (
                "prep",
                TrainPreprocessor(feature_names=feature_names, log1p=log1p, scale=True),
            ),
            ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma, cache_size=256)),
        ]
    )


def formula_predict(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    prep: TrainPreprocessor = pipeline.named_steps["prep"]
    svr: SVR = pipeline.named_steps["svr"]
    z = prep.transform(X)
    if svr.kernel != "rbf":
        return pipeline.predict(X)
    return predict_from_dual(
        svr.support_vectors_,
        svr.dual_coef_,
        float(svr.intercept_[0]),
        z,
        float(svr._gamma),
    )


def suggest_svr(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.05, 0.5),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
    }


def svr_complexity(params: dict) -> float:
    return float(np.log(params["C"]) + params["epsilon"])
