"""SVR with Gaussian (RBF) kernel — main intelligent method.

Inference formula implemented explicitly in `predict_from_dual` (plan §5):

    f(z) = Σ_{i ∈ SV} (α_i − α_i*) K(z_i, z) + b
    K(z_i, z) = exp(−γ ‖z_i − z‖²)

Training (once, via LIBSVM/SMO in sklearn): ε-insensitive primal

    min_{w,b,ξ,ξ*}  (1/2)‖w‖² + C Σ_i (ξ_i + ξ_i*)
    s.t.  y_i − wᵀφ(z_i) − b ≤ ε + ξ_i
          wᵀφ(z_i) + b − y_i ≤ ε + ξ_i*
          ξ_i, ξ_i* ≥ 0

sklearn stores dual_coef_ = (α − α*) and intercept_ = b; support_vectors_ = {z_i}.
`predict_from_dual` reconstructs f from those arrays so the formula is the code
path used in tests and in the CLI, not only a comment.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from winequality.transform import Log1pStandardScaler


def rbf_kernel(support: np.ndarray, z: np.ndarray, gamma: float) -> np.ndarray:
    """K(z_i, z) = exp(−γ ‖z_i − z‖²).

    support: (n_sv, d), z: (n, d) → (n, n_sv)
    """
    # ‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b
    sv_norm = np.sum(support**2, axis=1)
    z_norm = np.sum(z**2, axis=1)
    sq = sv_norm[None, :] + z_norm[:, None] - 2.0 * z @ support.T
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-gamma * sq)


def predict_from_dual(
    support_vectors: np.ndarray,
    dual_coef: np.ndarray,
    intercept: float,
    z: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """f(z) = Σ_i (α_i − α_i*) K(z_i, z) + b.

    dual_coef is sklearn's dual_coef_.ravel() ≡ (α − α*) on support vectors.
    This function is the formula-to-code link required by the exam.
    """
    k = rbf_kernel(support_vectors, np.asarray(z, dtype=float), gamma)
    return k @ np.asarray(dual_coef, dtype=float).ravel() + float(intercept)


def data_driven_epsilon(y: np.ndarray) -> float:
    """Cherkassky-style starting value used as an extra ε candidate.

    ε ≈ 3 σ √(ln n / n), clipped to the search grid range. Wu & Wang (2022)
    motivate a data-driven insensitivity width; this is the cheap initial
    estimate used before inner CV picks the final ε.
    """
    y = np.asarray(y, dtype=float)
    n = max(len(y), 2)
    sigma = float(np.std(y, ddof=1))
    eps = 3.0 * sigma * np.sqrt(np.log(n) / n)
    return float(np.clip(eps, 0.05, 0.5))


def svr_pipeline(
    C: float = 1.0,
    epsilon: float = 0.2,
    gamma: float | str = "scale",
    kernel: str = "rbf",
    seed: int = 0,
) -> Pipeline:
    return Pipeline(
        [
            ("prep", Log1pStandardScaler()),
            (
                "svr",
                SVR(
                    kernel=kernel,
                    C=C,
                    epsilon=epsilon,
                    gamma=gamma,
                    cache_size=256,
                ),
            ),
        ]
    )


def formula_predict(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Apply the stored SVR dual formula after the same preprocessing."""
    prep: Log1pStandardScaler = pipeline.named_steps["prep"]
    svr: SVR = pipeline.named_steps["svr"]
    if svr.kernel != "rbf":
        return pipeline.predict(X)
    z = prep.transform(X)
    gamma = svr._gamma
    return predict_from_dual(
        svr.support_vectors_,
        svr.dual_coef_,
        float(svr.intercept_[0]),
        z,
        gamma,
    )
