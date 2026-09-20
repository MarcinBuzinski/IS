"""RBF network — plan method M3 (LD3). SVR with *fixed* centres.

Training (two stages, once):
  1. centres c_k by k-means on the standardised training inputs
  2. widths σ_k = mean distance from c_k to its n_neighbors nearest centres
  3. output weights by ridge least squares (closed form)

Hidden activations and prediction (the formula used at test time):

    φ_k(z) = exp( − ‖z − c_k‖² / (2 σ_k²) )
    f(z)   = Σ_k w_k φ_k(z) + b

    w = (Φᵀ Φ + λ I)^{-1} Φᵀ y     (bias column included in Φ; λ not applied to b)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from winequality.transform import Log1pStandardScaler


def hidden_activations(z: np.ndarray, centres: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """φ_k(z) = exp( − ‖z − c_k‖² / (2 σ_k²) ).  z: (n, d) → (n, K)."""
    z = np.asarray(z, dtype=float)
    # ‖z − c‖² = ‖z‖² + ‖c‖² − 2 z·c
    z_norm = np.sum(z**2, axis=1)
    c_norm = np.sum(centres**2, axis=1)
    sq = z_norm[:, None] + c_norm[None, :] - 2.0 * z @ centres.T
    np.maximum(sq, 0.0, out=sq)
    denom = 2.0 * np.square(widths)
    denom = np.where(denom < 1e-18, 1e-18, denom)
    return np.exp(-sq / denom)


def design_matrix(phi: np.ndarray) -> np.ndarray:
    """Φ with a leading column of ones for the bias b."""
    return np.column_stack([np.ones(phi.shape[0]), phi])


def ridge_output_weights(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve (Φᵀ Φ + λ I) w = Φᵀ y, with λ = 0 on the bias term."""
    phi_b = design_matrix(phi)
    k = phi_b.shape[1]
    gram = phi_b.T @ phi_b
    pen = np.eye(k) * float(ridge)
    pen[0, 0] = 0.0
    return np.linalg.solve(gram + pen, phi_b.T @ y)


def predict_from_rbf(
    z: np.ndarray,
    centres: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """f(z) = Σ_k w_k φ_k(z) + b, with weights[0] = b."""
    phi = hidden_activations(z, centres, widths)
    return design_matrix(phi) @ weights


def _centre_widths(centres: np.ndarray, n_neighbors: int) -> np.ndarray:
    k = len(centres)
    if k == 1:
        return np.ones(1)
    d = np.sqrt(np.maximum(0.0, np.sum((centres[:, None, :] - centres[None, :, :]) ** 2, axis=2)))
    np.fill_diagonal(d, np.inf)
    nn = min(n_neighbors, k - 1)
    parts = np.partition(d, nn - 1, axis=1)[:, :nn]
    widths = parts.mean(axis=1)
    fallback = np.median(d[np.isfinite(d)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    return np.where(widths < 1e-12, fallback, widths)


class RBFNetwork(BaseEstimator, RegressorMixin):
    """sklearn-compatible RBF network; constructor args stored unmodified for clone()."""

    def __init__(
        self,
        n_centers: int = 40,
        ridge: float = 1e-2,
        n_neighbors: int = 2,
        random_state: int = 0,
    ) -> None:
        self.n_centers = n_centers
        self.ridge = ridge
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        k = int(min(self.n_centers, len(X)))
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        km.fit(X)
        self.centres_ = km.cluster_centers_
        self.widths_ = _centre_widths(self.centres_, self.n_neighbors)
        phi = hidden_activations(X, self.centres_, self.widths_)
        self.weights_ = ridge_output_weights(phi, y, self.ridge)
        return self

    def predict(self, X):
        if not hasattr(self, "weights_"):
            raise RuntimeError("RBFNetwork must be fit before predict.")
        return predict_from_rbf(
            np.asarray(X, dtype=float),
            self.centres_,
            self.widths_,
            self.weights_,
        )


def rbf_pipeline(
    n_centers: int = 40,
    ridge: float = 1e-2,
    n_neighbors: int = 2,
    seed: int = 0,
) -> Pipeline:
    return Pipeline(
        [
            ("prep", Log1pStandardScaler()),
            (
                "rbf",
                RBFNetwork(
                    n_centers=n_centers,
                    ridge=ridge,
                    n_neighbors=n_neighbors,
                    random_state=seed,
                ),
            ),
        ]
    )
