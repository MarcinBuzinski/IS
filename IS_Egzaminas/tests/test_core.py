"""Unit tests that do not touch the frozen-test protocol numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winequality.config import FEATURE_NAMES, TARGET
from winequality.metrics import evaluate, mae
from winequality.splits import feature_group_id, frozen_split, load_wine
from winequality.svr_model import formula_predict, svr_pipeline
from winequality.rbf import hidden_activations, predict_from_rbf, rbf_pipeline
from winequality.transform import Log1pStandardScaler


def test_group_split_has_no_overlap():
    df = load_wine("red")
    tr, te = frozen_split(df, seed=0)
    assert set(df.iloc[tr]["group"]).isdisjoint(set(df.iloc[te]["group"]))
    assert len(tr) + len(te) == len(df)


def test_identical_rows_share_group():
    row = {c: 1.0 for c in FEATURE_NAMES}
    a = pd.Series({**row, TARGET: 5})
    b = pd.Series({**row, TARGET: 5})
    assert feature_group_id(a) == feature_group_id(b)


def test_scaler_matches_manual_formula():
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(40, len(FEATURE_NAMES)))) + 0.01
    sc = Log1pStandardScaler().fit(X)
    z = sc.transform(X)
    X_t = X.copy()
    for j, name in enumerate(FEATURE_NAMES):
        if name in {"residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "sulphates"}:
            X_t[:, j] = np.log1p(X[:, j])
    mu = X_t.mean(axis=0)
    sd = X_t.std(axis=0, ddof=1)
    expected = (X_t - mu) / sd
    np.testing.assert_allclose(z, expected, atol=1e-12)


def test_svr_formula_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, len(FEATURE_NAMES)))
    X = np.abs(X)
    y = 5 + 0.3 * X[:, -1] + rng.normal(scale=0.2, size=80)
    pipe = svr_pipeline(C=2.0, epsilon=0.1, gamma=0.1)
    pipe.fit(X, y)
    a = formula_predict(pipe, X)
    b = pipe.predict(X)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_rbf_predict_matches_formula():
    rng = np.random.default_rng(2)
    X = np.abs(rng.normal(size=(60, len(FEATURE_NAMES))))
    y = 5 + 0.4 * X[:, -1] - 0.2 * X[:, 1] + rng.normal(scale=0.15, size=60)
    pipe = rbf_pipeline(n_centers=12, ridge=0.01, seed=0)
    pipe.fit(X, y)
    z = pipe.named_steps["prep"].transform(X)
    net = pipe.named_steps["rbf"]
    from_formula = predict_from_rbf(z, net.centres_, net.widths_, net.weights_)
    np.testing.assert_allclose(from_formula, pipe.predict(X), atol=1e-10)
    phi = hidden_activations(z, net.centres_, net.widths_)
    assert phi.shape == (len(X), net.centres_.shape[0])
    assert np.all(phi >= 0) and np.all(phi <= 1 + 1e-12)


def test_metrics_mae_known():
    y = np.array([5.0, 6.0, 7.0])
    f = np.array([5.0, 5.0, 7.0])
    assert mae(y, f) == pytest.approx(1.0 / 3.0)
    ev = evaluate(y, f)
    assert ev["acc_1.0"] == 1.0
