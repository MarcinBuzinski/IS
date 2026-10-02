"""Formulių, skaidymo ir sprendimo taisyklių testai."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from winequality.config import FEATURE_NAMES, LOG1P_FEATURES, TARGET
from winequality.corn import corn_loss
from winequality.hypotheses import decide_deployment, h1_pass, h2_pass
from winequality.metrics import evaluate, mae, quadratic_weighted_kappa
from winequality.ordinal import chain_rule, class_proba_from_thresholds, expected_score, project_monotone
from winequality.splits import feature_group_id, frozen_split, load_wine
from winequality.svr_model import formula_predict, predict_from_dual, rbf_kernel, svr_pipeline
from winequality.transform import TrainPreprocessor
from winequality.tune import select_trial
from winequality.xgb_ordinal import XGBOrdinalRegressor


def test_threshold_probabilities_match_formula():
    conditional = np.array([[0.8, 0.5]])
    g = chain_rule(conditional)
    np.testing.assert_allclose(g, [[0.8, 0.4]])
    proba = class_proba_from_thresholds(g)
    np.testing.assert_allclose(proba, [[0.2, 0.4, 0.4]])
    assert proba.sum() == pytest.approx(1.0)
    score = expected_score(proba, np.array([3, 4, 5], dtype=float))
    assert score[0] == pytest.approx(0.2 * 3 + 0.4 * 4 + 0.4 * 5)


def test_monotone_projection_removes_negative_class_mass():
    g = project_monotone(np.array([[0.4, 0.9, 0.2]]))
    assert np.all(np.diff(g, axis=1) <= 1e-12)
    proba = class_proba_from_thresholds(g)
    assert np.all(proba >= -1e-12)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_corn_loss_is_finite():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.2, 0.2, 0.2]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([3, 1])
    loss = corn_loss(logits, target, num_classes=4)
    assert torch.isfinite(loss)
    loss.backward()


def test_xgb_ordinal_probabilities():
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(size=(120, len(FEATURE_NAMES))))
    y = np.clip(np.rint(4 + x[:, -1] - x[:, 1]), 3, 8)
    model = XGBOrdinalRegressor(n_estimators=15, max_depth=2, learning_rate=0.2, seed=0)
    model.fit(x, y)
    g = model.threshold_proba(x)
    assert np.all(np.diff(g, axis=1) <= 1e-8)
    proba = model.predict_proba(x)
    assert np.all(proba >= -1e-8)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(model.predict(x), expected_score(proba, model.ranks_.astype(float)))


def test_svr_formula_matches_sklearn():
    rng = np.random.default_rng(1)
    x = np.abs(rng.normal(size=(70, len(FEATURE_NAMES))))
    y = 5 + 0.3 * x[:, -1] + rng.normal(scale=0.15, size=70)
    pipe = svr_pipeline(C=2.0, epsilon=0.1, gamma=0.1)
    pipe.fit(x, y)
    np.testing.assert_allclose(formula_predict(pipe, x), pipe.predict(x), atol=1e-6)
    z = pipe.named_steps["prep"].transform(x[:2])
    manual = predict_from_dual(
        pipe.named_steps["svr"].support_vectors_,
        pipe.named_steps["svr"].dual_coef_,
        float(pipe.named_steps["svr"].intercept_[0]),
        z,
        float(pipe.named_steps["svr"]._gamma),
    )
    np.testing.assert_allclose(manual, rbf_kernel(pipe.named_steps["svr"].support_vectors_, z, pipe.named_steps["svr"]._gamma) @ pipe.named_steps["svr"].dual_coef_.ravel() + pipe.named_steps["svr"].intercept_[0])


def test_preprocessor_matches_manual_formula():
    rng = np.random.default_rng(2)
    x = np.abs(rng.normal(size=(40, len(FEATURE_NAMES)))) + 0.01
    prep = TrainPreprocessor(log1p=True, scale=True).fit(x)
    transformed = x.copy()
    for j, name in enumerate(FEATURE_NAMES):
        if name in LOG1P_FEATURES:
            transformed[:, j] = np.log1p(transformed[:, j])
    expected = (transformed - transformed.mean(axis=0)) / transformed.std(axis=0, ddof=1)
    np.testing.assert_allclose(prep.transform(x), expected, atol=1e-12)


def test_group_split_has_no_overlap():
    frame = load_wine("red")
    train_idx, test_idx = frozen_split(frame, seed=0)
    assert set(frame.iloc[train_idx]["group"]).isdisjoint(set(frame.iloc[test_idx]["group"]))
    assert len(train_idx) + len(test_idx) == len(frame)


def test_identical_rows_share_group():
    row = {column: 1.0 for column in FEATURE_NAMES}
    left = pd.Series({**row, TARGET: 5})
    right = pd.Series({**row, TARGET: 6})
    assert feature_group_id(left) == feature_group_id(right)


def test_metrics_and_kappa_bounds():
    y = np.array([5.0, 6.0, 7.0])
    pred = np.array([5.0, 5.0, 7.0])
    assert mae(y, pred) == pytest.approx(1.0 / 3.0)
    assert evaluate(y, pred)["acc_1.0"] == 1.0
    perfect = quadratic_weighted_kappa(y, y)
    assert perfect == pytest.approx(1.0)


def test_deployment_rule():
    assert decide_deployment(False, True) == "linear"
    assert decide_deployment(True, False) == "catboost"
    assert decide_deployment(True, True) == "xgb_ordinal"
    assert h1_pass(0.90, 1.0, 0.40, 0.34)
    assert not h1_pass(0.97, 1.0, 0.40, 0.34)
    assert h2_pass({"lo": -0.08, "hi": -0.02}, {"lo": -0.01, "hi": 0.02})
    assert not h2_pass({"lo": -0.02, "hi": 0.02}, {"lo": -0.02, "hi": 0.02})


def test_select_trial_prefers_kappa_inside_mae_band():
    chosen = select_trial(
        [
            {"number": 0, "mae": 0.50, "kappa": 0.20, "complexity": 10},
            {"number": 1, "mae": 0.505, "kappa": 0.40, "complexity": 30},
            {"number": 2, "mae": 0.505, "kappa": 0.40, "complexity": 12},
            {"number": 3, "mae": 0.60, "kappa": 0.90, "complexity": 1},
        ]
    )
    assert chosen["number"] == 2
