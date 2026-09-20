"""Robustness: group-split leakage vs random split, and feature noise."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from winequality.config import DEFAULT_SEED
from winequality.metrics import evaluate
from winequality.splits import stratify_label
from winequality.tune import tune_svr


def random_split_indices(df, test_size: float = 0.2, seed: int = DEFAULT_SEED):
    """Leakage-prone split: duplicate groups may land on both sides."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    y = stratify_label(df["quality"]).to_numpy()
    train_idx, test_idx = next(sss.split(np.zeros((len(df), 1)), y))
    return train_idx, test_idx


def group_overlap(df, train_idx, test_idx) -> int:
    return len(set(df.iloc[train_idx]["group"]) & set(df.iloc[test_idx]["group"]))


def noise_stress(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale: float = 0.10,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Add N(0, scale · train-style column std of the *test* features) noise.

    Uses the test matrix std only to set the perturbation size, not to retrain.
    """
    rng = np.random.default_rng(seed)
    std = X_test.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    X_noisy = X_test + rng.normal(0.0, scale, size=X_test.shape) * std
    clean = evaluate(y_test, pipeline.predict(X_test))
    noisy = evaluate(y_test, pipeline.predict(X_noisy))
    return {
        "noise_scale_of_std": scale,
        "mae_clean": clean["mae"],
        "mae_noisy": noisy["mae"],
        "mae_delta": noisy["mae"] - clean["mae"],
        "acc_1.0_clean": clean["acc_1.0"],
        "acc_1.0_noisy": noisy["acc_1.0"],
    }


def leakage_comparison(
    df,
    train_idx_group,
    test_idx_group,
    seed: int,
    group_metrics: dict | None = None,
    group_tune: dict | None = None,
) -> dict:
    """A6: group-aware split vs random split, same SVR protocol."""
    from winequality.splits import xy

    def _run(tr, te, label: str, metrics=None, info=None) -> dict:
        if metrics is None or info is None:
            X_tr, y_tr = xy(df, tr)
            X_te, y_te = xy(df, te)
            groups_tr = df.iloc[tr]["group"].to_numpy()
            model, info = tune_svr(X_tr, y_tr, groups_tr, seed=seed)
            metrics = evaluate(y_te, model.predict(X_te))
        return {
            "label": label,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "overlapping_groups": group_overlap(df, tr, te),
            "metrics": metrics,
            "tune": info,
        }

    rand_tr, rand_te = random_split_indices(df, seed=seed)
    return {
        "group_aware": _run(
            train_idx_group, test_idx_group, "group_aware", group_metrics, group_tune
        ),
        "random": _run(rand_tr, rand_te, "random"),
    }
