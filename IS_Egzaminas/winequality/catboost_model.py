"""CatBoost regresija kaip boosting atskaita be ordinalinės galvutės.

    F_m(x) = F_{m−1}(x) + η h_m(x)
"""

from __future__ import annotations

from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from winequality.transform import TrainPreprocessor


def make_catboost(
    iterations: int = 300,
    depth: int = 6,
    learning_rate: float = 0.08,
    l2_leaf_reg: float = 3.0,
    random_strength: float = 1.0,
    loss: str = "MAE",
    seed: int = 0,
    thread_count: int = 1,
    feature_names: list[str] | None = None,
) -> Pipeline:
    return Pipeline(
        [
            ("prep", TrainPreprocessor(feature_names=feature_names, log1p=False, scale=False)),
            (
                "cat",
                CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg,
                    random_strength=random_strength,
                    loss_function=loss,
                    random_seed=seed,
                    thread_count=thread_count,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


def suggest_catboost(trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "loss": trial.suggest_categorical("loss", ["MAE", "Huber:delta=1.0"]),
    }


def catboost_complexity(params: dict) -> float:
    return float(params["iterations"] * params["depth"])
