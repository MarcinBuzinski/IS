"""Random forest regressor — second intelligent method (plan M4, H2 reference)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from winequality.transform import Log1pStandardScaler


def forest_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    seed: int = 0,
    log1p: bool = True,
) -> Pipeline:
    # Trees do not need standardisation; log1p is optional and kept for a
    # fair feature-space comparison with SVR unless an ablation turns it off.
    return Pipeline(
        [
            ("prep", Log1pStandardScaler(log1p=log1p)),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
