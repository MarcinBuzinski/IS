"""Multi-layer perceptron — plan method M2 (LD2 / course MLP).

sklearn MLPRegressor (Adam, L2, early stopping) is the method used in the
comparison. Forward pass of a 1-hidden-layer net, for the formula ↔ code link:

    h = act(W1 z + b1)
    f(z) = w2ᵀ h + b2

with act ∈ {ReLU, tanh}. Training (backprop) is done by sklearn once; the
exam allows skipping training-time formulas when the model is fit once.
"""

from __future__ import annotations

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from winequality.transform import Log1pStandardScaler


def mlp_pipeline(
    hidden_layer_sizes: tuple[int, ...] = (32,),
    activation: str = "relu",
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    seed: int = 0,
) -> Pipeline:
    return Pipeline(
        [
            ("prep", Log1pStandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    solver="adam",
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=seed,
                ),
            ),
        ]
    )
