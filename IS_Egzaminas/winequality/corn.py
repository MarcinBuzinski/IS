"""CORN: rangų nuoseklumas per sąlygines tikimybes (Shi, Cao, Raschka 2023).

    P(y > r_k | x) = Π_{j=1}^{k} P(y > r_j | y > r_{j−1}, x)

Nuostolis skaičiuojamas tik toms eilutėms, kurios pasiekė ankstesnį slenkstį.
Klasių indeksai mokyme yra 0…K−1, atitinkantys balus 3…9.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from winequality.config import RANKS, SCORE_MIN
from winequality.ordinal import chain_rule, class_proba_from_thresholds, expected_score
from winequality.transform import TrainPreprocessor


def corn_loss(logits: torch.Tensor, y_index: torch.Tensor, num_classes: int) -> torch.Tensor:
    """CORN nuostolis. y_index prasideda nuo 0."""
    losses = []
    counts = []
    for task in range(num_classes - 1):
        mask = y_index > task - 1
        if int(mask.sum()) == 0:
            continue
        labels = (y_index[mask] > task).to(logits.dtype)
        pred = logits[mask, task]
        task_loss = -(
            torch.nn.functional.logsigmoid(pred) * labels
            + (torch.nn.functional.logsigmoid(pred) - pred) * (1.0 - labels)
        )
        losses.append(task_loss.sum())
        counts.append(int(mask.sum()))
    if not counts:
        return logits.sum() * 0.0
    return torch.stack(losses).sum() / float(sum(counts))


class _CornNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], n_out: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for units in hidden:
            layers.extend([nn.Linear(width, units), nn.ReLU(), nn.Dropout(dropout)])
            width = units
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CornRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden: tuple[int, ...] = (32,),
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        learning_rate: float = 1e-3,
        max_epochs: int = 40,
        patience: int = 6,
        batch_size: int = 64,
        seed: int = 0,
        feature_names: list[str] | None = None,
        use_log1p: bool = True,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.feature_names = feature_names
        self.use_log1p = use_log1p

    def _hidden(self) -> tuple[int, ...]:
        hidden = self.hidden
        if isinstance(hidden, int):
            return (hidden,)
        return tuple(int(v) for v in hidden)

    def _build(self) -> _CornNet:
        return _CornNet(self.n_features_in_, self._hidden(), len(self.ranks_) - 1, self.dropout)

    def fit(self, X, y, groups=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.ranks_ = np.asarray(RANKS, dtype=int)
        self.prep_ = TrainPreprocessor(
            feature_names=self.feature_names,
            log1p=self.use_log1p,
            scale=True,
        )
        z = self.prep_.fit_transform(X).astype(np.float32)
        y_index = (np.clip(np.rint(y), int(self.ranks_[0]), int(self.ranks_[-1])).astype(int) - SCORE_MIN)
        self.n_features_in_ = z.shape[1]
        self._net = self._build()
        if groups is None:
            groups = np.arange(len(z))
        try:
            train_idx, valid_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=self.seed).split(z, y_index, groups)
            )
        except ValueError:
            cut = max(1, int(0.85 * len(z)))
            order = rng.permutation(len(z))
            train_idx, valid_idx = order[:cut], order[cut:]
        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        z_train = z[train_idx]
        y_train = y_index[train_idx].astype(np.int64)
        batch_size = max(8, int(self.batch_size))
        best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
        best_mae = float("inf")
        waited = 0
        self.n_epochs_ = 0
        for epoch in range(self.max_epochs):
            self._net.train()
            order = rng.permutation(len(z_train))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                xb = torch.from_numpy(z_train[batch])
                yb = torch.from_numpy(y_train[batch])
                optimizer.zero_grad()
                loss = corn_loss(self._net(xb), yb, len(self.ranks_))
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                optimizer.step()
            self.n_epochs_ = epoch + 1
            val_pred = self._predict_matrix(z[valid_idx])
            val_mae = float(np.mean(np.abs(y[valid_idx] - val_pred)))
            if val_mae + 1e-6 < best_mae:
                best_mae = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self._net.load_state_dict(best_state)
        self.best_val_mae_ = best_mae
        return self

    def _predict_matrix(self, z: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.from_numpy(np.asarray(z, dtype=np.float32))).cpu().numpy()
        conditional = 1.0 / (1.0 + np.exp(-logits))
        proba = class_proba_from_thresholds(chain_rule(conditional))
        return expected_score(proba, self.ranks_.astype(float))

    def predict_proba(self, X) -> np.ndarray:
        z = self.prep_.transform(np.asarray(X, dtype=float)).astype(np.float32)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.from_numpy(z)).cpu().numpy()
        conditional = 1.0 / (1.0 + np.exp(-logits))
        return class_proba_from_thresholds(chain_rule(conditional))

    def expected_score(self, X) -> np.ndarray:
        return expected_score(self.predict_proba(X), self.ranks_.astype(float))

    def predict(self, X) -> np.ndarray:
        return self.expected_score(X)

    def __getstate__(self):
        state = self.__dict__.copy()
        net = state.pop("_net", None)
        if net is not None:
            state["_net_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        return state

    def __setstate__(self, state):
        net_state = state.pop("_net_state", None)
        self.__dict__.update(state)
        self._net = None
        if net_state is not None:
            self._net = self._build()
            self._net.load_state_dict(net_state)


def suggest_corn(trial) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 2)
    units = trial.suggest_categorical("units", [16, 32, 64])
    hidden = (units,) if n_layers == 1 else (units, max(8, units // 2))
    return {
        "hidden": hidden,
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_layers": n_layers,
        "units": units,
    }


def corn_complexity(params: dict) -> float:
    return float(params["n_layers"] * params["units"])
