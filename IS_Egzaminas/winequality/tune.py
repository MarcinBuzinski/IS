"""Train-only GroupKFold paieška. Testo aibė į šį modulį nepatenka."""

from __future__ import annotations

import optuna
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from winequality.catboost_model import catboost_complexity, make_catboost, suggest_catboost
from winequality.corn import corn_complexity, suggest_corn
from winequality.corn import CornRegressor
from winequality.metrics import mae, quadratic_weighted_kappa
from winequality.splits import stratify_label
from winequality.svr_model import suggest_svr, svr_complexity, svr_pipeline
from winequality.xgb_ordinal import XGBOrdinalRegressor, suggest_xgb, xgb_complexity

optuna.logging.set_verbosity(optuna.logging.WARNING)

SUGGEST = {
    "xgb_ordinal": suggest_xgb,
    "catboost": suggest_catboost,
    "corn": suggest_corn,
    "svr": suggest_svr,
}
COMPLEXITY = {
    "xgb_ordinal": xgb_complexity,
    "catboost": catboost_complexity,
    "corn": corn_complexity,
    "svr": svr_complexity,
}


def _jsonable(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def make_model(name: str, params: dict, cfg: dict, seed: int, feature_names: list[str] | None = None):
    params = dict(params)
    if name == "xgb_ordinal":
        return XGBOrdinalRegressor(
            seed=seed,
            n_jobs=int(cfg.get("n_jobs", 1)),
            feature_names=feature_names,
            **params,
        )
    if name == "catboost":
        return make_catboost(
            seed=seed,
            thread_count=int(cfg.get("n_jobs", 1)),
            feature_names=feature_names,
            **params,
        )
    if name == "corn":
        params.pop("n_layers", None)
        params.pop("units", None)
        hidden = params.get("hidden")
        if isinstance(hidden, list):
            params["hidden"] = tuple(hidden)
        return CornRegressor(
            seed=seed,
            feature_names=feature_names,
            max_epochs=int(cfg["corn"]["max_epochs"]),
            patience=int(cfg["corn"]["patience"]),
            **params,
        )
    if name == "svr":
        return svr_pipeline(seed=seed, feature_names=feature_names, **params)
    raise KeyError(name)


def _fit(model, x, y, groups):
    if isinstance(model, CornRegressor):
        model.fit(x, y, groups=groups)
    else:
        model.fit(x, y)
    return model


def _folds(x, y, groups, n_splits: int, seed: int):
    stratum = stratify_label(y)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(x, stratum, groups))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(x, y, groups))


def select_trial(records: list[dict]) -> dict:
    """Mažiausias MAE; jei MAE skiriasi < 0,01, didesnė κw ir paprastesnis modelis."""
    usable = [row for row in records if row.get("mae", 1e9) < 10]
    if not usable:
        raise RuntimeError("Hiperparametrų paieška neturi sėkmingų bandymų")
    best_mae = min(row["mae"] for row in usable)
    pool = [row for row in usable if row["mae"] <= best_mae + 0.01]
    pool.sort(key=lambda row: (-row["kappa"], row["complexity"], row["number"]))
    return pool[0]


def tune_model(name: str, x, y, groups, cfg: dict, seed: int, feature_names: list[str] | None = None) -> dict:
    n_trials = int(cfg["trials"][name])
    n_splits = int(cfg["inner_splits"])

    def objective(trial: optuna.Trial) -> float:
        params = SUGGEST[name](trial)
        fold_mae = []
        fold_kappa = []
        for train_idx, valid_idx in _folds(x, y, groups, n_splits, seed):
            model = make_model(name, params, cfg, seed, feature_names)
            _fit(model, x[train_idx], y[train_idx], groups[train_idx])
            pred = model.predict(x[valid_idx])
            fold_mae.append(mae(y[valid_idx], pred))
            fold_kappa.append(quadratic_weighted_kappa(y[valid_idx], pred))
        mean_mae = float(sum(fold_mae) / len(fold_mae))
        mean_kappa = float(sum(fold_kappa) / len(fold_kappa))
        trial.set_user_attr("kappa", mean_kappa)
        trial.set_user_attr("complexity", COMPLEXITY[name](params))
        trial.set_user_attr("params", _jsonable(params))
        trial.set_user_attr("mae_folds", fold_mae)
        return mean_mae

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        records.append(
            {
                "number": trial.number,
                "mae": float(trial.value),
                "kappa": float(trial.user_attrs.get("kappa", 0.0)),
                "complexity": float(trial.user_attrs.get("complexity", 0.0)),
                "params": trial.user_attrs.get("params", {}),
                "mae_folds": trial.user_attrs.get("mae_folds", []),
            }
        )
    chosen = select_trial(records)
    model = make_model(name, chosen["params"], cfg, seed, feature_names)
    _fit(model, x, y, groups)
    return {
        "model": model,
        "chosen": chosen,
        "trials": records,
        "test_used": False,
        "n_train": int(len(y)),
        "n_trials": n_trials,
        "inner_splits": n_splits,
    }
