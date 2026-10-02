"""Abliacijos A1–A7. Jos nekeičia jau pasirinkto modelio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from winequality.config import FEATURE_GROUPS, FEATURE_NAMES, RANKS, TARGET, TYPE_COL
from winequality.metrics import evaluate
from winequality.transform import TrainPreprocessor
from winequality.tune import _fit, make_model


def _score(model, x, y) -> dict:
    pred = np.asarray(model.predict(x), dtype=float)
    proba = model.predict_proba(x) if hasattr(model, "predict_proba") and type(model).__name__ in {
        "XGBOrdinalRegressor",
        "CornRegressor",
    } else None
    out = evaluate(y, pred, proba)
    return {"mae": out["mae"], "kappa_w": out["kappa_w"], "mae_rounded": out["mae_rounded"]}


def _tree_matrix(x_train, x_test):
    prep = TrainPreprocessor(log1p=False, scale=False).fit(x_train)
    return prep.transform(x_train), prep.transform(x_test)


def ablation_a1(params: dict, cfg: dict, seed: int, x_train, y_train, x_test, y_test) -> dict:
    """Ta pati medžių parametrų aibė: ordinalinis, nominalus klasifikatorius, regresorius."""
    ordinal = make_model("xgb_ordinal", params, cfg, seed)
    _fit(ordinal, x_train, y_train, np.arange(len(y_train)))
    z_train, z_test = _tree_matrix(x_train, x_test)
    y_index = np.clip(np.rint(y_train), RANKS[0], RANKS[-1]).astype(int) - RANKS[0]
    clf = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        objective="multi:softprob",
        random_state=seed,
        n_jobs=int(cfg.get("n_jobs", 1)),
        tree_method="hist",
        verbosity=0,
    )
    clf.fit(z_train, y_index)
    proba = clf.predict_proba(z_test)
    full = np.zeros((len(z_test), len(RANKS)))
    for column, cls in enumerate(clf.classes_):
        full[:, int(cls)] = proba[:, column]
    class_score = full @ np.asarray(RANKS, dtype=float)
    reg = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=seed,
        n_jobs=int(cfg.get("n_jobs", 1)),
        tree_method="hist",
        verbosity=0,
    )
    reg.fit(z_train, y_train)
    return {
        "xgb_ordinal": _score(ordinal, x_test, y_test),
        "xgb_classifier": evaluate(y_test, class_score),
        "xgb_regressor": evaluate(y_test, reg.predict(z_test)),
    }


def ablation_a4(params: dict, cfg: dict, seed: int, x_train, y_train, x_test, y_test) -> dict:
    chosen = make_model("svr", params, cfg, seed)
    zero = make_model("svr", {**params, "epsilon": 0.0}, cfg, seed)
    for model in (chosen, zero):
        _fit(model, x_train, y_train, np.arange(len(y_train)))
    return {"epsilon_selected": _score(chosen, x_test, y_test), "epsilon_0": _score(zero, x_test, y_test)}


def ablation_a5(name: str, params: dict, cfg: dict, seed: int, x_train, y_train, x_test, y_test) -> dict:
    raw = make_model(name, params, cfg, seed)
    _fit(raw, x_train, y_train, np.arange(len(y_train)))
    prep = TrainPreprocessor(log1p=True, scale=True).fit(x_train)
    scaled = make_model(name, params, cfg, seed)
    _fit(scaled, prep.transform(x_train), y_train, np.arange(len(y_train)))
    return {
        "raw_scale": _score(raw, x_test, y_test),
        "log1p_standard": _score(scaled, prep.transform(x_test), y_test),
    }


def ablation_a6(params: dict, cfg: dict, seed: int, frames: dict[str, pd.DataFrame], splits: dict) -> dict:
    pieces = []
    for kind, frame in frames.items():
        part = frame.copy()
        part["group"] = kind + ":" + part["group"].astype(str)
        train_idx, test_idx = splits[kind]
        part["_split"] = "train"
        part.iloc[test_idx, part.columns.get_loc("_split")] = "test"
        pieces.append(part)
    del train_idx
    combined = pd.concat(pieces, ignore_index=True)
    columns = FEATURE_NAMES + [TYPE_COL]
    train = combined[combined["_split"] == "train"]
    test = combined[combined["_split"] == "test"]
    model = make_model("xgb_ordinal", params, cfg, seed, feature_names=columns)
    _fit(model, train[columns].to_numpy(float), train[TARGET].to_numpy(float), train["group"].to_numpy())
    pred = model.predict(test[columns].to_numpy(float))
    per_type = {}
    for kind, code in (("red", 0), ("white", 1)):
        mask = test[TYPE_COL].to_numpy() == code
        per_type[kind] = evaluate(test.loc[mask, TARGET].to_numpy(float), pred[mask])
    per_type["all"] = evaluate(test[TARGET].to_numpy(float), pred)
    return per_type


def ablation_a7(params: dict, cfg: dict, seed: int, x_train, y_train, x_test, y_test) -> dict:
    out = {}
    for group_name, columns in FEATURE_GROUPS.items():
        keep = [name for name in FEATURE_NAMES if name not in columns]
        idx = [FEATURE_NAMES.index(name) for name in keep]
        model = make_model("xgb_ordinal", params, cfg, seed, feature_names=keep)
        _fit(model, x_train[:, idx], y_train, np.arange(len(y_train)))
        out[group_name] = _score(model, x_test[:, idx], y_test)
    return out


def slim(metrics: dict) -> dict:
    if "mae" in metrics and "confusion_score" in metrics:
        return {key: metrics[key] for key in ("mae", "mae_rounded", "kappa_w", "macro_f1")}
    return {key: slim(value) if isinstance(value, dict) else value for key, value in metrics.items()}
