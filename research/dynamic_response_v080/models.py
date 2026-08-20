from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import output_dir
from .data import WalkForwardFold, select_fold_rows
from .features import STAGE_A_CATEGORICAL_FEATURES, STAGE_A_NUMERIC_FEATURES


LOGGER = logging.getLogger(__name__)


@dataclass
class FittedSurfaceModel:
    name: str
    preprocessor: Any
    estimator: Any
    feature_columns: list[str]
    categorical_columns: list[str]
    best_iteration: int | None
    runtime_seconds: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame[self.feature_columns])
        return np.asarray(self.estimator.predict(matrix), dtype=float)


def _linear_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _tree_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def deterministic_row_sample(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0 or len(frame) <= cap:
        return frame
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), size=cap, replace=False))
    return frame.iloc[positions].copy()


def _valid_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    mask = pd.to_numeric(frame[target], errors="coerce").notna()
    return frame.loc[mask].copy()


def _inner_early_stop_split(
    frame: pd.DataFrame, days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(frame["trade_date"]).dt.normalize().unique())
    if len(dates) <= days + 5:
        split = max(1, int(len(dates) * 0.8))
        early_dates = dates[split:]
    else:
        early_dates = dates[-days:]
    early = frame[frame["trade_date"].isin(early_dates)].copy()
    core = frame[~frame["trade_date"].isin(early_dates)].copy()
    if core.empty or early.empty:
        raise ValueError("Cannot construct chronological inner early-stopping split")
    return core, early


def fit_ridge_surface(
    train: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, Any],
) -> FittedSurfaceModel:
    started = time.perf_counter()
    features = numeric + categorical
    preprocessor = _linear_preprocessor(numeric, categorical)
    matrix = preprocessor.fit_transform(train[features])
    estimator = Ridge(
        alpha=float(config["surface_models"]["ridge"]["alpha"]),
        solver="lsqr",
    )
    estimator.fit(matrix, pd.to_numeric(train[target], errors="coerce").to_numpy())
    return FittedSurfaceModel(
        name="ridge",
        preprocessor=preprocessor,
        estimator=estimator,
        feature_columns=features,
        categorical_columns=categorical,
        best_iteration=None,
        runtime_seconds=time.perf_counter() - started,
    )


def _lightgbm_estimator(
    config: dict[str, Any], objective: str, alpha: float | None, n_estimators: int
):
    import lightgbm as lgb

    params = dict(config["surface_models"]["lightgbm"])
    params["objective"] = objective
    params["n_estimators"] = int(n_estimators)
    params["random_state"] = int(config["strategy"]["random_seed"])
    params["n_jobs"] = int(config["strategy"]["n_jobs"])
    params["verbosity"] = -1
    params["deterministic"] = True
    params["force_col_wise"] = True
    if alpha is not None:
        params["alpha"] = float(alpha)
    return lgb.LGBMRegressor(**params)


def fit_lightgbm_surface(
    train: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, Any],
    objective: str = "regression_l1",
    alpha: float | None = None,
    name: str = "lightgbm_mean",
) -> FittedSurfaceModel:
    import lightgbm as lgb

    started = time.perf_counter()
    features = numeric + categorical
    early_days = int(config["surface_models"]["early_stopping_days"])
    core, early = _inner_early_stop_split(train, early_days)
    early_preprocessor = _tree_preprocessor(numeric, categorical)
    x_core = early_preprocessor.fit_transform(core[features])
    x_early = early_preprocessor.transform(early[features])
    estimator = _lightgbm_estimator(
        config,
        objective=objective,
        alpha=alpha,
        n_estimators=int(config["surface_models"]["lightgbm"]["n_estimators"]),
    )
    estimator.fit(
        x_core,
        pd.to_numeric(core[target], errors="coerce"),
        eval_X=x_early,
        eval_y=pd.to_numeric(early[target], errors="coerce"),
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iteration = max(int(estimator.best_iteration_ or 1), 10)
    preprocessor = _tree_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(train[features])
    estimator = _lightgbm_estimator(
        config, objective=objective, alpha=alpha, n_estimators=best_iteration
    )
    estimator.fit(x_train, pd.to_numeric(train[target], errors="coerce"))
    return FittedSurfaceModel(
        name=name,
        preprocessor=preprocessor,
        estimator=estimator,
        feature_columns=features,
        categorical_columns=categorical,
        best_iteration=best_iteration,
        runtime_seconds=time.perf_counter() - started,
    )


def fit_xgboost_surface(
    train: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, Any],
) -> FittedSurfaceModel:
    import xgboost as xgb

    started = time.perf_counter()
    features = numeric + categorical
    core, early = _inner_early_stop_split(
        train, int(config["surface_models"]["early_stopping_days"])
    )
    early_preprocessor = _tree_preprocessor(numeric, categorical)
    x_core = early_preprocessor.fit_transform(core[features])
    x_early = early_preprocessor.transform(early[features])
    params = dict(config["surface_models"]["xgboost"])
    params.update(
        {
            "objective": "reg:squarederror",
            "random_state": int(config["strategy"]["random_seed"]),
            "n_jobs": int(config["strategy"]["n_jobs"]),
            "early_stopping_rounds": 50,
        }
    )
    estimator = xgb.XGBRegressor(**params)
    estimator.fit(
        x_core,
        pd.to_numeric(core[target], errors="coerce"),
        eval_set=[(x_early, pd.to_numeric(early[target], errors="coerce"))],
        verbose=False,
    )
    best_iteration = max(int(getattr(estimator, "best_iteration", 0)) + 1, 10)
    params["n_estimators"] = best_iteration
    params.pop("early_stopping_rounds", None)
    preprocessor = _tree_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(train[features])
    estimator = xgb.XGBRegressor(**params)
    estimator.fit(x_train, pd.to_numeric(train[target], errors="coerce"), verbose=False)
    return FittedSurfaceModel(
        name="xgboost_mean",
        preprocessor=preprocessor,
        estimator=estimator,
        feature_columns=features,
        categorical_columns=categorical,
        best_iteration=best_iteration,
        runtime_seconds=time.perf_counter() - started,
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 2:
        return {"sample_count": int(valid.sum()), "mae": np.nan, "rmse": np.nan, "spearman": np.nan}
    actual_valid = actual[valid]
    predicted_valid = predicted[valid]
    correlation = spearmanr(actual_valid, predicted_valid).statistic
    return {
        "sample_count": int(valid.sum()),
        "mae": float(mean_absolute_error(actual_valid, predicted_valid)),
        "rmse": float(np.sqrt(mean_squared_error(actual_valid, predicted_valid))),
        "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
        "directional_accuracy": float(np.mean(np.sign(actual_valid) == np.sign(predicted_valid))),
    }


def enforce_quantile_order(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort row-wise q10/q50/q90 predictions and retain a crossing flag."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Quantile matrix must have shape (n_samples, 3)")
    crossing = (matrix[:, 0] > matrix[:, 1]) | (matrix[:, 1] > matrix[:, 2])
    return np.sort(matrix, axis=1), crossing


def _metric_record(
    fold: WalkForwardFold,
    name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    runtime_seconds: float,
    best_iteration: int | None,
    train_rows: int,
) -> dict[str, Any]:
    return {
        "fold_id": fold.fold_id,
        "fold_role": fold.role,
        "official": fold.official,
        "model": name,
        "target": "surface_target_cb_return",
        "train_start": str(fold.train_start.date()),
        "train_end": str(fold.train_end.date()),
        "validation_start": str(fold.validation_start.date()),
        "validation_end": str(fold.validation_end.date()),
        "train_rows": train_rows,
        "runtime_seconds": runtime_seconds,
        "best_iteration": best_iteration,
        **regression_metrics(actual, predicted),
    }


def _save_model(model: FittedSurfaceModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def fit_predict_surface_folds(
    frame: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: dict[str, Any],
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate strict chronological OOF predictions for every surface model."""
    requested = set(model_names or config["surface_models"]["enabled"])
    target = "surface_target_cb_return"
    numeric = [column for column in STAGE_A_NUMERIC_FEATURES if column in frame.columns]
    categorical = [column for column in STAGE_A_CATEGORICAL_FEATURES if column in frame.columns]
    id_columns = [
        "trade_date", "bar_start", "bar_end", "bar_slot", "session_part",
        "bond_code", "stock_code", "window_id", "window_role",
        "research_pool_scope", "cb_close", "stock_close", target,
        "baseline_v071_prediction", "baseline_fair_prediction",
    ]
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    artifacts = output_dir(config) / "models" / "surface"
    seed = int(config["strategy"]["random_seed"])
    train_cap = int(config["surface_models"]["training_row_cap"])

    for fold_index, fold in enumerate(folds):
        raw_train, validation = select_fold_rows(frame, fold)
        train = deterministic_row_sample(_valid_target(raw_train, target), train_cap, seed + fold_index)
        validation_valid = _valid_target(validation, target)
        if train.empty or validation_valid.empty:
            raise ValueError(f"No target rows for surface fold {fold.fold_id}")
        fold_pred = validation_valid[id_columns].copy()
        fold_pred["fold_id"] = fold.fold_id
        fold_pred["fold_role"] = fold.role
        fold_pred["official"] = fold.official
        fold_pred["train_start"] = fold.train_start
        fold_pred["train_end"] = fold.train_end
        fold_pred["validation_start"] = fold.validation_start
        fold_pred["validation_end"] = fold.validation_end
        fold_pred["purge_days"] = fold.purge_days
        fold_pred["purge_bars"] = int(config["walk_forward"]["maximum_future_horizon_bars"])
        fold_pred["model_seed"] = seed
        actual = pd.to_numeric(validation_valid[target], errors="coerce").to_numpy(dtype=float)

        for baseline_name, baseline_column in [
            ("v071_linear", "baseline_v071_prediction"),
            ("v071_fair_linear", "baseline_fair_prediction"),
        ]:
            values = pd.to_numeric(validation_valid[baseline_column], errors="coerce").to_numpy(dtype=float)
            fold_pred[f"prediction_{baseline_name}"] = values.astype("float32")
            metric_rows.append(
                _metric_record(fold, baseline_name, actual, values, 0.0, None, len(train))
            )

        fitted: list[FittedSurfaceModel] = []
        if "ridge" in requested:
            fitted.append(fit_ridge_surface(train, target, numeric, categorical, config))
        if "lightgbm" in requested:
            fitted.append(
                fit_lightgbm_surface(
                    train, target, numeric, categorical, config, name="lightgbm_mean"
                )
            )
            for quantile in config["surface_models"]["quantiles"]:
                fitted.append(
                    fit_lightgbm_surface(
                        train,
                        target,
                        numeric,
                        categorical,
                        config,
                        objective="quantile",
                        alpha=float(quantile),
                        name=f"lightgbm_q{int(float(quantile) * 100):02d}",
                    )
                )
        if "xgboost" in requested:
            fitted.append(fit_xgboost_surface(train, target, numeric, categorical, config))
        if config["features"].get("include_bond_id_comparison", False) and "lightgbm" in requested:
            fitted.append(
                fit_lightgbm_surface(
                    train,
                    target,
                    numeric,
                    categorical + ["bond_code"],
                    config,
                    objective="quantile",
                    alpha=0.5,
                    name="lightgbm_q50_with_bond_id",
                )
            )

        for model in fitted:
            values = model.predict(validation_valid)
            fold_pred[f"prediction_{model.name}"] = values.astype("float32")
            metric_rows.append(
                _metric_record(
                    fold,
                    model.name,
                    actual,
                    values,
                    model.runtime_seconds,
                    model.best_iteration,
                    len(train),
                )
            )
            estimator = model.estimator
            if hasattr(estimator, "feature_importances_"):
                names = model.preprocessor.get_feature_names_out()
                for feature_name, importance in zip(names, estimator.feature_importances_, strict=False):
                    importance_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "model": model.name,
                            "feature": str(feature_name),
                            "importance": float(importance),
                        }
                    )
            _save_model(model, artifacts / fold.fold_id / f"{model.name}.joblib")

        quantile_columns = ["prediction_lightgbm_q10", "prediction_lightgbm_q50", "prediction_lightgbm_q90"]
        if all(column in fold_pred.columns for column in quantile_columns):
            raw_quantiles = fold_pred[quantile_columns].to_numpy(dtype=float)
            ordered, crossing = enforce_quantile_order(raw_quantiles)
            fold_pred["quantile_crossing_raw"] = crossing
            for index, column in enumerate(quantile_columns):
                fold_pred[column] = ordered[:, index].astype("float32")
            fold_pred["prediction_interval_width"] = (
                fold_pred["prediction_lightgbm_q90"] - fold_pred["prediction_lightgbm_q10"]
            ).astype("float32")
        predictions.append(fold_pred)
        LOGGER.info(
            "surface fold %s complete train_rows=%d validation_rows=%d",
            fold.fold_id,
            len(train),
            len(validation_valid),
        )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(importance_rows),
    )


def quantile_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    required = [
        "surface_target_cb_return",
        "prediction_lightgbm_q10",
        "prediction_lightgbm_q50",
        "prediction_lightgbm_q90",
    ]
    if not all(column in oof.columns for column in required):
        return pd.DataFrame()
    rows = []
    for fold_id, group in oof.groupby("fold_id", sort=False):
        actual = pd.to_numeric(group["surface_target_cb_return"], errors="coerce")
        q10 = pd.to_numeric(group["prediction_lightgbm_q10"], errors="coerce")
        q50 = pd.to_numeric(group["prediction_lightgbm_q50"], errors="coerce")
        q90 = pd.to_numeric(group["prediction_lightgbm_q90"], errors="coerce")
        valid = actual.notna() & q10.notna() & q50.notna() & q90.notna()
        rows.append(
            {
                "fold_id": fold_id,
                "sample_count": int(valid.sum()),
                "q10_pinball": mean_pinball_loss(actual[valid], q10[valid], alpha=0.1),
                "q50_pinball": mean_pinball_loss(actual[valid], q50[valid], alpha=0.5),
                "q90_pinball": mean_pinball_loss(actual[valid], q90[valid], alpha=0.9),
                "q10_empirical_coverage": float((actual[valid] <= q10[valid]).mean()),
                "q50_empirical_coverage": float((actual[valid] <= q50[valid]).mean()),
                "q90_empirical_coverage": float((actual[valid] <= q90[valid]).mean()),
                "q10_q90_interval_coverage": float(((actual[valid] >= q10[valid]) & (actual[valid] <= q90[valid])).mean()),
                "mean_interval_width": float((q90[valid] - q10[valid]).mean()),
                "raw_crossing_rate": float(group.get("quantile_crossing_raw", False).mean()),
            }
        )
    return pd.DataFrame(rows)


def tune_lightgbm_with_optuna(
    train: pd.DataFrame,
    config: dict[str, Any],
    trials: int | None = None,
) -> dict[str, Any]:
    """Tune only inside the supplied training period using chronological holdout."""
    import lightgbm as lgb
    import optuna

    target = "surface_target_cb_return"
    numeric = [column for column in STAGE_A_NUMERIC_FEATURES if column in train.columns]
    categorical = [column for column in STAGE_A_CATEGORICAL_FEATURES if column in train.columns]
    features = numeric + categorical
    core, validation = _inner_early_stop_split(
        _valid_target(train, target), int(config["surface_models"]["early_stopping_days"])
    )
    preprocessor = _tree_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(core[features])
    x_validation = preprocessor.transform(validation[features])
    y_train = pd.to_numeric(core[target], errors="coerce")
    y_validation = pd.to_numeric(validation[target], errors="coerce")
    seed = int(config["strategy"]["random_seed"])

    def objective(trial: Any) -> float:
        estimator = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=500,
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 250),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            random_state=seed,
            n_jobs=int(config["strategy"]["n_jobs"]),
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        estimator.fit(
            x_train,
            y_train,
            eval_X=x_validation,
            eval_y=y_validation,
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        return float(mean_absolute_error(y_validation, estimator.predict(x_validation)))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(trials or config["surface_models"]["optuna_trials"]))
    output = output_dir(config) / "optuna"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "trial_count": len(study.trials),
        "train_start": str(core["trade_date"].min()),
        "train_end": str(core["trade_date"].max()),
        "validation_start": str(validation["trade_date"].min()),
        "validation_end": str(validation["trade_date"].max()),
        "seed": seed,
    }
    (output / "lightgbm_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
