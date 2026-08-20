from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from .config import output_dir
from .data import WalkForwardFold
from .features import (
    BASELINE_OUTCOME_FEATURES,
    STAGE_A_CATEGORICAL_FEATURES,
    STAGE_A_NUMERIC_FEATURES,
)
from .models import _linear_preprocessor, deterministic_row_sample


LOGGER = logging.getLogger(__name__)
KEYS = ["trade_date", "bar_start", "bond_code", "stock_code"]


def merge_surface_oof(dataset: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = [
        column
        for column in oof.columns
        if column.startswith("prediction_")
        or column in {
            "fold_id", "fold_role", "official", "train_start", "train_end",
            "validation_start", "validation_end", "purge_days", "purge_bars",
            "model_seed", "prediction_interval_width", "quantile_crossing_raw",
        }
    ]
    right = oof[KEYS + prediction_columns].drop_duplicates(KEYS, keep="last")
    return dataset.merge(right, on=KEYS, how="inner", validate="one_to_one")


def _state_for_fold(
    state_frame: pd.DataFrame,
    fold_id: str,
    role: str,
) -> pd.DataFrame:
    selected = state_frame[
        state_frame["fold_id"].eq(fold_id) & state_frame["state_data_role"].eq(role)
    ].copy()
    return selected.drop_duplicates(KEYS, keep="last")


def _attach_state(
    frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    fold_id: str,
    role: str,
) -> pd.DataFrame:
    selected = _state_for_fold(state_frame, fold_id, role)
    columns = [
        column
        for column in selected.columns
        if column not in {"fold_id", "trade_date", "bar_start", "bond_code", "stock_code", "bar_slot", "session_part"}
    ]
    return frame.merge(selected[KEYS + columns], on=KEYS, how="left", validate="one_to_one")


def outcome_feature_sets(frame: pd.DataFrame, state_count: int) -> dict[str, tuple[list[str], list[str]]]:
    baseline_numeric = [column for column in BASELINE_OUTCOME_FEATURES if column in frame.columns]
    baseline_categorical = ["session_part"] if "session_part" in frame.columns else []
    surface_extra = [
        "prediction_lightgbm_mean",
        "prediction_lightgbm_q10",
        "prediction_lightgbm_q50",
        "prediction_lightgbm_q90",
        "prediction_xgboost_mean",
        "prediction_ridge",
        "prediction_interval_width",
        "response_gap",
        "response_gap_z",
        "response_gap_change",
    ]
    surface_numeric = list(
        dict.fromkeys(
            baseline_numeric
            + [column for column in STAGE_A_NUMERIC_FEATURES if column in frame.columns]
            + [column for column in surface_extra if column in frame.columns]
        )
    )
    surface_categorical = [
        column for column in STAGE_A_CATEGORICAL_FEATURES if column in frame.columns
    ]
    state_numeric = [f"state_probability_{index}" for index in range(state_count)] + [
        "state_argmax_probability",
        "state_entropy",
        "state_expected_duration",
    ]
    return {
        "A_BASELINE": (baseline_numeric, baseline_categorical),
        "B_SURFACE": (surface_numeric, surface_categorical),
        "C_SURFACE_STATE": (
            surface_numeric + [column for column in state_numeric if column in frame.columns],
            surface_categorical,
        ),
    }


def _regression_metric_rows(
    fold: WalkForwardFold,
    model_name: str,
    horizon: str,
    actual: pd.DataFrame,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(actual.columns):
        truth = pd.to_numeric(actual[target], errors="coerce").to_numpy(dtype=float)
        estimate = np.asarray(predicted[:, index], dtype=float)
        valid = np.isfinite(truth) & np.isfinite(estimate)
        correlation = spearmanr(truth[valid], estimate[valid]).statistic if valid.sum() > 2 else np.nan
        rows.append(
            {
                "fold_id": fold.fold_id,
                "fold_role": fold.role,
                "model": model_name,
                "horizon": horizon,
                "target": target,
                "sample_count": int(valid.sum()),
                "mae": float(mean_absolute_error(truth[valid], estimate[valid])) if valid.any() else np.nan,
                "rmse": float(np.sqrt(mean_squared_error(truth[valid], estimate[valid]))) if valid.any() else np.nan,
                "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
                "directional_accuracy": float(np.mean(np.sign(truth[valid]) == np.sign(estimate[valid]))) if valid.any() else np.nan,
            }
        )
    return rows


def _classification_metric(
    fold: WalkForwardFold,
    model_name: str,
    horizon: str,
    actual: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(actual) & np.isfinite(probability)
    truth = actual[valid].astype(int)
    estimate = probability[valid]
    both_classes = len(np.unique(truth)) == 2
    return {
        "fold_id": fold.fold_id,
        "fold_role": fold.role,
        "model": model_name,
        "horizon": horizon,
        "target": f"gap_significant_closure_{horizon}",
        "sample_count": int(valid.sum()),
        "positive_rate": float(truth.mean()) if len(truth) else np.nan,
        "brier_score": float(brier_score_loss(truth, estimate)) if len(truth) else np.nan,
        "roc_auc": float(roc_auc_score(truth, estimate)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(truth, estimate)) if both_classes else np.nan,
        "accuracy": float(accuracy_score(truth, estimate >= 0.5)) if len(truth) else np.nan,
    }


def fit_predict_outcome_folds(
    frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: dict[str, Any],
    state_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Model A/B/C on identical chronological folds and labels."""
    official_folds = [fold for fold in folds if fold.official]
    horizon_bars = [int(value) for value in config["labels"]["horizons_bars"]]
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    model_dir = output_dir(config) / "models" / "outcome"
    model_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["strategy"]["random_seed"])
    cap = int(config["outcome_models"]["training_row_cap"])

    for fold_index, fold in enumerate(official_folds):
        training = frame[frame["trade_date"] <= fold.train_end].copy()
        validation = frame[
            frame["trade_date"].between(fold.validation_start, fold.validation_end)
        ].copy()
        training = _attach_state(training, state_frame, fold.fold_id, "train")
        validation = _attach_state(validation, state_frame, fold.fold_id, "validation")
        validation_output = validation[KEYS + ["fold_id", "fold_role", "window_id", "window_role"]].copy()
        feature_sets = outcome_feature_sets(training, state_count)

        for model_name, (numeric, categorical) in feature_sets.items():
            if model_name == "C_SURFACE_STATE":
                required_state = "state_probability_0"
                training_model = training[training[required_state].notna()].copy()
                validation_model = validation[validation[required_state].notna()].copy()
            else:
                training_model = training.copy()
                validation_model = validation.copy()
            training_model = deterministic_row_sample(training_model, cap, seed + fold_index)
            if training_model.empty or validation_model.empty:
                continue
            preprocessor = _linear_preprocessor(numeric, categorical)
            x_train_all = preprocessor.fit_transform(training_model[numeric + categorical])
            x_validation_all = preprocessor.transform(validation_model[numeric + categorical])
            training_positions = pd.Series(np.arange(len(training_model)), index=training_model.index)
            validation_positions = pd.Series(np.arange(len(validation_model)), index=validation_model.index)

            for bars in horizon_bars:
                horizon = f"{bars * 5}m"
                continuous_targets = [
                    f"forward_cb_return_{horizon}",
                    f"forward_residual_return_{horizon}",
                    f"gap_abs_reduction_{horizon}",
                    f"gap_close_ratio_clipped_{horizon}",
                    f"mfe_{horizon}",
                    f"mae_{horizon}",
                ]
                train_valid = training_model[continuous_targets].notna().all(axis=1)
                validation_valid = validation_model[continuous_targets].notna().all(axis=1)
                if train_valid.sum() < 100 or validation_valid.sum() == 0:
                    continue
                train_pos = training_positions.loc[train_valid].to_numpy(dtype=int)
                validation_pos = validation_positions.loc[validation_valid].to_numpy(dtype=int)
                estimator = Ridge(
                    alpha=float(config["outcome_models"]["ridge_alpha"]), solver="lsqr"
                )
                estimator.fit(
                    x_train_all[train_pos],
                    training_model.loc[train_valid, continuous_targets].to_numpy(dtype=float),
                )
                predictions = estimator.predict(x_validation_all[validation_pos])
                metric_rows.extend(
                    _regression_metric_rows(
                        fold,
                        model_name,
                        horizon,
                        validation_model.loc[validation_valid, continuous_targets],
                        predictions,
                    )
                )
                for target_index, target in enumerate(continuous_targets):
                    column = f"{model_name}_prediction_{target}"
                    if column not in validation_output.columns:
                        validation_output[column] = np.nan
                    validation_output.loc[
                        validation_model.index[validation_valid], column
                    ] = predictions[:, target_index]

                classification_target = f"gap_significant_closure_{horizon}"
                available = f"label_available_{horizon}"
                train_class = (
                    training_model[available].fillna(False).astype(bool)
                    & training_model[classification_target].notna()
                )
                validation_class = (
                    validation_model[available].fillna(False).astype(bool)
                    & validation_model[classification_target].notna()
                )
                if train_class.sum() >= 100 and training_model.loc[train_class, classification_target].nunique() == 2:
                    classifier = LogisticRegression(
                        C=float(config["outcome_models"]["logistic_c"]),
                        class_weight="balanced",
                        max_iter=300,
                        solver="liblinear",
                        random_state=seed,
                    )
                    train_class_pos = training_positions.loc[train_class].to_numpy(dtype=int)
                    classifier.fit(
                        x_train_all[train_class_pos],
                        training_model.loc[train_class, classification_target].astype(int),
                    )
                    probability_column = f"{model_name}_probability_{classification_target}"
                    if probability_column not in validation_output.columns:
                        validation_output[probability_column] = np.nan
                    if validation_class.any():
                        validation_class_pos = validation_positions.loc[validation_class].to_numpy(dtype=int)
                        probability = classifier.predict_proba(x_validation_all[validation_class_pos])[:, 1]
                        validation_output.loc[
                            validation_model.index[validation_class], probability_column
                        ] = probability
                        metric_rows.append(
                            _classification_metric(
                                fold,
                                model_name,
                                horizon,
                                validation_model.loc[validation_class, classification_target].astype(float).to_numpy(),
                                probability,
                            )
                        )
                joblib.dump(
                    {
                        "preprocessor": preprocessor,
                        "regression": estimator,
                        "features": numeric + categorical,
                        "targets": continuous_targets,
                        "fold_id": fold.fold_id,
                    },
                    model_dir / f"{fold.fold_id}_{model_name}_{horizon}.joblib",
                    compress=3,
                )
        actual_columns = [
            column
            for column in validation.columns
            if column.startswith("forward_")
            or column.startswith("mfe_")
            or column.startswith("mae_")
            or column.startswith("gap_")
            or column.startswith("lagger_")
            or column.startswith("leader_")
        ]
        validation_output = validation_output.merge(
            validation[KEYS + actual_columns].drop_duplicates(KEYS),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        prediction_frames.append(validation_output)
        LOGGER.info("outcome fold %s complete rows=%d", fold.fold_id, len(validation_output))
    if not prediction_frames:
        raise RuntimeError("No forward outcome predictions were generated")
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(metric_rows)


def aggregate_model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    numeric = [
        column
        for column in [
            "sample_count", "mae", "rmse", "spearman", "directional_accuracy",
            "positive_rate", "brier_score", "roc_auc", "pr_auc", "accuracy",
        ]
        if column in metrics.columns
    ]
    return (
        metrics.groupby(["model", "horizon", "target"], dropna=False, as_index=False)[numeric]
        .mean(numeric_only=True)
        .sort_values(["target", "horizon", "model"])
    )


def state_conditioned_outcomes(
    frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    validation_state = state_frame[state_frame["state_data_role"].eq("validation")].copy()
    merged = frame.merge(
        validation_state.drop_duplicates(KEYS)[
            KEYS + ["canonical_state_id", "state_semantic_label", "state_entropy"]
        ],
        on=KEYS,
        how="inner",
    )
    rows = []
    for bars in config["labels"]["horizons_bars"]:
        horizon = f"{int(bars) * 5}m"
        grouped = merged.groupby(
            ["canonical_state_id", "state_semantic_label"], dropna=False
        )
        summary = grouped.agg(
            sample_count=("bond_code", "size"),
            bond_count=("bond_code", "nunique"),
            trading_days=("trade_date", "nunique"),
            forward_return=(f"forward_cb_return_{horizon}", "mean"),
            residual_return=(f"forward_residual_return_{horizon}", "mean"),
            gap_close_ratio=(f"gap_close_ratio_clipped_{horizon}", "mean"),
            mfe=(f"mfe_{horizon}", "mean"),
            mae=(f"mae_{horizon}", "mean"),
            leader_reversal_rate=(f"leader_reversal_flag_{horizon}", "mean"),
            lagger_catchup_rate=(f"lagger_catchup_flag_{horizon}", "mean"),
            state_entropy=("state_entropy", "mean"),
        ).reset_index()
        summary["horizon"] = horizon
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def response_gap_decile_outcomes(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame[frame["response_gap"].notna()].copy()
    out["response_gap_decile"] = out.groupby("fold_id", observed=True)["response_gap"].transform(
        lambda values: pd.qcut(values.rank(method="first"), int(config["evaluation"]["response_gap_bins"]), labels=False, duplicates="drop")
    )
    rows = []
    for bars in config["labels"]["horizons_bars"]:
        horizon = f"{int(bars) * 5}m"
        summary = out.groupby("response_gap_decile", dropna=False).agg(
            sample_count=("bond_code", "size"),
            response_gap=("response_gap", "mean"),
            forward_return=(f"forward_cb_return_{horizon}", "mean"),
            residual_return=(f"forward_residual_return_{horizon}", "mean"),
            gap_close_ratio=(f"gap_close_ratio_clipped_{horizon}", "mean"),
            leader_reversal_rate=(f"leader_reversal_flag_{horizon}", "mean"),
            lagger_catchup_rate=(f"lagger_catchup_flag_{horizon}", "mean"),
            mfe=(f"mfe_{horizon}", "mean"),
            mae=(f"mae_{horizon}", "mean"),
        ).reset_index()
        summary["horizon"] = horizon
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _fold_quantile_bucket(
    frame: pd.DataFrame,
    column: str,
    prefix: str,
    quantiles: int = 4,
) -> pd.Series:
    result = pd.Series("MISSING", index=frame.index, dtype="object")
    if column not in frame.columns:
        return result
    labels = [f"{prefix}_Q{index + 1}" for index in range(quantiles)]
    for _, indices in frame.groupby("fold_id", sort=False).groups.items():
        values = pd.to_numeric(frame.loc[indices, column], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        if valid.sum() < quantiles:
            continue
        ranked = values.loc[valid].rank(method="first")
        result.loc[ranked.index] = pd.qcut(
            ranked,
            quantiles,
            labels=labels,
            duplicates="drop",
        ).astype(str)
    return result


def stratified_outcome_diagnostics(
    frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize causal OOF outcomes across pre-specified diagnostic strata.

    Quantile buckets are formed independently inside each held-out fold using
    feature values only. They are descriptive evaluation bins and are never fed
    back into training, parameter selection, or signal construction.
    """
    if frame.empty or "fold_id" not in frame.columns:
        return pd.DataFrame()
    selected = frame[frame.get("official", False).fillna(False).astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame()
    validation_state = state_frame[
        state_frame.get("state_data_role", pd.Series(index=state_frame.index, dtype=str)).eq(
            "validation"
        )
    ].copy()
    if not validation_state.empty:
        selected = selected.merge(
            validation_state.drop_duplicates(KEYS)[
                KEYS + ["canonical_state_id", "state_semantic_label"]
            ],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    stock_shock = pd.to_numeric(selected.get("stock_idio_return_current"), errors="coerce")
    selected["shock_direction_bucket"] = np.select(
        [stock_shock.gt(0), stock_shock.lt(0)],
        ["UP", "DOWN"],
        default="FLAT_OR_MISSING",
    )
    selected["shock_size_bucket"] = _fold_quantile_bucket(
        selected, "stock_shock_z", "ABS_SHOCK"
    )
    # Shock magnitude, unlike the signed direction bucket, uses absolute values.
    absolute_shock = pd.to_numeric(selected.get("stock_shock_z"), errors="coerce").abs()
    selected["_absolute_stock_shock"] = absolute_shock
    selected["shock_size_bucket"] = _fold_quantile_bucket(
        selected, "_absolute_stock_shock", "ABS_SHOCK"
    )
    selected["premium_bucket"] = _fold_quantile_bucket(
        selected, "premium_lag1", "PREMIUM"
    )
    selected["volatility_bucket"] = _fold_quantile_bucket(
        selected, "stock_realized_vol_6bar", "VOL"
    )
    selected["liquidity_bucket"] = _fold_quantile_bucket(
        selected, "relative_amount_lag1", "LIQUIDITY"
    )
    slot = pd.to_numeric(selected.get("bar_slot"), errors="coerce")
    selected["time_slot_bucket"] = np.select(
        [slot.between(0, 5), slot.between(6, 23), slot.between(24, 29), slot.between(30, 41), slot.between(42, 47)],
        ["OPEN_30M", "AM_LATE", "PM_OPEN_30M", "PM_MID", "CLOSE_30M"],
        default="MISSING",
    )
    if "canonical_state_id" in selected.columns:
        selected["hidden_state_bucket"] = (
            "STATE_"
            + selected["canonical_state_id"].astype("Int64").astype(str)
            + "__"
            + selected["state_semantic_label"].fillna("UNAVAILABLE").astype(str)
        )
    else:
        selected["hidden_state_bucket"] = "UNAVAILABLE"

    dimensions = {
        "fold": "fold_id",
        "bond": "bond_code",
        "trade_date": "trade_date",
        "stock_shock_direction": "shock_direction_bucket",
        "stock_shock_size": "shock_size_bucket",
        "premium": "premium_bucket",
        "volatility": "volatility_bucket",
        "liquidity": "liquidity_bucket",
        "time_slot": "time_slot_bucket",
        "hidden_state": "hidden_state_bucket",
    }
    if "cb_source_vendor" in selected.columns:
        dimensions["source_vendor"] = "cb_source_vendor"
    if "research_pool_scope" in selected.columns:
        dimensions["pool_scope"] = "research_pool_scope"

    rows: list[pd.DataFrame] = []
    for bars in config["labels"]["horizons_bars"]:
        horizon = f"{int(bars) * 5}m"
        required = [
            f"forward_cb_return_{horizon}",
            f"forward_residual_return_{horizon}",
            f"gap_close_ratio_clipped_{horizon}",
            f"mfe_{horizon}",
            f"mae_{horizon}",
        ]
        if not all(column in selected.columns for column in required):
            continue
        horizon_frame = selected[selected[f"forward_cb_return_{horizon}"].notna()].copy()
        for dimension, bucket_column in dimensions.items():
            summary = horizon_frame.groupby(bucket_column, dropna=False).agg(
                sample_count=("bond_code", "size"),
                bond_count=("bond_code", "nunique"),
                trading_days=("trade_date", "nunique"),
                mean_return=(f"forward_cb_return_{horizon}", "mean"),
                median_return=(f"forward_cb_return_{horizon}", "median"),
                hit_rate=(f"forward_cb_return_{horizon}", lambda values: values.gt(0).mean()),
                residual_return=(f"forward_residual_return_{horizon}", "mean"),
                gap_close_ratio=(f"gap_close_ratio_clipped_{horizon}", "mean"),
                mfe=(f"mfe_{horizon}", "mean"),
                mae=(f"mae_{horizon}", "mean"),
                lagger_catchup_rate=(f"lagger_catchup_flag_{horizon}", "mean"),
                leader_reversal_rate=(f"leader_reversal_flag_{horizon}", "mean"),
            ).reset_index().rename(columns={bucket_column: "bucket"})
            summary["bucket"] = summary["bucket"].astype(str)
            summary["dimension"] = dimension
            summary["horizon"] = horizon
            summary["data_scope"] = "official_causal_oof"
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cost_sensitivity(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    horizon: str = "60m",
) -> pd.DataFrame:
    predicted_column = f"C_SURFACE_STATE_prediction_forward_cb_return_{horizon}"
    executable_column = f"forward_executable_cb_return_{horizon}"
    actual_column = executable_column if executable_column in predictions.columns else f"forward_cb_return_{horizon}"
    if predicted_column not in predictions.columns:
        predicted_column = f"B_SURFACE_prediction_forward_cb_return_{horizon}"
    if predicted_column not in predictions.columns or actual_column not in predictions.columns:
        return pd.DataFrame()
    rows = []
    actual = pd.to_numeric(predictions[actual_column], errors="coerce")
    predicted = pd.to_numeric(predictions[predicted_column], errors="coerce")
    for scenario, values in config["costs"]["scenarios"].items():
        component_sum = (
            float(values["half_spread_bps"])
            + float(values["slippage_bps"])
            + float(values["impact_bps"])
        )
        total_bps = max(float(values["minimum_cost_floor_bps"]), component_sum)
        threshold = total_bps / 10000.0
        selected = predicted.gt(threshold) & actual.notna()
        net = actual[selected] - threshold
        rows.append(
            {
                "scenario": scenario,
                "horizon": horizon,
                "prediction_model": predicted_column,
                "realized_return_column": actual_column,
                "selected_sample_count": int(selected.sum()),
                "selection_rate": float(selected.mean()),
                "half_spread_bps": float(values["half_spread_bps"]),
                "slippage_bps": float(values["slippage_bps"]),
                "impact_bps": float(values["impact_bps"]),
                "total_cost_bps": total_bps,
                "gross_mean_return": float(actual[selected].mean()) if selected.any() else np.nan,
                "net_mean_return": float(net.mean()) if selected.any() else np.nan,
                "gross_hit_rate": float(actual[selected].gt(0).mean()) if selected.any() else np.nan,
                "net_hit_rate": float(net.gt(0).mean()) if selected.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def two_dimensional_response_surfaces(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    pairs = [
        ("stock_idio_return_current", "premium_lag1"),
        ("stock_idio_return_current", "stock_realized_vol_6bar"),
        ("stock_idio_return_current", "relative_amount_lag1"),
        ("response_gap", "relative_amount_lag1"),
        ("response_gap", "bar_slot_numeric"),
        ("moneyness_current", "stock_idio_return_current"),
    ]
    prediction = "prediction_lightgbm_q50"
    bins = int(config["evaluation"]["surface_grid_bins"])
    minimum = int(config["evaluation"]["minimum_bucket_rows"])
    rows: list[pd.DataFrame] = []
    for x_column, y_column in pairs:
        if not all(column in frame.columns for column in [x_column, y_column, prediction]):
            continue
        selected = frame[[x_column, y_column, prediction, "surface_target_cb_return"]].dropna().copy()
        if len(selected) < bins * bins:
            continue
        selected["x_bin"] = pd.qcut(selected[x_column].rank(method="first"), bins, labels=False, duplicates="drop")
        selected["y_bin"] = pd.qcut(selected[y_column].rank(method="first"), bins, labels=False, duplicates="drop")
        summary = selected.groupby(["x_bin", "y_bin"], as_index=False).agg(
            sample_count=(prediction, "size"),
            x_mean=(x_column, "mean"),
            y_mean=(y_column, "mean"),
            predicted_cb_return=(prediction, "mean"),
            actual_cb_return=("surface_target_cb_return", "mean"),
        )
        summary["sparse_bucket"] = summary["sample_count"].lt(minimum)
        summary["x_feature"] = x_column
        summary["y_feature"] = y_column
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def heldout_shap_importance(
    frame: pd.DataFrame,
    config: dict[str, Any],
    fold_id: str,
    model_name: str = "lightgbm_q50",
) -> pd.DataFrame:
    import shap

    model_path = output_dir(config) / "models" / "surface" / fold_id / f"{model_name}.joblib"
    if not model_path.exists():
        return pd.DataFrame()
    fitted = joblib.load(model_path)
    heldout = frame[frame["fold_id"].eq(fold_id)].dropna(subset=["surface_target_cb_return"])
    heldout = deterministic_row_sample(
        heldout,
        int(config["evaluation"]["shap_sample_rows"]),
        int(config["strategy"]["random_seed"]),
    )
    if heldout.empty:
        return pd.DataFrame()
    matrix = fitted.preprocessor.transform(heldout[fitted.feature_columns])
    explainer = shap.TreeExplainer(fitted.estimator)
    values = np.asarray(explainer.shap_values(matrix), dtype=float)
    names = fitted.preprocessor.get_feature_names_out()
    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame(
        {
            "fold_id": fold_id,
            "model": model_name,
            "feature": names,
            "importance": importance,
            "importance_type": "heldout_mean_abs_shap",
            "sample_count": len(heldout),
        }
    ).sort_values("importance", ascending=False)


def heldout_ale_profiles(
    frame: pd.DataFrame,
    config: dict[str, Any],
    fold_id: str,
    model_name: str = "lightgbm_q50",
) -> pd.DataFrame:
    """Estimate first-order ALE profiles on one strictly held-out OOF fold.

    The fitted model is never refit here. For each numeric feature, observations
    from the requested validation fold are binned by held-out quantiles. Local
    prediction differences at each bin boundary are accumulated and centered
    with held-out occupancy weights. This is an interpretation diagnostic, not
    an input to training or signal generation.
    """
    model_path = output_dir(config) / "models" / "surface" / fold_id / f"{model_name}.joblib"
    if not model_path.exists():
        return pd.DataFrame()
    fitted = joblib.load(model_path)
    heldout = frame[frame["fold_id"].eq(fold_id)].dropna(
        subset=["surface_target_cb_return"]
    )
    heldout = deterministic_row_sample(
        heldout,
        int(config["evaluation"].get("ale_sample_rows", 20000)),
        int(config["strategy"]["random_seed"]),
    )
    if heldout.empty:
        return pd.DataFrame()

    requested = config["evaluation"].get(
        "ale_features",
        [
            "stock_idio_return_current",
            "premium_lag1",
            "stock_realized_vol_6bar",
            "relative_amount_lag1",
            "moneyness_current",
        ],
    )
    bins = int(config["evaluation"].get("ale_bins", 12))
    minimum = int(config["evaluation"].get("minimum_ale_bin_rows", 50))
    rows: list[dict[str, Any]] = []
    for feature in requested:
        if feature not in fitted.feature_columns or feature not in heldout.columns:
            continue
        values = pd.to_numeric(heldout[feature], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        selected = heldout.loc[valid].copy()
        selected_values = values.loc[valid].to_numpy(dtype=float)
        if len(selected) < max(bins * minimum, 2 * bins):
            continue
        edges = np.unique(
            np.quantile(selected_values, np.linspace(0.0, 1.0, bins + 1))
        )
        if len(edges) < 3:
            continue
        bin_index = np.searchsorted(edges, selected_values, side="right") - 1
        bin_index = np.clip(bin_index, 0, len(edges) - 2)
        local_rows: list[dict[str, Any]] = []
        for index in range(len(edges) - 1):
            mask = bin_index == index
            sample_count = int(mask.sum())
            if sample_count < minimum:
                continue
            group = selected.iloc[np.flatnonzero(mask)].copy()
            low = group.copy()
            high = group.copy()
            low[feature] = float(edges[index])
            high[feature] = float(edges[index + 1])
            local_effect = float(np.mean(fitted.predict(high) - fitted.predict(low)))
            local_rows.append(
                {
                    "fold_id": fold_id,
                    "model": model_name,
                    "feature": feature,
                    "bin_index": index,
                    "bin_lower": float(edges[index]),
                    "bin_upper": float(edges[index + 1]),
                    "feature_mean": float(selected_values[mask].mean()),
                    "sample_count": sample_count,
                    "local_effect": local_effect,
                    "heldout_target_mean": float(
                        pd.to_numeric(group["surface_target_cb_return"], errors="coerce").mean()
                    ),
                }
            )
        if not local_rows:
            continue
        cumulative = np.cumsum([row["local_effect"] for row in local_rows])
        weights = np.asarray([row["sample_count"] for row in local_rows], dtype=float)
        centered = cumulative - np.average(cumulative, weights=weights)
        for row, ale_value in zip(local_rows, centered, strict=True):
            row["ale_value"] = float(ale_value)
            row["data_scope"] = "strictly_heldout_validation"
            rows.append(row)
    return pd.DataFrame(rows)


def model_increment_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    primary = metrics[
        metrics["target"].str.startswith("forward_cb_return", na=False)
    ].copy()
    if primary.empty:
        return pd.DataFrame()
    pivot = primary.pivot_table(
        index=["fold_id", "horizon", "target"],
        columns="model",
        values=["mae", "rmse", "spearman", "directional_accuracy"],
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    for metric in ["mae", "rmse"]:
        if f"{metric}_A_BASELINE" in pivot and f"{metric}_B_SURFACE" in pivot:
            pivot[f"B_vs_A_{metric}_improvement"] = pivot[f"{metric}_A_BASELINE"] - pivot[f"{metric}_B_SURFACE"]
        if f"{metric}_B_SURFACE" in pivot and f"{metric}_C_SURFACE_STATE" in pivot:
            pivot[f"C_vs_B_{metric}_improvement"] = pivot[f"{metric}_B_SURFACE"] - pivot[f"{metric}_C_SURFACE_STATE"]
    return pivot
