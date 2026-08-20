from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import environment_manifest, output_dir
from .data import (
    build_walk_forward_folds,
    load_source_frame,
    select_fold_rows,
    write_dataset_manifest,
)
from .eval import (
    aggregate_model_comparison,
    cost_sensitivity,
    fit_predict_outcome_folds,
    heldout_ale_profiles,
    heldout_shap_importance,
    merge_surface_oof,
    model_increment_summary,
    response_gap_decile_outcomes,
    stratified_outcome_diagnostics,
    state_conditioned_outcomes,
    two_dimensional_response_surfaces,
)
from .features import (
    STAGE_A_CATEGORICAL_FEATURES,
    STAGE_A_NUMERIC_FEATURES,
    add_forward_labels,
    add_oof_gap_and_closure_labels,
    build_feature_dataset,
    write_feature_schema,
)
from .models import (
    fit_lightgbm_surface,
    fit_predict_surface_folds,
    fit_ridge_surface,
    fit_xgboost_surface,
    quantile_metrics,
    tune_lightgbm_with_optuna,
)
from .report import generate_plots, generate_report
from .state import fit_predict_state_folds, write_transition_payload


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
    mode: str
    start_date: str | None = None
    end_date: str | None = None
    days: int | None = None
    models: tuple[str, ...] | None = None
    retune: bool = False
    use_cache: bool = True
    output_override: str | None = None


def _with_run_overrides(
    config: dict[str, Any], options: RunOptions
) -> dict[str, Any]:
    result = deepcopy(config)
    if options.output_override:
        result["strategy"]["output_dir"] = options.output_override
    if options.mode == "smoke-test":
        result["strategy"]["output_dir"] = str(
            Path(result["strategy"]["output_dir"]) / "smoke"
        )
        result["strategy"]["n_jobs"] = min(8, int(result["strategy"]["n_jobs"]))
        result["surface_models"]["training_row_cap"] = 30000
        result["surface_models"]["validation_row_cap"] = 30000
        result["surface_models"]["lightgbm"]["n_estimators"] = 120
        result["surface_models"]["xgboost"]["n_estimators"] = 120
        result["surface_models"]["early_stopping_days"] = 4
        result["hmm"]["candidate_states"] = [3]
        result["hmm"]["covariance_types"] = ["diag"]
        result["hmm"]["candidate_seeds"] = [int(result["strategy"]["random_seed"])]
        result["hmm"]["candidate_row_cap"] = 8000
        result["hmm"]["training_row_cap"] = 12000
        result["hmm"]["n_iter"] = 15
        result["outcome_models"]["training_row_cap"] = 30000
        result["evaluation"]["shap_sample_rows"] = 500
    return result


def _add_versions(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    out["strategy_version"] = config["strategy"]["version"]
    out["factor_version"] = config["strategy"]["factor_version"]
    out["config_version"] = config["strategy"]["config_version"]
    out["model_seed"] = int(config["strategy"]["random_seed"])
    return out


def _write_csv(frame: pd.DataFrame, path: Path, config: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _add_versions(frame, config).to_csv(path, index=False)
    return path


def _write_parquet(frame: pd.DataFrame, path: Path, config: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _add_versions(frame, config).to_parquet(path, index=False, compression="zstd")
    return path


def _dataset_cache_path(config: dict[str, Any]) -> Path:
    return output_dir(config) / str(config["data"]["dataset_cache"])


def _cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def build_dataset(
    config: dict[str, Any], options: RunOptions
) -> tuple[pd.DataFrame, list[Any], list[str]]:
    cache = _dataset_cache_path(config)
    metadata_path = _cache_metadata_path(cache)
    expected_days = options.days if options.mode != "smoke-test" else (options.days or 45)
    can_use_cache = options.use_cache and cache.exists() and metadata_path.exists()
    if can_use_cache:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        can_use_cache = (
            metadata.get("config_version") == config["strategy"]["config_version"]
            and metadata.get("start_date_filter") == options.start_date
            and metadata.get("end_date_filter") == options.end_date
            and metadata.get("days_filter") == expected_days
        )
    if can_use_cache:
        LOGGER.info("loading V0.8 dataset cache: %s", cache)
        dataset = pd.read_parquet(cache)
        label_columns = metadata["label_columns"]
        expected_executable = {
            f"forward_executable_cb_return_{int(bars) * 5}m"
            for bars in config["labels"]["horizons_bars"]
        }
        if not expected_executable.issubset(dataset.columns):
            LOGGER.info("upgrading cached labels to next-bar executable entry contract")
            dataset, label_columns = add_forward_labels(dataset, config)
            dataset.to_parquet(cache, index=False, compression="zstd")
            metadata["label_columns"] = label_columns
            metadata["rows"] = len(dataset)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        source = load_source_frame(
            config,
            start_date=options.start_date,
            end_date=options.end_date,
            days=expected_days,
        )
        LOGGER.info("building V0.8 feature contract rows=%d", len(source))
        dataset = build_feature_dataset(source, config)
        dataset, label_columns = add_forward_labels(dataset, config)
        cache.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(cache, index=False, compression="zstd")
        metadata_path.write_text(
            json.dumps(
                {
                    "config_version": config["strategy"]["config_version"],
                    "start_date_filter": options.start_date,
                    "end_date_filter": options.end_date,
                    "days_filter": expected_days,
                    "rows": len(dataset),
                    "label_columns": label_columns,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"]).dt.normalize()
    dataset["bar_start"] = pd.to_datetime(dataset["bar_start"])
    dataset["bar_end"] = pd.to_datetime(dataset["bar_end"])
    folds = build_walk_forward_folds(dataset, config)
    return dataset, folds, label_columns


def _mean_missing_rate(frame: pd.DataFrame, columns: list[str]) -> float:
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        return float("nan")
    missing = sum(int(frame[column].isna().sum()) for column in available)
    return float(missing / (len(frame) * len(available)))


def _fold_contract_frame(
    dataset: pd.DataFrame,
    folds: list[Any],
) -> pd.DataFrame:
    features = [
        column
        for column in STAGE_A_NUMERIC_FEATURES + STAGE_A_CATEGORICAL_FEATURES
        if column in dataset.columns
    ]
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train, validation = select_fold_rows(dataset, fold)
        record = fold.to_record()
        record.update(
            {
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "train_bond_count": int(train["bond_code"].nunique()),
                "validation_bond_count": int(validation["bond_code"].nunique()),
                "train_feature_missing_rate": _mean_missing_rate(train, features),
                "validation_feature_missing_rate": _mean_missing_rate(validation, features),
                "feature_count": len(features),
                "train_target_rows": int(train["surface_target_cb_return"].notna().sum()),
                "validation_target_rows": int(
                    validation["surface_target_cb_return"].notna().sum()
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _write_model_parameter_manifest(
    config: dict[str, Any],
    folds: list[Any],
    surface_metrics: pd.DataFrame | None = None,
) -> Path:
    results: list[dict[str, Any]] = []
    if surface_metrics is not None and not surface_metrics.empty:
        columns = [
            column
            for column in [
                "fold_id", "model", "best_iteration", "train_rows",
                "validation_start", "validation_end", "runtime_seconds",
            ]
            if column in surface_metrics.columns
        ]
        results = json.loads(surface_metrics[columns].to_json(orient="records"))
    payload = {
        "strategy_version": config["strategy"]["version"],
        "config_version": config["strategy"]["config_version"],
        "seed": int(config["strategy"]["random_seed"]),
        "n_jobs": int(config["strategy"]["n_jobs"]),
        "surface_models": config["surface_models"],
        "outcome_models": config["outcome_models"],
        "hmm": config["hmm"],
        "folds": [fold.to_record() for fold in folds],
        "fold_fit_results": results,
        "parameter_scope": "Static parameters are shared; best_iteration is fold-specific.",
    }
    path = output_dir(config) / "model_parameter_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_base_artifacts(
    dataset: pd.DataFrame,
    folds: list[Any],
    label_columns: list[str],
    config: dict[str, Any],
) -> None:
    out = output_dir(config)
    (out / "environment_manifest.json").write_text(
        json.dumps(environment_manifest(), indent=2), encoding="utf-8"
    )
    _fold_contract_frame(dataset, folds).to_csv(out / "fold_definition.csv", index=False)
    _write_model_parameter_manifest(config, folds)
    write_dataset_manifest(
        dataset,
        folds,
        config,
        STAGE_A_NUMERIC_FEATURES + STAGE_A_CATEGORICAL_FEATURES,
        label_columns,
    )
    write_feature_schema(dataset, config, label_columns)


def fit_latest_research_artifacts(
    dataset: pd.DataFrame,
    config: dict[str, Any],
    model_names: tuple[str, ...] | None,
) -> Path:
    target = "surface_target_cb_return"
    train = dataset[dataset[target].notna()].copy()
    cap = int(config["surface_models"]["training_row_cap"])
    if len(train) > cap:
        train = train.sample(cap, random_state=int(config["strategy"]["random_seed"]))
    numeric = [column for column in STAGE_A_NUMERIC_FEATURES if column in train.columns]
    categorical = [column for column in STAGE_A_CATEGORICAL_FEATURES if column in train.columns]
    requested = set(model_names or tuple(config["surface_models"]["enabled"]))
    fitted = []
    if "ridge" in requested:
        fitted.append(fit_ridge_surface(train, target, numeric, categorical, config))
    if "lightgbm" in requested:
        fitted.append(fit_lightgbm_surface(train, target, numeric, categorical, config, name="lightgbm_mean"))
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
    destination = output_dir(config) / "models" / "latest_research"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in fitted:
        joblib.dump(model, destination / f"{model.name}.joblib", compress=3)
        rows.append(
            {
                "model": model.name,
                "train_start": str(train["trade_date"].min().date()),
                "train_end": str(train["trade_date"].max().date()),
                "train_rows": len(train),
                "best_iteration": model.best_iteration,
                "runtime_seconds": model.runtime_seconds,
            }
        )
    manifest = destination / "fit_latest_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "research_only": True,
                "production_alert_integration": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "models": rows,
                "config_version": config["strategy"]["config_version"],
                "seed": config["strategy"]["random_seed"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def run_backtest(config: dict[str, Any], options: RunOptions) -> dict[str, Any]:
    config = _with_run_overrides(config, options)
    out = output_dir(config)
    dataset, folds, base_labels = build_dataset(config, options)
    _write_base_artifacts(dataset, folds, base_labels, config)
    if options.mode == "build-dataset":
        return {"output_dir": str(out), "rows": len(dataset), "folds": len(folds)}
    if options.mode == "fit-latest":
        manifest = fit_latest_research_artifacts(dataset, config, options.models)
        return {"output_dir": str(out), "fit_latest_manifest": str(manifest)}
    if options.retune:
        first_official = next((fold for fold in folds if fold.official), folds[-1])
        train, _ = select_fold_rows(dataset, first_official)
        tune_lightgbm_with_optuna(train, config)

    surface_path = out / "oof_surface_predictions.parquet"
    surface_metrics_path = out / "surface_fold_metrics.csv"
    importance_path = out / "feature_importance_tree.csv"
    if options.use_cache and all(path.exists() for path in [surface_path, surface_metrics_path, importance_path]):
        LOGGER.info("loading cached OOF surface artifacts")
        surface_oof = pd.read_parquet(surface_path)
        surface_metrics = pd.read_csv(surface_metrics_path)
        tree_importance = pd.read_csv(importance_path)
    else:
        surface_oof_raw, surface_metrics, tree_importance = fit_predict_surface_folds(
            dataset, folds, config, list(options.models) if options.models else None
        )
        enriched = merge_surface_oof(dataset, surface_oof_raw)
        enriched, gap_labels = add_oof_gap_and_closure_labels(
            enriched, "prediction_lightgbm_q50", config
        )
        surface_oof = enriched
        _write_parquet(surface_oof, surface_path, config)
        _write_csv(surface_metrics, surface_metrics_path, config)
        _write_csv(tree_importance, importance_path, config)
        base_labels = base_labels + gap_labels
    expected_executable = {
        f"forward_executable_cb_return_{int(bars) * 5}m"
        for bars in config["labels"]["horizons_bars"]
    }
    if not expected_executable.issubset(surface_oof.columns):
        LOGGER.info("upgrading cached OOF artifact to executable-entry labels")
        surface_oof, refreshed_labels = add_forward_labels(surface_oof, config)
        base_labels = list(dict.fromkeys(base_labels + refreshed_labels))
        _write_parquet(surface_oof, surface_path, config)
    if "response_gap" not in surface_oof.columns:
        surface_oof, gap_labels = add_oof_gap_and_closure_labels(
            surface_oof, "prediction_lightgbm_q50", config
        )
        base_labels = base_labels + gap_labels
    _write_model_parameter_manifest(config, folds, surface_metrics)

    quantile = quantile_metrics(surface_oof)
    _write_csv(quantile, out / "quantile_calibration.csv", config)
    if not tree_importance.empty and "importance_type" not in tree_importance:
        tree_importance["importance_type"] = "tree_split"

    state_all, state_summary, state_stability, transition_payload = fit_predict_state_folds(
        surface_oof, folds, config
    )
    state_validation = state_all[state_all["state_data_role"].eq("validation")].copy()
    _write_parquet(state_validation, out / "oof_state_probabilities.parquet", config)
    _write_parquet(state_all[state_all["state_data_role"].eq("train")], out / "fold_training_state_probabilities.parquet", config)
    _write_csv(state_summary, out / "state_summary.csv", config)
    _write_csv(state_stability, out / "state_stability.csv", config)
    write_transition_payload(transition_payload, config)

    state_count = int(transition_payload["selected_n_states"])
    forward_predictions, outcome_metrics = fit_predict_outcome_folds(
        surface_oof, state_all, folds, config, state_count
    )
    _write_parquet(forward_predictions, out / "oof_forward_predictions.parquet", config)
    _write_csv(outcome_metrics, out / "outcome_fold_metrics.csv", config)
    model_comparison = aggregate_model_comparison(outcome_metrics)
    _write_csv(model_comparison, out / "model_comparison.csv", config)
    increment = model_increment_summary(outcome_metrics)
    _write_csv(increment, out / "model_increment_by_fold.csv", config)

    fold_metrics = pd.concat(
        [
            surface_metrics.assign(metric_stage="surface"),
            outcome_metrics.assign(metric_stage="forward_outcome"),
        ],
        ignore_index=True,
        sort=False,
    )
    _write_csv(fold_metrics, out / "fold_metrics.csv", config)
    state_outcomes = state_conditioned_outcomes(surface_oof, state_all, config)
    _write_csv(state_outcomes, out / "state_conditioned_outcomes.csv", config)
    stratified = stratified_outcome_diagnostics(surface_oof, state_all, config)
    _write_csv(stratified, out / "stratified_outcomes.csv", config)
    gap_deciles = response_gap_decile_outcomes(surface_oof, config)
    _write_csv(gap_deciles, out / "response_gap_decile_outcomes.csv", config)
    costs = cost_sensitivity(forward_predictions, config)
    _write_csv(costs, out / "cost_sensitivity.csv", config)
    response_surfaces = two_dimensional_response_surfaces(surface_oof, config)
    _write_csv(response_surfaces, out / "response_surfaces_2d.csv", config)

    official_folds = [fold for fold in folds if fold.official]
    latest_fold = official_folds[-1].fold_id
    shap_importance = heldout_shap_importance(surface_oof, config, latest_fold)
    ale_profiles = heldout_ale_profiles(surface_oof, config, latest_fold)
    _write_csv(ale_profiles, out / "ale_profiles.csv", config)
    combined_importance = pd.concat([tree_importance, shap_importance], ignore_index=True, sort=False)
    _write_csv(combined_importance, out / "feature_importance.csv", config)

    bond_dispersion = surface_oof.groupby("bond_code", as_index=False).agg(
        sample_count=("surface_target_cb_return", "count"),
        actual_mean=("surface_target_cb_return", "mean"),
        predicted_mean=("prediction_lightgbm_q50", "mean"),
        response_gap_mean=("response_gap", "mean"),
        response_gap_std=("response_gap", "std"),
    )
    _write_csv(bond_dispersion, out / "bond_level_performance_dispersion.csv", config)
    date_dispersion = surface_oof.groupby(["fold_id", "trade_date"], as_index=False).agg(
        sample_count=("surface_target_cb_return", "count"),
        actual_mean=("surface_target_cb_return", "mean"),
        predicted_mean=("prediction_lightgbm_q50", "mean"),
        absolute_error=("response_gap", lambda values: values.abs().mean()),
    )
    _write_csv(date_dispersion, out / "date_level_performance_dispersion.csv", config)

    plot_paths = generate_plots(
        surface_oof,
        forward_predictions,
        state_all,
        gap_deciles,
        state_outcomes,
        outcome_metrics,
        quantile,
        response_surfaces,
        ale_profiles,
        transition_payload,
        config,
    )
    report_path = generate_report(config)
    run_manifest = {
        "strategy_version": config["strategy"]["version"],
        "config_version": config["strategy"]["config_version"],
        "mode": options.mode,
        "rows": int(len(dataset)),
        "oof_surface_rows": int(len(surface_oof)),
        "official_forward_rows": int(len(forward_predictions)),
        "state_probability_rows": int(len(state_validation)),
        "trade_dates": int(dataset["trade_date"].nunique()),
        "bond_count": int(dataset["bond_code"].nunique()),
        "models": list(options.models or tuple(config["surface_models"]["enabled"])),
        "seed": int(config["strategy"]["random_seed"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report": str(report_path),
        "plots": [str(path) for path in plot_paths],
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return {"output_dir": str(out), **run_manifest}


def regenerate_report(config: dict[str, Any], options: RunOptions) -> Path:
    config = _with_run_overrides(config, options)
    return generate_report(config)
