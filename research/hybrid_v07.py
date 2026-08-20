from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config

from db import get_conn
from research.hybrid_v07_core import (
    LSMCResult,
    add_qiv_and_mc,
    classify_bonds,
    compute_liquidity_and_friction,
    load_hybrid_config,
    portfolio_metrics,
    run_lsmc_row,
    select_trade_horizon,
)
from research.hybrid_v07_data import build_daily_research_panel, write_csv


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_csv(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V0.7 convertible-bond hybrid intraday/daily/LSMC research system."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "classify-bonds",
            "price-daily",
            "build-lsmc-dataset",
            "run-lsmc",
            "run-lsmc-no-qiv-ablation",
            "select-horizon",
            "backtest",
            "ablation",
            "report",
            "all",
        ],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--backtest-paths", type=int, default=None)
    parser.add_argument("--path-model", default="gbm,empirical_bootstrap")
    parser.add_argument("--regression-model", default="ridge")
    parser.add_argument("--strategies", default="intraday,daily_lsmc,hybrid")
    parser.add_argument("--cost-bps", default="5,10,20,30")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--include-screening-extension", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def output_dir(config: dict[str, Any]) -> Path:
    path = Path(config["strategy"]["output_dir"])
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_dates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in ["trade_date", "decision_time", "static_snapshot_time", "maturity_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in [
        "lsmc_eligible",
        "intraday_eligible",
        "daily_holding_eligible",
        "clause_driven_flag",
        "distress_flag",
        "qiv_microstructure_contaminated",
        "qiv_contamination_assessable",
        "qiv_solve_success",
        "static_asof_valid",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.lower().map({"true": True, "false": False})
    return frame


def date_filter(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if df.empty or "trade_date" not in df.columns:
        return df.copy()
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    if start_date:
        out = out[out["trade_date"] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        out = out[out["trade_date"] <= pd.Timestamp(end_date).normalize()]
    return out


def build_or_load_panel(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    refresh: bool,
    include_screening_extension: bool,
) -> pd.DataFrame:
    path = output_dir(config) / "daily_research_panel.csv"
    if path.exists() and not refresh:
        panel = read_csv_dates(path)
        return date_filter(panel, start_date, end_date)
    conn = get_conn()
    try:
        panel = build_daily_research_panel(
            conn,
            None,
            end_date,
            config,
            include_screening_extension=include_screening_extension,
        )
    finally:
        conn.close()
    panel = classify_bonds(panel, config)
    panel = compute_liquidity_and_friction(panel, config)
    write_csv(panel, path)
    return date_filter(panel, start_date, end_date)


def run_classification(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    refresh: bool,
    include_screening_extension: bool,
) -> dict[str, Any]:
    panel = build_or_load_panel(
        config, start_date, end_date, refresh, include_screening_extension
    )
    if panel.empty:
        raise RuntimeError("No daily research panel rows in requested range")
    snapshot_columns = [
        "trade_date",
        "bond_code",
        "stock_code",
        "bond_name",
        "bond_type",
        "bond_open",
        "bond_close",
        "bond_type_version",
        "bond_type_confidence",
        "classification_reason",
        "clause_driven_flag",
        "distress_flag",
        "lsmc_eligible",
        "intraday_eligible",
        "daily_holding_eligible",
        "lsmc_exclusion_reason",
        "classification_point_in_time",
        "static_asof_valid",
        "research_pool_scope",
    ]
    snapshot = panel[[c for c in snapshot_columns if c in panel.columns]].copy()
    out = output_dir(config)
    write_csv(snapshot, out / "cb_bond_type_snapshot.csv")
    latest = snapshot[snapshot["trade_date"] == snapshot["trade_date"].max()]
    counts = latest.groupby("bond_type", dropna=False).size().rename("sample_count").reset_index()
    eligibility = latest.groupby("bond_type", dropna=False).agg(
        lsmc_eligible=("lsmc_eligible", "sum"),
        intraday_eligible=("intraday_eligible", "sum"),
        daily_holding_eligible=("daily_holding_eligible", "sum"),
        confidence_mean=("bond_type_confidence", "mean"),
    ).reset_index()
    report = counts.merge(eligibility, on="bond_type", how="left")
    write_csv(report, out / "bond_type_report.csv")
    acceptance = panel.groupby("bond_code", dropna=False).agg(
        panel_daily_rows=("trade_date", "size"),
        first_trade_date=("trade_date", "min"),
        last_trade_date=("trade_date", "max"),
        conversion_price_covered_days=("conversion_price_asof_valid", "sum"),
        unadjusted_stock_price_covered_days=("qiv_stock_close_asof_valid", "sum"),
        joint_pricing_input_covered_days=("pricing_inputs_asof_valid", "sum"),
    ).reset_index()
    for numerator, output_column in [
        ("conversion_price_covered_days", "conversion_price_coverage_ratio"),
        ("unadjusted_stock_price_covered_days", "unadjusted_stock_price_coverage_ratio"),
        ("joint_pricing_input_covered_days", "joint_pricing_input_coverage_ratio"),
    ]:
        acceptance[output_column] = (
            acceptance[numerator] / acceptance["panel_daily_rows"].replace(0, np.nan)
        )
    acceptance["strategy_version"] = config["strategy"]["version"]
    acceptance["config_version"] = config["strategy"]["config_version"]
    write_csv(acceptance, out / "conversion_price_acceptance.csv")
    return {
        "rows": int(len(snapshot)),
        "latest_trade_date": str(latest["trade_date"].max().date()),
        "latest_bonds": int(latest["bond_code"].nunique()),
        "bond_type_counts": latest["bond_type"].value_counts().to_dict(),
        "lsmc_eligible": int(latest["lsmc_eligible"].fillna(False).sum()),
        "lsmc_excluded": int((~latest["lsmc_eligible"].fillna(False)).sum()),
        "point_in_time_static_coverage": float(latest["static_asof_valid"].fillna(False).mean()),
        "conversion_price_coverage": float(
            panel["conversion_price_asof_valid"].fillna(False).mean()
        ),
        "unadjusted_stock_price_coverage": float(
            panel["qiv_stock_close_asof_valid"].fillna(False).mean()
        ),
        "joint_pricing_input_coverage": float(
            panel["pricing_inputs_asof_valid"].fillna(False).mean()
        ),
        "conversion_price_acceptance": str(out / "conversion_price_acceptance.csv"),
    }


def run_pricing(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    paths: int | None,
    refresh: bool,
    include_screening_extension: bool,
) -> dict[str, Any]:
    panel = build_or_load_panel(
        config, start_date, end_date, refresh, include_screening_extension
    )
    priced = add_qiv_and_mc(panel, config, paths=paths)
    keep = [
        "trade_date",
        "bond_code",
        "stock_code",
        "bond_type",
        "bond_close",
        "stock_close",
        "qiv_stock_close",
        "qiv_stock_close_source",
        "qiv_stock_close_asof_valid",
        "conversion_price",
        "conversion_price_source",
        "conversion_price_asof_valid",
        "conversion_price_effective_start",
        "conversion_price_effective_end_exclusive",
        "conversion_price_confidence",
        "parity",
        "parity_source",
        "parity_asof_valid",
        "premium",
        "premium_source",
        "premium_asof_valid",
        "pricing_inputs_asof_valid",
        "days_to_maturity",
        "qiv_raw",
        "qiv_vwap",
        "qiv_eod",
        "qiv_smooth",
        "qiv_confidence",
        "qiv_microstructure_contaminated",
        "qiv_contamination_assessable",
        "qiv_contamination_status",
        "qiv_solve_success",
        "qiv_solve_error",
        "qiv_iterations",
        "qiv_model_assumptions",
        "qiv_rank_cross_section",
        "qiv_rank_within_bond_type",
        "qiv_rank_within_parity_bucket",
        "qiv_rank_within_price_bucket",
        "qiv_rank_within_liquidity_bucket",
        "qiv_minus_stock_rv_20d",
        "qiv_minus_stock_rv_5d",
        "qiv_relative_value_score",
        "mc_model_price",
        "mc_price_std",
        "mc_price_p05",
        "mc_price_p50",
        "mc_price_p95",
        "mc_price_gap",
        "mc_price_gap_pct",
        "mc_delta",
        "mc_gamma",
        "mc_vega",
        "mc_theta_1d",
        "mc_runtime_ms",
        "mc_path_count",
        "mc_seed",
        "mc_model_version",
        "model_assumption_flags",
        "credit_spread_proxy",
        "credit_spread_source",
        "mc_credit_spread_sensitivity",
        "static_asof_valid",
    ]
    priced_out = priced[[c for c in keep if c in priced.columns]].copy()
    out = output_dir(config)
    write_csv(priced_out, out / "cb_daily_relative_value_snapshot.csv")
    write_csv(priced_out, out / "qiv_rv_report.csv")
    coverage = float(priced["qiv_solve_success"].fillna(False).mean())
    solved = priced[priced["qiv_solve_success"].fillna(False)]
    assessable = solved[solved["qiv_contamination_assessable"].fillna(False)]
    return {
        "rows": int(len(priced)),
        "qiv_solved": int(len(solved)),
        "qiv_coverage": coverage,
        "qiv_contaminated": int(solved["qiv_microstructure_contaminated"].fillna(False).sum()),
        "qiv_contamination_assessable": int(len(assessable)),
        "qiv_contamination_unavailable": int(len(solved) - len(assessable)),
        "qiv_contamination_rate_among_assessable": float(
            assessable["qiv_microstructure_contaminated"].fillna(False).mean()
        )
        if not assessable.empty
        else np.nan,
        "mc_coverage": float(priced["mc_model_price"].notna().mean()),
        "mc_runtime_ms_total": float(priced["mc_runtime_ms"].fillna(0).sum()),
        "output": str(out / "cb_daily_relative_value_snapshot.csv"),
    }


def load_priced_or_build(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    paths: int | None,
    refresh: bool,
    include_screening_extension: bool,
) -> pd.DataFrame:
    out = output_dir(config)
    panel = build_or_load_panel(
        config, None, end_date, refresh, include_screening_extension
    )
    pricing_path = out / "cb_daily_relative_value_snapshot.csv"
    if pricing_path.exists() and not refresh:
        pricing = read_csv_dates(pricing_path)
        pricing_cols = [
            c for c in pricing.columns
            if c not in panel.columns or c in ["trade_date", "bond_code", "stock_code", "bond_type"]
        ]
        merge_keys = ["trade_date", "bond_code"]
        pricing_cols = list(dict.fromkeys(merge_keys + pricing_cols))
        panel = panel.merge(pricing[pricing_cols], on=merge_keys, how="left", suffixes=("", "_priced"))
    else:
        panel = add_qiv_and_mc(panel, config, paths=paths)
        keep = [c for c in panel.columns if c.startswith("qiv_") or c.startswith("mc_")]
        keep += [
            "trade_date",
            "bond_code",
            "stock_code",
            "bond_type",
            "credit_spread_proxy",
            "credit_spread_source",
            "model_assumption_flags",
        ]
        write_csv(panel[list(dict.fromkeys([c for c in keep if c in panel.columns]))], pricing_path)
    for column, default in [
        ("qiv_microstructure_contaminated", False),
        ("qiv_contamination_assessable", False),
        ("qiv_solve_success", False),
        ("qiv_relative_value_score", np.nan),
        ("mc_price_gap_pct", np.nan),
        ("qiv_smooth", np.nan),
    ]:
        if column not in panel.columns:
            panel[column] = default
    return date_filter(panel, start_date, end_date)


def build_lsmc_dataset(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    paths: int | None,
    refresh: bool,
    include_screening_extension: bool,
) -> dict[str, Any]:
    panel = load_priced_or_build(
        config, start_date, end_date, paths, refresh, include_screening_extension
    )
    columns = [
        "trade_date",
        "bond_code",
        "stock_code",
        "research_pool_scope",
        "bond_type",
        "bond_type_confidence",
        "bond_close",
        "stock_close",
        "parity",
        "premium",
        "qiv_smooth",
        "qiv_relative_value_score",
        "mc_price_gap_pct",
        "empirical_delta",
        "elasticity",
        "days_to_maturity",
        "debt_ratio",
        "operating_cash_flow",
        "free_cash_flow",
        "fundamental_quality_bucket",
        "distress_flag",
        "credit_spread_proxy",
        "daily_turnover",
        "turnover_z",
        "bond_amount",
        "amount_z",
        "adv20_amount",
        "amihud",
        "liquidity_regime",
        "hot_money_risk_score",
        "residual_z_daily",
        "residual_change",
        "stock_return_1d",
        "bond_return_1d",
        "stock_rv_20d",
        "bond_rv_20d",
        "market_regime",
        "cb_ma20d_dev",
        "stock_ma20d_dev",
        "cb_ma90min",
        "stock_ma90min",
        "cb_ma90min_dev",
        "stock_ma90min_dev",
        "ma_alignment_state",
        "ma_monitor_score",
        "premium_change",
        "qiv_microstructure_contaminated",
        "qiv_contamination_assessable",
        "qiv_contamination_status",
        "lsmc_eligible",
        "lsmc_exclusion_reason",
        "forward_return_1d",
        "forward_return_2d",
        "forward_return_3d",
        "forward_return_5d",
        "forward_return_10d",
        "mfe_5d",
        "mae_5d",
        "excess_return_vs_cb_market",
        "beta_hedged_return",
        "next_open",
        "signal_execution_lag",
    ]
    dataset = panel[[c for c in columns if c in panel.columns]].copy()
    dataset["strategy_version"] = config["strategy"]["version"]
    dataset["config_version"] = config["strategy"]["config_version"]
    path = output_dir(config) / "lsmc_dataset.csv"
    write_csv(dataset, path)
    return {
        "rows": int(len(dataset)),
        "bonds": int(dataset["bond_code"].nunique()),
        "trade_days": int(dataset["trade_date"].nunique()),
        "eligible_rows": int(dataset["lsmc_eligible"].fillna(False).sum()),
        "output": str(path),
    }


def _lsmc_result_dict(result, row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    price = float(row["bond_close"])
    sensitivity = row.get("mc_credit_spread_sensitivity", "{}")
    return {
        "trade_date": row["trade_date"],
        "bond_code": row["bond_code"],
        "stock_code": row.get("stock_code"),
        "bond_type": row.get("bond_type"),
        "current_bond_price": price,
        "lsmc_immediate_liquidation_value": result.immediate_liquidation_value,
        "lsmc_continuation_value": result.continuation_value,
        "lsmc_hold_edge": result.hold_edge,
        "lsmc_hold_edge_pct": result.hold_edge / price if np.isfinite(result.hold_edge) else np.nan,
        "lsmc_hold_probability": result.hold_probability,
        "lsmc_expected_holding_days": result.expected_holding_days,
        "lsmc_recommended_action": result.recommended_action,
        "lsmc_confidence": result.confidence,
        "lsmc_path_count": result.path_count,
        "lsmc_regression_r2": result.regression_r2,
        "lsmc_basis_version": config["lsmc"]["basis_version"],
        "lsmc_model_version": config["lsmc"]["model_version"],
        "lsmc_credit_spread_sensitivity": sensitivity,
        "lsmc_exclusion_reason": result.exclusion_reason,
        "lsmc_path_model": result.path_model,
        "lsmc_structural_mapping_terminal_mean": result.structural_mapping_terminal_mean,
        "lsmc_empirical_mapping_terminal_mean": result.empirical_mapping_terminal_mean,
        "lsmc_structural_mapping_available": result.structural_mapping_available,
        "lsmc_empirical_mapping_available": result.empirical_mapping_available,
        "strategy_version": config["strategy"]["version"],
        "config_version": config["strategy"]["config_version"],
    }


def run_lsmc(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    paths: int | None,
    backtest_paths: int | None,
    path_models: list[str],
    regression_model: str,
    n_jobs: int | None,
    refresh: bool,
    include_screening_extension: bool,
    output_filename: str = "cb_lsmc_daily_snapshot.csv",
) -> dict[str, Any]:
    full_panel = load_priced_or_build(
        config, None, end_date, paths, refresh, include_screening_extension
    )
    target = date_filter(full_panel, start_date, end_date)
    if target.empty:
        raise RuntimeError("No LSMC rows in requested range")
    requested_days = target["trade_date"].nunique()
    effective_paths = int(
        paths
        or (
            config["lsmc"]["simulation_paths"]
            if requested_days <= 5
            else backtest_paths or config["lsmc"]["backtest_simulation_paths"]
        )
    )
    workers = n_jobs if n_jobs is not None else int(config["strategy"].get("n_jobs", -1))
    history_by_type = {
        bond_type: group.copy()
        for bond_type, group in full_panel.groupby("bond_type", sort=False)
    }

    def one(row_dict: dict[str, Any]) -> dict[str, Any]:
        row = pd.Series(row_dict)
        history = history_by_type.get(str(row["bond_type"]), full_panel.iloc[0:0])
        try:
            result = run_lsmc_row(
                row,
                history,
                config,
                path_models=path_models,
                regression_model=regression_model,
                paths=effective_paths,
            )
        except Exception as exc:
            LOGGER.warning(
                "LSMC row failed bond=%s date=%s: %s",
                row.get("bond_code"),
                row.get("trade_date"),
                exc,
            )
            price = pd.to_numeric(row.get("bond_close"), errors="coerce")
            friction = pd.to_numeric(row.get("total_friction_bps"), errors="coerce")
            immediate = (
                float(price) * (1 - float(friction) / 10000)
                if np.isfinite(price) and np.isfinite(friction)
                else np.nan
            )
            result = LSMCResult(
                immediate,
                np.nan,
                np.nan,
                0.0,
                0.0,
                "WATCH_ONLY",
                np.nan,
                0.0,
                0,
                f"LSMC_RUNTIME_ERROR:{type(exc).__name__}",
                "",
            )
        return _lsmc_result_dict(result, row, config)

    start = time.perf_counter()
    rows = target.to_dict("records")
    if workers == 1 or len(rows) < 10:
        results = [one(row) for row in rows]
    else:
        worker_count = effective_n_jobs(workers)
        chunk_count = min(
            len(rows),
            worker_count * int(config["strategy"].get("parallel_chunks_per_worker", 2)),
        )
        chunks = [
            list(chunk)
            for chunk in np.array_split(np.asarray(rows, dtype=object), chunk_count)
            if len(chunk)
        ]

        def run_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [one(row) for row in chunk]

        backend = str(config["strategy"].get("parallel_backend", "loky"))
        with parallel_config(backend=backend, inner_max_num_threads=1):
            nested = Parallel(n_jobs=workers, batch_size=1)(
                delayed(run_chunk)(chunk) for chunk in chunks
            )
        results = [item for chunk in nested for item in chunk]
    snapshot = pd.DataFrame(results)
    out = output_dir(config)
    snapshot_path = out / output_filename
    write_csv(snapshot, snapshot_path)
    if output_filename == "cb_lsmc_daily_snapshot.csv":
        write_csv(snapshot, out / "lsmc_snapshot.csv")
    diagnostics = (
        snapshot.groupby(["bond_type", "lsmc_recommended_action"], dropna=False)
        .agg(
            sample_count=("bond_code", "size"),
            hold_edge_mean=("lsmc_hold_edge", "mean"),
            hold_probability_mean=("lsmc_hold_probability", "mean"),
            expected_holding_days_mean=("lsmc_expected_holding_days", "mean"),
            regression_r2_mean=("lsmc_regression_r2", "mean"),
            confidence_mean=("lsmc_confidence", "mean"),
        )
        .reset_index()
    )
    diagnostics_name = (
        "lsmc_diagnostics.csv"
        if output_filename == "cb_lsmc_daily_snapshot.csv"
        else f"{Path(output_filename).stem}_diagnostics.csv"
    )
    write_csv(diagnostics, out / diagnostics_name)
    return {
        "rows": int(len(snapshot)),
        "eligible_evaluated": int((snapshot["lsmc_path_count"] > 0).sum()),
        "excluded_or_insufficient": int((snapshot["lsmc_path_count"] == 0).sum()),
        "recommended_action_counts": snapshot["lsmc_recommended_action"].value_counts().to_dict(),
        "path_count_per_model_row": effective_paths * len(path_models),
        "runtime_seconds": time.perf_counter() - start,
        "output": str(snapshot_path),
    }


def select_horizon(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    paths: int | None,
    refresh: bool,
    include_screening_extension: bool,
) -> dict[str, Any]:
    panel = load_priced_or_build(
        config, start_date, end_date, paths, refresh, include_screening_extension
    )
    lsmc_path = output_dir(config) / "cb_lsmc_daily_snapshot.csv"
    if not lsmc_path.exists():
        raise RuntimeError("Run --mode run-lsmc before select-horizon")
    lsmc = date_filter(read_csv_dates(lsmc_path), start_date, end_date)
    merge_columns = [
        "trade_date",
        "bond_code",
        "lsmc_immediate_liquidation_value",
        "lsmc_continuation_value",
        "lsmc_hold_edge",
        "lsmc_hold_edge_pct",
        "lsmc_hold_probability",
        "lsmc_expected_holding_days",
        "lsmc_recommended_action",
        "lsmc_confidence",
        "lsmc_path_count",
        "lsmc_regression_r2",
        "lsmc_exclusion_reason",
    ]
    merged = panel.merge(
        lsmc[[c for c in merge_columns if c in lsmc.columns]],
        on=["trade_date", "bond_code"],
        how="left",
        suffixes=("", "_lsmc"),
    )
    decisions = select_trade_horizon(merged, config)
    decisions["decision_time"] = decisions["trade_date"] + pd.Timedelta(hours=14, minutes=50)
    decisions["current_position_state"] = "FLAT_RESEARCH_ASSUMPTION"
    decisions["current_weight"] = 0.0
    max_positions = int(config["daily"]["max_positions"])
    decisions["target_weight"] = np.where(
        decisions["selected_horizon"] == "OVERNIGHT_DAILY",
        min(1 / max_positions, float(config["portfolio"]["max_weight_per_bond"])),
        0.0,
    )
    keep = [
        "decision_time",
        "trade_date",
        "bond_code",
        "stock_code",
        "bond_type",
        "bond_open",
        "bond_close",
        "current_position_state",
        "selected_horizon",
        "expected_intraday_edge_bps",
        "intraday_net_edge_bps",
        "daily_hold_edge_bps",
        "total_friction_bps",
        "commission_bps",
        "spread_proxy_bps",
        "slippage_bps",
        "market_impact_bps",
        "liquidity_risk_bps",
        "lsmc_immediate_liquidation_value",
        "lsmc_continuation_value",
        "lsmc_hold_edge",
        "lsmc_hold_probability",
        "lsmc_expected_holding_days",
        "lsmc_recommended_action",
        "carry_overnight_action",
        "loss_to_overnight_prohibition",
        "recommended_action",
        "current_weight",
        "target_weight",
        "reason_text",
        "risk_text",
        "research_only",
        "version",
        "next_open",
        "next_close",
        "next_trade_date",
        "next_open_to_close_return",
        "forward_return_1d",
        "mfe_5d",
        "mae_5d",
        "liquidity_regime",
        "liquidity_sweet_spot_score",
        "hot_money_risk_score",
        "intraday_eligible",
        "daily_holding_eligible",
        "lsmc_eligible",
        "qiv_microstructure_contaminated",
        "qiv_relative_value_score",
        "mc_price_gap_pct",
        "residual_z_daily",
        "residual_std_20d",
        "stock_return_1d",
        "research_pool_scope",
    ]
    decisions_out = decisions[[c for c in keep if c in decisions.columns]].copy()
    out = output_dir(config)
    write_csv(decisions_out, out / "cb_hybrid_trade_decision.csv")
    write_csv(decisions_out, out / "hybrid_decisions.csv")
    return {
        "rows": int(len(decisions_out)),
        "selected_horizon_counts": decisions_out["selected_horizon"].value_counts().to_dict(),
        "carry_action_counts": decisions_out["carry_overnight_action"].value_counts().to_dict(),
        "output": str(out / "cb_hybrid_trade_decision.csv"),
    }


def _execution_cost_bps(
    trades: pd.DataFrame,
    nominal_cost_bps: float,
    config: dict[str, Any],
    include_liquidity_friction: bool = True,
) -> pd.Series:
    stressed = nominal_cost_bps * float(config["friction"]["stress_multiplier"])
    if not include_liquidity_friction:
        return pd.Series(stressed, index=trades.index)
    model_total = pd.to_numeric(trades["total_friction_bps"], errors="coerce").fillna(stressed)
    commission_model = pd.to_numeric(trades.get("commission_bps", 0), errors="coerce").fillna(0)
    non_fee_friction = (model_total - commission_model).clip(lower=0)
    return stressed + non_fee_friction


def _strategy_daily_returns(
    decisions: pd.DataFrame,
    strategy: str,
    cost_bps: float,
    config: dict[str, Any],
    include_liquidity_friction: bool = True,
    use_lsmc: bool = True,
    use_bond_type: bool = True,
    include_clause: bool = False,
    momentum_weight: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = decisions.copy()
    if not include_clause:
        frame = frame[frame["bond_type"] != "CLAUSE_DRIVEN"]
    if not use_bond_type:
        frame.loc[frame["bond_type"] != "ILLIQUID", "bond_type"] = "UNIFIED"
    intraday_daily, intraday_trades = _intraday_trade_returns(
        frame,
        cost_bps,
        config,
        include_liquidity_friction,
        momentum_weight,
    )
    daily_daily, holding_trades = _daily_holding_returns(
        frame,
        cost_bps,
        config,
        include_liquidity_friction,
        use_lsmc,
        hybrid_only=strategy == "hybrid",
    )
    if strategy == "intraday":
        return intraday_daily, intraday_trades
    if strategy == "daily_lsmc":
        return daily_daily, holding_trades
    all_dates = pd.Index(
        sorted(
            set(pd.to_datetime(intraday_daily.get("trade_date", pd.Series(dtype="datetime64[ns]"))).dropna())
            | set(pd.to_datetime(daily_daily.get("trade_date", pd.Series(dtype="datetime64[ns]"))).dropna())
        )
    )
    if all_dates.empty:
        return pd.DataFrame(columns=["trade_date", "net_return"]), pd.DataFrame()
    intra = intraday_daily.set_index("trade_date") if not intraday_daily.empty else pd.DataFrame(index=all_dates)
    hold = daily_daily.set_index("trade_date") if not daily_daily.empty else pd.DataFrame(index=all_dates)
    hybrid = pd.DataFrame(index=all_dates)
    intra_weight = float(config["portfolio"]["intraday_capital_fraction"])
    daily_weight = float(config["portfolio"]["daily_capital_fraction"])
    hybrid["net_return"] = (
        intra.get("net_return", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * intra_weight
        + hold.get("net_return", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * daily_weight
    )
    hybrid["gross_return"] = (
        intra.get("gross_return", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * intra_weight
        + hold.get("gross_return", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * daily_weight
    )
    hybrid["trade_count"] = (
        intra.get("trade_count", pd.Series(0, index=all_dates)).reindex(all_dates).fillna(0)
        + hold.get("trade_count", pd.Series(0, index=all_dates)).reindex(all_dates).fillna(0)
    )
    hybrid["turnover"] = (
        intra.get("turnover", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * intra_weight
        + hold.get("turnover", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * daily_weight
    )
    hybrid["friction_bps"] = (
        intra.get("friction_bps", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * intra_weight
        + hold.get("friction_bps", pd.Series(0.0, index=all_dates)).reindex(all_dates).fillna(0) * daily_weight
    )
    hybrid = hybrid.rename_axis("trade_date").reset_index()
    trades = pd.concat([intraday_trades, holding_trades], ignore_index=True, sort=False)
    trades["strategy"] = "hybrid"
    return hybrid, trades


def _intraday_trade_returns(
    frame: pd.DataFrame,
    cost_bps: float,
    config: dict[str, Any],
    include_liquidity_friction: bool,
    momentum_weight: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame[frame["selected_horizon"] == "INTRADAY_T0"].copy()
    if momentum_weight is not None and momentum_weight > float(config["intraday"]["momentum_weight"]):
        momentum = pd.to_numeric(selected.get("stock_return_1d", 0), errors="coerce").fillna(0)
        selected = selected[momentum > 0].copy()
    if selected.empty:
        return pd.DataFrame(columns=["trade_date", "net_return"]), selected
    selected["execution_cost_bps"] = _execution_cost_bps(
        selected, cost_bps, config, include_liquidity_friction
    )
    selected["gross_return"] = pd.to_numeric(selected["next_open_to_close_return"], errors="coerce")
    selected["net_return"] = selected["gross_return"] - selected["execution_cost_bps"] / 10000
    selected["execution_date"] = pd.to_datetime(selected.get("next_trade_date"), errors="coerce")
    selected["execution_date"] = selected["execution_date"].fillna(
        selected["trade_date"] + pd.offsets.BDay(1)
    )
    selected["holding_return_until_lsmc_exit"] = selected["net_return"]
    selected["realized_holding_days"] = 0
    selected["strategy"] = "intraday"
    selected["cost_bps_nominal"] = cost_bps
    selected["strategy_version"] = config["strategy"]["version"]
    selected["config_version"] = config["strategy"]["config_version"]
    daily = (
        selected.dropna(subset=["net_return"])
        .groupby("execution_date", as_index=False)
        .agg(
            net_return=("net_return", "mean"),
            gross_return=("gross_return", "mean"),
            trade_count=("bond_code", "size"),
            turnover=("bond_code", lambda s: float(len(s) * 2)),
            friction_bps=("execution_cost_bps", "mean"),
        )
        .rename(columns={"execution_date": "trade_date"})
    )
    daily["turnover"] = (
        daily["turnover"] / max(int(config["daily"]["max_positions"]), 1)
    ).clip(upper=2.0)
    return daily, selected


def _daily_holding_returns(
    frame: pd.DataFrame,
    cost_bps: float,
    config: dict[str, Any],
    include_liquidity_friction: bool,
    use_lsmc: bool,
    hybrid_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes: list[dict[str, Any]] = []
    marks: list[dict[str, Any]] = []
    max_days = int(config["daily"]["max_expected_holding_days"])
    for bond_code, group in frame.groupby("bond_code", sort=False):
        g = group.sort_values("trade_date").reset_index(drop=True).copy()
        if len(g) < 2:
            continue
        if use_lsmc:
            open_mask = (
                (g["lsmc_recommended_action"] == "HOLD")
                & (pd.to_numeric(g["daily_hold_edge_bps"], errors="coerce") > 0)
            )
        else:
            open_mask = (
                (pd.to_numeric(g.get("qiv_relative_value_score", np.nan), errors="coerce") > 0)
                | (pd.to_numeric(g.get("residual_z_daily", np.nan), errors="coerce") < -1)
            )
        if hybrid_only:
            open_mask &= g["selected_horizon"] == "OVERNIGHT_DAILY"
        i = 0
        while i < len(g) - 1:
            if not bool(open_mask.iloc[i]) or not np.isfinite(pd.to_numeric(g.loc[i, "next_open"], errors="coerce")):
                i += 1
                continue
            entry_price = float(g.loc[i, "next_open"])
            entry_date = pd.Timestamp(g.loc[i, "next_trade_date"])
            max_signal_idx = min(i + max_days, len(g) - 2)
            exit_signal_idx = max_signal_idx
            exit_reason = "MAX_HOLDING_DAYS"
            for j in range(i + 1, max_signal_idx + 1):
                if use_lsmc:
                    should_exit = (
                        str(g.loc[j, "lsmc_recommended_action"]) == "SELL"
                        or pd.to_numeric(g.loc[j, "daily_hold_edge_bps"], errors="coerce") <= 0
                    )
                else:
                    should_exit = not bool(open_mask.iloc[j])
                if should_exit:
                    exit_signal_idx = j
                    exit_reason = "LSMC_EXIT" if use_lsmc else "RULE_EXIT"
                    break
            exit_price = pd.to_numeric(g.loc[exit_signal_idx, "next_open"], errors="coerce")
            exit_date = pd.to_datetime(g.loc[exit_signal_idx, "next_trade_date"], errors="coerce")
            if not np.isfinite(exit_price) or pd.isna(exit_date):
                exit_price = float(g.loc[exit_signal_idx, "bond_close"])
                exit_date = pd.Timestamp(g.loc[exit_signal_idx, "trade_date"])
                exit_reason = "END_OF_SAMPLE_CLOSE"
            entry_cost = float(
                _execution_cost_bps(
                    g.iloc[[i]], cost_bps, config, include_liquidity_friction
                ).iloc[0]
            )
            exit_cost = float(
                _execution_cost_bps(
                    g.iloc[[exit_signal_idx]], cost_bps, config, include_liquidity_friction
                ).iloc[0]
            )
            cumulative = 1.0
            cumulative_path: list[float] = []
            first_daily_idx = i + 1
            last_close_idx = exit_signal_idx
            for k in range(first_daily_idx, last_close_idx + 1):
                if k == first_daily_idx:
                    gross = float(g.loc[k, "bond_close"]) / entry_price - 1
                    net = gross - entry_cost / 10000
                    turnover = 1.0
                    friction_for_day = entry_cost
                else:
                    gross = float(g.loc[k, "bond_close"]) / float(g.loc[k - 1, "bond_close"]) - 1
                    net = gross
                    turnover = 0.0
                    friction_for_day = 0.0
                cumulative *= 1 + net
                cumulative_path.append(cumulative - 1)
                marks.append(
                    {
                        "trade_date": pd.Timestamp(g.loc[k, "trade_date"]),
                        "bond_code": bond_code,
                        "net_return": net,
                        "gross_return": gross,
                        "turnover": turnover,
                        "friction_bps": friction_for_day,
                    }
                )
            if exit_date > pd.Timestamp(g.loc[last_close_idx, "trade_date"]):
                gross = float(exit_price) / float(g.loc[last_close_idx, "bond_close"]) - 1
                net = gross - exit_cost / 10000
                cumulative *= 1 + net
                cumulative_path.append(cumulative - 1)
                marks.append(
                    {
                        "trade_date": exit_date,
                        "bond_code": bond_code,
                        "net_return": net,
                        "gross_return": gross,
                        "turnover": 1.0,
                        "friction_bps": exit_cost,
                    }
                )
            else:
                if cumulative_path:
                    cumulative /= 1 + marks[-1]["net_return"]
                    replacement = marks[-1]["gross_return"] - exit_cost / 10000
                    cumulative *= 1 + replacement
                    marks[-1]["net_return"] = replacement
                    marks[-1]["turnover"] += 1.0
                    marks[-1]["friction_bps"] += exit_cost
                    cumulative_path[-1] = cumulative - 1
            episode = g.loc[i].to_dict()
            episode.update(
                {
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": float(exit_price),
                    "exit_reason": exit_reason,
                    "gross_return": float(exit_price / entry_price - 1),
                    "net_return": cumulative - 1,
                    "holding_return_until_lsmc_exit": cumulative - 1,
                    "realized_holding_days": max(1, exit_signal_idx - i),
                    "realized_mfe": max(cumulative_path) if cumulative_path else np.nan,
                    "realized_mae": min(cumulative_path) if cumulative_path else np.nan,
                    "execution_cost_bps": entry_cost + exit_cost,
                    "strategy": "daily_lsmc",
                    "cost_bps_nominal": cost_bps,
                    "strategy_version": config["strategy"]["version"],
                    "config_version": config["strategy"]["config_version"],
                }
            )
            episodes.append(episode)
            i = exit_signal_idx + 1
    trades = pd.DataFrame(episodes)
    if not marks:
        return pd.DataFrame(columns=["trade_date", "net_return"]), trades
    mark_frame = pd.DataFrame(marks)
    daily = (
        mark_frame.groupby("trade_date", as_index=False)
        .agg(
            net_return=("net_return", "mean"),
            gross_return=("gross_return", "mean"),
            trade_count=("bond_code", "nunique"),
            turnover=("turnover", "sum"),
            friction_bps=("friction_bps", "mean"),
        )
    )
    daily["turnover"] = (
        daily["turnover"] / max(int(config["daily"]["max_positions"]), 1)
    ).clip(upper=2.0)
    return daily, trades


def run_backtest(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    strategies: list[str],
    costs: list[float],
) -> dict[str, Any]:
    decisions_path = output_dir(config) / "cb_hybrid_trade_decision.csv"
    if not decisions_path.exists():
        raise RuntimeError("Run --mode select-horizon before backtest")
    decisions = date_filter(read_csv_dates(decisions_path), start_date, end_date)
    rows: list[dict[str, Any]] = []
    trade_rows: list[pd.DataFrame] = []
    for cost in costs:
        for strategy in strategies:
            daily, trades = _strategy_daily_returns(decisions, strategy, cost, config)
            metrics = portfolio_metrics(daily["net_return"], daily.get("turnover"))
            rows.append(
                {
                    "strategy": strategy,
                    "cost_bps_nominal": cost,
                    "cost_stress_multiplier": config["friction"]["stress_multiplier"],
                    "trade_count": int(len(trades)),
                    "average_holding_days": float(
                        pd.to_numeric(
                            trades.get(
                                "realized_holding_days",
                                trades.get("lsmc_expected_holding_days"),
                            ),
                            errors="coerce",
                        ).mean()
                    )
                    if not trades.empty
                    else np.nan,
                    "overnight_exposure_count": int(
                        (trades.get("selected_horizon", pd.Series(dtype=str)) == "OVERNIGHT_DAILY").sum()
                    ),
                    **metrics,
                    "strategy_version": config["strategy"]["version"],
                    "config_version": config["strategy"]["config_version"],
                }
            )
            if not trades.empty:
                trade_rows.append(trades)
    report = pd.DataFrame(rows)
    all_trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    out = output_dir(config)
    write_csv(report, out / "intraday_vs_daily_vs_hybrid.csv")
    write_csv(report, out / "cost_sensitivity.csv")
    if not all_trades.empty:
        write_csv(all_trades, out / "hybrid_trade_episodes.csv")
        all_trades["evaluation_holding_days"] = pd.to_numeric(
            all_trades.get("realized_holding_days"), errors="coerce"
        ).combine_first(
            pd.to_numeric(all_trades.get("lsmc_expected_holding_days"), errors="coerce")
        )
        all_trades["evaluation_mfe"] = pd.to_numeric(
            all_trades.get("realized_mfe"), errors="coerce"
        ).combine_first(pd.to_numeric(all_trades.get("mfe_5d"), errors="coerce"))
        all_trades["evaluation_mae"] = pd.to_numeric(
            all_trades.get("realized_mae"), errors="coerce"
        ).combine_first(pd.to_numeric(all_trades.get("mae_5d"), errors="coerce"))
        type_perf = (
            all_trades.groupby(["strategy", "cost_bps_nominal", "bond_type"], dropna=False)
            .agg(
                sample_count=("bond_code", "size"),
                average_return=("net_return", "mean"),
                hit_rate=("net_return", lambda s: float((s > 0).mean())),
                friction=("execution_cost_bps", "mean"),
                mfe=("evaluation_mfe", "mean"),
                mae=("evaluation_mae", "mean"),
                turnover=("target_weight", lambda s: float(np.abs(pd.to_numeric(s, errors="coerce")).sum())),
                lsmc_hold_edge=("lsmc_hold_edge", "mean"),
                hold_probability=("lsmc_hold_probability", "mean"),
                average_holding_days=("evaluation_holding_days", "mean"),
            )
            .reset_index()
        )
        risk_rows = []
        for keys, group in all_trades.groupby(
            ["strategy", "cost_bps_nominal", "bond_type"], dropna=False
        ):
            ordered = group.sort_values(
                [c for c in ["exit_date", "entry_date", "trade_date"] if c in group.columns]
            )
            returns = pd.to_numeric(ordered["net_return"], errors="coerce").dropna()
            if returns.empty:
                max_drawdown = worst_period = np.nan
            else:
                curve = (1 + returns).cumprod()
                max_drawdown = float((curve / curve.cummax() - 1).min())
                worst_period = float(returns.min())
            risk_rows.append(
                {
                    "strategy": keys[0],
                    "cost_bps_nominal": keys[1],
                    "bond_type": keys[2],
                    "max_drawdown": max_drawdown,
                    "worst_period": worst_period,
                }
            )
        type_perf = type_perf.merge(
            pd.DataFrame(risk_rows),
            on=["strategy", "cost_bps_nominal", "bond_type"],
            how="left",
        )
    else:
        type_perf = pd.DataFrame()
    write_csv(type_perf, out / "type_performance.csv")
    event_benchmark = build_intraday_event_benchmark(config, costs)
    write_csv(event_benchmark, out / "intraday_event_benchmark.csv")
    return {
        "rows": int(len(report)),
        "trade_rows": int(len(all_trades)),
        "strategies": strategies,
        "costs": costs,
        "output": str(out / "intraday_vs_daily_vs_hybrid.csv"),
        "intraday_event_benchmark_rows": int(len(event_benchmark)),
    }


def build_intraday_event_benchmark(
    config: dict[str, Any],
    costs: list[float],
) -> pd.DataFrame:
    path = PROJECT_DIR / "outputs" / "ml_shadow" / "cb_intraday_ml_event_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    events = pd.read_csv(path, low_memory=False)
    official = {"LAG_REPAIR_LONG", "MOMENTUM_BREAKOUT_LONG"}
    late_raw = events.get("late_signal_shadow_only", pd.Series(False, index=events.index))
    if late_raw.dtype == object:
        late = late_raw.astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        late = late_raw.fillna(False).astype(bool)
    actions = events[
        (events["event_type"].astype(str) == "ACTION")
        & events["signal_type"].astype(str).isin(official)
        & ~late
    ].copy()
    if actions.empty:
        return pd.DataFrame()
    rows = []
    for cost in costs:
        stress_cost = float(cost) * float(config["friction"]["stress_multiplier"])
        returns = pd.to_numeric(actions["return_30m"], errors="coerce")
        net = returns - stress_cost / 10000
        valid = net.dropna()
        rows.append(
            {
                "evidence_layer": "v04_5min_action_events",
                "cost_bps_nominal": cost,
                "cost_stress_multiplier": config["friction"]["stress_multiplier"],
                "action_count": int(len(actions)),
                "label_available_count": int(valid.notna().sum()),
                "mean_return_30m_after_stressed_fee": float(valid.mean()) if not valid.empty else np.nan,
                "median_return_30m_after_stressed_fee": float(valid.median()) if not valid.empty else np.nan,
                "hit_rate_30m_after_stressed_fee": float((valid > 0).mean()) if not valid.empty else np.nan,
                "mfe_6bar_mean": float(pd.to_numeric(actions["mfe_6bar"], errors="coerce").mean()),
                "mae_6bar_mean": float(pd.to_numeric(actions["mae_6bar"], errors="coerce").mean()),
                "first_trade_date": actions["trade_date"].min(),
                "last_trade_date": actions["trade_date"].max(),
                "signal_type_counts": json.dumps(
                    actions["signal_type"].value_counts().to_dict(), sort_keys=True
                ),
                "data_mode_counts": json.dumps(
                    actions["data_mode"].value_counts(dropna=False).to_dict(), sort_keys=True
                ),
                "friction_scope": "stressed_nominal_fee_only;daily_liquidity_model_not_joined",
                "comparable_to_daily_backtest": False,
                "strategy_version": config["strategy"]["version"],
                "config_version": config["strategy"]["config_version"],
            }
        )
    return pd.DataFrame(rows)


def run_ablation(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    decisions_path = output_dir(config) / "cb_hybrid_trade_decision.csv"
    if not decisions_path.exists():
        raise RuntimeError("Run --mode select-horizon before ablation")
    decisions = date_filter(read_csv_dates(decisions_path), start_date, end_date)
    base_cost = 10.0
    scenarios = [
        ("A_NO_LSMC", "hybrid", {"use_lsmc": False}),
        ("B_WITH_LSMC", "hybrid", {"use_lsmc": True}),
        ("C_NO_QIV_MC", "hybrid", {"use_lsmc": True}),
        ("D_NO_LIQUIDITY_FRICTION", "hybrid", {"include_liquidity_friction": False}),
        ("E_NO_BOND_TYPE", "hybrid", {"use_bond_type": False}),
        ("F_INCLUDE_CLAUSE_DRIVEN", "hybrid", {"include_clause": True}),
        ("G_HIGH_MOMENTUM_WEIGHT", "hybrid", {"momentum_weight": 0.25}),
        ("H_LOW_MOMENTUM_WEIGHT", "hybrid", {"momentum_weight": float(config["intraday"]["momentum_weight"])}),
        ("I_INTRADAY_ONLY", "intraday", {}),
        ("J_DAILY_ONLY", "daily_lsmc", {}),
        ("K_HYBRID_HORIZON", "hybrid", {}),
    ]
    rows = []
    for name, strategy, kwargs in scenarios:
        scenario_decisions = decisions.copy()
        identified = True
        note = ""
        if name == "A_NO_LSMC":
            rule_daily = (
                scenario_decisions["daily_holding_eligible"].fillna(False)
                & (scenario_decisions["liquidity_regime"] != "ILLIQUID")
                & ~scenario_decisions["qiv_microstructure_contaminated"].fillna(False)
                & (
                    (
                        pd.to_numeric(
                            scenario_decisions["qiv_relative_value_score"], errors="coerce"
                        )
                        > 0
                    )
                    | (
                        pd.to_numeric(
                            scenario_decisions["residual_z_daily"], errors="coerce"
                        )
                        < -1
                    )
                )
            )
            preserve_intraday = scenario_decisions["selected_horizon"].eq("INTRADAY_T0")
            excluded_type = scenario_decisions["bond_type"].isin(["CLAUSE_DRIVEN", "ILLIQUID"])
            scenario_decisions["selected_horizon"] = np.select(
                [excluded_type, preserve_intraday, rule_daily],
                ["EXCLUDE", "INTRADAY_T0", "OVERNIGHT_DAILY"],
                default="WATCH_ONLY",
            )
            note = "Rule RV/residual daily baseline rebuilt without LSMC hold decisions."
        if name == "C_NO_QIV_MC":
            had_qiv = int(
                pd.to_numeric(
                    scenario_decisions["qiv_relative_value_score"], errors="coerce"
                ).notna().sum()
            )
            scenario_decisions["qiv_relative_value_score"] = np.nan
            scenario_decisions["mc_price_gap_pct"] = np.nan
            no_qiv_path = output_dir(config) / "lsmc_no_qiv_mc_ablation_snapshot.csv"
            if had_qiv == 0:
                identified = False
                note = "Historical PIT qIV coverage is zero in this evaluation slice."
            elif no_qiv_path.exists():
                no_qiv = date_filter(read_csv_dates(no_qiv_path), start_date, end_date)
                replacement_columns = [
                    "lsmc_immediate_liquidation_value",
                    "lsmc_continuation_value",
                    "lsmc_hold_edge",
                    "lsmc_hold_probability",
                    "lsmc_expected_holding_days",
                    "lsmc_recommended_action",
                ]
                scenario_decisions = scenario_decisions.drop(
                    columns=replacement_columns, errors="ignore"
                ).merge(
                    no_qiv[
                        ["trade_date", "bond_code"]
                        + [column for column in replacement_columns if column in no_qiv.columns]
                    ],
                    on=["trade_date", "bond_code"],
                    how="left",
                    validate="one_to_one",
                )
                scenario_decisions["qiv_microstructure_contaminated"] = False
                scenario_decisions = select_trade_horizon(scenario_decisions, config)
                note = (
                    "Full research ablation uses empirical-bootstrap-only LSMC continuation, "
                    "removes qIV/MC entry fields, and recomputes the horizon selector."
                )
            else:
                scenario_decisions = select_trade_horizon(scenario_decisions, config)
                identified = False
                note = "No empirical-only LSMC ablation snapshot is available."
        if name == "E_NO_BOND_TYPE":
            unified_hold = (
                (scenario_decisions["lsmc_recommended_action"] == "HOLD")
                & (pd.to_numeric(scenario_decisions["daily_hold_edge_bps"], errors="coerce") > 0)
                & (scenario_decisions["liquidity_regime"] != "ILLIQUID")
            )
            scenario_decisions.loc[unified_hold, "selected_horizon"] = "OVERNIGHT_DAILY"
            note = "Selector/type-gate ablation; continuation regressions remain type-conditioned."
        if name == "F_INCLUDE_CLAUSE_DRIVEN" and not (
            scenario_decisions["bond_type"] == "CLAUSE_DRIVEN"
        ).any():
            identified = False
            note = "No point-in-time CLAUSE_DRIVEN sample is available."
        if name == "G_HIGH_MOMENTUM_WEIGHT":
            old_momentum = (
                (pd.to_numeric(scenario_decisions["stock_return_1d"], errors="coerce") > 0.015)
                & (scenario_decisions["liquidity_regime"] == "LIQUIDITY_SWEET_SPOT")
                & ~scenario_decisions["bond_type"].isin(["CLAUSE_DRIVEN", "ILLIQUID"])
            )
            scenario_decisions.loc[old_momentum, "selected_horizon"] = "INTRADAY_T0"
            note = "Old-style monitoring baseline permits high momentum to dominate entry."
        daily, trades = _strategy_daily_returns(
            scenario_decisions,
            strategy,
            base_cost,
            config,
            **kwargs,
        )
        rows.append(
            {
                "ablation": name,
                "strategy": strategy,
                "trade_count": int(len(trades)),
                **portfolio_metrics(daily["net_return"], daily.get("turnover")),
                "cost_bps_nominal": base_cost,
                "cost_stress_multiplier": config["friction"]["stress_multiplier"],
                "ablation_identified": identified,
                "notes": note,
                "strategy_version": config["strategy"]["version"],
                "config_version": config["strategy"]["config_version"],
            }
        )
    report = pd.DataFrame(rows)
    path = output_dir(config) / "ablation_report.csv"
    write_csv(report, path)
    return {"rows": int(len(report)), "output": str(path)}


def _fmt_pct(value: Any) -> str:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if pd.isna(value) else f"{float(value):.3%}"


def build_report(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    out = output_dir(config)
    classification = read_csv_dates(out / "cb_bond_type_snapshot.csv")
    pricing_all = read_csv_dates(out / "cb_daily_relative_value_snapshot.csv")
    pricing = pricing_all.copy()
    lsmc = read_csv_dates(out / "cb_lsmc_daily_snapshot.csv")
    decisions = read_csv_dates(out / "cb_hybrid_trade_decision.csv")
    performance = pd.read_csv(out / "intraday_vs_daily_vs_hybrid.csv")
    event_benchmark_path = out / "intraday_event_benchmark.csv"
    event_benchmark = (
        pd.read_csv(event_benchmark_path) if event_benchmark_path.exists() else pd.DataFrame()
    )
    ablation = pd.read_csv(out / "ablation_report.csv")
    classification = date_filter(classification, start_date, end_date)
    pricing = date_filter(pricing, start_date, end_date)
    lsmc = date_filter(lsmc, start_date, end_date)
    decisions = date_filter(decisions, start_date, end_date)
    latest_date = decisions["trade_date"].max()
    latest_types = classification[classification["trade_date"] == classification["trade_date"].max()]
    solved = pricing[pricing["qiv_solve_success"].fillna(False)]
    solved_all = pricing_all[pricing_all["qiv_solve_success"].fillna(False)]
    contamination_assessable_all = solved_all[
        solved_all.get(
            "qiv_contamination_assessable",
            pd.Series(False, index=solved_all.index),
        ).fillna(False)
    ]
    conversion_covered = pricing_all.get(
        "conversion_price_asof_valid", pd.Series(False, index=pricing_all.index)
    ).fillna(False)
    stock_pricing_covered = pricing_all.get(
        "qiv_stock_close_asof_valid", pd.Series(False, index=pricing_all.index)
    ).fillna(False)
    joint_pricing_covered = pricing_all.get(
        "pricing_inputs_asof_valid", pd.Series(False, index=pricing_all.index)
    ).fillna(False)
    eligible = lsmc[lsmc["lsmc_path_count"] > 0]
    holds = decisions[decisions["selected_horizon"] == "OVERNIGHT_DAILY"]
    intraday = decisions[decisions["selected_horizon"] == "INTRADAY_T0"]
    excluded = decisions[decisions["selected_horizon"] == "EXCLUDE"]
    cost_10 = performance[
        pd.to_numeric(performance["cost_bps_nominal"], errors="coerce").eq(10.0)
    ]
    cost_10_summary = ", ".join(
        f"{row['strategy']}={_fmt_pct(row['total_return'])} ({int(row['trade_count'])} trades)"
        for _, row in cost_10.iterrows()
    )
    ablation_index = ablation.set_index("ablation") if not ablation.empty else pd.DataFrame()
    lsmc_delta_text = "unavailable"
    qiv_delta_text = "unavailable"
    qiv_ablation_status = "not identified"
    if {
        "A_NO_LSMC",
        "B_WITH_LSMC",
    }.issubset(set(ablation_index.index)):
        lsmc_delta = (
            float(ablation_index.loc["B_WITH_LSMC", "total_return"])
            - float(ablation_index.loc["A_NO_LSMC", "total_return"])
        )
        lsmc_delta_text = _fmt_pct(lsmc_delta)
    if {
        "B_WITH_LSMC",
        "C_NO_QIV_MC",
    }.issubset(set(ablation_index.index)):
        qiv_delta = (
            float(ablation_index.loc["B_WITH_LSMC", "total_return"])
            - float(ablation_index.loc["C_NO_QIV_MC", "total_return"])
        )
        qiv_delta_text = _fmt_pct(qiv_delta)
        qiv_ablation_status = (
            "identified with empirical-bootstrap-only LSMC"
            if bool(ablation_index.loc["C_NO_QIV_MC", "ablation_identified"])
            else "not identified"
        )

    lines = [
        "# Convertible Bond Hybrid Strategy V0.7 Research Report",
        "",
        f"- Strategy version: `{config['strategy']['version']}`",
        f"- Config version: `{config['strategy']['config_version']}`",
        f"- Requested range: `{start_date or 'all'} .. {end_date or 'all'}`",
        f"- Latest decision date in outputs: `{latest_date.date() if pd.notna(latest_date) else 'NA'}`",
        "- Scope: research only; no automatic orders.",
        "",
        "## Data and point-in-time status",
        "",
        f"- Latest classified bonds: {latest_types['bond_code'].nunique()}",
        f"- qIV solved rows: {len(solved)} / {len(pricing)} ({len(solved) / len(pricing):.2%})" if len(pricing) else "- qIV solved rows: 0",
        f"- Full local panel qIV/MC coverage: {len(solved_all)} / {len(pricing_all)} ({len(solved_all) / len(pricing_all):.2%})" if len(pricing_all) else "- Full local panel qIV/MC coverage: 0",
        f"- qIV contamination assessable: {len(contamination_assessable_all)} / {len(solved_all)} solved rows; unavailable rows are not called clean.",
        f"- LSMC evaluated rows: {len(eligible)} / {len(lsmc)}",
        "- Reliable YTM coverage is zero. Discounting uses risk-free plus a documented credit-spread proxy.",
        f"- PIT conversion-price coverage: {int(conversion_covered.sum())} / {len(pricing_all)} ({conversion_covered.mean():.2%}).",
        f"- Same-day unadjusted stock-price coverage: {int(stock_pricing_covered.sum())} / {len(pricing_all)} ({stock_pricing_covered.mean():.2%}).",
        f"- Joint PIT pricing-input coverage: {int(joint_pricing_covered.sum())} / {len(pricing_all)} ({joint_pricing_covered.mean():.2%}). Missing rows remain unavailable and are never filled with future conversion prices or adjusted stock prices.",
        f"- The {classification['bond_code'].nunique()}-bond historical universe was selected in 2026-07, so historical results retain survivorship/selection bias.",
        "",
        "## Bond type snapshot",
        "",
        latest_types["bond_type"].value_counts().rename_axis("bond_type").to_frame("count").to_markdown(),
        "",
        "## Horizon decisions",
        "",
        decisions["selected_horizon"].value_counts().rename_axis("selected_horizon").to_frame("count").to_markdown(),
        "",
        f"- BALANCED_CORE HOLD examples: {int(((decisions['bond_type'] == 'BALANCED_CORE') & (decisions['selected_horizon'] == 'OVERNIGHT_DAILY')).sum())}",
        f"- INTRADAY_T0 examples: {len(intraday)}",
        f"- CARRY_OVERNIGHT examples: {int((decisions['carry_overnight_action'] == 'CARRY_OVERNIGHT').sum())}",
        f"- CLAUSE_DRIVEN/ILLIQUID excluded examples: {len(excluded)}",
        "",
        "## Portfolio comparison",
        "",
        performance.to_markdown(index=False),
        "",
        f"At 10 bps nominal cost, fees are stressed by 5x before liquidity impact. Results: {cost_10_summary}. These are research backtests, not deployment evidence.",
        "",
        "## Existing 5-minute ACTION benchmark",
        "",
        event_benchmark.to_markdown(index=False) if not event_benchmark.empty else "No V0.4 event dataset was available.",
        "",
        "This event benchmark reuses official V0.4 ACTION episodes and is not merged into the daily/Hybrid equity curve because its calendar differs. It applies the 5x nominal fee stress but does not pretend that the daily liquidity-friction model was observed at each old event.",
        "",
        "## Ablation",
        "",
        ablation.to_markdown(index=False),
        "",
        f"Ablations are evaluated after stressed costs. Adding LSMC changes total return versus the rebuilt no-LSMC rule baseline by {lsmc_delta_text}. Adding qIV/MC structural mapping versus empirical-only LSMC changes total return by {qiv_delta_text}; this ablation is {qiv_ablation_status}.",
        "",
        "## Research answers",
        "",
        "1. BALANCED_CORE and selected EQUITY_LIKE rows are the intended intraday candidates, but only when residual/RV edge clears stressed friction. Momentum alone never triggers a trade.",
        "2. BALANCED_CORE, BOND_LIKE, and selected EQUITY_LIKE rows are eligible for daily holding. BOND_LIKE is not treated as ordinary lag repair.",
        "3. CLAUSE_DRIVEN, ILLIQUID, and extreme SPECULATIVE_HIGH_PREMIUM rows are excluded from ordinary LSMC. DISTRESS_REVERSAL remains experimental.",
        "4. LSMC improves the no-LSMC prototype in this slice but does not produce positive cost-after performance. Positive simulated hold edge is therefore miscalibrated as a tradable forecast.",
        "5. Hybrid is not proven superior. It loses less than Daily LSMC at the tested costs, while Intraday has too few observations for a fair comparison.",
        "6. Type and per-date outputs should be checked for concentration; a small number of bonds/dates can dominate this limited sample.",
        "7. qIV/MC now have historical point-in-time inputs for most canonical rows, but independence from momentum still requires the reported ablation and walk-forward stability rather than coverage alone.",
        "8. qIV contamination is reported only among solved rows; unsolved rows are not silently treated as clean.",
        "9. Wind daily turnover is included when available and directly enters liquidity/friction states.",
        "10. Momentum weight is 0.05 and monitoring-only. The high-momentum ablation is worse in this slice, which supports the downgrade direction but not a final parameter.",
        "11. Clause data are unavailable historically, so clause-pollution conclusions cannot be made from the current dataset.",
        "12. DISTRESS_REVERSAL should remain a separate research framework because price dynamics and event information are not captured by ordinary residual models.",
        "13. LSMC is most sensitive to path residuals, volatility, friction buffers, horizon, and credit spread.",
        "14. Credit spread sensitivity is stored for -200/-100/0/+100/+200 bps, but it remains a proxy because YTM and complete terms are unavailable.",
        "15. Intraday positions should flatten when the repair edge is gone, friction rises, liquidity deteriorates, or no independent daily hold case exists.",
        "16. Carry overnight is allowed only when an independent LSMC hold edge clears rebalance, overnight-risk, and model-uncertainty buffers; current backtest calibration does not yet justify acting on it.",
        "17. `loss_to_overnight_prohibition=true` prevents a losing intraday trade from becoming an overnight position solely to avoid realizing a loss.",
        "18. Current data are sufficient to exercise the research pipeline, not to support deployment conclusions.",
        "19. Bond-type thresholds, credit spreads, qIV structural assumptions, and LSMC path mappings remain research hypotheses.",
        "",
        "## Current limitations",
        "",
        "- No reliable YTM, coupon schedule, call/put/reset history, or rating history. Conversion-price history is reconstructed point-in-time from AkShare conversion-value anchors, JQData unadjusted closes, and JQData effective adjustment events.",
        "- Structural MC is a discounted-par plus terminal-conversion proxy, not a complete convertible-bond pricer.",
        "- Historical universe construction is not point-in-time and has survivorship bias.",
        "- PIT conversion-price and unadjusted-stock support are not perfectly complete. Missing rows remain unavailable; qIV solve coverage is also constrained by the deliberately simplified proxy-pricer range.",
        "- Daily holding episodes enter at the next trading-day open, mark to market each day, and exit at the next executable open after an LSMC/rule exit; same-close execution is forbidden.",
        "- The common-slice `INTRADAY_T0` portfolio is an EOD-planned next-day T+0 proxy. Existing V0.4 five-minute ACTION events are reported separately because their dates do not overlap the daily training history.",
        "- Intraday event and daily evidence cover different calendar samples. Direct hybrid claims require a common, point-in-time dataset.",
    ]
    report_path = out / "hybrid_v07_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report": str(report_path),
        "latest_type_counts": latest_types["bond_type"].value_counts().to_dict(),
        "qiv_coverage": float(len(solved) / len(pricing)) if len(pricing) else np.nan,
        "lsmc_evaluated": int(len(eligible)),
        "decision_counts": decisions["selected_horizon"].value_counts().to_dict(),
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_hybrid_config(args.config)
    n_jobs = args.n_jobs if args.n_jobs is not None else int(config["strategy"].get("n_jobs", -1))
    paths = args.paths
    costs = [float(v) for v in parse_csv(args.cost_bps)]
    path_models = parse_csv(args.path_model)
    strategies = parse_csv(args.strategies)
    try:
        if args.mode == "classify-bonds":
            result = run_classification(
                config,
                args.start_date,
                args.end_date,
                args.refresh_cache,
                args.include_screening_extension,
            )
        elif args.mode == "price-daily":
            result = run_pricing(
                config,
                args.start_date,
                args.end_date,
                paths,
                args.refresh_cache,
                args.include_screening_extension,
            )
        elif args.mode == "build-lsmc-dataset":
            result = build_lsmc_dataset(
                config,
                args.start_date,
                args.end_date,
                paths,
                args.refresh_cache,
                args.include_screening_extension,
            )
        elif args.mode == "run-lsmc":
            result = run_lsmc(
                config,
                args.start_date,
                args.end_date,
                paths,
                args.backtest_paths,
                path_models,
                args.regression_model,
                n_jobs,
                args.refresh_cache,
                args.include_screening_extension,
            )
        elif args.mode == "run-lsmc-no-qiv-ablation":
            result = run_lsmc(
                config,
                args.start_date,
                args.end_date,
                paths or int(config["lsmc"]["backtest_simulation_paths"]),
                args.backtest_paths,
                ["empirical_bootstrap"],
                args.regression_model,
                n_jobs,
                args.refresh_cache,
                args.include_screening_extension,
                output_filename="lsmc_no_qiv_mc_ablation_snapshot.csv",
            )
        elif args.mode == "select-horizon":
            result = select_horizon(
                config,
                args.start_date,
                args.end_date,
                paths,
                args.refresh_cache,
                args.include_screening_extension,
            )
        elif args.mode == "backtest":
            result = run_backtest(
                config,
                args.start_date,
                args.end_date,
                strategies,
                costs,
            )
        elif args.mode == "ablation":
            result = run_ablation(config, args.start_date, args.end_date)
        elif args.mode == "report":
            result = build_report(config, args.start_date, args.end_date)
        else:
            results: dict[str, Any] = {}
            results["classify"] = run_classification(
                config, args.start_date, args.end_date, args.refresh_cache, args.include_screening_extension
            )
            results["pricing"] = run_pricing(
                config, args.start_date, args.end_date, paths, args.refresh_cache, args.include_screening_extension
            )
            results["dataset"] = build_lsmc_dataset(
                config, args.start_date, args.end_date, paths, False, args.include_screening_extension
            )
            results["lsmc"] = run_lsmc(
                config,
                args.start_date,
                args.end_date,
                paths,
                args.backtest_paths,
                path_models,
                args.regression_model,
                n_jobs,
                False,
                args.include_screening_extension,
            )
            results["selector"] = select_horizon(
                config, args.start_date, args.end_date, paths, False, args.include_screening_extension
            )
            results["backtest"] = run_backtest(config, args.start_date, args.end_date, strategies, costs)
            results["ablation"] = run_ablation(config, args.start_date, args.end_date)
            results["report"] = build_report(config, args.start_date, args.end_date)
            result = results
        print_json(result)
        return 0
    except Exception as exc:
        LOGGER.exception("hybrid v0.7 research command failed")
        print_json({"status": "FAILED", "mode": args.mode, "error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
