from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.diagnostic_linkage_v071_core import deduplicate_event_episodes
from research.diagnostic_linkage_v071_data import add_versions, output_dir, write_csv


EPS = 1e-12


INTRADAY_STRATEGIES = {
    "D1_SYNCHRONOUS_RESIDUAL": ("baseline_residual_lag_repair", "fixed"),
    "D2_DISTRIBUTED_LAG_GAP": ("strategy_linkage_gap", "fixed"),
    "D3_NO_MARKET_ADJUSTMENT": ("strategy_linkage_gap_unadjusted", "fixed"),
    "D3_GAP_REPAIR_STARTED": ("strategy_linkage_repair", "fixed"),
    "D4_GAP_REPAIR_LIQUIDITY": ("strategy_linkage_liquidity", "fixed"),
    "D5_DYNAMIC_LINKAGE_EXIT": ("strategy_linkage_liquidity", "dynamic"),
    "D6_HOT_MONEY_REJECT": ("strategy_linkage_hot_reject", "dynamic"),
    "D7_NO_HOT_MONEY_REJECT": ("strategy_linkage_liquidity_no_hot_reject", "dynamic"),
    "D9_FIXED_PERCENT_EXIT_BENCHMARK": ("strategy_linkage_hot_reject", "fixed_percent"),
}


def _safe_mean(values: pd.Series | Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else np.nan


def _safe_median(values: pd.Series | Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.median()) if not series.empty else np.nan


def _compound(values: pd.Series | Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float((1.0 + series).prod() - 1.0) if not series.empty else np.nan


def drawdown_from_returns(values: pd.Series | Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    if series.empty:
        return np.nan
    curve = (1.0 + series).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def block_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    seed: int,
    iterations: int = 500,
    date_column: str = "trade_date",
) -> tuple[float, float]:
    source = frame[[date_column, value_column]].copy()
    source[value_column] = pd.to_numeric(source[value_column], errors="coerce")
    daily = source.groupby(date_column)[value_column].mean().dropna()
    if len(daily) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = daily.to_numpy(float)
    means = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        means[iteration] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _round_trip_cost_components(
    signal: pd.Series,
    entry_price: float,
    scenario: str,
    config: dict[str, Any],
) -> dict[str, float]:
    scenario_cfg = config["costs"]["scenarios"][scenario]
    if scenario == "zero":
        return {
            "commission_cost_bps": 0.0,
            "spread_cost_bps": 0.0,
            "latency_slippage_bps": 0.0,
            "market_impact_bps": 0.0,
            "liquidity_penalty_bps": 0.0,
            "total_cost_bps": 0.0,
            "total_implementation_shortfall_bps": 0.0,
        }
    close = float(pd.to_numeric(signal.get("cb_close"), errors="coerce"))
    high = float(pd.to_numeric(signal.get("cb_high"), errors="coerce"))
    low = float(pd.to_numeric(signal.get("cb_low"), errors="coerce"))
    amount = float(pd.to_numeric(signal.get("cb_amount"), errors="coerce"))
    adv20 = float(pd.to_numeric(signal.get("adv20_amount"), errors="coerce"))
    range_fraction = (high - low) / close if close > 0 and np.isfinite(high + low) else 0.0
    spread = (
        range_fraction
        * float(config["costs"]["spread_range_fraction"])
        * 10000.0
        * float(scenario_cfg["spread_multiplier"])
    )
    raw_latency = (entry_price / close - 1.0) * 10000.0 if close > 0 else 0.0
    if bool(config["costs"].get("latency_adverse_only", True)):
        raw_latency = max(raw_latency, 0.0)
    latency = raw_latency * float(scenario_cfg["latency_multiplier"])
    order_notional = float(config["costs"]["order_notional"])
    bar_share = order_notional / max(amount, order_notional)
    adv_share = order_notional / max(adv20, order_notional) if np.isfinite(adv20) else 1.0
    impact = (
        float(config["costs"]["market_impact_coefficient_bps"])
        * (math.sqrt(bar_share) + math.sqrt(adv_share))
        * float(scenario_cfg["impact_multiplier"])
    )
    regime = str(signal.get("liquidity_regime_5m", "NORMAL"))
    regime_penalties = config["costs"]["liquidity_regime_penalty_bps"]
    base_liquidity_penalty = float(regime_penalties.get(regime, regime_penalties["NORMAL"]))
    liquidity = (
        base_liquidity_penalty
        * float(scenario_cfg["liquidity_multiplier"])
    )
    commission = float(scenario_cfg["commission_bps"])
    total = commission + spread + impact + liquidity
    return {
        "commission_cost_bps": commission,
        "spread_cost_bps": spread,
        "latency_slippage_bps": latency,
        "market_impact_bps": impact,
        "liquidity_penalty_bps": liquidity,
        "total_cost_bps": total,
        "total_implementation_shortfall_bps": total + latency,
    }


def _deduplicate_signal_rows(
    frame: pd.DataFrame,
    mask: pd.Series,
    merge_bars: int,
) -> pd.DataFrame:
    selected = frame[mask.fillna(False)].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["trade_date", "bond_code", "bar_slot"])
    keep = []
    for _, group in selected.groupby(["trade_date", "bond_code"], sort=False):
        last_slot = -999
        for index, row in group.iterrows():
            slot = int(row["bar_slot"])
            if slot - last_slot > merge_bars:
                keep.append(index)
                last_slot = slot
    return selected.loc[keep].copy()


def _forward_bar_return(path: pd.DataFrame, entry_position: int, bars: int, entry_price: float) -> float:
    target = entry_position + bars - 1
    if target >= len(path):
        return np.nan
    close = float(pd.to_numeric(path.iloc[target]["cb_close"], errors="coerce"))
    return close / entry_price - 1.0 if entry_price > 0 and close > 0 else np.nan


def _simulate_intraday_trade(
    path: pd.DataFrame,
    signal_position: int,
    entry_delay_bars: int,
    exit_mode: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    entry_position = signal_position + 1 + int(entry_delay_bars)
    if entry_position >= len(path):
        return None
    signal = path.iloc[signal_position]
    entry = path.iloc[entry_position]
    entry_price = float(pd.to_numeric(entry["cb_open"], errors="coerce"))
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    maximum = int(config["exits"]["maximum_holding_bars"])
    fixed_horizon = int(config["exits"]["fixed_horizon_bars"])
    fixed_mode = exit_mode in {"fixed", "fixed_percent"}
    final_position = min(
        len(path) - 1,
        entry_position + (fixed_horizon if fixed_mode else maximum) - 1,
    )
    exit_reason = "FIXED_HORIZON" if fixed_mode else "MAXIMUM_HOLDING_BARS"
    initial_gap = max(float(pd.to_numeric(signal.get("linkage_gap_3bar"), errors="coerce")), EPS)
    initial_shock = float(
        pd.to_numeric(signal.get("stock_idiosyncratic_return"), errors="coerce")
    )
    half_life = float(pd.to_numeric(signal.get("response_half_life_bars"), errors="coerce"))
    if not np.isfinite(half_life):
        half_life = 3.0
    half_life = float(np.clip(math.ceil(half_life), 1, 3))
    daily_range = float(pd.to_numeric(signal.get("high_low_range_proxy"), errors="coerce"))
    hard_stop_bps = np.clip(
        daily_range * float(config["exits"]["hard_stop_vol_multiple"]) * 10000.0,
        float(config["exits"]["hard_stop_floor_bps"]),
        float(config["exits"]["hard_stop_cap_bps"]),
    )
    repaired_threshold = float(config["exits"]["default_linkage_repaired_threshold"])
    trigger_position = final_position
    if exit_mode == "fixed_percent":
        benchmark_barrier_bps = np.clip(
            daily_range
            * float(config["exits"]["fixed_benchmark_stop_vol_multiple"])
            * 10000.0,
            float(config["exits"]["hard_stop_floor_bps"]),
            float(config["exits"]["hard_stop_cap_bps"]),
        )
        for position in range(entry_position, final_position + 1):
            row = path.iloc[position]
            low_return = float(row["cb_low"]) / entry_price - 1.0
            high_return = float(row["cb_high"]) / entry_price - 1.0
            if low_return <= -benchmark_barrier_bps / 10000.0:
                exit_reason = "FIXED_VOL_NORMALIZED_STOP"
                trigger_position = position
                break
            if high_return >= benchmark_barrier_bps / 10000.0:
                exit_reason = "FIXED_VOL_NORMALIZED_TAKE_PROFIT"
                trigger_position = position
                break
    elif exit_mode == "dynamic":
        for position in range(entry_position, final_position + 1):
            row = path.iloc[position]
            elapsed = position - entry_position + 1
            current_close = float(pd.to_numeric(row["cb_close"], errors="coerce"))
            current_low = float(pd.to_numeric(row["cb_low"], errors="coerce"))
            cumulative_return = current_close / entry_price - 1.0
            repaired_fraction = cumulative_return / initial_gap
            stock_path = pd.to_numeric(
                path.iloc[signal_position : position + 1]["stock_idiosyncratic_return"],
                errors="coerce",
            ).sum(min_count=1)
            hard_stop = current_low / entry_price - 1.0 <= -hard_stop_bps / 10000.0
            stock_reversal = (
                bool(config["exits"]["use_stock_reversal_stop"])
                and np.isfinite(initial_shock)
                and stock_path <= -abs(initial_shock) * float(config["exits"]["stock_reversal_fraction"])
            )
            climax = (
                bool(config["exits"]["use_volume_climax_stop"])
                and row.get("liquidity_regime_5m") == "VOLUME_CLIMAX"
                and float(pd.to_numeric(row.get("close_location"), errors="coerce")) < 0.35
            )
            remaining_gap = initial_gap - cumulative_return
            overshoot = (
                bool(config["exits"]["use_overshoot_exit"])
                and (remaining_gap <= EPS or repaired_fraction + EPS >= 1.0)
            )
            repaired = repaired_fraction + EPS >= repaired_threshold
            failure = (
                bool(config["exits"]["use_linkage_failure_stop"])
                and elapsed >= half_life
                and remaining_gap >= initial_gap
                * float(config["exits"]["linkage_failure_remaining_gap_fraction"])
                and cumulative_return <= 0
            )
            time_stop = (
                bool(config["exits"]["use_response_half_life_time_stop"])
                and elapsed >= half_life
                and repaired_fraction < repaired_threshold
            )
            if hard_stop:
                exit_reason = "HARD_PRICE_STOP"
            elif stock_reversal:
                exit_reason = "STOCK_REVERSAL_STOP"
            elif climax:
                exit_reason = "VOLUME_CLIMAX_STOP"
            elif overshoot:
                exit_reason = "OVERSHOOT_EXIT"
            elif repaired:
                exit_reason = "LINKAGE_REPAIRED"
            elif failure:
                exit_reason = "LINKAGE_FAILURE_STOP"
            elif time_stop:
                exit_reason = "TIME_STOP"
            else:
                continue
            trigger_position = position
            break
    if trigger_position + 1 < len(path):
        exit_position = trigger_position + 1
        exit_price = float(pd.to_numeric(path.iloc[exit_position]["cb_open"], errors="coerce"))
        exit_time = path.iloc[exit_position]["bar_start"]
        exit_execution = "NEXT_BAR_OPEN_AFTER_EXIT_HINT"
    else:
        exit_position = trigger_position
        exit_price = float(pd.to_numeric(path.iloc[exit_position]["cb_close"], errors="coerce"))
        exit_time = path.iloc[exit_position]["bar_end"]
        exit_execution = "SAME_DAY_LAST_AVAILABLE_CLOSE"
    observed = path.iloc[entry_position : exit_position + 1]
    mfe = float(pd.to_numeric(observed["cb_high"], errors="coerce").max() / entry_price - 1.0)
    mae = float(pd.to_numeric(observed["cb_low"], errors="coerce").min() / entry_price - 1.0)
    return {
        "signal_time": signal["bar_end"],
        "first_seen_time": signal["bar_end"],
        "entry_time": entry["bar_start"],
        "entry_price": entry_price,
        "signal_price": float(signal["cb_close"]),
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "exit_execution_assumption": exit_execution,
        "holding_bars": int(exit_position - entry_position + 1),
        "gross_return": exit_price / entry_price - 1.0,
        "mfe": mfe,
        "mae": mae,
        "return_1bar": _forward_bar_return(path, entry_position, 1, entry_price),
        "return_2bar": _forward_bar_return(path, entry_position, 2, entry_price),
        "return_3bar": _forward_bar_return(path, entry_position, 3, entry_price),
        "return_30m": _forward_bar_return(path, entry_position, 6, entry_price),
        "return_60m": _forward_bar_return(path, entry_position, 12, entry_price),
        "initial_linkage_gap": initial_gap,
        "response_half_life_bars": half_life,
        "hard_stop_bps": hard_stop_bps,
    }


def backtest_intraday_linkage(
    features: pd.DataFrame,
    config: dict[str, Any],
    entry_delays: Iterable[int] | None = None,
    cost_scenarios: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    delays = list(entry_delays or config["evaluation"]["entry_delay_bars"])
    scenarios = list(cost_scenarios or config["evaluation"]["cost_scenarios"])
    ordered = features.sort_values(["trade_date", "bond_code", "bar_start"]).copy()
    ordered["path_position"] = ordered.groupby(["trade_date", "bond_code"]).cumcount()
    path_lookup = {
        key: group.reset_index(drop=True)
        for key, group in ordered.groupby(["trade_date", "bond_code"], sort=False)
    }
    trades = []
    merge_bars = int(config["linkage"]["episode_merge_bars"])
    for strategy, (mask_column, exit_mode) in INTRADAY_STRATEGIES.items():
        signal_rows = _deduplicate_signal_rows(ordered, ordered[mask_column], merge_bars)
        signal_rows = signal_rows[signal_rows["pit_universe_eligible"].fillna(False)].copy()
        for _, signal in signal_rows.iterrows():
            key = (signal["trade_date"], signal["bond_code"])
            path = path_lookup[key]
            signal_position = int(signal["path_position"])
            for delay in delays:
                result = _simulate_intraday_trade(
                    path, signal_position, int(delay), exit_mode, config
                )
                if result is None:
                    continue
                base = {
                    "trade_date": signal["trade_date"],
                    "window_id": signal["window_id"],
                    "window_role": signal["window_role"],
                    "bond_code": signal["bond_code"],
                    "stock_code": signal["stock_code"],
                    "bond_type": signal.get("bond_type", "UNKNOWN"),
                    "bond_price": signal.get("cb_close"),
                    "parity": signal.get("parity"),
                    "premium": signal.get("premium"),
                    "daily_turnover": signal.get("daily_turnover"),
                    "adv20_amount": signal.get("adv20_amount"),
                    "relative_amount": signal.get("relative_amount"),
                    "strategy": strategy,
                    "entry_delay_bars": int(delay),
                    "stock_shock_z": signal.get("stock_shock_z"),
                    "stock_shock_amount_z": signal.get("stock_shock_amount_z"),
                    "linkage_gap_1bar": signal.get("linkage_gap_1bar"),
                    "linkage_gap_2bar": signal.get("linkage_gap_2bar"),
                    "linkage_gap_3bar": signal.get("linkage_gap_3bar"),
                    "linkage_repair_started": signal.get("linkage_repair_started"),
                    "liquidity_regime": signal.get("liquidity_regime_5m"),
                    "hot_money_risk_score": signal.get("hot_money_risk_score_5m"),
                    "volume_climax_score": signal.get("volume_climax_score"),
                    "chosen_market_index": signal.get("chosen_market_index"),
                    "actionable_beta_sum": signal.get("stock_to_cb_beta_actionable_sum"),
                    "beta_lag1": signal.get("stock_to_cb_beta_lag1"),
                    "beta_lag2": signal.get("stock_to_cb_beta_lag2"),
                    "beta_lag3": signal.get("stock_to_cb_beta_lag3"),
                    "linkage_model_r2": signal.get("linkage_regression_r2"),
                    "linkage_confidence": signal.get("linkage_model_confidence"),
                    "response_half_life_bars_signal": signal.get("response_half_life_bars"),
                    "market_regime": signal.get("market_regime"),
                    "source_vendor": signal.get("cb_source_vendor"),
                    "pool_scope": signal.get("research_pool_scope"),
                    "universe_bias_flag": signal.get("universe_bias_flag"),
                }
                base.update(result)
                for scenario in scenarios:
                    costs = _round_trip_cost_components(
                        signal, result["entry_price"], scenario, config
                    )
                    row = base.copy()
                    row["cost_scenario"] = scenario
                    row.update(costs)
                    row["net_return"] = row["gross_return"] - row["total_cost_bps"] / 10000.0
                    trades.append(row)
    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        return trade_frame, pd.DataFrame(), pd.DataFrame()
    summary = summarize_intraday_trades(trade_frame, config)
    waterfall = trade_frame.groupby(
        ["strategy", "entry_delay_bars", "cost_scenario"], dropna=False
    ).agg(
        trade_count=("bond_code", "size"),
        gross_return=("gross_return", "mean"),
        commission_cost_bps=("commission_cost_bps", "mean"),
        spread_cost_bps=("spread_cost_bps", "mean"),
        latency_slippage_bps=("latency_slippage_bps", "mean"),
        market_impact_bps=("market_impact_bps", "mean"),
        liquidity_penalty_bps=("liquidity_penalty_bps", "mean"),
        total_cost_bps=("total_cost_bps", "mean"),
        total_implementation_shortfall_bps=("total_implementation_shortfall_bps", "mean"),
        net_return=("net_return", "mean"),
    ).reset_index()
    return trade_frame, summary, waterfall


def summarize_intraday_trades(
    trades: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    seed = int(config["strategy"]["random_seed"])
    iterations = int(config["diagnostic"]["bootstrap_iterations"])
    keys = ["strategy", "entry_delay_bars", "cost_scenario"]
    for values, group in trades.groupby(keys, dropna=False):
        daily = group.groupby("trade_date")["net_return"].mean()
        low, high = block_bootstrap_interval(group, "net_return", seed, iterations)
        row = dict(zip(keys, values))
        row.update(
            {
                "event_count": int(group[["trade_date", "bond_code", "signal_time"]].drop_duplicates().shape[0]),
                "trade_count": int(len(group)),
                "gross_return_mean": _safe_mean(group["gross_return"]),
                "gross_return_median": _safe_median(group["gross_return"]),
                "net_return_mean": _safe_mean(group["net_return"]),
                "net_return_median": _safe_median(group["net_return"]),
                "return_1bar": _safe_mean(group["return_1bar"]),
                "return_2bar": _safe_mean(group["return_2bar"]),
                "return_3bar": _safe_mean(group["return_3bar"]),
                "return_30m": _safe_mean(group["return_30m"]),
                "return_60m": _safe_mean(group["return_60m"]),
                "mfe": _safe_mean(group["mfe"]),
                "mae": _safe_mean(group["mae"]),
                "hit_rate_gross": float((group["gross_return"] > 0).mean()),
                "hit_rate_net": float((group["net_return"] > 0).mean()),
                "average_holding_bars": _safe_mean(group["holding_bars"]),
                "daily_signal_count": float(group.groupby("trade_date").size().mean()),
                "per_bond_daily_signal_count": float(
                    group.groupby(["trade_date", "bond_code"]).size().mean()
                ),
                "worst_day": float(daily.min()) if not daily.empty else np.nan,
                "worst_bond": str(
                    group.groupby("bond_code")["net_return"].mean().idxmin()
                ),
                "block_bootstrap_ci_low": low,
                "block_bootstrap_ci_high": high,
                "daily_cohort_compound_return": _compound(daily),
                "daily_cohort_max_drawdown": drawdown_from_returns(daily),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def linkage_coefficient_reports(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = features.sort_values("bar_start").drop_duplicates(
        ["trade_date", "bond_code"], keep="last"
    )
    columns = [
        "trade_date",
        "window_id",
        "window_role",
        "bond_code",
        "stock_code",
        "bond_type",
        "chosen_market_index",
        "stock_index_beta",
        "stock_index_model_r2",
        "stock_to_cb_beta_lag0",
        "stock_to_cb_beta_lag1",
        "stock_to_cb_beta_lag2",
        "stock_to_cb_beta_lag3",
        "stock_to_cb_beta_actionable_sum",
        "cb_market_gamma",
        "linkage_regression_r2",
        "linkage_sample_count",
        "linkage_model_source",
        "linkage_model_confidence",
        "cb_to_stock_beta_lag1",
        "cb_to_stock_beta_lag2",
        "cb_to_stock_beta_lag3",
        "cb_to_stock_beta_actionable_sum",
        "dominant_lead_direction",
        "lead_lag_confidence",
        "response_half_life_bars",
        "universe_bias_flag",
    ]
    coefficients = daily[[c for c in columns if c in daily.columns]].copy()
    bond_groups = coefficients.groupby(["bond_code", "stock_code"], dropna=False)
    by_bond = bond_groups.agg(
        sample_days=("trade_date", "nunique"),
        available_days=("linkage_model_confidence", lambda s: pd.to_numeric(s, errors="coerce").gt(0).sum()),
        beta_lag0=("stock_to_cb_beta_lag0", "mean"),
        beta_lag1=("stock_to_cb_beta_lag1", "mean"),
        beta_lag2=("stock_to_cb_beta_lag2", "mean"),
        beta_lag3=("stock_to_cb_beta_lag3", "mean"),
        actionable_beta_sum=("stock_to_cb_beta_actionable_sum", "mean"),
        reverse_actionable_beta_sum=("cb_to_stock_beta_actionable_sum", "mean"),
        linkage_r2=("linkage_regression_r2", "mean"),
        response_half_life_bars=("response_half_life_bars", "median"),
        confidence=("linkage_model_confidence", "mean"),
    ).reset_index()
    type_history = bond_groups["bond_type"].agg(
        dominant_bond_type=lambda values: (
            values.dropna().astype(str).mode().iloc[0]
            if not values.dropna().empty
            else "UNKNOWN"
        ),
        bond_type_count=lambda values: int(values.dropna().nunique()),
    ).reset_index()
    by_bond = by_bond.merge(
        type_history,
        on=["bond_code", "stock_code"],
        how="left",
        validate="one_to_one",
    )
    by_type = coefficients.groupby("bond_type", dropna=False).agg(
        bond_count=("bond_code", "nunique"),
        bond_days=("bond_code", "size"),
        beta_lag0=("stock_to_cb_beta_lag0", "mean"),
        beta_lag1=("stock_to_cb_beta_lag1", "mean"),
        beta_lag2=("stock_to_cb_beta_lag2", "mean"),
        beta_lag3=("stock_to_cb_beta_lag3", "mean"),
        actionable_beta_sum=("stock_to_cb_beta_actionable_sum", "mean"),
        reverse_actionable_beta_sum=("cb_to_stock_beta_actionable_sum", "mean"),
        linkage_r2=("linkage_regression_r2", "mean"),
        response_half_life_bars=("response_half_life_bars", "median"),
        confidence=("linkage_model_confidence", "mean"),
    ).reset_index()
    return coefficients, by_bond, by_type


def grouped_linkage_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    formal = trades[
        trades["strategy"].eq("D6_HOT_MONEY_REJECT")
        & trades["entry_delay_bars"].eq(0)
        & trades["cost_scenario"].isin(["zero", "realistic"])
    ].copy()
    signal_time = pd.to_datetime(formal["signal_time"], errors="coerce")
    minutes = signal_time.dt.hour * 60 + signal_time.dt.minute
    formal["time_bucket"] = pd.cut(
        minutes,
        bins=[0, 600, 690, 840, 885, 1440],
        labels=["OPEN_0930_1000", "MORNING_1000_1130", "EARLY_PM", "LATE_PM", "TAIL"],
        right=False,
    )
    formal["parity_bucket"] = pd.cut(
        pd.to_numeric(formal["parity"], errors="coerce"),
        [-np.inf, 90, 100, 115, 130, np.inf],
        labels=["LT90", "90_100", "100_115", "115_130", "GE130"],
        right=False,
    )
    formal["bond_price_bucket"] = pd.cut(
        pd.to_numeric(formal["bond_price"], errors="coerce"),
        [-np.inf, 110, 120, 130, np.inf],
        labels=["LT110", "110_120", "120_130", "GE130"],
        right=False,
    )
    formal["premium_bucket"] = pd.cut(
        pd.to_numeric(formal["premium"], errors="coerce"),
        [-np.inf, 20, 25, 30, 40, np.inf],
        labels=["LT20", "20_25", "25_30", "30_40", "GE40"],
        right=False,
    )
    formal["stock_shock_strength"] = pd.cut(
        pd.to_numeric(formal["stock_shock_z"], errors="coerce"),
        [-np.inf, 1.0, 1.5, 2.0, 3.0, np.inf],
        labels=["LT1", "1_1.5", "1.5_2", "2_3", "GE3"],
        right=False,
    )
    beta_values = formal[["beta_lag1", "beta_lag2", "beta_lag3"]].abs()
    formal["linkage_beta_lag_structure"] = np.where(
        beta_values.notna().any(axis=1),
        "LAG" + (beta_values.to_numpy().argmax(axis=1) + 1).astype(str) + "_DOMINANT",
        "UNAVAILABLE",
    )
    dimensions = [
        "bond_type",
        "parity_bucket",
        "bond_price_bucket",
        "premium_bucket",
        "liquidity_regime",
        "stock_shock_strength",
        "linkage_beta_lag_structure",
        "market_regime",
        "time_bucket",
        "source_vendor",
        "pool_scope",
    ]
    rows = []
    for dimension in dimensions:
        if dimension not in formal.columns:
            continue
        for (scenario, bucket), group in formal.groupby(
            ["cost_scenario", dimension], dropna=False, observed=True
        ):
            rows.append(
                {
                    "group_dimension": dimension,
                    "group_value": bucket,
                    "cost_scenario": scenario,
                    "trade_count": int(len(group)),
                    "bond_count": int(group["bond_code"].nunique()),
                    "trading_days": int(group["trade_date"].nunique()),
                    "gross_return": _safe_mean(group["gross_return"]),
                    "net_return": _safe_mean(group["net_return"]),
                    "hit_rate": float((group["net_return"] > 0).mean()),
                    "mfe": _safe_mean(group["mfe"]),
                    "mae": _safe_mean(group["mae"]),
                    "average_holding_bars": _safe_mean(group["holding_bars"]),
                }
            )
    return pd.DataFrame(rows)


def prepare_daily_simple_rv(
    daily: pd.DataFrame,
    config: dict[str, Any],
    qiv_pricing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = daily.copy().sort_values(["trade_date", "bond_code"])
    if qiv_pricing is not None and not qiv_pricing.empty:
        pricing_columns = [
            "trade_date",
            "bond_code",
            "qiv_smooth",
            "qiv_rank_within_bond_type",
            "qiv_relative_value_score",
            "qiv_microstructure_contaminated",
            "mc_price_gap_pct",
            "mc_price_gap",
        ]
        pricing = qiv_pricing[[c for c in pricing_columns if c in qiv_pricing.columns]].copy()
        pricing["trade_date"] = pd.to_datetime(pricing["trade_date"], errors="coerce").dt.normalize()
        out = out.drop(columns=[c for c in pricing_columns[2:] if c in out.columns]).merge(
            pricing.drop_duplicates(["trade_date", "bond_code"]),
            on=["trade_date", "bond_code"],
            how="left",
        )
    out["bond_type"] = out.get("bond_type", "UNKNOWN").fillna("UNKNOWN")
    valid = (
        out.get("conversion_price_asof_valid", False).fillna(False).astype(bool)
        & pd.to_numeric(out.get("daily_turnover"), errors="coerce").notna()
        & pd.to_numeric(out.get("bond_close"), errors="coerce").gt(0)
        & pd.to_numeric(out.get("next_open"), errors="coerce").gt(0)
    )
    if bool(config["daily_rv"]["exclude_obvious_hot_money"]):
        valid &= pd.to_numeric(out.get("premium"), errors="coerce").lt(
            float(config["daily_rv"]["speculative_premium_threshold"])
        )
        valid &= pd.to_numeric(out.get("adv20_amount"), errors="coerce").ge(
            float(config["daily_rv"]["minimum_adv20_amount"])
        )
    out["daily_rv_eligible"] = valid
    eligible = out["daily_rv_eligible"]
    grouped = out[eligible].groupby(["trade_date", "bond_type"], sort=False)
    out["premium_rank_within_type"] = np.nan
    out.loc[eligible, "premium_rank_within_type"] = grouped["premium"].rank(pct=True)
    out["residual_rank_within_type"] = np.nan
    out.loc[eligible, "residual_rank_within_type"] = grouped["residual_z_daily"].rank(pct=True)
    out["liquidity_rank"] = np.nan
    out.loc[eligible, "liquidity_rank"] = out[eligible].groupby("trade_date")[
        "adv20_amount"
    ].rank(pct=True)
    cfg = config["daily_rv"]
    out["daily_rv_score_simple"] = (
        float(cfg["premium_weight"]) * (1.0 - out["premium_rank_within_type"])
        + float(cfg["residual_weight"]) * (1.0 - out["residual_rank_within_type"])
        + float(cfg["liquidity_weight"]) * out["liquidity_rank"]
    )
    out["type_neutral_rv_rank"] = out[eligible].groupby("trade_date")[
        "daily_rv_score_simple"
    ].rank(pct=True).reindex(out.index)
    out["liquidity_adjusted_rv_rank"] = out[eligible].groupby("trade_date")[
        "daily_rv_score_simple"
    ].rank(pct=True).reindex(out.index)
    out["qiv_formal_weight"] = 0.0
    out["mc_gap_formal_weight"] = 0.0
    out["momentum_formal_weight"] = 0.0
    out["lsmc_entry_enabled"] = False
    out["universe_bias_flag"] = config["common_slice"]["universe_bias_flag"]
    out["point_in_time_universe_claim"] = False
    return out


def evaluate_qiv_mc_restored_ablation(
    daily_snapshot: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    restored = daily_snapshot.copy()
    qiv_rank = restored.groupby("trade_date")["qiv_relative_value_score"].rank(pct=True)
    mc_rank = restored.groupby("trade_date")["mc_price_gap_pct"].rank(pct=True)
    restored["daily_rv_score_simple"] = (
        0.60 * pd.to_numeric(restored["daily_rv_score_simple"], errors="coerce")
        + 0.20 * qiv_rank.fillna(0.5)
        + 0.20 * mc_rank.fillna(0.5)
    )
    restored["type_neutral_rv_rank"] = restored[
        restored["daily_rv_eligible"]
    ].groupby("trade_date")["daily_rv_score_simple"].rank(pct=True).reindex(restored.index)
    trades, summary = backtest_daily_simple_rv(
        restored,
        config,
        selectors=["top10"],
        holding_days=[3],
        scenarios=["zero", "realistic"],
    )
    if not trades.empty:
        trades["formal_strategy"] = False
        trades["diagnostic_only"] = True
        trades["qiv_mc_restored_weight"] = 0.40
    if not summary.empty:
        summary["formal_strategy"] = False
        summary["diagnostic_only"] = True
        summary["qiv_mc_restored_weight"] = 0.40
    return trades, summary


def _daily_cost_components(
    signal: pd.Series,
    scenario: str,
    config: dict[str, Any],
) -> dict[str, float]:
    scenario_cfg = config["costs"]["scenarios"][scenario]
    if scenario == "zero":
        return {
            "commission_cost_bps": 0.0,
            "spread_cost_bps": 0.0,
            "latency_slippage_bps": 0.0,
            "market_impact_bps": 0.0,
            "liquidity_penalty_bps": 0.0,
            "total_cost_bps": 0.0,
        }
    commission = float(scenario_cfg["commission_bps"])
    spread = float(pd.to_numeric(signal.get("spread_proxy_bps"), errors="coerce"))
    impact = float(pd.to_numeric(signal.get("market_impact_bps"), errors="coerce"))
    liquidity = float(pd.to_numeric(signal.get("liquidity_risk_bps"), errors="coerce"))
    spread = (spread if np.isfinite(spread) else 0.0) * float(scenario_cfg["spread_multiplier"])
    impact = (impact if np.isfinite(impact) else 0.0) * float(scenario_cfg["impact_multiplier"])
    liquidity = (liquidity if np.isfinite(liquidity) else 0.0) * float(
        scenario_cfg["liquidity_multiplier"]
    )
    total = commission + spread + impact + liquidity
    return {
        "commission_cost_bps": commission,
        "spread_cost_bps": spread,
        "latency_slippage_bps": 0.0,
        "market_impact_bps": impact,
        "liquidity_penalty_bps": liquidity,
        "total_cost_bps": total,
    }


def _select_daily_signals(snapshot: pd.DataFrame, selector: str) -> pd.DataFrame:
    eligible = snapshot[snapshot["daily_rv_eligible"]].copy()
    if selector.startswith("top") and selector[3:].isdigit():
        count = int(selector[3:])
        return eligible.sort_values(
            ["trade_date", "daily_rv_score_simple"], ascending=[True, False]
        ).groupby("trade_date", sort=False).head(count)
    if selector == "top20pct":
        return eligible[eligible["type_neutral_rv_rank"].ge(0.80)].copy()
    raise ValueError(f"Unknown daily selector: {selector}")


def _simulate_daily_entry(
    path: pd.DataFrame,
    signal_position: int,
    holding_days: int,
    exit_rule: str = "fixed",
    lsmc_lookup: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    signal = path.iloc[signal_position]
    entry_position = signal_position + 1
    if entry_position >= len(path) or path.iloc[entry_position]["window_id"] != signal["window_id"]:
        return None
    entry = path.iloc[entry_position]
    entry_price = float(pd.to_numeric(entry["bond_open"], errors="coerce"))
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    maximum_exit = min(len(path) - 1, entry_position + int(holding_days))
    if path.iloc[maximum_exit]["window_id"] != signal["window_id"]:
        return None
    exit_signal_position = maximum_exit - 1
    exit_reason = f"FIXED_{holding_days}D_EXIT"
    if exit_rule != "fixed":
        for position in range(entry_position, maximum_exit):
            row = path.iloc[position]
            if exit_rule == "residual_repair" and float(
                pd.to_numeric(row.get("residual_z_daily"), errors="coerce")
            ) >= -0.1:
                exit_signal_position = position
                exit_reason = "RESIDUAL_REPAIRED"
                break
            if exit_rule == "rv_rank_repair" and float(
                pd.to_numeric(row.get("type_neutral_rv_rank"), errors="coerce")
            ) <= 0.50:
                exit_signal_position = position
                exit_reason = "RV_RANK_REPAIRED"
                break
            if exit_rule == "lsmc" and lsmc_lookup is not None:
                key = (row["trade_date"], row["bond_code"])
                if key in lsmc_lookup.index:
                    lsmc = lsmc_lookup.loc[key]
                    if isinstance(lsmc, pd.DataFrame):
                        lsmc = lsmc.iloc[-1]
                    hold_edge = float(pd.to_numeric(lsmc.get("lsmc_hold_edge"), errors="coerce"))
                    action = str(lsmc.get("lsmc_recommended_action", ""))
                    if action == "SELL" or (np.isfinite(hold_edge) and hold_edge <= 0):
                        exit_signal_position = position
                        exit_reason = "LSMC_EXIT"
                        break
    exit_position = min(exit_signal_position + 1, len(path) - 1)
    if path.iloc[exit_position]["window_id"] != signal["window_id"]:
        return None
    exit_row = path.iloc[exit_position]
    exit_price = float(pd.to_numeric(exit_row["bond_open"], errors="coerce"))
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None
    observed = path.iloc[entry_position : exit_position + 1]
    return {
        "signal_date": signal["trade_date"],
        "entry_date": entry["trade_date"],
        "entry_price": entry_price,
        "exit_date": exit_row["trade_date"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": int(exit_position - entry_position),
        "gross_return": exit_price / entry_price - 1.0,
        "mfe": float(pd.to_numeric(observed["bond_high"], errors="coerce").max() / entry_price - 1.0),
        "mae": float(pd.to_numeric(observed["bond_low"], errors="coerce").min() / entry_price - 1.0),
        "same_close_execution_forbidden": True,
        "entry_execution_assumption": "NEXT_TRADING_DAY_OPEN",
    }


def backtest_daily_simple_rv(
    snapshot: pd.DataFrame,
    config: dict[str, Any],
    selectors: Iterable[str] | None = None,
    holding_days: Iterable[int] | None = None,
    scenarios: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selectors = list(selectors or ["top5", "top10", "top20pct"])
    holding_days = list(holding_days or config["daily_rv"]["holding_days"])
    scenarios = list(scenarios or config["evaluation"]["cost_scenarios"])
    ordered = snapshot.sort_values(["bond_code", "trade_date"]).copy()
    ordered["daily_position"] = ordered.groupby("bond_code").cumcount()
    paths = {bond: group.reset_index(drop=True) for bond, group in ordered.groupby("bond_code")}
    rows = []
    for selector in selectors:
        signals = _select_daily_signals(ordered, selector)
        for _, signal in signals.iterrows():
            path = paths[signal["bond_code"]]
            position = int(signal["daily_position"])
            for days in holding_days:
                result = _simulate_daily_entry(path, position, int(days), "fixed")
                if result is None:
                    continue
                base = {
                    "trade_date": signal["trade_date"],
                    "window_id": signal["window_id"],
                    "window_role": signal["window_role"],
                    "bond_code": signal["bond_code"],
                    "stock_code": signal["stock_code"],
                    "bond_type": signal["bond_type"],
                    "selector": selector,
                    "exit_rule": f"fixed_{days}d",
                    "daily_rv_score_simple": signal["daily_rv_score_simple"],
                    "type_neutral_rv_rank": signal["type_neutral_rv_rank"],
                    "qiv_formal_weight": 0.0,
                    "mc_gap_formal_weight": 0.0,
                    "lsmc_entry_enabled": False,
                    "universe_bias_flag": signal["universe_bias_flag"],
                }
                base.update(result)
                for scenario in scenarios:
                    costs = _daily_cost_components(signal, scenario, config)
                    row = base.copy()
                    row["cost_scenario"] = scenario
                    row.update(costs)
                    row["net_return"] = row["gross_return"] - row["total_cost_bps"] / 10000.0
                    rows.append(row)
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, pd.DataFrame()
    summary = summarize_daily_trades(trades, config)
    return trades, summary


def summarize_daily_trades(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    keys = ["selector", "exit_rule", "cost_scenario"]
    for values, group in trades.groupby(keys, dropna=False):
        daily = group.groupby("trade_date")["net_return"].mean()
        low, high = block_bootstrap_interval(
            group,
            "net_return",
            int(config["strategy"]["random_seed"]),
            int(config["diagnostic"]["bootstrap_iterations"]),
        )
        row = dict(zip(keys, values))
        row.update(
            {
                "trade_count": int(len(group)),
                "gross_return_mean": _safe_mean(group["gross_return"]),
                "net_return_mean": _safe_mean(group["net_return"]),
                "net_return_median": _safe_median(group["net_return"]),
                "hit_rate": float((group["net_return"] > 0).mean()),
                "mfe": _safe_mean(group["mfe"]),
                "mae": _safe_mean(group["mae"]),
                "average_holding_days": _safe_mean(group["holding_days"]),
                "turnover_proxy": 2.0 * len(group) / max(group["trade_date"].nunique(), 1),
                "daily_cohort_compound_return": _compound(daily),
                "max_drawdown": drawdown_from_returns(daily),
                "worst_day": float(daily.min()) if not daily.empty else np.nan,
                "worst_bond": str(group.groupby("bond_code")["net_return"].mean().idxmin()),
                "block_bootstrap_ci_low": low,
                "block_bootstrap_ci_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_qiv_pricing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"bond_code": str, "stock_code": str}, low_memory=False)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    return frame


def qiv_mc_monotonicity(
    daily_snapshot: pd.DataFrame,
    qiv_pricing: pd.DataFrame,
    intraday_trades: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if qiv_pricing.empty:
        return pd.DataFrame()
    fields = [
        "qiv_smooth",
        "qiv_rank_within_bond_type",
        "qiv_relative_value_score",
        "mc_price_gap_pct",
    ]
    labels = ["forward_return_1d", "forward_return_3d", "forward_return_5d", "mfe_5d", "mae_5d"]
    context = daily_snapshot[[
        c for c in ["trade_date", "bond_code", "bond_type", *labels] if c in daily_snapshot.columns
    ]].copy()
    pricing_for_merge = qiv_pricing.drop(columns=["bond_type"], errors="ignore")
    merged = pricing_for_merge.merge(context, on=["trade_date", "bond_code"], how="inner")
    merged = merged[
        merged["trade_date"].isin(pd.to_datetime(daily_snapshot["trade_date"]).dt.normalize().unique())
    ].copy()
    rows = []
    monotonic_lookup: dict[tuple[str, str], float] = {}
    bins = int(config["qiv_mc"]["quantile_bins"])
    for field in fields:
        if field not in merged.columns:
            continue
        ranked = merged.groupby(["trade_date", "bond_type"])[field].rank(pct=True)
        merged["diagnostic_bin"] = np.minimum((ranked * bins).apply(np.ceil), bins)
        for (bond_type, diagnostic_bin), group in merged.groupby(
            ["bond_type", "diagnostic_bin"], dropna=False
        ):
            row = {
                "diagnostic_field": field,
                "bond_type": bond_type,
                "diagnostic_bin": diagnostic_bin,
                "sample_count": int(len(group)),
                "contamination_rate": float(
                    group.get("qiv_microstructure_contaminated", False)
                    .astype(str)
                    .str.lower()
                    .eq("true")
                    .mean()
                ),
                "formal_decision_weight": 0.0,
            }
            for label in labels:
                if label in group.columns:
                    row[label] = _safe_mean(group[label])
            rows.append(row)
    result = pd.DataFrame(rows)
    monotonic = []
    if not result.empty:
        for (field, bond_type), group in result.groupby(["diagnostic_field", "bond_type"]):
            valid = group.dropna(subset=["diagnostic_bin", "forward_return_3d"])
            rho = spearmanr(valid["diagnostic_bin"], valid["forward_return_3d"]).statistic if len(valid) >= 3 else np.nan
            monotonic.append((field, bond_type, rho))
        lookup = {(f, b): r for f, b, r in monotonic}
        monotonic_lookup = lookup
        result["spearman_bin_vs_3d"] = [
            lookup.get((row.diagnostic_field, row.bond_type), np.nan)
            for row in result.itertuples()
        ]
    if not intraday_trades.empty:
        subset = intraday_trades[
            intraday_trades["strategy"].eq("D6_HOT_MONEY_REJECT")
            & intraday_trades["entry_delay_bars"].eq(0)
            & intraday_trades["cost_scenario"].eq("zero")
        ].copy()
        qiv_event = subset.merge(
            qiv_pricing[[c for c in ["trade_date", "bond_code", *fields] if c in qiv_pricing.columns]],
            on=["trade_date", "bond_code"],
            how="left",
        )
        for field in fields:
            if field not in qiv_event.columns:
                continue
            qiv_event["diagnostic_bin"] = pd.qcut(
                qiv_event[field].rank(method="first"), bins, labels=False, duplicates="drop"
            )
            for diagnostic_bin, group in qiv_event.groupby("diagnostic_bin", dropna=False):
                rows.append(
                    {
                        "diagnostic_field": field,
                        "bond_type": "INTRADAY_ALL",
                        "diagnostic_bin": diagnostic_bin,
                        "sample_count": int(len(group)),
                        "formal_decision_weight": 0.0,
                        "return_1bar": _safe_mean(group["return_1bar"]),
                        "return_2bar": _safe_mean(group["return_2bar"]),
                        "return_3bar": _safe_mean(group["return_3bar"]),
                        "mfe": _safe_mean(group["mfe"]),
                        "mae": _safe_mean(group["mae"]),
                    }
                )
        result = pd.DataFrame(rows)
    if not result.empty:
        result["spearman_bin_vs_3d"] = [
            monotonic_lookup.get((str(row.diagnostic_field), str(row.bond_type)), np.nan)
            for row in result.itertuples()
        ]
    return result


def evaluate_lsmc_exit_overlay(
    daily_snapshot: pd.DataFrame,
    lsmc: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if lsmc.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    lsmc = lsmc.copy()
    lsmc["trade_date"] = pd.to_datetime(lsmc["trade_date"], errors="coerce").dt.normalize()
    lsmc["bond_code"] = lsmc["bond_code"].astype(str).str.zfill(6)
    lookup = lsmc.drop_duplicates(["trade_date", "bond_code"], keep="last").set_index(
        ["trade_date", "bond_code"]
    )
    ordered = daily_snapshot.sort_values(["bond_code", "trade_date"]).copy()
    ordered["daily_position"] = ordered.groupby("bond_code").cumcount()
    paths = {bond: group.reset_index(drop=True) for bond, group in ordered.groupby("bond_code")}
    signals = _select_daily_signals(ordered, "top10")
    signals = signals[
        signals.apply(lambda row: (row["trade_date"], row["bond_code"]) in lookup.index, axis=1)
    ]
    exit_rules = [
        ("fixed_1d", 1, "fixed"),
        ("fixed_3d", 3, "fixed"),
        ("fixed_5d", 5, "fixed"),
        ("residual_repair", 5, "residual_repair"),
        ("rv_rank_repair", 5, "rv_rank_repair"),
        ("lsmc_exit", 5, "lsmc"),
    ]
    rows = []
    for _, signal in signals.iterrows():
        path = paths[signal["bond_code"]]
        position = int(signal["daily_position"])
        entry_id = f"{signal['trade_date'].date()}_{signal['bond_code']}"
        for name, days, rule in exit_rules:
            result = _simulate_daily_entry(path, position, days, rule, lookup)
            if result is None:
                continue
            base = {
                "matched_entry_id": entry_id,
                "trade_date": signal["trade_date"],
                "window_id": signal["window_id"],
                "bond_code": signal["bond_code"],
                "bond_type": signal["bond_type"],
                "exit_rule": name,
                "daily_rv_score_simple": signal["daily_rv_score_simple"],
                "lsmc_changed_entry": False,
                "lsmc_entry_enabled": False,
            }
            base.update(result)
            for scenario in config["evaluation"]["cost_scenarios"]:
                costs = _daily_cost_components(signal, scenario, config)
                row = base.copy()
                row["cost_scenario"] = scenario
                row.update(costs)
                row["net_return"] = row["gross_return"] - row["total_cost_bps"] / 10000.0
                rows.append(row)
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, pd.DataFrame(), pd.DataFrame()
    summary_rows = []
    for (rule, scenario), group in trades.groupby(["exit_rule", "cost_scenario"]):
        common_entry_count = group["matched_entry_id"].nunique()
        daily_returns = group.groupby("trade_date")["net_return"].mean()
        summary_rows.append(
            {
                "experiment": "L2_FIXED_ENTRY_EXIT_COMPARISON",
                "exit_rule": rule,
                "cost_scenario": scenario,
                "matched_entry_count": common_entry_count,
                "gross_return": _safe_mean(group["gross_return"]),
                "net_return": _safe_mean(group["net_return"]),
                "hit_rate": float((group["net_return"] > 0).mean()),
                "mfe": _safe_mean(group["mfe"]),
                "mae": _safe_mean(group["mae"]),
                "average_holding_days": _safe_mean(group["holding_days"]),
                "max_drawdown": drawdown_from_returns(daily_returns),
            }
        )
    summary = pd.DataFrame(summary_rows)
    matched = lsmc_matched_experiments(daily_snapshot, lsmc, config)
    return trades, summary, matched


def lsmc_matched_experiments(
    daily_snapshot: pd.DataFrame,
    lsmc: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    merged = daily_snapshot.merge(
        lsmc[[
            c
            for c in [
                "trade_date",
                "bond_code",
                "lsmc_hold_edge",
                "lsmc_hold_edge_pct",
                "lsmc_recommended_action",
            ]
            if c in lsmc.columns
        ]],
        on=["trade_date", "bond_code"],
        how="inner",
    )
    merged = merged[merged["daily_rv_eligible"]].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["future_3d"] = pd.to_numeric(merged.get("forward_return_3d"), errors="coerce")
    for horizon in [1, 3, 5]:
        label = f"forward_return_{horizon}d"
        if label in merged.columns:
            benchmark = merged.groupby(["trade_date", "bond_type"])[label].transform("mean")
            merged[f"type_neutral_excess_{horizon}d"] = (
                pd.to_numeric(merged[label], errors="coerce") - benchmark
            )
    merged["hold_edge_bin"] = merged.groupby("trade_date")["lsmc_hold_edge"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    rows = []
    for bin_value, group in merged.groupby("hold_edge_bin", dropna=False):
        rows.append(
            {
                "experiment": "L4_HOLD_EDGE_MONOTONICITY",
                "selection": f"hold_edge_bin_{bin_value}",
                "sample_count": int(len(group)),
                "forward_return_1d": _safe_mean(group.get("forward_return_1d")),
                "forward_return_3d": _safe_mean(group.get("forward_return_3d")),
                "forward_return_5d": _safe_mean(group.get("forward_return_5d")),
                "type_neutral_excess_1d": _safe_mean(group.get("type_neutral_excess_1d")),
                "type_neutral_excess_3d": _safe_mean(group.get("type_neutral_excess_3d")),
                "type_neutral_excess_5d": _safe_mean(group.get("type_neutral_excess_5d")),
                "mfe_5d": _safe_mean(group.get("mfe_5d")),
                "mae_5d": _safe_mean(group.get("mae_5d")),
            }
        )
    rng = np.random.default_rng(int(config["strategy"]["random_seed"]))
    repeats = int(config["lsmc"]["matched_bootstrap_iterations"])
    for trade_date, group in merged.groupby("trade_date"):
        selected = group[pd.to_numeric(group["lsmc_hold_edge"], errors="coerce").gt(0)]
        count = len(selected)
        if count == 0:
            continue
        controls = {
            "LSMC_POSITIVE_EDGE": selected,
            "SIMPLE_RV_MATCHED_COUNT": group.nlargest(count, "daily_rv_score_simple"),
            "LIQUIDITY_MATCHED_COUNT": group.nlargest(count, "adv20_amount"),
        }
        for name, subset in controls.items():
            rows.append(
                {
                    "experiment": "L1_FIXED_TRADE_COUNT",
                    "trade_date": trade_date,
                    "selection": name,
                    "sample_count": int(len(subset)),
                    "forward_return_1d": _safe_mean(subset.get("forward_return_1d")),
                    "forward_return_3d": _safe_mean(subset.get("forward_return_3d")),
                    "forward_return_5d": _safe_mean(subset.get("forward_return_5d")),
                }
            )
        random_means = []
        for _ in range(repeats):
            sample_positions = rng.choice(len(group), size=count, replace=False)
            random_means.append(_safe_mean(group.iloc[sample_positions].get("forward_return_3d")))
        rows.append(
            {
                "experiment": "L1_FIXED_TRADE_COUNT",
                "trade_date": trade_date,
                "selection": "RANDOM_MATCHED_BOOTSTRAP",
                "sample_count": count,
                "forward_return_3d": _safe_mean(random_means),
                "bootstrap_p05": float(np.nanquantile(random_means, 0.05)),
                "bootstrap_p95": float(np.nanquantile(random_means, 0.95)),
            }
        )
        fixed_exit_controls = {
            "SIMPLE_RV_ENTRY_FIXED_3D": group.nlargest(count, "daily_rv_score_simple"),
            "LSMC_EDGE_ENTRY_FIXED_3D": group.nlargest(count, "lsmc_hold_edge"),
            "LIQUIDITY_ENTRY_FIXED_3D": group.nlargest(count, "adv20_amount"),
        }
        for name, subset in fixed_exit_controls.items():
            rows.append(
                {
                    "experiment": "L3_FIXED_EXIT_ENTRY_COMPARISON",
                    "trade_date": trade_date,
                    "selection": name,
                    "sample_count": int(len(subset)),
                    "forward_return_3d": _safe_mean(subset.get("forward_return_3d")),
                    "type_neutral_excess_3d": _safe_mean(
                        subset.get("type_neutral_excess_3d")
                    ),
                }
            )
    result = pd.DataFrame(rows)
    hold_bins = result[
        result["experiment"].eq("L4_HOLD_EDGE_MONOTONICITY")
        & result["selection"].astype(str).ne("hold_edge_bin_nan")
    ].dropna(subset=["forward_return_3d"])
    hold_bins = hold_bins.assign(
        _bin=pd.to_numeric(
            hold_bins["selection"].astype(str).str.replace("hold_edge_bin_", "", regex=False),
            errors="coerce",
        )
    ).sort_values("_bin")
    if len(hold_bins) >= 3:
        monotonicity = spearmanr(
            hold_bins["_bin"], hold_bins["forward_return_3d"]
        ).statistic
    else:
        monotonicity = np.nan
    adjacent_monotonic = bool(
        len(hold_bins) >= 3
        and np.all(np.diff(hold_bins["forward_return_3d"].to_numpy(float)) >= 0)
    )
    date_count = int(merged["trade_date"].nunique())
    window_count = int(merged["window_id"].nunique()) if "window_id" in merged.columns else 0
    result["hold_edge_monotonicity_spearman"] = monotonicity
    result["hold_edge_adjacent_bins_monotonic"] = adjacent_monotonic
    result["evaluation_trade_days"] = date_count
    result["evaluation_window_count"] = window_count
    result["lsmc_hold_edge_tradable_calibration"] = bool(
        np.isfinite(monotonicity)
        and monotonicity > 0.7
        and adjacent_monotonic
        and date_count >= 90
        and window_count >= 3
    )
    return result


def evaluate_cb_leads_stock(
    events: pd.DataFrame,
    daily_snapshot: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reverse = events[events["event_type"].eq("CB_LEADS_STOCK_INFORMATION")].copy()
    if reverse.empty:
        return reverse, pd.DataFrame()
    daily = daily_snapshot.sort_values(["bond_code", "trade_date"]).copy()
    grouped = daily.groupby("bond_code", sort=False)
    daily["next_stock_open"] = grouped["stock_open"].shift(-1)
    daily["next_stock_close"] = grouped["stock_close"].shift(-1)
    daily["next_bond_open"] = grouped["bond_open"].shift(-1)
    daily["next_bond_close"] = grouped["bond_close"].shift(-1)
    context = daily[[
        "trade_date",
        "bond_code",
        "bond_type",
        "stock_close",
        "bond_close",
        "next_stock_open",
        "next_stock_close",
        "next_bond_open",
        "next_bond_close",
        "forward_return_3d",
        "market_regime",
    ]]
    reverse = reverse.merge(context, on=["trade_date", "bond_code"], how="left", suffixes=("", "_daily"))
    stock_base_column = "stock_close_daily" if "stock_close_daily" in reverse.columns else "stock_close"
    bond_base_column = "bond_close_daily" if "bond_close_daily" in reverse.columns else "bond_close"
    reverse["stock_next_open_return"] = reverse["next_stock_open"] / reverse[stock_base_column] - 1.0
    reverse["stock_next_close_return"] = reverse["next_stock_close"] / reverse[stock_base_column] - 1.0
    reverse["bond_next_open_return"] = reverse["next_bond_open"] / reverse[bond_base_column] - 1.0
    reverse["bond_next_close_return"] = reverse["next_bond_close"] / reverse[bond_base_column] - 1.0
    reverse["research_only"] = True
    reverse["intraday_trade_allowed"] = False
    summary = reverse.groupby(["bond_type", "liquidity_regime_5m"], dropna=False).agg(
        event_count=("episode_id", "nunique"),
        stock_next_open_return=("stock_next_open_return", "mean"),
        stock_next_close_return=("stock_next_close_return", "mean"),
        bond_next_open_return=("bond_next_open_return", "mean"),
        bond_next_close_return=("bond_next_close_return", "mean"),
        forward_return_3d=("forward_return_3d", "mean"),
    ).reset_index()
    return reverse, summary


def qiv_mc_coverage_report(
    pricing: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Report pricing-field availability without promoting qIV/MC into a signal."""
    if pricing.empty:
        return pd.DataFrame()
    frame = pricing.copy()
    frame["bond_type"] = frame.get("bond_type", "UNKNOWN").fillna("UNKNOWN")
    assessable = frame.get(
        "qiv_contamination_assessable", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    contaminated = frame.get(
        "qiv_microstructure_contaminated", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)

    def summarize(group: pd.DataFrame, scope: str) -> dict[str, Any]:
        group_assessable = assessable.reindex(group.index).fillna(False)
        group_contaminated = contaminated.reindex(group.index).fillna(False)
        assessable_count = int(group_assessable.sum())
        return {
            "scope": scope,
            "bond_type": scope if scope != "ALL" else "ALL",
            "row_count": int(len(group)),
            "bond_count": int(group["bond_code"].nunique()),
            "trading_days": int(pd.to_datetime(group["trade_date"]).nunique()),
            "qiv_smooth_coverage": float(group["qiv_smooth"].notna().mean()),
            "qiv_solve_success_rate": float(
                group.get("qiv_solve_success", pd.Series(False, index=group.index))
                .fillna(False)
                .astype(bool)
                .mean()
            ),
            "mc_price_gap_coverage": float(group["mc_price_gap_pct"].notna().mean()),
            "contamination_assessable_rate": float(group_assessable.mean()),
            "contamination_rate_all_rows": float(group_contaminated.mean()),
            "contamination_rate_when_assessable": (
                float(group_contaminated[group_assessable].mean())
                if assessable_count
                else np.nan
            ),
            "formal_decision_weight": float(config["qiv_mc"]["formal_decision_weight"]),
            "diagnostic_only": bool(config["qiv_mc"]["diagnostic_only"]),
        }

    rows = [summarize(frame, "ALL")]
    rows.extend(
        summarize(group, str(bond_type))
        for bond_type, group in frame.groupby("bond_type", dropna=False)
    )
    return pd.DataFrame(rows)


def reverse_linkage_window_stability(reverse_events: pd.DataFrame) -> pd.DataFrame:
    """Keep CB-leads-stock conclusions honest across disjoint date windows."""
    if reverse_events.empty or "window_id" not in reverse_events.columns:
        return pd.DataFrame()
    rows = []
    return_columns = [
        "stock_next_open_return",
        "stock_next_close_return",
        "bond_next_open_return",
        "bond_next_close_return",
        "forward_return_3d",
    ]
    for window_id, group in reverse_events.groupby("window_id", dropna=False):
        row: dict[str, Any] = {
            "window_id": window_id,
            "event_count": int(group["episode_id"].nunique()),
            "bond_count": int(group["bond_code"].nunique()),
            "trading_days": int(pd.to_datetime(group["trade_date"]).nunique()),
        }
        for column in return_columns:
            row[column] = _safe_mean(group.get(column))
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_comparison(
    daily_trades: pd.DataFrame,
    daily_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    if daily_trades.empty:
        return pd.DataFrame()
    eligible = daily_snapshot[daily_snapshot["daily_rv_eligible"]].copy().sort_values(
        ["bond_code", "trade_date"]
    )
    grouped = eligible.groupby("bond_code", sort=False)
    rows = []
    for (selector, exit_rule, scenario), trades in daily_trades.groupby(
        ["selector", "exit_rule", "cost_scenario"]
    ):
        holding = int(str(exit_rule).split("_")[1].replace("d", ""))
        label = f"matched_open_to_open_{holding}d"
        if label not in eligible.columns:
            entry_open = grouped["bond_open"].shift(-1)
            exit_open = grouped["bond_open"].shift(-(holding + 1))
            entry_window = grouped["window_id"].shift(-1)
            exit_window = grouped["window_id"].shift(-(holding + 1))
            eligible[label] = (exit_open / entry_open - 1.0).where(
                entry_window.eq(eligible["window_id"]) & exit_window.eq(eligible["window_id"])
            )
        pool = eligible.groupby("trade_date")[label].mean()
        median = eligible.groupby("trade_date")[label].median()
        type_returns = eligible.groupby(["trade_date", "bond_type"])[label].mean()
        enriched = trades.copy()
        enriched["full_pool_benchmark"] = enriched["trade_date"].map(pool)
        enriched["cb_market_median_benchmark"] = enriched["trade_date"].map(median)
        enriched["same_type_benchmark"] = [
            type_returns.get((date, bond_type), np.nan)
            for date, bond_type in zip(enriched["trade_date"], enriched["bond_type"])
        ]
        rows.append(
            {
                "selector": selector,
                "exit_rule": exit_rule,
                "cost_scenario": scenario,
                "trade_count": int(len(enriched)),
                "absolute_return": _safe_mean(enriched["net_return"]),
                "excess_vs_full_pool": _safe_mean(
                    enriched["net_return"] - enriched["full_pool_benchmark"]
                ),
                "excess_vs_same_type": _safe_mean(
                    enriched["net_return"] - enriched["same_type_benchmark"]
                ),
                "excess_vs_cb_median": _safe_mean(
                    enriched["net_return"] - enriched["cb_market_median_benchmark"]
                ),
                "beta_adjusted_return": np.nan,
                "beta_adjusted_status": "UNAVAILABLE_WITHOUT_RELIABLE_DAILY_BETA_HEDGE_EXECUTION",
            }
        )
    return pd.DataFrame(rows)


def common_slice_comparison(
    intraday_trades: pd.DataFrame,
    daily_trades: pd.DataFrame,
    overlay_trades: pd.DataFrame,
    daily_snapshot: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(sorted(pd.to_datetime(daily_snapshot["trade_date"]).dt.normalize().unique()))
    intraday = intraday_trades[
        intraday_trades["strategy"].eq("D6_HOT_MONEY_REJECT")
        & intraday_trades["entry_delay_bars"].eq(0)
        & intraday_trades["cost_scenario"].eq("realistic")
    ].groupby("trade_date")["net_return"].mean()
    daily = daily_trades[
        daily_trades["selector"].eq("top10")
        & daily_trades["exit_rule"].eq("fixed_3d")
        & daily_trades["cost_scenario"].eq("realistic")
    ].groupby("trade_date")["net_return"].mean()
    overlay = overlay_trades[
        overlay_trades["exit_rule"].eq("lsmc_exit")
        & overlay_trades["cost_scenario"].eq("realistic")
    ].groupby("trade_date")["net_return"].mean() if not overlay_trades.empty else pd.Series(dtype=float)
    panel = pd.DataFrame(index=dates)
    panel.index.name = "trade_date"
    panel["intraday_linkage"] = intraday.reindex(dates).fillna(0.0)
    panel["daily_simple_rv"] = daily.reindex(dates).fillna(0.0)
    panel["daily_rv_lsmc_exit"] = overlay.reindex(dates)
    panel["hybrid_selector"] = np.where(
        intraday.reindex(dates).notna(), panel["intraday_linkage"], panel["daily_simple_rv"]
    )
    panel["hybrid_selected_horizon"] = np.where(
        intraday.reindex(dates).notna(), "INTRADAY_LINKAGE", "DAILY_SIMPLE_RV"
    )
    panel["metric_basis"] = "equal_weight_signal_date_cohort_return"
    panel["common_slice_pool"] = "daily_observable_canonical_pool"
    panel["universe_bias_flag"] = config["common_slice"]["universe_bias_flag"]
    panel = panel.reset_index()
    rows = []
    strict_dates = pd.Index(sorted(pd.to_datetime(overlay_trades["trade_date"]).dt.normalize().unique())) if not overlay_trades.empty else pd.Index([])
    panel["strict_lsmc_overlap_day"] = panel["trade_date"].isin(strict_dates)
    scopes = [
        (
            "CORE_180D_NO_LSMC_ENTRY",
            panel,
            ["intraday_linkage", "daily_simple_rv", "hybrid_selector"],
        )
    ]
    if len(strict_dates):
        strict = panel[panel["strict_lsmc_overlap_day"]].copy()
        strict["daily_rv_lsmc_exit"] = strict["daily_rv_lsmc_exit"].fillna(0.0)
        scopes.append(
            (
                "STRICT_ALL_STRATEGIES_LSMC_OVERLAP",
                strict,
                [
                    "intraday_linkage",
                    "daily_simple_rv",
                    "daily_rv_lsmc_exit",
                    "hybrid_selector",
                ],
            )
        )
    for scope, scoped_panel, strategies in scopes:
        for strategy in strategies:
            values = pd.to_numeric(scoped_panel[strategy], errors="coerce")
            rows.append(
                {
                    "comparison_scope": scope,
                    "strategy": strategy,
                    "common_slice_days": int(len(scoped_panel)),
                    "mean_daily_cohort_return": _safe_mean(values),
                    "median_daily_cohort_return": _safe_median(values),
                    "compound_cohort_return": _compound(values.fillna(0)),
                    "hit_rate": float((values.dropna() > 0).mean()) if values.notna().any() else np.nan,
                    "max_drawdown": drawdown_from_returns(values.fillna(0)),
                    "worst_day": float(values.min()) if values.notna().any() else np.nan,
                    "metric_basis": "equal_weight_signal_date_cohort_return",
                    "qiv_mc_entry_weight": 0.0,
                    "lsmc_entry_enabled": False,
                    "universe_bias_flag": config["common_slice"]["universe_bias_flag"],
                }
            )
    return panel, pd.DataFrame(rows)


def window_stability(
    intraday_trades: pd.DataFrame,
    daily_trades: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    seed = int(config["strategy"]["random_seed"])

    def append_groups(frame: pd.DataFrame, strategy_name: str) -> None:
        for (window_id, scenario), group in frame.groupby(
            ["window_id", "cost_scenario"]
        ):
            low, high = block_bootstrap_interval(group, "net_return", seed)
            rows.append(
                {
                    "window_id": window_id,
                    "strategy": strategy_name,
                    "cost_scenario": scenario,
                    "sample_count": int(len(group)),
                    "mean_return": _safe_mean(group["net_return"]),
                    "median_return": _safe_median(group["net_return"]),
                    "hit_rate": float((group["net_return"] > 0).mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "worst_date": str(
                        group.groupby("trade_date")["net_return"].mean().idxmin().date()
                    ),
                    "worst_bond": str(
                        group.groupby("bond_code")["net_return"].mean().idxmin()
                    ),
                }
            )

    intraday_specs = {
        "intraday_gap_unconfirmed": "D2_DISTRIBUTED_LAG_GAP",
        "intraday_linkage_dynamic": "D6_HOT_MONEY_REJECT",
        "intraday_linkage_fixed_exit": "D9_FIXED_PERCENT_EXIT_BENCHMARK",
    }
    for name, strategy in intraday_specs.items():
        frame = intraday_trades[
            intraday_trades["strategy"].eq(strategy)
            & intraday_trades["entry_delay_bars"].eq(0)
            & intraday_trades["cost_scenario"].isin(["zero", "realistic"])
        ]
        append_groups(frame, name)

    daily_specs = {
        "daily_top10_fixed3": ("top10", "fixed_3d"),
        "daily_top5_fixed5": ("top5", "fixed_5d"),
    }
    for name, (selector, exit_rule) in daily_specs.items():
        frame = daily_trades[
            daily_trades["selector"].eq(selector)
            & daily_trades["exit_rule"].eq(exit_rule)
            & daily_trades["cost_scenario"].isin(["zero", "realistic"])
        ]
        append_groups(frame, name)
    return pd.DataFrame(rows)


def build_walk_forward_plan(
    daily_snapshot: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    windows = config["common_slice"]["windows"]
    all_dates = pd.Index(sorted(pd.to_datetime(daily_snapshot["trade_date"]).dt.normalize().unique()))
    embargo_days = int(config["evaluation"]["embargo_days_daily"])
    rows = []
    prior_dates: list[pd.Timestamp] = []
    for index, window in enumerate(windows):
        start = pd.Timestamp(window["start"]).normalize()
        end = pd.Timestamp(window["end"]).normalize()
        test_dates = [date for date in all_dates if start <= date <= end]
        if index == 0:
            train_dates: list[pd.Timestamp] = []
            purged_dates: list[pd.Timestamp] = []
        else:
            train_dates = list(prior_dates)
            purged_dates = train_dates[-embargo_days:] if embargo_days else []
            if purged_dates:
                train_dates = train_dates[:-embargo_days]
        rows.append(
            {
                "window_id": window["id"],
                "window_role": window["role"],
                "train_start": min(train_dates) if train_dates else pd.NaT,
                "train_end": max(train_dates) if train_dates else pd.NaT,
                "train_days_after_purge": len(train_dates),
                "purged_days": len(purged_dates),
                "embargo_days": embargo_days,
                "test_start": min(test_dates) if test_dates else pd.NaT,
                "test_end": max(test_dates) if test_dates else pd.NaT,
                "test_days": len(test_dates),
                "random_split": False,
                "purging_applied": index > 0 and embargo_days > 0,
                "overlapping_labels_grouped_by_trade_date": True,
            }
        )
        prior_dates.extend(test_dates)
    return pd.DataFrame(rows)


def build_ablation_report(
    intraday_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    overlay_summary: pd.DataFrame,
    qiv_monotonicity: pd.DataFrame,
    config: dict[str, Any],
    intraday_trades: pd.DataFrame | None = None,
    qiv_restored_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    mapping = {
        "D1_SYNCHRONOUS_RESIDUAL": "D1_SYNCHRONOUS_RESIDUAL",
        "D2_DISTRIBUTED_LAG_GAP": "D2_DISTRIBUTED_LAG",
        "D3_NO_MARKET_ADJUSTMENT": "D3_NO_MARKET_ADJUSTMENT",
        "D3_GAP_REPAIR_STARTED": "D4_MARKET_ADJUSTED_REPAIR",
        "D4_GAP_REPAIR_LIQUIDITY": "D6_WITH_LIQUIDITY_SWEET_SPOT",
        "D5_DYNAMIC_LINKAGE_EXIT": "D10_LINKAGE_DYNAMIC_EXIT",
        "D6_HOT_MONEY_REJECT": "D8_WITH_HOT_MONEY_REJECT",
        "D7_NO_HOT_MONEY_REJECT": "D7_NO_HOT_MONEY_REJECT",
        "D9_FIXED_PERCENT_EXIT_BENCHMARK": "D9_FIXED_VOL_NORMALIZED_EXIT",
    }
    selected = intraday_summary[
        intraday_summary["entry_delay_bars"].eq(0)
        & intraday_summary["cost_scenario"].isin(["zero", "realistic"])
    ]
    for _, row in selected.iterrows():
        rows.append(
            {
                "ablation_name": mapping.get(row["strategy"], row["strategy"]),
                "strategy": row["strategy"],
                "cost_scenario": row["cost_scenario"],
                "gross_return": row["gross_return_mean"],
                "net_return": row["net_return_mean"],
                "trade_count": row["trade_count"],
                "hit_rate": row["hit_rate_net"],
                "mfe": row["mfe"],
                "mae": row["mae"],
                "max_drawdown": row["daily_cohort_max_drawdown"],
            }
        )
    daily_selected = daily_summary[
        daily_summary["selector"].eq("top10")
        & daily_summary["exit_rule"].eq("fixed_3d")
        & daily_summary["cost_scenario"].isin(["zero", "realistic"])
    ]
    for _, row in daily_selected.iterrows():
        rows.append(
            {
                "ablation_name": "D13_NO_LSMC_DAILY_SIMPLE_RV",
                "strategy": "daily_simple_rv_fixed_3d",
                "cost_scenario": row["cost_scenario"],
                "gross_return": row["gross_return_mean"],
                "net_return": row["net_return_mean"],
                "trade_count": row["trade_count"],
                "hit_rate": row["hit_rate"],
                "mfe": row["mfe"],
                "mae": row["mae"],
                "max_drawdown": row["max_drawdown"],
            }
        )
    if not overlay_summary.empty:
        overlay = overlay_summary[
            overlay_summary["exit_rule"].eq("lsmc_exit")
            & overlay_summary["cost_scenario"].isin(["zero", "realistic"])
        ]
        for _, row in overlay.iterrows():
            rows.append(
                {
                    "ablation_name": "D14_LSMC_EXIT_OVERLAY",
                    "strategy": "daily_rv_lsmc_exit_same_entries",
                    "cost_scenario": row["cost_scenario"],
                    "gross_return": row["gross_return"],
                    "net_return": row["net_return"],
                    "trade_count": row["matched_entry_count"],
                    "hit_rate": row["hit_rate"],
                    "mfe": row["mfe"],
                    "mae": row["mae"],
                    "max_drawdown": row["max_drawdown"],
                }
            )
    rows.append(
        {
            "ablation_name": "D15_QIV_MC_WEIGHT_ZERO_FORMAL_BASELINE",
            "strategy": "daily_simple_rv",
            "cost_scenario": "diagnostic",
            "qiv_mc_weight": 0.0,
            "qiv_monotonicity_mean": _safe_mean(
                qiv_monotonicity.get("spearman_bin_vs_3d", pd.Series(dtype=float))
            ),
        }
    )
    if qiv_restored_summary is not None and not qiv_restored_summary.empty:
        for _, row in qiv_restored_summary.iterrows():
            rows.append(
                {
                    "ablation_name": "D16_QIV_MC_RESTORED_DIAGNOSTIC_ONLY",
                    "strategy": "daily_rv_qiv_mc_restored",
                    "cost_scenario": row["cost_scenario"],
                    "gross_return": row["gross_return_mean"],
                    "net_return": row["net_return_mean"],
                    "trade_count": row["trade_count"],
                    "hit_rate": row["hit_rate"],
                    "mfe": row["mfe"],
                    "mae": row["mae"],
                    "max_drawdown": row["max_drawdown"],
                    "qiv_mc_weight": 0.40,
                    "qiv_monotonicity_mean": _safe_mean(
                        qiv_monotonicity.get(
                            "spearman_bin_vs_3d", pd.Series(dtype=float)
                        )
                    ),
                }
            )
    if intraday_trades is not None and not intraday_trades.empty:
        formal = intraday_trades[
            intraday_trades["strategy"].eq("D6_HOT_MONEY_REJECT")
            & intraday_trades["entry_delay_bars"].eq(0)
            & intraday_trades["cost_scenario"].isin(["zero", "realistic"])
        ]
        allowed_types = ["BALANCED_CORE", "EQUITY_LIKE", "BOND_LIKE"]
        for scenario, group in formal.groupby("cost_scenario"):
            for name, subset in [
                ("D11_NO_BOND_TYPE_HARD_GATE", group),
                ("D12_WITH_BOND_TYPE_HARD_GATE", group[group["bond_type"].isin(allowed_types)]),
            ]:
                rows.append(
                    {
                        "ablation_name": name,
                        "strategy": "D6_HOT_MONEY_REJECT",
                        "cost_scenario": scenario,
                        "gross_return": _safe_mean(subset["gross_return"]),
                        "net_return": _safe_mean(subset["net_return"]),
                        "trade_count": int(len(subset)),
                        "hit_rate": float((subset["net_return"] > 0).mean()) if len(subset) else np.nan,
                        "mfe": _safe_mean(subset["mfe"]),
                        "mae": _safe_mean(subset["mae"]),
                        "max_drawdown": drawdown_from_returns(
                            subset.groupby("trade_date")["net_return"].mean()
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result["experiment_id"] = [f"v071_ablation_{i:03d}" for i in range(len(result))]
    return result
