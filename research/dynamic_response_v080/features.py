from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir


LOGGER = logging.getLogger(__name__)
EPS = 1e-12
GROUP_KEYS = ["bond_code", "trade_date", "session_part"]


STAGE_A_NUMERIC_FEATURES = [
    "stock_return_current",
    "stock_idio_return_current",
    "stock_return_2bar",
    "stock_return_3bar",
    "stock_return_6bar",
    "stock_return_12bar",
    "stock_path_max_6bar",
    "stock_path_min_6bar",
    "stock_path_range_6bar",
    "stock_return_acceleration",
    "stock_realized_vol_6bar",
    "stock_realized_vol_12bar",
    "stock_shock_z",
    "stock_shock_amount_z",
    "stock_index_beta",
    "cb_return_lag1",
    "cb_return_past_2bar",
    "cb_return_past_3bar",
    "cb_return_past_6bar",
    "cb_return_past_12bar",
    "cb_path_max_past_6bar",
    "cb_path_min_past_6bar",
    "cb_realized_vol_past_6bar",
    "linkage_residual_lag1",
    "linkage_residual_z_lag1",
    "response_gap_lag1_proxy",
    "response_gap_change_lagged",
    "dynamic_beta_lag0",
    "dynamic_beta_lag1",
    "dynamic_beta_lag2",
    "dynamic_beta_lag3",
    "linkage_regression_r2_lag1",
    "linkage_model_confidence_lag1",
    "conversion_value_current",
    "moneyness_current",
    "premium_lag1",
    "premium_slot_residual_lag1",
    "relative_amount_lag1",
    "cb_amount_z_lag1",
    "high_low_range_z_lag1",
    "vwap_gap_lag1",
    "liquidity_sweet_spot_lag1",
    "hot_money_risk_lag1",
    "volume_climax_lag1",
    "daily_turnover_lag1d",
    "adv20_amount_lag1d",
    "active_bar_ratio_lag1d",
    "zero_bar_ratio_lag1d",
    "amihud_lag1d",
    "bar_slot_numeric",
    "minutes_from_session_open",
    "minutes_to_session_close",
    "time_sin",
    "time_cos",
    "weekday",
    "is_afternoon",
    "data_coverage_flag",
]

STAGE_A_CATEGORICAL_FEATURES = [
    "session_part",
    "chosen_market_index",
    "liquidity_regime_lag1",
]

BASELINE_OUTCOME_FEATURES = [
    "stock_return_current",
    "stock_idio_return_current",
    "dynamic_beta_lag0",
    "cb_market_return_loo",
    "cb_return_lag1",
    "linkage_residual_z_lag1",
    "bar_slot_numeric",
]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _grouped_rolling_sum(values: pd.Series, frame: pd.DataFrame, window: int, shift: int = 0) -> pd.Series:
    working = values.groupby([frame[key] for key in GROUP_KEYS], sort=False).shift(shift)
    return (
        working.groupby([frame[key] for key in GROUP_KEYS], sort=False)
        .rolling(window, min_periods=window)
        .sum()
        .reset_index(level=[0, 1, 2], drop=True)
        .reindex(frame.index)
    )


def _grouped_rolling_stat(
    values: pd.Series,
    frame: pd.DataFrame,
    window: int,
    statistic: str,
    shift: int = 0,
) -> pd.Series:
    working = values.groupby([frame[key] for key in GROUP_KEYS], sort=False).shift(shift)
    rolling = working.groupby([frame[key] for key in GROUP_KEYS], sort=False).rolling(
        window, min_periods=window
    )
    method = getattr(rolling, statistic)
    return method().reset_index(level=[0, 1, 2], drop=True).reindex(frame.index)


def _lag(values: pd.Series, frame: pd.DataFrame, periods: int = 1) -> pd.Series:
    return values.groupby([frame[key] for key in GROUP_KEYS], sort=False).shift(periods)


def _prior_daily_value(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    daily = (
        frame[["bond_code", "trade_date", column]]
        .drop_duplicates(["bond_code", "trade_date"], keep="last")
        .sort_values(["bond_code", "trade_date"])
    )
    daily[f"{column}_lag1d"] = daily.groupby("bond_code", sort=False)[column].shift(1)
    lookup = daily.set_index(["bond_code", "trade_date"])[f"{column}_lag1d"]
    index = pd.MultiIndex.from_frame(frame[["bond_code", "trade_date"]])
    return pd.Series(lookup.reindex(index).to_numpy(), index=frame.index, dtype=float)


def _leave_one_out_market_return(frame: pd.DataFrame) -> pd.Series:
    values = _numeric(frame, "cb_return")
    keys = [frame["trade_date"], frame["bar_slot"]]
    total = values.groupby(keys, sort=False).transform("sum")
    count = values.notna().groupby(keys, sort=False).transform("sum")
    return (total - values) / (count - 1).where(count.gt(1))


def build_feature_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build the V0.8 stage-A feature and future-label base table.

    Stage A predicts the current five-minute CB return after observing the current
    stock bar. The current CB return and any field mechanically derived from the
    current CB close are excluded from its feature list. CB path and microstructure
    inputs are lagged by at least one bar. Rolling paths reset at lunch and overnight.
    """
    base_columns = [
        "trade_date", "bar_start", "bar_end", "bar_slot", "session_part",
        "session_slot", "bond_code", "stock_code", "research_pool_scope",
        "window_id", "window_role", "pair_data_quality_pass", "cb_open",
        "cb_high", "cb_low", "cb_close", "cb_amount", "stock_open",
        "stock_high", "stock_low", "stock_close", "stock_amount",
        "cb_return", "stock_return", "stock_idiosyncratic_return",
        "conversion_price", "conversion_price_asof_valid",
    ]
    optional = [
        "cb_source_vendor", "stock_source_vendor", "cb_market_return",
        "chosen_market_index", "stock_index_beta", "stock_shock_z",
        "stock_shock_amount_z", "cb_amount_z_5m", "relative_amount",
        "high_low_range_proxy_z", "vwap_gap", "premium_expansion_proxy_z",
        "stock_to_cb_beta_lag0", "stock_to_cb_beta_lag1",
        "stock_to_cb_beta_lag2", "stock_to_cb_beta_lag3", "cb_market_gamma",
        "linkage_regression_intercept", "linkage_regression_r2",
        "linkage_model_confidence", "linkage_residual", "linkage_residual_z",
        "linkage_gap_z", "liquidity_sweet_spot_score_5m",
        "hot_money_risk_score_5m", "volume_climax_score",
        "liquidity_regime_5m", "daily_turnover", "adv20_amount",
        "active_bar_ratio", "zero_bar_ratio", "amihud",
    ]
    selected = list(dict.fromkeys(base_columns + [c for c in optional if c in frame.columns]))
    out = frame[selected].copy()
    out = out.sort_values(["bond_code", "trade_date", "bar_slot"]).reset_index(drop=True)

    stock_return = _numeric(out, "stock_return")
    stock_idio = _numeric(out, "stock_idiosyncratic_return")
    cb_return = _numeric(out, "cb_return")
    out["stock_return_current"] = stock_return
    out["stock_idio_return_current"] = stock_idio
    for window in [2, 3, 6, 12]:
        out[f"stock_return_{window}bar"] = _grouped_rolling_sum(stock_return, out, window)
        out[f"cb_return_past_{window}bar"] = _grouped_rolling_sum(cb_return, out, window, shift=1)
    out["stock_path_max_6bar"] = _grouped_rolling_stat(stock_return, out, 6, "max")
    out["stock_path_min_6bar"] = _grouped_rolling_stat(stock_return, out, 6, "min")
    out["stock_path_range_6bar"] = out["stock_path_max_6bar"] - out["stock_path_min_6bar"]
    out["stock_return_acceleration"] = stock_return - _lag(stock_return, out)
    out["stock_realized_vol_6bar"] = _grouped_rolling_stat(stock_return, out, 6, "std")
    out["stock_realized_vol_12bar"] = _grouped_rolling_stat(stock_return, out, 12, "std")
    out["cb_return_lag1"] = _lag(cb_return, out)
    out["cb_path_max_past_6bar"] = _grouped_rolling_stat(cb_return, out, 6, "max", shift=1)
    out["cb_path_min_past_6bar"] = _grouped_rolling_stat(cb_return, out, 6, "min", shift=1)
    out["cb_realized_vol_past_6bar"] = _grouped_rolling_stat(cb_return, out, 6, "std", shift=1)

    residual = _numeric(out, "linkage_residual")
    residual_z = _numeric(out, "linkage_residual_z")
    gap_proxy = _numeric(out, "linkage_gap_z")
    out["linkage_residual_lag1"] = _lag(residual, out)
    out["linkage_residual_z_lag1"] = _lag(residual_z, out)
    out["response_gap_lag1_proxy"] = _lag(gap_proxy, out)
    out["response_gap_change_lagged"] = out["response_gap_lag1_proxy"] - _lag(gap_proxy, out, 2)
    for lag in range(4):
        out[f"dynamic_beta_lag{lag}"] = _numeric(out, f"stock_to_cb_beta_lag{lag}")
    out["linkage_regression_r2_lag1"] = _lag(_numeric(out, "linkage_regression_r2"), out)
    out["linkage_model_confidence_lag1"] = _lag(_numeric(out, "linkage_model_confidence"), out)

    conversion_price = _numeric(out, "conversion_price")
    conversion_value = _numeric(out, "stock_close") / conversion_price.where(conversion_price.gt(0)) * 100.0
    out["conversion_value_current"] = conversion_value
    out["moneyness_current"] = conversion_value / 100.0
    previous_cb_close = _lag(_numeric(out, "cb_close"), out)
    previous_stock_close = _lag(_numeric(out, "stock_close"), out)
    previous_conversion_value = previous_stock_close / conversion_price.where(conversion_price.gt(0)) * 100.0
    out["premium_lag1"] = previous_cb_close / previous_conversion_value.where(previous_conversion_value.gt(0)) - 1.0
    out["premium_slot_residual_lag1"] = _lag(_numeric(out, "premium_expansion_proxy_z"), out)

    lag_sources = {
        "relative_amount_lag1": "relative_amount",
        "cb_amount_z_lag1": "cb_amount_z_5m",
        "high_low_range_z_lag1": "high_low_range_proxy_z",
        "vwap_gap_lag1": "vwap_gap",
        "liquidity_sweet_spot_lag1": "liquidity_sweet_spot_score_5m",
        "hot_money_risk_lag1": "hot_money_risk_score_5m",
        "volume_climax_lag1": "volume_climax_score",
    }
    for target, source in lag_sources.items():
        out[target] = _lag(_numeric(out, source), out)
    liquidity = out.get("liquidity_regime_5m", pd.Series("UNKNOWN", index=out.index)).astype(str)
    out["liquidity_regime_lag1"] = liquidity.groupby(
        [out[key] for key in GROUP_KEYS], sort=False
    ).shift(1).fillna("UNKNOWN")
    for column in ["daily_turnover", "adv20_amount", "active_bar_ratio", "zero_bar_ratio", "amihud"]:
        out[f"{column}_lag1d"] = _prior_daily_value(out, column)

    out["cb_market_return_loo"] = _leave_one_out_market_return(out)
    out["baseline_v071_prediction"] = (
        _numeric(out, "linkage_regression_intercept")
        + out["dynamic_beta_lag0"] * stock_idio
        + _numeric(out, "cb_market_gamma") * _numeric(out, "cb_market_return")
    )
    out["baseline_fair_prediction"] = (
        _numeric(out, "linkage_regression_intercept")
        + out["dynamic_beta_lag0"] * stock_idio
        + _numeric(out, "cb_market_gamma") * out["cb_market_return_loo"]
    )

    out["bar_slot_numeric"] = _numeric(out, "bar_slot")
    out["minutes_from_session_open"] = _numeric(out, "session_slot") * 5.0
    out["minutes_to_session_close"] = (23.0 - _numeric(out, "session_slot")) * 5.0
    angle = 2.0 * np.pi * out["bar_slot_numeric"] / 48.0
    out["time_sin"] = np.sin(angle)
    out["time_cos"] = np.cos(angle)
    out["weekday"] = pd.to_datetime(out["trade_date"]).dt.weekday.astype(float)
    out["is_afternoon"] = out["session_part"].eq("PM").astype(float)
    out["data_coverage_flag"] = out["pair_data_quality_pass"].fillna(False).astype(float)

    out["surface_target_cb_return"] = cb_return
    for column in STAGE_A_NUMERIC_FEATURES + ["surface_target_cb_return"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float32")
    for column in STAGE_A_CATEGORICAL_FEATURES:
        if column not in out.columns:
            out[column] = "UNKNOWN"
        out[column] = out[column].astype(str).fillna("UNKNOWN")
    if "surface_target_cb_return" in STAGE_A_NUMERIC_FEATURES:
        raise AssertionError("Current CB return leaked into Stage A features")
    return out


def _future_shift(frame: pd.DataFrame, column: str, bars: int) -> pd.Series:
    values = _numeric(frame, column)
    return values.groupby([frame[key] for key in GROUP_KEYS], sort=False).shift(-bars)


def add_forward_labels(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Add same-session future labels.

    For horizon h, theoretical returns are close(t+h)/close(t)-1. Executable
    returns use open(t+1) as entry and close(t+h) as exit. Labels are available
    only when both entry and exit slots are exact members of the same AM/PM
    session. MFE and MAE use highs/lows from t+1 through t+h. Residual return is
    r_CB(t,t+h) - beta_lag0(t) * r_stock(t,t+h).
    """
    out = frame.copy()
    label_columns: list[str] = []
    current_cb = _numeric(out, "cb_close")
    current_stock = _numeric(out, "stock_close")
    beta = _numeric(out, "dynamic_beta_lag0")
    for bars in [int(value) for value in config["labels"]["horizons_bars"]]:
        suffix = f"{bars * 5}m"
        future_cb = _future_shift(out, "cb_close", bars)
        future_stock = _future_shift(out, "stock_close", bars)
        future_slot = _future_shift(out, "bar_slot", bars)
        next_cb_open = _future_shift(out, "cb_open", 1)
        next_slot = _future_shift(out, "bar_slot", 1)
        valid = future_slot.sub(_numeric(out, "bar_slot")).eq(bars).fillna(False)
        executable = valid & next_slot.sub(_numeric(out, "bar_slot")).eq(1).fillna(False)
        cb_forward = (future_cb / current_cb - 1.0).where(valid & current_cb.gt(0) & future_cb.gt(0))
        stock_forward = (future_stock / current_stock - 1.0).where(
            valid & current_stock.gt(0) & future_stock.gt(0)
        )
        residual_forward = cb_forward - beta * stock_forward
        highs = [_future_shift(out, "cb_high", step) for step in range(1, bars + 1)]
        lows = [_future_shift(out, "cb_low", step) for step in range(1, bars + 1)]
        high_path = pd.concat(highs, axis=1).max(axis=1, skipna=False)
        low_path = pd.concat(lows, axis=1).min(axis=1, skipna=False)
        names = {
            f"forward_cb_return_{suffix}": cb_forward,
            f"forward_executable_cb_return_{suffix}": (
                future_cb / next_cb_open - 1.0
            ).where(executable & next_cb_open.gt(0) & future_cb.gt(0)),
            f"forward_stock_return_{suffix}": stock_forward,
            f"forward_residual_return_{suffix}": residual_forward,
            f"mfe_{suffix}": (high_path / current_cb - 1.0).where(valid),
            f"mae_{suffix}": (low_path / current_cb - 1.0).where(valid),
            f"executable_mfe_{suffix}": (high_path / next_cb_open - 1.0).where(executable),
            f"executable_mae_{suffix}": (low_path / next_cb_open - 1.0).where(executable),
            f"label_available_{suffix}": valid.astype(bool),
        }
        for name, values in names.items():
            out[name] = values.astype("bool" if name.startswith("label_available") else "float32")
            label_columns.append(name)
    return out, label_columns


def add_oof_gap_and_closure_labels(
    frame: pd.DataFrame,
    prediction_column: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Add causal response-gap labels and a leader/lagger decomposition.

    g_t = r_CB,t - rhat_CB,t. For horizon h, a first-order future gap is
    g_(t+h) = g_t + R_CB(t,t+h) - beta_t * R_stock(t,t+h).

    Let d=-sign(g_t), the direction required to close the gap. Lagger catch-up
    contribution is d*R_CB; leader reversal contribution is -d*beta*R_stock.
    Positive values of either contribution reduce the original gap. This keeps a
    falling absolute gap caused by stock reversal distinct from CB catch-up.
    """
    if prediction_column not in frame.columns:
        raise ValueError(f"Missing OOF prediction column: {prediction_column}")
    out = frame.copy()
    prediction = _numeric(out, prediction_column)
    target = _numeric(out, "surface_target_cb_return")
    out["response_gap"] = (target - prediction).astype("float32")
    # Standardization is fitted elsewhere per training fold; this online proxy uses
    # only prior bars in the same bond/session and is never fit on future data.
    grouped = out["response_gap"].groupby([out[key] for key in GROUP_KEYS], sort=False)
    past_mean = grouped.transform(lambda values: values.shift(1).expanding(min_periods=12).mean())
    past_std = grouped.transform(lambda values: values.shift(1).expanding(min_periods=12).std())
    out["response_gap_z"] = ((out["response_gap"] - past_mean) / past_std.where(past_std.gt(EPS))).astype("float32")
    out["response_gap_change"] = grouped.diff().astype("float32")
    label_columns = ["response_gap", "response_gap_z", "response_gap_change"]
    threshold = float(config["labels"]["significant_gap_closure_ratio"])
    beta = _numeric(out, "dynamic_beta_lag0")
    gap = _numeric(out, "response_gap")
    required_direction = -np.sign(gap)
    for bars in [int(value) for value in config["labels"]["horizons_bars"]]:
        suffix = f"{bars * 5}m"
        cb_forward = _numeric(out, f"forward_cb_return_{suffix}")
        stock_forward = _numeric(out, f"forward_stock_return_{suffix}")
        future_gap = gap + cb_forward - beta * stock_forward
        abs_reduction = gap.abs() - future_gap.abs()
        minimum_gap = float(config["labels"].get("minimum_gap_for_ratio_bps", 1.0)) / 10000.0
        close_ratio = abs_reduction / gap.abs().where(gap.abs().ge(minimum_gap))
        clip_low, clip_high = config["labels"].get("gap_close_ratio_clip", [-5.0, 5.0])
        close_ratio_clipped = close_ratio.clip(float(clip_low), float(clip_high))
        lagger = required_direction * cb_forward
        leader = -required_direction * beta * stock_forward
        direction_consistent = np.sign(future_gap).eq(np.sign(gap))
        available = close_ratio.notna()
        significant = pd.Series(pd.NA, index=out.index, dtype="boolean")
        significant.loc[available] = close_ratio.loc[available].ge(threshold)
        lagger_flag = pd.Series(pd.NA, index=out.index, dtype="boolean")
        leader_flag = pd.Series(pd.NA, index=out.index, dtype="boolean")
        direction_flag = pd.Series(pd.NA, index=out.index, dtype="boolean")
        lagger_flag.loc[available] = (lagger.loc[available].gt(0) & lagger.loc[available].ge(leader.loc[available])).to_numpy()
        leader_flag.loc[available] = (leader.loc[available].gt(0) & leader.loc[available].gt(lagger.loc[available])).to_numpy()
        direction_flag.loc[available] = direction_consistent.loc[available].to_numpy()
        names = {
            f"future_response_gap_{suffix}": future_gap,
            f"gap_abs_reduction_{suffix}": abs_reduction,
            f"gap_close_ratio_{suffix}": close_ratio,
            f"gap_close_ratio_clipped_{suffix}": close_ratio_clipped,
            f"gap_significant_closure_{suffix}": significant,
            f"lagger_catchup_contribution_{suffix}": lagger,
            f"leader_reversal_contribution_{suffix}": leader,
            f"lagger_catchup_flag_{suffix}": lagger_flag,
            f"leader_reversal_flag_{suffix}": leader_flag,
            f"gap_direction_consistent_{suffix}": direction_flag,
        }
        for name, values in names.items():
            out[name] = values.astype("boolean" if name.endswith("flag_" + suffix) or "significant_closure" in name or "direction_consistent" in name else "float32")
            label_columns.append(name)
    return out, label_columns


def write_feature_schema(
    frame: pd.DataFrame,
    config: dict[str, Any],
    label_columns: list[str],
) -> Path:
    path = output_dir(config) / "feature_schema.json"
    schema = {
        "stage_a_numeric_features": STAGE_A_NUMERIC_FEATURES,
        "stage_a_categorical_features": STAGE_A_CATEGORICAL_FEATURES,
        "baseline_outcome_features": BASELINE_OUTCOME_FEATURES,
        "labels": label_columns,
        "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
        "same_bar_cb_return_forbidden": True,
        "same_day_daily_aggregate_forbidden": True,
        "current_bar_cutoff": "after current stock and CB bars close; current CB return is target only",
        "lunch_policy": "AM and PM rolling windows and labels are separate",
        "overnight_policy": "no intraday feature window or label crosses trade_date",
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path
