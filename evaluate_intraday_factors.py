from __future__ import annotations

import argparse
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from intraday_factors.config import load_config, load_schema
from intraday_factors.data_source import IntradayDataAdapter, get_connection, query_to_dataframe, sql_string
from intraday_factors.engine import FactorEngine
from intraday_factors.pool_state import ACTIVE, ACTIVE_MANUAL, ACTIVE_PENDING_HISTORY, OBSERVE, OBSERVE_MANUAL, SHADOW_RESEARCH, normalize_pool_state


LOGGER = logging.getLogger(__name__)


def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    return float(x[valid].rank().corr(y[valid].rank()))


def max_drawdown(returns: pd.Series) -> float:
    if returns.dropna().empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def add_forward_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["bond_code", "bar_start"]).copy()
    panel["signal_price"] = panel["cb_close"]
    panel["next_bar_open"] = panel.groupby("bond_code")["cb_open"].shift(-1)
    panel["first_executable_price_after_seen"] = panel["next_bar_open"]
    for bars, label in [(1, "1bar"), (2, "2bar"), (3, "3bar"), (6, "30m"), (12, "60m")]:
        panel[f"forward_cb_return_{label}"] = panel.groupby("bond_code")["cb_close"].shift(-bars) / panel["cb_close"] - 1.0
    panel["forward_stock_return_30m"] = panel.groupby("bond_code")["stock_close"].shift(-6) / panel["stock_close"] - 1.0
    panel["forward_stock_return_60m"] = panel.groupby("bond_code")["stock_close"].shift(-12) / panel["stock_close"] - 1.0
    for label in ["30m", "60m"]:
        market_forward = panel.groupby("bar_start")[f"forward_cb_return_{label}"].median().rename(f"forward_cb_market_return_{label}")
        panel = panel.merge(market_forward, on="bar_start", how="left")

    for horizon, label in [(3, "3bar"), (6, "6bar")]:
        future_returns = []
        future_prices = []
        for step in range(1, horizon + 1):
            future_returns.append(panel.groupby("bond_code")["cb_close"].shift(-step) / panel["cb_close"] - 1.0)
            future_prices.append(panel.groupby("bond_code")["cb_close"].shift(-step))
        future = pd.concat(future_returns, axis=1)
        prices = pd.concat(future_prices, axis=1)
        panel[f"mfe_{label}"] = future.max(axis=1)
        panel[f"mae_{label}"] = future.min(axis=1)
        panel[f"best_price_{label}"] = prices.max(axis=1)
        panel[f"worst_price_{label}"] = prices.min(axis=1)

    for horizon, label in [(6, "30m"), (12, "60m")]:
        future_returns = []
        for step in range(1, horizon + 1):
            future_returns.append(panel.groupby("bond_code")["cb_close"].shift(-step) / panel["cb_close"] - 1.0)
        future = pd.concat(future_returns, axis=1)
        panel[f"mfe_{label}"] = future.max(axis=1)
        panel[f"mae_{label}"] = future.min(axis=1)
        future_prices = []
        for step in range(1, horizon + 1):
            future_prices.append(panel.groupby("bond_code")["cb_close"].shift(-step))
        prices = pd.concat(future_prices, axis=1)
        panel[f"best_price_{label}"] = prices.max(axis=1)
        panel[f"worst_price_{label}"] = prices.min(axis=1)
    return panel


def normalize_time_window(
    start: str | None,
    end: str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[str, str]:
    if start_date:
        start = f"{start_date} 09:30:00"
    if end_date:
        end = f"{end_date} 15:00:00"
    if not start or not end:
        raise SystemExit("requires --start/--end or --start-date/--end-date")
    return start, end


def pool_scope_states(pool_scope: str | None) -> set[str]:
    scope = (pool_scope or "ACTIVE_ONLY").upper()
    if scope == "ACTIVE_ONLY":
        return {ACTIVE, ACTIVE_MANUAL}
    if scope == "ACTIVE_OBSERVE":
        return {ACTIVE, ACTIVE_MANUAL, OBSERVE, OBSERVE_MANUAL, ACTIVE_PENDING_HISTORY}
    if scope == "ACTIVE_OBSERVE_SHADOW":
        return {ACTIVE, ACTIVE_MANUAL, OBSERVE, OBSERVE_MANUAL, ACTIVE_PENDING_HISTORY, SHADOW_RESEARCH}
    states = set()
    for item in scope.split(","):
        token = item.strip()
        if token:
            states.add(normalize_pool_state(token))
    return states or {ACTIVE, ACTIVE_MANUAL}


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["parity_bucket"] = pd.cut(
        out["parity"],
        bins=[-np.inf, 90, 100, 115, 130, np.inf],
        labels=["parity_lt_90", "parity_90_100", "parity_100_115", "parity_115_130", "parity_ge_130"],
    ).astype("object")
    close_col = "cb_close" if "cb_close" in out.columns else "signal_price"
    out["bond_price_bucket"] = pd.cut(
        out[close_col],
        bins=[-np.inf, 110, 120, 130, np.inf],
        labels=["price_lt_110", "price_110_120", "price_120_130", "price_ge_130"],
    ).astype("object")
    out["premium_bucket"] = pd.cut(
        out["premium_raw"],
        bins=[-np.inf, 0.20, 0.25, 0.30, np.inf],
        labels=["premium_lt_20", "premium_20_25", "premium_25_30", "premium_ge_30"],
    ).astype("object")
    liquidity_source = out["adv20_amount"].combine_first(out.get("relative_amount", pd.Series(np.nan, index=out.index)))
    try:
        out["liquidity_bucket"] = pd.qcut(liquidity_source.rank(method="first"), 3, labels=["liq_low", "liq_mid", "liq_high"]).astype("object")
    except ValueError:
        out["liquidity_bucket"] = "liq_unavailable"
    if "empirical_delta_proxy" in out.columns:
        try:
            out["empirical_delta_bucket"] = pd.qcut(
                out["empirical_delta_proxy"].rank(method="first"),
                3,
                labels=["delta_low", "delta_mid", "delta_high"],
            ).astype("object")
        except ValueError:
            out["empirical_delta_bucket"] = "delta_unavailable"
    else:
        out["empirical_delta_bucket"] = "delta_unavailable"
    hour_min = out["bar_start"].dt.hour * 60 + out["bar_start"].dt.minute
    out["time_bucket"] = np.select(
        [
            hour_min < 10 * 60 + 30,
            hour_min < 11 * 60 + 30,
            hour_min < 14 * 60,
            hour_min < 14 * 60 + 30,
            hour_min < 15 * 60,
        ],
        ["open_0930_1030", "late_morning", "early_pm", "pm_1400_1430", "tail_1430_1500"],
        default="close_or_other",
    )
    return out


def research_experiment_grid() -> dict[str, object]:
    return {
        "lag_repair_version_a_early": {
            "residual_z_max": [-0.6, -0.8, -1.0],
            "stock_mom_30m_z_min": [0.5, 0.7, 1.0],
            "relative_amount_min": [0.5, 0.8, 1.0],
            "premium_slot_z_max": [0.8, 1.0, 1.2],
            "flow_confirmation_min": [-0.2, 0.0, 0.2],
        },
        "lag_repair_version_b_repair_confirmed": {
            "inherits": "lag_repair_version_a_early",
            "additional": ["residual_z_change_1bar > 0 or residual_z_change_2bar > 0", "cb_mom_10m_z >= 0 or cb_flow_z >= 0"],
        },
        "momentum_breakout": {
            "stock_mom_30m_z_min": [0.8, 1.0, 1.2],
            "stock_impulse_score_min": [0.5, 0.8, 1.0],
            "cb_mom_10m_z_min": [0.2, 0.5, 0.8],
            "cb_mom_15m_z_min": [0.2, 0.5, 0.8],
            "relative_amount_min": [0.7, 1.0, 1.3],
            "residual_z_upper": [0.5, 0.8, 1.2],
            "cb_flow_z_min": [0.0, 0.3, 0.6],
            "premium_change_10m_z_max": [1.0, 1.5],
            "premium_slot_z_max": [1.0, 1.2, 1.5],
        },
        "no_new_action_after": ["14:20:00", "14:30:00", "14:40:00", "14:50:00"],
    }


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def _as_bool(df: pd.DataFrame, col: str) -> pd.Series:
    return _bool_series(df, col)


def _scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return np.nan if np.isnan(value) else value
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def _records(df: pd.DataFrame, columns: list[str], limit: int = 50) -> list[dict[str, object]]:
    if df.empty:
        return []
    out = []
    for row in df[columns].head(limit).to_dict("records"):
        out.append({key: _scalar(value) for key, value in row.items()})
    return out


def _return_stats(sample: pd.DataFrame, col: str, cost: pd.Series | float = 0.0) -> dict[str, object]:
    values = sample[col] if col in sample.columns else pd.Series(dtype=float)
    net = values - cost
    return {
        "valid_count": int(values.notna().sum()),
        "mean": float(values.mean()) if values.notna().any() else np.nan,
        "median": float(values.median()) if values.notna().any() else np.nan,
        "hit_rate_net": float((net.dropna() > 0).mean()) if net.notna().any() else np.nan,
    }


def _metric_summary(rows: pd.DataFrame, signal_type: str | None = None) -> dict[str, object]:
    sample = rows if signal_type is None else rows[rows["signal_type"] == signal_type]
    cost = sample.get("cost", pd.Series(0.0, index=sample.index))
    out: dict[str, object] = {"sample_count": int(len(sample))}
    for label in ["1bar", "2bar", "3bar"]:
        stats = _return_stats(sample, f"forward_cb_return_{label}", cost)
        out[f"return_after_{label}"] = stats
    for horizon in ["30m", "60m"]:
        gross_stats = _return_stats(sample, f"forward_cb_return_{horizon}", cost)
        residual = sample[f"forward_residual_return_{horizon}"] if f"forward_residual_return_{horizon}" in sample.columns else pd.Series(dtype=float)
        out[f"return_{horizon}"] = gross_stats
        out[f"residual_return_{horizon}"] = {
            "valid_count": int(residual.notna().sum()),
            "mean": float(residual.mean()) if residual.notna().any() else np.nan,
            "median": float(residual.median()) if residual.notna().any() else np.nan,
        }
        out[f"mfe_{horizon}_mean"] = float(sample[f"mfe_{horizon}"].mean()) if sample[f"mfe_{horizon}"].notna().any() else np.nan
        out[f"mae_{horizon}_mean"] = float(sample[f"mae_{horizon}"].mean()) if sample[f"mae_{horizon}"].notna().any() else np.nan
    return out


def gate_stats(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    gates = [
        "data_quality_pass",
        "liquidity_pass",
        "factor_coverage_pass",
        "volatility_gate_pass",
        "trend_gate",
        "missing_cb_bar",
        "missing_stock_bar",
        "stale_price_flag",
        "missing_vol_gate",
        "missing_range_gate",
        "fallback_standardization_flag",
    ]
    out: dict[str, dict[str, int]] = {}
    for col in gates:
        if col not in df.columns:
            continue
        values = df[col]
        bool_values = values.fillna(False).astype(bool)
        out[col] = {
            "true": int(bool_values.sum()),
            "false": int((~bool_values).sum()),
            "missing": int(values.isna().sum()),
        }
    return out


def watch_conversion_summary(df: pd.DataFrame, action_types: set[str], armed_types: set[str] | None = None) -> dict[str, object]:
    armed_types = armed_types or set()
    watch = df[df["signal_type"] == "WATCH_LONG"].copy()
    actions = df[(df["alert_state"] == "ACTION_LONG") & df["signal_type"].isin(action_types)].copy()
    armed = df[(df["alert_state"] == "ARMED_LONG") | df["signal_type"].isin(armed_types)].copy()
    if watch.empty:
        return {"watch_count": 0, "to_armed_count": 0, "to_action_count": 0, "to_armed_rate": np.nan, "to_action_rate": np.nan}
    to_action = 0
    to_armed = 0
    for row in watch.itertuples(index=False):
        later = actions[
            (actions["bond_code"].astype(str) == str(row.bond_code))
            & (actions["bar_start"].dt.strftime("%Y-%m-%d") == row.bar_start.strftime("%Y-%m-%d"))
            & (actions["bar_start"] > row.bar_start)
        ]
        later_armed = armed[
            (armed["bond_code"].astype(str) == str(row.bond_code))
            & (armed["bar_start"].dt.strftime("%Y-%m-%d") == row.bar_start.strftime("%Y-%m-%d"))
            & (armed["bar_start"] > row.bar_start)
        ]
        to_action += int(not later.empty)
        to_armed += int(not later_armed.empty)
    return {
        "watch_count": int(len(watch)),
        "to_armed_count": int(to_armed),
        "to_action_count": int(to_action),
        "to_armed_rate": float(to_armed / len(watch)) if len(watch) else np.nan,
        "to_action_rate": float(to_action / len(watch)) if len(watch) else np.nan,
        "watch_forward_summary": _metric_summary(watch),
        "by_watch_signal_type": per_group_summary(watch, "watch_signal_type") if "watch_signal_type" in watch.columns else {},
    }


def armed_conversion_summary(df: pd.DataFrame, action_types: set[str], armed_types: set[str]) -> dict[str, object]:
    armed = df[(df["alert_state"] == "ARMED_LONG") | df["signal_type"].isin(armed_types)].copy()
    actions = df[(df["alert_state"] == "ACTION_LONG") & df["signal_type"].isin(action_types)].copy()
    if armed.empty:
        return {"armed_count": 0, "to_action_count": 0, "to_action_rate": np.nan, "avg_bars_to_action": np.nan}
    converted = 0
    bars_to_action = []
    false_positive_rows = []
    for idx, row in armed.iterrows():
        later = actions[
            (actions["bond_code"].astype(str) == str(row["bond_code"]))
            & (actions["bar_start"].dt.strftime("%Y-%m-%d") == row["bar_start"].strftime("%Y-%m-%d"))
            & (actions["bar_start"] > row["bar_start"])
        ]
        if later.empty:
            false_positive_rows.append(idx)
            continue
        converted += 1
        bars_to_action.append((later["bar_start"].iloc[0] - row["bar_start"]).total_seconds() / 300.0)
    false_positive = armed.loc[[idx for idx in false_positive_rows if idx in armed.index]]
    return {
        "armed_count": int(len(armed)),
        "to_action_count": int(converted),
        "to_action_rate": float(converted / len(armed)) if len(armed) else np.nan,
        "avg_bars_to_action": float(np.nanmean(bars_to_action)) if bars_to_action else np.nan,
        "armed_forward_summary": _metric_summary(armed),
        "armed_false_positive_summary": _metric_summary(false_positive),
        "by_armed_signal_type": per_group_summary(armed, "signal_type"),
    }


def action_time_distribution(action_rows: pd.DataFrame) -> dict[str, int]:
    if action_rows.empty:
        return {}
    return action_rows["bar_start"].dt.strftime("%H:%M").value_counts().sort_index().astype(int).to_dict()


def daily_action_counts(action_rows: pd.DataFrame) -> dict[str, int]:
    if action_rows.empty:
        return {}
    return action_rows.groupby(action_rows["bar_start"].dt.strftime("%Y-%m-%d")).size().astype(int).to_dict()


def bond_day_action_counts(action_rows: pd.DataFrame) -> dict[tuple[str, str], int]:
    if action_rows.empty:
        return {}
    return action_rows.groupby([action_rows["bar_start"].dt.strftime("%Y-%m-%d"), "bond_code"]).size().astype(int).to_dict()


def infer_daily_health(
    factors: pd.DataFrame,
    alerts: pd.DataFrame,
    latency_threshold_minutes: float,
    exclude_dates: set[str],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    if factors.empty:
        return out
    work = factors.copy()
    work["trade_date"] = work["bar_start"].dt.strftime("%Y-%m-%d")
    work["excluded_by_user"] = work["trade_date"].isin(exclude_dates)
    if "calculated_at" in work.columns:
        work["calculated_at"] = pd.to_datetime(work["calculated_at"], errors="coerce")
        raw_delay = (work["calculated_at"] - work["bar_start"]).dt.total_seconds() / 60.0
        work["delay_minutes_raw"] = raw_delay
    else:
        work["delay_minutes_raw"] = np.nan

    alert_counts = {}
    if not alerts.empty and "bar_start" in alerts.columns:
        tmp = alerts.copy()
        tmp["bar_start"] = pd.to_datetime(tmp["bar_start"], errors="coerce")
        tmp["trade_date"] = tmp["bar_start"].dt.strftime("%Y-%m-%d")
        alert_counts = tmp.groupby("trade_date").size().astype(int).to_dict()

    for trade_date, group in work.groupby("trade_date"):
        min_delay = float(group["delay_minutes_raw"].min()) if group["delay_minutes_raw"].notna().any() else np.nan
        max_delay = float(group["delay_minutes_raw"].max()) if group["delay_minutes_raw"].notna().any() else np.nan
        clock_mismatch = bool(pd.notna(min_delay) and min_delay < -60.0)
        delayed = bool(pd.notna(max_delay) and max_delay > latency_threshold_minutes)
        out[trade_date] = {
            "factor_rows": int(len(group)),
            "alert_rows": int(alert_counts.get(trade_date, 0)),
            "max_bar_start": _scalar(group["bar_start"].max()),
            "max_calculated_at": _scalar(group["calculated_at"].max()) if "calculated_at" in group.columns else None,
            "min_delay_minutes_raw": min_delay,
            "max_delay_minutes_raw": max_delay,
            "latency_clock_mismatch": clock_mismatch,
            "delay_over_threshold": delayed,
            "excluded_by_user": bool(group["excluded_by_user"].any()),
            "unreliable_live_day": bool(clock_mismatch or delayed or group["excluded_by_user"].any()),
            "delivery_mode_inferred": "unknown_clock_mismatch" if clock_mismatch else ("delayed_or_backfill" if delayed else "live_like"),
        }
    return out


def make_action_explanation_rows(action_rows: pd.DataFrame, limit: int = 50) -> list[dict[str, object]]:
    columns = [
        "bar_start",
        "bond_code",
        "stock_code",
        "signal_type",
        "watch_signal_type",
        "alert_state",
        "pool_state",
        "setup_score",
        "trigger_score",
        "exit_type",
        "residual_z",
        "residual_z_change_1bar",
        "residual_z_change_2bar",
        "stock_mom_30m_z",
        "stock_mom_60m_z",
        "stock_impulse_score",
        "cb_mom_10m_z",
        "cb_mom_15m_z",
        "premium_slot_z",
        "premium_change_10m_z",
        "premium_change_30m_z",
        "relative_amount",
        "cb_flow_z",
        "flow_confirmation",
        "vwap_gap",
        "market_regime_intraday",
        "volatility_gate_pass",
        "signal_price",
        "next_bar_open",
        "first_executable_price_after_seen",
        "slippage_vs_next_bar_open",
        "slippage_vs_first_seen_price",
        "forward_cb_return_1bar",
        "forward_cb_return_2bar",
        "forward_cb_return_3bar",
        "forward_cb_return_30m",
        "forward_residual_return_30m",
        "mfe_30m",
        "mae_30m",
        "action_reason_text",
        "watch_missing_to_action",
        "risk_hint_text",
        "cooldown_suppressed",
        "action_blocked_by_time",
        "first_seen_time",
        "alert_expire_time",
    ]
    existing = [col for col in columns if col in action_rows.columns]
    return _records(action_rows.sort_values(["bar_start", "bond_code"]), existing, limit)


def per_group_summary(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, object]]:
    if df.empty or group_col not in df.columns:
        return {}
    out = {}
    for key, group in df.groupby(group_col):
        out[str(key)] = _metric_summary(group)
    return out


def _episode_anchor_row(group: pd.DataFrame, action_types: set[str], armed_types: set[str]) -> pd.Series:
    actions = group[(group["alert_state"] == "ACTION_LONG") & group["signal_type"].isin(action_types)]
    if not actions.empty:
        return actions.iloc[0]
    armed = group[(group["alert_state"] == "ARMED_LONG") | group["signal_type"].isin(armed_types)]
    if not armed.empty:
        return armed.iloc[0]
    return group.iloc[0]


def build_episodes(df: pd.DataFrame, action_types: set[str], armed_types: set[str], daily_health: dict[str, dict[str, object]] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.sort_values(["bond_code", "bar_start"]).copy()
    work["trade_date"] = work["bar_start"].dt.strftime("%Y-%m-%d")
    signal_mask = (
        work["signal_type"].isin(["WATCH_LONG", *action_types, *armed_types])
        | work["alert_state"].isin(["WATCH_ONLY", "ARMED_LONG", "ACTION_LONG", "EXIT_HINT"])
    )
    episodes = []
    for (trade_date, bond_code), group in work.groupby(["trade_date", "bond_code"], sort=False):
        active = signal_mask.loc[group.index]
        current_indices: list[int] = []
        episode_no = 0
        previous_ts = None
        for idx, row in group.iterrows():
            gap_reset = previous_ts is not None and (row["bar_start"] - previous_ts).total_seconds() > 30 * 60
            if (not bool(active.loc[idx]) or gap_reset) and current_indices:
                episode_no += 1
                episodes.append(_make_episode(work.loc[current_indices], action_types, armed_types, trade_date, str(bond_code), episode_no, daily_health))
                current_indices = []
            if bool(active.loc[idx]):
                current_indices.append(idx)
            previous_ts = row["bar_start"]
        if current_indices:
            episode_no += 1
            episodes.append(_make_episode(work.loc[current_indices], action_types, armed_types, trade_date, str(bond_code), episode_no, daily_health))
    return pd.DataFrame(episodes)


def _make_episode(
    group: pd.DataFrame,
    action_types: set[str],
    armed_types: set[str],
    trade_date: str,
    bond_code: str,
    episode_no: int,
    daily_health: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    anchor = _episode_anchor_row(group, action_types, armed_types)
    actions = group[(group["alert_state"] == "ACTION_LONG") & group["signal_type"].isin(action_types)]
    armed = group[(group["alert_state"] == "ARMED_LONG") | group["signal_type"].isin(armed_types)]
    watch = group[group["signal_type"] == "WATCH_LONG"]
    exits = group[(group["alert_state"] == "EXIT_HINT") | (group.get("exit_type", pd.Series("HOLD", index=group.index)) != "HOLD")]
    has_action = not actions.empty
    has_armed = not armed.empty
    if has_action:
        primary = "ACTION_EPISODE"
    elif has_armed:
        primary = "ARMED_ONLY"
    else:
        primary = "WATCH_ONLY"
    health = (daily_health or {}).get(trade_date, {})
    first_seen = anchor.get("first_seen_time", anchor.get("calculated_at", pd.NaT))
    signal_time = anchor.get("bar_start", pd.NaT)
    live_or_backfill = "unknown"
    if pd.notna(first_seen) and pd.notna(signal_time):
        delay = (pd.Timestamp(first_seen) - pd.Timestamp(signal_time)).total_seconds() / 60.0
        live_or_backfill = "live_like" if -60 <= delay <= 10 else "backfill_or_delayed"
    return {
        "episode_id": f"{trade_date}_{bond_code}_{episode_no:03d}",
        "trade_date": trade_date,
        "bond_code": bond_code,
        "stock_code": str(anchor.get("stock_code", "")),
        "pool_state": str(anchor.get("pool_state", "")),
        "episode_start_time": _scalar(group["bar_start"].min()),
        "first_watch_time": _scalar(watch["bar_start"].min()) if not watch.empty else None,
        "first_armed_time": _scalar(armed["bar_start"].min()) if not armed.empty else None,
        "first_action_time": _scalar(actions["bar_start"].min()) if not actions.empty else None,
        "last_action_time": _scalar(actions["bar_start"].max()) if not actions.empty else None,
        "exit_time": _scalar(exits["bar_start"].min()) if not exits.empty else None,
        "expire_time": _scalar(actions["alert_expire_time"].dropna().max()) if has_action and "alert_expire_time" in actions.columns and actions["alert_expire_time"].notna().any() else None,
        "episode_primary_type": primary,
        "max_setup_score": _scalar(group["setup_score"].max()) if "setup_score" in group.columns else np.nan,
        "max_trigger_score": _scalar(group["trigger_score"].max()) if "trigger_score" in group.columns else np.nan,
        "min_residual_z": _scalar(group["residual_z"].min()) if "residual_z" in group.columns else np.nan,
        "max_stock_impulse_score": _scalar(group["stock_impulse_score"].max()) if "stock_impulse_score" in group.columns else np.nan,
        "max_relative_amount": _scalar(group["relative_amount"].max()) if "relative_amount" in group.columns else np.nan,
        "entry_candidate_price": _scalar(anchor.get("signal_price", np.nan)),
        "next_bar_open_after_action": _scalar(anchor.get("next_bar_open", np.nan)),
        "best_price_3bar": _scalar(anchor.get("best_price_3bar", np.nan)),
        "worst_price_3bar": _scalar(anchor.get("worst_price_3bar", np.nan)),
        "best_price_6bar": _scalar(anchor.get("best_price_6bar", np.nan)),
        "worst_price_6bar": _scalar(anchor.get("worst_price_6bar", np.nan)),
        "mfe_3bar": _scalar(anchor.get("mfe_3bar", np.nan)),
        "mae_3bar": _scalar(anchor.get("mae_3bar", np.nan)),
        "mfe_6bar": _scalar(anchor.get("mfe_6bar", np.nan)),
        "mae_6bar": _scalar(anchor.get("mae_6bar", np.nan)),
        "return_1bar": _scalar(anchor.get("forward_cb_return_1bar", np.nan)),
        "return_2bar": _scalar(anchor.get("forward_cb_return_2bar", np.nan)),
        "return_3bar": _scalar(anchor.get("forward_cb_return_3bar", np.nan)),
        "return_30m": _scalar(anchor.get("forward_cb_return_30m", np.nan)),
        "return_60m": _scalar(anchor.get("forward_cb_return_60m", np.nan)),
        "residual_return_30m": _scalar(anchor.get("forward_residual_return_30m", np.nan)),
        "residual_return_60m": _scalar(anchor.get("forward_residual_return_60m", np.nan)),
        "final_exit_type": str(exits["exit_type"].iloc[-1]) if not exits.empty and "exit_type" in exits.columns else ("EXPIRED" if has_action else "NO_ACTION"),
        "episode_outcome": "HIT_30M" if pd.notna(anchor.get("forward_cb_return_30m", np.nan)) and anchor.get("forward_cb_return_30m", np.nan) > 0 else "UNRESOLVED_OR_MISS",
        "live_or_backfill": live_or_backfill,
        "unreliable_live_day": bool(health.get("unreliable_live_day", False)),
    }


def episode_summary(episodes: pd.DataFrame) -> dict[str, object]:
    if episodes.empty:
        return {"episode_count": 0}
    out = {
        "episode_count": int(len(episodes)),
        "action_episode_count": int((episodes["episode_primary_type"] == "ACTION_EPISODE").sum()),
        "armed_only_episode_count": int((episodes["episode_primary_type"] == "ARMED_ONLY").sum()),
        "watch_only_episode_count": int((episodes["episode_primary_type"] == "WATCH_ONLY").sum()),
        "episode_hit_rate_30m": float((pd.to_numeric(episodes["return_30m"], errors="coerce").dropna() > 0).mean())
        if pd.to_numeric(episodes["return_30m"], errors="coerce").notna().any()
        else np.nan,
        "episode_return_30m_mean": float(pd.to_numeric(episodes["return_30m"], errors="coerce").mean()),
        "episode_mfe_6bar_mean": float(pd.to_numeric(episodes["mfe_6bar"], errors="coerce").mean()),
        "episode_mae_6bar_mean": float(pd.to_numeric(episodes["mae_6bar"], errors="coerce").mean()),
        "daily_episode_count": episodes.groupby("trade_date").size().astype(int).to_dict(),
        "bond_day_episode_count": episodes.groupby(["trade_date", "bond_code"]).size().astype(int).to_dict(),
        "by_episode_primary_type": episodes.groupby("episode_primary_type").size().astype(int).to_dict(),
    }
    return out


def evaluate_precomputed(
    factors: pd.DataFrame,
    alerts: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict[str, object],
    factor_version: str,
    cost_bps: float,
    pool_scope: str,
    allowed_pool_states: set[str],
    exclude_dates: set[str] | None = None,
    latency_threshold_minutes: float = 10.0,
    exclude_unreliable_live_days: bool = False,
    include_backfill: bool = False,
    include_shadow: bool = False,
    group_by: list[str] | None = None,
    source: str = "snapshot",
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    exclude_dates = exclude_dates or set()
    if factors.empty or panel.empty:
        return {
            "sample_count": 0,
            "message": "No factor or panel rows available",
            "factor_version": factor_version,
            "pool_scope": pool_scope,
            "source": source,
        }

    factors = factors.copy()
    alerts = alerts.copy() if alerts is not None else pd.DataFrame()
    panel = panel.copy()
    factors["bar_start"] = pd.to_datetime(factors["bar_start"], errors="coerce")
    if not alerts.empty:
        alerts["bar_start"] = pd.to_datetime(alerts["bar_start"], errors="coerce")
    if "pool_state" not in factors.columns:
        factors["pool_state"] = ACTIVE
    factors["pool_state"] = factors["pool_state"].apply(normalize_pool_state)
    if not alerts.empty:
        if "pool_state" not in alerts.columns:
            alerts["pool_state"] = ACTIVE
        alerts["pool_state"] = alerts["pool_state"].apply(normalize_pool_state)
    factors = factors[factors["pool_state"].isin(allowed_pool_states)].copy()
    if not alerts.empty:
        alerts = alerts[alerts["pool_state"].isin(allowed_pool_states)].copy()
    daily_health = infer_daily_health(factors, alerts, latency_threshold_minutes, exclude_dates) if source == "snapshot" else {}
    if exclude_dates:
        factors = factors[~factors["bar_start"].dt.strftime("%Y-%m-%d").isin(exclude_dates)].copy()
        if not alerts.empty:
            alerts = alerts[~alerts["bar_start"].dt.strftime("%Y-%m-%d").isin(exclude_dates)].copy()
    if exclude_unreliable_live_days and daily_health:
        unreliable_days = {day for day, health in daily_health.items() if bool(health.get("unreliable_live_day"))}
        factors = factors[~factors["bar_start"].dt.strftime("%Y-%m-%d").isin(unreliable_days)].copy()
        if not alerts.empty:
            alerts = alerts[~alerts["bar_start"].dt.strftime("%Y-%m-%d").isin(unreliable_days)].copy()
    if factors.empty:
        return {
            "sample_count": 0,
            "message": "No rows remain after factor_version, pool_scope, and reliability filters",
            "factor_version": factor_version,
            "pool_scope": pool_scope,
            "source": source,
            "daily_health": daily_health,
        }
    if not alerts.empty:
        alert_time_cols = [
            "bar_start",
            "bond_code",
            "signal_time",
            "first_seen_time",
            "alert_inserted_at",
            "effective_asof",
        ]
        existing_alert_time_cols = [col for col in alert_time_cols if col in alerts.columns]
        if {"bar_start", "bond_code"}.issubset(existing_alert_time_cols):
            factors = factors.merge(
                alerts[existing_alert_time_cols].drop_duplicates(subset=["bar_start", "bond_code"], keep="last"),
                on=["bar_start", "bond_code"],
                how="left",
            )
    panel["bar_start"] = pd.to_datetime(panel["bar_start"], errors="coerce")
    panel = add_forward_metrics(panel)

    merged = factors.merge(
        panel[
            [
                "bar_start",
                "bond_code",
                "cb_close",
                "cb_open",
                "signal_price",
                "next_bar_open",
                "first_executable_price_after_seen",
                "forward_cb_return_1bar",
                "forward_cb_return_2bar",
                "forward_cb_return_3bar",
                "forward_cb_return_30m",
                "forward_cb_return_60m",
                "forward_stock_return_30m",
                "forward_stock_return_60m",
                "forward_cb_market_return_30m",
                "forward_cb_market_return_60m",
                "mfe_30m",
                "mae_30m",
                "mfe_60m",
                "mae_60m",
                "mfe_3bar",
                "mae_3bar",
                "mfe_6bar",
                "mae_6bar",
                "best_price_3bar",
                "worst_price_3bar",
                "best_price_6bar",
                "worst_price_6bar",
                "best_price_30m",
                "worst_price_30m",
            ]
        ],
        on=["bar_start", "bond_code"],
        how="left",
    )
    merged = merged.sort_values(["bar_start", "bond_code"]).reset_index(drop=True)
    merged["forward_residual_return_30m"] = (
        merged["forward_cb_return_30m"]
        - merged["beta"].fillna(0.0) * merged["forward_stock_return_30m"].fillna(0.0)
        - merged["gamma"].fillna(0.0) * merged["forward_cb_market_return_30m"].fillna(0.0)
    )
    merged["forward_residual_return_60m"] = (
        merged["forward_cb_return_60m"]
        - merged["beta"].fillna(0.0) * merged["forward_stock_return_60m"].fillna(0.0)
        - merged["gamma"].fillna(0.0) * merged["forward_cb_market_return_60m"].fillna(0.0)
    )
    merged["slippage_vs_next_bar_open"] = merged["next_bar_open"] / merged["signal_price"] - 1.0
    merged["slippage_vs_first_seen_price"] = merged["first_executable_price_after_seen"] / merged["signal_price"] - 1.0
    merged["theoretical_signal_return_1bar"] = merged["forward_cb_return_1bar"]
    merged["live_visible_return_1bar_proxy"] = (
        merged.groupby("bond_code")["cb_close"].shift(-1) / merged["first_executable_price_after_seen"] - 1.0
    )
    merged = assign_buckets(merged)
    for time_col in ["signal_time", "first_seen_time", "alert_inserted_at", "effective_asof", "calculated_at", "alert_expire_time"]:
        if time_col in merged.columns:
            merged[time_col] = pd.to_datetime(merged[time_col], errors="coerce")
    if "alert_state" not in merged.columns:
        merged["alert_state"] = "INFO_ONLY"
    if "signal_type" not in merged.columns:
        merged["signal_type"] = merged.get("signal_state", "NONE")
    action_types = set(config["manual_alert_mode"]["action_signal_types"])  # type: ignore[index]
    armed_types = set(config.get("manual_alert_mode", {}).get("armed_signal_types", config.get("alert_controls", {}).get("armed_signal_types", [])))  # type: ignore[union-attr]
    action_rows = merged[(merged["alert_state"] == "ACTION_LONG") & merged["signal_type"].isin(action_types)].copy()
    cost = cost_bps / 10000.0
    action_rows["cost"] = cost
    watch_rows = merged[merged["signal_type"] == "WATCH_LONG"].copy()
    armed_rows = merged[(merged["alert_state"] == "ARMED_LONG") | merged["signal_type"].isin(armed_types)].copy()

    by_time_ic = merged.groupby("bar_start").apply(lambda g: spearman_ic(g["signal_score"], g["forward_cb_return_30m"]))
    by_bond_ic = merged.groupby("bond_code").apply(lambda g: spearman_ic(g["signal_score"], g["forward_cb_return_30m"]))
    by_signal_type = {
        signal_type: _metric_summary(action_rows, signal_type)
        for signal_type in sorted(action_types)
        if len(action_rows[action_rows["signal_type"] == signal_type])
    }
    triggers_per_bond_day = bond_day_action_counts(action_rows)
    daily_summary = {}
    for trade_date, group in merged.groupby(merged["bar_start"].dt.strftime("%Y-%m-%d")):
        day_actions = action_rows[action_rows["bar_start"].dt.strftime("%Y-%m-%d") == trade_date]
        daily_summary[trade_date] = {
            "sample_count": int(len(group)),
            "action_count": int(len(day_actions)),
            "armed_count": int(((group["alert_state"] == "ARMED_LONG") | group["signal_type"].isin(armed_types)).sum()),
            "watch_count": int((group["signal_type"] == "WATCH_LONG").sum()),
            "valid_30m_count": int(group["forward_cb_return_30m"].notna().sum()),
            "action_summary": _metric_summary(day_actions),
        }
    by_bond_summary = {}
    for bond_code, group in action_rows.groupby("bond_code"):
        by_bond_summary[str(bond_code)] = _metric_summary(group)
    episodes = build_episodes(merged, action_types, armed_types, daily_health)
    exit_rows = merged[(merged.get("exit_type", pd.Series("HOLD", index=merged.index)) != "HOLD") | (merged["alert_state"] == "EXIT_HINT")].copy()
    bucket_summaries = {
        "parity_bucket": per_group_summary(action_rows, "parity_bucket"),
        "bond_price_bucket": per_group_summary(action_rows, "bond_price_bucket"),
        "premium_bucket": per_group_summary(action_rows, "premium_bucket"),
        "liquidity_bucket": per_group_summary(action_rows, "liquidity_bucket"),
        "empirical_delta_bucket": per_group_summary(action_rows, "empirical_delta_bucket"),
        "time_bucket": per_group_summary(action_rows, "time_bucket"),
        "market_regime_intraday": per_group_summary(action_rows, "market_regime_intraday"),
    }

    return {
        "sample_count": int(len(merged)),
        "factor_version": factor_version,
        "pool_scope": pool_scope,
        "source": source,
        "elapsed_seconds": elapsed_seconds,
        "allowed_pool_states": sorted(allowed_pool_states),
        "include_backfill": bool(include_backfill),
        "include_shadow": bool(include_shadow),
        "requested_group_by": group_by or [],
        "factor_coverage_mean": float(merged["factor_coverage"].mean()),
        "pooled_panel_ic": spearman_ic(merged["signal_score"], merged["forward_cb_return_30m"]),
        "cross_section_ic_mean": float(by_time_ic.mean()) if not by_time_ic.empty else np.nan,
        "cross_section_icir": float(by_time_ic.mean() / (by_time_ic.std() + 1e-9)) if len(by_time_ic.dropna()) > 1 else np.nan,
        "time_series_ic_mean_by_bond": float(by_bond_ic.mean()) if not by_bond_ic.empty else np.nan,
        "action_signal_types": sorted(action_types),
        "armed_signal_types": sorted(armed_types),
        "action_sample_count": int(len(action_rows)),
        "armed_sample_count": int(len(armed_rows)),
        "action_by_signal_type": by_signal_type,
        "watch_long_excluded_from_trading_backtest": True,
        "armed_long_excluded_from_trading_backtest": True,
        "action_summary": _metric_summary(action_rows),
        "armed_summary": _metric_summary(armed_rows),
        "by_signal_type": by_signal_type,
        "watch_long_summary": _metric_summary(watch_rows),
        "watch_by_watch_signal_type": per_group_summary(watch_rows, "watch_signal_type"),
        "watch_to_action": watch_conversion_summary(merged, action_types, armed_types),
        "armed_to_action": armed_conversion_summary(merged, action_types, armed_types),
        "watch_risky_summary": _metric_summary(
            watch_rows[watch_rows["watch_signal_type"] == "WATCH_RISKY"] if "watch_signal_type" in watch_rows.columns else watch_rows.iloc[0:0]
        ),
        "exit_type_summary": per_group_summary(exit_rows, "exit_type"),
        "cooldown_suppressed_count": int(_bool_series(merged, "cooldown_suppressed").sum()),
        "after_no_new_action_time_filtered_count": int(_bool_series(merged, "action_blocked_by_time").sum()),
        "no_new_action_after_filtered_summary": _metric_summary(merged[_bool_series(merged, "action_blocked_by_time")]),
        "gate_stats": gate_stats(merged),
        "action_time_distribution": action_time_distribution(action_rows),
        "daily_action_count": daily_action_counts(action_rows),
        "triggers_per_bond_day": triggers_per_bond_day,
        "daily_summary": daily_summary,
        "by_bond_summary": by_bond_summary,
        "bucket_summaries": bucket_summaries,
        "episode_summary": episode_summary(episodes),
        "episode_rows": _records(episodes, list(episodes.columns), limit=50) if not episodes.empty else [],
        "action_explanations": make_action_explanation_rows(action_rows),
        "daily_health": daily_health,
        "research_experiment_grid": research_experiment_grid(),
        "action_max_drawdown_30m_net": max_drawdown(action_rows["forward_cb_return_30m"] - cost) if len(action_rows) else np.nan,
        "cost_bps": cost_bps,
        "by_bond_return": action_rows.groupby("bond_code")["forward_cb_return_30m"].mean().dropna().to_dict(),
        "by_month_return": action_rows.groupby(action_rows["bar_start"].dt.strftime("%Y-%m"))["forward_cb_return_30m"].mean().dropna().to_dict()
        if not action_rows.empty
        else {},
    }


def evaluate(
    start: str,
    end: str,
    cost_bps: float = 10.0,
    schema_path: str | None = None,
    config_path: str | None = None,
    exclude_dates: set[str] | None = None,
    latency_threshold_minutes: float = 10.0,
    factor_version_override: str | None = None,
    pool_scope: str = "ACTIVE_ONLY",
    exclude_unreliable_live_days: bool = False,
    include_backfill: bool = False,
    include_shadow: bool = False,
    group_by: list[str] | None = None,
) -> dict[str, object]:
    schema = load_schema(schema_path)
    config = load_config(config_path)
    factor_version = str(factor_version_override or config["factor_version"])
    exclude_dates = exclude_dates or set()
    allowed_pool_states = pool_scope_states(pool_scope)
    conn = get_connection()
    try:
        factors = query_to_dataframe(
            conn,
            f"""
            SELECT * FROM {schema['tables']['factor_snapshot']}
            WHERE bar_start >= {sql_string(start)}
              AND bar_start <= {sql_string(end)}
              AND factor_version = {sql_string(factor_version)}
            ORDER BY bond_code, bar_start
            """,
        )
        alerts = query_to_dataframe(
            conn,
            f"""
            SELECT * FROM {schema['tables'].get('alert_table', 'cb_intraday_alerts')}
            WHERE bar_start >= {sql_string(start)}
              AND bar_start <= {sql_string(end)}
              AND factor_version = {sql_string(factor_version)}
            ORDER BY bond_code, bar_start
            """,
        )
        panel = IntradayDataAdapter.from_files(schema_path).build_panel(conn, start, end, pool_states=allowed_pool_states)
    finally:
        conn.close()

    return evaluate_precomputed(
        factors=factors,
        alerts=alerts,
        panel=panel,
        config=config,
        factor_version=factor_version,
        cost_bps=cost_bps,
        pool_scope=pool_scope,
        allowed_pool_states=allowed_pool_states,
        exclude_dates=exclude_dates,
        latency_threshold_minutes=latency_threshold_minutes,
        exclude_unreliable_live_days=exclude_unreliable_live_days,
        include_backfill=include_backfill,
        include_shadow=include_shadow,
        group_by=group_by,
        source="snapshot",
    )

    if factors.empty or panel.empty:
        return {"sample_count": 0, "message": "No factor or panel rows available"}

    factors["bar_start"] = pd.to_datetime(factors["bar_start"], errors="coerce")
    if not alerts.empty:
        alerts["bar_start"] = pd.to_datetime(alerts["bar_start"], errors="coerce")
    if "pool_state" not in factors.columns:
        factors["pool_state"] = ACTIVE
    factors["pool_state"] = factors["pool_state"].apply(normalize_pool_state)
    if not alerts.empty:
        if "pool_state" not in alerts.columns:
            alerts["pool_state"] = ACTIVE
        alerts["pool_state"] = alerts["pool_state"].apply(normalize_pool_state)
    factors = factors[factors["pool_state"].isin(allowed_pool_states)].copy()
    if not alerts.empty:
        alerts = alerts[alerts["pool_state"].isin(allowed_pool_states)].copy()
    daily_health = infer_daily_health(factors, alerts, latency_threshold_minutes, exclude_dates)
    if exclude_dates:
        factors = factors[~factors["bar_start"].dt.strftime("%Y-%m-%d").isin(exclude_dates)].copy()
        if not alerts.empty:
            alerts = alerts[~alerts["bar_start"].dt.strftime("%Y-%m-%d").isin(exclude_dates)].copy()
    if exclude_unreliable_live_days and daily_health:
        unreliable_days = {day for day, health in daily_health.items() if bool(health.get("unreliable_live_day"))}
        factors = factors[~factors["bar_start"].dt.strftime("%Y-%m-%d").isin(unreliable_days)].copy()
        if not alerts.empty:
            alerts = alerts[~alerts["bar_start"].dt.strftime("%Y-%m-%d").isin(unreliable_days)].copy()
    if factors.empty:
        return {
            "sample_count": 0,
            "message": "No rows remain after factor_version, pool_scope, and reliability filters",
            "factor_version": factor_version,
            "pool_scope": pool_scope,
            "daily_health": daily_health,
        }
    if not alerts.empty:
        alert_time_cols = [
            "bar_start",
            "bond_code",
            "signal_time",
            "first_seen_time",
            "alert_inserted_at",
            "effective_asof",
        ]
        existing_alert_time_cols = [col for col in alert_time_cols if col in alerts.columns]
        if {"bar_start", "bond_code"}.issubset(existing_alert_time_cols):
            factors = factors.merge(
                alerts[existing_alert_time_cols].drop_duplicates(subset=["bar_start", "bond_code"], keep="last"),
                on=["bar_start", "bond_code"],
                how="left",
            )
    panel["bar_start"] = pd.to_datetime(panel["bar_start"], errors="coerce")
    panel = add_forward_metrics(panel)

    merged = factors.merge(
        panel[
            [
                "bar_start",
                "bond_code",
                "cb_close",
                "cb_open",
                "signal_price",
                "next_bar_open",
                "first_executable_price_after_seen",
                "forward_cb_return_1bar",
                "forward_cb_return_2bar",
                "forward_cb_return_3bar",
                "forward_cb_return_30m",
                "forward_cb_return_60m",
                "forward_stock_return_30m",
                "forward_stock_return_60m",
                "forward_cb_market_return_30m",
                "forward_cb_market_return_60m",
                "mfe_30m",
                "mae_30m",
                "mfe_60m",
                "mae_60m",
                "mfe_3bar",
                "mae_3bar",
                "mfe_6bar",
                "mae_6bar",
                "best_price_3bar",
                "worst_price_3bar",
                "best_price_6bar",
                "worst_price_6bar",
                "best_price_30m",
                "worst_price_30m",
            ]
        ],
        on=["bar_start", "bond_code"],
        how="left",
    )
    merged = merged.sort_values(["bar_start", "bond_code"]).reset_index(drop=True)
    merged["forward_residual_return_30m"] = (
        merged["forward_cb_return_30m"]
        - merged["beta"].fillna(0.0) * merged["forward_stock_return_30m"].fillna(0.0)
        - merged["gamma"].fillna(0.0) * merged["forward_cb_market_return_30m"].fillna(0.0)
    )
    merged["forward_residual_return_60m"] = (
        merged["forward_cb_return_60m"]
        - merged["beta"].fillna(0.0) * merged["forward_stock_return_60m"].fillna(0.0)
        - merged["gamma"].fillna(0.0) * merged["forward_cb_market_return_60m"].fillna(0.0)
    )
    merged["slippage_vs_next_bar_open"] = merged["next_bar_open"] / merged["signal_price"] - 1.0
    merged["slippage_vs_first_seen_price"] = merged["first_executable_price_after_seen"] / merged["signal_price"] - 1.0
    merged["theoretical_signal_return_1bar"] = merged["forward_cb_return_1bar"]
    merged["live_visible_return_1bar_proxy"] = (
        merged.groupby("bond_code")["cb_close"].shift(-1) / merged["first_executable_price_after_seen"] - 1.0
    )
    merged = assign_buckets(merged)
    for time_col in ["signal_time", "first_seen_time", "alert_inserted_at", "effective_asof", "calculated_at", "alert_expire_time"]:
        if time_col in merged.columns:
            merged[time_col] = pd.to_datetime(merged[time_col], errors="coerce")
    if "alert_state" not in merged.columns:
        merged["alert_state"] = "INFO_ONLY"
    if "signal_type" not in merged.columns:
        merged["signal_type"] = merged.get("signal_state", "NONE")
    action_types = set(config["manual_alert_mode"]["action_signal_types"])
    armed_types = set(config.get("manual_alert_mode", {}).get("armed_signal_types", config.get("alert_controls", {}).get("armed_signal_types", [])))
    action_rows = merged[(merged["alert_state"] == "ACTION_LONG") & merged["signal_type"].isin(action_types)].copy()
    cost = cost_bps / 10000.0
    action_rows["cost"] = cost
    watch_rows = merged[merged["signal_type"] == "WATCH_LONG"].copy()
    armed_rows = merged[(merged["alert_state"] == "ARMED_LONG") | merged["signal_type"].isin(armed_types)].copy()

    by_time_ic = merged.groupby("bar_start").apply(lambda g: spearman_ic(g["signal_score"], g["forward_cb_return_30m"]))
    by_bond_ic = merged.groupby("bond_code").apply(lambda g: spearman_ic(g["signal_score"], g["forward_cb_return_30m"]))
    by_signal_type = {
        signal_type: _metric_summary(action_rows, signal_type)
        for signal_type in sorted(action_types)
        if len(action_rows[action_rows["signal_type"] == signal_type])
    }
    triggers_per_bond_day = (
        bond_day_action_counts(action_rows)
    )
    daily_summary = {}
    for trade_date, group in merged.groupby(merged["bar_start"].dt.strftime("%Y-%m-%d")):
        day_actions = action_rows[action_rows["bar_start"].dt.strftime("%Y-%m-%d") == trade_date]
        daily_summary[trade_date] = {
            "sample_count": int(len(group)),
            "action_count": int(len(day_actions)),
            "watch_count": int((group["signal_type"] == "WATCH_LONG").sum()),
            "valid_30m_count": int(group["forward_cb_return_30m"].notna().sum()),
            "action_summary": _metric_summary(day_actions),
        }
    by_bond_summary = {}
    for bond_code, group in action_rows.groupby("bond_code"):
        by_bond_summary[str(bond_code)] = _metric_summary(group)
    episodes = build_episodes(merged, action_types, armed_types, daily_health)
    exit_rows = merged[(merged.get("exit_type", pd.Series("HOLD", index=merged.index)) != "HOLD") | (merged["alert_state"] == "EXIT_HINT")].copy()
    bucket_summaries = {
        "parity_bucket": per_group_summary(action_rows, "parity_bucket"),
        "bond_price_bucket": per_group_summary(action_rows, "bond_price_bucket"),
        "premium_bucket": per_group_summary(action_rows, "premium_bucket"),
        "liquidity_bucket": per_group_summary(action_rows, "liquidity_bucket"),
        "empirical_delta_bucket": per_group_summary(action_rows, "empirical_delta_bucket"),
        "time_bucket": per_group_summary(action_rows, "time_bucket"),
        "market_regime_intraday": per_group_summary(action_rows, "market_regime_intraday"),
    }

    return {
        "sample_count": int(len(merged)),
        "factor_version": factor_version,
        "pool_scope": pool_scope,
        "allowed_pool_states": sorted(allowed_pool_states),
        "include_backfill": bool(include_backfill),
        "include_shadow": bool(include_shadow),
        "requested_group_by": group_by or [],
        "factor_coverage_mean": float(merged["factor_coverage"].mean()),
        "pooled_panel_ic": spearman_ic(merged["signal_score"], merged["forward_cb_return_30m"]),
        "cross_section_ic_mean": float(by_time_ic.mean()) if not by_time_ic.empty else np.nan,
        "cross_section_icir": float(by_time_ic.mean() / (by_time_ic.std() + 1e-9)) if len(by_time_ic.dropna()) > 1 else np.nan,
        "time_series_ic_mean_by_bond": float(by_bond_ic.mean()) if not by_bond_ic.empty else np.nan,
        "action_signal_types": sorted(action_types),
        "armed_signal_types": sorted(armed_types),
        "action_sample_count": int(len(action_rows)),
        "armed_sample_count": int(len(armed_rows)),
        "action_by_signal_type": by_signal_type,
        "watch_long_excluded_from_trading_backtest": True,
        "armed_long_excluded_from_trading_backtest": True,
        "action_summary": _metric_summary(action_rows),
        "armed_summary": _metric_summary(armed_rows),
        "by_signal_type": by_signal_type,
        "watch_long_summary": _metric_summary(watch_rows),
        "watch_by_watch_signal_type": per_group_summary(watch_rows, "watch_signal_type"),
        "watch_to_action": watch_conversion_summary(merged, action_types, armed_types),
        "armed_to_action": armed_conversion_summary(merged, action_types, armed_types),
        "watch_risky_summary": _metric_summary(
            watch_rows[watch_rows["watch_signal_type"] == "WATCH_RISKY"] if "watch_signal_type" in watch_rows.columns else watch_rows.iloc[0:0]
        ),
        "exit_type_summary": per_group_summary(exit_rows, "exit_type"),
        "cooldown_suppressed_count": int(_bool_series(merged, "cooldown_suppressed").sum()),
        "after_no_new_action_time_filtered_count": int(_bool_series(merged, "action_blocked_by_time").sum()),
        "no_new_action_after_filtered_summary": _metric_summary(merged[_bool_series(merged, "action_blocked_by_time")]),
        "gate_stats": gate_stats(merged),
        "action_time_distribution": action_time_distribution(action_rows),
        "daily_action_count": daily_action_counts(action_rows),
        "triggers_per_bond_day": triggers_per_bond_day,
        "daily_summary": daily_summary,
        "by_bond_summary": by_bond_summary,
        "bucket_summaries": bucket_summaries,
        "episode_summary": episode_summary(episodes),
        "episode_rows": _records(episodes, list(episodes.columns), limit=50) if not episodes.empty else [],
        "action_explanations": make_action_explanation_rows(action_rows),
        "daily_health": daily_health,
        "research_experiment_grid": research_experiment_grid(),
        "action_max_drawdown_30m_net": max_drawdown(action_rows["forward_cb_return_30m"] - cost) if len(action_rows) else np.nan,
        "cost_bps": cost_bps,
        "by_bond_return": action_rows.groupby("bond_code")["forward_cb_return_30m"].mean().dropna().to_dict(),
        "by_month_return": action_rows.groupby(action_rows["bar_start"].dt.strftime("%Y-%m"))["forward_cb_return_30m"].mean().dropna().to_dict()
        if not action_rows.empty
        else {},
    }


def evaluate_live_compute(
    start: str,
    end: str,
    cost_bps: float = 10.0,
    schema_path: str | None = None,
    config_path: str | None = None,
    factor_version_override: str | None = None,
    pool_scope: str = "ACTIVE_ONLY",
    lookback_days: int = 70,
    include_backfill: bool = True,
    include_shadow: bool = False,
    group_by: list[str] | None = None,
) -> dict[str, object]:
    import time

    started = time.perf_counter()
    schema = load_schema(schema_path)
    config = load_config(config_path)
    factor_version = str(factor_version_override or config["factor_version"])
    allowed_pool_states = pool_scope_states(pool_scope)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    calc_start = start_ts - pd.Timedelta(days=int(lookback_days))
    adapter = IntradayDataAdapter.from_files(schema_path, bar_minutes=int(config["bar_minutes"]))
    conn = get_connection()
    try:
        panel_full = adapter.build_panel(conn, calc_start, end_ts, pool_states=allowed_pool_states)
        if panel_full.empty:
            return {"sample_count": 0, "message": "No live panel rows available", "source": "live-compute"}
        daily_amounts = adapter.load_daily_amounts(
            conn,
            sorted(panel_full["bond_code"].dropna().astype(str).unique()),
            end=end_ts,
            lookback_days=int(config["liquidity"]["adv20_days"]),
        )
    finally:
        conn.close()
    engine = FactorEngine(config, schema)
    factors_full = engine.compute_panel_factors(panel_full, daily_amounts=daily_amounts)
    factors = factors_full[(pd.to_datetime(factors_full["bar_start"]) >= start_ts) & (pd.to_datetime(factors_full["bar_start"]) <= end_ts)].copy()
    panel_eval = panel_full[(pd.to_datetime(panel_full["bar_start"]) >= start_ts) & (pd.to_datetime(panel_full["bar_start"]) <= end_ts)].copy()
    return evaluate_precomputed(
        factors=factors,
        alerts=pd.DataFrame(),
        panel=panel_eval,
        config=config,
        factor_version=factor_version,
        cost_bps=cost_bps,
        pool_scope=pool_scope,
        allowed_pool_states=allowed_pool_states,
        include_backfill=include_backfill,
        include_shadow=include_shadow,
        group_by=group_by,
        source="live-compute",
        elapsed_seconds=time.perf_counter() - started,
    )


def available_live_dates(start: str, end: str, schema_path: str | None = None) -> list[str]:
    schema = load_schema(schema_path)
    conn = get_connection()
    try:
        rows = query_to_dataframe(
            conn,
            f"""
            SELECT bar_start
            FROM {schema['tables']['live_5m']}
            WHERE bar_start >= {sql_string(start)}
              AND bar_start <= {sql_string(end)}
              AND asset_type = 'CB'
            ORDER BY bar_start
            """,
        )
    finally:
        conn.close()
    if rows.empty:
        return []
    return sorted(pd.to_datetime(rows["bar_start"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist())


def _evaluate_live_day_worker(params: dict[str, object]) -> tuple[str, dict[str, object]]:
    day = str(params["day"])
    start = str(params["start"])
    end = str(params["end"])
    start_ts = max(pd.Timestamp(start), pd.Timestamp(f"{day} 09:30:00"))
    end_ts = min(pd.Timestamp(end), pd.Timestamp(f"{day} 15:00:00"))
    result = evaluate_live_compute(
        start=start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        end=end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        cost_bps=float(params["cost_bps"]),
        schema_path=params.get("schema_path") or None,
        config_path=params.get("config_path") or None,
        factor_version_override=params.get("factor_version") or None,
        pool_scope=str(params["pool_scope"]),
        lookback_days=int(params["lookback_days"]),
        include_backfill=bool(params.get("include_backfill", True)),
        include_shadow=bool(params.get("include_shadow", False)),
        group_by=list(params.get("group_by") or []),
    )
    return day, result


def summarize_parallel_results(results_by_day: dict[str, dict[str, object]], elapsed_seconds: float, jobs: int) -> dict[str, object]:
    ordered = {day: results_by_day[day] for day in sorted(results_by_day)}
    episode_counts = [res.get("episode_summary", {}) for res in ordered.values()]
    return {
        "source": "live-compute-parallel-by-day",
        "jobs": jobs,
        "elapsed_seconds": elapsed_seconds,
        "day_count": len(ordered),
        "sample_count": int(sum(int(res.get("sample_count") or 0) for res in ordered.values())),
        "action_sample_count": int(sum(int(res.get("action_sample_count") or 0) for res in ordered.values())),
        "armed_sample_count": int(sum(int(res.get("armed_sample_count") or 0) for res in ordered.values())),
        "episode_count": int(sum(int(ep.get("episode_count") or 0) for ep in episode_counts)),
        "action_episode_count": int(sum(int(ep.get("action_episode_count") or 0) for ep in episode_counts)),
        "armed_only_episode_count": int(sum(int(ep.get("armed_only_episode_count") or 0) for ep in episode_counts)),
        "watch_only_episode_count": int(sum(int(ep.get("watch_only_episode_count") or 0) for ep in episode_counts)),
        "daily_results": ordered,
        "note": "Parallel by day uses independent lookback windows per day; use source=live-compute with jobs=1 when exact pooled IC across the whole range is required.",
    }


def evaluate_live_compute_parallel_by_day(
    start: str,
    end: str,
    cost_bps: float = 10.0,
    schema_path: str | None = None,
    config_path: str | None = None,
    factor_version_override: str | None = None,
    pool_scope: str = "ACTIVE_ONLY",
    lookback_days: int = 70,
    jobs: int | None = None,
    include_backfill: bool = True,
    include_shadow: bool = False,
    group_by: list[str] | None = None,
) -> dict[str, object]:
    import time

    started = time.perf_counter()
    days = available_live_dates(start, end, schema_path)
    if not days:
        return {"source": "live-compute-parallel-by-day", "sample_count": 0, "message": "No live dates available"}
    worker_count = max(1, int(jobs or min(len(days), os.cpu_count() or 1)))
    worker_count = min(worker_count, len(days))
    params = [
        {
            "day": day,
            "start": start,
            "end": end,
            "cost_bps": cost_bps,
            "schema_path": schema_path,
            "config_path": config_path,
            "factor_version": factor_version_override,
            "pool_scope": pool_scope,
            "lookback_days": lookback_days,
            "include_backfill": include_backfill,
            "include_shadow": include_shadow,
            "group_by": group_by or [],
        }
        for day in days
    ]
    results_by_day: dict[str, dict[str, object]] = {}
    executor = ProcessPoolExecutor(max_workers=worker_count)
    try:
        futures = [executor.submit(_evaluate_live_day_worker, item) for item in params]
        for future in as_completed(futures):
            day, result = future.result()
            results_by_day[day] = result
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return summarize_parallel_results(results_by_day, time.perf_counter() - started, worker_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight intraday factor evaluator.")
    parser.add_argument("--mode", choices=["evaluate"], default="evaluate")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--factor-version", default=None)
    parser.add_argument("--source", choices=["snapshot", "live-compute"], default="snapshot")
    parser.add_argument("--jobs", type=int, default=1, help="Use >1 with --source live-compute to evaluate independent days in parallel.")
    parser.add_argument("--live-lookback-days", type=int, default=70, help="Calendar-day lookback used for live-compute factor warmup.")
    parser.add_argument("--pool-scope", default="ACTIVE_ONLY", choices=["ACTIVE_ONLY", "ACTIVE_OBSERVE", "ACTIVE_OBSERVE_SHADOW"])
    parser.add_argument("--exclude-unreliable-live-days", action="store_true")
    parser.add_argument("--include-backfill", action="store_true", help="Accepted for report labeling; theoretical returns remain separated from live/manual fields.")
    parser.add_argument("--include-shadow", action="store_true", help="Use with --pool-scope ACTIVE_OBSERVE_SHADOW for shadow research rows.")
    parser.add_argument("--group-by", action="append", default=[])
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--exclude-date", action="append", default=[])
    parser.add_argument("--latency-threshold-minutes", type=float, default=10.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    start, end = normalize_time_window(args.start, args.end, args.start_date, args.end_date)
    if args.source == "live-compute" and args.jobs > 1:
        result = evaluate_live_compute_parallel_by_day(
            start,
            end,
            args.cost_bps,
            args.schema,
            args.config,
            factor_version_override=args.factor_version,
            pool_scope=args.pool_scope,
            lookback_days=args.live_lookback_days,
            jobs=args.jobs,
            include_backfill=args.include_backfill,
            include_shadow=args.include_shadow,
            group_by=args.group_by,
        )
    elif args.source == "live-compute":
        result = evaluate_live_compute(
            start,
            end,
            args.cost_bps,
            args.schema,
            args.config,
            factor_version_override=args.factor_version,
            pool_scope=args.pool_scope,
            lookback_days=args.live_lookback_days,
            include_backfill=args.include_backfill,
            include_shadow=args.include_shadow,
            group_by=args.group_by,
        )
    else:
        result = evaluate(
            start,
            end,
            args.cost_bps,
            args.schema,
            args.config,
            exclude_dates=set(args.exclude_date),
            latency_threshold_minutes=args.latency_threshold_minutes,
            factor_version_override=args.factor_version,
            pool_scope=args.pool_scope,
            exclude_unreliable_live_days=args.exclude_unreliable_live_days,
            include_backfill=args.include_backfill,
            include_shadow=args.include_shadow,
            group_by=args.group_by,
        )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
