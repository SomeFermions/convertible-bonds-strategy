from __future__ import annotations

import argparse
import json
import logging
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_intraday_factors import assign_buckets, normalize_time_window, pool_scope_states
from intraday_factors.config import load_config, load_schema
from intraday_factors.data_source import IntradayDataAdapter, get_connection
from intraday_factors.engine import FactorEngine
from intraday_factors.pool_state import ACTIVE, ACTIVE_MANUAL, normalize_pool_state


LOGGER = logging.getLogger(__name__)


FEATURE_VERSION = "v0.4_feature_001"
LABEL_VERSION = "v0.4_label_001"
MODEL_VERSION_PREFIX = "ml_shadow_v04"


NUMERIC_FEATURES = [
    "is_actionable_bar",
    "minutes_to_close",
    "no_new_action_after_suppressed",
    "parity",
    "premium_raw",
    "premium_slot_z",
    "empirical_delta_proxy",
    "elasticity_proxy",
    "residual_z",
    "residual_ewm",
    "factor_residual_mr",
    "residual_z_change_1bar",
    "residual_z_change_2bar",
    "residual_z_slope_3bar",
    "residual_repair_started",
    "stock_mom_5m_z",
    "stock_mom_10m_z",
    "stock_mom_15m_z",
    "stock_mom_30m_z",
    "stock_mom_60m_z",
    "stock_impulse_score",
    "stock_amount_burst_z",
    "cb_mom_5m_z",
    "cb_mom_10m_z",
    "cb_mom_15m_z",
    "cb_mom_30m_z",
    "premium_change_5m_z",
    "premium_change_10m_z",
    "premium_change_30m_z",
    "premium_expansion_risk_flag",
    "cb_vwap_dev",
    "stock_vwap_dev",
    "vwap_gap",
    "relative_amount",
    "cb_flow_z",
    "stock_flow_z",
    "flow_confirmation",
    "flow_lead",
    "residual_vol_ratio_z",
    "high_low_range_proxy_z",
    "volatility_gate_pass",
    "noise_penalty",
    "active_bar_ratio",
    "cb_market_mom_30m",
    "cb_market_breadth_30m",
    "cb_market_breadth_60m",
    "cb_market_amount_burst",
    "universe_count",
    "first_seen_delay_seconds",
    "live_pipeline_delay_seconds",
    "setup_score",
    "trigger_score",
    "signal_score",
]


CATEGORICAL_FEATURES = [
    "signal_type",
    "watch_subtype",
    "episode_primary_type",
    "pool_state",
    "event_type",
    "time_bucket",
    "parity_bucket",
    "bond_price_bucket",
    "premium_bucket",
    "empirical_delta_bucket",
    "liquidity_bucket",
    "amount_source",
    "market_regime_intraday",
    "data_mode",
    "unreliable_live_day",
]


LABEL_COLUMNS = [
    "return_1bar",
    "return_2bar",
    "return_3bar",
    "return_30m",
    "return_60m",
    "return_to_eod",
    "residual_return_30m",
    "residual_return_60m",
    "residual_return_to_eod",
    "mfe_3bar",
    "mae_3bar",
    "mfe_6bar",
    "mae_6bar",
    "mfe_to_eod",
    "mae_to_eod",
    "label_action_good_30m",
    "label_action_good_to_eod",
    "label_action_bad_3bar",
    "label_triple_barrier_6bar",
    "label_fake_breakout",
    "label_lag_repair_success",
    "label_lag_repair_fail",
    "label_armed_to_action_1bar",
    "label_armed_to_action_3bar",
    "label_armed_false_alarm",
    "label_unavailable_due_to_eod",
]


@dataclass
class DatasetBuildResult:
    dataset: pd.DataFrame
    factors: pd.DataFrame
    panel: pd.DataFrame
    output_path: Path | None = None
    parquet_path: Path | None = None


def _ml_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("ml_shadow_v04", {}))


def _parse_hhmmss(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in str(value).split(":")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _same_day_time(trade_date: str, hhmmss: str) -> pd.Timestamp:
    h, m, s = _parse_hhmmss(hhmmss)
    return pd.Timestamp(trade_date) + pd.Timedelta(hours=h, minutes=m, seconds=s)


def _time_after(ts: pd.Timestamp, hhmmss: str) -> bool:
    h, m, s = _parse_hhmmss(hhmmss)
    seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
    cutoff = h * 3600 + m * 60 + s
    return seconds > cutoff


def _safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if pd.isna(value):
        return None
    return str(value)


def _output_dir(config: dict[str, Any]) -> Path:
    out = Path(str(_ml_cfg(config).get("output_dir", "outputs/ml_shadow")))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _table_name(schema: dict[str, Any], key: str, fallback: str) -> str:
    return str(schema.get("tables", {}).get(key, fallback))


def write_dataframe(df: pd.DataFrame, output_dir: Path, logical_name: str) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{logical_name}.csv"
    df.to_csv(csv_path, index=False)
    parquet_path = output_dir / f"{logical_name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        LOGGER.info("Parquet export skipped for %s: %s", logical_name, exc)
        parquet_path = None
    return csv_path, parquet_path


def load_live_factors_and_panel(
    start: str,
    end: str,
    config: dict[str, Any],
    schema: dict[str, Any],
    pool_scope: str = "ACTIVE_ONLY",
    lookback_days: int = 70,
    schema_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    calc_start = start_ts - pd.Timedelta(days=int(lookback_days))
    allowed_pool_states = pool_scope_states(pool_scope)
    adapter = IntradayDataAdapter.from_files(schema_path, bar_minutes=int(config["bar_minutes"]))
    conn = get_connection()
    try:
        panel_full = adapter.build_panel(conn, calc_start, end_ts, pool_states=allowed_pool_states)
        if panel_full.empty:
            return pd.DataFrame(), pd.DataFrame()
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
    mask = (pd.to_datetime(factors_full["bar_start"]) >= start_ts) & (pd.to_datetime(factors_full["bar_start"]) <= end_ts)
    panel_mask = (pd.to_datetime(panel_full["bar_start"]) >= start_ts) & (pd.to_datetime(panel_full["bar_start"]) <= end_ts)
    return factors_full.loc[mask].copy(), panel_full.loc[panel_mask].copy()


def _prepare_merged(factors: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if factors.empty:
        return pd.DataFrame()
    out = factors.copy()
    out["bar_start"] = pd.to_datetime(out["bar_start"], errors="coerce")
    if not panel.empty:
        panel_cols = [
            "bar_start",
            "bond_code",
            "cb_open",
            "cb_high",
            "cb_low",
            "cb_close",
            "stock_open",
            "stock_high",
            "stock_low",
            "stock_close",
            "amount_used",
            "stock_amount_used",
        ]
        panel_work = panel[[col for col in panel_cols if col in panel.columns]].copy()
        panel_work["bar_start"] = pd.to_datetime(panel_work["bar_start"], errors="coerce")
        out = out.merge(panel_work.drop_duplicates(["bar_start", "bond_code"]), on=["bar_start", "bond_code"], how="left")
    if "cb_close" not in out.columns:
        out["cb_close"] = np.nan
    if "cb_open" not in out.columns:
        out["cb_open"] = np.nan
    out["trade_date"] = out["bar_start"].dt.strftime("%Y-%m-%d")
    if "pool_state" not in out.columns:
        out["pool_state"] = ACTIVE
    out["pool_state"] = out["pool_state"].apply(normalize_pool_state)
    out = assign_buckets(out)
    out["watch_subtype"] = out.get("watch_signal_type", "").fillna("").astype(str)
    out["premium_expansion_risk_flag"] = (
        (pd.to_numeric(out.get("premium_change_10m_z", np.nan), errors="coerce") > 1.0)
        | (pd.to_numeric(out.get("premium_change_30m_z", np.nan), errors="coerce") > 1.0)
    )
    out["universe_count"] = out.groupby("bar_start")["bond_code"].transform("count")
    return out.sort_values(["bond_code", "bar_start"]).reset_index(drop=True)


def _event_type(row: pd.Series, action_types: set[str], armed_types: set[str]) -> str | None:
    signal_type = str(row.get("signal_type", ""))
    alert_state = str(row.get("alert_state", ""))
    if alert_state == "ACTION_LONG" and signal_type in action_types:
        return "ACTION"
    if alert_state == "ARMED_LONG" or signal_type in armed_types:
        return "ARMED"
    if alert_state == "EXIT_HINT" or str(row.get("exit_type", "HOLD")) != "HOLD":
        return "EXIT"
    if signal_type == "WATCH_LONG":
        return "WATCH"
    return None


def _episode_ids(merged: pd.DataFrame, action_types: set[str], armed_types: set[str], cfg: dict[str, Any]) -> pd.DataFrame:
    candidates = merged.copy()
    candidates["event_type_raw"] = candidates.apply(lambda row: _event_type(row, action_types, armed_types), axis=1)
    candidates = candidates[candidates["event_type_raw"].notna()].copy()
    if candidates.empty:
        return candidates
    gap = pd.Timedelta(minutes=float(cfg.get("dedupe_episode_gap_minutes", 30)))
    rows = []
    for (trade_date, bond_code), group in candidates.groupby(["trade_date", "bond_code"], sort=False):
        ep_no = 0
        last_ts: pd.Timestamp | None = None
        first_by_event: dict[tuple[int, str], bool] = {}
        for idx, row in group.sort_values("bar_start").iterrows():
            ts = pd.Timestamp(row["bar_start"])
            if last_ts is None or ts - last_ts > gap:
                ep_no += 1
                first_by_event = {}
            event_type = str(row["event_type_raw"])
            key = (ep_no, event_type)
            if key not in first_by_event:
                item = row.copy()
                item["episode_id"] = f"{trade_date}_{bond_code}_{ep_no:03d}"
                item["episode_primary_type"] = "ACTION_EPISODE" if event_type == "ACTION" else ("ARMED_ONLY" if event_type == "ARMED" else f"{event_type}_ONLY")
                rows.append(item)
                first_by_event[key] = True
            last_ts = ts
    return pd.DataFrame(rows)


def _future_window(group: pd.DataFrame, pos: int, max_bars: int, force_flatten_ts: pd.Timestamp) -> pd.DataFrame:
    future = group.iloc[pos + 1 : pos + max_bars + 1].copy()
    return future[(future["bar_start"] <= force_flatten_ts) & (future["trade_date"] == group.iloc[pos]["trade_date"])]


def _return_at(group: pd.DataFrame, pos: int, bars: int, entry: float, force_flatten_ts: pd.Timestamp) -> tuple[float, bool]:
    target = pos + bars
    if target >= len(group) or not np.isfinite(entry) or entry <= 0:
        return np.nan, True
    row = group.iloc[target]
    if row["trade_date"] != group.iloc[pos]["trade_date"] or pd.Timestamp(row["bar_start"]) > force_flatten_ts:
        return np.nan, True
    close = _safe_float(row.get("cb_close"))
    return (close / entry - 1.0) if np.isfinite(close) and close > 0 else np.nan, False


def _stock_return_at(group: pd.DataFrame, pos: int, bars: int) -> float:
    target = pos + bars
    if target >= len(group):
        return np.nan
    base = _safe_float(group.iloc[pos].get("stock_close"))
    close = _safe_float(group.iloc[target].get("stock_close"))
    if not np.isfinite(base) or not np.isfinite(close) or base <= 0:
        return np.nan
    return close / base - 1.0


def _mfe_mae(group: pd.DataFrame, pos: int, entry: float, bars: int, force_flatten_ts: pd.Timestamp) -> tuple[float, float]:
    future = _future_window(group, pos, bars, force_flatten_ts)
    if future.empty or not np.isfinite(entry) or entry <= 0:
        return np.nan, np.nan
    returns = pd.to_numeric(future["cb_close"], errors="coerce") / entry - 1.0
    return float(returns.max()), float(returns.min())


def _to_eod_return(group: pd.DataFrame, pos: int, entry: float, force_flatten_ts: pd.Timestamp) -> float:
    future = group.iloc[pos + 1 :].copy()
    future = future[(future["trade_date"] == group.iloc[pos]["trade_date"]) & (future["bar_start"] <= force_flatten_ts)]
    if future.empty or not np.isfinite(entry) or entry <= 0:
        return np.nan
    close = _safe_float(future.iloc[-1].get("cb_close"))
    return close / entry - 1.0 if np.isfinite(close) and close > 0 else np.nan


def _triple_barrier(group: pd.DataFrame, pos: int, entry: float, cfg: dict[str, Any], signal_type: str, cost: float, force_flatten_ts: pd.Timestamp) -> float:
    barrier_cfg = cfg.get("triple_barrier", {})
    if signal_type == "MOMENTUM_BREAKOUT_LONG":
        params = barrier_cfg.get("breakout", barrier_cfg.get("default", {}))
    elif signal_type == "LAG_REPAIR_LONG":
        params = barrier_cfg.get("lag_repair", barrier_cfg.get("default", {}))
    else:
        params = barrier_cfg.get("default", {})
    profit = float(params.get("profit_pct", 0.003))
    stop = float(params.get("stop_pct", -0.002))
    vertical = int(params.get("vertical_bars", 6))
    future = _future_window(group, pos, vertical, force_flatten_ts)
    if future.empty or not np.isfinite(entry) or entry <= 0:
        return np.nan
    last_ret = np.nan
    for _, row in future.iterrows():
        ret = _safe_float(row.get("cb_close")) / entry - 1.0
        if not np.isfinite(ret):
            continue
        last_ret = ret
        if ret >= profit:
            return 1.0
        if ret <= stop:
            return 0.0
    if not np.isfinite(last_ret):
        return np.nan
    return 1.0 if last_ret - cost > 0 else 0.0


def add_t0_labels(events: pd.DataFrame, merged: pd.DataFrame, config: dict[str, Any], cost_bps: float | None = None) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    cfg = _ml_cfg(config)
    cost = float(cost_bps if cost_bps is not None else cfg.get("default_cost_bps", 10)) / 10000.0
    force_time = str(cfg.get("force_flatten_time", "14:55:00"))
    no_new_action_after = str(cfg.get("no_new_action_after", "14:30:00"))
    stop_threshold = float(cfg.get("stop_threshold", -0.002))

    work = events.copy()
    merged_sorted = merged.sort_values(["bond_code", "trade_date", "bar_start"]).copy()
    label_rows: list[dict[str, Any]] = []
    action_types = set(cfg.get("official_action_signal_types", []))

    for (trade_date, bond_code), group in merged_sorted.groupby(["trade_date", "bond_code"], sort=False):
        group = group.reset_index(drop=False).rename(columns={"index": "_merged_index"})
        force_flatten_ts = _same_day_time(str(trade_date), force_time)
        group_events = work[(work["trade_date"] == trade_date) & (work["bond_code"].astype(str) == str(bond_code))]
        action_times = group[
            (group.get("alert_state", "") == "ACTION_LONG") & group.get("signal_type", "").isin(action_types)
        ]["bar_start"].tolist()
        for event_idx, event in group_events.iterrows():
            matches = group[group["bar_start"] == event["bar_start"]]
            if matches.empty:
                continue
            pos = int(matches.index[0])
            row = group.iloc[pos]
            next_pos = pos + 1
            entry = np.nan
            if next_pos < len(group) and group.iloc[next_pos]["trade_date"] == trade_date:
                next_ts = pd.Timestamp(group.iloc[next_pos]["bar_start"])
                if next_ts <= force_flatten_ts:
                    entry = _safe_float(group.iloc[next_pos].get("cb_open"))
            labels: dict[str, Any] = {"_event_index": event_idx}
            unavailable_due_to_eod = False
            for bars, name in [(1, "1bar"), (2, "2bar"), (3, "3bar"), (6, "30m"), (12, "60m")]:
                ret, unavailable = _return_at(group, pos, bars, entry, force_flatten_ts)
                labels[f"return_{name}"] = ret
                unavailable_due_to_eod = unavailable_due_to_eod or unavailable
                if name in {"30m", "60m"}:
                    stock_ret = _stock_return_at(group, pos, bars)
                    beta = _safe_float(row.get("beta"))
                    gamma = _safe_float(row.get("gamma"))
                    market_ret = _safe_float(row.get(f"forward_cb_market_return_{name}", np.nan))
                    labels[f"residual_return_{name}"] = ret - np.nan_to_num(beta) * np.nan_to_num(stock_ret) - np.nan_to_num(gamma) * np.nan_to_num(market_ret) if np.isfinite(ret) else np.nan
            for bars, name in [(3, "3bar"), (6, "6bar")]:
                mfe, mae = _mfe_mae(group, pos, entry, bars, force_flatten_ts)
                labels[f"mfe_{name}"] = mfe
                labels[f"mae_{name}"] = mae
            labels["return_to_eod"] = _to_eod_return(group, pos, entry, force_flatten_ts)
            mfe_eod, mae_eod = _mfe_mae(group, pos, entry, 10_000, force_flatten_ts)
            labels["mfe_to_eod"] = mfe_eod
            labels["mae_to_eod"] = mae_eod
            stock_to_eod = np.nan
            future_to_eod = group.iloc[pos + 1 :].copy()
            future_to_eod = future_to_eod[(future_to_eod["trade_date"] == trade_date) & (future_to_eod["bar_start"] <= force_flatten_ts)]
            if not future_to_eod.empty:
                base_stock = _safe_float(row.get("stock_close"))
                last_stock = _safe_float(future_to_eod.iloc[-1].get("stock_close"))
                if np.isfinite(base_stock) and base_stock > 0 and np.isfinite(last_stock):
                    stock_to_eod = last_stock / base_stock - 1.0
            labels["residual_return_to_eod"] = (
                labels["return_to_eod"] - np.nan_to_num(_safe_float(row.get("beta"))) * np.nan_to_num(stock_to_eod)
                if np.isfinite(labels["return_to_eod"])
                else np.nan
            )
            labels["label_unavailable_due_to_eod"] = bool(unavailable_due_to_eod)
            labels["label_action_good_30m"] = (
                float(labels["return_30m"] - cost > 0 and labels["mae_6bar"] > stop_threshold)
                if np.isfinite(labels["return_30m"]) and np.isfinite(labels["mae_6bar"])
                else np.nan
            )
            labels["label_action_good_to_eod"] = (
                float(labels["return_to_eod"] - cost > 0 and labels["mae_to_eod"] > stop_threshold)
                if np.isfinite(labels["return_to_eod"]) and np.isfinite(labels["mae_to_eod"])
                else np.nan
            )
            labels["label_action_bad_3bar"] = (
                float(labels["return_3bar"] - cost < 0 or labels["mae_3bar"] <= stop_threshold)
                if np.isfinite(labels["return_3bar"]) or np.isfinite(labels["mae_3bar"])
                else np.nan
            )
            labels["label_triple_barrier_6bar"] = _triple_barrier(group, pos, entry, cfg, str(event.get("signal_type", "")), cost, force_flatten_ts)
            labels["label_fake_breakout"] = (
                float((labels["return_3bar"] - cost < 0) or (labels["mae_3bar"] <= -0.002))
                if str(event.get("signal_type")) == "MOMENTUM_BREAKOUT_LONG" and (np.isfinite(labels["return_3bar"]) or np.isfinite(labels["mae_3bar"]))
                else np.nan
            )
            future6 = _future_window(group, pos, 6, force_flatten_ts)
            residual_now = _safe_float(row.get("residual_z"))
            future_max_residual = pd.to_numeric(future6.get("residual_z", pd.Series(dtype=float)), errors="coerce").max() if not future6.empty else np.nan
            repaired = np.isfinite(residual_now) and np.isfinite(future_max_residual) and future_max_residual > residual_now
            labels["label_lag_repair_success"] = (
                float(repaired and labels["return_30m"] - cost > 0 and labels["mae_6bar"] > stop_threshold)
                if str(event.get("signal_type")) == "LAG_REPAIR_LONG" and np.isfinite(labels["return_30m"]) and np.isfinite(labels["mae_6bar"])
                else np.nan
            )
            labels["label_lag_repair_fail"] = (
                float(1.0 - labels["label_lag_repair_success"])
                if np.isfinite(labels["label_lag_repair_success"])
                else np.nan
            )
            event_time = pd.Timestamp(event["bar_start"])
            future_action_times = [ts for ts in action_times if event_time < pd.Timestamp(ts) <= force_flatten_ts]
            labels["label_armed_to_action_1bar"] = (
                float(any(pd.Timestamp(ts) <= event_time + pd.Timedelta(minutes=5) for ts in future_action_times))
                if str(event.get("event_type")) == "ARMED"
                else np.nan
            )
            labels["label_armed_to_action_3bar"] = (
                float(any(pd.Timestamp(ts) <= event_time + pd.Timedelta(minutes=15) for ts in future_action_times))
                if str(event.get("event_type")) == "ARMED"
                else np.nan
            )
            labels["label_armed_false_alarm"] = (
                float(1.0 - labels["label_armed_to_action_3bar"])
                if np.isfinite(labels["label_armed_to_action_3bar"])
                else np.nan
            )
            labels["entry_price_next_bar_open"] = entry
            labels["next_bar_open"] = entry
            labels["signal_price"] = _safe_float(row.get("cb_close"))
            labels["first_executable_price_after_seen"] = entry
            labels["slippage_vs_next_bar_open"] = entry / labels["signal_price"] - 1.0 if np.isfinite(entry) and np.isfinite(labels["signal_price"]) and labels["signal_price"] > 0 else np.nan
            labels["slippage_vs_first_seen_price"] = labels["slippage_vs_next_bar_open"]
            labels["late_signal_shadow_only"] = bool(_time_after(event_time, no_new_action_after))
            label_rows.append(labels)

    labels_df = pd.DataFrame(label_rows).set_index("_event_index") if label_rows else pd.DataFrame()
    out = work.copy()
    for col in labels_df.columns:
        out.loc[labels_df.index, col] = labels_df[col]
    return out


def build_ml_event_dataset_from_frames(
    factors: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict[str, Any],
    factor_version: str = "v0.3",
    config_version: str | None = None,
    pool_scope: str = "ACTIVE_ONLY",
    event_scope: str = "ARMED,ACTION",
    data_mode: str = "manual_backfill",
    unreliable_live_days: set[str] | None = None,
    cost_bps: float | None = None,
) -> pd.DataFrame:
    cfg = _ml_cfg(config)
    action_types = set(cfg.get("official_action_signal_types", config.get("manual_alert_mode", {}).get("action_signal_types", [])))
    armed_types = set(cfg.get("official_armed_signal_types", config.get("manual_alert_mode", {}).get("armed_signal_types", [])))
    merged = _prepare_merged(factors, panel)
    if merged.empty:
        return pd.DataFrame()

    events = _episode_ids(merged, action_types, armed_types, cfg)
    if events.empty:
        return pd.DataFrame()
    requested_scopes = {item.strip().upper() for item in event_scope.split(",") if item.strip()}
    events = events[events["event_type_raw"].isin(requested_scopes)].copy()
    if events.empty:
        return pd.DataFrame()

    events = events.rename(columns={"event_type_raw": "event_type", "bar_start": "signal_time"})
    events["bar_start"] = pd.to_datetime(events["signal_time"], errors="coerce")
    events["signal_time"] = pd.to_datetime(events["signal_time"], errors="coerce")
    events["trade_date"] = events["signal_time"].dt.strftime("%Y-%m-%d")
    events["event_id"] = [
        f"{row.trade_date}_{row.bond_code}_{row.event_type}_{pd.Timestamp(row.signal_time).strftime('%H%M%S')}_{i:04d}"
        for i, row in enumerate(events.itertuples(index=False))
    ]
    events["first_seen_time"] = pd.NaT
    for col in ["first_seen_time", "alert_inserted_at", "effective_asof", "calculated_at"]:
        if col not in events.columns:
            events[col] = pd.NaT
        events[col] = pd.to_datetime(events[col], errors="coerce")
    events["first_seen_time"] = events["first_seen_time"].combine_first(events["alert_inserted_at"]).combine_first(events["calculated_at"]).combine_first(events["signal_time"])
    events["alert_inserted_at"] = events["alert_inserted_at"].combine_first(events["first_seen_time"])
    events["effective_asof"] = events["effective_asof"].combine_first(events["signal_time"])
    events["data_mode"] = data_mode
    events["factor_version"] = factor_version
    events["config_version"] = config_version or str(cfg.get("config_version", "intraday_factors_v0.4"))
    events["feature_version"] = str(cfg.get("feature_version", FEATURE_VERSION))
    events["label_version"] = str(cfg.get("label_version", LABEL_VERSION))
    unreliable_live_days = unreliable_live_days or set()
    events["unreliable_live_day"] = events["trade_date"].isin(unreliable_live_days)
    no_new_after = str(cfg.get("no_new_action_after", "14:30:00"))
    force_time = str(cfg.get("force_flatten_time", "14:55:00"))
    events["minutes_to_close"] = [
        (_same_day_time(str(row.trade_date), force_time) - pd.Timestamp(row.signal_time)).total_seconds() / 60.0
        for row in events.itertuples(index=False)
    ]
    events["force_flatten_time"] = [str(_same_day_time(str(day), force_time)) for day in events["trade_date"]]
    events["eod_cutoff_time"] = events["force_flatten_time"]
    events["no_new_action_after_suppressed"] = events["signal_time"].apply(lambda ts: _time_after(pd.Timestamp(ts), no_new_after))
    if "is_actionable_bar" not in events.columns:
        events["is_actionable_bar"] = True
    events["is_actionable_bar"] = events["is_actionable_bar"].fillna(False).astype(bool) & ~events["no_new_action_after_suppressed"]
    events.loc[events["signal_time"].dt.strftime("%H:%M:%S").eq("15:00:00"), "is_actionable_bar"] = False
    events["first_seen_delay_seconds"] = (events["first_seen_time"] - events["signal_time"]).dt.total_seconds()
    events["live_pipeline_delay_seconds"] = (events["alert_inserted_at"] - events["signal_time"]).dt.total_seconds()
    events["pool_state"] = events["pool_state"].apply(normalize_pool_state)
    events["premium_expansion_risk_flag"] = events["premium_expansion_risk_flag"].fillna(False).astype(bool)
    events["volatility_gate_pass"] = events.get("volatility_gate_pass", True)
    events["residual_repair_started"] = events.get("residual_repair_started", False)
    events = add_t0_labels(events, merged, config, cost_bps=cost_bps)
    events["eligible_for_action_training"] = (
        (events["event_type"] == "ACTION")
        & events["signal_type"].isin(action_types)
        & events["pool_state"].isin({ACTIVE, ACTIVE_MANUAL})
        & events["is_actionable_bar"].fillna(False).astype(bool)
        & ~events["no_new_action_after_suppressed"].fillna(False).astype(bool)
    )
    events["eligible_for_armed_training"] = (events["event_type"] == "ARMED") & events["signal_type"].isin(armed_types)
    events["force_flatten_marker"] = "FORCE_FLATTEN"
    return events.sort_values(["signal_time", "bond_code", "event_type"]).reset_index(drop=True)


def build_ml_event_dataset(
    start: str,
    end: str,
    config: dict[str, Any],
    schema: dict[str, Any],
    factor_version: str = "v0.3",
    config_version: str | None = None,
    pool_scope: str = "ACTIVE_ONLY",
    event_scope: str = "ARMED,ACTION",
    data_mode: str = "manual_backfill",
    live_lookback_days: int = 70,
    cost_bps: float | None = None,
    schema_path: str | None = None,
) -> DatasetBuildResult:
    factors, panel = load_live_factors_and_panel(start, end, config, schema, pool_scope, live_lookback_days, schema_path=schema_path)
    dataset = build_ml_event_dataset_from_frames(
        factors,
        panel,
        config,
        factor_version=factor_version,
        config_version=config_version,
        pool_scope=pool_scope,
        event_scope=event_scope,
        data_mode=data_mode,
        cost_bps=cost_bps,
    )
    return DatasetBuildResult(dataset=dataset, factors=factors, panel=panel)


def load_dataset(path_or_name: str, config: dict[str, Any], schema: dict[str, Any]) -> pd.DataFrame:
    output_dir = _output_dir(config)
    logical_name = path_or_name
    if path_or_name == _table_name(schema, "ml_event_dataset", "cb_intraday_ml_event_dataset"):
        path = output_dir / f"{logical_name}.csv"
    else:
        path = Path(path_or_name)
    if not path.exists():
        raise FileNotFoundError(f"ML dataset not found: {path}")
    df = pd.read_csv(path)
    for col in ["signal_time", "first_seen_time", "alert_inserted_at", "effective_asof"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_walk_forward_splits(df: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dates = sorted(pd.Series(df["trade_date"].dropna().astype(str).unique()).tolist())
    val_cfg = cfg.get("validation", {})
    min_train = int(val_cfg.get("min_train_days", 15))
    min_test = int(val_cfg.get("min_test_days", 5))
    if len(dates) <= min_train:
        return []
    splits = []
    for test_idx in range(min_train, len(dates)):
        train_dates = dates[:test_idx]
        test_date = dates[test_idx]
        splits.append(
            {
                "split_id": f"wf_{test_date}",
                "train_dates": train_dates,
                "validation_dates": train_dates[-min(min_test, len(train_dates)) :],
                "test_dates": [test_date],
                "embargo_bars": int(val_cfg.get("embargo_bars", 6)),
                "purged": bool(val_cfg.get("use_purging", True)),
            }
        )
    return splits


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def _one_hot_encoder(dense: bool = False) -> Any:
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = []
    for col in NUMERIC_FEATURES:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            numeric.append(col)
    categorical = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].notna().any():
            categorical.append(col)
    return numeric, categorical


class SimplePreprocessor:
    def __init__(self) -> None:
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.levels: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "SimplePreprocessor":
        self.numeric, self.categorical = _feature_columns(df)
        self.feature_names = []
        for col in self.numeric:
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.medians[col] = median
            self.means[col] = mean
            self.stds[col] = std if std > 1e-12 else 1.0
            self.feature_names.append(col)
        for col in self.categorical:
            levels = sorted(df[col].fillna("__MISSING__").astype(str).unique().tolist())
            self.levels[col] = levels
            self.feature_names.extend([f"{col}={level}" for level in levels])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        columns = []
        for col in self.numeric:
            values = pd.to_numeric(df[col], errors="coerce").fillna(self.medians[col]).to_numpy(dtype=float)
            columns.append((values - self.means[col]) / self.stds[col])
        for col in self.categorical:
            values = df[col].fillna("__MISSING__").astype(str)
            for level in self.levels[col]:
                columns.append((values == level).to_numpy(dtype=float))
        if not columns:
            return np.empty((len(df), 0), dtype=float)
        return np.column_stack(columns)


class NumpyLogisticClassifier:
    def __init__(self, learning_rate: float = 0.05, max_iter: int = 1000, l2: float = 1.0, class_weight: str | None = "balanced") -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.class_weight = class_weight
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticClassifier":
        y = y.astype(float)
        n, p = x.shape
        self.coef_ = np.zeros(p, dtype=float)
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        if self.class_weight == "balanced":
            weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
        else:
            weights = np.ones(len(y), dtype=float)
        self.intercept_ = float(np.log(pos / neg))
        for _ in range(int(self.max_iter)):
            logits = x @ self.coef_ + self.intercept_
            probs = _sigmoid(logits)
            err = (probs - y) * weights
            grad_w = x.T @ err / n + self.l2 * self.coef_ / max(n, 1)
            grad_b = float(err.mean())
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
        return self

    def predict_proba_matrix(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fit")
        probs = _sigmoid(x @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - probs, probs])


class NumpyStumpBoostClassifier:
    def __init__(self, max_iter: int = 80, learning_rate: float = 0.05, max_thresholds: int = 5) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_thresholds = max_thresholds
        self.init_: float = 0.0
        self.stumps: list[tuple[int, float, float, float]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyStumpBoostClassifier":
        y = y.astype(float)
        pos = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        self.init_ = float(np.log(pos / (1.0 - pos)))
        logits = np.full(len(y), self.init_, dtype=float)
        self.stumps = []
        if x.size == 0:
            return self
        for _ in range(int(self.max_iter)):
            residual = y - _sigmoid(logits)
            best: tuple[float, int, float, float, float] | None = None
            for feature_idx in range(x.shape[1]):
                values = x[:, feature_idx]
                finite = np.isfinite(values)
                if finite.sum() < 5 or np.nanmax(values[finite]) <= np.nanmin(values[finite]):
                    continue
                thresholds = np.unique(np.nanquantile(values[finite], np.linspace(0.1, 0.9, self.max_thresholds)))
                for threshold in thresholds:
                    left = values <= threshold
                    right = ~left
                    if left.sum() < 2 or right.sum() < 2:
                        continue
                    left_value = float(residual[left].mean())
                    right_value = float(residual[right].mean())
                    pred = np.where(left, left_value, right_value)
                    loss = float(((residual - pred) ** 2).mean())
                    if best is None or loss < best[0]:
                        best = (loss, feature_idx, float(threshold), left_value, right_value)
            if best is None:
                break
            _, feature_idx, threshold, left_value, right_value = best
            update = np.where(x[:, feature_idx] <= threshold, left_value, right_value)
            logits += self.learning_rate * update
            self.stumps.append((feature_idx, threshold, left_value, right_value))
        return self

    def predict_proba_matrix(self, x: np.ndarray) -> np.ndarray:
        logits = np.full(x.shape[0], self.init_, dtype=float)
        for feature_idx, threshold, left_value, right_value in self.stumps:
            logits += self.learning_rate * np.where(x[:, feature_idx] <= threshold, left_value, right_value)
        probs = _sigmoid(logits)
        return np.column_stack([1.0 - probs, probs])


class NumpyPipeline:
    def __init__(self, estimator: Any) -> None:
        self.preprocessor = SimplePreprocessor()
        self.estimator = estimator
        self.feature_names_: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "NumpyPipeline":
        matrix = self.preprocessor.fit(x).transform(x)
        self.feature_names_ = self.preprocessor.feature_names
        self.estimator.fit(matrix, y.to_numpy(dtype=int))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba_matrix(self.preprocessor.transform(x))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def coefficients(self) -> dict[str, float]:
        coef = getattr(self.estimator, "coef_", None)
        if coef is None:
            return {}
        return {name: float(value) for name, value in zip(self.feature_names_, coef)}


class SklearnDynamicPreprocessor:
    def __init__(self, model: Any, dense_ohe: bool = False) -> None:
        self.model = model
        self.dense_ohe = dense_ohe
        self.pipeline = None
        self.numeric: list[str] = []
        self.categorical: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "SklearnDynamicPreprocessor":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.numeric, self.categorical = _feature_columns(x)
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder(dense=self.dense_ohe))])
        preprocessor = ColumnTransformer(
            [("num", num_pipe, self.numeric), ("cat", cat_pipe, self.categorical)],
            remainder="drop",
        )
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("model", self.model)])
        self.pipeline.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model not fit")
        return self.pipeline.predict_proba(x)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model not fit")
        return self.pipeline.predict(x)

    def coefficients(self) -> dict[str, float]:
        if self.pipeline is None:
            return {}
        model = self.pipeline.named_steps.get("model")
        coef = getattr(model, "coef_", None)
        if coef is None:
            return {}
        try:
            names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            names = [f"f{i}" for i in range(coef.ravel().shape[0])]
        return {str(name): float(value) for name, value in zip(names, coef.ravel())}


def make_model(model_type: str, config: dict[str, Any], dense: bool = False) -> Any:
    model_cfg = _ml_cfg(config).get("models", {})
    model_type = model_type.lower()
    if model_type == "logistic":
        cfg = model_cfg.get("logistic", {})
        if _sklearn_available():
            from sklearn.linear_model import LogisticRegression

            penalty = str(cfg.get("penalty", "l2"))
            kwargs = {
                "C": float(cfg.get("C", 1.0)),
                "class_weight": cfg.get("class_weight", "balanced"),
                "max_iter": int(cfg.get("max_iter", 1000)),
            }
            if penalty == "l1":
                kwargs["penalty"] = "l1"
                kwargs["solver"] = "liblinear"
            model = LogisticRegression(**kwargs)
            return SklearnDynamicPreprocessor(model, dense_ohe=False)
        l2 = 1.0 / max(float(cfg.get("C", 1.0)), 1e-6)
        return NumpyPipeline(
            NumpyLogisticClassifier(
                learning_rate=0.05,
                max_iter=int(cfg.get("max_iter", 1000)),
                l2=l2,
                class_weight=cfg.get("class_weight", "balanced"),
            )
        )
    if model_type in {"gbdt", "hist_gbdt", "tree"}:
        cfg = model_cfg.get("gbdt", {})
        if _sklearn_available():
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                max_iter=int(cfg.get("max_iter", 200)),
                learning_rate=float(cfg.get("learning_rate", 0.05)),
                max_leaf_nodes=int(cfg.get("max_leaf_nodes", 15)),
                random_state=42,
            )
            return SklearnDynamicPreprocessor(model, dense_ohe=True)
        return NumpyPipeline(
            NumpyStumpBoostClassifier(
            max_iter=int(cfg.get("max_iter", 200)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
                max_thresholds=max(3, min(int(cfg.get("max_leaf_nodes", 15)), 9)),
            )
        )
    if model_type == "calibrated":
        base = make_model("logistic", config)
        return CalibratedWrapper(base, config)
    raise ValueError(f"unsupported model_type={model_type}")


class CalibratedWrapper:
    def __init__(self, base: Any, config: dict[str, Any]) -> None:
        self.base = base
        self.config = config
        self.calibrator = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CalibratedWrapper":
        self.base.fit(x, y)
        probs = self.base.predict_proba(x)[:, 1]
        lr = NumpyLogisticClassifier(learning_rate=0.1, max_iter=500, l2=0.1, class_weight=None)
        lr.fit(probs.reshape(-1, 1), y.to_numpy(dtype=int))
        self.calibrator = lr
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        base_probs = self.base.predict_proba(x)[:, 1]
        if self.calibrator is None:
            probs = base_probs
        else:
            probs = self.calibrator.predict_proba_matrix(base_probs.reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - probs, probs])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def classification_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, Any]:
    y = pd.Series(y_true).astype(int)
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (y.to_numpy() == 1)).sum())
    fp = int(((pred == 1) & (y.to_numpy() == 0)).sum())
    fn = int(((pred == 0) & (y.to_numpy() == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    out = {
        "sample_count": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "accuracy": float((pred == y.to_numpy()).mean()) if len(y) else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "brier_score": float(np.mean((prob - y.to_numpy()) ** 2)) if len(y) else np.nan,
    }
    out["auc"] = _roc_auc(y.to_numpy(), prob) if y.nunique() > 1 else np.nan
    out["pr_auc"] = _average_precision(y.to_numpy(), prob) if y.nunique() > 1 else np.nan
    buckets = pd.DataFrame({"y": y, "p": prob})
    try:
        buckets["bucket"] = pd.qcut(buckets["p"].rank(method="first"), 5, labels=["p0_20", "p20_40", "p40_60", "p60_80", "p80_100"])
        out["calibration_bucket"] = buckets.groupby("bucket", observed=False).agg(count=("y", "size"), avg_p=("p", "mean"), hit_rate=("y", "mean")).reset_index().to_dict("records")
    except ValueError:
        out["calibration_bucket"] = []
    return out


def _roc_auc(y: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(prob) + 1)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(-prob)
    y_sorted = y[order]
    positives = y_sorted.sum()
    if positives <= 0:
        return np.nan
    precision_at_k = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float((precision_at_k * y_sorted).sum() / positives)


def probability_bucket_metrics(df: pd.DataFrame, prob_col: str = "p_good") -> dict[str, Any]:
    if df.empty or prob_col not in df.columns:
        return {}
    work = df.copy()
    work["prob_rank"] = work[prob_col].rank(pct=True, method="first")
    buckets = {
        "top_10": work[work["prob_rank"] >= 0.90],
        "top_20": work[work["prob_rank"] >= 0.80],
        "top_30": work[work["prob_rank"] >= 0.70],
        "middle": work[(work["prob_rank"] >= 0.30) & (work["prob_rank"] < 0.70)],
        "bottom": work[work["prob_rank"] < 0.30],
    }
    out: dict[str, Any] = {}
    for name, group in buckets.items():
        out[name] = {
            "sample_count": int(len(group)),
            "mean_return_1bar": float(group["return_1bar"].mean()) if "return_1bar" in group else np.nan,
            "mean_return_2bar": float(group["return_2bar"].mean()) if "return_2bar" in group else np.nan,
            "mean_return_3bar": float(group["return_3bar"].mean()) if "return_3bar" in group else np.nan,
            "mean_return_30m": float(group["return_30m"].mean()) if "return_30m" in group else np.nan,
            "mean_return_to_eod": float(group["return_to_eod"].mean()) if "return_to_eod" in group else np.nan,
            "median_return_30m": float(group["return_30m"].median()) if "return_30m" in group else np.nan,
            "hit_rate_30m": float((group["return_30m"].dropna() > 0).mean()) if "return_30m" in group and group["return_30m"].notna().any() else np.nan,
            "mfe_6bar": float(group["mfe_6bar"].mean()) if "mfe_6bar" in group else np.nan,
            "mae_6bar": float(group["mae_6bar"].mean()) if "mae_6bar" in group else np.nan,
            "average_minutes_to_close": float(group["minutes_to_close"].mean()) if "minutes_to_close" in group else np.nan,
            "daily_sample_count": group.groupby("trade_date").size().astype(int).to_dict() if "trade_date" in group else {},
            "daily_mean_return": group.groupby("trade_date")["return_30m"].mean().dropna().to_dict() if "trade_date" in group and "return_30m" in group else {},
        }
        if "trade_date" in group and "return_30m" in group and not group.empty:
            daily = group.groupby("trade_date")["return_30m"].mean().dropna()
            out[name]["worst_day"] = str(daily.idxmin()) if not daily.empty else None
            by_bond = group.groupby("bond_code")["return_30m"].mean().dropna()
            out[name]["worst_bond"] = str(by_bond.idxmin()) if not by_bond.empty else None
        out[name]["signal_reduction_ratio"] = 1.0 - (len(group) / len(work) if len(work) else np.nan)
    return out


def _training_frame(dataset: pd.DataFrame, target: str, event_scope: str | None = None, signal_scope: str | None = None) -> pd.DataFrame:
    df = dataset.copy()
    if event_scope:
        scopes = {item.strip().upper() for item in event_scope.split(",") if item.strip()}
        df = df[df["event_type"].isin(scopes)].copy()
    if signal_scope:
        signals = {item.strip() for item in signal_scope.split(",") if item.strip()}
        df = df[df["signal_type"].isin(signals)].copy()
    if target.startswith("label_action") or target in {"label_fake_breakout", "label_lag_repair_success", "label_triple_barrier_6bar"}:
        if "eligible_for_action_training" in df.columns:
            df = df[df["eligible_for_action_training"].fillna(False).astype(bool)].copy()
    if target.startswith("label_armed"):
        if "eligible_for_armed_training" in df.columns:
            df = df[df["eligible_for_armed_training"].fillna(False).astype(bool)].copy()
    df = df[df[target].notna()].copy()
    return df


def train_walk_forward(
    dataset: pd.DataFrame,
    target: str,
    model_type: str,
    config: dict[str, Any],
    event_scope: str | None = None,
    signal_scope: str | None = None,
) -> dict[str, Any]:
    df = _training_frame(dataset, target, event_scope, signal_scope)
    small_sample_warning = len(df) < 300 or df["trade_date"].nunique() < 20 or df[target].nunique() < 2
    splits = build_walk_forward_splits(df, _ml_cfg(config))
    predictions = []
    split_metrics = []
    for split in splits:
        train = df[df["trade_date"].isin(split["train_dates"])].copy()
        test = df[df["trade_date"].isin(split["test_dates"])].copy()
        if train.empty or test.empty or train[target].nunique() < 2:
            continue
        model = make_model(model_type, config)
        model.fit(train, train[target].astype(int))
        prob = model.predict_proba(test)[:, 1]
        test_out = test.copy()
        test_out["p_good"] = prob
        test_out["split_id"] = split["split_id"]
        predictions.append(test_out)
        split_metrics.append({"split_id": split["split_id"], **classification_metrics(test[target].astype(int), prob)})
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics = classification_metrics(pred_df[target].astype(int), pred_df["p_good"].to_numpy()) if not pred_df.empty else {"sample_count": 0}
    metrics["small_sample_warning"] = bool(small_sample_warning)
    metrics["walk_forward_split_count"] = int(len(split_metrics))
    metrics["split_metrics"] = split_metrics
    metrics["probability_bucket_metrics"] = probability_bucket_metrics(pred_df, "p_good")
    metrics["rule_baseline"] = rule_baseline_metrics(df, target)
    model_final = None
    if not df.empty and df[target].nunique() >= 2:
        model_final = make_model(model_type, config)
        model_final.fit(df, df[target].astype(int))
    return {
        "training_rows": int(len(df)),
        "target": target,
        "model_type": model_type,
        "metrics": metrics,
        "predictions": pred_df,
        "model": model_final,
        "feature_columns": {"numeric": _feature_columns(df)[0], "categorical": _feature_columns(df)[1]},
    }


def rule_baseline_metrics(df: pd.DataFrame, target: str) -> dict[str, Any]:
    if df.empty or target not in df.columns or df[target].nunique() < 2:
        return {"sample_count": int(len(df)), "message": "insufficient baseline rows"}
    score = pd.to_numeric(df.get("trigger_score", df.get("setup_score", 0.0)), errors="coerce").fillna(0.0)
    if score.max() > score.min():
        prob = ((score - score.min()) / (score.max() - score.min())).to_numpy()
    else:
        prob = np.full(len(df), 0.5)
    return classification_metrics(df[target].astype(int), prob)


def save_experiment(
    result: dict[str, Any],
    dataset: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
    start_date: str,
    end_date: str,
    pool_scope: str,
    event_scope: str,
    signal_scope: str | None,
    factor_version: str,
) -> dict[str, Any]:
    cfg = _ml_cfg(config)
    output_dir = _output_dir(config)
    now_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
    experiment_id = f"exp_{now_ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    model_version = f"{cfg.get('model_version_prefix', MODEL_VERSION_PREFIX)}_{result['model_type']}_{result['target']}_{experiment_id[-8:]}"
    artifact_path = output_dir / f"{model_version}.pkl"
    if result.get("model") is not None:
        with artifact_path.open("wb") as fh:
            pickle.dump(
                {
                    "model": result["model"],
                    "model_version": model_version,
                    "target": result["target"],
                    "model_type": result["model_type"],
                    "feature_columns": result["feature_columns"],
                    "config": cfg,
                },
                fh,
            )
    metrics = result["metrics"]
    record = {
        "experiment_id": experiment_id,
        "model_version": model_version,
        "feature_version": str(cfg.get("feature_version", FEATURE_VERSION)),
        "label_version": str(cfg.get("label_version", LABEL_VERSION)),
        "factor_version": factor_version,
        "config_version": str(cfg.get("config_version", "intraday_factors_v0.4")),
        "train_start_date": str(dataset["trade_date"].min()) if not dataset.empty else None,
        "train_end_date": str(dataset["trade_date"].max()) if not dataset.empty else None,
        "validation_start_date": None,
        "validation_end_date": None,
        "test_start_date": start_date,
        "test_end_date": end_date,
        "pool_scope": pool_scope,
        "event_scope": event_scope,
        "signal_scope": signal_scope or "",
        "target_label": result["target"],
        "model_type": result["model_type"],
        "hyperparameters_json": json.dumps(_ml_cfg(config).get("models", {}).get(result["model_type"], {}), ensure_ascii=False),
        "feature_list_json": json.dumps(result["feature_columns"], ensure_ascii=False),
        "metrics_json": json.dumps(metrics, ensure_ascii=False, default=_json_default),
        "artifact_path": str(artifact_path) if result.get("model") is not None else "",
        "created_at": now_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "V0.4 shadow-only ML meta-filter; not a production gate.",
    }
    exp_name = _table_name(schema, "ml_experiments", "cb_intraday_ml_experiments")
    jsonl_path = output_dir / f"{exp_name}.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return record


def load_model_artifact(model_version_or_path: str, config: dict[str, Any]) -> dict[str, Any]:
    path = Path(model_version_or_path)
    if not path.exists():
        path = _output_dir(config) / f"{model_version_or_path}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_version_or_path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def build_predictions(dataset: pd.DataFrame, artifact: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    model = artifact["model"]
    target = str(artifact.get("target", ""))
    out = dataset.copy()
    prob = model.predict_proba(out)[:, 1]
    out["prediction_id"] = [f"pred_{uuid.uuid4().hex}" for _ in range(len(out))]
    out["model_version"] = artifact.get("model_version", "")
    out["p_action_good_30m"] = np.nan
    out["p_action_good_to_eod"] = np.nan
    out["p_action_bad_3bar"] = np.nan
    out["p_fake_breakout"] = np.nan
    out["p_lag_repair_success"] = np.nan
    out["p_armed_to_action_1bar"] = np.nan
    out["p_armed_to_action_3bar"] = np.nan
    target_to_col = {
        "label_action_good_30m": "p_action_good_30m",
        "label_action_good_to_eod": "p_action_good_to_eod",
        "label_action_bad_3bar": "p_action_bad_3bar",
        "label_fake_breakout": "p_fake_breakout",
        "label_lag_repair_success": "p_lag_repair_success",
        "label_armed_to_action_1bar": "p_armed_to_action_1bar",
        "label_armed_to_action_3bar": "p_armed_to_action_3bar",
    }
    out[target_to_col.get(target, "p_action_good_30m")] = prob
    thresholds = _ml_cfg(config).get("ml_action_hint_thresholds", {})
    boost = float(thresholds.get("boost_p_good_min", 0.65))
    suppress_bad = float(thresholds.get("suppress_p_bad_min", 0.55))
    suppress_fake = float(thresholds.get("fake_breakout_suppress_min", 0.60))
    good_prob = out["p_action_good_30m"].fillna(out["p_action_good_to_eod"]).fillna(0.5)
    bad_prob = out["p_action_bad_3bar"].fillna(0.0)
    fake_prob = out["p_fake_breakout"].fillna(0.0)
    out["confidence_bucket"] = pd.cut(
        prob,
        bins=[-np.inf, 0.35, 0.50, 0.65, np.inf],
        labels=["LOW", "MID_LOW", "MID_HIGH", "HIGH"],
    ).astype("object")
    out["ml_action_hint"] = "NEUTRAL"
    out.loc[good_prob >= boost, "ml_action_hint"] = "BOOST"
    out.loc[(bad_prob >= suppress_bad) | (fake_prob >= suppress_fake), "ml_action_hint"] = "SUPPRESS"
    out.loc[out["event_type"].eq("WATCH"), "ml_action_hint"] = "WATCH_ONLY"
    out["ml_reason_text"] = out.apply(_ml_reason_text, axis=1)
    out["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
    keep = [
        "prediction_id",
        "event_id",
        "episode_id",
        "trade_date",
        "bond_code",
        "stock_code",
        "signal_time",
        "first_seen_time",
        "event_type",
        "signal_type",
        "pool_state",
        "model_version",
        "feature_version",
        "label_version",
        "factor_version",
        "config_version",
        "data_mode",
        "p_action_good_30m",
        "p_action_good_to_eod",
        "p_action_bad_3bar",
        "p_fake_breakout",
        "p_lag_repair_success",
        "p_armed_to_action_1bar",
        "p_armed_to_action_3bar",
        "confidence_bucket",
        "ml_action_hint",
        "ml_reason_text",
        "created_at",
    ]
    return out[[col for col in keep if col in out.columns]].copy()


def _ml_reason_text(row: pd.Series) -> str:
    if row.get("ml_action_hint") == "BOOST":
        p_good = _safe_float(row.get("p_action_good_30m"))
        if not np.isfinite(p_good):
            p_good = _safe_float(row.get("p_action_good_to_eod"))
        return f"ML shadow: 高置信增强，p_good={p_good:.2f}。仅供人工参考。"
    if row.get("ml_action_hint") == "SUPPRESS":
        return (
            f"ML shadow: 风险压制，p_bad_3bar={_safe_float(row.get('p_action_bad_3bar')):.2f}，"
            f"p_fake_breakout={_safe_float(row.get('p_fake_breakout')):.2f}。不改变规则ACTION。"
        )
    if row.get("event_type") == "ARMED":
        p1 = _safe_float(row.get("p_armed_to_action_1bar"))
        p3 = _safe_float(row.get("p_armed_to_action_3bar"))
        if np.isfinite(p1) or np.isfinite(p3):
            return f"ML shadow: ARMED转ACTION概率1bar={p1:.2f}，3bar={p3:.2f}。"
        p_good = _safe_float(row.get("p_action_good_30m"))
        if np.isfinite(p_good):
            return f"ML shadow: ARMED事件的ACTION-good影子评分={p_good:.2f}，仅用于研究排序。"
        return "ML shadow: ARMED事件暂无匹配目标概率，仅保留为shadow展示。"
    return "ML shadow: 中性，仅作为排序/解释字段，不改变正式alert_state。"


def shadow_research_summary(dataset: pd.DataFrame, predictions: pd.DataFrame | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset_rows": int(len(dataset)),
        "action_rows": int((dataset["event_type"] == "ACTION").sum()) if "event_type" in dataset else 0,
        "armed_rows": int((dataset["event_type"] == "ARMED").sum()) if "event_type" in dataset else 0,
        "lag_rows": int((dataset["signal_type"] == "LAG_REPAIR_LONG").sum()) if "signal_type" in dataset else 0,
        "breakout_rows": int((dataset["signal_type"] == "MOMENTUM_BREAKOUT_LONG").sum()) if "signal_type" in dataset else 0,
        "small_sample_warning": bool(len(dataset) < 1000 or (dataset["event_type"].eq("ACTION").sum() if "event_type" in dataset else 0) < 300),
    }
    for signal in ["LAG_REPAIR_LONG", "MOMENTUM_BREAKOUT_LONG", "ARMED_LAG_LONG", "ARMED_BREAKOUT_LONG"]:
        group = dataset[dataset["signal_type"] == signal] if "signal_type" in dataset else pd.DataFrame()
        out[f"{signal}_summary"] = {
            "sample_count": int(len(group)),
            "return_1bar_mean": float(group["return_1bar"].mean()) if "return_1bar" in group and group["return_1bar"].notna().any() else np.nan,
            "return_3bar_mean": float(group["return_3bar"].mean()) if "return_3bar" in group and group["return_3bar"].notna().any() else np.nan,
            "return_30m_mean": float(group["return_30m"].mean()) if "return_30m" in group and group["return_30m"].notna().any() else np.nan,
            "return_to_eod_mean": float(group["return_to_eod"].mean()) if "return_to_eod" in group and group["return_to_eod"].notna().any() else np.nan,
            "mae_6bar_mean": float(group["mae_6bar"].mean()) if "mae_6bar" in group and group["mae_6bar"].notna().any() else np.nan,
        }
    if predictions is not None and not predictions.empty:
        out["prediction_rows"] = int(len(predictions))
        out["ml_hint_counts"] = predictions["ml_action_hint"].value_counts().astype(int).to_dict()
        joined = dataset.merge(predictions[["event_id", "ml_action_hint"]], on="event_id", how="inner")
        out["shadow_filter"] = {
            hint: {
                "sample_count": int(len(group)),
                "return_30m_mean": float(group["return_30m"].mean()) if group["return_30m"].notna().any() else np.nan,
                "hit_rate_30m": float((group["return_30m"].dropna() > 0).mean()) if group["return_30m"].notna().any() else np.nan,
                "mae_6bar_mean": float(group["mae_6bar"].mean()) if group["mae_6bar"].notna().any() else np.nan,
            }
            for hint, group in joined.groupby("ml_action_hint")
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V0.4 ML shadow meta-filter research CLI.")
    parser.add_argument("--mode", required=True, choices=["build-ml-dataset", "train-ml-shadow", "predict-ml-shadow", "evaluate-ml-shadow", "report-ml-shadow"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--factor-version", default="v0.3")
    parser.add_argument("--config-version", default=None)
    parser.add_argument("--pool-scope", default="ACTIVE_ONLY")
    parser.add_argument("--event-scope", default="ARMED,ACTION")
    parser.add_argument("--signal-scope", default=None)
    parser.add_argument("--data-mode", default="manual_backfill")
    parser.add_argument("--dataset", default="cb_intraday_ml_event_dataset")
    parser.add_argument("--output-table", default="cb_intraday_ml_event_dataset")
    parser.add_argument("--prediction-table", default="cb_intraday_ml_predictions")
    parser.add_argument("--target", default="label_action_good_30m")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--live-lookback-days", type=int, default=70)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--full-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    config = load_config(args.config)
    schema = load_schema(args.schema)
    output_dir = _output_dir(config)

    if args.mode == "build-ml-dataset":
        start, end = normalize_time_window(args.start, args.end, args.start_date, args.end_date)
        result = build_ml_event_dataset(
            start,
            end,
            config,
            schema,
            factor_version=args.factor_version,
            config_version=args.config_version,
            pool_scope=args.pool_scope,
            event_scope=args.event_scope,
            data_mode=args.data_mode,
            live_lookback_days=args.live_lookback_days,
            cost_bps=args.cost_bps,
            schema_path=args.schema,
        )
        logical_name = args.output_table or _table_name(schema, "ml_event_dataset", "cb_intraday_ml_event_dataset")
        csv_path, parquet_path = write_dataframe(result.dataset, output_dir, logical_name)
        print(json.dumps({"rows": len(result.dataset), "csv_path": str(csv_path), "parquet_path": str(parquet_path) if parquet_path else None, **shadow_research_summary(result.dataset)}, ensure_ascii=False, default=_json_default, indent=2))
        return

    dataset = load_dataset(args.dataset, config, schema)
    if args.mode == "train-ml-shadow":
        result = train_walk_forward(dataset, args.target, args.model_type, config, args.event_scope, args.signal_scope)
        start = args.start_date or str(dataset["trade_date"].min())
        end = args.end_date or str(dataset["trade_date"].max())
        record = save_experiment(result, dataset, config, schema, start, end, args.pool_scope, args.event_scope, args.signal_scope, args.factor_version)
        pred_name = f"{record['model_version']}_walk_forward_predictions"
        pred_path, _ = write_dataframe(result["predictions"], output_dir, pred_name)
        print(json.dumps({"experiment": record, "metrics": result["metrics"], "walk_forward_prediction_path": str(pred_path)}, ensure_ascii=False, default=_json_default, indent=2))
        return

    if args.mode == "predict-ml-shadow":
        if not args.model_version:
            raise SystemExit("--model-version is required for predict-ml-shadow")
        artifact = load_model_artifact(args.model_version, config)
        predictions = build_predictions(dataset, artifact, config)
        name = args.prediction_table or _table_name(schema, "ml_predictions", "cb_intraday_ml_predictions")
        csv_path, parquet_path = write_dataframe(predictions, output_dir, name)
        print(json.dumps({"rows": len(predictions), "csv_path": str(csv_path), "parquet_path": str(parquet_path) if parquet_path else None, "summary": shadow_research_summary(dataset, predictions), "sample": predictions.head(1).to_dict("records")}, ensure_ascii=False, default=_json_default, indent=2))
        return

    if args.mode == "evaluate-ml-shadow":
        predictions_path = output_dir / f"{args.prediction_table}.csv"
        predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
        print(json.dumps(shadow_research_summary(dataset, predictions), ensure_ascii=False, default=_json_default, indent=2))
        return

    if args.mode == "report-ml-shadow":
        exp_name = _table_name(schema, "ml_experiments", "cb_intraday_ml_experiments")
        exp_path = output_dir / f"{exp_name}.jsonl"
        records = []
        if exp_path.exists():
            for line in exp_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        if args.experiment_id:
            records = [record for record in records if record.get("experiment_id") == args.experiment_id]
        selected = records[-10:]
        if not args.full_report:
            summaries = []
            for record in selected:
                try:
                    metrics = json.loads(record.get("metrics_json", "{}"))
                except json.JSONDecodeError:
                    metrics = {}
                summaries.append(
                    {
                        "experiment_id": record.get("experiment_id"),
                        "model_version": record.get("model_version"),
                        "target_label": record.get("target_label"),
                        "model_type": record.get("model_type"),
                        "event_scope": record.get("event_scope"),
                        "signal_scope": record.get("signal_scope"),
                        "sample_count": metrics.get("sample_count"),
                        "positive_rate": metrics.get("positive_rate"),
                        "auc": metrics.get("auc"),
                        "pr_auc": metrics.get("pr_auc"),
                        "brier_score": metrics.get("brier_score"),
                        "walk_forward_split_count": metrics.get("walk_forward_split_count"),
                        "small_sample_warning": metrics.get("small_sample_warning"),
                    }
                )
            print(json.dumps({"experiment_count": len(records), "experiments": summaries}, ensure_ascii=False, default=_json_default, indent=2))
            return
        print(json.dumps({"experiment_count": len(records), "experiments": selected}, ensure_ascii=False, default=_json_default, indent=2))


if __name__ == "__main__":
    main()
