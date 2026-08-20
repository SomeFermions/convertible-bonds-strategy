from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from intraday_factors.config import load_config, load_schema
from intraday_factors.data_source import IntradayDataAdapter, get_connection, insert_alert_rows, insert_factor_rows
from intraday_factors.pool_state import DEFAULT_ACTIONABLE_POOL_STATES, normalize_pool_state
from intraday_factors.trading_session import expected_slots_per_day, session_bar_role
from intraday_factors.utils import (
    clip_zscore,
    rolling_regression,
    rolling_rv,
    rolling_sum,
    safe_log_return,
    same_slot_robust_z,
)


LOGGER = logging.getLogger(__name__)


def _resolve_worker_count(max_workers: int | None, group_count: int) -> int:
    if group_count <= 1:
        return 1
    if max_workers is None or int(max_workers) <= 0:
        max_workers = os.cpu_count() or 1
    return max(1, min(int(max_workers), int(group_count)))


def _residual_model_group_worker(payload: tuple[pd.DataFrame, dict[str, Any]]) -> pd.DataFrame:
    group, residual_cfg = payload
    group = group.copy()
    if group["r_cb_market"].notna().sum() >= int(residual_cfg["min_observations"]):
        x = group[["r_stock", "r_cb_market"]]
    else:
        x = group[["r_stock"]]
        group["r_cb_market"] = np.nan
    params = rolling_regression(
        group["r_cb"],
        x,
        window=int(residual_cfg["window_bars"]),
        min_observations=int(residual_cfg["min_observations"]),
        ridge_alpha=float(residual_cfg["ridge_alpha"]),
    )
    group["alpha"] = params["alpha"]
    group["beta"] = params.get("r_stock")
    group["gamma"] = params.get("r_cb_market", np.nan)
    group["residual_model_nobs"] = params["nobs"]
    group["residual_model_r2"] = params["r2"]
    group["residual"] = group["r_cb"] - group["alpha"] - group["beta"] * group["r_stock"]
    group["residual"] = group["residual"] - group["gamma"].fillna(0.0) * group["r_cb_market"].fillna(0.0)
    return group


def _empirical_delta_group_worker(payload: tuple[pd.DataFrame, dict[str, Any]]) -> pd.Series:
    group, residual_cfg = payload
    params = rolling_regression(
        group["delta_bond_price"],
        group[["delta_parity"]],
        window=int(residual_cfg["window_bars"]),
        min_observations=max(10, int(residual_cfg["min_observations"]) // 2),
        ridge_alpha=float(residual_cfg["ridge_alpha"]),
    )
    return params["delta_parity"]


RESULT_COLUMNS = [
    "bar_start",
    "bond_code",
    "stock_code",
    "pool_state",
    "is_actionable_bar",
    "session_bar_role",
    "parity",
    "premium_raw",
    "premium_slot_z",
    "factor_premium_mr",
    "premium_change_5m",
    "premium_change_10m",
    "premium_change_30m",
    "premium_change_5m_z",
    "premium_change_10m_z",
    "premium_change_30m_z",
    "premium_model_residual",
    "premium_model_residual_z",
    "premium_factor_source",
    "r_cb",
    "r_stock",
    "r_cb_market",
    "alpha",
    "beta",
    "gamma",
    "residual",
    "residual_ewm",
    "residual_z",
    "residual_z_change_1bar",
    "residual_z_change_2bar",
    "residual_z_slope_3bar",
    "residual_repair_started",
    "factor_residual_mr",
    "stock_mom_5m_raw",
    "stock_mom_10m_raw",
    "stock_mom_15m_raw",
    "stock_mom_5m_z",
    "stock_mom_10m_z",
    "stock_mom_15m_z",
    "stock_amount_burst_z",
    "stock_impulse_score",
    "stock_mom_30m",
    "stock_mom_30m_z",
    "stock_mom_60m",
    "stock_mom_60m_z",
    "stock_mom_1d",
    "stock_mom_1d_z",
    "stock_mom_5d",
    "stock_mom_5d_z",
    "stock_momentum_score",
    "cb_mom_5m",
    "cb_mom_10m",
    "cb_mom_15m",
    "cb_mom_30m",
    "cb_mom_60m",
    "cb_mom_5m_z",
    "cb_mom_10m_z",
    "cb_mom_15m_z",
    "cb_mom_30m_z",
    "cb_intraday_vwap",
    "stock_intraday_vwap",
    "cb_vwap_dev",
    "stock_vwap_dev",
    "vwap_gap",
    "stock_rv_short",
    "stock_rv_long",
    "cb_rv_short",
    "cb_rv_long",
    "residual_rv_short",
    "residual_rv_long",
    "residual_rv_long_fallback",
    "residual_vol_ratio",
    "residual_vol_ratio_z",
    "residual_vol_ratio_source",
    "residual_vol_ratio_fallback",
    "residual_vol_ratio_fallback_z",
    "residual_vol_ratio_fallback_source",
    "fallback_vol_available",
    "factor_vol_mr",
    "factor_vol_mr_fallback",
    "empirical_delta_proxy",
    "elasticity_proxy",
    "empirical_gamma_proxy",
    "convexity_asymmetry_proxy",
    "factor_vol_response_gap",
    "cb_flow_30m",
    "stock_flow_30m",
    "cb_flow_z",
    "stock_flow_z",
    "flow_confirmation",
    "flow_lead",
    "cb_market_mom_30m",
    "cb_market_breadth_30m",
    "cb_market_breadth_60m",
    "cb_market_amount_burst",
    "stock_market_breadth_30m",
    "market_regime_intraday",
    "adv20_amount",
    "relative_amount",
    "active_bar_ratio",
    "amihud",
    "high_low_range_proxy",
    "high_low_range_proxy_z",
    "high_low_range_proxy_z_source",
    "volatility_gate_pass",
    "missing_vol_gate",
    "missing_range_gate",
    "noise_penalty",
    "data_quality_pass",
    "liquidity_pass",
    "trend_gate",
    "factor_coverage",
    "factor_coverage_pass",
    "missing_cb_bar",
    "missing_stock_bar",
    "stale_price_flag",
    "insufficient_history_flag",
    "fallback_standardization_flag",
    "setup_score",
    "trigger_score",
    "exit_score",
    "exit_state",
    "exit_type",
    "signal_score",
    "signal_type",
    "watch_signal_type",
    "watch_missing_to_action",
    "signal_state",
    "alert_state",
    "action_reason_text",
    "risk_hint_text",
    "cooldown_suppressed",
    "suppression_reason",
    "no_new_action_after_suppressed",
    "action_blocked_by_time",
    "alert_expire_time",
    "last_action_reason",
    "amount_source",
]


@dataclass
class FactorEngine:
    config: dict[str, Any]
    schema: dict[str, Any]

    @classmethod
    def from_files(cls, config_path: str | None = None, schema_path: str | None = None) -> "FactorEngine":
        return cls(load_config(config_path), load_schema(schema_path))

    @property
    def factor_version(self) -> str:
        return str(self.config["factor_version"])

    @property
    def adapter(self) -> IntradayDataAdapter:
        return IntradayDataAdapter(self.schema, bar_minutes=int(self.config["bar_minutes"]))

    def compute_panel_factors(self, panel: pd.DataFrame, daily_amounts: pd.DataFrame | None = None) -> pd.DataFrame:
        """Compute V0 factors. All rolling standardization uses past-only history."""
        if panel.empty:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        df = panel.copy().sort_values(["bond_code", "bar_start"]).reset_index(drop=True)
        eps = float(self.config["epsilon"])
        clip = float(self.config["clip_zscore"])
        fallback_window = int(self.config["fallback_rolling_window"])
        same_slot_samples = int(self.config["same_slot_lookback_days"])
        min_same_slot = int(self.config["minimum_same_slot_samples"])
        momentum_cfg = self.config["momentum"]
        vol_cfg = self.config["volatility"]
        residual_cfg = self.config["residual_model"]
        performance_cfg = self.config.get("performance", {})

        if daily_amounts is not None and not daily_amounts.empty:
            df = df.merge(daily_amounts, on="bond_code", how="left")
        else:
            df["adv20_amount"] = np.nan
        if "pool_state" not in df.columns:
            df["pool_state"] = "ACTIVE"
        df["pool_state"] = df["pool_state"].apply(lambda value: normalize_pool_state(value))
        if "session_bar_role" not in df.columns:
            df["session_bar_role"] = df["bar_start"].apply(session_bar_role)
        else:
            df["session_bar_role"] = df["session_bar_role"].combine_first(df["bar_start"].apply(session_bar_role))
        if "trade_date" not in df.columns:
            df["trade_date"] = pd.to_datetime(df["bar_start"], errors="coerce").dt.strftime("%Y-%m-%d")

        df["missing_cb_bar"] = df["cb_close"].isna()
        df["missing_stock_bar"] = df["stock_close"].isna()
        df["stale_price_flag"] = (
            df["cb_close"].isna()
            | (df["cb_close"] <= 0)
            | (df["cb_high"] < df["cb_low"])
            | (df["cb_quote_count"].fillna(0) <= 0)
        )

        df["parity"] = 100.0 * df["stock_close"] / df["conversion_price"]
        df.loc[(df["conversion_price"] <= 0) | (df["stock_close"] <= 0), "parity"] = np.nan
        df["premium_raw"] = df["cb_close"] / df["parity"] - 1.0

        df["r_cb"] = df.groupby("bond_code", group_keys=False)["cb_close"].apply(safe_log_return)
        df["r_stock"] = df.groupby("bond_code", group_keys=False)["stock_close"].apply(safe_log_return)
        df["r_parity"] = df.groupby("bond_code", group_keys=False)["parity"].apply(safe_log_return)

        # Short-cycle impulse fields are research-only in V0.3. They use the
        # completed current bar and past-only standardization.
        impulse_fallback_any = pd.Series(False, index=df.index, dtype="bool")
        for name, window in [("stock_mom_5m_raw", 1), ("stock_mom_10m_raw", 2), ("stock_mom_15m_raw", 3)]:
            df[name] = df.groupby("bond_code", group_keys=False)["r_stock"].apply(
                lambda s, w=window: rolling_sum(s, w, max(1, min(w, 2)))
            )
            z, impulse_fallback = same_slot_robust_z(
                df,
                name,
                ["bond_code"],
                "slot_id",
                same_slot_samples,
                min_same_slot,
                fallback_window,
                eps,
                clip,
            )
            df[f"{name.replace('_raw', '')}_z"] = z
            df[f"{name}_fallback"] = impulse_fallback
            impulse_fallback_any = impulse_fallback_any | impulse_fallback
        df["stock_amount_log"] = np.log1p(df["stock_amount_used"].clip(lower=0))
        df["stock_amount_burst_z"], stock_amount_fallback = same_slot_robust_z(
            df,
            "stock_amount_log",
            ["bond_code"],
            "slot_id",
            same_slot_samples,
            min_same_slot,
            fallback_window,
            eps,
            clip,
        )
        impulse_fallback_any = impulse_fallback_any | stock_amount_fallback
        df["stock_impulse_score"] = (
            0.50 * df["stock_mom_5m_z"] + 0.30 * df["stock_mom_10m_z"] + 0.20 * df["stock_amount_burst_z"]
        )

        grouped_day = df.groupby(["bond_code", "trade_date"], group_keys=False)
        cb_cum_volume = grouped_day["cb_volume"].cumsum()
        cb_cum_amount = grouped_day["amount_used"].cumsum()
        stock_cum_volume = grouped_day["stock_volume"].cumsum()
        stock_cum_amount = grouped_day["stock_amount_used"].cumsum()
        df["cb_intraday_vwap"] = cb_cum_amount / cb_cum_volume.replace(0, np.nan)
        df["stock_intraday_vwap"] = stock_cum_amount / stock_cum_volume.replace(0, np.nan)
        df["cb_vwap_dev"] = df["cb_close"] / df["cb_intraday_vwap"] - 1.0
        df["stock_vwap_dev"] = df["stock_close"] / df["stock_intraday_vwap"] - 1.0
        df["vwap_gap"] = df["stock_vwap_dev"] - df["cb_vwap_dev"]

        market_stats = df.groupby("bar_start")["r_cb"].agg(["median", "count"]).rename(columns={"median": "r_cb_market", "count": "r_cb_market_count"})
        df = df.merge(market_stats, on="bar_start", how="left")
        df.loc[df["r_cb_market_count"] < int(residual_cfg["min_market_count"]), "r_cb_market"] = np.nan

        # Momentum raw and z-scores.
        for name, window in [
            ("stock_mom_30m", int(momentum_cfg["bars_30m"])),
            ("stock_mom_60m", int(momentum_cfg["bars_60m"])),
            ("stock_mom_1d", int(momentum_cfg["bars_1d"])),
            ("stock_mom_5d", int(momentum_cfg["bars_5d"])),
        ]:
            df[name] = df.groupby("bond_code", group_keys=False)["r_stock"].apply(lambda s, w=window: rolling_sum(s.shift(1), w, max(2, min(w, 6))))
            z, fallback = same_slot_robust_z(
                df,
                name,
                ["bond_code"],
                "slot_id",
                same_slot_samples,
                min_same_slot,
                fallback_window,
                eps,
                clip,
            )
            df[f"{name}_z"] = z
            df[f"{name}_fallback"] = fallback

        df["stock_momentum_score"] = (
            0.50 * df["stock_mom_30m_z"] + 0.30 * df["stock_mom_1d_z"] + 0.20 * df["stock_mom_5d_z"]
        )
        if df["stock_mom_1d_z"].isna().all() and df["stock_mom_5d_z"].isna().all():
            df["stock_momentum_score"] = df["stock_mom_30m_z"]

        cb_mom_fallback_any = pd.Series(False, index=df.index, dtype="bool")
        for name, window in [("cb_mom_5m", 1), ("cb_mom_10m", 2), ("cb_mom_15m", 3), ("cb_mom_30m", 6), ("cb_mom_60m", 12)]:
            df[name] = df.groupby("bond_code", group_keys=False)["r_cb"].apply(
                lambda s, w=window: rolling_sum(s, w, max(1, min(w, 3)))
            )
            z, fallback = same_slot_robust_z(
                df,
                name,
                ["bond_code"],
                "slot_id",
                same_slot_samples,
                min_same_slot,
                fallback_window,
                eps,
                clip,
            )
            if name != "cb_mom_60m":
                df[f"{name}_z"] = z
            df[f"{name}_fallback"] = fallback
            cb_mom_fallback_any = cb_mom_fallback_any | fallback

        # Residual mean-reversion model.
        pieces = self._map_bond_groups(
            df,
            _residual_model_group_worker,
            residual_cfg,
            performance_cfg,
            task_name="residual_model",
        )
        df = pd.concat(pieces, ignore_index=True).sort_values(["bond_code", "bar_start"]).reset_index(drop=True)
        df["residual_ewm"] = df.groupby("bond_code", group_keys=False)["residual"].apply(
            lambda s: s.ewm(span=int(residual_cfg["residual_ewm_span"]), adjust=False, min_periods=2).mean()
        )
        df["residual_z"], residual_fallback = same_slot_robust_z(
            df,
            "residual_ewm",
            ["bond_code"],
            "slot_id",
            same_slot_samples,
            min_same_slot,
            fallback_window,
            eps,
            clip,
        )
        df["factor_residual_mr"] = -clip_zscore(df["residual_z"], clip)
        df["residual_z_change_1bar"] = df["residual_z"] - df.groupby("bond_code")["residual_z"].shift(1)
        df["residual_z_change_2bar"] = df["residual_z"] - df.groupby("bond_code")["residual_z"].shift(2)
        df["residual_z_slope_3bar"] = df.groupby("bond_code", group_keys=False)["residual_z"].apply(
            lambda s: s.rolling(3, min_periods=2).apply(
                lambda values: np.polyfit(np.arange(len(values)), values, 1)[0] if np.isfinite(values).sum() >= 2 else np.nan,
                raw=True,
            )
        )
        df["residual_repair_started"] = (df["residual_z_change_1bar"] > 0) | (df["residual_z_change_2bar"] > 0)

        # Premium mean-reversion. Conditional premium model is deliberately disabled in V0.
        df["premium_slot_z"], premium_fallback = same_slot_robust_z(
            df,
            "premium_raw",
            ["bond_code"],
            "slot_id",
            same_slot_samples,
            min_same_slot,
            fallback_window,
            eps,
            clip,
        )
        df["premium_model_residual"] = np.nan
        df["premium_model_residual_z"] = np.nan
        df["premium_factor_source"] = "slot_z"
        df["factor_premium_mr"] = -clip_zscore(df["premium_slot_z"], clip)
        premium_change_fallback_any = pd.Series(False, index=df.index, dtype="bool")
        for name, bars in [("premium_change_5m", 1), ("premium_change_10m", 2), ("premium_change_30m", 6)]:
            df[name] = df["premium_raw"] - df.groupby("bond_code")["premium_raw"].shift(bars)
            z, premium_change_fallback = same_slot_robust_z(
                df,
                name,
                ["bond_code"],
                "slot_id",
                same_slot_samples,
                min_same_slot,
                fallback_window,
                eps,
                clip,
            )
            df[f"{name}_z"] = z
            df[f"{name}_fallback"] = premium_change_fallback
            premium_change_fallback_any = premium_change_fallback_any | premium_change_fallback

        # Volatility and response gap. Primary vol uses multi-day history; fallback is display-only.
        short_w = int(vol_cfg.get("short_window_bars", vol_cfg.get("short_window", 6)))
        primary_long_w = int(vol_cfg.get("primary_long_window_bars", vol_cfg.get("long_window", 240)))
        fallback_long_w = int(vol_cfg.get("fallback_long_window_bars", 48))
        min_primary_obs = int(vol_cfg.get("min_primary_obs", max(10, min(primary_long_w, 60))))
        min_fallback_obs = int(vol_cfg.get("min_fallback_obs", max(10, min(fallback_long_w, 36))))
        df["stock_rv_short"] = df.groupby("bond_code", group_keys=False)["r_stock"].apply(lambda s: rolling_rv(s, short_w, max(2, min(short_w, 3))))
        df["stock_rv_long"] = df.groupby("bond_code", group_keys=False)["r_stock"].apply(lambda s: rolling_rv(s, primary_long_w, min_primary_obs))
        df["cb_rv_short"] = df.groupby("bond_code", group_keys=False)["r_cb"].apply(lambda s: rolling_rv(s, short_w, max(2, min(short_w, 3))))
        df["cb_rv_long"] = df.groupby("bond_code", group_keys=False)["r_cb"].apply(lambda s: rolling_rv(s, primary_long_w, min_primary_obs))
        df["residual_rv_short"] = df.groupby("bond_code", group_keys=False)["residual"].apply(lambda s: rolling_rv(s, short_w, max(2, min(short_w, 3))))
        df["residual_rv_long"] = df.groupby("bond_code", group_keys=False)["residual"].apply(lambda s: rolling_rv(s, primary_long_w, min_primary_obs))
        df["residual_rv_long_fallback"] = df.groupby("bond_code", group_keys=False)["residual"].apply(
            lambda s: rolling_rv(s, fallback_long_w, min_fallback_obs)
        )
        df["stock_vol_ratio"] = np.log((df["stock_rv_short"] + eps) / (df["stock_rv_long"] + eps))
        df["residual_vol_ratio"] = np.log((df["residual_rv_short"] + eps) / (df["residual_rv_long"] + eps))
        df["residual_vol_ratio_fallback"] = np.log((df["residual_rv_short"] + eps) / (df["residual_rv_long_fallback"] + eps))
        df["stock_vol_ratio_z"], stock_vol_fallback = same_slot_robust_z(
            df, "stock_vol_ratio", ["bond_code"], "slot_id", same_slot_samples, min_same_slot, fallback_window, eps, clip
        )
        df["residual_vol_ratio_z"], vol_fallback = same_slot_robust_z(
            df, "residual_vol_ratio", ["bond_code"], "slot_id", same_slot_samples, min_same_slot, fallback_window, eps, clip
        )
        df["residual_vol_ratio_fallback_z"], residual_vol_fallback_used = same_slot_robust_z(
            df, "residual_vol_ratio_fallback", ["bond_code"], "slot_id", same_slot_samples, min_same_slot, fallback_window, eps, clip
        )
        df["residual_vol_ratio_source"] = np.where(df["residual_vol_ratio_z"].notna(), "primary", "unavailable")
        df["residual_vol_ratio_fallback_source"] = np.where(df["residual_vol_ratio_fallback_z"].notna(), "fallback", "unavailable")
        df["fallback_vol_available"] = df["residual_vol_ratio_fallback_z"].notna()
        df["factor_vol_mr"] = -clip_zscore(df["residual_vol_ratio_z"], clip)
        df["factor_vol_mr_fallback"] = -clip_zscore(df["residual_vol_ratio_fallback_z"], clip)

        # Empirical option proxies.
        df["delta_bond_price"] = df.groupby("bond_code")["cb_close"].diff()
        df["delta_parity"] = df.groupby("bond_code")["parity"].diff()
        delta_parts = self._map_bond_groups(
            df,
            _empirical_delta_group_worker,
            residual_cfg,
            performance_cfg,
            task_name="empirical_delta",
        )
        df["empirical_delta_proxy"] = pd.concat(delta_parts).sort_index() if delta_parts else np.nan
        df["empirical_delta_proxy_clipped"] = df["empirical_delta_proxy"].clip(-5, 5)
        df["elasticity_proxy"] = df["empirical_delta_proxy"] * df["parity"] / df["cb_close"]
        df["empirical_gamma_proxy"] = np.nan
        df["convexity_asymmetry_proxy"] = self._convexity_asymmetry(df, residual_cfg)
        df["volatility_response_gap"] = df["elasticity_proxy"].abs() * df["stock_rv_short"] - df["cb_rv_short"]
        df["volatility_response_gap_z"], response_fallback = same_slot_robust_z(
            df,
            "volatility_response_gap",
            ["bond_code"],
            "slot_id",
            same_slot_samples,
            min_same_slot,
            fallback_window,
            eps,
            clip,
        )
        df["factor_vol_response_gap"] = clip_zscore(df["volatility_response_gap_z"], clip)

        # Flow proxies.
        cb_direction = np.sign(df["cb_close"] - df["cb_open"])
        cb_direction = cb_direction.mask(cb_direction == 0, np.sign(df.groupby("bond_code")["cb_close"].diff()))
        stock_direction = np.sign(df["stock_close"] - df["stock_open"])
        stock_direction = stock_direction.mask(stock_direction == 0, np.sign(df.groupby("bond_code")["stock_close"].diff()))
        df["cb_signed_amount"] = cb_direction.fillna(0.0) * df["amount_used"]
        df["stock_signed_amount"] = stock_direction.fillna(0.0) * df["stock_amount_used"]
        df["cb_flow_30m"] = self._group_rolling_ratio(df, "cb_signed_amount", "amount_used", short_w, eps)
        df["stock_flow_30m"] = self._group_rolling_ratio(df, "stock_signed_amount", "stock_amount_used", short_w, eps)
        df["cb_flow_z"], cb_flow_fallback = same_slot_robust_z(
            df, "cb_flow_30m", ["bond_code"], "slot_id", same_slot_samples, min_same_slot, fallback_window, eps, clip
        )
        df["stock_flow_z"], stock_flow_fallback = same_slot_robust_z(
            df, "stock_flow_30m", ["bond_code"], "slot_id", same_slot_samples, min_same_slot, fallback_window, eps, clip
        )
        df["flow_confirmation"] = 0.5 * df["cb_flow_z"] + 0.5 * df["stock_flow_z"]
        df["flow_lead"] = df["cb_flow_z"] - df["stock_flow_z"]

        df = df.copy()

        # Liquidity proxies.
        df["median_bar_amount_20d"] = df.groupby(["bond_code", "slot_id"], group_keys=False)["amount_used"].apply(
            lambda s: s.shift(1).rolling(same_slot_samples, min_periods=max(3, min_same_slot // 2)).median()
        )
        fallback_median = df.groupby("bond_code", group_keys=False)["amount_used"].apply(
            lambda s: s.shift(1).rolling(fallback_window, min_periods=3).median()
        )
        df["median_bar_amount_20d"] = df["median_bar_amount_20d"].combine_first(fallback_median)
        df["relative_amount"] = df["amount_used"] / (df["median_bar_amount_20d"] + eps)
        market_cfg = self.config.get("market_regime", {})
        min_market_count = int(market_cfg.get("min_active_count", residual_cfg.get("min_market_count", 5)))
        regime_stats = df.groupby("bar_start").agg(
            cb_market_mom_30m=("cb_mom_30m", "median"),
            cb_market_breadth_30m=("cb_mom_30m", lambda values: float((values.dropna() > 0).mean()) if values.notna().any() else np.nan),
            cb_market_breadth_60m=("cb_mom_60m", lambda values: float((values.dropna() > 0).mean()) if values.notna().any() else np.nan),
            cb_market_amount_burst=("relative_amount", "median"),
            stock_market_breadth_30m=("stock_mom_30m", lambda values: float((values.dropna() > 0).mean()) if values.notna().any() else np.nan),
            market_count=("bond_code", "count"),
        )
        df = df.merge(regime_stats, on="bar_start", how="left")
        risk_on_breadth = float(market_cfg.get("risk_on_breadth_30m", 0.60))
        risk_off_breadth = float(market_cfg.get("risk_off_breadth_30m", 0.30))
        thin_amount = float(market_cfg.get("thin_market_relative_amount", 0.30))
        df["market_regime_intraday"] = "NEUTRAL"
        df.loc[
            (df["market_count"] < min_market_count) | (df["cb_market_amount_burst"].notna() & (df["cb_market_amount_burst"] < thin_amount)),
            "market_regime_intraday",
        ] = "THIN_MARKET"
        df.loc[
            (df["market_regime_intraday"] == "NEUTRAL")
            & (df["cb_market_breadth_30m"] >= risk_on_breadth)
            & (df["cb_market_mom_30m"] > 0),
            "market_regime_intraday",
        ] = "RISK_ON"
        df.loc[
            (df["market_regime_intraday"] == "NEUTRAL")
            & (df["cb_market_breadth_30m"] <= risk_off_breadth)
            & (df["cb_market_mom_30m"] < 0),
            "market_regime_intraday",
        ] = "RISK_OFF"
        df["active_bar"] = df["amount_used"].notna() & (df["amount_used"] > 0)
        df["active_bar_ratio"] = df.groupby("bond_code", group_keys=False)["active_bar"].apply(
            lambda s: s.shift(1).rolling(min(expected_slots_per_day(), fallback_window), min_periods=3).mean()
        )
        df["zero_or_missing_bar_ratio"] = 1.0 - df["active_bar_ratio"]
        df["amihud"] = self._group_amihud(df, fallback_window, eps)
        df["high_low_range_proxy"] = (df["cb_high"] - df["cb_low"]) / df["cb_close"]
        df["high_low_range_proxy_z"], range_fallback = same_slot_robust_z(
            df,
            "high_low_range_proxy",
            ["bond_code"],
            "slot_id",
            same_slot_samples,
            min_same_slot,
            fallback_window,
            eps,
            clip,
        )
        df["high_low_range_proxy_z_source"] = np.select(
            [df["high_low_range_proxy_z"].notna() & range_fallback, df["high_low_range_proxy_z"].notna()],
            ["fallback", "primary"],
            default="unavailable",
        )
        vol_gate_cfg = self.config["volatility_gate"]
        vol_z_max = float(vol_gate_cfg["residual_vol_ratio_z_max"])
        range_z_max = float(vol_gate_cfg["high_low_range_proxy_z_max"])
        df["missing_vol_gate"] = df["residual_vol_ratio_z"].isna()
        df["missing_range_gate"] = df["high_low_range_proxy_z"].isna()
        residual_vol_gate = df["residual_vol_ratio_z"].isna() | (df["residual_vol_ratio_z"] <= vol_z_max)
        range_gate = df["high_low_range_proxy_z"].isna() | (df["high_low_range_proxy_z"] <= range_z_max)
        df["volatility_gate_pass"] = residual_vol_gate & range_gate
        df["noise_penalty"] = (
            (df["residual_vol_ratio_z"] - vol_z_max).clip(lower=0).fillna(0.0)
            + (df["high_low_range_proxy_z"] - range_z_max).clip(lower=0).fillna(0.0)
        )
        df["liquidity_pass"] = self._liquidity_pass(df)

        df["data_quality_pass"] = (
            (~df["missing_cb_bar"])
            & (~df["missing_stock_bar"])
            & (~df["stale_price_flag"])
            & df["conversion_price"].notna()
            & df["parity"].notna()
        )
        df["insufficient_history_flag"] = df["residual_model_nobs"].fillna(0) < int(residual_cfg["min_observations"])
        df["fallback_standardization_flag"] = (
            residual_fallback
            | premium_fallback
            | vol_fallback
            | response_fallback
            | cb_flow_fallback
            | stock_flow_fallback
            | residual_vol_fallback_used
            | range_fallback
            | impulse_fallback_any
            | cb_mom_fallback_any
            | premium_change_fallback_any
        )

        df = df.copy()
        self._compose_signal(df)

        missing_result_columns = {col: pd.Series(np.nan, index=df.index) for col in RESULT_COLUMNS if col not in df.columns}
        if missing_result_columns:
            df = pd.concat([df, pd.DataFrame(missing_result_columns, index=df.index)], axis=1)
        return df.loc[:, RESULT_COLUMNS].sort_values(["bar_start", "bond_code"]).reset_index(drop=True)

    @staticmethod
    def _map_bond_groups(
        df: pd.DataFrame,
        worker: Any,
        worker_config: dict[str, Any],
        performance_cfg: dict[str, Any],
        task_name: str,
    ) -> list[Any]:
        groups = [group for _, group in df.groupby("bond_code", sort=False)]
        if not groups:
            return []

        enabled = bool(performance_cfg.get("parallel_by_bond", True))
        min_rows = int(performance_cfg.get("min_rows_for_parallel", 5000))
        disabled_tasks = set(performance_cfg.get("disable_parallel_tasks", []) or [])
        worker_count = _resolve_worker_count(performance_cfg.get("max_workers"), len(groups))
        should_parallel = enabled and task_name not in disabled_tasks and worker_count > 1 and len(df) >= min_rows
        payloads = [(group, worker_config) for group in groups]
        if not should_parallel:
            return [worker(payload) for payload in payloads]

        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                return list(executor.map(worker, payloads))
        except Exception as exc:
            LOGGER.warning("Parallel %s failed; falling back to serial. error=%s", task_name, exc)
            return [worker(payload) for payload in payloads]

    def _convexity_asymmetry(self, df: pd.DataFrame, residual_cfg: dict[str, Any]) -> pd.Series:
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        min_obs = max(8, int(residual_cfg["min_observations"]) // 3)
        window = int(residual_cfg["window_bars"])

        def rolling_conditional_slope(group: pd.DataFrame, mask: pd.Series) -> np.ndarray:
            y = group["delta_bond_price"].to_numpy(dtype=float, copy=False)
            x = group["delta_parity"].to_numpy(dtype=float, copy=False)
            valid = mask.to_numpy(dtype=bool, copy=False) & np.isfinite(y) & np.isfinite(x)
            x_clean = np.where(valid, x, 0.0)
            y_clean = np.where(valid, y, 0.0)
            count_prefix = np.concatenate([[0.0], np.cumsum(valid.astype(float))])
            x_prefix = np.concatenate([[0.0], np.cumsum(x_clean)])
            y_prefix = np.concatenate([[0.0], np.cumsum(y_clean)])
            x2_prefix = np.concatenate([[0.0], np.cumsum(x_clean * x_clean)])
            xy_prefix = np.concatenate([[0.0], np.cumsum(x_clean * y_clean)])
            slope = np.full(len(group), np.nan, dtype=float)
            for pos in range(len(group)):
                start = max(0, pos - window)
                nobs = int(count_prefix[pos] - count_prefix[start])
                if nobs < min_obs:
                    continue
                sum_x = x_prefix[pos] - x_prefix[start]
                sum_y = y_prefix[pos] - y_prefix[start]
                sum_x2 = x2_prefix[pos] - x2_prefix[start]
                sum_xy = xy_prefix[pos] - xy_prefix[start]
                denom = sum_x2 - (sum_x * sum_x / nobs)
                if denom <= 0:
                    continue
                slope[pos] = (sum_xy - (sum_x * sum_y / nobs)) / denom
            return slope

        for _, group in df.groupby("bond_code", sort=False):
            up_beta = rolling_conditional_slope(group, group["delta_parity"] > 0)
            down_beta = rolling_conditional_slope(group, group["delta_parity"] < 0)
            out.loc[group.index] = up_beta - down_beta
        return out

    @staticmethod
    def _slope(y: pd.Series, x: pd.Series) -> float:
        valid = y.notna() & x.notna()
        if valid.sum() < 2:
            return np.nan
        xv = x[valid].to_numpy(dtype=float)
        yv = y[valid].to_numpy(dtype=float)
        denom = float(((xv - xv.mean()) ** 2).sum())
        if denom <= 0:
            return np.nan
        return float(((xv - xv.mean()) * (yv - yv.mean())).sum() / denom)

    @staticmethod
    def _group_rolling_ratio(df: pd.DataFrame, numerator_col: str, denominator_col: str, window: int, eps: float) -> pd.Series:
        grouped = df.groupby("bond_code", sort=False)
        numerator = grouped[numerator_col].transform(lambda s: rolling_sum(s, window, 2))
        denominator = grouped[denominator_col].transform(lambda s: rolling_sum(s, window, 2))
        return numerator / (denominator + eps)

    @staticmethod
    def _group_amihud(df: pd.DataFrame, window: int, eps: float) -> pd.Series:
        raw = df["r_cb"].abs() / (df["amount_used"] + eps)
        return raw.groupby(df["bond_code"], sort=False).transform(lambda s: s.shift(1).rolling(window, min_periods=3).median())

    def _liquidity_pass(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.config["liquidity"]
        passed = pd.Series(True, index=df.index)
        rules = [
            ("adv20_amount", cfg.get("min_adv20_amount"), ">="),
            ("amount_used", cfg.get("min_current_bar_amount"), ">="),
            ("relative_amount", cfg.get("min_relative_amount"), ">="),
            ("active_bar_ratio", cfg.get("min_active_bar_ratio"), ">="),
            ("zero_or_missing_bar_ratio", cfg.get("max_zero_bar_ratio"), "<="),
            ("amihud", cfg.get("max_amihud"), "<="),
            ("high_low_range_proxy", cfg.get("max_high_low_range_proxy"), "<="),
        ]
        for col, threshold, op in rules:
            if threshold is None or isinstance(threshold, dict):
                continue
            if op == ">=":
                passed &= df[col].isna() | (df[col] >= float(threshold))
            else:
                passed &= df[col].isna() | (df[col] <= float(threshold))
        return passed

    @staticmethod
    def _ge(series: pd.Series, threshold: float) -> pd.Series:
        return series.notna() & (series >= float(threshold))

    @staticmethod
    def _le(series: pd.Series, threshold: float) -> pd.Series:
        return series.notna() & (series <= float(threshold))

    @staticmethod
    def _between(series: pd.Series, low: float, high: float) -> pd.Series:
        return series.notna() & (series >= float(low)) & (series <= float(high))

    @staticmethod
    def _positive_score(series: pd.Series, threshold: float, higher_is_better: bool = True) -> pd.Series:
        if higher_is_better:
            value = series / (abs(float(threshold)) if not np.isclose(threshold, 0.0) else 1.0)
        else:
            value = (-series) / (abs(float(threshold)) if not np.isclose(threshold, 0.0) else 1.0)
        return value.clip(lower=0.0, upper=2.0).fillna(0.0)

    @staticmethod
    def _parse_time(value: str) -> tuple[int, int, int]:
        parts = [int(item) for item in str(value).split(":")]
        while len(parts) < 3:
            parts.append(0)
        return parts[0], parts[1], parts[2]

    @classmethod
    def _after_time(cls, values: pd.Series, hhmmss: str) -> pd.Series:
        hour, minute, second = cls._parse_time(hhmmss)
        cutoff = hour * 3600 + minute * 60 + second
        times = pd.to_datetime(values, errors="coerce")
        seconds = times.dt.hour * 3600 + times.dt.minute * 60 + times.dt.second
        return seconds > cutoff

    def _action_reason_text(self, row: pd.Series) -> str:
        if row.get("signal_type") == "LAG_REPAIR_LONG":
            return (
                f"正股30m动量强({row.get('stock_mom_30m_z', np.nan):.2f})，"
                f"转债残差滞涨({row.get('residual_z', np.nan):.2f})，"
                f"成交额为同slot中位数{row.get('relative_amount', np.nan):.2f}倍，溢价未过热。"
            )
        if row.get("signal_type") == "MOMENTUM_BREAKOUT_LONG":
            return (
                f"正股30m动量强({row.get('stock_mom_30m_z', np.nan):.2f})，"
                f"转债15m动量同步转强({row.get('cb_mom_15m_z', np.nan):.2f})，"
                f"成交额放大到同slot中位数{row.get('relative_amount', np.nan):.2f}倍，残差尚未明显透支。"
            )
        if row.get("signal_type") == "ARMED_LAG_LONG":
            return (
                f"接近滞涨补涨触发：正股30m动量{row.get('stock_mom_30m_z', np.nan):.2f}，"
                f"残差{row.get('residual_z', np.nan):.2f}，成交额{row.get('relative_amount', np.nan):.2f}倍；等待成交/残差修复确认。"
            )
        if row.get("signal_type") == "ARMED_BREAKOUT_LONG":
            return (
                f"接近共振突破触发：正股脉冲{row.get('stock_impulse_score', np.nan):.2f}，"
                f"转债15m动量{row.get('cb_mom_15m_z', np.nan):.2f}，成交额{row.get('relative_amount', np.nan):.2f}倍；等待突破强度确认。"
            )
        if row.get("signal_type") == "WATCH_LONG":
            subtype = row.get("watch_signal_type") or "WATCH_LONG"
            missing = row.get("watch_missing_to_action") or ""
            return f"{subtype}：仅作为监控面板候选。{missing}"
        if row.get("exit_state") != "HOLD":
            return f"{row.get('exit_type') or row.get('exit_state')}：提示人工复核止盈/退出。"
        return ""

    def _watch_missing_to_action(self, row: pd.Series) -> str:
        subtype = str(row.get("watch_signal_type") or "")
        if not subtype:
            return ""

        def value(name: str) -> float:
            item = row.get(name)
            return float(item) if item is not None and pd.notna(item) else np.nan

        trigger_cfg = self.config["trigger_thresholds"]
        missing: list[str] = []
        if subtype == "WATCH_RISKY":
            return "风险点：溢价/残差/噪声或流动性存在追涨风险，不应默认下单。"
        if subtype in {"WATCH_LAG", "WATCH_SETUP"}:
            checks = [
                ("正股30m动量", value("stock_mom_30m_z"), ">=", float(trigger_cfg["lag_stock_mom_30m_z"])),
                ("残差滞涨", value("residual_z"), "<=", float(trigger_cfg["lag_residual_z_max"])),
                ("成交额确认", value("relative_amount"), ">=", float(trigger_cfg["lag_relative_amount_min"])),
                ("资金确认", value("flow_confirmation"), ">=", float(trigger_cfg["lag_flow_confirmation_min"])),
                ("溢价不过热", value("premium_slot_z"), "<=", float(trigger_cfg["lag_premium_slot_z_max"])),
            ]
        else:
            checks = [
                ("正股30m动量", value("stock_mom_30m_z"), ">=", float(trigger_cfg["breakout_stock_mom_30m_z"])),
                ("转债15m动量", value("cb_mom_15m_z"), ">=", float(trigger_cfg["breakout_cb_mom_15m_z"])),
                ("成交额放大", value("relative_amount"), ">=", float(trigger_cfg["breakout_relative_amount_min"])),
                ("转债资金确认", value("cb_flow_z"), ">=", float(trigger_cfg["breakout_cb_flow_z_min"])),
                ("残差未透支", value("residual_z"), "<=", float(trigger_cfg["breakout_residual_z_max"])),
            ]
        for label, item, op, threshold in checks:
            if pd.isna(item) or (op == ">=" and item < threshold) or (op == "<=" and item > threshold):
                missing.append(f"{label}{item:.2f}未达{op}{threshold:.2f}" if pd.notna(item) else f"{label}缺失")
        return "差一步：" + "，".join(missing[:4]) if missing else "条件接近 ACTION，等待确认。"

    def _risk_hint_text(self, row: pd.Series) -> str:
        hints = []
        if bool(row.get("missing_vol_gate")):
            hints.append("主波动率门缺失，已按不否决处理")
        if bool(row.get("missing_range_gate")):
            hints.append("高低价噪声门缺失")
        if row.get("premium_slot_z") is not None and pd.notna(row.get("premium_slot_z")) and row.get("premium_slot_z") > 1.0:
            hints.append("溢价偏热")
        if row.get("noise_penalty") is not None and pd.notna(row.get("noise_penalty")) and row.get("noise_penalty") > 0:
            hints.append(f"噪声惩罚{row.get('noise_penalty'):.2f}")
        if row.get("premium_change_10m_z") is not None and pd.notna(row.get("premium_change_10m_z")) and row.get("premium_change_10m_z") > 1.0:
            hints.append(f"10m溢价抬升偏快({row.get('premium_change_10m_z'):.2f})")
        if row.get("cb_vwap_dev") is not None and pd.notna(row.get("cb_vwap_dev")) and row.get("cb_vwap_dev") > 0.01 and (
            row.get("stock_vwap_dev") is None or pd.isna(row.get("stock_vwap_dev")) or row.get("stock_vwap_dev") <= 0
        ):
            hints.append("转债高于VWAP但正股未站稳VWAP")
        if row.get("market_regime_intraday") in {"RISK_OFF", "THIN_MARKET"}:
            hints.append(f"市场环境{row.get('market_regime_intraday')}")
        return "；".join(hints)

    def _compose_signal(self, df: pd.DataFrame) -> None:
        if "pool_state" not in df.columns:
            df["pool_state"] = "ACTIVE"
        df["pool_state"] = df["pool_state"].apply(lambda value: normalize_pool_state(value))
        if "session_bar_role" not in df.columns:
            df["session_bar_role"] = df["bar_start"].apply(session_bar_role)
        required_numeric = [
            "factor_residual_mr",
            "factor_premium_mr",
            "stock_momentum_score",
            "factor_vol_response_gap",
            "flow_confirmation",
            "factor_vol_mr",
            "stock_mom_5m_z",
            "stock_mom_10m_z",
            "stock_mom_15m_z",
            "stock_mom_30m_z",
            "stock_mom_60m_z",
            "stock_impulse_score",
            "cb_mom_5m_z",
            "cb_mom_10m_z",
            "cb_mom_15m_z",
            "cb_mom_30m_z",
            "cb_flow_z",
            "residual_z",
            "residual_z_change_1bar",
            "residual_z_change_2bar",
            "premium_slot_z",
            "premium_change_10m_z",
            "premium_change_30m_z",
            "relative_amount",
            "high_low_range_proxy_z",
            "residual_vol_ratio_z",
            "noise_penalty",
            "vwap_gap",
            "cb_vwap_dev",
            "stock_vwap_dev",
        ]
        missing_numeric = {col: pd.Series(np.nan, index=df.index, dtype="float64") for col in required_numeric if col not in df.columns}
        missing_bool = {
            col: pd.Series(False, index=df.index, dtype="bool")
            for col in ["data_quality_pass", "liquidity_pass", "volatility_gate_pass", "residual_repair_started"]
            if col not in df.columns
        }
        if missing_numeric or missing_bool:
            df_extra = pd.DataFrame({**missing_numeric, **missing_bool}, index=df.index)
            df[df_extra.columns] = df_extra

        weights = self.config["signal_weights"]
        thresholds = self.config["signal_thresholds"]
        weighted_sum = pd.Series(0.0, index=df.index)
        active_weight = pd.Series(0.0, index=df.index)
        for col, weight in weights.items():
            valid = df[col].notna()
            weighted_sum.loc[valid] += df.loc[valid, col] * float(weight)
            active_weight.loc[valid] += float(weight)
        total_weight = sum(float(value) for value in weights.values())
        df["factor_coverage"] = active_weight / total_weight
        df["factor_coverage_pass"] = df["factor_coverage"] >= float(thresholds["minimum_factor_coverage"])
        df["signal_score"] = weighted_sum / active_weight.replace(0, np.nan)
        df["trend_gate"] = df["stock_momentum_score"].notna() & (
            df["stock_momentum_score"] >= float(thresholds["trend_gate_min_stock_momentum"])
        )

        setup_weights = self.config["setup_score_weights"]
        setup_sum = pd.Series(0.0, index=df.index)
        setup_active = pd.Series(0.0, index=df.index)
        setup_values = {
            "stock_mom_30m_z": df["stock_mom_30m_z"],
            "stock_mom_60m_z": df["stock_mom_60m_z"],
            "stock_impulse_score": df["stock_impulse_score"],
            "factor_residual_mr": df["factor_residual_mr"],
            "factor_premium_mr": df["factor_premium_mr"],
            "flow_confirmation": df["flow_confirmation"],
            "relative_amount_log": np.log1p(df["relative_amount"].clip(lower=0)),
            "factor_vol_response_gap": df["factor_vol_response_gap"],
            "vwap_gap": df["vwap_gap"],
            "noise_penalty": df["noise_penalty"],
        }
        for col, weight in setup_weights.items():
            series = setup_values.get(col, pd.Series(np.nan, index=df.index, dtype="float64"))
            valid_series = series.notna()
            setup_sum.loc[valid_series] += series.loc[valid_series] * float(weight)
            setup_active.loc[valid_series] += abs(float(weight))
        df["setup_score"] = (setup_sum / setup_active.replace(0, np.nan)).clip(-3.0, 3.0)

        trigger_cfg = self.config["trigger_thresholds"]
        lag_trigger_score = (
            self._positive_score(df["stock_mom_30m_z"], trigger_cfg["lag_stock_mom_30m_z"])
            + self._positive_score(df["stock_mom_60m_z"], 0.5)
            + self._positive_score(df["residual_z"], trigger_cfg["lag_residual_z_max"], higher_is_better=False)
            + self._positive_score(df["relative_amount"], trigger_cfg["lag_relative_amount_min"])
            + self._positive_score(df["flow_confirmation"], max(trigger_cfg["lag_flow_confirmation_min"], 0.5))
        ) / 5.0
        breakout_trigger_score = (
            self._positive_score(df["stock_mom_30m_z"], trigger_cfg["breakout_stock_mom_30m_z"])
            + self._positive_score(df["cb_mom_15m_z"], trigger_cfg["breakout_cb_mom_15m_z"])
            + self._positive_score(df["relative_amount"], trigger_cfg["breakout_relative_amount_min"])
            + self._positive_score(df["cb_flow_z"], trigger_cfg["breakout_cb_flow_z_min"])
        ) / 4.0
        df["trigger_score"] = pd.concat([lag_trigger_score, breakout_trigger_score], axis=1).max(axis=1).clip(0.0, 3.0)

        exit_cfg = self.config["exit_thresholds"]
        exit_residual = self._positive_score(df["residual_z"], abs(float(exit_cfg["mean_reversion_complete_residual_z"])))
        exit_premium = self._positive_score(df["premium_slot_z"], float(exit_cfg["premium_overheat_slot_z"]))
        momentum_decay = self._positive_score(df["stock_mom_30m_z"], abs(float(exit_cfg["momentum_decay_stock_mom_30m_z"])), higher_is_better=False)
        df["exit_score"] = (0.5 * exit_residual + 0.25 * exit_premium + 0.25 * momentum_decay).clip(0.0, 3.0)
        df["exit_state"] = "HOLD"
        df["exit_type"] = "HOLD"
        residual_repaired = self._ge(df["residual_z"], float(exit_cfg["mean_reversion_complete_residual_z"])) & (df["signal_score"].fillna(0.0) <= 0.0)
        momentum_decay_short = (
            self._le(df["stock_mom_10m_z"], float(exit_cfg.get("momentum_decay_stock_mom_short_z", exit_cfg["momentum_decay_stock_mom_30m_z"])))
            | self._le(df["stock_mom_15m_z"], float(exit_cfg.get("momentum_decay_stock_mom_short_z", exit_cfg["momentum_decay_stock_mom_30m_z"])))
            | self._le(df["stock_mom_30m_z"], float(exit_cfg["momentum_decay_stock_mom_30m_z"]))
        )
        premium_overheat = (
            self._ge(df["premium_slot_z"], float(exit_cfg["premium_overheat_slot_z"]))
            | self._ge(df["premium_change_10m_z"], float(exit_cfg.get("premium_change_10m_z_overheat", 1.5)))
            | self._ge(df["premium_change_30m_z"], float(exit_cfg.get("premium_change_30m_z_overheat", 1.5)))
        )
        noise_spike = (
            self._ge(df["high_low_range_proxy_z"], float(exit_cfg.get("noise_spike_range_z", 2.0)))
            | self._ge(df["residual_vol_ratio_z"], float(exit_cfg.get("noise_spike_residual_vol_z", 1.2)))
        )
        end_of_day = df["session_bar_role"].astype(str).str.upper().eq("CLOSE")
        df.loc[residual_repaired, "exit_type"] = "EXIT_RESIDUAL_REPAIRED"
        df.loc[momentum_decay_short, "exit_type"] = "EXIT_MOMENTUM_DECAY"
        df.loc[premium_overheat, "exit_type"] = "EXIT_PREMIUM_OVERHEAT"
        df.loc[noise_spike, "exit_type"] = "EXIT_NOISE_SPIKE"
        df.loc[end_of_day, "exit_type"] = "EXIT_END_OF_DAY"
        df["exit_state"] = df["exit_type"]

        action_coverage = float(self.config["manual_alert_mode"]["min_factor_coverage_for_action"])
        action_base = (
            df["data_quality_pass"]
            & df["liquidity_pass"]
            & df["volatility_gate_pass"]
            & (df["factor_coverage"] >= action_coverage)
        )
        lag_vol_gate = df["residual_vol_ratio_z"].isna() | self._le(df["residual_vol_ratio_z"], float(trigger_cfg["lag_residual_vol_ratio_z_max"]))
        lag_repair = (
            action_base
            & self._ge(df["stock_mom_30m_z"], float(trigger_cfg["lag_stock_mom_30m_z"]))
            & self._ge(df["stock_mom_60m_z"], float(trigger_cfg["lag_stock_mom_60m_z"]))
            & self._le(df["residual_z"], float(trigger_cfg["lag_residual_z_max"]))
            & self._ge(df["relative_amount"], float(trigger_cfg["lag_relative_amount_min"]))
            & self._le(df["premium_slot_z"], float(trigger_cfg["lag_premium_slot_z_max"]))
            & self._ge(df["flow_confirmation"], float(trigger_cfg["lag_flow_confirmation_min"]))
            & lag_vol_gate
        )
        breakout = (
            action_base
            & self._ge(df["stock_mom_30m_z"], float(trigger_cfg["breakout_stock_mom_30m_z"]))
            & self._ge(df["cb_mom_15m_z"], float(trigger_cfg["breakout_cb_mom_15m_z"]))
            & self._ge(df["relative_amount"], float(trigger_cfg["breakout_relative_amount_min"]))
            & self._between(df["residual_z"], float(trigger_cfg["breakout_residual_z_min"]), float(trigger_cfg["breakout_residual_z_max"]))
            & self._le(df["premium_slot_z"], float(trigger_cfg["breakout_premium_slot_z_max"]))
            & self._ge(df["cb_flow_z"], float(trigger_cfg["breakout_cb_flow_z_min"]))
        )
        armed_cfg = self.config.get("armed_thresholds", {})
        armed_lag = (
            action_base
            & (~lag_repair)
            & self._ge(df["stock_mom_30m_z"], float(armed_cfg.get("lag_stock_mom_30m_z", 0.50)))
            & self._le(df["residual_z"], float(armed_cfg.get("lag_residual_z_max", -0.60)))
            & self._le(df["premium_slot_z"], float(armed_cfg.get("lag_premium_slot_z_max", 1.20)))
            & self._ge(df["relative_amount"], float(armed_cfg.get("lag_relative_amount_min", 0.40)))
            & self._ge(df["flow_confirmation"], float(armed_cfg.get("lag_flow_confirmation_min", -0.20)))
        )
        armed_breakout = (
            action_base
            & (~breakout)
            & self._ge(df["stock_mom_30m_z"], float(armed_cfg.get("breakout_stock_mom_30m_z", 0.80)))
            & self._ge(df["cb_mom_15m_z"], float(armed_cfg.get("breakout_cb_mom_15m_z", 0.20)))
            & self._le(df["residual_z"], float(armed_cfg.get("breakout_residual_z_max", 1.00)))
            & self._ge(df["relative_amount"], float(armed_cfg.get("breakout_relative_amount_min", 0.70)))
            & self._ge(df["cb_flow_z"], float(armed_cfg.get("breakout_cb_flow_z_min", 0.00)))
        )
        watch_cfg = self.config.get("watch_thresholds", {})
        watch_base = (
            df["data_quality_pass"]
            & df["liquidity_pass"]
            & (df["factor_coverage"] >= float(watch_cfg.get("minimum_factor_coverage", thresholds["minimum_factor_coverage"])))
        )
        watch_risky = watch_base & (
            self._ge(df["premium_slot_z"], float(watch_cfg.get("risky_premium_slot_z_min", 1.50)))
            | self._ge(df["premium_change_10m_z"], float(watch_cfg.get("risky_premium_change_10m_z_min", 1.50)))
            | self._ge(df["premium_change_30m_z"], float(watch_cfg.get("risky_premium_change_30m_z_min", 1.50)))
            | self._ge(df["residual_z"], float(watch_cfg.get("risky_residual_z_min", 0.80)))
            | self._ge(df["high_low_range_proxy_z"], float(watch_cfg.get("risky_high_low_range_proxy_z_min", 1.50)))
            | self._ge(df["residual_vol_ratio_z"], float(watch_cfg.get("risky_residual_vol_ratio_z_min", 0.80)))
            | (~df["volatility_gate_pass"])
        )
        watch_lag = (
            watch_base
            & (~watch_risky)
            & self._ge(df["stock_mom_30m_z"], float(trigger_cfg["watch_stock_mom_30m_z_min"]))
            & self._le(df["residual_z"], float(trigger_cfg["watch_residual_z_max"]))
            & self._le(df["premium_slot_z"], float(trigger_cfg["watch_premium_slot_z_max"]))
        )
        watch_momentum = (
            watch_base
            & (~watch_risky)
            & self._ge(df["stock_mom_30m_z"], float(watch_cfg.get("momentum_stock_mom_30m_z_min", 0.0)))
            & self._ge(df["cb_mom_15m_z"], float(watch_cfg.get("momentum_cb_mom_15m_z_min", 0.0)))
            & self._le(df["residual_z"], float(watch_cfg.get("momentum_residual_z_max", 0.80)))
            & self._le(df["premium_slot_z"], float(watch_cfg.get("momentum_premium_slot_z_max", 1.50)))
        )
        watch_setup = (
            watch_base
            & (~watch_risky)
            & self._le(df["premium_slot_z"], float(watch_cfg.get("setup_premium_slot_z_max", 1.20)))
            & self._le(df["residual_z"], float(watch_cfg.get("setup_residual_z_max", 0.80)))
        )
        any_watch = watch_setup | watch_lag | watch_momentum | watch_risky

        signal_type = pd.Series("NONE", index=df.index, dtype="object")
        signal_type.loc[any_watch] = "WATCH_LONG"
        signal_type.loc[armed_lag] = "ARMED_LAG_LONG"
        signal_type.loc[armed_breakout] = "ARMED_BREAKOUT_LONG"
        signal_type.loc[lag_repair] = "LAG_REPAIR_LONG"
        signal_type.loc[breakout] = "MOMENTUM_BREAKOUT_LONG"
        signal_type.loc[(signal_type == "NONE") & (df["exit_state"] != "HOLD")] = "MEAN_REVERSION_COMPLETE"
        signal_type.loc[~df["data_quality_pass"]] = "INVALID"
        signal_type.loc[df["data_quality_pass"] & ~df["liquidity_pass"]] = "ILLIQUID"
        df["signal_type"] = signal_type
        watch_signal_type = pd.Series("", index=df.index, dtype="object")
        watch_signal_type.loc[watch_setup] = "WATCH_SETUP"
        watch_signal_type.loc[watch_lag] = "WATCH_LAG"
        watch_signal_type.loc[watch_momentum] = "WATCH_MOMENTUM"
        watch_signal_type.loc[watch_risky] = "WATCH_RISKY"
        watch_signal_type.loc[df["signal_type"].isin(["LAG_REPAIR_LONG", "MOMENTUM_BREAKOUT_LONG", "ARMED_LAG_LONG", "ARMED_BREAKOUT_LONG"])] = ""
        df["watch_signal_type"] = watch_signal_type
        df["watch_missing_to_action"] = df.apply(self._watch_missing_to_action, axis=1)

        alert_state = pd.Series("INFO_ONLY", index=df.index, dtype="object")
        alert_state.loc[df["signal_type"] == "WATCH_LONG"] = "WATCH_ONLY"
        alert_state.loc[df["exit_state"] != "HOLD"] = "EXIT_HINT"
        alert_state.loc[~df["data_quality_pass"]] = "INVALID"
        alert_state.loc[df["data_quality_pass"] & ~df["liquidity_pass"]] = "ILLIQUID"

        action_types = set(self.config["manual_alert_mode"]["action_signal_types"])
        armed_types = set(self.config.get("alert_controls", {}).get("armed_signal_types", []))
        raw_action = df["signal_type"].isin(action_types)
        raw_armed = df["signal_type"].isin(armed_types)
        controls = self.config["alert_controls"]
        no_new_after = str(controls.get("no_new_action_after", self.config["manual_alert_mode"]["no_new_action_after"]))
        pool_allowed = df["pool_state"].isin(set(controls.get("actionable_pool_states", DEFAULT_ACTIONABLE_POOL_STATES)))
        close_bar = df["session_bar_role"].astype(str).str.upper().eq("CLOSE")
        after_no_new = self._after_time(df["bar_start"], no_new_after)
        df["is_actionable_bar"] = (~close_bar) & (~after_no_new) & pool_allowed
        df["action_blocked_by_time"] = raw_action & after_no_new
        df["no_new_action_after_suppressed"] = raw_action & after_no_new
        action_allowed = raw_action & df["is_actionable_bar"]
        df["cooldown_suppressed"] = False
        df["suppression_reason"] = ""
        df.loc[raw_action & close_bar, "suppression_reason"] = "close_bar_not_actionable"
        df.loc[raw_action & after_no_new, "suppression_reason"] = "no_new_action_after"
        df.loc[raw_action & ~pool_allowed, "suppression_reason"] = "pool_state_not_actionable"
        df.loc[raw_armed & ~pool_allowed, "suppression_reason"] = "pool_state_not_actionable"
        df.loc[raw_armed & close_bar, "suppression_reason"] = "close_bar_not_actionable"
        df["last_action_reason"] = ""
        df["alert_expire_time"] = pd.NaT

        per_bond_last: dict[tuple[str, str], pd.Timestamp] = {}
        per_bond_count: dict[tuple[str, str], int] = {}
        total_by_day: dict[str, int] = {}
        min_gap = pd.Timedelta(minutes=float(controls["min_minutes_between_action_alerts_per_bond"]))
        max_per_bond = int(controls["max_action_alerts_per_bond_per_day"])
        max_total = int(controls["max_total_action_alerts_per_day"])
        expire_delta = pd.Timedelta(minutes=int(controls["expire_action_after_bars"]) * int(self.config["bar_minutes"]))

        for row_idx in df.sort_values(["bar_start", "bond_code"]).index:
            if not bool(action_allowed.loc[row_idx]):
                continue
            bond_code = str(df.at[row_idx, "bond_code"])
            bar_start = pd.Timestamp(df.at[row_idx, "bar_start"]).tz_localize(None)
            trade_date = bar_start.strftime("%Y-%m-%d")
            key = (trade_date, bond_code)
            last_ts = per_bond_last.get(key)
            bond_count = per_bond_count.get(key, 0)
            total_count = total_by_day.get(trade_date, 0)
            suppressed = (
                (last_ts is not None and bar_start - last_ts < min_gap)
                or bond_count >= max_per_bond
                or total_count >= max_total
            )
            if suppressed:
                df.at[row_idx, "cooldown_suppressed"] = True
                df.at[row_idx, "suppression_reason"] = "cooldown_or_daily_limit"
                continue
            alert_state.loc[row_idx] = "ACTION_LONG"
            per_bond_last[key] = bar_start
            per_bond_count[key] = bond_count + 1
            total_by_day[trade_date] = total_count + 1
            df.at[row_idx, "alert_expire_time"] = bar_start + expire_delta

        alert_state.loc[raw_action & df["cooldown_suppressed"]] = "WATCH_ONLY"
        alert_state.loc[raw_action & ~df["is_actionable_bar"]] = "INFO_ONLY"
        alert_state.loc[raw_armed & (~raw_action) & df["is_actionable_bar"]] = "ARMED_LONG"
        alert_state.loc[~df["data_quality_pass"]] = "INVALID"
        alert_state.loc[df["data_quality_pass"] & ~df["liquidity_pass"]] = "ILLIQUID"
        df["alert_state"] = alert_state
        df["signal_state"] = df["signal_type"]
        df["action_reason_text"] = df.apply(self._action_reason_text, axis=1)
        df["risk_hint_text"] = df.apply(self._risk_hint_text, axis=1)
        df.loc[raw_action & df["cooldown_suppressed"], "last_action_reason"] = df.loc[
            raw_action & df["cooldown_suppressed"], "action_reason_text"
        ]

    def compute_from_db(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        asof: str | pd.Timestamp | None = None,
        write: bool = False,
    ) -> pd.DataFrame:
        conn = get_connection()
        try:
            panel = self.adapter.build_panel(conn, start=start, end=end, asof=asof)
            if panel.empty:
                LOGGER.warning("No panel bars available start=%s end=%s asof=%s", start, end, asof)
                return pd.DataFrame(columns=RESULT_COLUMNS)
            daily_amounts = self.adapter.load_daily_amounts(
                conn,
                sorted(panel["bond_code"].dropna().astype(str).unique()),
                end=panel["bar_start"].max(),
                lookback_days=int(self.config["liquidity"]["adv20_days"]),
            )
            result = self.compute_panel_factors(panel, daily_amounts=daily_amounts)
            if write:
                inserted = insert_factor_rows(conn, result, self.schema["tables"]["factor_snapshot"], self.factor_version)
                alert_inserted = insert_alert_rows(conn, result, self.schema["tables"]["alert_table"], self.factor_version)
                LOGGER.info("Inserted factor rows=%s alert rows=%s factor_version=%s", inserted, alert_inserted, self.factor_version)
            return result
        finally:
            conn.close()
