from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "intraday_factors.yaml"
DEFAULT_SCHEMA_PATH = PROJECT_DIR / "config" / "data_schema.yaml"


DEFAULT_CONFIG: dict[str, Any] = {
    "factor_version": "v0.3",
    "bar_minutes": 5,
    "same_slot_lookback_days": 20,
    "minimum_same_slot_samples": 10,
    "fallback_rolling_window": 120,
    "clip_zscore": 3.0,
    "epsilon": 1e-9,
    "performance": {
        "parallel_by_bond": True,
        "max_workers": 0,
        "min_rows_for_parallel": 5000,
        "disable_parallel_tasks": [],
    },
    "residual_model": {
        "window_bars": 960,
        "min_observations": 30,
        "ridge_alpha": 1e-4,
        "min_market_count": 5,
        "residual_ewm_span": 6,
    },
    "momentum": {"bars_30m": 6, "bars_60m": 12, "bars_1d": 48, "bars_5d": 240},
    "volatility": {
        "short_window_bars": 6,
        "primary_long_window_bars": 240,
        "fallback_long_window_bars": 48,
        "min_primary_obs": 180,
        "min_fallback_obs": 36,
        "use_fallback_in_composite_score": False,
        "expose_fallback_fields": True,
    },
    "liquidity": {
        "adv20_days": 20,
        "min_adv20_amount": None,
        "min_current_bar_amount": 0.0,
        "min_relative_amount": None,
        "min_active_bar_ratio": None,
        "max_zero_bar_ratio": None,
        "max_amihud": None,
        "max_high_low_range_proxy": None,
    },
    "signal_weights": {
        "factor_residual_mr": 0.35,
        "factor_premium_mr": 0.20,
        "stock_momentum_score": 0.20,
        "factor_vol_response_gap": 0.10,
        "flow_confirmation": 0.10,
        "factor_vol_mr": 0.05,
    },
    "signal_thresholds": {
        "minimum_factor_coverage": 0.50,
        "trend_gate_min_stock_momentum": 0.0,
        "watch_long_score": 0.40,
        "long_candidate_score": 0.80,
        "long_candidate_residual_z": -1.0,
        "long_candidate_relative_amount": 0.50,
        "mean_reversion_complete_residual_z": -0.10,
    },
    "manual_alert_mode": {
        "enabled": True,
        "action_signal_types": ["LAG_REPAIR_LONG", "MOMENTUM_BREAKOUT_LONG"],
        "watch_signal_types": ["WATCH_LONG", "WATCH_SETUP", "WATCH_LAG", "WATCH_MOMENTUM", "WATCH_RISKY"],
        "armed_signal_types": ["ARMED_LAG_LONG", "ARMED_BREAKOUT_LONG"],
        "min_factor_coverage_for_action": 0.60,
        "no_new_action_after": "14:50:00",
    },
    "alert_controls": {
        "min_minutes_between_action_alerts_per_bond": 30,
        "max_action_alerts_per_bond_per_day": 5,
        "max_total_action_alerts_per_day": 20,
        "no_new_action_after": "14:50:00",
        "expire_action_after_bars": 2,
        "armed_signal_types": ["ARMED_LAG_LONG", "ARMED_BREAKOUT_LONG"],
        "actionable_pool_states": ["ACTIVE", "ACTIVE_MANUAL"],
    },
    "trigger_thresholds": {
        "lag_stock_mom_30m_z": 0.60,
        "lag_stock_mom_60m_z": 0.0,
        "lag_residual_z_max": -0.80,
        "lag_relative_amount_min": 0.50,
        "lag_premium_slot_z_max": 1.0,
        "lag_flow_confirmation_min": 0.0,
        "lag_residual_vol_ratio_z_max": 0.80,
        "breakout_stock_mom_30m_z": 1.0,
        "breakout_cb_mom_15m_z": 0.50,
        "breakout_relative_amount_min": 1.0,
        "breakout_residual_z_min": -0.50,
        "breakout_residual_z_max": 0.80,
        "breakout_premium_slot_z_max": 1.20,
        "breakout_cb_flow_z_min": 0.30,
        "watch_stock_mom_30m_z_min": 0.0,
        "watch_residual_z_max": 0.0,
        "watch_premium_slot_z_max": 1.50,
    },
    "armed_thresholds": {
        "lag_stock_mom_30m_z": 0.50,
        "lag_residual_z_max": -0.60,
        "lag_premium_slot_z_max": 1.20,
        "lag_relative_amount_min": 0.40,
        "lag_flow_confirmation_min": -0.20,
        "breakout_stock_mom_30m_z": 0.80,
        "breakout_cb_mom_15m_z": 0.20,
        "breakout_residual_z_max": 1.00,
        "breakout_relative_amount_min": 0.70,
        "breakout_cb_flow_z_min": 0.00,
    },
    "watch_thresholds": {
        "minimum_factor_coverage": 0.50,
        "setup_premium_slot_z_max": 1.20,
        "setup_residual_z_max": 0.80,
        "momentum_stock_mom_30m_z_min": 0.0,
        "momentum_cb_mom_15m_z_min": 0.0,
        "momentum_residual_z_max": 0.80,
        "momentum_premium_slot_z_max": 1.50,
        "risky_premium_slot_z_min": 1.50,
        "risky_premium_change_10m_z_min": 1.50,
        "risky_premium_change_30m_z_min": 1.50,
        "risky_residual_z_min": 0.80,
        "risky_high_low_range_proxy_z_min": 1.50,
        "risky_residual_vol_ratio_z_min": 0.80,
    },
    "market_regime": {
        "min_active_count": 5,
        "risk_on_breadth_30m": 0.60,
        "risk_off_breadth_30m": 0.30,
        "thin_market_relative_amount": 0.30,
    },
    "volatility_gate": {
        "residual_vol_ratio_z_max": 0.80,
        "high_low_range_proxy_z_max": 1.50,
        "noise_penalty_weight": 0.25,
    },
    "setup_score_weights": {
        "stock_impulse_score": 0.10,
        "stock_mom_30m_z": 0.25,
        "stock_mom_60m_z": 0.15,
        "factor_residual_mr": 0.25,
        "factor_premium_mr": 0.10,
        "flow_confirmation": 0.10,
        "relative_amount_log": 0.10,
        "factor_vol_response_gap": 0.05,
        "vwap_gap": 0.05,
        "noise_penalty": -0.25,
    },
    "exit_thresholds": {
        "mean_reversion_complete_residual_z": -0.10,
        "momentum_decay_stock_mom_30m_z": -0.20,
        "momentum_decay_stock_mom_short_z": -0.20,
        "premium_overheat_slot_z": 1.50,
        "premium_change_10m_z_overheat": 1.50,
        "premium_change_30m_z_overheat": 1.50,
        "noise_spike_range_z": 2.00,
        "noise_spike_residual_vol_z": 1.20,
    },
    "data_quality": {"max_stale_minutes": 10},
    "ml_shadow_v04": {
        "enabled": True,
        "shadow_only": True,
        "no_overnight": True,
        "no_new_action_after": "14:30:00",
        "exit_hint_after": "14:50:00",
        "force_flatten_time": "14:55:00",
        "allow_15_00_bar_for_label": True,
        "allow_15_00_bar_for_new_action": False,
        "event_scopes": ["ARMED", "ACTION"],
        "official_action_signal_types": ["LAG_REPAIR_LONG", "MOMENTUM_BREAKOUT_LONG"],
        "official_armed_signal_types": ["ARMED_LAG_LONG", "ARMED_BREAKOUT_LONG"],
        "exclude_watch_from_trade_labels": True,
        "exclude_shadow_research_from_official_metrics": True,
        "cost_bps_grid": [5, 10, 20],
        "default_cost_bps": 10,
        "feature_version": "v0.4_feature_001",
        "label_version": "v0.4_label_001",
        "model_version_prefix": "ml_shadow_v04",
        "config_version": "intraday_factors_v0.4",
        "dataset_name": "cb_intraday_ml_event_dataset",
        "prediction_name": "cb_intraday_ml_predictions",
        "experiment_name": "cb_intraday_ml_experiments",
        "output_dir": "outputs/ml_shadow",
        "dedupe_episode_gap_minutes": 30,
        "stop_threshold": -0.002,
        "triple_barrier": {
            "lag_repair": {"profit_pct": 0.0030, "stop_pct": -0.0020, "vertical_bars": 6},
            "breakout": {"profit_pct": 0.0025, "stop_pct": -0.0015, "vertical_bars": 3},
            "default": {"profit_pct": 0.0030, "stop_pct": -0.0020, "vertical_bars": 6},
        },
        "validation": {
            "split_type": "walk_forward",
            "use_purging": True,
            "embargo_bars": 6,
            "min_train_days": 15,
            "min_test_days": 5,
            "group_by_trade_date": True,
        },
        "models": {
            "logistic": {"enabled": True, "penalty": "l2", "C": 1.0, "class_weight": "balanced", "max_iter": 1000},
            "gbdt": {"enabled": True, "max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
            "calibrated": {"enabled": True, "method": "sigmoid", "cv": 3},
            "small_mlp": {"enabled": False, "require_cuda": False},
        },
        "ml_action_hint_thresholds": {
            "boost_p_good_min": 0.65,
            "suppress_p_bad_min": 0.55,
            "fake_breakout_suppress_min": 0.60,
        },
    },
}

DEFAULT_SCHEMA: dict[str, Any] = {
    "tables": {
        "live_5m": "cb_watchlist_bar_5m_live",
        "watchlist_daily": "cb_watchlist_daily",
        "factor_snapshot": "cb_intraday_factor_snapshot",
        "alert_table": "cb_intraday_alerts",
        "job_runs": "cb_intraday_job_runs",
        "live_backfill_diff": "cb_intraday_live_backfill_diff",
        "ml_event_dataset": "cb_intraday_ml_event_dataset",
        "ml_predictions": "cb_intraday_ml_predictions",
        "ml_experiments": "cb_intraday_ml_experiments",
    },
    "live_5m_fields": {
        "row_timestamp": "ts",
        "bar_start": "bar_start",
        "trade_date": "trade_date",
        "asset_type": "asset_type",
        "watchlist_status": "watchlist_status",
        "pool_state": "pool_state",
        "instrument_code": "instrument_code",
        "instrument_name": "instrument_name",
        "bond_code": "bond_code",
        "bond_name": "bond_name",
        "stock_code": "stock_code",
        "stock_name": "stock_name",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
        "premium_rate": "premium_rate",
        "conversion_price": "conversion_price",
        "ytm_available": "ytm_available",
        "ytm_source": "ytm_source",
        "turnover_available": "turnover_available",
        "turnover_source": "turnover_source",
        "premium_source": "premium_source",
    },
    "asset_type_values": {"cb": "CB", "stock": "STOCK"},
    "watchlist_fields": {
        "timestamp": "ts",
        "snapshot_id": "snapshot_id",
        "bond_code": "bond_code",
        "bond_name": "bond_name",
        "stock_code": "stock_code",
        "stock_name": "stock_name",
        "conversion_price": "conversion_price",
        "watchlist_status": "watchlist_status",
        "pool_state": "pool_state",
        "candidate_since": "candidate_since",
        "active_since": "active_since",
    },
}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the small YAML subset used by this project without adding PyYAML."""
    if not path.exists():
        return {}

    parsed_lines: list[tuple[int, str]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        parsed_lines.append((len(line) - len(line.lstrip(" ")), line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for idx, (indent, content) in enumerate(parsed_lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unsupported YAML list placement in {path}: {content}")
            parent.append(_parse_scalar(content[2:].strip()))
            continue

        key, _, value = content.partition(":")
        if value.strip() == "":
            next_is_list = False
            if idx + 1 < len(parsed_lines):
                next_indent, next_content = parsed_lines[idx + 1]
                next_is_list = next_indent > indent and next_content.startswith("- ")
            child: Any = [] if next_is_list else {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return _deep_update(DEFAULT_CONFIG, _load_simple_yaml(Path(path) if path else DEFAULT_CONFIG_PATH))


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    return _deep_update(DEFAULT_SCHEMA, _load_simple_yaml(Path(path) if path else DEFAULT_SCHEMA_PATH))
