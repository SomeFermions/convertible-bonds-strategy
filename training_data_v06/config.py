from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "training_data_v06.yaml"
DEFAULT_JQDATA_CREDENTIALS_PATH = PROJECT_DIR / "config" / "jqdata_credentials.toml"


DEFAULT_CONFIG: dict[str, Any] = {
    "version": "v0.6",
    "default_sample_version": "jq_cb_train_v1",
    "research_only": True,
    "jqdata": {
        "account_type": "personal_trial",
        "credentials_file": str(DEFAULT_JQDATA_CREDENTIALS_PATH),
        "daily_quota_limit": 500000,
        "default_daily_hard_cap": 450000,
        "reserve_quota_for_research": True,
        "available_range": "trial_15m_to_3m",
        "trial_window_anchor_date": None,
        "trial_download_start_buffer_days": 14,
        "trade_calendar_cache": "outputs/cn_trade_calendar.csv",
        "available_start_date": None,
        "available_end_date": None,
        "target_trading_days": 200,
        "max_smoke_bonds": 2,
        "max_smoke_days": 2,
        "max_smoke_instruments": 4,
        "large_download_instrument_threshold": 2,
        "large_download_row_threshold": 10000,
        "frequency": "5m",
        "bars_per_day_estimate": 48,
        "plan_chunk_trading_days": 10,
        "request_cooldown_seconds": 1.0,
        "max_failed_plans_per_run": 20,
        "max_attempts_per_plan": 3,
        "stale_running_minutes": 60,
        "cb_fields": ["open", "high", "low", "close", "volume", "money", "paused"],
        "stock_fields": ["open", "high", "low", "close", "volume", "money", "paused", "live_turnover"],
    },
    "wind": {
        "account_type": "personal_trial",
        "provider": "auto",
        "close_after_request": True,
        "wsd_options": "Days=Trading",
        "field_map": {
            "live_turnover": "turn",
            "ytm": "ytm_b",
        },
        "weekly_quota_limit": 50000,
        "weekly_hard_cap": 45000,
        "reserve_rows": 5000,
        "validation_run_target_cells": 7000,
        "validation_fields": ["live_turnover"],
        "allowed_fields": ["live_turnover", "ytm"],
        "forbidden_fields": [
            "call_trigger_count",
            "put_trigger_count",
            "reset_trigger_count",
            "redeem_terms",
            "put_terms",
            "rating",
            "coupon",
            "clause",
        ],
    },
    "akshare_static": {
        "source_vendor": "akshare",
        "conversion_price_cache_dir": "outputs/akshare_conversion_price_history",
        "request_cooldown_seconds": 2.0,
        "request_cooldown_jitter_seconds": 0.5,
        "max_retries": 3,
        "conversion_price_mismatch_tolerance": 0.08,
        "allowed_fields": [
            "maturity_date",
            "st_status",
            "conversion_price",
            "bond_code",
            "stock_code",
            "bond_name",
            "stock_name",
            "listing_date",
            "delisting_date",
        ],
    },
    "sample_selection": {
        "max_sample_size": 200,
        "active_like_premium_min": 20.0,
        "active_like_premium_max": 30.0,
        "shadow_premium_min": 10.0,
        "shadow_premium_max": 40.0,
        "broad_premium_min": -10.0,
        "broad_premium_max": 80.0,
        "min_days_to_maturity": 183,
        "max_debt_asset_ratio": 70.0,
        "severe_debt_asset_ratio": 90.0,
    },
    "canonical": {
        "canonical_version": "jq_training_canonical_v1",
        "bar_minutes": 5,
        "no_cross_lunch": True,
    },
    "benchmarks": [
        {"benchmark_code": "CSI300", "benchmark_name": "沪深300", "vendor_code": "000300.XSHG"},
        {"benchmark_code": "CSI500", "benchmark_name": "中证500", "vendor_code": "000905.XSHG"},
        {"benchmark_code": "CSI1000", "benchmark_name": "中证1000", "vendor_code": "000852.XSHG"},
        {"benchmark_code": "CHINEXT", "benchmark_name": "创业板指", "vendor_code": "399006.XSHE"},
        {"benchmark_code": "STAR50", "benchmark_name": "科创50", "vendor_code": "000688.XSHG"},
    ],
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return deepcopy(DEFAULT_CONFIG)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError(
            f"PyYAML is required to load the existing training config: {config_path}"
        ) from exc
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"training config must be a mapping: {config_path}")
    return _deep_merge(DEFAULT_CONFIG, loaded)
