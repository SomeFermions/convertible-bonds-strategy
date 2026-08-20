from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import output_dir, resolve_project_path


LOGGER = logging.getLogger(__name__)


IDENTIFIER_COLUMNS = [
    "trade_date",
    "bar_start",
    "bar_end",
    "bar_slot",
    "bond_code",
    "stock_code",
    "research_pool_scope",
    "window_id",
    "window_role",
]

REQUIRED_SOURCE_COLUMNS = [
    *IDENTIFIER_COLUMNS,
    "pair_data_quality_pass",
    "cb_open",
    "cb_high",
    "cb_low",
    "cb_close",
    "cb_amount",
    "stock_open",
    "stock_high",
    "stock_low",
    "stock_close",
    "stock_amount",
    "cb_return",
    "stock_return",
    "stock_idiosyncratic_return",
    "conversion_price",
    "conversion_price_asof_valid",
]

# Explicitly excluded from the V0.8 model contract even when present upstream.
FORBIDDEN_SOURCE_FEATURES = {
    "debt_ratio",
    "operating_cash_flow",
    "free_cash_flow",
    "clause_driven_flag",
    "distress_flag",
    "bond_type",
    "classification_point_in_time",
    "qiv_raw",
    "qiv_smooth",
    "mc_price_gap_pct",
    "ytm",
}

OPTIONAL_SOURCE_COLUMNS = [
    "cb_source_vendor",
    "stock_source_vendor",
    "cb_market_return",
    "chosen_market_index",
    "stock_index_beta",
    "stock_market_component",
    "stock_shock_z",
    "stock_shock_amount_z",
    "cb_amount_z_5m",
    "relative_amount",
    "high_low_range_proxy",
    "high_low_range_proxy_z",
    "close_location",
    "cb_vwap_dev",
    "stock_vwap_dev",
    "vwap_gap",
    "premium_expansion_proxy_z",
    "stock_to_cb_beta_lag0",
    "stock_to_cb_beta_lag1",
    "stock_to_cb_beta_lag2",
    "stock_to_cb_beta_lag3",
    "cb_market_gamma",
    "linkage_regression_intercept",
    "linkage_regression_r2",
    "linkage_sample_count",
    "linkage_model_confidence",
    "linkage_residual",
    "linkage_residual_z",
    "linkage_gap_z",
    "liquidity_sweet_spot_score_5m",
    "hot_money_risk_score_5m",
    "volume_climax_score",
    "liquidity_regime_5m",
    "daily_turnover",
    "adv20_amount",
    "active_bar_ratio",
    "zero_bar_ratio",
    "amihud",
    "amount_z",
    "turnover_z",
]


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_days: int
    role: str
    official: bool

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ["train_start", "train_end", "validation_start", "validation_end"]:
            record[key] = str(record[key].date())
        return record


def validate_source_schema(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"V0.8 source cache is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("V0.8 source cache is empty")


def _normalize_source(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    validate_source_schema(frame)
    available = IDENTIFIER_COLUMNS + [
        column
        for column in REQUIRED_SOURCE_COLUMNS + OPTIONAL_SOURCE_COLUMNS
        if column not in IDENTIFIER_COLUMNS and column in frame.columns
    ]
    available = list(dict.fromkeys(available))
    out = frame.loc[:, available].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["bar_start"] = pd.to_datetime(out["bar_start"], errors="coerce")
    out["bar_end"] = pd.to_datetime(out["bar_end"], errors="coerce")
    out["bond_code"] = out["bond_code"].astype(str).str.zfill(6)
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    out["bar_slot"] = pd.to_numeric(out["bar_slot"], errors="coerce").astype("Int16")
    out = out[out["bar_slot"].between(0, int(config["data"]["maximum_bar_slot"]))].copy()
    if config["data"].get("require_pair_quality", True):
        out = out[out["pair_data_quality_pass"].fillna(False).astype(bool)].copy()
    duplicate_keys = ["trade_date", "bar_start", "bond_code", "stock_code"]
    duplicate_count = int(out.duplicated(duplicate_keys, keep=False).sum())
    if duplicate_count:
        raise ValueError(f"V0.8 source contains {duplicate_count} duplicate aligned bars")
    out["session_part"] = np.where(out["bar_slot"].lt(24), "AM", "PM")
    out["session_slot"] = np.where(
        out["bar_slot"].lt(24), out["bar_slot"], out["bar_slot"] - 24
    ).astype("int16")
    return out.sort_values(["bond_code", "trade_date", "bar_slot"]).reset_index(drop=True)


def load_source_frame(
    config: dict[str, Any],
    start_date: str | None = None,
    end_date: str | None = None,
    days: int | None = None,
) -> pd.DataFrame:
    """Load the audited V0.7.1 PIT-aligned cache without querying new data."""
    path = resolve_project_path(config["data"]["source_feature_cache"])
    if not path.exists():
        raise FileNotFoundError(
            f"Audited V0.7.1 feature cache is required but missing: {path}"
        )
    LOGGER.info("loading audited V0.7.1 feature cache: %s", path)
    frame = _normalize_source(pd.read_pickle(path), config)
    if start_date:
        frame = frame[frame["trade_date"] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        frame = frame[frame["trade_date"] <= pd.Timestamp(end_date).normalize()]
    if days is not None:
        dates = sorted(frame["trade_date"].dropna().unique())
        if len(dates) > days:
            frame = frame[frame["trade_date"].isin(dates[-int(days) :])]
    if frame.empty:
        raise ValueError("No V0.8 source rows remain after date filtering")
    return frame.reset_index(drop=True)


def _purged_train_end(dates: list[pd.Timestamp], before: pd.Timestamp, purge_days: int) -> pd.Timestamp:
    eligible = [date for date in dates if date < before]
    if len(eligible) <= purge_days:
        raise ValueError("Not enough dates before validation after purge")
    return eligible[-purge_days - 1]


def build_walk_forward_folds(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> list[WalkForwardFold]:
    """Create chronological expanding folds with date-level purging.

    The warm-up fold creates causal OOF response gaps inside W1. Official model
    comparison starts at W2 and therefore never treats the warm-up segment as a
    final validation result.
    """
    dates = [pd.Timestamp(value).normalize() for value in sorted(frame["trade_date"].unique())]
    if len(dates) < int(config["walk_forward"]["minimum_train_days"]) + 2:
        raise ValueError("Insufficient trading dates for chronological walk-forward")
    purge_days = int(config["walk_forward"]["purge_days"])
    folds: list[WalkForwardFold] = []
    first_window = config["walk_forward"]["windows"][0]
    w1_dates = [
        date
        for date in dates
        if pd.Timestamp(first_window["start"]) <= date <= pd.Timestamp(first_window["end"])
    ]
    warmup_days = int(config["walk_forward"]["warmup_train_days"])
    if len(w1_dates) > warmup_days + purge_days:
        validation_start = w1_dates[warmup_days + purge_days]
        folds.append(
            WalkForwardFold(
                fold_id="V080_WARMUP",
                train_start=w1_dates[0],
                train_end=w1_dates[warmup_days - 1],
                validation_start=validation_start,
                validation_end=w1_dates[-1],
                purge_days=purge_days,
                role="warmup_oof",
                official=False,
            )
        )
    for window in config["walk_forward"]["windows"][1:]:
        validation_dates = [
            date
            for date in dates
            if pd.Timestamp(window["start"]) <= date <= pd.Timestamp(window["end"])
        ]
        if not validation_dates:
            continue
        earlier_count = sum(date < validation_dates[0] for date in dates)
        if earlier_count <= int(config["walk_forward"]["minimum_train_days"]) + purge_days:
            continue
        train_end = _purged_train_end(dates, validation_dates[0], purge_days)
        folds.append(
            WalkForwardFold(
                fold_id=str(window["id"]),
                train_start=dates[0],
                train_end=train_end,
                validation_start=validation_dates[0],
                validation_end=validation_dates[-1],
                purge_days=purge_days,
                role=str(window["role"]),
                official=True,
            )
        )
    if not folds:
        # Smoke-test fallback for arbitrary short ranges. A non-official OOF
        # segment precedes the official validation so the HMM still trains on
        # genuinely earlier response gaps.
        minimum = int(config["walk_forward"]["minimum_train_days"])
        if len(dates) < minimum + purge_days + 8:
            raise ValueError("Ad-hoc walk-forward requires more trading dates")
        warmup_train_end = min(max(minimum - 1, int(len(dates) * 0.45)), len(dates) - 9)
        warmup_validation_start = warmup_train_end + purge_days + 1
        official_train_end = min(max(warmup_validation_start + 5, int(len(dates) * 0.70)), len(dates) - purge_days - 2)
        official_validation_start = official_train_end + purge_days + 1
        folds.append(
            WalkForwardFold(
                fold_id="V080_ADHOC_WARMUP",
                train_start=dates[0],
                train_end=dates[warmup_train_end],
                validation_start=dates[warmup_validation_start],
                validation_end=dates[official_train_end - 1],
                purge_days=purge_days,
                role="warmup_oof",
                official=False,
            )
        )
        folds.append(
            WalkForwardFold(
                fold_id="V080_ADHOC",
                train_start=dates[0],
                train_end=dates[official_train_end],
                validation_start=dates[official_validation_start],
                validation_end=dates[-1],
                purge_days=purge_days,
                role="adhoc_validation",
                official=True,
            )
        )
    return folds


def fold_frame(folds: Iterable[WalkForwardFold]) -> pd.DataFrame:
    return pd.DataFrame([fold.to_record() for fold in folds])


def select_fold_rows(
    frame: pd.DataFrame, fold: WalkForwardFold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    train = frame[dates.between(fold.train_start, fold.train_end)].copy()
    validation = frame[dates.between(fold.validation_start, fold.validation_end)].copy()
    if train.empty or validation.empty:
        raise ValueError(f"Empty train or validation set for fold {fold.fold_id}")
    if train["trade_date"].max() >= validation["trade_date"].min():
        raise AssertionError(f"Fold chronology violated for {fold.fold_id}")
    return train, validation


def write_dataset_manifest(
    frame: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: dict[str, Any],
    feature_columns: list[str],
    label_columns: list[str],
) -> Path:
    path = output_dir(config) / "dataset_manifest.json"
    manifest = {
        "strategy_version": config["strategy"]["version"],
        "factor_version": config["strategy"]["factor_version"],
        "config_version": config["strategy"]["config_version"],
        "source_cache": str(resolve_project_path(config["data"]["source_feature_cache"])),
        "rows": int(len(frame)),
        "trade_dates": int(frame["trade_date"].nunique()),
        "bond_count": int(frame["bond_code"].nunique()),
        "stock_count": int(frame["stock_code"].nunique()),
        "start_date": str(frame["trade_date"].min().date()),
        "end_date": str(frame["trade_date"].max().date()),
        "feature_count": len(feature_columns),
        "label_count": len(label_columns),
        "features": feature_columns,
        "labels": label_columns,
        "forbidden_source_features": sorted(FORBIDDEN_SOURCE_FEATURES),
        "same_bar_cb_return_is_feature": "cb_return" in feature_columns,
        "universe_bias_flag": config["data"]["universe_bias_flag"],
        "fundamental_pit_status": config["data"]["fundamental_pit_status"],
        "folds": [fold.to_record() for fold in folds],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
