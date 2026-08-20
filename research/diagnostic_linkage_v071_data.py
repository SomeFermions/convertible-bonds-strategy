from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intraday_factors.data_source import query_to_dataframe, sql_string


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "diagnostic_linkage_v071.yaml"
DAILY_PANEL_PATH = PROJECT_DIR / "outputs" / "research" / "hybrid_v07" / "daily_research_panel.csv"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    return config


def output_dir(config: dict[str, Any]) -> Path:
    path = Path(config["strategy"]["output_dir"])
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_versions(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    out["strategy_version"] = config["strategy"]["version"]
    out["factor_version"] = config["strategy"]["factor_version"]
    out["config_version"] = config["strategy"]["config_version"]
    return out


def write_csv(frame: pd.DataFrame, path: Path, config: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    add_versions(frame, config).to_csv(path, index=False)
    return path


def configured_windows(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in config["common_slice"]["windows"]:
        rows.append(
            {
                "window_id": str(item["id"]),
                "window_role": str(item["role"]),
                "window_start": pd.Timestamp(item["start"]).normalize(),
                "window_end": pd.Timestamp(item["end"]).normalize(),
            }
        )
    return pd.DataFrame(rows)


def assign_window(
    frame: pd.DataFrame,
    config: dict[str, Any],
    date_column: str = "trade_date",
) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out[date_column], errors="coerce").dt.normalize()
    out[date_column] = dates
    out["window_id"] = None
    out["window_role"] = None
    for item in config["common_slice"]["windows"]:
        start = pd.Timestamp(item["start"]).normalize()
        end = pd.Timestamp(item["end"]).normalize()
        mask = dates.between(start, end)
        out.loc[mask, "window_id"] = str(item["id"])
        out.loc[mask, "window_role"] = str(item["role"])
    return out[out["window_id"].notna()].copy()


def filter_dates(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    date_column: str = "trade_date",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce").dt.normalize()
    if start_date:
        out = out[out[date_column] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        out = out[out[date_column] <= pd.Timestamp(end_date).normalize()]
    return out


def session_slot(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="coerce")
    minute = ts.dt.hour * 60 + ts.dt.minute
    morning = (minute >= 570) & (minute <= 685)
    afternoon = (minute >= 780) & (minute <= 895)
    values = np.full(len(ts), np.nan)
    values[morning.to_numpy()] = ((minute[morning] - 570) // 5).to_numpy()
    values[afternoon.to_numpy()] = (24 + (minute[afternoon] - 780) // 5).to_numpy()
    return pd.Series(values, index=timestamp.index, dtype="Float64")


def valid_session_bar(timestamp: pd.Series) -> pd.Series:
    slot = session_slot(timestamp)
    return slot.notna() & slot.between(0, 47)


def load_daily_panel(
    config: dict[str, Any],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if not DAILY_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Required local V0.7 daily panel is missing: {DAILY_PANEL_PATH}"
        )
    panel = pd.read_csv(
        DAILY_PANEL_PATH,
        dtype={"bond_code": str, "stock_code": str},
        low_memory=False,
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    panel = panel[panel["data_mode"].eq("scheduled_backfill")].copy()
    panel = assign_window(panel, config)
    panel = filter_dates(panel, start_date, end_date)
    for column in [
        "conversion_price_asof_valid",
        "qiv_stock_close_asof_valid",
        "pricing_inputs_asof_valid",
        "classification_point_in_time",
        "clause_driven_flag",
        "distress_flag",
    ]:
        if column in panel.columns:
            panel[column] = (
                panel[column]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
            )
    panel["bond_code"] = panel["bond_code"].str.zfill(6)
    panel["stock_code"] = panel["stock_code"].str.zfill(6)
    return panel.sort_values(["trade_date", "bond_code"]).reset_index(drop=True)


def _where_for_window(config: dict[str, Any], window: dict[str, Any]) -> str:
    sample = sql_string(str(config["strategy"]["sample_version"]))
    start = sql_string(str(window["start"]))
    end = sql_string(str(window["end"]))
    return f"sample_version = {sample} AND trade_date >= {start} AND trade_date <= {end}"


def _normalize_pair_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    raw["bar_start"] = pd.to_datetime(raw["bar_start"], errors="coerce")
    raw["bar_end"] = pd.to_datetime(raw["bar_end"], errors="coerce")
    raw["bond_code"] = raw["bond_code"].astype(str).str.zfill(6)
    raw["stock_code"] = raw["stock_code"].astype(str).str.zfill(6)
    raw["asset_type"] = raw["asset_type"].astype(str).str.upper()
    raw = raw[valid_session_bar(raw["bar_start"])].copy()
    keys = ["trade_date", "bar_start", "bar_end", "bond_code", "stock_code"]
    cb = raw[raw["asset_type"].eq("CB")].drop(columns=["asset_type"]).copy()
    stock = raw[raw["asset_type"].eq("STOCK")].drop(columns=["asset_type"]).copy()
    common_values = ["research_pool_scope"]
    value_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "paused",
        "data_quality_flag",
        "source_vendor",
    ]
    cb = cb[keys + common_values + value_columns].rename(
        columns={column: f"cb_{column}" for column in value_columns}
    )
    stock = stock[keys + value_columns].rename(
        columns={column: f"stock_{column}" for column in value_columns}
    )
    cb = cb.drop_duplicates(keys, keep="last")
    stock = stock.drop_duplicates(keys, keep="last")
    paired = cb.merge(stock, on=keys, how="inner", validate="one_to_one")
    paired["bar_slot"] = session_slot(paired["bar_start"]).astype("Int16")
    paired["pair_data_quality_pass"] = (
        paired["cb_data_quality_flag"].eq("OK")
        & paired["stock_data_quality_flag"].eq("OK")
        & ~paired["cb_paused"].fillna(False).astype(bool)
        & ~paired["stock_paused"].fillna(False).astype(bool)
    )
    for column in [
        "cb_open",
        "cb_high",
        "cb_low",
        "cb_close",
        "cb_volume",
        "cb_amount",
        "stock_open",
        "stock_high",
        "stock_low",
        "stock_close",
        "stock_volume",
        "stock_amount",
    ]:
        paired[column] = pd.to_numeric(paired[column], errors="coerce").astype("float64")
    return paired.sort_values(["bond_code", "trade_date", "bar_start"]).reset_index(drop=True)


def load_pair_bars(
    conn,
    config: dict[str, Any],
    refresh_cache: bool = False,
) -> pd.DataFrame:
    out = output_dir(config)
    pieces = []
    for window in config["common_slice"]["windows"]:
        cache = out / f"cache_pair_bars_{window['id']}.pkl"
        if cache.exists() and not refresh_cache:
            piece = pd.read_pickle(cache)
        else:
            where = _where_for_window(config, window)
            LOGGER.info("querying canonical pair bars for %s", window["id"])
            raw = query_to_dataframe(
                conn,
                f"""
                SELECT trade_date, bar_start, bar_end, bond_code, stock_code,
                       asset_type, research_pool_scope, open, high, low, close,
                       volume, amount, paused, is_valid_bar, data_quality_flag,
                       source_vendor
                FROM market_bar_5m_training_canonical
                WHERE {where} AND is_valid_bar = true
                """,
            )
            piece = _normalize_pair_raw(raw)
            piece = assign_window(piece, config)
            piece.to_pickle(cache)
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["bond_code", "trade_date", "bar_start"]
    ).reset_index(drop=True)


def load_benchmark_bars(
    conn,
    config: dict[str, Any],
    refresh_cache: bool = False,
) -> pd.DataFrame:
    cache = output_dir(config) / "cache_benchmark_bars.pkl"
    if cache.exists() and not refresh_cache:
        return pd.read_pickle(cache)
    pieces = []
    for window in config["common_slice"]["windows"]:
        where = _where_for_window(config, window)
        raw = query_to_dataframe(
            conn,
            f"""
            SELECT trade_date, benchmark_code, benchmark_name, vendor_code,
                   bar_start, bar_end, open, high, low, close, volume, amount,
                   paused, is_valid_bar, data_quality_flag, source_vendor
            FROM market_benchmark_bar_5m_canonical
            WHERE {where} AND is_valid_bar = true
            """,
        )
        pieces.append(raw)
    benchmark = pd.concat(pieces, ignore_index=True)
    benchmark["trade_date"] = pd.to_datetime(
        benchmark["trade_date"], errors="coerce"
    ).dt.normalize()
    benchmark["bar_start"] = pd.to_datetime(benchmark["bar_start"], errors="coerce")
    benchmark["bar_end"] = pd.to_datetime(benchmark["bar_end"], errors="coerce")
    benchmark = benchmark[valid_session_bar(benchmark["bar_start"])].copy()
    benchmark["bar_slot"] = session_slot(benchmark["bar_start"]).astype("Int16")
    benchmark = assign_window(benchmark, config)
    benchmark = benchmark.sort_values(["benchmark_code", "trade_date", "bar_start"])
    first_bar = benchmark["bar_slot"].eq(0)
    prior = benchmark.groupby(["benchmark_code", "trade_date"], sort=False)["close"].shift(1)
    benchmark["benchmark_return"] = np.where(
        first_bar,
        pd.to_numeric(benchmark["close"], errors="coerce")
        / pd.to_numeric(benchmark["open"], errors="coerce")
        - 1.0,
        pd.to_numeric(benchmark["close"], errors="coerce") / prior - 1.0,
    )
    benchmark.to_pickle(cache)
    return benchmark.reset_index(drop=True)


def benchmark_wide(benchmark: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    keys = ["trade_date", "bar_start"]
    codes = list(config["linkage"]["benchmark_codes"])
    pieces = []
    for value, suffix in [
        ("benchmark_return", "return"),
        ("close", "close"),
        ("amount", "amount"),
    ]:
        pivot = benchmark.pivot_table(
            index=keys,
            columns="benchmark_code",
            values=value,
            aggfunc="last",
        )
        pivot = pivot.reindex(columns=codes)
        pivot.columns = [f"index_{code}_{suffix}" for code in pivot.columns]
        pieces.append(pivot)
    return pd.concat(pieces, axis=1).reset_index()


def common_benchmark_dates(benchmark: pd.DataFrame, config: dict[str, Any]) -> pd.Index:
    bars_per_day = int(config["common_slice"]["bars_per_day"])
    codes = set(config["linkage"]["benchmark_codes"])
    coverage = (
        benchmark.groupby(["trade_date", "benchmark_code"])["bar_start"]
        .nunique()
        .unstack(fill_value=0)
    )
    for code in codes:
        if code not in coverage.columns:
            return pd.Index([], dtype="datetime64[ns]")
    complete = coverage[list(codes)].ge(bars_per_day).all(axis=1)
    return coverage.index[complete]


def attach_daily_context(
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "trade_date",
        "bond_code",
        "stock_code",
        "bond_type",
        "bond_type_confidence",
        "classification_point_in_time",
        "conversion_price",
        "conversion_price_asof_valid",
        "parity",
        "premium",
        "pricing_inputs_asof_valid",
        "daily_turnover",
        "adv20_amount",
        "active_bar_ratio",
        "zero_bar_ratio",
        "amihud",
        "amount_z",
        "turnover_z",
        "empirical_delta",
        "elasticity",
        "residual_z_daily",
        "residual_daily",
        "premium_change_z",
        "liquidity_sweet_spot_score",
        "hot_money_risk_score",
        "liquidity_regime",
        "qiv_smooth",
        "qiv_relative_value_score",
        "qiv_microstructure_contaminated",
        "mc_price_gap_pct",
        "research_pool_scope",
        "static_asof_valid",
        "debt_ratio",
        "operating_cash_flow",
        "free_cash_flow",
        "clause_driven_flag",
        "distress_flag",
    ]
    available = [column for column in columns if column in daily.columns]
    context = daily[available].drop_duplicates(["trade_date", "bond_code"], keep="last")
    context = context.rename(columns={"research_pool_scope": "daily_research_pool_scope"})
    out = bars.merge(context, on=["trade_date", "bond_code", "stock_code"], how="left")
    min_bars = int(config["common_slice"]["minimum_valid_bars_per_asset"])
    daily_counts = out.groupby(["trade_date", "bond_code"], sort=False).agg(
        paired_bar_count=("bar_start", "nunique"),
        quality_bar_count=("pair_data_quality_pass", "sum"),
    )
    out = out.merge(daily_counts.reset_index(), on=["trade_date", "bond_code"], how="left")
    conversion_ok = out.get(
        "conversion_price_asof_valid", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    turnover_ok = pd.to_numeric(out.get("daily_turnover"), errors="coerce").notna()
    out["pit_universe_eligible"] = (
        out["paired_bar_count"].ge(min_bars)
        & out["quality_bar_count"].ge(min_bars)
        & conversion_ok
        & turnover_ok
    )
    reasons = np.select(
        [
            out["paired_bar_count"].lt(min_bars),
            out["quality_bar_count"].lt(min_bars),
            ~conversion_ok,
            ~turnover_ok,
        ],
        [
            "INSUFFICIENT_PAIRED_BARS",
            "DATA_QUALITY_FAILED",
            "MISSING_PIT_CONVERSION_PRICE",
            "MISSING_DAILY_TURNOVER",
        ],
        default="PIT_OBSERVABLE_ELIGIBLE",
    )
    out["pit_universe_reason"] = reasons
    out["universe_bias_flag"] = config["common_slice"]["universe_bias_flag"]
    out["fundamental_pit_status"] = config["common_slice"]["fundamental_pit_status"]
    out["point_in_time_universe_claim"] = False
    return out


def build_common_slice(
    conn,
    config: dict[str, Any],
    start_date: str | None = None,
    end_date: str | None = None,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = output_dir(config) / "common_slice_bars.pkl"
    benchmark_cache = output_dir(config) / "common_slice_benchmarks.pkl"
    daily = load_daily_panel(config)
    if cache.exists() and benchmark_cache.exists() and not refresh_cache:
        bars = pd.read_pickle(cache)
        benchmark = pd.read_pickle(benchmark_cache)
    else:
        pairs = load_pair_bars(conn, config, refresh_cache=refresh_cache)
        benchmark = load_benchmark_bars(conn, config, refresh_cache=refresh_cache)
        complete_dates = common_benchmark_dates(benchmark, config)
        pairs = pairs[pairs["trade_date"].isin(complete_dates)].copy()
        wide = benchmark_wide(benchmark, config)
        bars = pairs.merge(wide, on=["trade_date", "bar_start"], how="inner", validate="many_to_one")
        bars = attach_daily_context(bars, daily, config)
        bars.to_pickle(cache)
        benchmark.to_pickle(benchmark_cache)
    bars = filter_dates(bars, start_date, end_date)
    benchmark = filter_dates(benchmark, start_date, end_date)
    daily = filter_dates(daily, start_date, end_date)
    return bars.reset_index(drop=True), benchmark.reset_index(drop=True), daily.reset_index(drop=True)


def build_universe_report(
    bars: pd.DataFrame,
    benchmark: pd.DataFrame,
    daily: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_bond_day = bars.sort_values("bar_start").drop_duplicates(
        ["trade_date", "bond_code"], keep="last"
    )
    snapshot_columns = [
        "trade_date",
        "window_id",
        "window_role",
        "bond_code",
        "stock_code",
        "bond_type",
        "research_pool_scope",
        "daily_research_pool_scope",
        "paired_bar_count",
        "quality_bar_count",
        "conversion_price_asof_valid",
        "daily_turnover",
        "pit_universe_eligible",
        "pit_universe_reason",
        "universe_bias_flag",
        "fundamental_pit_status",
        "point_in_time_universe_claim",
    ]
    snapshot = per_bond_day[[c for c in snapshot_columns if c in per_bond_day.columns]].copy()
    benchmark_counts = benchmark.groupby("trade_date").agg(
        benchmark_codes=("benchmark_code", "nunique"),
        benchmark_bars=("bar_start", "nunique"),
    )
    summary = snapshot.groupby(["window_id", "window_role"], dropna=False).agg(
        first_date=("trade_date", "min"),
        last_date=("trade_date", "max"),
        trading_days=("trade_date", "nunique"),
        observable_bonds=("bond_code", "nunique"),
        bond_days=("bond_code", "size"),
        pit_eligible_bond_days=("pit_universe_eligible", "sum"),
        conversion_price_coverage=("conversion_price_asof_valid", "mean"),
        turnover_coverage=("daily_turnover", lambda values: pd.to_numeric(values, errors="coerce").notna().mean()),
    ).reset_index()
    benchmark_summary = benchmark_counts.reset_index().groupby(
        pd.cut(
            benchmark_counts.reset_index()["trade_date"],
            bins=[
                pd.Timestamp("1900-01-01"),
                *[pd.Timestamp(item["end"]) for item in config["common_slice"]["windows"]],
                pd.Timestamp("2100-01-01"),
            ],
            duplicates="drop",
        ),
        observed=True,
    ).agg(
        benchmark_days=("trade_date", "nunique"),
        minimum_benchmark_count=("benchmark_codes", "min"),
        minimum_slots=("benchmark_bars", "min"),
    )
    summary["benchmark_complete"] = (
        summary["trading_days"].eq(45)
        & (len(benchmark_summary) >= len(summary))
    )
    summary["universe_bias_flag"] = config["common_slice"]["universe_bias_flag"]
    summary["fundamental_pit_status"] = config["common_slice"]["fundamental_pit_status"]
    return snapshot, summary
