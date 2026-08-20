from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_factors.data_source import query_to_dataframe, sql_string
from research.hybrid_v07_core import add_daily_features


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _date_where(start_date: str | None, end_date: str | None, column: str = "trade_date") -> str:
    conditions = []
    if start_date:
        conditions.append(f"{column} >= {sql_string(start_date)}")
    if end_date:
        conditions.append(f"{column} <= {sql_string(end_date)}")
    return " AND ".join(conditions)


def load_training_daily_panel(
    conn,
    start_date: str | None,
    end_date: str | None,
    sample_version: str,
) -> pd.DataFrame:
    where = [
        f"sample_version = {sql_string(sample_version)}",
        "is_valid_bar = true",
    ]
    date_where = _date_where(start_date, end_date)
    if date_where:
        where.append(date_where)
    sql = f"""
    SELECT
        trade_date,
        bond_code,
        stock_code,
        asset_type,
        research_pool_scope,
        FIRST(open) AS open,
        MAX(high) AS high,
        MIN(low) AS low,
        LAST(close) AS close,
        SUM(volume) AS volume,
        SUM(amount) AS amount,
        COUNT(*) AS valid_bars,
        SUM(CASE WHEN amount <= 0 THEN 1 ELSE 0 END) AS zero_bars,
        AVG(CASE WHEN instrument_slot >= 30 THEN close ELSE NULL END) AS ma90min
    FROM market_bar_5m_training_canonical
    WHERE {' AND '.join(where)}
    GROUP BY trade_date, bond_code, stock_code, asset_type, research_pool_scope
    """
    raw = query_to_dataframe(conn, sql)
    if raw.empty:
        return raw
    raw["asset_type"] = raw["asset_type"].astype(str).str.upper()
    keys = ["trade_date", "bond_code", "stock_code", "research_pool_scope"]
    cb = raw[raw["asset_type"] == "CB"].drop(columns=["asset_type"]).copy()
    stock = raw[raw["asset_type"] == "STOCK"].drop(columns=["asset_type"]).copy()
    cb = cb.rename(
        columns={
            "open": "bond_open",
            "high": "bond_high",
            "low": "bond_low",
            "close": "bond_close",
            "volume": "bond_volume",
            "amount": "bond_amount",
            "valid_bars": "bond_valid_bars",
            "zero_bars": "bond_zero_bars",
            "ma90min": "cb_ma90min",
        }
    )
    stock = stock.rename(
        columns={
            "open": "stock_open",
            "high": "stock_high",
            "low": "stock_low",
            "close": "stock_close",
            "volume": "stock_volume",
            "amount": "stock_amount",
            "valid_bars": "stock_valid_bars",
            "zero_bars": "stock_zero_bars",
            "ma90min": "stock_ma90min",
        }
    )
    panel = cb.merge(stock, on=keys, how="inner", validate="one_to_one")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    panel["source_vendor"] = "jqdata_canonical_5m_daily_aggregate"
    panel["data_mode"] = "scheduled_backfill"
    panel["point_in_time_price_data"] = True
    return panel


def load_training_sample_identity(conn, sample_version: str) -> pd.DataFrame:
    sample = query_to_dataframe(
        conn,
        f"SELECT * FROM training_cb_sample WHERE sample_version = {sql_string(sample_version)}",
    )
    if sample.empty:
        return sample
    sample["selected_at"] = pd.to_datetime(sample["selected_at"], errors="coerce")
    sample = sample.sort_values(["bond_code", "selected_at"]).drop_duplicates("bond_code", keep="last")
    keep = [
        "bond_code",
        "stock_code",
        "bond_name",
        "stock_name",
        "maturity_date",
        "selected_at",
        "snapshot_date",
        "source_vendor",
    ]
    return sample[[col for col in keep if col in sample.columns]]


def load_wind_daily_turnover(
    conn,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    where = _date_where(start_date, end_date)
    sql = """
    SELECT bond_code, trade_date, LAST(live_turnover) AS daily_turnover,
           LAST(live_turnover_source) AS daily_turnover_source,
           LAST(ytm) AS ytm, LAST(ytm_source) AS ytm_source
    FROM wind_cb_ytm_turnover_raw
    """
    if where:
        sql += " WHERE " + where
    sql += " GROUP BY bond_code, trade_date"
    result = query_to_dataframe(conn, sql)
    if not result.empty:
        result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    return result


def load_stock_daily_unadjusted_support(
    conn,
    sample_version: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    where = [f"sample_version = {sql_string(sample_version)}", "fq_mode = 'none'"]
    date_where = _date_where(start_date, end_date)
    if date_where:
        where.append(date_where)
    result = query_to_dataframe(
        conn,
        f"""
        SELECT stock_code, trade_date, LAST(close) AS stock_close_unadjusted,
               LAST(is_filled) AS stock_close_unadjusted_filled,
               LAST(price_status) AS stock_close_unadjusted_status
        FROM jq_stock_daily_unadjusted_support
        WHERE {' AND '.join(where)}
        GROUP BY stock_code, trade_date
        """,
    )
    if not result.empty:
        result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
        result["stock_close_unadjusted"] = pd.to_numeric(
            result["stock_close_unadjusted"], errors="coerce"
        )
    return result


def load_screening_snapshots(
    conn,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    raw = query_to_dataframe(conn, "SELECT * FROM cb_screening_daily")
    if raw.empty:
        return raw
    raw["snapshot_time"] = pd.to_datetime(raw["ts"], errors="coerce")
    raw["trade_date"] = raw["snapshot_time"].dt.normalize()
    if start_date:
        raw = raw[raw["trade_date"] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        raw = raw[raw["trade_date"] <= pd.Timestamp(end_date).normalize()]
    raw = raw.sort_values(["trade_date", "bond_code", "snapshot_time"]).drop_duplicates(
        ["trade_date", "bond_code"], keep="last"
    )
    rename = {
        "bond_price": "bond_close",
        "stock_price": "stock_close",
        "debt_asset_ratio": "debt_ratio",
        "premium_rate": "premium",
    }
    raw = raw.rename(columns=rename)
    raw["bond_open"] = raw["bond_close"]
    raw["bond_high"] = raw["bond_close"]
    raw["bond_low"] = raw["bond_close"]
    raw["stock_open"] = raw["stock_close"]
    raw["stock_high"] = raw["stock_close"]
    raw["stock_low"] = raw["stock_close"]
    for column in [
        "bond_volume",
        "bond_amount",
        "stock_volume",
        "stock_amount",
        "daily_turnover",
        "cb_ma90min",
        "stock_ma90min",
    ]:
        raw[column] = np.nan
    raw["bond_valid_bars"] = np.nan
    raw["bond_zero_bars"] = np.nan
    raw["stock_valid_bars"] = np.nan
    raw["stock_zero_bars"] = np.nan
    raw["research_pool_scope"] = raw.get("pool_state", "FULL_MARKET_SCREENING")
    raw["source_vendor"] = raw.get("source", "akshare_screening")
    raw["data_mode"] = "daily_screening_snapshot"
    raw["point_in_time_price_data"] = True
    raw["static_snapshot_time"] = raw["snapshot_time"]
    raw["static_asof_valid"] = True
    raw["parity"] = raw.get("conversion_value")
    raw["days_to_maturity"] = pd.to_numeric(raw.get("days_to_maturity"), errors="coerce")
    keep = [
        "trade_date",
        "bond_code",
        "stock_code",
        "bond_name",
        "stock_name",
        "research_pool_scope",
        "bond_open",
        "bond_high",
        "bond_low",
        "bond_close",
        "bond_volume",
        "bond_amount",
        "bond_valid_bars",
        "bond_zero_bars",
        "stock_open",
        "stock_high",
        "stock_low",
        "stock_close",
        "stock_volume",
        "stock_amount",
        "stock_valid_bars",
        "stock_zero_bars",
        "daily_turnover",
        "premium",
        "parity",
        "conversion_price",
        "maturity_date",
        "days_to_maturity",
        "debt_ratio",
        "operating_cash_flow",
        "free_cash_flow",
        "source_vendor",
        "data_mode",
        "point_in_time_price_data",
        "static_snapshot_time",
        "static_asof_valid",
    ]
    return raw[[column for column in keep if column in raw.columns]].copy()


def load_conversion_price_history(
    conn,
    sample_version: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    where = [f"sample_version = {sql_string(sample_version)}"]
    if start_date:
        where.append(f"effective_end_exclusive > {sql_string(start_date)}")
    if end_date:
        where.append(f"effective_start <= {sql_string(end_date)}")
    result = query_to_dataframe(
        conn,
        f"""
        SELECT bond_code, stock_code, effective_start, effective_end_exclusive,
               conversion_price, source_vendor, price_source, confidence
        FROM cb_conversion_price_history_pit
        WHERE {' AND '.join(where)}
        """,
    )
    if result.empty:
        return result
    result["effective_start"] = pd.to_datetime(result["effective_start"], errors="coerce").dt.normalize()
    result["effective_end_exclusive"] = pd.to_datetime(
        result["effective_end_exclusive"], errors="coerce"
    ).dt.normalize()
    result["conversion_price"] = pd.to_numeric(result["conversion_price"], errors="coerce")
    return result.dropna(subset=["bond_code", "effective_start", "conversion_price"]).copy()


def attach_point_in_time_conversion_price(
    historical: pd.DataFrame,
    conversion_history: pd.DataFrame,
) -> pd.DataFrame:
    out = historical.copy().reset_index(drop=True)
    if "conversion_price" not in out.columns:
        out["conversion_price"] = np.nan
    out["conversion_price_source"] = np.where(
        pd.to_numeric(out["conversion_price"], errors="coerce").notna(),
        "daily_screening_snapshot",
        None,
    )
    out["conversion_price_asof_valid"] = pd.to_numeric(
        out["conversion_price"], errors="coerce"
    ).notna()
    if conversion_history.empty or out.empty:
        return out

    left = out[["bond_code", "trade_date"]].copy()
    left["_row_id"] = left.index
    candidates = left.merge(conversion_history, on="bond_code", how="left")
    valid = (
        candidates["effective_start"].notna()
        & (candidates["effective_start"] <= candidates["trade_date"])
        & (
            candidates["effective_end_exclusive"].isna()
            | (candidates["trade_date"] < candidates["effective_end_exclusive"])
        )
    )
    matched = (
        candidates[valid]
        .sort_values(["_row_id", "effective_start"])
        .drop_duplicates("_row_id", keep="last")
        .set_index("_row_id")
    )
    if matched.empty:
        return out
    row_ids = matched.index.to_numpy()
    existing = pd.to_numeric(out.loc[row_ids, "conversion_price"], errors="coerce")
    fill_mask = existing.isna().to_numpy()
    fill_ids = row_ids[fill_mask]
    if len(fill_ids):
        selected = matched.loc[fill_ids]
        out.loc[fill_ids, "conversion_price"] = selected["conversion_price"].to_numpy()
        out.loc[fill_ids, "conversion_price_source"] = (
            selected["source_vendor"].astype(str) + ":" + selected["price_source"].astype(str)
        ).to_numpy()
        out.loc[fill_ids, "conversion_price_asof_valid"] = True
        out.loc[fill_ids, "conversion_price_effective_start"] = selected[
            "effective_start"
        ].to_numpy()
        out.loc[fill_ids, "conversion_price_effective_end_exclusive"] = selected[
            "effective_end_exclusive"
        ].to_numpy()
        out.loc[fill_ids, "conversion_price_confidence"] = selected["confidence"].to_numpy()
    return out


def derive_point_in_time_parity_and_premium(panel: pd.DataFrame) -> pd.DataFrame:
    """Build pricing fields only when same-day unadjusted inputs are available."""
    out = panel.copy()
    for column in ["qiv_stock_close", "parity", "premium", "conversion_price"]:
        if column not in out.columns:
            out[column] = np.nan
    if "qiv_stock_close_source" not in out.columns:
        out["qiv_stock_close_source"] = None

    screening = out.get(
        "data_mode", pd.Series(index=out.index, dtype=object)
    ).eq("daily_screening_snapshot")
    same_day_screening_price = (
        screening
        & pd.to_numeric(out["qiv_stock_close"], errors="coerce").isna()
        & pd.to_numeric(out.get("stock_close"), errors="coerce").gt(0)
    )
    out.loc[same_day_screening_price, "qiv_stock_close"] = pd.to_numeric(
        out.loc[same_day_screening_price, "stock_close"], errors="coerce"
    )
    out.loc[same_day_screening_price, "qiv_stock_close_source"] = (
        "daily_screening_snapshot_spot"
    )

    pricing_stock = pd.to_numeric(out["qiv_stock_close"], errors="coerce")
    conversion = pd.to_numeric(out["conversion_price"], errors="coerce")
    static_valid = out.get(
        "static_asof_valid", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    out["qiv_stock_close_asof_valid"] = pricing_stock.gt(0)
    conversion_valid = out.get(
        "conversion_price_asof_valid",
        pd.Series(False, index=out.index),
    ).fillna(False).astype(bool)
    conversion_valid = conversion_valid | (conversion.gt(0) & static_valid)
    out["conversion_price_asof_valid"] = conversion_valid
    if "conversion_price_source" not in out.columns:
        out["conversion_price_source"] = None
    screening_conversion = conversion.gt(0) & static_valid & out[
        "conversion_price_source"
    ].isna()
    out.loc[screening_conversion, "conversion_price_source"] = (
        "daily_screening_snapshot"
    )
    derivable = pricing_stock.gt(0) & conversion.gt(0) & conversion_valid
    derived_parity = 100.0 * pricing_stock / conversion

    existing_parity = pd.to_numeric(out["parity"], errors="coerce")
    out["parity_source"] = np.where(
        existing_parity.gt(0) & static_valid,
        "daily_screening_snapshot",
        None,
    )
    fill_parity = existing_parity.isna() & derivable
    out.loc[fill_parity, "parity"] = derived_parity.loc[fill_parity]
    out.loc[fill_parity, "parity_source"] = (
        "pit_conversion_price+jqdata_fq_none_stock_close"
    )
    out["parity_asof_valid"] = (
        pd.to_numeric(out["parity"], errors="coerce").gt(0)
        & (static_valid | derivable)
    )

    parity = pd.to_numeric(out["parity"], errors="coerce")
    bond_price = pd.to_numeric(out.get("bond_close"), errors="coerce")
    existing_premium = pd.to_numeric(out["premium"], errors="coerce")
    out["premium_source"] = np.where(
        existing_premium.notna() & static_valid,
        "daily_screening_snapshot",
        None,
    )
    fill_premium = existing_premium.isna() & parity.gt(0) & bond_price.gt(0) & out[
        "parity_asof_valid"
    ]
    out.loc[fill_premium, "premium"] = (
        bond_price.loc[fill_premium] / parity.loc[fill_premium] - 1.0
    ) * 100.0
    out.loc[fill_premium, "premium_source"] = "derived_from_point_in_time_parity"
    out["premium_asof_valid"] = (
        pd.to_numeric(out["premium"], errors="coerce").notna()
        & (static_valid | out["parity_asof_valid"])
    )
    out["pricing_inputs_asof_valid"] = (
        conversion_valid
        & out["qiv_stock_close_asof_valid"]
        & out["parity_asof_valid"]
        & out["premium_asof_valid"]
    )
    return out


def attach_point_in_time_static(
    historical: pd.DataFrame,
    screening: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    out = historical.copy()
    for column in [
        "premium",
        "parity",
        "conversion_price",
        "maturity_date",
        "days_to_maturity",
        "debt_ratio",
        "operating_cash_flow",
        "free_cash_flow",
        "static_snapshot_time",
    ]:
        if column not in out.columns:
            out[column] = np.nan
    out["static_asof_valid"] = False
    if screening.empty:
        return out
    enrich_columns = [
        "trade_date",
        "bond_code",
        "premium",
        "parity",
        "conversion_price",
        "maturity_date",
        "days_to_maturity",
        "debt_ratio",
        "operating_cash_flow",
        "free_cash_flow",
        "static_snapshot_time",
    ]
    exact = screening[[col for col in enrich_columns if col in screening.columns]].copy()
    exact = exact.drop_duplicates(["trade_date", "bond_code"], keep="last")
    out = out.drop(columns=[col for col in enrich_columns[2:] if col in out.columns]).merge(
        exact,
        on=["trade_date", "bond_code"],
        how="left",
    )
    out["static_snapshot_time"] = pd.to_datetime(out["static_snapshot_time"], errors="coerce")
    out["static_asof_valid"] = (
        out["static_snapshot_time"].notna()
        & (out["static_snapshot_time"].dt.normalize() <= out["trade_date"])
    )
    invalid = ~out["static_asof_valid"]
    for column in enrich_columns[2:-1]:
        out.loc[invalid, column] = np.nan
    return out


def build_daily_research_panel(
    conn,
    start_date: str | None,
    end_date: str | None,
    config: dict[str, Any],
    include_screening_extension: bool = True,
) -> pd.DataFrame:
    sample_version = str(config["strategy"]["sample_version"])
    historical = load_training_daily_panel(conn, start_date, end_date, sample_version)
    screening = load_screening_snapshots(conn, start_date, end_date)
    if historical.empty and screening.empty:
        return pd.DataFrame()
    if not historical.empty:
        historical = attach_point_in_time_static(historical, screening, config)
        conversion_history = load_conversion_price_history(
            conn,
            sample_version,
            start_date,
            end_date,
        )
        historical = attach_point_in_time_conversion_price(historical, conversion_history)
        unadjusted_stock = load_stock_daily_unadjusted_support(
            conn,
            sample_version,
            start_date,
            end_date,
        )
        if not unadjusted_stock.empty:
            historical = historical.merge(
                unadjusted_stock,
                on=["trade_date", "stock_code"],
                how="left",
                validate="many_to_one",
            )
            historical["qiv_stock_close"] = historical["stock_close_unadjusted"]
            historical["qiv_stock_close_source"] = (
                "jqdata_fq_none_daily:" + historical["stock_close_unadjusted_status"].astype(str)
            )
        turnover = load_wind_daily_turnover(conn, start_date, end_date)
        if not turnover.empty:
            historical = historical.merge(turnover, on=["trade_date", "bond_code"], how="left")
        identity = load_training_sample_identity(conn, sample_version)
        if not identity.empty:
            identity = identity.rename(
                columns={
                    "bond_name": "identity_bond_name",
                    "stock_name": "identity_stock_name",
                    "maturity_date": "identity_maturity_date",
                    "source_vendor": "identity_source_vendor",
                }
            )
            historical = historical.merge(
                identity.drop(columns=["stock_code"], errors="ignore"),
                on="bond_code",
                how="left",
            )
            historical["bond_name"] = historical.get("bond_name", pd.Series(index=historical.index, dtype=object)).combine_first(
                historical.get("identity_bond_name")
            )
            historical["stock_name"] = historical.get("stock_name", pd.Series(index=historical.index, dtype=object)).combine_first(
                historical.get("identity_stock_name")
            )
            historical["maturity_date"] = pd.to_datetime(
                historical.get("maturity_date"), errors="coerce"
            ).combine_first(
                pd.to_datetime(historical.get("identity_maturity_date"), errors="coerce")
            )
            historical["days_to_maturity"] = (
                historical["maturity_date"].dt.normalize() - historical["trade_date"]
            ).dt.days
            historical["maturity_date_source"] = np.where(
                historical["maturity_date"].notna(),
                "akshare_training_sample_static",
                "unavailable",
            )
    pieces = [historical] if not historical.empty else []
    if include_screening_extension and not screening.empty:
        historical_keys = set(
            zip(historical["trade_date"], historical["bond_code"])
        ) if not historical.empty else set()
        extension = screening[
            ~screening.apply(lambda row: (row["trade_date"], row["bond_code"]) in historical_keys, axis=1)
        ].copy()
        if not extension.empty:
            pieces.append(extension)
    panel = pd.concat(pieces, ignore_index=True, sort=False)
    panel = panel.sort_values(["bond_code", "trade_date"]).drop_duplicates(
        ["trade_date", "bond_code"], keep="last"
    )
    panel["ytm"] = pd.to_numeric(panel.get("ytm"), errors="coerce")
    panel["ytm_available"] = panel["ytm"].notna()
    panel["ytm_proxy_used"] = False
    panel["static_snapshot_future_rejected"] = ~panel["static_asof_valid"].fillna(False)
    panel = derive_point_in_time_parity_and_premium(panel)
    return add_daily_features(panel)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
