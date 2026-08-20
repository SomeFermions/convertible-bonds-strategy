from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from db import get_conn
from intraday_factors.data_source import query_to_dataframe, sql_string
from intraday_factors.trading_session import is_session_bar
from training_data_v06.config import load_config
from training_data_v06.providers import (
    JQ_CB_FIELD_WHITELIST,
    JQ_STOCK_FIELD_WHITELIST,
    build_jqdata_provider,
    build_wind_provider,
    jq_code_from_bond,
    jq_code_from_stock,
    validate_jq_fields,
    validate_wind_fields,
)
from training_data_v06.schemas import (
    AKSHARE_STATIC_TABLE,
    BENCHMARK_CANONICAL_TABLE,
    BENCHMARK_COVERAGE_TABLE,
    BENCHMARK_PLAN_TABLE,
    CANONICAL_TABLE,
    CONVERSION_PRICE_COVERAGE_TABLE,
    CONVERSION_PRICE_HISTORY_TABLE,
    COVERAGE_TABLE,
    DOWNLOAD_PLAN_TABLE,
    JQ_CB_RAW_TABLE,
    JQ_BENCHMARK_RAW_TABLE,
    JQ_STOCK_DAILY_UNADJUSTED_TABLE,
    JQ_STOCK_RAW_TABLE,
    QUOTA_JOB_TABLE,
    QUOTA_LEDGER_TABLE,
    TRAINING_SAMPLE_TABLE,
    WIND_RAW_TABLE,
    WINDOW_MANIFEST_TABLE,
    ensure_training_tables,
    insert_rows,
)


LOGGER = logging.getLogger(__name__)
SOURCE_VENDOR_JQ = "jqdata"
SOURCE_VENDOR_WIND = "wind"
SOURCE_VENDOR_AKSHARE = "akshare"
SELECTION_VERSION = "v0.6_sample_selection_001"
RUNNABLE_JQ_PLAN_STATUS = "READY"


def now_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC)).tz_convert(None)


def bool_arg(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def stable_ts(base: pd.Timestamp, *parts: Any) -> pd.Timestamp:
    offset_ms = zlib.crc32(":".join(str(part) for part in parts).encode("utf-8")) % 300000
    return pd.Timestamp(base).tz_localize(None) + pd.Timedelta(milliseconds=offset_ms)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def raw_payload_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def week_id(value: pd.Timestamp | None = None) -> str:
    ts = pd.Timestamp(value or now_ts())
    iso = ts.isocalendar()
    return f"{int(iso.year)}W{int(iso.week):02d}"


def available_trial_range(asof: str | pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(asof or now_ts()).normalize() - pd.DateOffset(months=3)
    start = pd.Timestamp(asof or now_ts()).normalize() - pd.DateOffset(months=15)
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def trade_calendar_cache_path(config: dict[str, Any]) -> Path:
    configured = Path(config["jqdata"].get("trade_calendar_cache", "outputs/cn_trade_calendar.csv")).expanduser()
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[1] / configured


def trading_dates_between(
    config: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[str], str]:
    cache_path = trade_calendar_cache_path(config)
    if cache_path.exists():
        calendar = pd.read_csv(cache_path)
        if "trade_date" not in calendar.columns:
            raise ValueError(f"trade calendar cache lacks trade_date: {cache_path}")
        values = pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().dt.normalize()
        selected = values[(values >= pd.Timestamp(start).normalize()) & (values <= pd.Timestamp(end).normalize())]
        dates = sorted(set(selected.dt.strftime("%Y-%m-%d")))
        if dates:
            return dates, f"akshare_cache:{cache_path}"
    return business_dates(start, end), "weekday_fallback"


def refresh_cn_trade_calendar(config: dict[str, Any]) -> dict[str, Any]:
    import akshare as ak

    raw = ak.tool_trade_date_hist_sina()
    if raw is None or raw.empty or "trade_date" not in raw.columns:
        raise RuntimeError("AkShare returned an empty CN trade calendar")
    dates = pd.to_datetime(raw["trade_date"], errors="coerce").dropna().dt.normalize()
    out = pd.DataFrame({"trade_date": sorted(set(dates.dt.strftime("%Y-%m-%d")))})
    path = trade_calendar_cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return {
        "mode": "refresh-cn-trade-calendar",
        "source_vendor": SOURCE_VENDOR_AKSHARE,
        "cache_path": str(path),
        "trading_days": int(len(out)),
        "first_trade_date": out.iloc[0]["trade_date"],
        "last_trade_date": out.iloc[-1]["trade_date"],
    }


def permission_safe_trial_range(
    config: dict[str, Any],
    asof: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    jq_cfg = config["jqdata"]
    anchor = asof or jq_cfg.get("trial_window_anchor_date")
    start, end = available_trial_range(anchor)
    buffer_days = int(jq_cfg.get("trial_download_start_buffer_days", 14))
    if buffer_days < 0:
        raise ValueError("trial_download_start_buffer_days cannot be negative")

    start = start + pd.Timedelta(days=buffer_days)

    # Optional overrides may tighten a plan, but may never widen it beyond the
    # moving permission-safe trial interval.
    if jq_cfg.get("available_start_date"):
        configured_start = pd.Timestamp(jq_cfg["available_start_date"]).normalize()
        start = max(start, configured_start)
    if jq_cfg.get("available_end_date"):
        configured_end = pd.Timestamp(jq_cfg["available_end_date"]).normalize()
        end = min(end, configured_end)
    safe_dates, _ = trading_dates_between(config, start, end)
    if not safe_dates:
        raise ValueError(f"no trading dates in permission-safe JQData range start={start.date()} end={end.date()}")
    start = pd.Timestamp(safe_dates[0])
    end = pd.Timestamp(safe_dates[-1])
    if start > end:
        raise ValueError(f"invalid permission-safe JQData range start={start.date()} end={end.date()}")
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def business_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [ts.strftime("%Y-%m-%d") for ts in pd.bdate_range(start, end)]


def selected_training_dates(config: dict[str, Any], asof: str | pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    start, end = permission_safe_trial_range(config, asof)
    jq_cfg = config["jqdata"]
    dates, _ = trading_dates_between(config, start, end)
    target_days = int(jq_cfg.get("target_trading_days", 200))
    if len(dates) > target_days:
        dates = dates[-target_days:]
    if dates:
        start = pd.Timestamp(dates[0])
        end = pd.Timestamp(dates[-1])
    return start, end, dates


def chunk_dates(dates: list[str], chunk_size: int) -> list[tuple[str, str, int]]:
    out = []
    for offset in range(0, len(dates), max(1, int(chunk_size))):
        chunk = dates[offset:offset + max(1, int(chunk_size))]
        if chunk:
            out.append((chunk[0], chunk[-1], len(chunk)))
    return out


def estimate_rows(instrument_count: int, trading_days: int, bars_per_day: int) -> int:
    return int(instrument_count) * int(trading_days) * int(bars_per_day)


def require_smoke_scope(bond_codes: list[str], days: int, config: dict[str, Any]) -> None:
    jq_cfg = config["jqdata"]
    if len(bond_codes) > int(jq_cfg["max_smoke_bonds"]):
        raise ValueError("jq-smoke-test is limited to at most 2 convertible bonds")
    if int(days) > int(jq_cfg["max_smoke_days"]):
        raise ValueError("jq-smoke-test is limited to at most 2 trading days")


def require_large_download_confirmation(
    instrument_count: int,
    estimated: int,
    confirm_large_download: bool,
    config: dict[str, Any],
) -> None:
    jq_cfg = config["jqdata"]
    if instrument_count > int(jq_cfg["large_download_instrument_threshold"]) or estimated > int(jq_cfg["large_download_row_threshold"]):
        if not confirm_large_download:
            raise ValueError(
                "large JQData download refused without --confirm-large-download "
                f"(instruments={instrument_count}, estimated_rows={estimated})"
            )


def split_jq_fields(fields: list[str], config: dict[str, Any]) -> tuple[list[str], list[str]]:
    requested = [item.strip() for item in fields if item and item.strip()]
    if not requested:
        requested = list(config["jqdata"].get("cb_fields", []))
    allowed_any = JQ_CB_FIELD_WHITELIST | JQ_STOCK_FIELD_WHITELIST
    invalid = sorted(set(requested) - allowed_any)
    if invalid:
        raise ValueError(f"JQData fields not allowed: {invalid}")

    cb_fields = [field for field in requested if field in JQ_CB_FIELD_WHITELIST]
    stock_fields = [field for field in requested if field in JQ_STOCK_FIELD_WHITELIST]

    for field in config["jqdata"].get("stock_fields", []):
        if field in JQ_STOCK_FIELD_WHITELIST and field not in stock_fields:
            stock_fields.append(field)
    for field in config["jqdata"].get("cb_fields", []):
        if field in JQ_CB_FIELD_WHITELIST and field not in cb_fields:
            cb_fields.append(field)

    return validate_jq_fields("CB", cb_fields), validate_jq_fields("STOCK", stock_fields)


def jq_plan_can_run(status: Any, attempt_count: Any, max_attempts: int) -> bool:
    if str(status) != RUNNABLE_JQ_PLAN_STATUS:
        return False
    attempts = pd.to_numeric(attempt_count, errors="coerce")
    attempts_int = 0 if pd.isna(attempts) else int(attempts)
    return attempts_int < int(max_attempts)


def source_snapshot(conn) -> pd.DataFrame:
    errors: list[str] = []
    for table in ("cb_screening_daily", "cb_watchlist_daily"):
        try:
            snap = query_to_dataframe(conn, f"SELECT LAST(snapshot_id) AS snapshot_id FROM {table}")
            if snap.empty or pd.isna(snap.loc[0, "snapshot_id"]):
                df = query_to_dataframe(conn, f"SELECT * FROM {table} ORDER BY ts DESC LIMIT 1000")
            else:
                snapshot_id = str(snap.loc[0, "snapshot_id"])
                df = query_to_dataframe(conn, f"SELECT * FROM {table} WHERE snapshot_id = {sql_string(snapshot_id)}")
            if not df.empty:
                return df
        except Exception as exc:
            message = str(exc)
            errors.append(f"{table}: {message}")
            if "License expired" in message or "Invalid user" in message or "Unable to establish connection" in message:
                raise RuntimeError(f"failed to read local slow-data snapshot: {message}") from exc
            continue
    if errors:
        LOGGER.warning("No local snapshot loaded; suppressed source errors: %s", "; ".join(errors))
    return pd.DataFrame()


def non_st(row: pd.Series) -> bool:
    for col in ("st_status", "stock_name"):
        value = row.get(col)
        if value is not None and not pd.isna(value) and "ST" in str(value).upper():
            return False
    return True


def classify_sample(row: pd.Series, config: dict[str, Any]) -> tuple[str | None, str, str | None, float]:
    cfg = config["sample_selection"]
    reasons: list[str] = []
    exclusions: list[str] = []
    score = 0.0

    stock_code = str(row.get("stock_code") or "").strip()
    premium = pd.to_numeric(row.get("premium_rate"), errors="coerce")
    days_to_maturity = pd.to_numeric(row.get("days_to_maturity"), errors="coerce")
    debt = pd.to_numeric(row.get("debt_asset_ratio"), errors="coerce")
    ocf = pd.to_numeric(row.get("operating_cash_flow"), errors="coerce")
    fcf = pd.to_numeric(row.get("free_cash_flow"), errors="coerce")

    if not stock_code:
        exclusions.append("missing_stock_code")
    if not non_st(row):
        exclusions.append("stock_st")
    if pd.notna(days_to_maturity) and days_to_maturity <= int(cfg["min_days_to_maturity"]):
        exclusions.append("near_maturity")
    if pd.notna(debt) and debt >= float(cfg["severe_debt_asset_ratio"]):
        exclusions.append("severe_debt_asset_ratio")
    if pd.isna(premium):
        exclusions.append("missing_premium")

    if exclusions:
        return None, "", ",".join(exclusions), -1.0

    active_financial_ok = True
    if pd.notna(debt) and debt >= float(cfg["max_debt_asset_ratio"]):
        active_financial_ok = False
    if pd.notna(ocf) and ocf <= 0:
        active_financial_ok = False
    if pd.notna(fcf) and fcf <= 0:
        active_financial_ok = False

    if float(cfg["active_like_premium_min"]) <= float(premium) <= float(cfg["active_like_premium_max"]) and active_financial_ok:
        reasons.append("premium_20_30")
        reasons.append("basic_financial_pass_or_unavailable")
        score = 300.0 - abs(float(premium) - 25.0)
        return "ACTIVE_LIKE", ";".join(reasons), None, score

    if float(cfg["shadow_premium_min"]) <= float(premium) <= float(cfg["shadow_premium_max"]):
        reasons.append("eligible_shadow_premium_10_40")
        if not active_financial_ok:
            reasons.append("active_like_financial_not_full_pass")
        score = 200.0 - abs(float(premium) - 25.0)
        return "ELIGIBLE_SHADOW", ";".join(reasons), None, score

    if float(cfg["broad_premium_min"]) <= float(premium) <= float(cfg["broad_premium_max"]):
        reasons.append("broad_but_clean")
        score = 100.0 - abs(float(premium) - 25.0) * 0.1
        return "BROAD_BUT_CLEAN", ";".join(reasons), None, score

    return None, "", "premium_out_of_clean_range", -1.0


def build_training_sample(conn, sample_size: int, sample_version: str, config: dict[str, Any]) -> dict[str, Any]:
    ensure_training_tables(conn)
    raw = source_snapshot(conn)
    if raw.empty:
        return {"sample_version": sample_version, "selected_count": 0, "reason": "no_local_screening_snapshot"}

    for col in ["bond_code", "stock_code", "bond_name", "stock_name"]:
        if col not in raw.columns:
            raw[col] = ""
    raw["bond_code"] = raw["bond_code"].astype(str).str.zfill(6)
    raw["stock_code"] = raw["stock_code"].astype(str).str.zfill(6)
    raw = raw.drop_duplicates(subset=["bond_code"], keep="last").copy()

    classified = []
    excluded = 0
    for _, row in raw.iterrows():
        scope, reason, exclusion, score = classify_sample(row, config)
        if scope is None:
            excluded += 1
            continue
        item = row.to_dict()
        item.update(
            {
                "research_pool_scope": scope,
                "sample_selection_reason": reason,
                "exclusion_reason": exclusion,
                "_score": score,
            }
        )
        classified.append(item)

    selected = pd.DataFrame(classified)
    if selected.empty:
        return {"sample_version": sample_version, "selected_count": 0, "excluded_count": excluded}
    selected = selected.sort_values(["_score", "bond_code"], ascending=[False, True]).head(int(sample_size)).copy()

    existing = query_to_dataframe(
        conn,
        f"SELECT bond_code FROM {TRAINING_SAMPLE_TABLE} WHERE sample_version = {sql_string(sample_version)}",
    )
    existing_codes = set(existing["bond_code"].astype(str)) if not existing.empty else set()
    selected_at = now_ts()
    rows = []
    for row_index, (_, row) in enumerate(selected.iterrows()):
        bond_code = str(row.get("bond_code")).zfill(6)
        if bond_code in existing_codes:
            continue
        rows.append(
            {
                "ts": selected_at + pd.Timedelta(milliseconds=row_index),
                "sample_version": sample_version,
                "selection_version": SELECTION_VERSION,
                "selected_at": selected_at,
                "bond_code": bond_code,
                "stock_code": str(row.get("stock_code")).zfill(6),
                "bond_name": row.get("bond_name"),
                "stock_name": row.get("stock_name"),
                "research_pool_scope": row.get("research_pool_scope"),
                "sample_selection_reason": row.get("sample_selection_reason"),
                "exclusion_reason": row.get("exclusion_reason"),
                "premium_rate": row.get("premium_rate"),
                "days_to_maturity": row.get("days_to_maturity"),
                "debt_asset_ratio": row.get("debt_asset_ratio"),
                "operating_cash_flow": row.get("operating_cash_flow"),
                "free_cash_flow": row.get("free_cash_flow"),
                "maturity_date": row.get("maturity_date"),
                "conversion_price": row.get("conversion_price"),
                "source_vendor": SOURCE_VENDOR_AKSHARE,
                "snapshot_date": row.get("ts"),
            }
        )
    inserted = insert_rows(conn, TRAINING_SAMPLE_TABLE, rows)
    counts = selected["research_pool_scope"].value_counts().to_dict()
    return {
        "sample_version": sample_version,
        "selected_count": int(len(selected)),
        "inserted_count": int(inserted),
        "excluded_count": int(excluded),
        "pool_counts": {str(key): int(value) for key, value in counts.items()},
    }


def refresh_akshare_static_cache_from_local(conn) -> dict[str, Any]:
    ensure_training_tables(conn)
    raw = source_snapshot(conn)
    if raw.empty:
        return {"mode": "refresh-akshare-static-cache", "inserted_rows": 0, "reason": "no_local_screening_snapshot"}
    now = now_ts()
    snapshot_date = now.strftime("%Y-%m-%d")
    rows = []
    for row_index, (_, item) in enumerate(raw.drop_duplicates(subset=["bond_code"], keep="last").iterrows()):
        bond_code = str(item.get("bond_code")).zfill(6)
        stock_code = str(item.get("stock_code")).zfill(6)
        stock_name = item.get("stock_name")
        st_status = "ST" if stock_name is not None and not pd.isna(stock_name) and "ST" in str(stock_name).upper() else "NORMAL"
        rows.append(
            {
                "ts": now + pd.Timedelta(milliseconds=row_index),
                "source_vendor": SOURCE_VENDOR_AKSHARE,
                "snapshot_date": snapshot_date,
                "bond_code": bond_code,
                "stock_code": stock_code,
                "maturity_date": item.get("maturity_date"),
                "st_status": st_status,
                "conversion_price": item.get("conversion_price"),
                "conversion_price_source": "akshare_static",
                "bond_name": item.get("bond_name"),
                "stock_name": stock_name,
                "listing_date": item.get("listed_date") if "listed_date" in item else item.get("listing_date"),
                "delisting_date": item.get("delisting_date"),
                "downloaded_at": now,
            }
        )
    inserted = insert_rows(conn, AKSHARE_STATIC_TABLE, rows)
    return {"mode": "refresh-akshare-static-cache", "source": "local_slow_snapshot", "inserted_rows": int(inserted)}


def load_sample(conn, sample_version: str) -> pd.DataFrame:
    df = query_to_dataframe(conn, f"SELECT * FROM {TRAINING_SAMPLE_TABLE} WHERE sample_version = {sql_string(sample_version)}")
    if df.empty:
        return df
    df["bond_code"] = df["bond_code"].astype(str).str.zfill(6)
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df


def raw_table(asset_type: str) -> str:
    return JQ_CB_RAW_TABLE if asset_type.upper() == "CB" else JQ_STOCK_RAW_TABLE


def sample_instrument_records(sample: pd.DataFrame) -> list[dict[str, Any]]:
    instruments: dict[tuple[str, str], dict[str, Any]] = {}
    ordered = sample.sort_values(["bond_code", "stock_code"]).copy()
    for _, item in ordered.iterrows():
        bond_code = str(item["bond_code"]).zfill(6)
        stock_code = str(item["stock_code"]).zfill(6)
        priority = 0 if item.get("research_pool_scope") == "ACTIVE_LIKE" else 1
        candidates = (
            ("CB", jq_code_from_bond(bond_code)),
            ("STOCK", jq_code_from_stock(stock_code)),
        )
        for asset_type, vendor_code in candidates:
            key = (asset_type, vendor_code)
            if key not in instruments:
                instruments[key] = {
                    "asset_type": asset_type,
                    "instrument_code": vendor_code,
                    "bond_code": bond_code,
                    "stock_code": stock_code,
                    "priority": priority,
                }
            else:
                instruments[key]["priority"] = min(int(instruments[key]["priority"]), priority)
    records = []
    for slot, key in enumerate(sorted(instruments)):
        record = dict(instruments[key])
        record["instrument_slot"] = slot
        records.append(record)
    return records


def canonical_association_slots(sample: pd.DataFrame) -> dict[tuple[str, str], int]:
    keys = []
    for bond_code in sorted(sample["bond_code"].astype(str).str.zfill(6).unique()):
        keys.extend((("CB", bond_code), ("STOCK", bond_code)))
    return {key: slot for slot, key in enumerate(sorted(keys))}


def raw_rows_complete(
    conn,
    table: str,
    vendor_code: str,
    start_date: str,
    end_date: str,
    expected_rows: int,
) -> bool:
    try:
        df = query_to_dataframe(
            conn,
            f"""
            SELECT COUNT(*) AS n
            FROM {table}
            WHERE vendor_code = {sql_string(vendor_code)}
              AND trade_date >= {sql_string(start_date)}
              AND trade_date <= {sql_string(end_date)}
            """,
        )
        return not df.empty and int(df.loc[0, "n"] or 0) >= int(expected_rows)
    except Exception:
        return False


def load_raw_coverage(
    conn,
    table: str,
    start_date: str,
    end_date: str,
) -> dict[tuple[str, str], int]:
    try:
        df = query_to_dataframe(
            conn,
            f"""
            SELECT vendor_code, trade_date, COUNT(*) AS n
            FROM {table}
            WHERE trade_date >= {sql_string(start_date)}
              AND trade_date <= {sql_string(end_date)}
            GROUP BY vendor_code, trade_date
            """,
        )
    except Exception:
        return {}
    if df.empty:
        return {}
    return {
        (str(item["vendor_code"]), str(item["trade_date"])[:10]): int(item["n"] or 0)
        for _, item in df.iterrows()
    }


def raw_chunk_complete_from_coverage(
    coverage: dict[tuple[str, str], int],
    vendor_code: str,
    start_date: str,
    end_date: str,
    expected_rows: int,
) -> bool:
    actual = sum(
        count
        for (code, trade_date), count in coverage.items()
        if code == str(vendor_code) and start_date <= trade_date <= end_date
    )
    return actual >= int(expected_rows)


def build_jq_download_plan(
    conn,
    sample_version: str,
    frequency: str,
    fields: list[str],
    dry_run: bool,
    confirm_large_download: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    ensure_training_tables(conn)
    if frequency != "5m":
        raise ValueError("V0.6 only supports JQData 5m training bars")
    sample = load_sample(conn, sample_version)
    if sample.empty:
        return {"sample_version": sample_version, "planned_rows": 0, "reason": "sample_not_found"}

    permission_anchor = pd.Timestamp(
        config["jqdata"].get("trial_window_anchor_date") or now_ts()
    ).normalize()
    permission_safe_start, permission_safe_end = permission_safe_trial_range(config, permission_anchor)
    _, calendar_source = trading_dates_between(config, permission_safe_start, permission_safe_end)
    start, end, dates = selected_training_dates(config)
    jq_cfg = config["jqdata"]
    chunks = chunk_dates(dates, int(jq_cfg["plan_chunk_trading_days"]))
    cb_fields, stock_fields = split_jq_fields(fields, config)
    bars_per_day = int(jq_cfg["bars_per_day_estimate"])
    instruments = sample_instrument_records(sample)
    coverage_by_table = {
        JQ_CB_RAW_TABLE: load_raw_coverage(conn, JQ_CB_RAW_TABLE, dates[0], dates[-1]),
        JQ_STOCK_RAW_TABLE: load_raw_coverage(conn, JQ_STOCK_RAW_TABLE, dates[0], dates[-1]),
    }
    total_estimated = estimate_rows(len(instruments), len(dates), bars_per_day)
    if not dry_run:
        require_large_download_confirmation(len(instruments), total_estimated, confirm_large_download, config)

    batch_id = stable_id("jqbatch", sample_version, frequency, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), ",".join(fields))
    now = now_ts()
    rows = []
    new_plan_rows = 0
    updated_plan_rows = 0
    skipped_existing = 0
    existing_df = query_to_dataframe(
        conn,
        f"SELECT * FROM {DOWNLOAD_PLAN_TABLE} WHERE sample_version = {sql_string(sample_version)}",
    )
    existing_plans = {
        str(item["plan_id"]): item.to_dict()
        for _, item in existing_df.iterrows()
    } if not existing_df.empty else {}
    for instrument in instruments:
        asset_type = instrument["asset_type"]
        vendor_code = instrument["instrument_code"]
        plan_fields = cb_fields if asset_type == "CB" else stock_fields
        table = raw_table(asset_type)
        for start_date, end_date, day_count in chunks:
            plan_id = stable_id("jqplan", sample_version, vendor_code, start_date, end_date, frequency)
            status = "DRY_RUN_ESTIMATED" if dry_run else "READY"
            if raw_chunk_complete_from_coverage(
                coverage_by_table[table],
                vendor_code,
                start_date,
                end_date,
                day_count * bars_per_day,
            ):
                status = "SKIPPED_ALREADY_EXISTS"
                skipped_existing += 1
            if plan_id in existing_plans:
                current = dict(existing_plans[plan_id])
                current_status = str(current.get("status"))
                mutable_preflight_statuses = {"DRY_RUN_ESTIMATED", "SKIPPED_ALREADY_EXISTS"}
                if current_status in mutable_preflight_statuses and current_status != status:
                    current["status"] = status
                    current["updated_at"] = now
                    rows.append(current)
                    updated_plan_rows += 1
                continue
            rows.append(
                {
                    "ts": now + pd.Timedelta(milliseconds=len(rows)),
                    "plan_id": plan_id,
                    "batch_id": batch_id,
                    "data_source": SOURCE_VENDOR_JQ,
                    "research_only": True,
                    "sample_version": sample_version,
                    "permission_anchor_date": permission_anchor.strftime("%Y-%m-%d"),
                    "permission_safe_start": permission_safe_start.strftime("%Y-%m-%d"),
                    "permission_safe_end": permission_safe_end.strftime("%Y-%m-%d"),
                    "instrument_code": vendor_code,
                    "bond_code": instrument["bond_code"],
                    "stock_code": instrument["stock_code"],
                    "asset_type": asset_type,
                    "instrument_slot": instrument["instrument_slot"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "frequency": frequency,
                    "fields_json": json.dumps(plan_fields, ensure_ascii=True),
                    "estimated_rows": day_count * bars_per_day,
                    "priority": instrument["priority"],
                    "status": status,
                    "attempt_count": 0,
                    "last_error": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            new_plan_rows += 1
    inserted = insert_rows(conn, DOWNLOAD_PLAN_TABLE, rows)
    return {
        "sample_version": sample_version,
        "batch_id": batch_id,
        "dry_run": bool(dry_run),
        "permission_anchor_date": permission_anchor.strftime("%Y-%m-%d"),
        "permission_safe_start": permission_safe_start.strftime("%Y-%m-%d"),
        "permission_safe_end": permission_safe_end.strftime("%Y-%m-%d"),
        "selected_start": start.strftime("%Y-%m-%d"),
        "selected_end": end.strftime("%Y-%m-%d"),
        "available_start": permission_safe_start.strftime("%Y-%m-%d"),
        "available_end": permission_safe_end.strftime("%Y-%m-%d"),
        "business_days_estimated": len(dates),
        "trading_days_selected": len(dates),
        "trade_calendar_source": calendar_source,
        "sample_bond_count": int(len(sample)),
        "unique_stock_count": int(sum(1 for item in instruments if item["asset_type"] == "STOCK")),
        "instrument_count": int(len(instruments)),
        "estimated_rows_total": int(total_estimated),
        "inserted_plan_rows": int(inserted),
        "new_plan_rows": int(new_plan_rows),
        "updated_plan_rows": int(updated_plan_rows),
        "skipped_existing_chunks": int(skipped_existing),
    }


def update_plan_status(
    conn,
    plan_id: str,
    status: str,
    error: str | None = None,
    increment_attempt: bool = False,
    current_row: dict[str, Any] | pd.Series | None = None,
) -> dict[str, Any]:
    now = now_ts()
    if current_row is None:
        current = query_to_dataframe(
            conn,
            f"SELECT * FROM {DOWNLOAD_PLAN_TABLE} WHERE plan_id = {sql_string(plan_id)} LIMIT 1",
        )
        if current.empty:
            raise ValueError(f"download plan not found: {plan_id}")
        row = current.iloc[0].to_dict()
    elif isinstance(current_row, pd.Series):
        row = current_row.to_dict()
    else:
        row = dict(current_row)
    attempts = pd.to_numeric(row.get("attempt_count"), errors="coerce")
    attempts_int = 0 if pd.isna(attempts) else int(attempts)
    row["status"] = status
    row["last_error"] = error
    row["updated_at"] = now
    row["attempt_count"] = attempts_int + (1 if increment_attempt else 0)
    insert_rows(conn, DOWNLOAD_PLAN_TABLE, [row])
    return row


def promote_existing_jq_plan_if_needed(conn, plan_id: str, target_status: str, dry_run: bool) -> bool:
    exists = query_to_dataframe(conn, f"SELECT status FROM {DOWNLOAD_PLAN_TABLE} WHERE plan_id = {sql_string(plan_id)} LIMIT 1")
    if exists.empty:
        return False
    current_status = str(exists.loc[0, "status"])
    mutable_preflight_statuses = {"DRY_RUN_ESTIMATED", "SKIPPED_ALREADY_EXISTS"}
    if current_status in mutable_preflight_statuses and current_status != target_status:
        update_plan_status(conn, plan_id, target_status)
    return True


def current_quota_used(conn, source_vendor: str, trade_date: str) -> int:
    try:
        jobs = query_to_dataframe(
            conn,
            f"""
            SELECT SUM(estimated_rows) AS estimated, SUM(actual_rows) AS actual
            FROM {QUOTA_JOB_TABLE}
            WHERE source_vendor = {sql_string(source_vendor)}
              AND trade_date = {sql_string(trade_date)}
              AND status IN ('SUCCESS','PARTIAL_SUCCESS','RUNNING','PLANNED','FAILED')
            """,
        )
        if jobs.empty:
            return 0
        estimated = int(jobs.loc[0, "estimated"] or 0)
        actual = int(jobs.loc[0, "actual"] or 0)
        return max(estimated, actual)
    except Exception:
        return 0


def current_weekly_quota_used(conn, source_vendor: str, current_week_id: str) -> int:
    try:
        jobs = query_to_dataframe(
            conn,
            f"""
            SELECT SUM(estimated_rows) AS estimated, SUM(actual_rows) AS actual
            FROM {QUOTA_JOB_TABLE}
            WHERE source_vendor = {sql_string(source_vendor)}
              AND week_id = {sql_string(current_week_id)}
              AND status IN ('SUCCESS','PARTIAL_SUCCESS','RUNNING','PLANNED','FAILED')
            """,
        )
        if jobs.empty:
            return 0
        estimated = int(jobs.loc[0, "estimated"] or 0)
        actual = int(jobs.loc[0, "actual"] or 0)
        return max(estimated, actual)
    except Exception:
        return 0


def record_quota_job(
    conn,
    source_vendor: str,
    account_id_hash: str,
    mode: str,
    sample_version: str,
    estimated_rows: int,
    actual_rows: int,
    status: str,
    error_message: str | None = None,
) -> str:
    now = now_ts()
    trade_date = now.strftime("%Y-%m-%d")
    job_id = stable_id("quota", source_vendor, mode, sample_version, now.isoformat(), estimated_rows, actual_rows)
    insert_rows(
        conn,
        QUOTA_JOB_TABLE,
        [
            {
                "ts": stable_ts(now, job_id),
                "job_id": job_id,
                "source_vendor": source_vendor,
                "account_id_hash": account_id_hash,
                "trade_date": trade_date,
                "week_id": week_id(now),
                "mode": mode,
                "sample_version": sample_version,
                "estimated_rows": estimated_rows,
                "actual_rows": actual_rows,
                "status": status,
                "error_message": error_message,
                "created_at": now,
                "updated_at": now,
            }
        ],
    )
    return job_id


def record_quota_ledger(
    conn,
    source_vendor: str,
    account_id_hash: str,
    daily_limit: int,
    weekly_limit: int,
    estimated_rows: int,
    actual_rows: int,
    job_id: str,
    failed: bool = False,
) -> None:
    now = now_ts()
    trade_date = now.strftime("%Y-%m-%d")
    used_est = current_quota_used(conn, source_vendor, trade_date)
    if int(daily_limit) > 0:
        quota_remaining = max(0, int(daily_limit) - int(used_est))
    else:
        used_week = current_weekly_quota_used(conn, source_vendor, week_id(now))
        quota_remaining = max(0, int(weekly_limit) - int(used_week))
    insert_rows(
        conn,
        QUOTA_LEDGER_TABLE,
        [
            {
                "ts": stable_ts(now, source_vendor, trade_date, job_id),
                "source_vendor": source_vendor,
                "account_id_hash": account_id_hash,
                "trade_date": trade_date,
                "week_id": week_id(now),
                "quota_limit_daily": daily_limit,
                "quota_limit_weekly": weekly_limit,
                "quota_used_estimated": used_est,
                "quota_used_actual": actual_rows,
                "quota_remaining_estimated": quota_remaining,
                "job_count": 1,
                "failed_job_count": 1 if failed else 0,
                "last_job_id": job_id,
                "created_at": now,
                "updated_at": now,
            }
        ],
    )


def raw_rows_from_jq_frame(plan: pd.Series, frame: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    now = now_ts()
    rows = []
    asset_type = str(plan["asset_type"]).upper()
    account_type = config["jqdata"]["account_type"]
    bar_minutes = int(config["canonical"].get("bar_minutes", 5))
    slot_value = pd.to_numeric(plan.get("instrument_slot"), errors="coerce")
    if pd.isna(slot_value) or int(slot_value) < 0 or int(slot_value) >= bar_minutes * 60 * 1000:
        raise ValueError(f"invalid JQData instrument_slot={plan.get('instrument_slot')}")
    instrument_slot = int(slot_value)
    for _, item in frame.iterrows():
        vendor_bar_end = pd.Timestamp(item["bar_start"]).tz_localize(None)
        bar_start = vendor_bar_end - pd.Timedelta(minutes=bar_minutes)
        if not is_session_bar(bar_start):
            continue
        bar_end = vendor_bar_end
        amount = item.get("amount")
        money = item.get("money")
        if pd.isna(amount):
            amount = money
        row = {
            "source_vendor": SOURCE_VENDOR_JQ,
            "account_type": account_type,
            "research_only": True,
            "download_batch_id": plan["batch_id"],
            "vendor_code": plan["instrument_code"],
            "instrument_slot": instrument_slot,
            "bond_code": str(plan["bond_code"]).zfill(6),
            "stock_code": str(plan["stock_code"]).zfill(6),
            "trade_date": bar_start.strftime("%Y-%m-%d"),
            "bar_start": bar_start,
            "bar_end": bar_end,
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "volume": item.get("volume"),
            "money": money,
            "amount": amount,
            "paused": item.get("paused") if not pd.isna(item.get("paused")) else None,
            "bar_time_semantics": "vendor_bar_end_normalized",
            "downloaded_at": now,
        }
        row["raw_payload_hash"] = raw_payload_hash(row)
        row["ts"] = bar_start + pd.Timedelta(milliseconds=instrument_slot)
        rows.append(row)
    return rows


def existing_bar_keys(conn, table: str, rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    if not rows:
        return set()
    start = min(pd.Timestamp(row["bar_start"]).tz_localize(None) for row in rows)
    end = max(pd.Timestamp(row["bar_start"]).tz_localize(None) for row in rows)
    if table == CANONICAL_TABLE:
        code_column = "instrument_code"
        key_columns = ["instrument_code", "bond_code", "bar_start"]
    else:
        code_column = "vendor_code"
        key_columns = ["vendor_code", "bar_start"]
    codes = sorted({str(row[code_column]) for row in rows})
    code_filter = ", ".join(sql_string(code) for code in codes)
    df = query_to_dataframe(
        conn,
        f"""
        SELECT {', '.join(key_columns)} FROM {table}
        WHERE {code_column} IN ({code_filter})
          AND bar_start >= {sql_string(start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])}
          AND bar_start <= {sql_string(end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])}
        """,
    )
    if df.empty:
        return set()
    keys = set()
    for _, item in df.iterrows():
        values: list[Any] = []
        for column in key_columns:
            value = item[column]
            values.append(pd.Timestamp(value).tz_localize(None) if column == "bar_start" else str(value))
        keys.add(tuple(values))
    return keys


def bar_row_key(table: str, row: dict[str, Any]) -> tuple[Any, ...]:
    bar_start = pd.Timestamp(row["bar_start"]).tz_localize(None)
    if table == CANONICAL_TABLE:
        return str(row["instrument_code"]), str(row["bond_code"]), bar_start
    return str(row["vendor_code"]), bar_start


def insert_raw_rows_idempotent(conn, table: str, rows: list[dict[str, Any]]) -> int:
    existing = existing_bar_keys(conn, table, rows)
    filtered = [row for row in rows if bar_row_key(table, row) not in existing]
    return insert_rows(conn, table, filtered)


def repair_jq_raw_bar_semantics(conn, batch_id: str, config: dict[str, Any]) -> dict[str, Any]:
    ensure_training_tables(conn)
    if not batch_id:
        raise ValueError("repair-jq-bar-semantics requires --batch-id")
    bar_minutes = int(config["canonical"].get("bar_minutes", 5))
    repaired_total = 0
    table_counts: dict[str, int] = {}
    for asset_type, table in (("CB", JQ_CB_RAW_TABLE), ("STOCK", JQ_STOCK_RAW_TABLE)):
        raw = query_to_dataframe(
            conn,
            f"""
            SELECT * FROM {table}
            WHERE download_batch_id = {sql_string(batch_id)}
              AND (bar_time_semantics IS NULL OR bar_time_semantics != 'vendor_bar_end_normalized')
            ORDER BY ts
            """,
        )
        if raw.empty:
            table_counts[table] = 0
            continue
        for _, item in raw.iterrows():
            old_ts = pd.Timestamp(item["ts"]).tz_localize(None)
            vendor_bar_end = pd.Timestamp(item["bar_start"]).tz_localize(None)
            bar_start = vendor_bar_end - pd.Timedelta(minutes=bar_minutes)
            original = item.to_dict()
            original["ts"] = old_ts
            row = item.to_dict()
            row["bar_start"] = bar_start
            row["bar_end"] = vendor_bar_end
            row["bar_time_semantics"] = "vendor_bar_end_normalized"
            row["ts"] = stable_ts(bar_start, SOURCE_VENDOR_JQ, asset_type, row["vendor_code"], row["bond_code"])
            row["raw_payload_hash"] = raw_payload_hash(row)
            conn.execute(
                f"DELETE FROM {table} WHERE ts = "
                + sql_string(old_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
            )
            try:
                inserted = insert_rows(conn, table, [row])
            except Exception:
                insert_rows(conn, table, [original])
                raise
            table_counts[table] = table_counts.get(table, 0) + int(inserted)
            repaired_total += int(inserted)
    return {
        "mode": "repair-jq-bar-semantics",
        "batch_id": batch_id,
        "repaired_rows": repaired_total,
        "tables": table_counts,
    }


def recover_stale_running_plans(
    conn,
    sample_version: str,
    stale_minutes: int,
) -> int:
    running = query_to_dataframe(
        conn,
        f"""
        SELECT * FROM {DOWNLOAD_PLAN_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
          AND status = 'RUNNING'
        """,
    )
    if running.empty:
        return 0
    cutoff = now_ts() - pd.Timedelta(minutes=max(1, int(stale_minutes)))
    recovered = 0
    for _, item in running.iterrows():
        updated_at = pd.to_datetime(item.get("updated_at"), errors="coerce")
        if pd.notna(updated_at) and pd.Timestamp(updated_at).tz_localize(None) > cutoff:
            continue
        update_plan_status(
            conn,
            str(item["plan_id"]),
            "READY",
            "recovered stale RUNNING plan",
            current_row=item,
        )
        recovered += 1
    return recovered


def run_jq_download_plan(
    conn,
    sample_version: str,
    max_rows_today: int,
    confirm_large_download: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    ensure_training_tables(conn)
    jq_cfg = config["jqdata"]
    max_attempts = int(jq_cfg.get("max_attempts_per_plan", 3))
    recovered_stale = recover_stale_running_plans(
        conn,
        sample_version,
        int(jq_cfg.get("stale_running_minutes", 60)),
    )
    pending = query_to_dataframe(
        conn,
        f"""
        SELECT * FROM {DOWNLOAD_PLAN_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
          AND data_source = 'jqdata'
          AND status = {sql_string(RUNNABLE_JQ_PLAN_STATUS)}
          AND attempt_count < {max_attempts}
        ORDER BY priority, start_date, instrument_code
        """,
    )
    if pending.empty:
        return {"sample_version": sample_version, "downloaded_rows": 0, "reason": "no_ready_plan"}
    total_estimated = int(pd.to_numeric(pending["estimated_rows"], errors="coerce").fillna(0).sum())
    instrument_count = pending["instrument_code"].nunique()
    require_large_download_confirmation(instrument_count, total_estimated, confirm_large_download, config)

    today = now_ts().strftime("%Y-%m-%d")
    used = current_quota_used(conn, SOURCE_VENDOR_JQ, today)
    allowed_today = min(int(max_rows_today), int(jq_cfg["default_daily_hard_cap"]))
    if used >= allowed_today:
        return {"sample_version": sample_version, "downloaded_rows": 0, "reason": "daily_quota_cap_reached", "used_estimated": used}

    provider = build_jqdata_provider(config)
    account_id = provider.account_id_hash()
    downloaded = 0
    actual_vendor_rows = 0
    estimated_run = 0
    failed = 0
    fatal_error: str | None = None
    processed_plans = 0
    deferred_quota = 0
    daily_cap_reached = False
    skipped_existing = 0
    max_failed_plans = int(jq_cfg.get("max_failed_plans_per_run", 20))
    cooldown = float(jq_cfg.get("request_cooldown_seconds", 0) or 0)
    coverage_by_table = {
        JQ_CB_RAW_TABLE: load_raw_coverage(conn, JQ_CB_RAW_TABLE, str(pending["start_date"].min()), str(pending["end_date"].max())),
        JQ_STOCK_RAW_TABLE: load_raw_coverage(conn, JQ_STOCK_RAW_TABLE, str(pending["start_date"].min()), str(pending["end_date"].max())),
    }
    for pending_index, (_, plan) in enumerate(pending.iterrows()):
        plan_estimate = int(plan.get("estimated_rows") or 0)
        if used + estimated_run + plan_estimate > allowed_today:
            deferred_quota = int(len(pending) - pending_index)
            daily_cap_reached = True
            break
        table = raw_table(plan["asset_type"])
        if raw_chunk_complete_from_coverage(
            coverage_by_table[table],
            str(plan["instrument_code"]),
            str(plan["start_date"]),
            str(plan["end_date"]),
            plan_estimate,
        ):
            update_plan_status(conn, plan["plan_id"], "SKIPPED_ALREADY_EXISTS", current_row=plan)
            skipped_existing += 1
            continue
        estimated_run += plan_estimate
        processed_plans += 1
        plan_state = plan.to_dict()
        try:
            plan_state = update_plan_status(
                conn,
                plan["plan_id"],
                "RUNNING",
                increment_attempt=True,
                current_row=plan_state,
            )
            fields = json.loads(plan["fields_json"])
            fields = validate_jq_fields(plan["asset_type"], fields)
            frame = provider.get_price_5m(plan["instrument_code"], plan["start_date"], plan["end_date"], fields)
            actual_vendor_rows += int(len(frame))
            rows = raw_rows_from_jq_frame(plan, frame, config)
            inserted = insert_raw_rows_idempotent(conn, table, rows)
            downloaded += inserted
            update_plan_status(conn, plan["plan_id"], "SUCCESS", current_row=plan_state)
        except Exception as exc:
            failed += 1
            message = str(exc)
            LOGGER.exception("JQData plan failed plan_id=%s instrument=%s", plan["plan_id"], plan["instrument_code"])
            attempts_value = pd.to_numeric(plan.get("attempt_count"), errors="coerce")
            attempts_before = 0 if pd.isna(attempts_value) else int(attempts_value)
            attempts_after = attempts_before + 1
            retry_status = "READY" if attempts_after < max_attempts else "FAILED"
            update_plan_status(
                conn,
                plan["plan_id"],
                retry_status,
                message[:1000],
                increment_attempt=False,
                current_row=plan_state,
            )
            if "credentials are not configured" in message or "jqdatasdk is not installed" in message:
                fatal_error = message
                LOGGER.error("Stopping JQData run due to provider configuration error")
                break
            if failed >= max_failed_plans:
                LOGGER.error("Stopping JQData run after %s failed plans", failed)
                break
        if cooldown > 0:
            time.sleep(cooldown)

    if failed == 0:
        status = "SUCCESS"
    elif fatal_error and downloaded == 0:
        status = "FAILED"
    else:
        status = "PARTIAL_SUCCESS"
    job_id = record_quota_job(conn, SOURCE_VENDOR_JQ, account_id, "run-jq-download-plan", sample_version, estimated_run, actual_vendor_rows, status, fatal_error[:1000] if fatal_error else None)
    record_quota_ledger(
        conn,
        SOURCE_VENDOR_JQ,
        account_id,
        int(jq_cfg["daily_quota_limit"]),
        0,
        estimated_run,
        actual_vendor_rows,
        job_id,
        failed=failed > 0,
    )
    return {
        "sample_version": sample_version,
        "recovered_stale_plans": int(recovered_stale),
        "processed_plans": int(processed_plans),
        "skipped_quota_plans": 0,
        "deferred_quota_plans": int(deferred_quota),
        "daily_cap_reached": bool(daily_cap_reached),
        "skipped_existing_plans": int(skipped_existing),
        "estimated_rows": int(estimated_run),
        "actual_vendor_rows": int(actual_vendor_rows),
        "downloaded_rows": int(downloaded),
        "failed_plans": int(failed),
        "fatal_error": fatal_error,
        "quota_job_id": job_id,
        "status": status,
    }


def jq_smoke_test(
    conn,
    bond_codes: list[str],
    days: int,
    dry_run: bool,
    config: dict[str, Any],
    sample_version: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    ensure_training_tables(conn)
    require_smoke_scope(bond_codes, days, config)
    selected_sample_version = sample_version or config["default_sample_version"]
    sample = load_sample(conn, selected_sample_version)
    if sample.empty:
        sample = source_snapshot(conn)
    sample["bond_code"] = sample["bond_code"].astype(str).str.zfill(6)
    sample["stock_code"] = sample["stock_code"].astype(str).str.zfill(6)
    rows = sample[sample["bond_code"].isin([code.zfill(6) for code in bond_codes])].drop_duplicates("bond_code")
    if rows.empty:
        return {
            "mode": "jq-smoke-test",
            "sample_version": selected_sample_version,
            "dry_run": dry_run,
            "reason": "bond_codes_not_found_locally",
            "bond_codes": bond_codes,
        }
    safe_start, safe_end = permission_safe_trial_range(config)
    _, default_end, _ = selected_training_dates(config)
    end = pd.offsets.BDay().rollback(pd.Timestamp(end_date).normalize()) if end_date else default_end
    if end < safe_start or end > safe_end:
        raise ValueError(
            f"JQData smoke end date {end.date()} is outside permission-safe range "
            f"{safe_start.date()}..{safe_end.date()}"
        )
    available_smoke_dates, _ = trading_dates_between(config, safe_start, end)
    dates = available_smoke_dates[-int(days):]
    if not dates or pd.Timestamp(dates[0]) < safe_start:
        raise ValueError("JQData smoke date range crosses the permission-safe lower boundary")
    instruments = sample_instrument_records(rows)
    estimated = estimate_rows(len(instruments), len(dates), int(config["jqdata"]["bars_per_day_estimate"]))
    if dry_run:
        return {
            "mode": "jq-smoke-test",
            "sample_version": selected_sample_version,
            "dry_run": True,
            "bond_count": int(len(rows)),
            "instrument_count": int(len(instruments)),
            "start_date": dates[0],
            "end_date": dates[-1],
            "estimated_rows": int(estimated),
            "would_write_raw_tables": [JQ_CB_RAW_TABLE, JQ_STOCK_RAW_TABLE],
        }

    provider = build_jqdata_provider(config)
    account_id = provider.account_id_hash()
    inserted = 0
    actual_rows = 0
    estimated_run = 0
    skipped_existing = 0
    batch_id = stable_id("jqsmoke", ",".join(sorted(bond_codes)), dates[0], dates[-1])
    used_today = current_quota_used(conn, SOURCE_VENDOR_JQ, now_ts().strftime("%Y-%m-%d"))
    hard_cap = int(config["jqdata"]["default_daily_hard_cap"])
    if used_today + estimated > hard_cap:
        raise ValueError(f"JQData smoke test would exceed daily hard cap used={used_today} estimated={estimated} cap={hard_cap}")
    try:
        for instrument in instruments:
            asset_type = instrument["asset_type"]
            vendor_code = instrument["instrument_code"]
            table = raw_table(asset_type)
            if raw_rows_complete(
                conn,
                table,
                vendor_code,
                dates[0],
                dates[-1],
                len(dates) * int(config["jqdata"]["bars_per_day_estimate"]),
            ):
                skipped_existing += 1
                continue
            estimated_run += len(dates) * int(config["jqdata"]["bars_per_day_estimate"])
            fields = config["jqdata"]["cb_fields"] if asset_type == "CB" else config["jqdata"]["stock_fields"]
            fields = validate_jq_fields(asset_type, fields)
            plan = pd.Series(
                {
                    "batch_id": batch_id,
                    "instrument_code": vendor_code,
                    "instrument_slot": instrument["instrument_slot"],
                    "asset_type": asset_type,
                    "bond_code": instrument["bond_code"],
                    "stock_code": instrument["stock_code"],
                    "start_date": dates[0],
                    "end_date": dates[-1],
                }
            )
            frame = provider.get_price_5m(vendor_code, dates[0], dates[-1], fields)
            actual_rows += int(len(frame))
            raw_rows = raw_rows_from_jq_frame(plan, frame, config)
            inserted += insert_raw_rows_idempotent(conn, table, raw_rows)
    except Exception as exc:
        job_id = record_quota_job(conn, SOURCE_VENDOR_JQ, account_id, "jq-smoke-test", selected_sample_version, estimated_run, actual_rows, "FAILED", str(exc)[:1000])
        record_quota_ledger(conn, SOURCE_VENDOR_JQ, account_id, int(config["jqdata"]["daily_quota_limit"]), 0, estimated_run, actual_rows, job_id, failed=True)
        raise
    job_id = record_quota_job(conn, SOURCE_VENDOR_JQ, account_id, "jq-smoke-test", selected_sample_version, estimated_run, actual_rows, "SUCCESS")
    record_quota_ledger(conn, SOURCE_VENDOR_JQ, account_id, int(config["jqdata"]["daily_quota_limit"]), 0, estimated_run, actual_rows, job_id)
    return {
        "mode": "jq-smoke-test",
        "sample_version": selected_sample_version,
        "dry_run": False,
        "inserted_rows": int(inserted),
        "actual_vendor_rows": int(actual_rows),
        "estimated_rows": int(estimated_run),
        "skipped_existing_instruments": int(skipped_existing),
        "batch_id": batch_id,
        "quota_job_id": job_id,
    }


def bar_quality_flags(row: pd.Series) -> tuple[bool, str]:
    flags: list[str] = []
    values = {col: pd.to_numeric(row.get(col), errors="coerce") for col in ["open", "high", "low", "close"]}
    if any(pd.isna(value) for value in values.values()):
        flags.append("missing_ohlc")
    else:
        high = float(values["high"])
        low = float(values["low"])
        open_ = float(values["open"])
        close = float(values["close"])
        if high < low or high < max(open_, close) or low > min(open_, close):
            flags.append("invalid_ohlc")
    if not is_session_bar(row.get("bar_start")):
        flags.append("non_session_bar")
    if row.get("paused") in {True, 1, "1", "true", "True"}:
        flags.append("paused")
    return len([flag for flag in flags if flag not in {"paused"}]) == 0, ",".join(flags) or "OK"


def canonical_ts(row: pd.Series, sample_version: str) -> pd.Timestamp:
    slot = pd.to_numeric(row.get("instrument_slot"), errors="coerce")
    if pd.isna(slot) or int(slot) < 0 or int(slot) >= 300000:
        raise ValueError(f"invalid canonical instrument_slot={row.get('instrument_slot')}")
    return pd.Timestamp(row["bar_start"]).tz_localize(None) + pd.Timedelta(milliseconds=int(slot))


def canonicalize_jq_training_bars(
    conn,
    sample_version: str,
    config: dict[str, Any],
    batch_id: str | None = None,
) -> dict[str, Any]:
    ensure_training_tables(conn)
    sample = load_sample(conn, sample_version)
    if sample.empty:
        return {"sample_version": sample_version, "canonical_rows": 0, "reason": "sample_not_found"}
    scope_map = sample.set_index("bond_code")["research_pool_scope"].to_dict()
    stock_bonds = (
        sample.groupby("stock_code")["bond_code"]
        .apply(lambda values: sorted(set(values.astype(str).str.zfill(6))))
        .to_dict()
    )
    association_slots = canonical_association_slots(sample)
    _, _, dates = selected_training_dates(config)
    start_ts = f"{dates[0]} 00:00:00.000"
    end_ts = f"{dates[-1]} 23:59:59.999"
    bond_filter = ", ".join(sql_string(str(code).zfill(6)) for code in sample["bond_code"].astype(str).unique())
    discovered_batches: set[str] = set()
    for table in (JQ_CB_RAW_TABLE, JQ_STOCK_RAW_TABLE):
        raw_batches = query_to_dataframe(
            conn,
            f"""
            SELECT download_batch_id FROM {table}
            WHERE bond_code IN ({bond_filter})
              AND bar_start >= {sql_string(start_ts)}
              AND bar_start <= {sql_string(end_ts)}
            GROUP BY download_batch_id
            """,
        )
        if not raw_batches.empty:
            discovered_batches.update(raw_batches["download_batch_id"].dropna().astype(str))
    batch_ids = [str(batch_id)] if batch_id else sorted(discovered_batches)
    batch_ids = [value for value in batch_ids if value in discovered_batches]
    if not batch_ids:
        return {"sample_version": sample_version, "canonical_rows": 0, "reason": "no_matching_jq_raw_batches"}

    canonical_version = config["canonical"]["canonical_version"]
    now = now_ts()
    inserted_total = 0
    for asset_type, table in [("CB", JQ_CB_RAW_TABLE), ("STOCK", JQ_STOCK_RAW_TABLE)]:
        batch_filter = ", ".join(sql_string(batch_id) for batch_id in batch_ids)
        raw = query_to_dataframe(
            conn,
            f"""
            SELECT * FROM {table}
            WHERE download_batch_id IN ({batch_filter})
              AND bond_code IN ({bond_filter})
              AND bar_start >= {sql_string(start_ts)}
              AND bar_start <= {sql_string(end_ts)}
            ORDER BY bar_start, vendor_code
            """,
        )
        if raw.empty:
            continue
        raw["bar_start"] = pd.to_datetime(raw["bar_start"], errors="coerce")
        raw["bar_end"] = pd.to_datetime(raw["bar_end"], errors="coerce")
        raw = raw.dropna(subset=["bar_start", "bond_code"]).copy()
        raw = raw.drop_duplicates(subset=["vendor_code", "bond_code", "bar_start"], keep="last")
        rows = []
        for _, item in raw.iterrows():
            valid, flag = bar_quality_flags(item)
            amount = item.get("amount")
            money = item.get("money")
            amount_source = "money" if pd.isna(amount) and pd.notna(money) else "amount"
            if pd.isna(amount):
                amount = money
            stock_code = str(item["stock_code"]).zfill(6)
            target_bonds = (
                stock_bonds.get(stock_code, [])
                if asset_type == "STOCK"
                else [str(item["bond_code"]).zfill(6)]
            )
            for bond_code in target_bonds:
                out = {
                    "source_vendor": SOURCE_VENDOR_JQ,
                    "research_only": True,
                    "sample_version": sample_version,
                    "research_pool_scope": scope_map.get(bond_code, "UNKNOWN"),
                    "trade_date": pd.Timestamp(item["bar_start"]).strftime("%Y-%m-%d"),
                    "bar_start": item["bar_start"],
                    "bar_end": item["bar_end"] if pd.notna(item.get("bar_end")) else pd.Timestamp(item["bar_start"]) + pd.Timedelta(minutes=5),
                    "asset_type": asset_type,
                    "instrument_code": item["vendor_code"],
                    "instrument_slot": association_slots[(asset_type, bond_code)],
                    "bond_code": bond_code,
                    "stock_code": stock_code,
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "close": item.get("close"),
                    "volume": item.get("volume"),
                    "amount": amount,
                    "money": money,
                    "amount_source": amount_source,
                    "paused": item.get("paused"),
                    "is_valid_bar": valid,
                    "data_quality_flag": flag,
                    "canonical_version": canonical_version,
                    "created_at": now,
                }
                out["ts"] = canonical_ts(pd.Series(out), sample_version)
                rows.append(out)
        inserted_total += insert_raw_rows_idempotent(conn, CANONICAL_TABLE, rows)
    return {"sample_version": sample_version, "canonical_rows_inserted": int(inserted_total), "batch_count": int(len(batch_ids))}


def report_jq_training_coverage(conn, sample_version: str, config: dict[str, Any]) -> dict[str, Any]:
    ensure_training_tables(conn)
    sample = load_sample(conn, sample_version)
    if sample.empty:
        return {"sample_version": sample_version, "coverage_rows": 0, "reason": "sample_not_found"}
    start, end, dates = selected_training_dates(config)
    expected_days = len(dates)
    expected_bars = expected_days * int(config["jqdata"]["bars_per_day_estimate"])
    batch = query_to_dataframe(conn, f"SELECT LAST(batch_id) AS batch_id FROM {DOWNLOAD_PLAN_TABLE} WHERE sample_version = {sql_string(sample_version)}")
    batch_id = str(batch.loc[0, "batch_id"]) if not batch.empty and pd.notna(batch.loc[0, "batch_id"]) else stable_id("coverage", sample_version)
    now = now_ts()
    existing_report = query_to_dataframe(
        conn,
        f"""
        SELECT bond_code, LAST(ts) AS ts
        FROM {COVERAGE_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
        GROUP BY bond_code
        """,
    )
    existing_ts = {
        str(item["bond_code"]).zfill(6): pd.Timestamp(item["ts"]).tz_localize(None)
        for _, item in existing_report.iterrows()
        if pd.notna(item.get("ts"))
    } if not existing_report.empty else {}
    rows = []
    for row_index, (_, item) in enumerate(sample.sort_values("bond_code").iterrows()):
        bond = str(item["bond_code"]).zfill(6)
        stock = str(item["stock_code"]).zfill(6)
        data = query_to_dataframe(
            conn,
            f"""
            SELECT asset_type, trade_date, is_valid_bar, COUNT(*) AS n
            FROM {CANONICAL_TABLE}
            WHERE sample_version = {sql_string(sample_version)}
              AND bond_code = {sql_string(bond)}
            GROUP BY asset_type, trade_date, is_valid_bar
            """,
        )
        cb_days = stock_days = valid_cb_days = valid_stock_days = 0
        cb_bars = stock_bars = valid_cb_bars = valid_stock_bars = 0
        if not data.empty:
            cb = data[data["asset_type"].astype(str).str.upper() == "CB"]
            st = data[data["asset_type"].astype(str).str.upper() == "STOCK"]
            valid_mask = data["is_valid_bar"].astype(str).str.lower().isin({"true", "1"})
            valid_cb = data[(data["asset_type"].astype(str).str.upper() == "CB") & valid_mask]
            valid_st = data[(data["asset_type"].astype(str).str.upper() == "STOCK") & valid_mask]
            cb_days = cb["trade_date"].nunique()
            stock_days = st["trade_date"].nunique()
            valid_cb_days = valid_cb["trade_date"].nunique()
            valid_stock_days = valid_st["trade_date"].nunique()
            cb_bars = int(pd.to_numeric(cb["n"], errors="coerce").fillna(0).sum())
            stock_bars = int(pd.to_numeric(st["n"], errors="coerce").fillna(0).sum())
            valid_cb_bars = int(pd.to_numeric(valid_cb["n"], errors="coerce").fillna(0).sum())
            valid_stock_bars = int(pd.to_numeric(valid_st["n"], errors="coerce").fillna(0).sum())
        invalid_cb_bars = cb_bars - valid_cb_bars
        invalid_stock_bars = stock_bars - valid_stock_bars
        row_ratio = min(cb_bars, stock_bars) / expected_bars if expected_bars else 0.0
        ratio = min(valid_cb_bars, valid_stock_bars) / expected_bars if expected_bars else 0.0
        missing = []
        if cb_days < expected_days:
            missing.append("cb_missing_days")
        if stock_days < expected_days:
            missing.append("stock_missing_days")
        if invalid_cb_bars:
            missing.append("cb_invalid_bars")
        if invalid_stock_bars:
            missing.append("stock_invalid_bars")
        rows.append(
            {
                "ts": existing_ts.get(bond, now + pd.Timedelta(milliseconds=row_index)),
                "batch_id": batch_id,
                "sample_version": sample_version,
                "bond_code": bond,
                "stock_code": stock,
                "research_pool_scope": item.get("research_pool_scope"),
                "expected_trading_days": expected_days,
                "available_cb_trading_days": cb_days,
                "available_stock_trading_days": stock_days,
                "valid_cb_trading_days": valid_cb_days,
                "valid_stock_trading_days": valid_stock_days,
                "expected_cb_bars": expected_bars,
                "available_cb_bars": cb_bars,
                "valid_cb_bars": valid_cb_bars,
                "invalid_cb_bars": invalid_cb_bars,
                "expected_stock_bars": expected_bars,
                "available_stock_bars": stock_bars,
                "valid_stock_bars": valid_stock_bars,
                "invalid_stock_bars": invalid_stock_bars,
                "row_coverage_ratio": row_ratio,
                "coverage_ratio": ratio,
                "coverage_pass": ratio >= 0.90,
                "missing_reason": ",".join(missing) or "OK",
                "checked_at": now,
            }
        )
    inserted = insert_rows(conn, COVERAGE_TABLE, rows)
    return {"sample_version": sample_version, "coverage_rows_inserted": int(inserted), "coverage_rows": int(len(rows))}


def build_local_training_windows(conn, sample_version: str, window_trading_days: int, max_windows: int) -> dict[str, Any]:
    ensure_training_tables(conn)
    daily_instruments = query_to_dataframe(
        conn,
        f"""
        SELECT trade_date, asset_type, instrument_code, COUNT(*) AS n
        FROM {CANONICAL_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
        GROUP BY trade_date, asset_type, instrument_code
        ORDER BY trade_date, asset_type, instrument_code
        """
    )
    if daily_instruments.empty:
        return {"sample_version": sample_version, "windows": 0, "reason": "no_canonical_data"}
    daily_instruments["trade_date"] = daily_instruments["trade_date"].astype(str)
    daily_instruments["n"] = pd.to_numeric(daily_instruments["n"], errors="coerce").fillna(0)
    dates = sorted(set(daily_instruments["trade_date"].dropna()))
    windows = []
    if len(dates) < int(window_trading_days):
        return {
            "sample_version": sample_version,
            "windows": 0,
            "reason": "insufficient_canonical_trading_days",
            "available_trading_days": int(len(dates)),
            "required_trading_days": int(window_trading_days),
        }
    else:
        step = max(1, math.floor((len(dates) - int(window_trading_days)) / max(1, int(max_windows) - 1))) if max_windows > 1 else int(window_trading_days)
        starts = []
        idx = 0
        while idx + int(window_trading_days) <= len(dates) and len(starts) < int(max_windows):
            starts.append(idx)
            idx += step
        for start_idx in starts[: int(max_windows)]:
            chunk = dates[start_idx:start_idx + int(window_trading_days)]
            windows.append((chunk[0], chunk[-1], len(chunk)))

    now = now_ts()
    existing_manifest = query_to_dataframe(
        conn,
        f"""
        SELECT local_window_id, LAST(ts) AS ts
        FROM {WINDOW_MANIFEST_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
        GROUP BY local_window_id
        """,
    )
    existing_ts = {
        str(item["local_window_id"]): pd.Timestamp(item["ts"]).tz_localize(None)
        for _, item in existing_manifest.iterrows()
        if pd.notna(item.get("ts"))
    } if not existing_manifest.empty else {}
    rows = []
    for idx, (start_date, end_date, day_count) in enumerate(windows, start=1):
        counts = daily_instruments[
            (daily_instruments["trade_date"] >= start_date)
            & (daily_instruments["trade_date"] <= end_date)
        ]
        cb_count = stock_count = instrument_count = 0
        if not counts.empty:
            cb = counts[counts["asset_type"].astype(str).str.upper() == "CB"]
            st = counts[counts["asset_type"].astype(str).str.upper() == "STOCK"]
            cb_count = int(cb["n"].sum()) if not cb.empty else 0
            stock_count = int(st["n"].sum()) if not st.empty else 0
            instrument_count = int(
                counts.groupby("asset_type")["instrument_code"].nunique().sum()
            )
        local_window_id = stable_id("jqwin", sample_version, start_date, end_date, idx)
        rows.append(
            {
                "ts": existing_ts.get(local_window_id, stable_ts(now, local_window_id)),
                "local_window_id": local_window_id,
                "sample_version": sample_version,
                "window_start_date": start_date,
                "window_end_date": end_date,
                "trading_days": day_count,
                "source_vendor": SOURCE_VENDOR_JQ,
                "instrument_count": instrument_count,
                "cb_bar_count": cb_count,
                "stock_bar_count": stock_count,
                "created_at": now,
            }
        )
    inserted = insert_rows(conn, WINDOW_MANIFEST_TABLE, rows)
    return {"sample_version": sample_version, "windows": int(len(rows)), "inserted_rows": int(inserted), "manifest": rows}


def contiguous_missing_date_ranges(all_dates: list[str], missing_dates: list[str]) -> list[tuple[str, str]]:
    missing = set(missing_dates)
    ranges: list[tuple[str, str]] = []
    start: str | None = None
    previous: str | None = None
    for trade_date in all_dates:
        if trade_date in missing:
            if start is None:
                start = trade_date
            previous = trade_date
        elif start is not None and previous is not None:
            ranges.append((start, previous))
            start = previous = None
    if start is not None and previous is not None:
        ranges.append((start, previous))
    return ranges


def wind_scope_candidates(sample: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "active_like_only":
        return sample[sample["research_pool_scope"] == "ACTIVE_LIKE"].copy()
    if scope != "validation_sample":
        raise ValueError(f"unsupported Wind enrichment scope: {scope}")

    candidates = sample[sample["research_pool_scope"] != "ACTIVE_LIKE"].copy()
    scope_rank = {"ELIGIBLE_SHADOW": 0, "BROAD_BUT_CLEAN": 1}
    candidates["_scope_rank"] = candidates["research_pool_scope"].map(scope_rank).fillna(99)
    return candidates.sort_values(["_scope_rank", "bond_code"]).drop(columns=["_scope_rank"])


def wind_fields_for_scope(config: dict[str, Any], scope: str) -> list[str]:
    if scope == "active_like_only":
        return validate_wind_fields(["live_turnover", "ytm"])
    if scope == "validation_sample":
        return validate_wind_fields(config["wind"].get("validation_fields", ["live_turnover"]))
    raise ValueError(f"unsupported Wind enrichment scope: {scope}")


def build_wind_missing_requests(
    bond_codes: list[str],
    dates: list[str],
    existing_keys: set[tuple[str, str]],
    field_count: int,
    max_cells: int | None = None,
) -> tuple[list[str], list[tuple[str, str, str]], int]:
    selected_bonds: list[str] = []
    requests: list[tuple[str, str, str]] = []
    estimated_cells = 0
    for bond_code in bond_codes:
        missing_dates = [date for date in dates if (bond_code, date) not in existing_keys]
        bond_cells = len(missing_dates) * int(field_count)
        if bond_cells == 0:
            continue
        if max_cells is not None and estimated_cells + bond_cells > int(max_cells):
            continue
        selected_bonds.append(bond_code)
        estimated_cells += bond_cells
        for range_start, range_end in contiguous_missing_date_ranges(dates, missing_dates):
            requests.append((bond_code, range_start, range_end))
    return selected_bonds, requests, estimated_cells


def run_wind_turnover_ytm_enrichment(
    conn,
    sample_version: str,
    scope: str,
    max_rows_week: int,
    confirm_wind_download: bool,
    config: dict[str, Any],
    max_cells_run: int | None = None,
) -> dict[str, Any]:
    ensure_training_tables(conn)
    if not confirm_wind_download:
        raise ValueError("Wind enrichment refused without --confirm-wind-download")
    fields = wind_fields_for_scope(config, scope)
    sample = load_sample(conn, sample_version)
    if sample.empty:
        return {"sample_version": sample_version, "wind_rows": 0, "reason": "sample_not_found"}
    sample = wind_scope_candidates(sample, scope)
    start, end, dates = selected_training_dates(config)
    bond_codes = sample["bond_code"].astype(str).str.zfill(6).drop_duplicates().tolist()
    existing = query_to_dataframe(
        conn,
        f"""
        SELECT bond_code, trade_date FROM {WIND_RAW_TABLE}
        WHERE trade_date >= {sql_string(dates[0])}
          AND trade_date <= {sql_string(dates[-1])}
        """,
    )
    existing_keys = {
        (str(item["bond_code"]).zfill(6), str(item["trade_date"])[:10])
        for _, item in existing.iterrows()
    } if not existing.empty else set()
    weekly_cap = min(int(max_rows_week), int(config["wind"]["weekly_hard_cap"]))
    used_this_week = current_weekly_quota_used(conn, SOURCE_VENDOR_WIND, week_id())
    remaining_week = max(0, weekly_cap - used_this_week)
    run_cap = None
    if scope == "validation_sample":
        configured_cap = int(config["wind"].get("validation_run_target_cells", 7000))
        requested_cap = configured_cap if max_cells_run is None else int(max_cells_run)
        if requested_cap <= 0:
            raise ValueError("--max-wind-cells-run must be positive")
        run_cap = min(requested_cap, remaining_week)
    has_missing_data = any(
        (bond_code, date) not in existing_keys
        for bond_code in bond_codes
        for date in dates
    )
    selected_bonds, requests, estimated = build_wind_missing_requests(
        bond_codes,
        dates,
        existing_keys,
        len(fields),
        max_cells=run_cap,
    )
    if estimated == 0:
        quota_limited = bool(has_missing_data and run_cap is not None)
        return {
            "sample_version": sample_version,
            "scope": scope,
            "status": "SKIPPED_QUOTA_LIMIT" if quota_limited else "SKIPPED_ALREADY_EXISTS",
            "wind_rows_inserted": 0,
            "estimated_rows": 0,
            "request_count": 0,
            "selected_bonds": [],
            "reason": "no_whole_bond_fits_run_or_weekly_cap" if quota_limited else "all_dates_already_exist",
        }
    if used_this_week + estimated > weekly_cap:
        raise ValueError(
            f"Wind enrichment refused: used_this_week={used_this_week} estimated_rows={estimated} "
            f"would exceed weekly cap {weekly_cap}"
        )
    provider = build_wind_provider(config)
    account_id = provider.account_id_hash()
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for bond_code, range_start, range_end in requests:
        try:
            frame = provider.fetch_turnover_ytm([bond_code], range_start, range_end, fields)
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{bond_code} {range_start}..{range_end}: {exc}")
            LOGGER.exception("Wind enrichment failed bond=%s range=%s..%s", bond_code, range_start, range_end)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["bond_code", "trade_date", *fields])

    batch_id = stable_id("windbatch", sample_version, scope, dates[0], dates[-1])
    rows = wind_raw_rows(frame, batch_id, config)
    inserted = insert_wind_rows_idempotent(conn, rows)
    actual_cells = int(len(frame) * len(fields))
    status = "SUCCESS" if not errors else ("PARTIAL_SUCCESS" if not frame.empty else "FAILED")
    error_message = "; ".join(errors)[:1000] if errors else None
    job_id = record_quota_job(conn, SOURCE_VENDOR_WIND, account_id, "run-wind-turnover-ytm-enrichment", sample_version, estimated, actual_cells, status, error_message)
    record_quota_ledger(conn, SOURCE_VENDOR_WIND, account_id, 0, int(config["wind"]["weekly_quota_limit"]), estimated, actual_cells, job_id, failed=bool(errors))
    return {
        "sample_version": sample_version,
        "scope": scope,
        "status": status,
        "request_count": int(len(requests)),
        "selected_bonds": selected_bonds,
        "wind_rows_inserted": int(inserted),
        "estimated_rows": int(estimated),
        "actual_vendor_cells": actual_cells,
        "error_count": int(len(errors)),
        "errors": errors,
        "quota_job_id": job_id,
    }


def wind_raw_rows(frame: pd.DataFrame, batch_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    now = now_ts()
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        trade_date = str(item.get("trade_date") or "")[:10]
        if not trade_date or pd.isna(pd.to_datetime(trade_date, errors="coerce")):
            continue
        bond_code = str(item.get("bond_code") or "").zfill(6)
        if not bond_code.strip("0"):
            continue
        rows.append(
            {
                "ts": pd.Timestamp(trade_date).normalize() + pd.Timedelta(milliseconds=int(bond_code)),
                "source_vendor": SOURCE_VENDOR_WIND,
                "account_type": config["wind"]["account_type"],
                "research_only": True,
                "download_batch_id": batch_id,
                "bond_code": bond_code,
                "trade_date": trade_date,
                "ytm": item.get("ytm"),
                "live_turnover": item.get("live_turnover"),
                "ytm_source": "wind" if pd.notna(item.get("ytm")) else "unavailable",
                "live_turnover_source": "wind" if pd.notna(item.get("live_turnover")) else "unavailable",
                "downloaded_at": now,
            }
        )
    return rows


def insert_wind_rows_idempotent(conn, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    start_date = min(row["trade_date"] for row in rows)
    end_date = max(row["trade_date"] for row in rows)
    existing = query_to_dataframe(
        conn,
        f"""
        SELECT bond_code, trade_date
        FROM {WIND_RAW_TABLE}
        WHERE trade_date >= {sql_string(start_date)}
          AND trade_date <= {sql_string(end_date)}
        """,
    )
    existing_keys = set()
    if not existing.empty:
        existing_keys = {
            (str(item["bond_code"]).zfill(6), str(item["trade_date"])[:10])
            for _, item in existing.iterrows()
        }
    filtered = [row for row in rows if (row["bond_code"], row["trade_date"]) not in existing_keys]
    return insert_rows(conn, WIND_RAW_TABLE, filtered)


def wind_smoke_test(
    conn,
    bond_codes: list[str],
    days: int,
    dry_run: bool,
    confirm_wind_download: bool,
    sample_version: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    ensure_training_tables(conn)
    require_smoke_scope(bond_codes, days, config)
    if not bond_codes:
        raise ValueError("wind-smoke-test requires at least one bond code")
    sample = load_sample(conn, sample_version)
    rows = sample[sample["bond_code"].isin([str(code).zfill(6) for code in bond_codes])].drop_duplicates("bond_code")
    if rows.empty:
        return {
            "mode": "wind-smoke-test",
            "sample_version": sample_version,
            "dry_run": dry_run,
            "reason": "bond_codes_not_found_in_sample",
        }
    safe_start, _ = permission_safe_trial_range(config)
    _, end, _ = selected_training_dates(config)
    available_smoke_dates, _ = trading_dates_between(config, safe_start, end)
    dates = available_smoke_dates[-int(days):]
    fields = validate_wind_fields(["live_turnover", "ytm"])
    estimated_cells = int(len(rows) * len(dates) * len(fields))
    weekly_cap = int(config["wind"]["weekly_hard_cap"])
    used_this_week = current_weekly_quota_used(conn, SOURCE_VENDOR_WIND, week_id())
    result = {
        "mode": "wind-smoke-test",
        "sample_version": sample_version,
        "dry_run": bool(dry_run),
        "bond_codes": rows["bond_code"].astype(str).tolist(),
        "start_date": dates[0],
        "end_date": dates[-1],
        "fields": fields,
        "estimated_vendor_cells": estimated_cells,
        "used_this_week_before": used_this_week,
        "weekly_hard_cap": weekly_cap,
    }
    if dry_run:
        return result
    if not confirm_wind_download:
        raise ValueError("Wind smoke test refused without --confirm-wind-download")
    if used_this_week + estimated_cells > weekly_cap:
        raise ValueError("Wind smoke test refused because the weekly hard cap would be exceeded")

    provider = build_wind_provider(config)
    account_id = provider.account_id_hash()
    batch_id = stable_id("windsmoke", sample_version, ",".join(sorted(result["bond_codes"])), dates[0], dates[-1])
    try:
        frame = provider.fetch_turnover_ytm(result["bond_codes"], dates[0], dates[-1], fields)
        inserted = insert_wind_rows_idempotent(conn, wind_raw_rows(frame, batch_id, config))
        actual_cells = int(len(frame) * len(fields))
        job_id = record_quota_job(conn, SOURCE_VENDOR_WIND, account_id, "wind-smoke-test", sample_version, estimated_cells, actual_cells, "SUCCESS")
        record_quota_ledger(conn, SOURCE_VENDOR_WIND, account_id, 0, int(config["wind"]["weekly_quota_limit"]), estimated_cells, actual_cells, job_id)
    except Exception as exc:
        job_id = record_quota_job(conn, SOURCE_VENDOR_WIND, account_id, "wind-smoke-test", sample_version, estimated_cells, 0, "FAILED", str(exc)[:1000])
        record_quota_ledger(conn, SOURCE_VENDOR_WIND, account_id, 0, int(config["wind"]["weekly_quota_limit"]), estimated_cells, 0, job_id, failed=True)
        raise
    return {
        **result,
        "dry_run": False,
        "batch_id": batch_id,
        "vendor_rows_returned": int(len(frame)),
        "actual_vendor_cells": actual_cells,
        "wind_rows_inserted": int(inserted),
        "quota_job_id": job_id,
    }


@contextmanager
def without_http_proxy():
    names = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is not None:
                os.environ[name] = value


def conversion_price_cache_dir(config: dict[str, Any]) -> Path:
    configured = Path(
        config["akshare_static"].get(
            "conversion_price_cache_dir",
            "outputs/akshare_conversion_price_history",
        )
    ).expanduser()
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[1] / configured


def normalize_akshare_conversion_value(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["trade_date", "conversion_value"])
    out = raw.rename(columns={"日期": "trade_date", "转股价值": "conversion_value"}).copy()
    if "trade_date" not in out.columns or "conversion_value" not in out.columns:
        raise ValueError(f"unexpected AkShare conversion value columns: {list(raw.columns)}")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["conversion_value"] = pd.to_numeric(out["conversion_value"], errors="coerce")
    return (
        out.dropna(subset=["trade_date", "conversion_value"])
        .sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")[["trade_date", "conversion_value"]]
    )


def fetch_akshare_conversion_value_history(
    bond_code: str,
    cache_dir: Path,
    refresh_cache: bool,
    retries: int,
    cooldown_seconds: float,
) -> tuple[pd.DataFrame, bool]:
    cache_path = cache_dir / f"{str(bond_code).zfill(6)}.csv"
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path)
        return normalize_akshare_conversion_value(cached), True

    import akshare as ak

    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            with without_http_proxy():
                raw = ak.bond_zh_cov_value_analysis(symbol=str(bond_code).zfill(6))
            normalized = normalize_akshare_conversion_value(raw)
            cache_dir.mkdir(parents=True, exist_ok=True)
            normalized.to_csv(cache_path, index=False)
            return normalized, False
        except Exception as exc:
            last_error = exc
            if attempt < max(1, int(retries)):
                time.sleep(max(1.0, cooldown_seconds) * attempt)
    raise RuntimeError(f"{bond_code} AkShare conversion value fetch failed: {last_error}")


def conversion_history_ts(sample_version: str, bond_code: str, effective_start: str) -> pd.Timestamp:
    key = f"{sample_version}|{bond_code}|{effective_start}"
    offset_ms = zlib.crc32(key.encode("utf-8"))
    return pd.Timestamp("2020-01-01") + pd.Timedelta(milliseconds=offset_ms)


def stock_daily_support_rows(
    stock_daily: pd.DataFrame,
    sample_version: str,
    batch_id: str,
    downloaded_at: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized = (
        stock_daily.dropna(subset=["trade_date", "stock_code"])
        .drop_duplicates(["stock_code", "trade_date"], keep="last")
        .sort_values(["stock_code", "trade_date"])
        .copy()
    )
    normalized["vendor_close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized["close"] = normalized.groupby("stock_code")["vendor_close"].ffill()
    normalized["is_filled"] = normalized["vendor_close"].isna() & normalized["close"].notna()
    normalized = normalized.dropna(subset=["close"])
    for _, item in normalized.iterrows():
        trade_date = pd.Timestamp(item["trade_date"]).normalize()
        stock_code = str(item["stock_code"]).zfill(6)
        rows.append(
            {
                "ts": trade_date + pd.Timedelta(milliseconds=int(stock_code)),
                "source_vendor": SOURCE_VENDOR_JQ,
                "research_only": True,
                "sample_version": sample_version,
                "download_batch_id": batch_id,
                "vendor_code": item.get("vendor_code"),
                "stock_code": stock_code,
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "close": item.get("close"),
                "vendor_close": item.get("vendor_close"),
                "is_filled": item.get("is_filled"),
                "price_status": "PREVIOUS_UNADJUSTED_CLOSE" if item.get("is_filled") else "VENDOR_CLOSE",
                "fq_mode": "none",
                "downloaded_at": downloaded_at,
            }
        )
    return rows


def infer_conversion_price_intervals(
    sample_version: str,
    bond_code: str,
    stock_code: str,
    target_dates: list[pd.Timestamp],
    stock_daily: pd.DataFrame,
    akshare_values: pd.DataFrame,
    adjustments: pd.DataFrame,
    downloaded_at: pd.Timestamp,
    mismatch_tolerance: float = 0.08,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dates = sorted({pd.Timestamp(value).normalize() for value in target_dates if not pd.isna(value)})
    expected_days = len(dates)
    base_report = {
        "bond_code": str(bond_code).zfill(6),
        "stock_code": str(stock_code).zfill(6),
        "expected_trading_days": expected_days,
        "covered_trading_days": 0,
        "interval_count": 0,
        "akshare_anchor_days": 0,
        "jq_adjustment_events": 0,
        "qa_mismatch_days": 0,
        "coverage_ratio": 1.0 if expected_days == 0 else 0.0,
        "coverage_pass": expected_days == 0,
        "baseline_anchor_date": None,
        "baseline_source": None,
        "missing_reason": "no_valid_cb_bars_in_windows" if expected_days == 0 else None,
    }
    if not dates:
        return [], base_report

    first_date = dates[0]
    last_date = dates[-1]
    stocks = stock_daily.copy()
    if not stocks.empty:
        stocks["trade_date"] = pd.to_datetime(stocks["trade_date"], errors="coerce").dt.normalize()
        stocks["close"] = pd.to_numeric(stocks["close"], errors="coerce")
    values = akshare_values.copy()
    if not values.empty:
        values["trade_date"] = pd.to_datetime(values["trade_date"], errors="coerce").dt.normalize()
        values["conversion_value"] = pd.to_numeric(values["conversion_value"], errors="coerce")
    anchors = values.merge(stocks[["trade_date", "close"]], on="trade_date", how="inner")
    anchors = anchors[
        anchors["trade_date"].isin(dates)
        & (anchors["conversion_value"] > 0)
        & (anchors["close"] > 0)
    ].copy()
    anchors["implied_conversion_price"] = 100.0 * anchors["close"] / anchors["conversion_value"]

    events = adjustments.copy()
    if not events.empty:
        events["effective_date"] = pd.to_datetime(events["effective_date"], errors="coerce").dt.normalize()
        events["conversion_price"] = pd.to_numeric(events["conversion_price"], errors="coerce")
        events = (
            events.dropna(subset=["effective_date", "conversion_price"])
            .sort_values(["effective_date", "announcement_date"])
            .drop_duplicates("effective_date", keep="last")
        )
        events = events[events["effective_date"] <= last_date].copy()

    prior_events = events[events["effective_date"] <= first_date] if not events.empty else events
    future_events = events[events["effective_date"] > first_date] if not events.empty else events
    baseline_price: float | None = None
    baseline_source: str | None = None
    baseline_source_date: pd.Timestamp | None = None
    baseline_announcement: pd.Timestamp | None = None
    baseline_reason: str | None = None

    if not prior_events.empty:
        baseline_event = prior_events.iloc[-1]
        baseline_price = float(baseline_event["conversion_price"])
        baseline_source = "jqdata_adjustment"
        baseline_source_date = pd.Timestamp(baseline_event["effective_date"])
        baseline_announcement = pd.to_datetime(baseline_event.get("announcement_date"), errors="coerce")
        baseline_reason = baseline_event.get("adjustment_reason")
    elif not anchors.empty:
        first_future_event = future_events["effective_date"].min() if not future_events.empty else None
        baseline_anchors = anchors
        if first_future_event is not None and not pd.isna(first_future_event):
            baseline_anchors = baseline_anchors[baseline_anchors["trade_date"] < first_future_event]
        if baseline_anchors.empty:
            baseline_anchors = anchors.head(10)
        else:
            baseline_anchors = baseline_anchors.head(20)
        baseline_price = round(float(baseline_anchors["implied_conversion_price"].median()), 2)
        baseline_source = "akshare_conversion_value+jqdata_unadjusted_close"
        baseline_source_date = pd.Timestamp(baseline_anchors.iloc[0]["trade_date"])

    if baseline_price is None or not math.isfinite(baseline_price) or baseline_price <= 0:
        base_report.update(
            {
                "akshare_anchor_days": int(len(anchors)),
                "jq_adjustment_events": int(len(events)),
                "missing_reason": "no_point_in_time_baseline_price",
            }
        )
        return [], base_report

    event_rows: list[dict[str, Any]] = [
        {
            "effective_start": first_date,
            "conversion_price": baseline_price,
            "price_source": baseline_source,
            "source_effective_date": baseline_source_date,
            "announcement_date": baseline_announcement,
            "adjustment_reason": baseline_reason or "window_baseline",
            "baseline_anchor_date": baseline_source_date,
        }
    ]
    for _, event in future_events.iterrows():
        event_rows.append(
            {
                "effective_start": pd.Timestamp(event["effective_date"]),
                "conversion_price": float(event["conversion_price"]),
                "price_source": "jqdata_adjustment",
                "source_effective_date": pd.Timestamp(event["effective_date"]),
                "announcement_date": pd.to_datetime(event.get("announcement_date"), errors="coerce"),
                "adjustment_reason": event.get("adjustment_reason"),
                "baseline_anchor_date": baseline_source_date,
            }
        )
    event_rows.sort(key=lambda item: item["effective_start"])

    intervals: list[dict[str, Any]] = []
    terminal_end = last_date + pd.Timedelta(days=1)
    for index, event in enumerate(event_rows):
        effective_start = pd.Timestamp(event["effective_start"]).normalize()
        effective_end = (
            pd.Timestamp(event_rows[index + 1]["effective_start"]).normalize()
            if index + 1 < len(event_rows)
            else terminal_end
        )
        source_vendor = "jqdata"
        if not anchors.empty:
            source_vendor = "akshare+jqdata"
        confidence = "HIGH" if not anchors.empty else "MEDIUM"
        intervals.append(
            {
                "ts": conversion_history_ts(sample_version, bond_code, effective_start.strftime("%Y-%m-%d")),
                "sample_version": sample_version,
                "research_only": True,
                "bond_code": str(bond_code).zfill(6),
                "stock_code": str(stock_code).zfill(6),
                "effective_start": effective_start.strftime("%Y-%m-%d"),
                "effective_end_exclusive": effective_end.strftime("%Y-%m-%d"),
                "conversion_price": event["conversion_price"],
                "source_vendor": source_vendor,
                "price_source": event["price_source"],
                "source_effective_date": (
                    pd.Timestamp(event["source_effective_date"]).strftime("%Y-%m-%d")
                    if event.get("source_effective_date") is not None
                    and not pd.isna(event.get("source_effective_date"))
                    else None
                ),
                "announcement_date": event.get("announcement_date"),
                "adjustment_reason": event.get("adjustment_reason"),
                "baseline_anchor_date": (
                    pd.Timestamp(event["baseline_anchor_date"]).strftime("%Y-%m-%d")
                    if event.get("baseline_anchor_date") is not None
                    and not pd.isna(event.get("baseline_anchor_date"))
                    else None
                ),
                "confidence": confidence,
                "source_detail": (
                    "AkShare Eastmoney historical conversion_value supplies point-in-time baseline/QA; "
                    "JQData fq=None stock close supplies the unadjusted underlying price; "
                    "JQData CONBOND_CONVERT_PRICE_ADJUST supplies effective adjustment events."
                ),
                "downloaded_at": downloaded_at,
            }
        )

    covered_dates = 0
    mismatch_days = 0
    implied_by_date = {
        pd.Timestamp(row["trade_date"]).normalize(): float(row["implied_conversion_price"])
        for _, row in anchors.iterrows()
    }
    for target_date in dates:
        matching = [
            item
            for item in intervals
            if pd.Timestamp(item["effective_start"]) <= target_date
            < pd.Timestamp(item["effective_end_exclusive"])
        ]
        if not matching:
            continue
        covered_dates += 1
        implied = implied_by_date.get(target_date)
        if implied is not None and abs(implied - float(matching[-1]["conversion_price"])) > float(mismatch_tolerance):
            mismatch_days += 1

    coverage_ratio = covered_dates / expected_days if expected_days else 1.0
    missing: list[str] = []
    if covered_dates < expected_days:
        missing.append(f"uncovered_trading_days={expected_days - covered_dates}")
    if anchors.empty:
        missing.append("akshare_anchor_unavailable")
    if mismatch_days:
        missing.append(f"akshare_jq_price_mismatch_days={mismatch_days}")
    base_report.update(
        {
            "covered_trading_days": covered_dates,
            "interval_count": len(intervals),
            "akshare_anchor_days": int(len(anchors)),
            "jq_adjustment_events": int(len(events)),
            "qa_mismatch_days": mismatch_days,
            "coverage_ratio": coverage_ratio,
            "coverage_pass": covered_dates == expected_days,
            "baseline_anchor_date": (
                baseline_source_date.strftime("%Y-%m-%d")
                if baseline_source_date is not None and not pd.isna(baseline_source_date)
                else None
            ),
            "baseline_source": baseline_source,
            "missing_reason": ",".join(missing) if missing else "OK",
        }
    )
    return intervals, base_report


def collect_akshare_conversion_price_history(
    conn,
    sample_version: str,
    config: dict[str, Any],
    confirm_download: bool,
    refresh_cache: bool = False,
    max_bonds: int | None = None,
) -> dict[str, Any]:
    ensure_training_tables(conn)
    sample = load_sample(conn, sample_version).drop_duplicates("bond_code", keep="last")
    if sample.empty:
        return {"mode": "collect-akshare-conversion-price-history", "reason": "sample_not_found"}
    if len(sample) > 2 and not confirm_download:
        raise ValueError("AkShare conversion-price collection refused without --confirm-akshare-download")

    manifests = query_to_dataframe(
        conn,
        f"""
        SELECT window_start_date, window_end_date
        FROM {WINDOW_MANIFEST_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
        ORDER BY window_start_date
        """,
    )
    if manifests.empty:
        return {"mode": "collect-akshare-conversion-price-history", "reason": "window_manifest_not_found"}
    ranges = [
        (str(row["window_start_date"])[:10], str(row["window_end_date"])[:10])
        for _, row in manifests.iterrows()
    ]
    range_sql = " OR ".join(
        f"(trade_date >= {sql_string(start)} AND trade_date <= {sql_string(end)})"
        for start, end in ranges
    )
    valid_cb_dates = query_to_dataframe(
        conn,
        f"""
        SELECT bond_code, stock_code, trade_date
        FROM {CANONICAL_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
          AND asset_type = 'CB'
          AND is_valid_bar = true
          AND ({range_sql})
        GROUP BY bond_code, stock_code, trade_date
        """,
    )
    if not valid_cb_dates.empty:
        valid_cb_dates["trade_date"] = pd.to_datetime(valid_cb_dates["trade_date"], errors="coerce").dt.normalize()
        valid_cb_dates["bond_code"] = valid_cb_dates["bond_code"].astype(str).str.zfill(6)

    if max_bonds is not None:
        sample = sample.sort_values("bond_code").head(max(1, int(max_bonds))).copy()
    bond_codes = sample["bond_code"].astype(str).str.zfill(6).tolist()
    stock_codes = sample["stock_code"].astype(str).str.zfill(6).tolist()
    start_date = min(start for start, _ in ranges)
    end_date = max(end for _, end in ranges)
    _, _, selected_dates = selected_training_dates(config)
    date_count = sum(start_date <= value <= end_date for value in selected_dates)

    cache_dir = conversion_price_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = stable_id("conversion", sample_version, start_date, end_date, len(sample))
    stock_cache = cache_dir / f"{cache_key}_jq_stock_daily_fq_none.csv"
    adjustment_cache = cache_dir / f"{cache_key}_jq_adjustments.csv"
    provider = build_jqdata_provider(config)
    estimated_jq_rows = 0
    if not stock_cache.exists() or refresh_cache:
        estimated_jq_rows += len(set(stock_codes)) * date_count
    if not adjustment_cache.exists() or refresh_cache:
        estimated_jq_rows += len(bond_codes) * 5
    used_before = current_quota_used(conn, SOURCE_VENDOR_JQ, now_ts().strftime("%Y-%m-%d"))
    hard_cap = int(config["jqdata"]["default_daily_hard_cap"])
    if estimated_jq_rows and used_before + estimated_jq_rows > hard_cap:
        raise ValueError(
            "JQData conversion-price support fetch refused because daily hard cap would be exceeded "
            f"(used={used_before}, estimated={estimated_jq_rows}, cap={hard_cap})"
        )

    actual_jq_rows = 0
    quota_job_id: str | None = None
    account_id = provider.account_id_hash()
    try:
        if stock_cache.exists() and not refresh_cache:
            stock_daily = pd.read_csv(stock_cache)
        else:
            vendor_codes = [jq_code_from_stock(code) for code in sorted(set(stock_codes))]
            with without_http_proxy():
                stock_daily = provider.get_stock_daily_unadjusted(vendor_codes, start_date, end_date)
            stock_daily.to_csv(stock_cache, index=False)
            actual_jq_rows += len(stock_daily)
        if adjustment_cache.exists() and not refresh_cache:
            adjustments = pd.read_csv(adjustment_cache)
        else:
            with without_http_proxy():
                adjustments = provider.get_conversion_price_adjustments(bond_codes)
            adjustments.to_csv(adjustment_cache, index=False)
            actual_jq_rows += len(adjustments)
        if estimated_jq_rows:
            quota_job_id = record_quota_job(
                conn,
                SOURCE_VENDOR_JQ,
                account_id,
                "collect-akshare-conversion-price-history",
                sample_version,
                estimated_jq_rows,
                actual_jq_rows,
                "SUCCESS",
            )
            record_quota_ledger(
                conn,
                SOURCE_VENDOR_JQ,
                account_id,
                int(config["jqdata"]["daily_quota_limit"]),
                0,
                estimated_jq_rows,
                actual_jq_rows,
                quota_job_id,
            )
    except Exception as exc:
        quota_job_id = record_quota_job(
            conn,
            SOURCE_VENDOR_JQ,
            account_id,
            "collect-akshare-conversion-price-history",
            sample_version,
            estimated_jq_rows,
            actual_jq_rows,
            "FAILED",
            str(exc)[:1000],
        )
        record_quota_ledger(
            conn,
            SOURCE_VENDOR_JQ,
            account_id,
            int(config["jqdata"]["daily_quota_limit"]),
            0,
            estimated_jq_rows,
            actual_jq_rows,
            quota_job_id,
            failed=True,
        )
        raise

    stock_daily["trade_date"] = pd.to_datetime(stock_daily["trade_date"], errors="coerce").dt.normalize()
    stock_daily["stock_code"] = stock_daily["vendor_code"].astype(str).str.extract(r"(\d{6})", expand=False)
    stock_daily["close"] = pd.to_numeric(stock_daily["close"], errors="coerce")
    stock_support_inserted = insert_rows(
        conn,
        JQ_STOCK_DAILY_UNADJUSTED_TABLE,
        stock_daily_support_rows(stock_daily, sample_version, cache_key, now_ts()),
    )
    if not adjustments.empty:
        adjustments["bond_code"] = adjustments["bond_code"].astype(str).str.zfill(6)
        adjustments["effective_date"] = pd.to_datetime(adjustments["effective_date"], errors="coerce")
        adjustments["announcement_date"] = pd.to_datetime(adjustments["announcement_date"], errors="coerce")

    ak_cfg = config["akshare_static"]
    cooldown = float(ak_cfg.get("request_cooldown_seconds", 2.0))
    jitter = float(ak_cfg.get("request_cooldown_jitter_seconds", 0.5))
    retries = int(ak_cfg.get("max_retries", 3))
    now = now_ts()
    interval_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    cache_hits = 0
    fetched = 0
    for position, (_, item) in enumerate(sample.sort_values("bond_code").iterrows(), start=1):
        bond_code = str(item["bond_code"]).zfill(6)
        stock_code = str(item["stock_code"]).zfill(6)
        cache_hit = True
        target = (
            valid_cb_dates.loc[valid_cb_dates["bond_code"] == bond_code, "trade_date"].dropna().tolist()
            if not valid_cb_dates.empty
            else []
        )
        try:
            akshare_values, cache_hit = fetch_akshare_conversion_value_history(
                bond_code,
                cache_dir,
                refresh_cache,
                retries,
                cooldown,
            )
            cache_hits += int(cache_hit)
            fetched += int(not cache_hit)
            intervals, report = infer_conversion_price_intervals(
                sample_version,
                bond_code,
                stock_code,
                target,
                stock_daily[stock_daily["stock_code"] == stock_code],
                akshare_values,
                adjustments[adjustments["bond_code"] == bond_code] if not adjustments.empty else adjustments,
                now,
                mismatch_tolerance=float(ak_cfg.get("conversion_price_mismatch_tolerance", 0.08)),
            )
            interval_rows.extend(intervals)
        except Exception as exc:
            errors.append(f"{bond_code}: {exc}")
            report = {
                "bond_code": bond_code,
                "stock_code": stock_code,
                "expected_trading_days": len(target),
                "covered_trading_days": 0,
                "interval_count": 0,
                "akshare_anchor_days": 0,
                "jq_adjustment_events": 0,
                "qa_mismatch_days": 0,
                "coverage_ratio": 0.0,
                "coverage_pass": False,
                "baseline_anchor_date": None,
                "baseline_source": None,
                "missing_reason": str(exc)[:1000],
            }
        report.update(
            {
                "ts": conversion_history_ts(sample_version, bond_code, "coverage"),
                "sample_version": sample_version,
                "window_start_date": start_date,
                "window_end_date": end_date,
                "checked_at": now,
            }
        )
        coverage_rows.append(report)
        if position % 10 == 0 or position == len(sample):
            LOGGER.info(
                "conversion-price progress=%s/%s intervals=%s errors=%s cache_hits=%s",
                position,
                len(sample),
                len(interval_rows),
                len(errors),
                cache_hits,
            )
        if not cache_hit:
            time.sleep(max(0.0, cooldown + random.uniform(0.0, max(0.0, jitter))))

    inserted_intervals = insert_rows(conn, CONVERSION_PRICE_HISTORY_TABLE, interval_rows)
    inserted_coverage = insert_rows(conn, CONVERSION_PRICE_COVERAGE_TABLE, coverage_rows)
    passed = sum(bool(row.get("coverage_pass")) for row in coverage_rows)
    no_valid_bars = sum(row.get("missing_reason") == "no_valid_cb_bars_in_windows" for row in coverage_rows)
    return {
        "mode": "collect-akshare-conversion-price-history",
        "sample_version": sample_version,
        "window_start_date": start_date,
        "window_end_date": end_date,
        "sample_bonds": int(len(sample)),
        "akshare_fetched_bonds": int(fetched),
        "akshare_cache_hits": int(cache_hits),
        "jq_estimated_rows": int(estimated_jq_rows),
        "jq_actual_rows": int(actual_jq_rows),
        "jq_quota_used_before": int(used_before),
        "jq_quota_job_id": quota_job_id,
        "interval_rows_upserted": int(inserted_intervals),
        "stock_daily_unadjusted_rows_upserted": int(stock_support_inserted),
        "coverage_rows_upserted": int(inserted_coverage),
        "coverage_pass_bonds": int(passed),
        "no_valid_cb_bar_bonds": int(no_valid_bars),
        "error_count": int(len(errors)),
        "errors": errors[:20],
        "cache_dir": str(cache_dir),
    }


def training_window_ranges(conn, sample_version: str) -> list[tuple[str, str, int]]:
    manifest = query_to_dataframe(
        conn,
        f"""
        SELECT window_start_date, window_end_date, trading_days
        FROM {WINDOW_MANIFEST_TABLE}
        WHERE sample_version = {sql_string(sample_version)}
        ORDER BY window_start_date
        """,
    )
    if manifest.empty:
        return []
    return [
        (str(row["window_start_date"])[:10], str(row["window_end_date"])[:10], int(row["trading_days"]))
        for _, row in manifest.iterrows()
    ]


def summarize_training_bar_days(day_rows: pd.DataFrame, asset_type: str, bars_per_day: int = 48) -> dict[str, Any]:
    asset = str(asset_type).upper()
    frame = day_rows[day_rows["asset_type"].astype(str).str.upper() == asset].copy()
    if frame.empty:
        return {
            "asset_type": asset,
            "instrument_days": 0,
            "row_complete_days": 0,
            "valid_days": 0,
            "repairable_days": 0,
            "repairable_examples": [],
        }
    for column in ("row_count", "valid_rows", "null_volume", "null_amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    row_incomplete = frame["row_count"] != int(bars_per_day)
    null_market_data = (frame["null_volume"] > 0) | (frame["null_amount"] > 0)
    repairable = row_incomplete.copy()
    expected_unavailable = pd.Series(False, index=frame.index)
    inside_active_span = pd.Series(False, index=frame.index)
    fully_unavailable_bonds = 0
    if asset == "STOCK":
        repairable |= (frame["valid_rows"] != int(bars_per_day)) | null_market_data
    else:
        for _, group in frame.groupby("bond_code"):
            valid = group[group["valid_rows"] == int(bars_per_day)]
            if valid.empty:
                fully_unavailable_bonds += 1
                expected_unavailable.loc[group.index] = True
                continue
            first_valid = valid["trade_date"].min()
            last_valid = valid["trade_date"].max()
            active = (group["trade_date"] >= first_valid) & (group["trade_date"] <= last_valid)
            inside_active_span.loc[group.index] = active.to_numpy()
            expected_unavailable.loc[group.index] = (group["trade_date"] < first_valid).to_numpy()
        repairable |= inside_active_span & (
            (frame["valid_rows"] != int(bars_per_day)) | null_market_data
        )
    examples = frame.loc[
        repairable,
        ["bond_code", "stock_code", "trade_date", "row_count", "valid_rows", "null_volume", "null_amount"],
    ].head(50).copy()
    if not examples.empty:
        examples["trade_date"] = examples["trade_date"].dt.strftime("%Y-%m-%d")
    return {
        "asset_type": asset,
        "instrument_days": int(len(frame)),
        "row_complete_days": int((frame["row_count"] == int(bars_per_day)).sum()),
        "valid_days": int((frame["valid_rows"] == int(bars_per_day)).sum()),
        "expected_unavailable_days": int(expected_unavailable.sum()),
        "inside_active_span_invalid_days": int(
            (inside_active_span & (frame["valid_rows"] != int(bars_per_day))).sum()
        ),
        "fully_unavailable_bonds": int(fully_unavailable_bonds),
        "repairable_days": int(repairable.sum()),
        "repairable_examples": examples.to_dict("records"),
    }


def benchmark_plan_ts(plan_id: str) -> pd.Timestamp:
    return pd.Timestamp("2021-01-01") + pd.Timedelta(milliseconds=zlib.crc32(plan_id.encode("utf-8")))


def benchmark_rows_from_jq_frame(
    frame: pd.DataFrame,
    definition: dict[str, Any],
    sample_version: str,
    batch_id: str,
    benchmark_slot: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    now = now_ts()
    bar_minutes = int(config["canonical"].get("bar_minutes", 5))
    for _, item in frame.iterrows():
        vendor_bar_end = pd.Timestamp(item["bar_start"]).tz_localize(None)
        bar_start = vendor_bar_end - pd.Timedelta(minutes=bar_minutes)
        if not is_session_bar(bar_start):
            continue
        money = item.get("money")
        amount = item.get("amount")
        if pd.isna(amount):
            amount = money
        paused = item.get("paused")
        if pd.isna(paused):
            paused = None
        raw = {
            "ts": bar_start + pd.Timedelta(milliseconds=int(benchmark_slot)),
            "source_vendor": SOURCE_VENDOR_JQ,
            "account_type": config["jqdata"]["account_type"],
            "research_only": True,
            "sample_version": sample_version,
            "download_batch_id": batch_id,
            "benchmark_code": definition["benchmark_code"],
            "benchmark_name": definition["benchmark_name"],
            "vendor_code": definition["vendor_code"],
            "benchmark_slot": benchmark_slot,
            "trade_date": bar_start.strftime("%Y-%m-%d"),
            "bar_start": bar_start,
            "bar_end": vendor_bar_end,
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "volume": item.get("volume"),
            "money": money,
            "amount": amount,
            "paused": paused,
            "bar_time_semantics": "vendor_bar_end_normalized",
            "downloaded_at": now,
        }
        raw["raw_payload_hash"] = raw_payload_hash(raw)
        valid, flag = bar_quality_flags(pd.Series(raw))
        canonical_rows.append(
            {
                "ts": raw["ts"],
                "source_vendor": SOURCE_VENDOR_JQ,
                "research_only": True,
                "sample_version": sample_version,
                "benchmark_code": definition["benchmark_code"],
                "benchmark_name": definition["benchmark_name"],
                "vendor_code": definition["vendor_code"],
                "trade_date": raw["trade_date"],
                "bar_start": bar_start,
                "bar_end": vendor_bar_end,
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "volume": raw["volume"],
                "money": money,
                "amount": amount,
                "amount_source": "money" if pd.isna(item.get("amount")) and pd.notna(money) else "amount",
                "paused": paused,
                "is_valid_bar": valid,
                "data_quality_flag": flag,
                "canonical_version": config["canonical"]["canonical_version"],
                "created_at": now,
            }
        )
        raw_rows.append(raw)
    return raw_rows, canonical_rows


def benchmark_day_coverage(conn, start_date: str, end_date: str) -> dict[tuple[str, str], int]:
    result = query_to_dataframe(
        conn,
        f"""
        SELECT vendor_code, trade_date, COUNT(*) AS row_count
        FROM {JQ_BENCHMARK_RAW_TABLE}
        WHERE trade_date >= {sql_string(start_date)} AND trade_date <= {sql_string(end_date)}
        GROUP BY vendor_code, trade_date
        """,
    )
    if result.empty:
        return {}
    return {
        (str(row["vendor_code"]), str(row["trade_date"])[:10]): int(row["row_count"])
        for _, row in result.iterrows()
    }


def report_jq_benchmark_coverage(conn, sample_version: str, config: dict[str, Any]) -> dict[str, Any]:
    ensure_training_tables(conn)
    ranges = training_window_ranges(conn, sample_version)
    if not ranges:
        return {"coverage_rows": 0, "reason": "window_manifest_not_found"}
    checked_at = now_ts()
    existing = query_to_dataframe(
        conn,
        f"SELECT ts, benchmark_code, window_start_date FROM {BENCHMARK_COVERAGE_TABLE} "
        f"WHERE sample_version = {sql_string(sample_version)}",
    )
    existing_ts = {
        (str(row["benchmark_code"]), str(row["window_start_date"])[:10]): row["ts"]
        for _, row in existing.iterrows()
    } if not existing.empty else {}
    rows: list[dict[str, Any]] = []
    for definition in config["benchmarks"]:
        for start_date, end_date, expected_days in ranges:
            stats = query_to_dataframe(
                conn,
                f"""
                SELECT COUNT(*) AS available_bars,
                       SUM(CASE WHEN is_valid_bar = true THEN 1 ELSE 0 END) AS valid_bars,
                       SUM(CASE WHEN is_valid_bar = false THEN 1 ELSE 0 END) AS invalid_bars,
                       SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_volume_bars,
                       SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS null_amount_bars
                FROM {BENCHMARK_CANONICAL_TABLE}
                WHERE sample_version = {sql_string(sample_version)}
                  AND vendor_code = {sql_string(definition['vendor_code'])}
                  AND trade_date >= {sql_string(start_date)} AND trade_date <= {sql_string(end_date)}
                """,
            )
            days = query_to_dataframe(
                conn,
                f"""
                SELECT trade_date FROM {BENCHMARK_CANONICAL_TABLE}
                WHERE sample_version = {sql_string(sample_version)}
                  AND vendor_code = {sql_string(definition['vendor_code'])}
                  AND trade_date >= {sql_string(start_date)} AND trade_date <= {sql_string(end_date)}
                GROUP BY trade_date
                """,
            )
            values = stats.iloc[0].to_dict() if not stats.empty else {}
            def count_value(name: str) -> int:
                value = pd.to_numeric(values.get(name), errors="coerce")
                return 0 if pd.isna(value) else int(value)

            expected_bars = int(expected_days) * int(config["jqdata"].get("bars_per_day_estimate", 48))
            available_bars = count_value("available_bars")
            valid_bars = count_value("valid_bars")
            invalid_bars = count_value("invalid_bars")
            null_volume = count_value("null_volume_bars")
            null_amount = count_value("null_amount_bars")
            missing: list[str] = []
            if available_bars != expected_bars:
                missing.append(f"missing_bars={expected_bars - available_bars}")
            if valid_bars != expected_bars:
                missing.append(f"invalid_bars={expected_bars - valid_bars}")
            if null_volume:
                missing.append(f"null_volume_bars={null_volume}")
            if null_amount:
                missing.append(f"null_amount_bars={null_amount}")
            key = (str(definition["benchmark_code"]), start_date)
            coverage_id = stable_id("jqbenchcoverage", sample_version, definition["benchmark_code"], start_date)
            rows.append(
                {
                    "ts": existing_ts.get(key, benchmark_plan_ts(coverage_id)),
                    "sample_version": sample_version,
                    "benchmark_code": definition["benchmark_code"],
                    "benchmark_name": definition["benchmark_name"],
                    "vendor_code": definition["vendor_code"],
                    "window_start_date": start_date,
                    "window_end_date": end_date,
                    "expected_trading_days": expected_days,
                    "available_trading_days": len(days),
                    "expected_bars": expected_bars,
                    "available_bars": available_bars,
                    "valid_bars": valid_bars,
                    "invalid_bars": invalid_bars,
                    "null_volume_bars": null_volume,
                    "null_amount_bars": null_amount,
                    "coverage_ratio": valid_bars / expected_bars if expected_bars else 1.0,
                    "coverage_pass": not missing,
                    "missing_reason": ",".join(missing) or "OK",
                    "checked_at": checked_at,
                }
            )
    insert_rows(conn, BENCHMARK_COVERAGE_TABLE, rows)
    return {
        "coverage_rows": len(rows),
        "coverage_pass_rows": sum(bool(row["coverage_pass"]) for row in rows),
        "coverage_failed_rows": sum(not bool(row["coverage_pass"]) for row in rows),
        "expected_bars": sum(int(row["expected_bars"]) for row in rows),
        "available_bars": sum(int(row["available_bars"]) for row in rows),
        "valid_bars": sum(int(row["valid_bars"]) for row in rows),
    }


def collect_jq_benchmarks(
    conn,
    sample_version: str,
    dry_run: bool,
    confirm_large_download: bool,
    max_rows_today: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    ensure_training_tables(conn)
    ranges = training_window_ranges(conn, sample_version)
    if not ranges:
        return {"mode": "collect-jq-benchmarks", "reason": "window_manifest_not_found"}
    bars_per_day = int(config["jqdata"].get("bars_per_day_estimate", 48))
    min_date = min(item[0] for item in ranges)
    max_date = max(item[1] for item in ranges)
    coverage = benchmark_day_coverage(conn, min_date, max_date)
    now = now_ts()
    batch_id = stable_id("jqbenchbatch", sample_version, min_date, max_date)
    existing = query_to_dataframe(
        conn,
        f"SELECT * FROM {BENCHMARK_PLAN_TABLE} WHERE sample_version = {sql_string(sample_version)}",
    )
    existing_by_id = {
        str(row["plan_id"]): row.to_dict() for _, row in existing.iterrows()
    } if not existing.empty else {}
    plans: list[dict[str, Any]] = []
    for slot, definition in enumerate(config["benchmarks"]):
        for start_date, end_date, _ in ranges:
            dates, _ = trading_dates_between(config, pd.Timestamp(start_date), pd.Timestamp(end_date))
            missing_dates = [
                trade_date
                for trade_date in dates
                if coverage.get((str(definition["vendor_code"]), trade_date), 0) < bars_per_day
            ]
            plan_id = stable_id("jqbenchplan", sample_version, definition["benchmark_code"], start_date, end_date)
            old = existing_by_id.get(plan_id, {})
            attempts = pd.to_numeric(old.get("attempt_count"), errors="coerce")
            status = "SKIPPED_ALREADY_EXISTS" if not missing_dates else ("DRY_RUN_ESTIMATED" if dry_run else "READY")
            plans.append(
                {
                    "ts": old.get("ts", benchmark_plan_ts(plan_id)),
                    "plan_id": plan_id,
                    "batch_id": batch_id,
                    "sample_version": sample_version,
                    "source_vendor": SOURCE_VENDOR_JQ,
                    "research_only": True,
                    "benchmark_code": definition["benchmark_code"],
                    "benchmark_name": definition["benchmark_name"],
                    "vendor_code": definition["vendor_code"],
                    "window_start_date": start_date,
                    "window_end_date": end_date,
                    "missing_dates_json": json.dumps(missing_dates, ensure_ascii=True),
                    "estimated_rows": len(missing_dates) * bars_per_day,
                    "status": status,
                    "attempt_count": 0 if pd.isna(attempts) else int(attempts),
                    "last_error": None,
                    "created_at": old.get("created_at", now),
                    "updated_at": now,
                    "_slot": slot,
                    "_missing_dates": missing_dates,
                }
            )
    insert_rows(
        conn,
        BENCHMARK_PLAN_TABLE,
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in plans],
    )
    estimated_rows = sum(int(row["estimated_rows"]) for row in plans)
    result = {
        "mode": "collect-jq-benchmarks",
        "sample_version": sample_version,
        "dry_run": bool(dry_run),
        "batch_id": batch_id,
        "benchmark_count": len(config["benchmarks"]),
        "window_count": len(ranges),
        "plan_count": len(plans),
        "ready_plan_count": sum(bool(row["_missing_dates"]) for row in plans),
        "estimated_rows": estimated_rows,
    }
    if dry_run or estimated_rows == 0:
        return {**result, "coverage": report_jq_benchmark_coverage(conn, sample_version, config)}

    require_large_download_confirmation(
        len(config["benchmarks"]), estimated_rows, confirm_large_download, config
    )
    hard_cap = min(int(max_rows_today), int(config["jqdata"]["default_daily_hard_cap"]))
    used_before = current_quota_used(conn, SOURCE_VENDOR_JQ, now.strftime("%Y-%m-%d"))
    if used_before + estimated_rows > hard_cap:
        raise ValueError(
            f"JQData benchmark download refused by daily cap used={used_before} "
            f"estimated={estimated_rows} cap={hard_cap}"
        )

    provider = build_jqdata_provider(config)
    account_id = provider.account_id_hash()
    fields = validate_jq_fields(
        "STOCK", ["open", "high", "low", "close", "volume", "money", "paused"]
    )
    actual_rows = 0
    raw_inserted = 0
    canonical_inserted = 0
    errors: list[str] = []
    cooldown = float(config["jqdata"].get("request_cooldown_seconds", 1.0))
    for plan in plans:
        if not plan["_missing_dates"]:
            continue
        plan["status"] = "RUNNING"
        plan["attempt_count"] = int(plan["attempt_count"]) + 1
        plan["updated_at"] = now_ts()
        insert_rows(
            conn,
            BENCHMARK_PLAN_TABLE,
            [{key: value for key, value in plan.items() if not key.startswith("_")}],
        )
        definition = {
            "benchmark_code": plan["benchmark_code"],
            "benchmark_name": plan["benchmark_name"],
            "vendor_code": plan["vendor_code"],
        }
        all_dates, _ = trading_dates_between(
            config,
            pd.Timestamp(plan["window_start_date"]),
            pd.Timestamp(plan["window_end_date"]),
        )
        try:
            for range_start, range_end in contiguous_missing_date_ranges(all_dates, plan["_missing_dates"]):
                with without_http_proxy():
                    frame = provider.get_price_5m(plan["vendor_code"], range_start, range_end, fields)
                actual_rows += int(len(frame))
                raw_rows, canonical_rows = benchmark_rows_from_jq_frame(
                    frame,
                    definition,
                    sample_version,
                    batch_id,
                    int(plan["_slot"]),
                    config,
                )
                raw_inserted += insert_raw_rows_idempotent(conn, JQ_BENCHMARK_RAW_TABLE, raw_rows)
                canonical_inserted += insert_raw_rows_idempotent(conn, BENCHMARK_CANONICAL_TABLE, canonical_rows)
                time.sleep(max(0.0, cooldown))
            plan["status"] = "SUCCESS"
            plan["last_error"] = None
        except Exception as exc:
            plan["status"] = "FAILED"
            plan["last_error"] = str(exc)[:1000]
            errors.append(f"{plan['benchmark_code']} {plan['window_start_date']}: {exc}")
        plan["updated_at"] = now_ts()
        insert_rows(
            conn,
            BENCHMARK_PLAN_TABLE,
            [{key: value for key, value in plan.items() if not key.startswith("_")}],
        )

    status = "SUCCESS" if not errors else ("PARTIAL_SUCCESS" if actual_rows else "FAILED")
    job_id = record_quota_job(
        conn,
        SOURCE_VENDOR_JQ,
        account_id,
        "collect-jq-benchmarks",
        sample_version,
        estimated_rows,
        actual_rows,
        status,
        "; ".join(errors)[:1000] if errors else None,
    )
    record_quota_ledger(
        conn,
        SOURCE_VENDOR_JQ,
        account_id,
        int(config["jqdata"]["daily_quota_limit"]),
        0,
        estimated_rows,
        actual_rows,
        job_id,
        failed=bool(errors),
    )
    return {
        **result,
        "dry_run": False,
        "used_before": used_before,
        "actual_vendor_rows": actual_rows,
        "raw_rows_inserted": raw_inserted,
        "canonical_rows_inserted": canonical_inserted,
        "error_count": len(errors),
        "errors": errors,
        "quota_job_id": job_id,
        "coverage": report_jq_benchmark_coverage(conn, sample_version, config),
    }


def audit_jq_training_bars(conn, sample_version: str, config: dict[str, Any]) -> dict[str, Any]:
    ensure_training_tables(conn)
    ranges = training_window_ranges(conn, sample_version)
    if not ranges:
        return {"mode": "audit-jq-training-bars", "sample_version": sample_version, "reason": "window_manifest_not_found"}
    range_sql = " OR ".join(
        f"(trade_date >= {sql_string(start)} AND trade_date <= {sql_string(end)})"
        for start, end, _ in ranges
    )
    day_rows = query_to_dataframe(
        conn,
        f"""
        SELECT asset_type, bond_code, stock_code, trade_date,
               COUNT(*) AS row_count,
               SUM(CASE WHEN is_valid_bar = true THEN 1 ELSE 0 END) AS valid_rows,
               SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_volume,
               SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS null_amount
        FROM {CANONICAL_TABLE}
        WHERE sample_version = {sql_string(sample_version)} AND ({range_sql})
        GROUP BY asset_type, bond_code, stock_code, trade_date
        """,
    )
    bars_per_day = int(config["jqdata"].get("bars_per_day_estimate", 48))
    stock = summarize_training_bar_days(day_rows, "STOCK", bars_per_day)
    cb = summarize_training_bar_days(day_rows, "CB", bars_per_day)
    return {
        "mode": "audit-jq-training-bars",
        "sample_version": sample_version,
        "window_count": len(ranges),
        "window_trading_days": int(sum(item[2] for item in ranges)),
        "bars_per_day": bars_per_day,
        "stock": stock,
        "cb": cb,
        "repair_required": bool(stock["repairable_days"] or cb["repairable_days"]),
    }


@dataclass
class TrainingPipeline:
    config_path: str | None = None

    def __post_init__(self) -> None:
        self.config = load_config(self.config_path)

    def with_conn(self):
        conn = get_conn()
        ensure_training_tables(conn)
        return conn
