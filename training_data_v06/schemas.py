from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from intraday_factors.data_source import sql_bool, sql_number, sql_string, try_execute


LOGGER = logging.getLogger(__name__)


TRAINING_SAMPLE_TABLE = "training_cb_sample"
QUOTA_LEDGER_TABLE = "vendor_quota_ledger"
QUOTA_JOB_TABLE = "vendor_quota_job"
DOWNLOAD_PLAN_TABLE = "jq_download_plan"
BENCHMARK_PLAN_TABLE = "jq_benchmark_download_plan"
JQ_CB_RAW_TABLE = "jq_cb_bar_5m_raw"
JQ_STOCK_RAW_TABLE = "jq_stock_bar_5m_raw"
JQ_BENCHMARK_RAW_TABLE = "jq_benchmark_bar_5m_raw"
JQ_STOCK_DAILY_UNADJUSTED_TABLE = "jq_stock_daily_unadjusted_support"
WIND_RAW_TABLE = "wind_cb_ytm_turnover_raw"
AKSHARE_STATIC_TABLE = "akshare_cb_static_snapshot"
CONVERSION_PRICE_HISTORY_TABLE = "cb_conversion_price_history_pit"
CONVERSION_PRICE_COVERAGE_TABLE = "cb_conversion_price_coverage_report"
CANONICAL_TABLE = "market_bar_5m_training_canonical"
COVERAGE_TABLE = "jq_training_coverage_report"
WINDOW_MANIFEST_TABLE = "training_window_manifest"
BENCHMARK_CANONICAL_TABLE = "market_benchmark_bar_5m_canonical"
BENCHMARK_COVERAGE_TABLE = "jq_benchmark_coverage_report"


TABLE_COLUMNS: dict[str, list[str]] = {
    TRAINING_SAMPLE_TABLE: [
        "ts TIMESTAMP",
        "sample_version NCHAR(64)",
        "selection_version NCHAR(64)",
        "selected_at TIMESTAMP",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "bond_name NCHAR(64)",
        "stock_name NCHAR(64)",
        "research_pool_scope NCHAR(32)",
        "sample_selection_reason NCHAR(1024)",
        "exclusion_reason NCHAR(1024)",
        "premium_rate DOUBLE",
        "days_to_maturity INT",
        "debt_asset_ratio DOUBLE",
        "operating_cash_flow DOUBLE",
        "free_cash_flow DOUBLE",
        "maturity_date TIMESTAMP",
        "conversion_price DOUBLE",
        "source_vendor NCHAR(32)",
        "snapshot_date TIMESTAMP",
    ],
    QUOTA_LEDGER_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "account_id_hash NCHAR(64)",
        "trade_date NCHAR(16)",
        "week_id NCHAR(16)",
        "quota_limit_daily INT",
        "quota_limit_weekly INT",
        "quota_used_estimated INT",
        "quota_used_actual INT",
        "quota_remaining_estimated INT",
        "job_count INT",
        "failed_job_count INT",
        "last_job_id NCHAR(64)",
        "created_at TIMESTAMP",
        "updated_at TIMESTAMP",
    ],
    QUOTA_JOB_TABLE: [
        "ts TIMESTAMP",
        "job_id NCHAR(64)",
        "source_vendor NCHAR(32)",
        "account_id_hash NCHAR(64)",
        "trade_date NCHAR(16)",
        "week_id NCHAR(16)",
        "mode NCHAR(64)",
        "sample_version NCHAR(64)",
        "estimated_rows INT",
        "actual_rows INT",
        "status NCHAR(32)",
        "error_message NCHAR(1024)",
        "created_at TIMESTAMP",
        "updated_at TIMESTAMP",
    ],
    DOWNLOAD_PLAN_TABLE: [
        "ts TIMESTAMP",
        "plan_id NCHAR(64)",
        "batch_id NCHAR(64)",
        "data_source NCHAR(32)",
        "research_only BOOL",
        "sample_version NCHAR(64)",
        "permission_anchor_date NCHAR(16)",
        "permission_safe_start NCHAR(16)",
        "permission_safe_end NCHAR(16)",
        "instrument_code NCHAR(32)",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "asset_type NCHAR(16)",
        "instrument_slot INT",
        "start_date NCHAR(16)",
        "end_date NCHAR(16)",
        "frequency NCHAR(16)",
        "fields_json NCHAR(1024)",
        "estimated_rows INT",
        "priority INT",
        "status NCHAR(32)",
        "attempt_count INT",
        "last_error NCHAR(1024)",
        "created_at TIMESTAMP",
        "updated_at TIMESTAMP",
    ],
    BENCHMARK_PLAN_TABLE: [
        "ts TIMESTAMP",
        "plan_id NCHAR(64)",
        "batch_id NCHAR(64)",
        "sample_version NCHAR(64)",
        "source_vendor NCHAR(32)",
        "research_only BOOL",
        "benchmark_code NCHAR(32)",
        "benchmark_name NCHAR(64)",
        "vendor_code NCHAR(32)",
        "window_start_date NCHAR(16)",
        "window_end_date NCHAR(16)",
        "missing_dates_json NCHAR(1024)",
        "estimated_rows INT",
        "status NCHAR(32)",
        "attempt_count INT",
        "last_error NCHAR(1024)",
        "created_at TIMESTAMP",
        "updated_at TIMESTAMP",
    ],
    JQ_CB_RAW_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "account_type NCHAR(32)",
        "research_only BOOL",
        "download_batch_id NCHAR(64)",
        "vendor_code NCHAR(32)",
        "instrument_slot INT",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "trade_date NCHAR(16)",
        "bar_start TIMESTAMP",
        "bar_end TIMESTAMP",
        "open DOUBLE",
        "high DOUBLE",
        "low DOUBLE",
        "close DOUBLE",
        "volume DOUBLE",
        "money DOUBLE",
        "amount DOUBLE",
        "paused BOOL",
        "bar_time_semantics NCHAR(64)",
        "raw_payload_hash NCHAR(64)",
        "downloaded_at TIMESTAMP",
    ],
    JQ_STOCK_RAW_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "account_type NCHAR(32)",
        "research_only BOOL",
        "download_batch_id NCHAR(64)",
        "vendor_code NCHAR(32)",
        "instrument_slot INT",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "trade_date NCHAR(16)",
        "bar_start TIMESTAMP",
        "bar_end TIMESTAMP",
        "open DOUBLE",
        "high DOUBLE",
        "low DOUBLE",
        "close DOUBLE",
        "volume DOUBLE",
        "money DOUBLE",
        "amount DOUBLE",
        "paused BOOL",
        "bar_time_semantics NCHAR(64)",
        "raw_payload_hash NCHAR(64)",
        "downloaded_at TIMESTAMP",
    ],
    JQ_BENCHMARK_RAW_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "account_type NCHAR(32)",
        "research_only BOOL",
        "sample_version NCHAR(64)",
        "download_batch_id NCHAR(64)",
        "benchmark_code NCHAR(32)",
        "benchmark_name NCHAR(64)",
        "vendor_code NCHAR(32)",
        "benchmark_slot INT",
        "trade_date NCHAR(16)",
        "bar_start TIMESTAMP",
        "bar_end TIMESTAMP",
        "open DOUBLE",
        "high DOUBLE",
        "low DOUBLE",
        "close DOUBLE",
        "volume DOUBLE",
        "money DOUBLE",
        "amount DOUBLE",
        "paused BOOL",
        "bar_time_semantics NCHAR(64)",
        "raw_payload_hash NCHAR(64)",
        "downloaded_at TIMESTAMP",
    ],
    JQ_STOCK_DAILY_UNADJUSTED_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "research_only BOOL",
        "sample_version NCHAR(64)",
        "download_batch_id NCHAR(64)",
        "vendor_code NCHAR(32)",
        "stock_code NCHAR(16)",
        "trade_date NCHAR(16)",
        "close DOUBLE",
        "vendor_close DOUBLE",
        "is_filled BOOL",
        "price_status NCHAR(32)",
        "fq_mode NCHAR(16)",
        "downloaded_at TIMESTAMP",
    ],
    WIND_RAW_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "account_type NCHAR(32)",
        "research_only BOOL",
        "download_batch_id NCHAR(64)",
        "bond_code NCHAR(16)",
        "trade_date NCHAR(16)",
        "ytm DOUBLE",
        "live_turnover DOUBLE",
        "ytm_source NCHAR(32)",
        "live_turnover_source NCHAR(32)",
        "downloaded_at TIMESTAMP",
    ],
    AKSHARE_STATIC_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "snapshot_date NCHAR(16)",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "maturity_date TIMESTAMP",
        "st_status NCHAR(32)",
        "conversion_price DOUBLE",
        "conversion_price_source NCHAR(32)",
        "bond_name NCHAR(64)",
        "stock_name NCHAR(64)",
        "listing_date TIMESTAMP",
        "delisting_date TIMESTAMP",
        "downloaded_at TIMESTAMP",
    ],
    CONVERSION_PRICE_HISTORY_TABLE: [
        "ts TIMESTAMP",
        "sample_version NCHAR(64)",
        "research_only BOOL",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "effective_start NCHAR(16)",
        "effective_end_exclusive NCHAR(16)",
        "conversion_price DOUBLE",
        "source_vendor NCHAR(64)",
        "price_source NCHAR(128)",
        "source_effective_date NCHAR(16)",
        "announcement_date TIMESTAMP",
        "adjustment_reason NCHAR(256)",
        "baseline_anchor_date NCHAR(16)",
        "confidence NCHAR(32)",
        "source_detail NCHAR(1024)",
        "downloaded_at TIMESTAMP",
    ],
    CONVERSION_PRICE_COVERAGE_TABLE: [
        "ts TIMESTAMP",
        "sample_version NCHAR(64)",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "window_start_date NCHAR(16)",
        "window_end_date NCHAR(16)",
        "expected_trading_days INT",
        "covered_trading_days INT",
        "interval_count INT",
        "akshare_anchor_days INT",
        "jq_adjustment_events INT",
        "qa_mismatch_days INT",
        "coverage_ratio DOUBLE",
        "coverage_pass BOOL",
        "baseline_anchor_date NCHAR(16)",
        "baseline_source NCHAR(128)",
        "missing_reason NCHAR(1024)",
        "checked_at TIMESTAMP",
    ],
    CANONICAL_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "research_only BOOL",
        "sample_version NCHAR(64)",
        "research_pool_scope NCHAR(32)",
        "trade_date NCHAR(16)",
        "bar_start TIMESTAMP",
        "bar_end TIMESTAMP",
        "asset_type NCHAR(16)",
        "instrument_code NCHAR(32)",
        "instrument_slot INT",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "open DOUBLE",
        "high DOUBLE",
        "low DOUBLE",
        "close DOUBLE",
        "volume DOUBLE",
        "amount DOUBLE",
        "money DOUBLE",
        "amount_source NCHAR(32)",
        "paused BOOL",
        "is_valid_bar BOOL",
        "data_quality_flag NCHAR(1024)",
        "canonical_version NCHAR(64)",
        "created_at TIMESTAMP",
    ],
    COVERAGE_TABLE: [
        "ts TIMESTAMP",
        "batch_id NCHAR(64)",
        "sample_version NCHAR(64)",
        "bond_code NCHAR(16)",
        "stock_code NCHAR(16)",
        "research_pool_scope NCHAR(32)",
        "expected_trading_days INT",
        "available_cb_trading_days INT",
        "available_stock_trading_days INT",
        "valid_cb_trading_days INT",
        "valid_stock_trading_days INT",
        "expected_cb_bars INT",
        "available_cb_bars INT",
        "valid_cb_bars INT",
        "invalid_cb_bars INT",
        "expected_stock_bars INT",
        "available_stock_bars INT",
        "valid_stock_bars INT",
        "invalid_stock_bars INT",
        "row_coverage_ratio DOUBLE",
        "coverage_ratio DOUBLE",
        "coverage_pass BOOL",
        "missing_reason NCHAR(1024)",
        "checked_at TIMESTAMP",
    ],
    WINDOW_MANIFEST_TABLE: [
        "ts TIMESTAMP",
        "local_window_id NCHAR(64)",
        "sample_version NCHAR(64)",
        "window_start_date NCHAR(16)",
        "window_end_date NCHAR(16)",
        "trading_days INT",
        "source_vendor NCHAR(32)",
        "instrument_count INT",
        "cb_bar_count INT",
        "stock_bar_count INT",
        "created_at TIMESTAMP",
    ],
    BENCHMARK_CANONICAL_TABLE: [
        "ts TIMESTAMP",
        "source_vendor NCHAR(32)",
        "research_only BOOL",
        "sample_version NCHAR(64)",
        "benchmark_code NCHAR(32)",
        "benchmark_name NCHAR(64)",
        "vendor_code NCHAR(32)",
        "trade_date NCHAR(16)",
        "bar_start TIMESTAMP",
        "bar_end TIMESTAMP",
        "open DOUBLE",
        "high DOUBLE",
        "low DOUBLE",
        "close DOUBLE",
        "volume DOUBLE",
        "money DOUBLE",
        "amount DOUBLE",
        "amount_source NCHAR(32)",
        "paused BOOL",
        "is_valid_bar BOOL",
        "data_quality_flag NCHAR(1024)",
        "canonical_version NCHAR(64)",
        "created_at TIMESTAMP",
    ],
    BENCHMARK_COVERAGE_TABLE: [
        "ts TIMESTAMP",
        "sample_version NCHAR(64)",
        "benchmark_code NCHAR(32)",
        "benchmark_name NCHAR(64)",
        "vendor_code NCHAR(32)",
        "window_start_date NCHAR(16)",
        "window_end_date NCHAR(16)",
        "expected_trading_days INT",
        "available_trading_days INT",
        "expected_bars INT",
        "available_bars INT",
        "valid_bars INT",
        "invalid_bars INT",
        "null_volume_bars INT",
        "null_amount_bars INT",
        "coverage_ratio DOUBLE",
        "coverage_pass BOOL",
        "missing_reason NCHAR(1024)",
        "checked_at TIMESTAMP",
    ],
}


NUMERIC_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "amount",
    "premium_rate",
    "debt_asset_ratio",
    "operating_cash_flow",
    "free_cash_flow",
    "conversion_price",
    "vendor_close",
    "ytm",
    "live_turnover",
    "row_coverage_ratio",
    "coverage_ratio",
}
INT_COLUMNS = {
    "quota_limit_daily",
    "quota_limit_weekly",
    "quota_used_estimated",
    "quota_used_actual",
    "quota_remaining_estimated",
    "job_count",
    "failed_job_count",
    "estimated_rows",
    "actual_rows",
    "priority",
    "attempt_count",
    "days_to_maturity",
    "expected_trading_days",
    "available_cb_trading_days",
    "available_stock_trading_days",
    "valid_cb_trading_days",
    "valid_stock_trading_days",
    "expected_cb_bars",
    "available_cb_bars",
    "valid_cb_bars",
    "invalid_cb_bars",
    "expected_stock_bars",
    "expected_bars",
    "available_stock_bars",
    "valid_stock_bars",
    "invalid_stock_bars",
    "trading_days",
    "instrument_count",
    "cb_bar_count",
    "stock_bar_count",
    "covered_trading_days",
    "interval_count",
    "akshare_anchor_days",
    "jq_adjustment_events",
    "qa_mismatch_days",
    "benchmark_slot",
    "available_trading_days",
    "available_bars",
    "valid_bars",
    "invalid_bars",
    "null_volume_bars",
    "null_amount_bars",
}
BOOL_COLUMNS = {"research_only", "paused", "is_valid_bar", "coverage_pass", "is_filled"}


def ensure_training_tables(conn) -> None:
    for table, columns in TABLE_COLUMNS.items():
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(columns)})")
        try:
            existing = {str(row[0]).lower() for row in conn.query(f"DESCRIBE {table}")}
        except Exception:
            existing = set()
        for definition in columns[1:]:
            if definition.split()[0].lower() in existing:
                continue
            try_execute(conn, f"ALTER TABLE {table} ADD COLUMN {definition}")


def table_column_names(table: str) -> list[str]:
    return [definition.split()[0] for definition in TABLE_COLUMNS[table]]


def column_sql_type(table: str, column: str) -> str:
    for definition in TABLE_COLUMNS[table]:
        parts = definition.split()
        if parts[0] == column:
            return parts[1].upper()
    raise KeyError(f"unknown column {table}.{column}")


def sql_value(table: str, column: str, value: Any) -> str:
    sql_type = column_sql_type(table, column)
    if sql_type == "BOOL":
        return sql_bool(value)
    if sql_type in {"DOUBLE", "FLOAT"}:
        return sql_number(value)
    if sql_type in {"INT", "BIGINT", "SMALLINT", "TINYINT"}:
        if value is None or pd.isna(value):
            return "NULL"
        return str(int(value))
    if sql_type == "TIMESTAMP":
        if value is None or pd.isna(value):
            return "NULL"
        return sql_string(pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    return sql_string(value)


def insert_rows(conn, table: str, rows: list[dict[str, Any]], chunk_size: int = 500) -> int:
    if not rows:
        return 0
    columns = table_column_names(table)
    inserted = 0
    for offset in range(0, len(rows), chunk_size):
        values = []
        for row in rows[offset:offset + chunk_size]:
            values.append("(" + ", ".join(sql_value(table, col, row.get(col)) for col in columns) + ")")
        conn.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES " + " ".join(values))
        inserted += len(values)
    return inserted
