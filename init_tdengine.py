# init_tdengine.py

import re

from config import TDENGINE_DATABASE
from db import get_conn
from intraday_factors.data_source import create_alert_table, create_factor_table, create_job_run_table, create_live_backfill_diff_table
from training_data_v06.schemas import ensure_training_tables


def try_execute(conn, sql: str):
    try:
        conn.execute(sql)
    except Exception:
        pass


def init_db():
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", TDENGINE_DATABASE):
        raise ValueError(f"invalid TDengine database name: {TDENGINE_DATABASE!r}")
    conn = get_conn(use_db=False)
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {TDENGINE_DATABASE}")
    conn.execute(f"USE {TDENGINE_DATABASE}")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS cb_watchlist_bar_5m_live (
        ts TIMESTAMP,
        bar_start TIMESTAMP,
        trade_date NCHAR(16),
        asset_type NCHAR(16),
        watchlist_status NCHAR(16),
        pool_state NCHAR(32),
        instrument_code NCHAR(16),
        instrument_name NCHAR(32),
        bond_code NCHAR(16),
        bond_name NCHAR(32),
        stock_code NCHAR(16),
        stock_name NCHAR(32),
        market NCHAR(8),
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        change_percent DOUBLE,
        premium_rate DOUBLE,
        ytm DOUBLE,
        conversion_value DOUBLE,
        conversion_price DOUBLE,
        volume DOUBLE,
        amount DOUBLE,
        turnover_rate DOUBLE,
        ytm_available BOOL,
        ytm_source NCHAR(32),
        turnover_available BOOL,
        turnover_source NCHAR(32),
        premium_source NCHAR(32),
        volatility DOUBLE,
        quote_count INT,
        source NCHAR(32)
    )
    """)

    conn.execute("""
    CREATE STABLE IF NOT EXISTS st_cb_bar_1d (
        ts TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        amount DOUBLE,
        source NCHAR(32)
    )
    TAGS (
        bond_code NCHAR(16),
        bond_name NCHAR(32),
        market NCHAR(8)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS cb_screening_daily (
        ts TIMESTAMP,
        snapshot_id NCHAR(32),
        bond_code NCHAR(16),
        bond_name NCHAR(32),
        ak_symbol NCHAR(16),
        bond_price DOUBLE,
        premium_rate DOUBLE,
        conversion_value DOUBLE,
        conversion_price DOUBLE,
        stock_code NCHAR(16),
        stock_name NCHAR(32),
        stock_price DOUBLE,
        listed_date TIMESTAMP,
        maturity_date TIMESTAMP,
        days_to_maturity INT,
        report_date TIMESTAMP,
        debt_asset_ratio DOUBLE,
        operating_cash_flow DOUBLE,
        free_cash_flow DOUBLE,
        watchlist_status NCHAR(16),
        pool_state NCHAR(32),
        candidate_since TIMESTAMP,
        active_since TIMESTAMP,
        source NCHAR(64)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS cb_watchlist_daily (
        ts TIMESTAMP,
        snapshot_id NCHAR(32),
        bond_code NCHAR(16),
        bond_name NCHAR(32),
        ak_symbol NCHAR(16),
        bond_price DOUBLE,
        premium_rate DOUBLE,
        conversion_value DOUBLE,
        conversion_price DOUBLE,
        stock_code NCHAR(16),
        stock_name NCHAR(32),
        stock_price DOUBLE,
        maturity_date TIMESTAMP,
        days_to_maturity INT,
        report_date TIMESTAMP,
        debt_asset_ratio DOUBLE,
        operating_cash_flow DOUBLE,
        free_cash_flow DOUBLE,
        watchlist_status NCHAR(16),
        pool_state NCHAR(32),
        candidate_since TIMESTAMP,
        active_since TIMESTAMP,
        source NCHAR(64)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS cb_intraday_factor_snapshot (
        ts TIMESTAMP,
        bar_start TIMESTAMP,
        bond_code NCHAR(16),
        stock_code NCHAR(16),
        factor_version NCHAR(16),
        parity DOUBLE,
        premium_raw DOUBLE,
        premium_slot_z DOUBLE,
        premium_model_residual DOUBLE,
        premium_model_residual_z DOUBLE,
        premium_factor_source NCHAR(32),
        r_cb DOUBLE,
        r_stock DOUBLE,
        r_cb_market DOUBLE,
        alpha DOUBLE,
        beta DOUBLE,
        gamma DOUBLE,
        residual DOUBLE,
        residual_ewm DOUBLE,
        residual_z DOUBLE,
        factor_residual_mr DOUBLE,
        stock_mom_30m DOUBLE,
        stock_mom_60m DOUBLE,
        stock_mom_1d DOUBLE,
        stock_mom_5d DOUBLE,
        stock_momentum_score DOUBLE,
        stock_rv_short DOUBLE,
        stock_rv_long DOUBLE,
        cb_rv_short DOUBLE,
        cb_rv_long DOUBLE,
        residual_rv_short DOUBLE,
        residual_rv_long DOUBLE,
        factor_vol_mr DOUBLE,
        empirical_delta_proxy DOUBLE,
        elasticity_proxy DOUBLE,
        empirical_gamma_proxy DOUBLE,
        convexity_asymmetry_proxy DOUBLE,
        factor_vol_response_gap DOUBLE,
        cb_flow_30m DOUBLE,
        stock_flow_30m DOUBLE,
        flow_confirmation DOUBLE,
        flow_lead DOUBLE,
        adv20_amount DOUBLE,
        relative_amount DOUBLE,
        active_bar_ratio DOUBLE,
        amihud DOUBLE,
        high_low_range_proxy DOUBLE,
        data_quality_pass BOOL,
        liquidity_pass BOOL,
        trend_gate BOOL,
        factor_coverage DOUBLE,
        factor_coverage_pass BOOL,
        missing_cb_bar BOOL,
        missing_stock_bar BOOL,
        stale_price_flag BOOL,
        insufficient_history_flag BOOL,
        fallback_standardization_flag BOOL,
        signal_score DOUBLE,
        signal_state NCHAR(32),
        amount_source NCHAR(32),
        calculated_at TIMESTAMP
    )
    """)
    create_factor_table(conn, "cb_intraday_factor_snapshot")
    create_alert_table(conn, "cb_intraday_alerts")
    create_job_run_table(conn, "cb_intraday_job_runs")
    create_live_backfill_diff_table(conn, "cb_intraday_live_backfill_diff")
    ensure_training_tables(conn)

    try_execute(conn, "ALTER TABLE cb_screening_daily ADD COLUMN snapshot_id NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN snapshot_id NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_screening_daily ADD COLUMN maturity_date TIMESTAMP")
    try_execute(conn, "ALTER TABLE cb_screening_daily ADD COLUMN days_to_maturity INT")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN maturity_date TIMESTAMP")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN days_to_maturity INT")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN asset_type NCHAR(16)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN watchlist_status NCHAR(16)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN pool_state NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN instrument_code NCHAR(16)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN instrument_name NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN stock_code NCHAR(16)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN stock_name NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN ytm_available BOOL")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN ytm_source NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN turnover_available BOOL")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN turnover_source NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_bar_5m_live ADD COLUMN premium_source NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN watchlist_status NCHAR(16)")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN pool_state NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN candidate_since TIMESTAMP")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN active_since TIMESTAMP")

    conn.close()
    print("TDengine schema initialized.")


if __name__ == "__main__":
    init_db()
