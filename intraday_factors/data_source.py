from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from db import get_conn
from intraday_factors.config import load_schema
from intraday_factors.pool_state import DEFAULT_ACTIONABLE_POOL_STATES, normalize_pool_state
from intraday_factors.trading_session import completed_bar_cutoff, ensure_market_naive, is_session_bar, session_bar_role, slot_id
from intraday_factors.utils import point_in_time_asof_join


LOGGER = logging.getLogger(__name__)


def sql_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    escaped = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def sql_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    return str(float(value))


def sql_bool(value: Any) -> str:
    if value is None or pd.isna(value):
        return "0"
    return "1" if bool(value) else "0"


def query_to_dataframe(conn, sql: str) -> pd.DataFrame:
    result = conn.query(sql)
    rows = list(result)
    fields = [field.name for field in result.fields]
    return pd.DataFrame(rows, columns=fields)


def try_execute(conn, sql: str) -> None:
    try:
        conn.execute(sql)
    except Exception as exc:
        LOGGER.debug("Ignored SQL failure: %s sql=%s", exc, sql)


@dataclass
class IntradayDataAdapter:
    schema: dict[str, Any]
    bar_minutes: int = 5

    @classmethod
    def from_files(cls, schema_path: str | None = None, bar_minutes: int = 5) -> "IntradayDataAdapter":
        return cls(schema=load_schema(schema_path), bar_minutes=bar_minutes)

    @property
    def live_table(self) -> str:
        return self.schema["tables"]["live_5m"]

    @property
    def watchlist_table(self) -> str:
        return self.schema["tables"]["watchlist_daily"]

    @property
    def factor_table(self) -> str:
        return self.schema["tables"]["factor_snapshot"]

    def load_pool_bond_codes(self, conn, asof: str | pd.Timestamp, pool_states: set[str] | None = None) -> set[str]:
        asof_ts = ensure_market_naive(asof)
        allowed_states = pool_states or DEFAULT_ACTIONABLE_POOL_STATES
        try:
            snapshot_df = query_to_dataframe(
                conn,
                f"""
                SELECT LAST(snapshot_id) AS snapshot_id
                FROM {self.watchlist_table}
                WHERE ts <= {sql_string(asof_ts.strftime('%Y-%m-%d %H:%M:%S'))}
                """,
            )
            if snapshot_df.empty or pd.isna(snapshot_df.loc[0, "snapshot_id"]):
                return set()

            snapshot_id = str(snapshot_df.loc[0, "snapshot_id"])
            watchlist = query_to_dataframe(
                conn,
                f"""
                SELECT bond_code, watchlist_status, pool_state
                FROM {self.watchlist_table}
                WHERE snapshot_id = {sql_string(snapshot_id)}
                """,
            )
        except Exception as exc:
            LOGGER.warning("Failed to load active watchlist asof=%s: %s", asof_ts, exc)
            return set()

        if watchlist.empty or "bond_code" not in watchlist.columns:
            return set()
        status_col = "pool_state" if "pool_state" in watchlist.columns else "watchlist_status"
        if status_col in watchlist.columns:
            status = watchlist[status_col].apply(lambda value: normalize_pool_state(value))
            watchlist = watchlist[status.isin(allowed_states)].copy()
        return set(watchlist["bond_code"].dropna().astype(str))

    def load_active_bond_codes(self, conn, asof: str | pd.Timestamp) -> set[str]:
        return self.load_pool_bond_codes(conn, asof, DEFAULT_ACTIONABLE_POOL_STATES)

    def load_live_bars(
        self,
        conn,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        asof: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        start_ts = ensure_market_naive(start)
        end_ts = ensure_market_naive(end)
        if asof is not None:
            end_ts = min(end_ts, completed_bar_cutoff(asof, self.bar_minutes))

        sql = f"""
            SELECT * FROM {self.live_table}
            WHERE bar_start >= {sql_string(start_ts.strftime('%Y-%m-%d %H:%M:%S'))}
              AND bar_start <= {sql_string(end_ts.strftime('%Y-%m-%d %H:%M:%S'))}
            ORDER BY bar_start, asset_type, bond_code, stock_code
        """
        df = query_to_dataframe(conn, sql)
        if df.empty:
            return df

        df["bar_start"] = pd.to_datetime(df["bar_start"], errors="coerce")
        df = df[df["bar_start"].notna()].copy()
        df["slot_id"] = df["bar_start"].apply(lambda value: slot_id(value, self.bar_minutes))
        df = df[df["slot_id"].notna()].copy()
        df["slot_id"] = df["slot_id"].astype(int)
        df["trade_date"] = df["bar_start"].dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume", "amount", "premium_rate", "conversion_price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def load_conversion_history(self, conn, end: str | pd.Timestamp) -> pd.DataFrame:
        end_ts = ensure_market_naive(end)
        sql = f"""
            SELECT ts, bond_code, stock_code, stock_name, conversion_price
            FROM {self.watchlist_table}
            WHERE ts <= {sql_string(end_ts.strftime('%Y-%m-%d %H:%M:%S'))}
              AND conversion_price IS NOT NULL
            ORDER BY bond_code, ts
        """
        df = query_to_dataframe(conn, sql)
        if df.empty:
            return pd.DataFrame(columns=["snapshot_ts", "bond_code", "stock_code", "stock_name", "conversion_price"])

        df = df.rename(columns={"ts": "snapshot_ts"}).copy()
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
        df["conversion_price"] = pd.to_numeric(df["conversion_price"], errors="coerce")
        df["bond_code"] = df["bond_code"].astype(str)
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        df = df.dropna(subset=["snapshot_ts", "bond_code", "conversion_price"])
        return df.drop_duplicates(subset=["bond_code", "snapshot_ts"], keep="last").sort_values(["bond_code", "snapshot_ts"])

    def build_panel(
        self,
        conn,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        asof: str | pd.Timestamp | None = None,
        pool_states: set[str] | None = None,
    ) -> pd.DataFrame:
        bars = self.load_live_bars(conn, start, end, asof=asof)
        if bars.empty:
            return pd.DataFrame()

        active_bond_codes = self.load_pool_bond_codes(conn, end if asof is None else asof, pool_states)
        if active_bond_codes:
            bars = bars[bars["bond_code"].astype(str).isin(active_bond_codes)].copy()
        elif "pool_state" in bars.columns or "watchlist_status" in bars.columns:
            status_col = "pool_state" if "pool_state" in bars.columns else "watchlist_status"
            status = bars[status_col].apply(lambda value: normalize_pool_state(value))
            bars = bars[status.isin(pool_states or DEFAULT_ACTIONABLE_POOL_STATES)].copy()
        if bars.empty:
            return pd.DataFrame()

        cb = bars[bars["asset_type"].astype(str).str.upper() == self.schema["asset_type_values"]["cb"]].copy()
        stock = bars[bars["asset_type"].astype(str).str.upper() == self.schema["asset_type_values"]["stock"]].copy()
        if cb.empty:
            return pd.DataFrame()

        cb_cols = [
            "bar_start",
            "trade_date",
            "slot_id",
            "bond_code",
            "bond_name",
            "stock_code",
            "stock_name",
            "watchlist_status",
            "pool_state",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "premium_rate",
            "conversion_price",
            "quote_count",
        ]
        cb = cb[[col for col in cb_cols if col in cb.columns]].rename(
            columns={
                "open": "cb_open",
                "high": "cb_high",
                "low": "cb_low",
                "close": "cb_close",
                "volume": "cb_volume",
                "amount": "cb_amount",
                "premium_rate": "source_premium_rate",
                "conversion_price": "bar_conversion_price",
                "quote_count": "cb_quote_count",
            }
        )
        cb["bond_code"] = cb["bond_code"].astype(str)
        cb["stock_code"] = cb["stock_code"].astype(str).str.zfill(6)
        if "pool_state" not in cb.columns:
            cb["pool_state"] = cb.get("watchlist_status", "ACTIVE")
        cb["pool_state"] = cb["pool_state"].apply(lambda value: normalize_pool_state(value))

        if not stock.empty:
            stock_cols = [
                "bar_start",
                "bond_code",
                "stock_code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "quote_count",
            ]
            stock = stock[[col for col in stock_cols if col in stock.columns]].rename(
                columns={
                    "open": "stock_open",
                    "high": "stock_high",
                    "low": "stock_low",
                    "close": "stock_close",
                    "volume": "stock_volume",
                    "amount": "stock_amount",
                    "quote_count": "stock_quote_count",
                }
            )
            stock["bond_code"] = stock["bond_code"].astype(str)
            stock["stock_code"] = stock["stock_code"].astype(str).str.zfill(6)
            panel = cb.merge(stock, on=["bar_start", "bond_code", "stock_code"], how="left")
        else:
            panel = cb.copy()
            for col in ["stock_open", "stock_high", "stock_low", "stock_close", "stock_volume", "stock_amount", "stock_quote_count"]:
                panel[col] = np.nan

        end_ts = ensure_market_naive(end if asof is None else asof)
        conversion_history = self.load_conversion_history(conn, end_ts)
        panel = point_in_time_asof_join(
            panel.sort_values(["bond_code", "bar_start"]),
            conversion_history,
            by="bond_code",
            left_on="bar_start",
            right_on="snapshot_ts",
            columns=["conversion_price", "stock_code", "stock_name"],
        )

        if "stock_code_y" in panel.columns:
            panel["stock_code"] = panel["stock_code_x"].fillna(panel["stock_code_y"])
            panel = panel.drop(columns=[col for col in ["stock_code_x", "stock_code_y"] if col in panel.columns])
        if "stock_name_y" in panel.columns:
            panel["stock_name"] = panel["stock_name_x"].fillna(panel["stock_name_y"])
            panel = panel.drop(columns=[col for col in ["stock_name_x", "stock_name_y"] if col in panel.columns])
        if "bond_code_y" in panel.columns:
            panel["bond_code"] = panel["bond_code_x"].fillna(panel["bond_code_y"])
            panel = panel.drop(columns=[col for col in ["bond_code_x", "bond_code_y"] if col in panel.columns])

        panel["conversion_price"] = pd.to_numeric(panel.get("conversion_price"), errors="coerce")
        panel["bar_conversion_price"] = pd.to_numeric(panel.get("bar_conversion_price"), errors="coerce")
        panel["conversion_price"] = panel["conversion_price"].combine_first(panel["bar_conversion_price"])
        panel["conversion_price_source"] = np.where(panel["conversion_price"].notna(), "point_in_time_snapshot_or_bar", "missing")
        panel["pool_state"] = panel.get("pool_state", "ACTIVE")
        panel["pool_state"] = panel["pool_state"].apply(lambda value: normalize_pool_state(value))
        panel["session_bar_role"] = panel["bar_start"].apply(lambda value: session_bar_role(value, self.bar_minutes))

        panel["amount_used"] = pd.to_numeric(panel["cb_amount"], errors="coerce")
        amount_proxy = pd.to_numeric(panel["cb_volume"], errors="coerce") * pd.to_numeric(panel["cb_close"], errors="coerce")
        panel["amount_source"] = np.where(panel["amount_used"].notna(), "reported_amount", "proxy_volume_close")
        panel["amount_used"] = panel["amount_used"].combine_first(amount_proxy)
        panel.loc[panel["amount_used"].isna(), "amount_source"] = "missing"

        panel["stock_amount_used"] = pd.to_numeric(panel["stock_amount"], errors="coerce")
        stock_proxy = pd.to_numeric(panel["stock_volume"], errors="coerce") * pd.to_numeric(panel["stock_close"], errors="coerce")
        panel["stock_amount_used"] = panel["stock_amount_used"].combine_first(stock_proxy)
        panel = panel.sort_values(["bond_code", "bar_start"]).reset_index(drop=True)
        return panel

    def load_daily_amounts(self, conn, bond_codes: list[str], end: str | pd.Timestamp, lookback_days: int) -> pd.DataFrame:
        end_ts = ensure_market_naive(end).normalize()
        start_ts = end_ts - pd.Timedelta(days=max(lookback_days * 3, 30))
        rows = []
        for bond_code in bond_codes:
            table_name = f"cb_bar_1d_{bond_code}"
            try:
                result = conn.query(
                    f"""
                    SELECT ts, amount
                    FROM {table_name}
                    WHERE ts < {sql_string(end_ts.strftime('%Y-%m-%d %H:%M:%S'))}
                      AND ts >= {sql_string(start_ts.strftime('%Y-%m-%d %H:%M:%S'))}
                    ORDER BY ts DESC
                    LIMIT {int(lookback_days)}
                    """
                )
                amounts = [float(row[1]) for row in result if row[1] is not None]
                rows.append({"bond_code": bond_code, "adv20_amount": float(np.nanmean(amounts)) if amounts else np.nan})
            except Exception:
                rows.append({"bond_code": bond_code, "adv20_amount": np.nan})
        return pd.DataFrame(rows)


def factor_row_timestamp(bar_start: pd.Timestamp, bond_code: str, factor_version: str) -> pd.Timestamp:
    offset_ms = zlib.crc32(f"{bond_code}:{factor_version}".encode("utf-8")) % 300000
    return pd.Timestamp(bar_start).tz_localize(None) + pd.Timedelta(milliseconds=offset_ms)


def alert_row_timestamp(bar_start: pd.Timestamp, bond_code: str, signal_type: str, factor_version: str) -> pd.Timestamp:
    offset_ms = zlib.crc32(f"{bond_code}:{signal_type}:{factor_version}".encode("utf-8")) % 300000
    return pd.Timestamp(bar_start).tz_localize(None) + pd.Timedelta(milliseconds=offset_ms)


FACTOR_TABLE_COLUMNS = [
    "ts TIMESTAMP",
    "bar_start TIMESTAMP",
    "bond_code NCHAR(16)",
    "stock_code NCHAR(16)",
    "pool_state NCHAR(32)",
    "factor_version NCHAR(16)",
    "is_actionable_bar BOOL",
    "session_bar_role NCHAR(32)",
    "parity DOUBLE",
    "premium_raw DOUBLE",
    "premium_slot_z DOUBLE",
    "factor_premium_mr DOUBLE",
    "premium_change_5m DOUBLE",
    "premium_change_10m DOUBLE",
    "premium_change_30m DOUBLE",
    "premium_change_5m_z DOUBLE",
    "premium_change_10m_z DOUBLE",
    "premium_change_30m_z DOUBLE",
    "premium_model_residual DOUBLE",
    "premium_model_residual_z DOUBLE",
    "premium_factor_source NCHAR(32)",
    "r_cb DOUBLE",
    "r_stock DOUBLE",
    "r_cb_market DOUBLE",
    "alpha DOUBLE",
    "beta DOUBLE",
    "gamma DOUBLE",
    "residual DOUBLE",
    "residual_ewm DOUBLE",
    "residual_z DOUBLE",
    "residual_z_change_1bar DOUBLE",
    "residual_z_change_2bar DOUBLE",
    "residual_z_slope_3bar DOUBLE",
    "residual_repair_started BOOL",
    "factor_residual_mr DOUBLE",
    "stock_mom_5m_raw DOUBLE",
    "stock_mom_10m_raw DOUBLE",
    "stock_mom_15m_raw DOUBLE",
    "stock_mom_5m_z DOUBLE",
    "stock_mom_10m_z DOUBLE",
    "stock_mom_15m_z DOUBLE",
    "stock_amount_burst_z DOUBLE",
    "stock_impulse_score DOUBLE",
    "stock_mom_30m DOUBLE",
    "stock_mom_30m_z DOUBLE",
    "stock_mom_60m DOUBLE",
    "stock_mom_60m_z DOUBLE",
    "stock_mom_1d DOUBLE",
    "stock_mom_1d_z DOUBLE",
    "stock_mom_5d DOUBLE",
    "stock_mom_5d_z DOUBLE",
    "stock_momentum_score DOUBLE",
    "cb_mom_5m DOUBLE",
    "cb_mom_10m DOUBLE",
    "cb_mom_15m DOUBLE",
    "cb_mom_30m DOUBLE",
    "cb_mom_60m DOUBLE",
    "cb_mom_5m_z DOUBLE",
    "cb_mom_10m_z DOUBLE",
    "cb_mom_15m_z DOUBLE",
    "cb_mom_30m_z DOUBLE",
    "cb_intraday_vwap DOUBLE",
    "stock_intraday_vwap DOUBLE",
    "cb_vwap_dev DOUBLE",
    "stock_vwap_dev DOUBLE",
    "vwap_gap DOUBLE",
    "stock_rv_short DOUBLE",
    "stock_rv_long DOUBLE",
    "cb_rv_short DOUBLE",
    "cb_rv_long DOUBLE",
    "residual_rv_short DOUBLE",
    "residual_rv_long DOUBLE",
    "residual_rv_long_fallback DOUBLE",
    "residual_vol_ratio DOUBLE",
    "residual_vol_ratio_z DOUBLE",
    "residual_vol_ratio_source NCHAR(32)",
    "residual_vol_ratio_fallback DOUBLE",
    "residual_vol_ratio_fallback_z DOUBLE",
    "residual_vol_ratio_fallback_source NCHAR(32)",
    "fallback_vol_available BOOL",
    "factor_vol_mr DOUBLE",
    "factor_vol_mr_fallback DOUBLE",
    "empirical_delta_proxy DOUBLE",
    "elasticity_proxy DOUBLE",
    "empirical_gamma_proxy DOUBLE",
    "convexity_asymmetry_proxy DOUBLE",
    "factor_vol_response_gap DOUBLE",
    "cb_flow_30m DOUBLE",
    "stock_flow_30m DOUBLE",
    "cb_flow_z DOUBLE",
    "stock_flow_z DOUBLE",
    "flow_confirmation DOUBLE",
    "flow_lead DOUBLE",
    "cb_market_mom_30m DOUBLE",
    "cb_market_breadth_30m DOUBLE",
    "cb_market_breadth_60m DOUBLE",
    "cb_market_amount_burst DOUBLE",
    "stock_market_breadth_30m DOUBLE",
    "market_regime_intraday NCHAR(32)",
    "adv20_amount DOUBLE",
    "relative_amount DOUBLE",
    "active_bar_ratio DOUBLE",
    "amihud DOUBLE",
    "high_low_range_proxy DOUBLE",
    "high_low_range_proxy_z DOUBLE",
    "high_low_range_proxy_z_source NCHAR(32)",
    "volatility_gate_pass BOOL",
    "missing_vol_gate BOOL",
    "missing_range_gate BOOL",
    "noise_penalty DOUBLE",
    "data_quality_pass BOOL",
    "liquidity_pass BOOL",
    "trend_gate BOOL",
    "factor_coverage DOUBLE",
    "factor_coverage_pass BOOL",
    "missing_cb_bar BOOL",
    "missing_stock_bar BOOL",
    "stale_price_flag BOOL",
    "insufficient_history_flag BOOL",
    "fallback_standardization_flag BOOL",
    "setup_score DOUBLE",
    "trigger_score DOUBLE",
    "exit_score DOUBLE",
    "exit_state NCHAR(32)",
    "exit_type NCHAR(32)",
    "signal_score DOUBLE",
    "signal_type NCHAR(32)",
    "watch_signal_type NCHAR(32)",
    "watch_missing_to_action NCHAR(512)",
    "signal_state NCHAR(32)",
    "alert_state NCHAR(32)",
    "action_reason_text NCHAR(512)",
    "risk_hint_text NCHAR(512)",
    "cooldown_suppressed BOOL",
    "suppression_reason NCHAR(256)",
    "action_blocked_by_time BOOL",
    "no_new_action_after_suppressed BOOL",
    "alert_expire_time TIMESTAMP",
    "last_action_reason NCHAR(512)",
    "amount_source NCHAR(32)",
    "calculated_at TIMESTAMP",
]


ALERT_TABLE_COLUMNS = [
    "ts TIMESTAMP",
    "bar_start TIMESTAMP",
    "signal_time TIMESTAMP",
    "bond_code NCHAR(16)",
    "stock_code NCHAR(16)",
    "pool_state NCHAR(32)",
    "alert_state NCHAR(32)",
    "signal_type NCHAR(32)",
    "watch_signal_type NCHAR(32)",
    "setup_score DOUBLE",
    "trigger_score DOUBLE",
    "exit_score DOUBLE",
    "exit_type NCHAR(32)",
    "residual_z DOUBLE",
    "residual_z_change_1bar DOUBLE",
    "residual_z_change_2bar DOUBLE",
    "stock_mom_30m_z DOUBLE",
    "stock_mom_60m_z DOUBLE",
    "stock_impulse_score DOUBLE",
    "cb_mom_10m_z DOUBLE",
    "cb_mom_15m_z DOUBLE",
    "premium_slot_z DOUBLE",
    "premium_change_10m_z DOUBLE",
    "premium_change_30m_z DOUBLE",
    "relative_amount DOUBLE",
    "cb_flow_z DOUBLE",
    "flow_confirmation DOUBLE",
    "vwap_gap DOUBLE",
    "cb_vwap_dev DOUBLE",
    "stock_vwap_dev DOUBLE",
    "market_regime_intraday NCHAR(32)",
    "residual_vol_ratio_z DOUBLE",
    "high_low_range_proxy_z DOUBLE",
    "volatility_gate_pass BOOL",
    "factor_coverage DOUBLE",
    "is_actionable_bar BOOL",
    "action_reason_text NCHAR(512)",
    "watch_missing_to_action NCHAR(512)",
    "risk_hint_text NCHAR(512)",
    "cooldown_suppressed BOOL",
    "suppression_reason NCHAR(256)",
    "no_new_action_after_suppressed BOOL",
    "alert_expire_time TIMESTAMP",
    "calculated_at TIMESTAMP",
    "alert_inserted_at TIMESTAMP",
    "first_seen_time TIMESTAMP",
    "effective_asof TIMESTAMP",
    "factor_version NCHAR(16)",
]


JOB_RUN_TABLE_COLUMNS = [
    "ts TIMESTAMP",
    "run_id NCHAR(64)",
    "job_name NCHAR(64)",
    "mode NCHAR(32)",
    "factor_version NCHAR(16)",
    "requested_asof TIMESTAMP",
    "effective_asof TIMESTAMP",
    "latest_cb_bar TIMESTAMP",
    "latest_stock_bar TIMESTAMP",
    "latest_factor_bar TIMESTAMP",
    "latest_alert_bar TIMESTAMP",
    "started_at TIMESTAMP",
    "finished_at TIMESTAMP",
    "status NCHAR(32)",
    "processed_bars INT",
    "processed_bonds INT",
    "factor_rows_upserted INT",
    "alert_rows_upserted INT",
    "skipped_bonds INT",
    "error_count INT",
    "error_message NCHAR(1024)",
    "log_path NCHAR(512)",
    "pool_scope NCHAR(128)",
    "history_coverage_min_days INT",
    "live_backfill_diff_status NCHAR(32)",
]


LIVE_BACKFILL_DIFF_COLUMNS = [
    "ts TIMESTAMP",
    "diff_date NCHAR(16)",
    "factor_version NCHAR(16)",
    "bond_code NCHAR(16)",
    "bar_start TIMESTAMP",
    "field_name NCHAR(64)",
    "live_value NCHAR(128)",
    "backfill_value NCHAR(128)",
    "abs_diff DOUBLE",
    "diff_type NCHAR(64)",
    "severity NCHAR(16)",
    "created_at TIMESTAMP",
]


def _column_name(definition: str) -> str:
    return definition.split()[0]


def _column_type(definition: str) -> str:
    return definition.split(None, 1)[1].upper()


def _create_table_with_columns(conn, table_name: str, columns: list[str]) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})")
    for definition in columns[1:]:
        try_execute(conn, f"ALTER TABLE {table_name} ADD COLUMN {definition}")


def create_factor_table(conn, table_name: str = "cb_intraday_factor_snapshot") -> None:
    _create_table_with_columns(conn, table_name, FACTOR_TABLE_COLUMNS)


def create_alert_table(conn, table_name: str = "cb_intraday_alerts") -> None:
    _create_table_with_columns(conn, table_name, ALERT_TABLE_COLUMNS)


def create_job_run_table(conn, table_name: str = "cb_intraday_job_runs") -> None:
    _create_table_with_columns(conn, table_name, JOB_RUN_TABLE_COLUMNS)


def create_live_backfill_diff_table(conn, table_name: str = "cb_intraday_live_backfill_diff") -> None:
    _create_table_with_columns(conn, table_name, LIVE_BACKFILL_DIFF_COLUMNS)


def _sql_value_for_type(value: Any, column_type: str) -> str:
    if "TIMESTAMP" in column_type:
        if value is None or pd.isna(value):
            return "NULL"
        ts = pd.Timestamp(value).tz_localize(None)
        return sql_string(ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    if "BOOL" in column_type:
        return sql_bool(value)
    if "NCHAR" in column_type or "VARCHAR" in column_type or "BINARY" in column_type:
        return sql_string(value)
    return sql_number(value)


def _insert_rows(conn, table_name: str, columns: list[str], rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    column_names = [_column_name(definition) for definition in columns]
    column_types = {name: _column_type(definition) for name, definition in zip(column_names, columns)}
    values = []
    for row in rows:
        values.append(
            "("
            + ", ".join(_sql_value_for_type(row.get(column), column_types[column]) for column in column_names)
            + ")"
        )
    conn.execute(
        f"""
        INSERT INTO {table_name} ({', '.join(column_names)}) VALUES
        """
        + " ".join(values)
    )
    return len(values)


def insert_factor_rows(conn, df: pd.DataFrame, table_name: str, factor_version: str) -> int:
    if df.empty:
        return 0
    create_factor_table(conn, table_name)
    now = pd.Timestamp.now().floor("s")
    rows = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        bar_start = pd.Timestamp(row_dict["bar_start"]).tz_localize(None)
        row_dict["ts"] = factor_row_timestamp(bar_start, str(row_dict["bond_code"]), factor_version)
        row_dict["bar_start"] = bar_start
        row_dict["factor_version"] = factor_version
        row_dict["pool_state"] = normalize_pool_state(row_dict.get("pool_state"))
        row_dict["session_bar_role"] = row_dict.get("session_bar_role") or session_bar_role(bar_start)
        row_dict["is_actionable_bar"] = bool(row_dict.get("is_actionable_bar", True))
        row_dict["calculated_at"] = now
        rows.append(row_dict)
    return _insert_rows(conn, table_name, FACTOR_TABLE_COLUMNS, rows)


def insert_alert_rows(conn, df: pd.DataFrame, table_name: str, factor_version: str) -> int:
    if df.empty:
        return 0
    create_alert_table(conn, table_name)
    now = pd.Timestamp.now().floor("s")
    rows = []
    alert_columns = {_column_name(definition) for definition in ALERT_TABLE_COLUMNS}
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        bar_start = pd.Timestamp(row_dict["bar_start"]).tz_localize(None)
        signal_type = str(row_dict.get("signal_type") or "NONE")
        out = {column: row_dict.get(column) for column in alert_columns}
        out["ts"] = alert_row_timestamp(bar_start, str(row_dict.get("bond_code")), signal_type, factor_version)
        out["bar_start"] = bar_start
        out["signal_time"] = row_dict.get("signal_time") or bar_start
        out["pool_state"] = normalize_pool_state(row_dict.get("pool_state"))
        out["factor_version"] = factor_version
        out["calculated_at"] = now
        out["alert_inserted_at"] = now
        out["first_seen_time"] = row_dict.get("first_seen_time") or now
        out["effective_asof"] = row_dict.get("effective_asof") or bar_start
        out["is_actionable_bar"] = bool(row_dict.get("is_actionable_bar", True))
        out["cooldown_suppressed"] = bool(row_dict.get("cooldown_suppressed", False))
        out["no_new_action_after_suppressed"] = bool(row_dict.get("no_new_action_after_suppressed", False))
        rows.append(out)
    return _insert_rows(conn, table_name, ALERT_TABLE_COLUMNS, rows)


def insert_job_run(conn, row: dict[str, Any], table_name: str = "cb_intraday_job_runs") -> int:
    create_job_run_table(conn, table_name)
    out = dict(row)
    finished_at = out.get("finished_at")
    out["ts"] = pd.Timestamp(finished_at if finished_at is not None and not pd.isna(finished_at) else pd.Timestamp.now()).tz_localize(None)
    return _insert_rows(conn, table_name, JOB_RUN_TABLE_COLUMNS, [out])


def insert_live_backfill_diff_rows(conn, rows: list[dict[str, Any]], table_name: str = "cb_intraday_live_backfill_diff") -> int:
    if not rows:
        return 0
    create_live_backfill_diff_table(conn, table_name)
    now = pd.Timestamp.now().floor("s")
    out_rows = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["ts"] = now + pd.Timedelta(milliseconds=idx)
        out["created_at"] = now
        out_rows.append(out)
    return _insert_rows(conn, table_name, LIVE_BACKFILL_DIFF_COLUMNS, out_rows)


def get_connection():
    return get_conn(use_db=True)
