from __future__ import annotations

import logging
import os
import time
import zlib
from typing import Any

import akshare as ak
import pandas as pd
import requests

from cb_universe import to_ak_bond_symbol
from config import SOURCE, USE_PROXY_FOR_MARKET_DATA
from intraday_factors.data_source import (
    insert_live_backfill_diff_rows,
    query_to_dataframe,
    sql_bool,
    sql_number,
    sql_string,
    try_execute,
)
from intraday_factors.pool_state import (
    ACTIVE,
    ACTIVE_PENDING_HISTORY,
    BLACKLIST_MANUAL,
    DEFAULT_ACTIONABLE_POOL_STATES,
    OBSERVE,
    SHADOW_RESEARCH,
    normalize_pool_state,
    parse_pool_scope,
)
from intraday_factors.trading_session import expected_slots_per_day, is_session_bar, session_bar_role


LOGGER = logging.getLogger(__name__)
LIVE_TABLE = "cb_watchlist_bar_5m_live"
WATCHLIST_TABLE = "cb_watchlist_daily"
DEFAULT_HISTORY_TRADING_DAYS = 45
DEFAULT_BAR_MINUTES = 5
DEFAULT_HISTORY_RETRIES = int(os.getenv("CB_HISTORY_RETRIES", "4"))
DEFAULT_HISTORY_TIMEOUT_SECONDS = float(os.getenv("CB_HISTORY_TIMEOUT_SECONDS", "20"))
DEFAULT_HISTORY_BACKOFF_SECONDS = float(os.getenv("CB_HISTORY_BACKOFF_SECONDS", "2"))
DEFAULT_HISTORY_EXPECTED_BARS_PER_DAY = max(1, expected_slots_per_day(DEFAULT_BAR_MINUTES) - 2)
EASTMONEY_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://quote.eastmoney.com/",
}


LIVE_TABLE_COLUMNS = [
    "ts TIMESTAMP",
    "bar_start TIMESTAMP",
    "trade_date NCHAR(16)",
    "asset_type NCHAR(16)",
    "watchlist_status NCHAR(16)",
    "pool_state NCHAR(32)",
    "instrument_code NCHAR(16)",
    "instrument_name NCHAR(32)",
    "bond_code NCHAR(16)",
    "bond_name NCHAR(32)",
    "stock_code NCHAR(16)",
    "stock_name NCHAR(32)",
    "market NCHAR(8)",
    "open DOUBLE",
    "high DOUBLE",
    "low DOUBLE",
    "close DOUBLE",
    "change_percent DOUBLE",
    "premium_rate DOUBLE",
    "ytm DOUBLE",
    "conversion_value DOUBLE",
    "conversion_price DOUBLE",
    "volume DOUBLE",
    "amount DOUBLE",
    "turnover_rate DOUBLE",
    "ytm_available BOOL",
    "ytm_source NCHAR(32)",
    "turnover_available BOOL",
    "turnover_source NCHAR(32)",
    "premium_source NCHAR(32)",
    "volatility DOUBLE",
    "quote_count INT",
    "source NCHAR(32)",
]


def disable_proxy_if_needed() -> None:
    if USE_PROXY_FOR_MARKET_DATA:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)


def ensure_live_table(conn, table_name: str = LIVE_TABLE) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(LIVE_TABLE_COLUMNS)})")
    for definition in LIVE_TABLE_COLUMNS[1:]:
        try_execute(conn, f"ALTER TABLE {table_name} ADD COLUMN {definition}")


def stock_market_from_code(stock_code: str) -> str | None:
    code = str(stock_code).strip().zfill(6)
    if code.startswith(("60", "68", "90")):
        return "SH"
    if code.startswith(("00", "30", "20")):
        return "SZ"
    return None


def live_row_timestamp(bar_start: pd.Timestamp, asset_type: str, bond_code: str, instrument_code: str) -> pd.Timestamp:
    offset_ms = zlib.crc32(f"{asset_type}:{bond_code}:{instrument_code}".encode("utf-8")) % 300000
    return pd.Timestamp(bar_start).tz_localize(None) + pd.Timedelta(milliseconds=offset_ms)


def normalize_minute_history(raw: pd.DataFrame, bar_minutes: int = DEFAULT_BAR_MINUTES) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["bar_start", "open", "high", "low", "close", "volume", "amount", "change_percent", "turnover_rate"])
    rename = {
        "时间": "bar_start",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "change_percent",
        "换手率": "turnover_rate",
    }
    df = raw.rename(columns=rename).copy()
    required = ["bar_start", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"minute history missing columns {missing}; raw columns={list(raw.columns)}")
    for col in ["open", "high", "low", "close", "volume", "amount", "change_percent", "turnover_rate"]:
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bar_start"] = pd.to_datetime(df["bar_start"], errors="coerce")
    df = df.dropna(subset=["bar_start", "open", "high", "low", "close"])
    df["bar_start"] = df["bar_start"].dt.tz_localize(None)
    df = df[df["bar_start"].apply(lambda value: is_session_bar(value, bar_minutes))].copy()
    return df.sort_values("bar_start").drop_duplicates(subset=["bar_start"], keep="last").reset_index(drop=True)


def history_retry_sleep(attempt: int) -> None:
    time.sleep(DEFAULT_HISTORY_BACKOFF_SECONDS * attempt)


def expected_history_slots_per_day() -> int:
    # Historical Eastmoney 5m bars are completed bars labelled 09:35..15:00,
    # without the 09:30 open marker or 13:00 afternoon boundary marker.
    return DEFAULT_HISTORY_EXPECTED_BARS_PER_DAY


def fetch_cb_5m_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp, lmt: int) -> pd.DataFrame:
    disable_proxy_if_needed()
    symbol = str(symbol)
    market_type = {"sh": "1", "sz": "0"}
    if symbol[:2] not in market_type:
        raise ValueError(f"unsupported convertible bond symbol: {symbol}")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": f"{market_type[symbol[:2]]}.{symbol[2:]}",
        "klt": "5",
        "fqt": "0",
        "lmt": str(lmt),
        "end": end.strftime("%Y%m%d"),
        "iscca": "1",
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "forcect": "1",
    }
    last_error = None
    session = requests.Session()
    session.trust_env = False
    for attempt in range(1, DEFAULT_HISTORY_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=DEFAULT_HISTORY_TIMEOUT_SECONDS, headers=EASTMONEY_HEADERS)
            response.raise_for_status()
            data = response.json().get("data") or {}
            klines = data.get("klines") or []
            if not klines:
                raise RuntimeError("empty kline payload")
            raw = pd.DataFrame([item.split(",") for item in klines])
            raw.columns = ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
            df = normalize_minute_history(raw)
            return df[(df["bar_start"] >= start) & (df["bar_start"] <= end)].reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("custom bond 5m history fetch failed symbol=%s attempt=%s/%s error=%s", symbol, attempt, DEFAULT_HISTORY_RETRIES, exc)
            if attempt < DEFAULT_HISTORY_RETRIES:
                history_retry_sleep(attempt)

    for attempt in range(1, DEFAULT_HISTORY_RETRIES + 1):
        try:
            raw = ak.bond_zh_hs_cov_min(
                symbol=symbol,
                period="5",
                adjust="",
                start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            df = normalize_minute_history(raw)
            return df[(df["bar_start"] >= start) & (df["bar_start"] <= end)].reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("akshare bond 5m history fetch failed symbol=%s attempt=%s/%s error=%s", symbol, attempt, DEFAULT_HISTORY_RETRIES, exc)
            if attempt < DEFAULT_HISTORY_RETRIES:
                history_retry_sleep(attempt)
    raise RuntimeError(f"bond 5m history fetch failed symbol={symbol}: {last_error}")


def fetch_stock_5m_history(stock_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    disable_proxy_if_needed()
    last_error = None
    for attempt in range(1, DEFAULT_HISTORY_RETRIES + 1):
        try:
            raw = ak.stock_zh_a_hist_min_em(
                symbol=str(stock_code).zfill(6),
                start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                period="5",
                adjust="",
            )
            df = normalize_minute_history(raw)
            return df[(df["bar_start"] >= start) & (df["bar_start"] <= end)].reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("stock 5m history fetch failed stock_code=%s attempt=%s/%s error=%s", stock_code, attempt, DEFAULT_HISTORY_RETRIES, exc)
            if attempt < DEFAULT_HISTORY_RETRIES:
                history_retry_sleep(attempt)
    raise RuntimeError(f"stock 5m history fetch failed stock_code={stock_code}: {last_error}")


def latest_watchlist_snapshot(conn, asof_date: str | pd.Timestamp, table_name: str = WATCHLIST_TABLE) -> pd.DataFrame:
    asof_ts = pd.Timestamp(asof_date).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    snap = query_to_dataframe(
        conn,
        f"""
        SELECT LAST(snapshot_id) AS snapshot_id
        FROM {table_name}
        WHERE ts <= {sql_string(asof_ts.strftime('%Y-%m-%d %H:%M:%S'))}
        """,
    )
    if snap.empty or pd.isna(snap.loc[0, "snapshot_id"]):
        return pd.DataFrame()
    snapshot_id = str(snap.loc[0, "snapshot_id"])
    df = query_to_dataframe(conn, f"SELECT * FROM {table_name} WHERE snapshot_id = {sql_string(snapshot_id)}")
    if df.empty:
        return df
    if "pool_state" not in df.columns:
        df["pool_state"] = df.get("watchlist_status", ACTIVE)
    df["pool_state"] = df["pool_state"].apply(lambda value: normalize_pool_state(value))
    df["watchlist_status"] = df["pool_state"]
    df["bond_code"] = df["bond_code"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df


def filter_pool(df: pd.DataFrame, pool: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    states = parse_pool_scope(pool)
    return df[df["pool_state"].isin(states)].copy()


def recent_trading_dates(conn, bond_code: str, asof_date: str | pd.Timestamp, trading_days: int, live_table: str = LIVE_TABLE) -> list[str]:
    asof_ts = pd.Timestamp(asof_date).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    dates: list[str] = []
    try:
        daily = query_to_dataframe(
            conn,
            f"""
            SELECT ts
            FROM cb_bar_1d_{bond_code}
            WHERE ts <= {sql_string(asof_ts.strftime('%Y-%m-%d %H:%M:%S'))}
            ORDER BY ts DESC
            LIMIT {int(trading_days)}
            """,
        )
        if not daily.empty:
            dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in daily["ts"].dropna()]
    except Exception:
        dates = []

    try:
        live = query_to_dataframe(
            conn,
            f"""
            SELECT trade_date
            FROM {live_table}
            WHERE bond_code = {sql_string(bond_code)}
              AND bar_start <= {sql_string(asof_ts.strftime('%Y-%m-%d %H:%M:%S'))}
            ORDER BY bar_start DESC
            LIMIT 20000
            """,
        )
        if not live.empty and "trade_date" in live.columns:
            dates.extend(str(value)[:10] for value in live["trade_date"].dropna())
    except Exception:
        pass

    unique = sorted({date for date in dates if date and date != "NaT"})
    return unique[-int(trading_days):]


def coverage_for_pair(
    conn,
    row: dict[str, Any] | pd.Series,
    trading_days: int,
    asof_date: str | pd.Timestamp,
    live_table: str = LIVE_TABLE,
) -> dict[str, Any]:
    data = dict(row)
    bond_code = str(data.get("bond_code"))
    stock_code = str(data.get("stock_code")).zfill(6)
    dates = recent_trading_dates(conn, bond_code, asof_date, trading_days, live_table=live_table)
    expected_per_day = expected_history_slots_per_day()
    expected_bars = len(dates) * expected_per_day
    report = {
        "bond_code": bond_code,
        "stock_code": stock_code,
        "pool_state": normalize_pool_state(data.get("pool_state", data.get("watchlist_status", ACTIVE))),
        "history_days_available": 0,
        "expected_trading_days": int(trading_days),
        "local_trading_days": len(dates),
        "expected_bars": expected_bars,
        "cb_bars": 0,
        "stock_bars": 0,
        "missing_cb_bars": expected_bars,
        "missing_stock_bars": expected_bars,
        "ready_for_active": False,
        "failure_reason": None,
    }
    if not dates:
        report["failure_reason"] = "no_local_trading_dates"
        return report

    start = f"{dates[0]} 00:00:00"
    end = f"{dates[-1]} 23:59:59"
    try:
        bars = query_to_dataframe(
            conn,
            f"""
            SELECT asset_type, trade_date, bar_start
            FROM {live_table}
            WHERE bond_code = {sql_string(bond_code)}
              AND bar_start >= {sql_string(start)}
              AND bar_start <= {sql_string(end)}
            ORDER BY bar_start
            """,
        )
    except Exception as exc:
        report["failure_reason"] = f"coverage_query_failed: {exc}"
        return report

    if bars.empty:
        report["failure_reason"] = "no_5m_history"
        return report

    bars["trade_date"] = bars["trade_date"].astype(str).str[:10]
    bars = bars[bars["trade_date"].isin(dates)].copy()
    cb_counts = bars[bars["asset_type"].astype(str).str.upper() == "CB"].groupby("trade_date").size()
    stock_counts = bars[bars["asset_type"].astype(str).str.upper() == "STOCK"].groupby("trade_date").size()
    cb_total = int(cb_counts.sum())
    stock_total = int(stock_counts.sum())
    complete_days = 0
    for date in dates:
        if int(cb_counts.get(date, 0)) >= expected_per_day and int(stock_counts.get(date, 0)) >= expected_per_day:
            complete_days += 1
    report.update(
        {
            "history_days_available": complete_days,
            "cb_bars": cb_total,
            "stock_bars": stock_total,
            "missing_cb_bars": max(0, expected_bars - cb_total),
            "missing_stock_bars": max(0, expected_bars - stock_total),
            "ready_for_active": complete_days >= trading_days and cb_total >= expected_bars and stock_total >= expected_bars,
        }
    )
    if not report["ready_for_active"]:
        report["failure_reason"] = "insufficient_5m_history"
    return report


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if value is None or pd.isna(value):
        return default
    return value


def build_live_rows_from_history(
    watch_row: dict[str, Any],
    cb_hist: pd.DataFrame,
    stock_hist: pd.DataFrame,
) -> list[dict[str, Any]]:
    bond_code = str(watch_row["bond_code"])
    stock_code = str(watch_row["stock_code"]).zfill(6)
    bond_name = str(_row_value(watch_row, "bond_name", ""))
    stock_name = str(_row_value(watch_row, "stock_name", ""))
    pool_state = normalize_pool_state(watch_row.get("pool_state", watch_row.get("watchlist_status", ACTIVE)))
    conversion_price = pd.to_numeric(watch_row.get("conversion_price"), errors="coerce")
    stock_by_time = stock_hist.set_index("bar_start")["close"].to_dict() if not stock_hist.empty else {}
    rows: list[dict[str, Any]] = []

    for _, item in cb_hist.iterrows():
        bar_start = pd.Timestamp(item["bar_start"]).tz_localize(None)
        stock_close = stock_by_time.get(bar_start)
        conversion_value = None
        premium_rate = None
        premium_source = "unavailable"
        if pd.notna(conversion_price) and conversion_price > 0 and stock_close is not None and pd.notna(stock_close) and stock_close > 0:
            conversion_value = 100.0 * float(stock_close) / float(conversion_price)
            if conversion_value > 0:
                premium_rate = (float(item["close"]) / conversion_value - 1.0) * 100.0
                premium_source = "calculated"
        turnover = item.get("turnover_rate")
        rows.append(
            {
                "ts": live_row_timestamp(bar_start, "CB", bond_code, bond_code),
                "bar_start": bar_start,
                "trade_date": bar_start.strftime("%Y-%m-%d"),
                "asset_type": "CB",
                "watchlist_status": pool_state,
                "pool_state": pool_state,
                "instrument_code": bond_code,
                "instrument_name": bond_name,
                "bond_code": bond_code,
                "bond_name": bond_name,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "market": "SH" if to_ak_bond_symbol(bond_code).startswith("sh") else "SZ",
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "change_percent": item.get("change_percent"),
                "premium_rate": premium_rate,
                "ytm": None,
                "conversion_value": conversion_value,
                "conversion_price": conversion_price,
                "volume": item.get("volume"),
                "amount": item.get("amount"),
                "turnover_rate": turnover,
                "ytm_available": False,
                "ytm_source": "unavailable",
                "turnover_available": turnover is not None and not pd.isna(turnover),
                "turnover_source": "vendor" if turnover is not None and not pd.isna(turnover) else "unavailable",
                "premium_source": premium_source,
                "volatility": None,
                "quote_count": 1,
                "source": SOURCE + "_history_5m",
            }
        )

    for _, item in stock_hist.iterrows():
        bar_start = pd.Timestamp(item["bar_start"]).tz_localize(None)
        turnover = item.get("turnover_rate")
        rows.append(
            {
                "ts": live_row_timestamp(bar_start, "STOCK", bond_code, stock_code),
                "bar_start": bar_start,
                "trade_date": bar_start.strftime("%Y-%m-%d"),
                "asset_type": "STOCK",
                "watchlist_status": pool_state,
                "pool_state": pool_state,
                "instrument_code": stock_code,
                "instrument_name": stock_name,
                "bond_code": bond_code,
                "bond_name": bond_name,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "market": stock_market_from_code(stock_code),
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "change_percent": item.get("change_percent"),
                "premium_rate": None,
                "ytm": None,
                "conversion_value": None,
                "conversion_price": None,
                "volume": item.get("volume"),
                "amount": item.get("amount"),
                "turnover_rate": turnover,
                "ytm_available": False,
                "ytm_source": "unavailable",
                "turnover_available": turnover is not None and not pd.isna(turnover),
                "turnover_source": "vendor" if turnover is not None and not pd.isna(turnover) else "unavailable",
                "premium_source": "unavailable",
                "volatility": None,
                "quote_count": 1,
                "source": SOURCE + "_history_5m",
            }
        )
    return rows


def existing_live_timestamps(conn, live_table: str, start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    try:
        df = query_to_dataframe(
            conn,
            f"""
            SELECT ts
            FROM {live_table}
            WHERE ts >= {sql_string(start.strftime('%Y-%m-%d %H:%M:%S'))}
              AND ts <= {sql_string(end.strftime('%Y-%m-%d %H:%M:%S'))}
            ORDER BY ts
            """,
        )
    except Exception:
        return set()
    if df.empty or "ts" not in df.columns:
        return set()
    return {pd.Timestamp(value).tz_localize(None) for value in df["ts"].dropna()}


def insert_history_rows(conn, rows: list[dict[str, Any]], live_table: str = LIVE_TABLE, chunk_size: int = 500) -> int:
    if not rows:
        return 0
    ensure_live_table(conn, live_table)
    start = min(pd.Timestamp(row["ts"]).tz_localize(None) for row in rows)
    end = max(pd.Timestamp(row["ts"]).tz_localize(None) for row in rows)
    existing = existing_live_timestamps(conn, live_table, start, end)
    rows = [row for row in rows if pd.Timestamp(row["ts"]).tz_localize(None) not in existing]
    if not rows:
        return 0

    columns = [definition.split()[0] for definition in LIVE_TABLE_COLUMNS]
    inserted = 0
    for offset in range(0, len(rows), chunk_size):
        values = []
        for row in rows[offset:offset + chunk_size]:
            values.append(
                "("
                + ", ".join(
                    sql_bool(row.get(col))
                    if col.endswith("_available")
                    else sql_string(pd.Timestamp(row[col]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                    if col in {"ts", "bar_start"}
                    else sql_number(row.get(col))
                    if col in {"open", "high", "low", "close", "change_percent", "premium_rate", "ytm", "conversion_value", "conversion_price", "volume", "amount", "turnover_rate", "volatility"}
                    else str(int(row.get(col))) if col == "quote_count" and row.get(col) is not None and not pd.isna(row.get(col))
                    else sql_string(row.get(col))
                    for col in columns
                )
                + ")"
            )
        conn.execute(f"INSERT INTO {live_table} ({', '.join(columns)}) VALUES " + " ".join(values))
        inserted += len(values)
    return inserted


def backfill_history_for_row(
    conn,
    watch_row: dict[str, Any] | pd.Series,
    trading_days: int,
    asof_date: str | pd.Timestamp,
    live_table: str = LIVE_TABLE,
    sleep_seconds: float = 0.5,
) -> dict[str, Any]:
    row = dict(watch_row)
    bond_code = str(row["bond_code"])
    stock_code = str(row["stock_code"]).zfill(6)
    dates = recent_trading_dates(conn, bond_code, asof_date, trading_days, live_table=live_table)
    report = {
        "bond_code": bond_code,
        "stock_code": stock_code,
        "pool_state": normalize_pool_state(row.get("pool_state", row.get("watchlist_status", ACTIVE))),
        "inserted_rows": 0,
        "cb_fetched_rows": 0,
        "stock_fetched_rows": 0,
        "error": None,
    }
    if not dates:
        report["error"] = "no_local_trading_dates"
        report.update(coverage_for_pair(conn, row, trading_days, asof_date, live_table=live_table))
        return report

    start = pd.Timestamp(f"{dates[0]} 09:30:00")
    end = pd.Timestamp(f"{dates[-1]} 15:00:00")
    lmt = max(5000, trading_days * expected_slots_per_day(DEFAULT_BAR_MINUTES) + 200)
    cb_hist = pd.DataFrame()
    stock_hist = pd.DataFrame()
    errors = []

    cb_symbol = to_ak_bond_symbol(bond_code)
    LOGGER.info("5m history backfill start bond_code=%s stock_code=%s trading_days=%s", bond_code, stock_code, trading_days)
    if cb_symbol is None:
        errors.append(f"cannot map bond_code to ak symbol: {bond_code}")
    else:
        try:
            cb_hist = fetch_cb_5m_history(cb_symbol, start, end, lmt=lmt)
        except Exception as exc:
            LOGGER.exception("CB 5m history fetch failed bond_code=%s symbol=%s", bond_code, cb_symbol)
            errors.append(str(exc))

    time.sleep(sleep_seconds)
    try:
        stock_hist = fetch_stock_5m_history(stock_code, start, end)
    except Exception as exc:
        LOGGER.exception("STOCK 5m history fetch failed bond_code=%s stock_code=%s", bond_code, stock_code)
        errors.append(str(exc))

    try:
        if not cb_hist.empty:
            cb_hist = cb_hist[cb_hist["bar_start"].dt.strftime("%Y-%m-%d").isin(dates)].copy()
        if not stock_hist.empty:
            stock_hist = stock_hist[stock_hist["bar_start"].dt.strftime("%Y-%m-%d").isin(dates)].copy()
        report["cb_fetched_rows"] = len(cb_hist)
        report["stock_fetched_rows"] = len(stock_hist)
        if not cb_hist.empty or not stock_hist.empty:
            rows = build_live_rows_from_history(row, cb_hist, stock_hist)
            report["inserted_rows"] = insert_history_rows(conn, rows, live_table=live_table)
        if errors:
            report["error"] = "; ".join(errors)[:1000]
    except Exception as exc:
        LOGGER.exception("5m history insert failed bond_code=%s stock_code=%s", bond_code, stock_code)
        report["error"] = str(exc)

    coverage = coverage_for_pair(conn, row, trading_days, asof_date, live_table=live_table)
    report.update(coverage)
    LOGGER.info(
        "5m history backfill done bond_code=%s stock_code=%s inserted=%s cb_rows=%s stock_rows=%s ready=%s missing_cb=%s missing_stock=%s error=%s",
        bond_code,
        stock_code,
        report.get("inserted_rows"),
        report.get("cb_fetched_rows"),
        report.get("stock_fetched_rows"),
        report.get("ready_for_active"),
        report.get("missing_cb_bars"),
        report.get("missing_stock_bars"),
        report.get("error"),
    )
    return report


def backfill_pool_history(
    conn,
    pool: str,
    trading_days: int,
    asof_date: str | pd.Timestamp,
    live_table: str = LIVE_TABLE,
    sleep_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    watchlist = filter_pool(latest_watchlist_snapshot(conn, asof_date), pool)
    reports = []
    for _, row in watchlist.iterrows():
        reports.append(
            backfill_history_for_row(
                conn,
                row,
                trading_days=trading_days,
                asof_date=asof_date,
                live_table=live_table,
                sleep_seconds=sleep_seconds,
            )
        )
    return reports


def prepare_active_history(
    conn,
    bond_code: str,
    trading_days: int,
    asof_date: str | pd.Timestamp,
    live_table: str = LIVE_TABLE,
    sleep_seconds: float = 0.5,
) -> dict[str, Any]:
    watchlist = latest_watchlist_snapshot(conn, asof_date)
    row = watchlist[watchlist["bond_code"].astype(str) == str(bond_code)]
    if row.empty:
        return {
            "bond_code": str(bond_code),
            "ready_for_active": False,
            "failure_reason": "bond_not_in_latest_watchlist",
        }
    return backfill_history_for_row(conn, row.iloc[0], trading_days, asof_date, live_table=live_table, sleep_seconds=sleep_seconds)


def readiness_check(
    conn,
    check_date: str | pd.Timestamp,
    factor_version: str,
    trading_days: int = DEFAULT_HISTORY_TRADING_DAYS,
    live_table: str = LIVE_TABLE,
) -> tuple[dict[str, Any], int]:
    watchlist = latest_watchlist_snapshot(conn, check_date)
    if watchlist.empty:
        return {"date": str(check_date)[:10], "ready_for_open": False, "failure_reason": "no_watchlist_snapshot"}, 2

    counts = watchlist["pool_state"].value_counts().to_dict()
    active = watchlist[watchlist["pool_state"].isin(DEFAULT_ACTIONABLE_POOL_STATES)].copy()
    observe = watchlist[watchlist["pool_state"] == OBSERVE].copy()
    shadow = watchlist[watchlist["pool_state"] == SHADOW_RESEARCH].copy()
    pending = watchlist[watchlist["pool_state"] == ACTIVE_PENDING_HISTORY].copy()
    coverage = [coverage_for_pair(conn, row, trading_days, check_date, live_table=live_table) for _, row in active.iterrows()]

    active_failures = []
    for report in coverage:
        row = active[active["bond_code"].astype(str) == report["bond_code"]].iloc[0]
        missing_mapping = pd.isna(row.get("stock_code")) or str(row.get("stock_code")).strip() == ""
        missing_conversion = pd.isna(row.get("conversion_price"))
        unable_premium = missing_mapping or missing_conversion
        report["stock_mapping_exists"] = not missing_mapping
        report["conversion_price_exists"] = not missing_conversion
        report["premium_calculable"] = not unable_premium
        if not report["ready_for_active"] or missing_mapping or missing_conversion:
            active_failures.append(report["bond_code"])

    latest_daily_snapshot = None
    try:
        daily = query_to_dataframe(conn, "SELECT MAX(ts) AS ts FROM cb_screening_daily")
        if not daily.empty and pd.notna(daily.loc[0, "ts"]):
            latest_daily_snapshot = pd.Timestamp(daily.loc[0, "ts"]).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    output = {
        "date": pd.Timestamp(check_date).strftime("%Y-%m-%d"),
        "factor_version": factor_version,
        "active_count": int(len(active)),
        "observe_count": int(len(observe)),
        "shadow_research_count": int(len(shadow)),
        "pool_counts": {str(key): int(value) for key, value in counts.items()},
        "active_history_coverage": coverage,
        "observe_tracking_days": [
            {
                "bond_code": str(row.get("bond_code")),
                "stock_code": str(row.get("stock_code")).zfill(6),
                "candidate_since": None if pd.isna(row.get("candidate_since")) else pd.Timestamp(row.get("candidate_since")).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for _, row in observe.iterrows()
        ],
        "active_pending_history_count": int(len(pending)),
        "latest_daily_snapshot": latest_daily_snapshot,
        "latest_financial_snapshot": latest_daily_snapshot,
        "optional_ytm_turnover_missing_blocks_open": False,
        "active_failures": active_failures,
        "ready_for_open": len(active_failures) == 0 and len(pending) == 0,
    }
    return output, 0 if output["ready_for_open"] else 2


def summarize_latest_diff_status(conn, diff_table: str = "cb_intraday_live_backfill_diff") -> str | None:
    try:
        df = query_to_dataframe(conn, f"SELECT severity FROM {diff_table} ORDER BY ts DESC LIMIT 200")
    except Exception:
        return None
    if df.empty:
        return None
    severities = set(df["severity"].dropna().astype(str))
    if "ERROR" in severities:
        return "ERROR"
    if "WARN" in severities:
        return "WARN"
    return "OK"


def diff_live_backfill(
    conn,
    diff_date: str | pd.Timestamp,
    factor_version: str,
    config: dict[str, Any],
    schema: dict[str, Any],
    write: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from intraday_factors.engine import FactorEngine

    date = pd.Timestamp(diff_date).strftime("%Y-%m-%d")
    start = f"{date} 09:30:00"
    end = f"{date} 15:00:00"
    engine = FactorEngine(config, schema)
    recomputed = engine.compute_from_db(start, end, write=False)
    recomputed = recomputed[recomputed["bar_start"].dt.strftime("%Y-%m-%d") == date].copy() if not recomputed.empty else recomputed

    factor_table = schema["tables"]["factor_snapshot"]
    alert_table = schema["tables"]["alert_table"]
    live_factor = query_to_dataframe(
        conn,
        f"""
        SELECT *
        FROM {factor_table}
        WHERE factor_version = {sql_string(factor_version)}
          AND bar_start >= {sql_string(start)}
          AND bar_start <= {sql_string(end)}
        """,
    )
    live_alert = query_to_dataframe(
        conn,
        f"""
        SELECT *
        FROM {alert_table}
        WHERE factor_version = {sql_string(factor_version)}
          AND bar_start >= {sql_string(start)}
          AND bar_start <= {sql_string(end)}
        """,
    )
    for df in [live_factor, live_alert, recomputed]:
        if not df.empty and "bar_start" in df.columns:
            df["bar_start"] = pd.to_datetime(df["bar_start"], errors="coerce")

    diff_rows: list[dict[str, Any]] = []
    compare_fields = [
        "signal_type",
        "alert_state",
        "setup_score",
        "trigger_score",
        "residual_z",
        "premium_slot_z",
    ]
    if not recomputed.empty and not live_alert.empty:
        live_keys = live_alert.set_index(["bar_start", "bond_code"])
        back_keys = recomputed.set_index(["bar_start", "bond_code"])
        all_keys = sorted(set(live_keys.index) | set(back_keys.index))
        for key in all_keys:
            live_row = live_keys.loc[key].iloc[0] if key in live_keys.index and isinstance(live_keys.loc[key], pd.DataFrame) else (live_keys.loc[key] if key in live_keys.index else None)
            back_row = back_keys.loc[key].iloc[0] if key in back_keys.index and isinstance(back_keys.loc[key], pd.DataFrame) else (back_keys.loc[key] if key in back_keys.index else None)
            if live_row is None:
                diff_rows.append(
                    {
                        "diff_date": date,
                        "factor_version": factor_version,
                        "bond_code": str(key[1]),
                        "bar_start": key[0],
                        "field_name": "row",
                        "live_value": None,
                        "backfill_value": "present",
                        "abs_diff": None,
                        "diff_type": "backfill_only",
                        "severity": "WARN",
                    }
                )
                continue
            if back_row is None:
                diff_rows.append(
                    {
                        "diff_date": date,
                        "factor_version": factor_version,
                        "bond_code": str(key[1]),
                        "bar_start": key[0],
                        "field_name": "row",
                        "live_value": "present",
                        "backfill_value": None,
                        "abs_diff": None,
                        "diff_type": "live_only",
                        "severity": "WARN",
                    }
                )
                continue
            for field in compare_fields:
                live_value = live_row.get(field)
                back_value = back_row.get(field)
                if pd.isna(live_value) and pd.isna(back_value):
                    continue
                if field in {"signal_type", "alert_state"}:
                    if str(live_value) != str(back_value):
                        diff_rows.append(
                            {
                                "diff_date": date,
                                "factor_version": factor_version,
                                "bond_code": str(key[1]),
                                "bar_start": key[0],
                                "field_name": field,
                                "live_value": live_value,
                                "backfill_value": back_value,
                                "abs_diff": None,
                                "diff_type": "categorical_mismatch",
                                "severity": "ERROR" if field == "alert_state" else "WARN",
                            }
                        )
                    continue
                live_num = pd.to_numeric(live_value, errors="coerce")
                back_num = pd.to_numeric(back_value, errors="coerce")
                if pd.isna(live_num) or pd.isna(back_num):
                    continue
                abs_diff = abs(float(live_num) - float(back_num))
                if abs_diff > 1e-9:
                    diff_rows.append(
                        {
                            "diff_date": date,
                            "factor_version": factor_version,
                            "bond_code": str(key[1]),
                            "bar_start": key[0],
                            "field_name": field,
                            "live_value": live_value,
                            "backfill_value": back_value,
                            "abs_diff": abs_diff,
                            "diff_type": "numeric_diff",
                            "severity": "WARN" if abs_diff > 1e-6 else "INFO",
                        }
                    )

    rows_to_insert = diff_rows
    if write and not rows_to_insert:
        rows_to_insert = [
            {
                "diff_date": date,
                "factor_version": factor_version,
                "bond_code": "",
                "bar_start": pd.Timestamp(f"{date} 00:00:00"),
                "field_name": "summary",
                "live_value": f"factor={len(live_factor)} alert={len(live_alert)}",
                "backfill_value": f"rows={len(recomputed)}",
                "abs_diff": 0.0,
                "diff_type": "no_diff",
                "severity": "OK",
            }
        ]
    inserted = (
        insert_live_backfill_diff_rows(conn, rows_to_insert, schema["tables"].get("live_backfill_diff", "cb_intraday_live_backfill_diff"))
        if write
        else 0
    )
    summary = {
        "diff_date": date,
        "factor_version": factor_version,
        "factor_live_rows": int(len(live_factor)),
        "alert_live_rows": int(len(live_alert)),
        "backfill_rows": int(len(recomputed)),
        "diff_rows": int(len(diff_rows)),
        "diff_rows_inserted": int(inserted),
        "status": "OK" if not diff_rows else "DIFF_FOUND",
    }
    return summary, diff_rows
