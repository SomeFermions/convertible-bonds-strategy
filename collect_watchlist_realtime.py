import argparse
import logging
import os
import time
import zlib
from datetime import datetime
from typing import Any

import akshare as ak
import pandas as pd

from cb_universe import normalize_bond_code
from config import SOURCE, USE_PROXY_FOR_MARKET_DATA
from db import get_conn
from intraday_factors.pool_state import (
    ACTIVE,
    ACTIVE_PENDING_HISTORY,
    OBSERVE,
    SHADOW_RESEARCH,
    normalize_pool_state,
)
from market_calendar import MARKET_TZ, is_cn_trade_day


LOGGER = logging.getLogger(__name__)
LIVE_BAR_TABLE = "cb_watchlist_bar_5m_live"
TRADING_SESSIONS = (("09:30", "11:30"), ("13:00", "15:00"))
STOCK_CACHE_MAX_AGE_SECONDS = 20 * 60
DEFAULT_LIVE_RETAIN_DAYS = 45
DEFAULT_WATCHLIST_REBALANCE_DAYS = 3
DEFAULT_OBSERVATION_DAYS = 1
WATCHLIST_STATUS_ACTIVE = ACTIVE
WATCHLIST_STATUS_OBSERVE = OBSERVE
WATCHLIST_STATUS_SHADOW_RESEARCH = SHADOW_RESEARCH
WATCHLIST_STATUS_ACTIVE_PENDING_HISTORY = ACTIVE_PENDING_HISTORY
os.environ.setdefault("TZ", "Asia/Shanghai")
if hasattr(time, "tzset"):
    time.tzset()


def disable_proxy_if_needed() -> None:
    if USE_PROXY_FOR_MARKET_DATA:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)


def sql_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    escaped = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def sql_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    return str(float(value))


def market_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None)


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def minutes_since_midnight(now: datetime) -> int:
    return now.hour * 60 + now.minute


def is_trading_session(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False

    current = minutes_since_midnight(now)
    for start, end in TRADING_SESSIONS:
        start_hour, start_minute = parse_hhmm(start)
        end_hour, end_minute = parse_hhmm(end)
        start_value = start_hour * 60 + start_minute
        end_value = end_hour * 60 + end_minute
        if start_value <= current <= end_value:
            return True
    return False


def is_after_stop_time(now: datetime, stop_time: str) -> bool:
    stop_hour, stop_minute = parse_hhmm(stop_time)
    return minutes_since_midnight(now) >= stop_hour * 60 + stop_minute


def query_to_dataframe(conn, sql: str) -> pd.DataFrame:
    result = conn.query(sql)
    rows = list(result)
    fields = [field.name for field in result.fields]
    return pd.DataFrame(rows, columns=fields)


def try_execute(conn, sql: str) -> None:
    try:
        conn.execute(sql)
    except Exception as exc:
        LOGGER.debug("Ignored SQL failure sql=%s error=%s", sql, exc)


def create_live_bar_table(conn) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LIVE_BAR_TABLE} (
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
        """
    )
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN asset_type NCHAR(16)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN watchlist_status NCHAR(16)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN pool_state NCHAR(32)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN instrument_code NCHAR(16)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN instrument_name NCHAR(32)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN stock_code NCHAR(16)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN stock_name NCHAR(32)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN ytm_available BOOL")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN ytm_source NCHAR(32)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN turnover_available BOOL")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN turnover_source NCHAR(32)")
    try_execute(conn, f"ALTER TABLE {LIVE_BAR_TABLE} ADD COLUMN premium_source NCHAR(32)")


def ensure_watchlist_columns(conn) -> None:
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN watchlist_status NCHAR(16)")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN pool_state NCHAR(32)")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN candidate_since TIMESTAMP")
    try_execute(conn, "ALTER TABLE cb_watchlist_daily ADD COLUMN active_since TIMESTAMP")


def reset_live_data(conn) -> None:
    try_execute(conn, f"DROP TABLE IF EXISTS {LIVE_BAR_TABLE}")
    create_live_bar_table(conn)
    LOGGER.info("Reset live intraday table %s", LIVE_BAR_TABLE)


def live_retention_cutoff_date(existing_trade_dates: list[Any], today: pd.Timestamp, retain_days: int) -> str | None:
    if retain_days <= 0:
        return None

    dates = {today.strftime("%Y-%m-%d")}
    for value in existing_trade_dates:
        if value is None or pd.isna(value):
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            continue
        dates.add(pd.Timestamp(parsed).strftime("%Y-%m-%d"))

    if len(dates) <= retain_days:
        return None
    return sorted(dates, reverse=True)[retain_days - 1]


def purge_old_live_data(conn, retain_days: int = DEFAULT_LIVE_RETAIN_DAYS) -> None:
    create_live_bar_table(conn)
    df = query_to_dataframe(conn, f"SELECT trade_date FROM {LIVE_BAR_TABLE} ORDER BY ts DESC LIMIT 20000")
    trade_dates = [] if df.empty or "trade_date" not in df.columns else list(df["trade_date"])
    cutoff_date = live_retention_cutoff_date(trade_dates, market_now(), retain_days)
    if cutoff_date is None:
        LOGGER.info("Live intraday retention skipped retain_days=%s existing_trade_dates=%s", retain_days, len(set(trade_dates)))
        return

    cutoff_ts = f"{cutoff_date} 00:00:00"
    conn.execute(f"DELETE FROM {LIVE_BAR_TABLE} WHERE ts < {sql_string(cutoff_ts)}")
    LOGGER.info("Purged live intraday rows before %s retain_days=%s", cutoff_ts, retain_days)


def normalize_watchlist_status(value: Any) -> str:
    return normalize_pool_state(value, default=WATCHLIST_STATUS_ACTIVE)


def normalize_watchlist_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "watchlist_status" not in out.columns:
        out["watchlist_status"] = WATCHLIST_STATUS_ACTIVE
    out["watchlist_status"] = out["watchlist_status"].apply(normalize_watchlist_status)
    if "pool_state" not in out.columns:
        out["pool_state"] = out["watchlist_status"]
    out["pool_state"] = out["pool_state"].combine_first(out["watchlist_status"])
    out["pool_state"] = out["pool_state"].apply(normalize_watchlist_status)
    out["watchlist_status"] = out["pool_state"]

    if "ts" not in out.columns:
        out["ts"] = pd.NaT
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce").astype("datetime64[ns]")
    for col in ["candidate_since", "active_since"]:
        if col not in out.columns:
            out[col] = pd.NaT
        out[col] = pd.to_datetime(out[col], errors="coerce").astype("datetime64[ns]")

    out["candidate_since"] = out["candidate_since"].combine_first(out["ts"])
    active_mask = out["watchlist_status"] == WATCHLIST_STATUS_ACTIVE
    out.loc[active_mask, "active_since"] = out.loc[active_mask, "active_since"].combine_first(out.loc[active_mask, "ts"])
    return out


def collect_known_trade_dates(conn, today: pd.Timestamp) -> set[str]:
    dates = {today.strftime("%Y-%m-%d")}
    queries = [
        f"SELECT trade_date FROM {LIVE_BAR_TABLE} ORDER BY ts DESC LIMIT 20000",
        "SELECT ts FROM cb_watchlist_daily ORDER BY ts DESC LIMIT 5000",
        "SELECT ts FROM cb_screening_daily ORDER BY ts DESC LIMIT 5000",
    ]
    for sql in queries:
        try:
            df = query_to_dataframe(conn, sql)
        except Exception as exc:
            LOGGER.debug("Ignored trade-date query failure sql=%s error=%s", sql, exc)
            continue
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            dates.update(ts.strftime("%Y-%m-%d") for ts in parsed.dropna())
    return dates


def trade_days_elapsed(start: Any, today: pd.Timestamp, known_trade_dates: set[str] | None = None) -> int:
    start_ts = pd.to_datetime(start, errors="coerce")
    today_ts = pd.to_datetime(today, errors="coerce")
    if pd.isna(start_ts) or pd.isna(today_ts):
        return 0

    start_date = pd.Timestamp(start_ts).normalize()
    today_date = pd.Timestamp(today_ts).normalize()
    if today_date <= start_date:
        return 0

    if known_trade_dates:
        dates = {str(day)[:10] for day in known_trade_dates}
        dates.add(today_date.strftime("%Y-%m-%d"))
    else:
        dates = {ts.strftime("%Y-%m-%d") for ts in pd.bdate_range(start_date, today_date)}
    return sum(start_date < pd.Timestamp(day) <= today_date for day in dates)


def should_rebalance_watchlist(
    previous_watchlist: pd.DataFrame,
    today: pd.Timestamp,
    known_trade_dates: set[str],
    rebalance_days: int,
) -> bool:
    previous = normalize_watchlist_frame(previous_watchlist)
    if previous.empty:
        return True

    previous_active = previous[previous["watchlist_status"] == WATCHLIST_STATUS_ACTIVE]
    if previous_active.empty:
        return True

    active_since = previous_active["active_since"].dropna()
    if active_since.empty:
        return True
    active_epoch = active_since.max()
    return trade_days_elapsed(active_epoch, today, known_trade_dates) >= rebalance_days


def recent_screening_snapshot_ids(conn, limit: int = 10) -> list[str]:
    df = query_to_dataframe(conn, "SELECT snapshot_id FROM cb_screening_daily ORDER BY ts DESC LIMIT 5000")
    if df.empty or "snapshot_id" not in df.columns:
        return []

    snapshot_ids = []
    seen = set()
    for value in df["snapshot_id"]:
        if pd.isna(value):
            continue
        snapshot_id = str(value)
        if snapshot_id in seen:
            continue
        snapshot_ids.append(snapshot_id)
        seen.add(snapshot_id)
        if len(snapshot_ids) >= limit:
            break
    return snapshot_ids


def load_screening_snapshot(conn, snapshot_id: str) -> pd.DataFrame:
    df = query_to_dataframe(
        conn,
        f"SELECT * FROM cb_screening_daily WHERE snapshot_id = {sql_string(snapshot_id)}",
    )
    LOGGER.info("Loaded screening snapshot snapshot_id=%s rows=%s", snapshot_id, len(df))
    return df


def is_non_st(stock_name: str) -> bool:
    return "ST" not in str(stock_name).upper()


def build_watchlist_from_local_snapshot(
    snapshot: pd.DataFrame,
    premium_min: float = 20.0,
    premium_max: float = 30.0,
    max_debt_asset_ratio: float = 70.0,
) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot

    df = snapshot.copy()
    for col in ["premium_rate", "debt_asset_ratio", "operating_cash_flow", "free_cash_flow"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "days_to_maturity" in df.columns:
        df["days_to_maturity"] = pd.to_numeric(df["days_to_maturity"], errors="coerce")
    else:
        df["days_to_maturity"] = None

    out = df[
        df["premium_rate"].between(premium_min, premium_max, inclusive="both")
        & (df["debt_asset_ratio"] < max_debt_asset_ratio)
        & (df["operating_cash_flow"] > 0)
        & (df["free_cash_flow"] > 0)
        & df["stock_name"].apply(is_non_st)
        & (df["days_to_maturity"] > 183)
    ].copy()

    out = out.sort_values(["premium_rate", "debt_asset_ratio"], ascending=[True, True])
    return out.reset_index(drop=True)


def build_monitoring_watchlist(
    candidates: pd.DataFrame,
    previous_watchlist: pd.DataFrame,
    today: pd.Timestamp,
    known_trade_dates: set[str],
    rebalance_days: int,
    observation_days: int,
    active_readiness: dict[str, bool] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    now_ts = pd.Timestamp(today).floor("s")
    candidates = candidates.copy()
    candidates["bond_code"] = candidates["bond_code"].astype(str)
    previous = normalize_watchlist_frame(previous_watchlist)
    if not previous.empty:
        previous["bond_code"] = previous["bond_code"].astype(str)

    previous_by_bond = {}
    if not previous.empty:
        previous_by_bond = previous.drop_duplicates(subset=["bond_code"], keep="last").set_index("bond_code").to_dict("index")

    previous_active = pd.DataFrame()
    previous_active_codes: set[str] = set()
    if not previous.empty:
        previous_active = previous[previous["watchlist_status"] == WATCHLIST_STATUS_ACTIVE].copy()
        previous_active_codes = set(previous_active["bond_code"].astype(str))

    rebalance_due = should_rebalance_watchlist(previous, now_ts, known_trade_dates, rebalance_days)
    rows = []

    if not rebalance_due:
        if not previous_active.empty:
            active_rows = previous_active.copy()
            active_rows["watchlist_status"] = WATCHLIST_STATUS_ACTIVE
            active_rows["pool_state"] = WATCHLIST_STATUS_ACTIVE
            rows.append(active_rows)

        observe = candidates[~candidates["bond_code"].isin(previous_active_codes)].copy()
        if not observe.empty:
            observe["watchlist_status"] = WATCHLIST_STATUS_OBSERVE
            observe["pool_state"] = WATCHLIST_STATUS_OBSERVE
            observe["active_since"] = pd.NaT
            observe["candidate_since"] = observe["bond_code"].map(
                lambda code: previous_by_bond.get(str(code), {}).get("candidate_since", now_ts)
            )
            rows.append(observe)

        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        LOGGER.info(
            "Kept active watchlist between rebalance windows active=%s observe=%s rebalance_days=%s",
            len(out[out["watchlist_status"] == WATCHLIST_STATUS_ACTIVE]) if not out.empty else 0,
            len(out[out["watchlist_status"] == WATCHLIST_STATUS_OBSERVE]) if not out.empty else 0,
            rebalance_days,
        )
        return normalize_watchlist_frame(out)

    promoted_rows = []
    readiness_required = active_readiness is not None
    active_readiness = active_readiness or {}
    for _, row in candidates.iterrows():
        bond_code = str(row["bond_code"])
        previous_row = previous_by_bond.get(bond_code, {})
        candidate_since = previous_row.get("candidate_since", now_ts)
        was_active = bond_code in previous_active_codes
        observed_long_enough = trade_days_elapsed(candidate_since, now_ts, known_trade_dates) >= observation_days
        if previous.empty or was_active:
            status = WATCHLIST_STATUS_ACTIVE
        elif observed_long_enough:
            status = (
                WATCHLIST_STATUS_ACTIVE
                if (not readiness_required or active_readiness.get(bond_code, False))
                else WATCHLIST_STATUS_ACTIVE_PENDING_HISTORY
            )
        else:
            status = WATCHLIST_STATUS_OBSERVE

        out_row = row.copy()
        out_row["watchlist_status"] = status
        out_row["pool_state"] = status
        out_row["candidate_since"] = candidate_since
        out_row["active_since"] = now_ts if status == WATCHLIST_STATUS_ACTIVE else pd.NaT
        promoted_rows.append(out_row)

    out = normalize_watchlist_frame(pd.DataFrame(promoted_rows))
    LOGGER.info(
        "Rebuilt watchlist at rebalance window active=%s observe=%s observation_days=%s",
        len(out[out["watchlist_status"] == WATCHLIST_STATUS_ACTIVE]),
        len(out[out["watchlist_status"] == WATCHLIST_STATUS_OBSERVE]),
        observation_days,
    )
    return out


def insert_watchlist(conn, watchlist: pd.DataFrame) -> int:
    if watchlist.empty:
        return 0

    ensure_watchlist_columns(conn)
    snapshot_id = datetime.now(MARKET_TZ).strftime("%Y%m%d%H%M%S")
    base_ts = market_now().floor("s")
    values = []
    for idx, row in enumerate(watchlist.itertuples(index=False)):
        row_dict = row._asdict()
        row_ts = (base_ts + pd.Timedelta(milliseconds=idx)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        report_date = None
        if pd.notna(row_dict.get("report_date")):
            report_date = pd.Timestamp(row_dict.get("report_date")).strftime("%Y-%m-%d %H:%M:%S")
        maturity_date = None
        if pd.notna(row_dict.get("maturity_date")):
            maturity_date = pd.Timestamp(row_dict.get("maturity_date")).strftime("%Y-%m-%d %H:%M:%S")
        candidate_since = None
        if pd.notna(row_dict.get("candidate_since")):
            candidate_since = pd.Timestamp(row_dict.get("candidate_since")).strftime("%Y-%m-%d %H:%M:%S")
        active_since = None
        if pd.notna(row_dict.get("active_since")):
            active_since = pd.Timestamp(row_dict.get("active_since")).strftime("%Y-%m-%d %H:%M:%S")
        status = normalize_watchlist_status(row_dict.get("watchlist_status"))
        pool_state = normalize_watchlist_status(row_dict.get("pool_state", status))

        values.append(
            "("
            f"{sql_string(row_ts)}, {sql_string(snapshot_id)}, "
            f"{sql_string(row_dict.get('bond_code'))}, {sql_string(row_dict.get('bond_name'))}, {sql_string(row_dict.get('ak_symbol'))}, "
            f"{sql_number(row_dict.get('bond_price'))}, {sql_number(row_dict.get('premium_rate'))}, "
            f"{sql_number(row_dict.get('conversion_value'))}, {sql_number(row_dict.get('conversion_price'))}, "
            f"{sql_string(row_dict.get('stock_code'))}, {sql_string(row_dict.get('stock_name'))}, {sql_number(row_dict.get('stock_price'))}, "
            f"{sql_string(maturity_date)}, "
            f"{int(row_dict.get('days_to_maturity')) if pd.notna(row_dict.get('days_to_maturity')) else 'NULL'}, "
            f"{sql_string(report_date)}, {sql_number(row_dict.get('debt_asset_ratio'))}, "
            f"{sql_number(row_dict.get('operating_cash_flow'))}, {sql_number(row_dict.get('free_cash_flow'))}, "
            f"{sql_string(status)}, {sql_string(pool_state)}, {sql_string(candidate_since)}, {sql_string(active_since)}, "
            f"{sql_string(SOURCE + '_watchlist_local')}"
            ")"
        )

    conn.execute(
        """
        INSERT INTO cb_watchlist_daily (
            ts, snapshot_id, bond_code, bond_name, ak_symbol,
            bond_price, premium_rate, conversion_value, conversion_price,
            stock_code, stock_name, stock_price, maturity_date, days_to_maturity, report_date,
            debt_asset_ratio, operating_cash_flow, free_cash_flow,
            watchlist_status, pool_state, candidate_since, active_since, source
        ) VALUES
        """
        + " ".join(values)
    )
    return len(values)


def build_watchlist_from_latest_screening(conn, args: argparse.Namespace) -> pd.DataFrame:
    ensure_watchlist_columns(conn)
    previous_watchlist = load_latest_watchlist(conn)
    today = market_now()
    known_trade_dates = collect_known_trade_dates(conn, today)

    for snapshot_id in recent_screening_snapshot_ids(conn):
        snapshot = load_screening_snapshot(conn, snapshot_id)
        candidates = build_watchlist_from_local_snapshot(
            snapshot,
            premium_min=args.premium_min,
            premium_max=args.premium_max,
            max_debt_asset_ratio=args.max_debt_asset_ratio,
        )
        if candidates.empty:
            LOGGER.warning("Screening snapshot_id=%s produced empty watchlist; trying older snapshot", snapshot_id)
            continue

        active_readiness: dict[str, bool] = {}
        normalized_previous = normalize_watchlist_frame(previous_watchlist)
        rebalance_due = should_rebalance_watchlist(normalized_previous, today, known_trade_dates, args.watchlist_rebalance_days)
        if rebalance_due and not normalized_previous.empty:
            previous_by_bond = normalized_previous.drop_duplicates(subset=["bond_code"], keep="last").set_index("bond_code").to_dict("index")
            previous_active_codes = set(
                normalized_previous[normalized_previous["watchlist_status"] == WATCHLIST_STATUS_ACTIVE]["bond_code"].astype(str)
            )
            for candidate in candidates.itertuples(index=False):
                bond_code = str(candidate.bond_code)
                previous_row = previous_by_bond.get(bond_code, {})
                if bond_code in previous_active_codes:
                    continue
                candidate_since = previous_row.get("candidate_since", today)
                if trade_days_elapsed(candidate_since, today, known_trade_dates) < args.observation_days:
                    continue
                try:
                    from intraday_factors.history_tools import coverage_for_pair, prepare_active_history

                    report = coverage_for_pair(conn, candidate._asdict(), args.history_trading_days, today)
                    if not report.get("ready_for_active") and args.prepare_history_on_promotion:
                        report = prepare_active_history(
                            conn,
                            bond_code=bond_code,
                            trading_days=args.history_trading_days,
                            asof_date=today,
                            sleep_seconds=args.history_sleep_seconds,
                        )
                    active_readiness[bond_code] = bool(report.get("ready_for_active"))
                    if not active_readiness[bond_code]:
                        LOGGER.warning(
                            "OBSERVE promotion blocked by missing 5m history bond_code=%s missing_cb=%s missing_stock=%s reason=%s",
                            bond_code,
                            report.get("missing_cb_bars"),
                            report.get("missing_stock_bars"),
                            report.get("failure_reason") or report.get("error"),
                        )
                except Exception as exc:
                    active_readiness[bond_code] = False
                    LOGGER.exception("OBSERVE promotion readiness failed bond_code=%s error=%s", bond_code, exc)

        watchlist = build_monitoring_watchlist(
            candidates,
            previous_watchlist=previous_watchlist,
            today=today,
            known_trade_dates=known_trade_dates,
            rebalance_days=args.watchlist_rebalance_days,
            observation_days=args.observation_days,
            active_readiness=active_readiness,
        )
        inserted = insert_watchlist(conn, watchlist)
        LOGGER.info(
            "Built monitoring watchlist rows=%s active=%s observe=%s inserted=%s into cb_watchlist_daily from screening snapshot_id=%s",
            len(watchlist),
            len(watchlist[watchlist["watchlist_status"] == WATCHLIST_STATUS_ACTIVE]) if not watchlist.empty else 0,
            len(watchlist[watchlist["watchlist_status"] == WATCHLIST_STATUS_OBSERVE]) if not watchlist.empty else 0,
            inserted,
            snapshot_id,
        )
        return watchlist

    LOGGER.warning("Local watchlist is empty across recent screening snapshots")
    return pd.DataFrame()


def latest_watchlist_snapshot_id(conn) -> str | None:
    ensure_watchlist_columns(conn)
    df = query_to_dataframe(conn, "SELECT LAST(snapshot_id) AS snapshot_id FROM cb_watchlist_daily")
    if df.empty or pd.isna(df.loc[0, "snapshot_id"]):
        return None
    return str(df.loc[0, "snapshot_id"])


def load_latest_watchlist(conn) -> pd.DataFrame:
    ensure_watchlist_columns(conn)
    snapshot_id = latest_watchlist_snapshot_id(conn)
    if snapshot_id is None:
        return pd.DataFrame()
    df = query_to_dataframe(
        conn,
        f"SELECT * FROM cb_watchlist_daily WHERE snapshot_id = {sql_string(snapshot_id)}",
    )
    df = normalize_watchlist_frame(df)
    LOGGER.info(
        "Loaded watchlist snapshot_id=%s rows=%s active=%s observe=%s",
        snapshot_id,
        len(df),
        len(df[df["watchlist_status"] == WATCHLIST_STATUS_ACTIVE]) if not df.empty else 0,
        len(df[df["watchlist_status"] == WATCHLIST_STATUS_OBSERVE]) if not df.empty else 0,
    )
    return df


def fetch_spot_quotes(watchlist: pd.DataFrame) -> pd.DataFrame:
    disable_proxy_if_needed()
    raw = ak.bond_zh_hs_cov_spot()
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["bond_code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    wanted = set(watchlist["bond_code"].astype(str))
    df = df[df["bond_code"].isin(wanted)].copy()
    if df.empty:
        return df

    for col in ["trade", "open", "high", "low", "volume", "amount", "changepercent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quote_ts"] = market_now().floor("s")

    pairs = watchlist[["bond_code", "bond_name", "stock_code", "stock_name"]].copy()
    pairs["watchlist_status"] = (
        watchlist["watchlist_status"].apply(normalize_watchlist_status)
        if "watchlist_status" in watchlist.columns
        else WATCHLIST_STATUS_ACTIVE
    )
    pairs["pool_state"] = (
        watchlist["pool_state"].apply(normalize_watchlist_status)
        if "pool_state" in watchlist.columns
        else pairs["watchlist_status"]
    )
    pairs["bond_code"] = pairs["bond_code"].astype(str)
    df = df.merge(pairs, on="bond_code", how="left")
    df["bond_name"] = df["bond_name"].fillna(df["name"])
    df["watchlist_status"] = df["watchlist_status"].apply(normalize_watchlist_status)
    return df[
        [
            "quote_ts",
            "bond_code",
            "bond_name",
            "stock_code",
            "stock_name",
            "watchlist_status",
            "pool_state",
            "symbol",
            "trade",
            "open",
            "high",
            "low",
            "volume",
            "amount",
            "changepercent",
            "ticktime",
        ]
    ]


def stock_market_from_code(stock_code: str) -> str | None:
    code = str(stock_code).strip().zfill(6)
    if code.startswith(("60", "68", "90")):
        return "SH"
    if code.startswith(("00", "30", "20")):
        return "SZ"
    return None


def watchlist_stock_pairs(watchlist: pd.DataFrame) -> pd.DataFrame:
    pairs = watchlist[["bond_code", "bond_name", "stock_code", "stock_name"]].copy()
    pairs["watchlist_status"] = (
        watchlist["watchlist_status"].apply(normalize_watchlist_status)
        if "watchlist_status" in watchlist.columns
        else WATCHLIST_STATUS_ACTIVE
    )
    pairs["pool_state"] = (
        watchlist["pool_state"].apply(normalize_watchlist_status)
        if "pool_state" in watchlist.columns
        else pairs["watchlist_status"]
    )
    pairs["bond_code"] = pairs["bond_code"].astype(str)
    pairs["stock_code"] = pairs["stock_code"].astype(str).str.zfill(6)
    pairs["stock_name"] = pairs["stock_name"].astype(str)
    pairs["pair_key"] = pairs["bond_code"] + ":" + pairs["stock_code"]
    return pairs.drop_duplicates(subset=["pair_key"]).reset_index(drop=True)


class WatchlistStockQuoteProvider:
    def __init__(self, bar_minutes: int):
        self.bar_minutes = bar_minutes
        self.cache: dict[str, tuple[pd.Timestamp, dict[str, Any]]] = {}
        self.seen_hist_bars: set[tuple[str, pd.Timestamp]] = set()

    def fetch(self, watchlist: pd.DataFrame) -> pd.DataFrame:
        disable_proxy_if_needed()
        rows = []
        bid_ask_failures = 0
        hist_failures = 0
        cache_hits = 0
        pairs = watchlist_stock_pairs(watchlist)
        for _, group in pairs.groupby("stock_code", sort=False):
            stock_pairs = list(group.itertuples(index=False))
            representative = stock_pairs[0]
            row = self._fetch_bid_ask_row(representative)
            if row is None:
                bid_ask_failures += 1
                row = self._fetch_hist_5m_row(representative)
                if row is None:
                    hist_failures += 1
                    for pair in stock_pairs:
                        cached = self._cached_row(pair.pair_key)
                        if cached is not None:
                            rows.append(cached)
                            cache_hits += 1
                    continue
            if row is not None:
                for pair in stock_pairs:
                    pair_row = self._copy_for_pair(row, pair)
                    self._store_cache(pair_row)
                    rows.append(pair_row)

        if bid_ask_failures or hist_failures or cache_hits:
            LOGGER.info(
                "Stock quote provider fallback stats bid_ask_stock_failures=%s hist_stock_failures=%s cache_pair_hits=%s",
                bid_ask_failures,
                hist_failures,
                cache_hits,
            )
        return pd.DataFrame(rows)

    def _base_row(self, pair) -> dict[str, Any]:
        return {
            "pair_key": pair.pair_key,
            "bond_code": str(pair.bond_code),
            "bond_name": str(pair.bond_name),
            "stock_code": str(pair.stock_code).zfill(6),
            "stock_name": str(pair.stock_name),
            "watchlist_status": normalize_watchlist_status(getattr(pair, "watchlist_status", WATCHLIST_STATUS_ACTIVE)),
            "pool_state": normalize_watchlist_status(getattr(pair, "pool_state", getattr(pair, "watchlist_status", WATCHLIST_STATUS_ACTIVE))),
            "market": stock_market_from_code(pair.stock_code),
        }

    def _copy_for_pair(self, row: dict[str, Any], pair) -> dict[str, Any]:
        out = dict(row)
        out.update(self._base_row(pair))
        return out

    def _fetch_bid_ask_row(self, pair) -> dict[str, Any] | None:
        try:
            raw = ak.stock_bid_ask_em(symbol=str(pair.stock_code).zfill(6))
            if raw is None or raw.empty:
                return None
            values = dict(zip(raw["item"].astype(str), raw["value"]))
            price = pd.to_numeric(values.get("最新"), errors="coerce")
            if pd.isna(price) or float(price) <= 0:
                return None

            row = self._base_row(pair)
            row.update(
                {
                    "quote_ts": market_now().floor("s"),
                    "trade": float(price),
                    "open": pd.to_numeric(values.get("今开"), errors="coerce"),
                    "high": pd.to_numeric(values.get("最高"), errors="coerce"),
                    "low": pd.to_numeric(values.get("最低"), errors="coerce"),
                    "volume": pd.to_numeric(values.get("总手"), errors="coerce"),
                    "amount": pd.to_numeric(values.get("金额"), errors="coerce"),
                    "changepercent": pd.to_numeric(values.get("涨幅"), errors="coerce"),
                    "turnover_rate": pd.to_numeric(values.get("换手"), errors="coerce"),
                    "volume_is_delta": False,
                    "amount_is_delta": False,
                    "quote_source": "stock_bid_ask_em",
                }
            )
            return row
        except Exception as exc:
            LOGGER.debug("stock_bid_ask_em failed stock_code=%s error=%s", pair.stock_code, exc)
            return None

    def _fetch_hist_5m_row(self, pair) -> dict[str, Any] | None:
        try:
            now = market_now()
            start = now.normalize().strftime("%Y-%m-%d 09:30:00")
            end = (now + pd.Timedelta(minutes=self.bar_minutes)).strftime("%Y-%m-%d %H:%M:%S")
            raw = ak.stock_zh_a_hist_min_em(
                symbol=str(pair.stock_code).zfill(6),
                start_date=start,
                end_date=end,
                period=str(self.bar_minutes),
                adjust="",
            )
            if raw is None or raw.empty:
                return None
            df = raw.copy()
            df["bar_start"] = pd.to_datetime(df["时间"], errors="coerce")
            df = df.dropna(subset=["bar_start"]).sort_values("bar_start")
            df = df[df["bar_start"].dt.date == now.date()]
            if df.empty:
                return None
            latest = df.iloc[-1]
            bar_start_ts = pd.Timestamp(latest["bar_start"]).tz_localize(None)
            seen_key = (str(pair.stock_code).zfill(6), bar_start_ts)
            if seen_key in self.seen_hist_bars:
                return None
            self.seen_hist_bars.add(seen_key)

            row = self._base_row(pair)
            row.update(
                {
                    "quote_ts": bar_start_ts,
                    "trade": pd.to_numeric(latest.get("收盘"), errors="coerce"),
                    "open": pd.to_numeric(latest.get("开盘"), errors="coerce"),
                    "high": pd.to_numeric(latest.get("最高"), errors="coerce"),
                    "low": pd.to_numeric(latest.get("最低"), errors="coerce"),
                    "volume": pd.to_numeric(latest.get("成交量"), errors="coerce"),
                    "amount": pd.to_numeric(latest.get("成交额"), errors="coerce"),
                    "changepercent": pd.to_numeric(latest.get("涨跌幅"), errors="coerce"),
                    "turnover_rate": pd.to_numeric(latest.get("换手率"), errors="coerce"),
                    "volume_is_delta": True,
                    "amount_is_delta": True,
                    "quote_source": "stock_zh_a_hist_min_em",
                }
            )
            if pd.isna(row["trade"]) or float(row["trade"]) <= 0:
                return None
            return row
        except Exception as exc:
            LOGGER.debug("stock_zh_a_hist_min_em failed stock_code=%s error=%s", pair.stock_code, exc)
            return None

    def _store_cache(self, row: dict[str, Any]) -> None:
        self.cache[str(row["pair_key"])] = (market_now(), dict(row))

    def _cached_row(self, pair_key: str) -> dict[str, Any] | None:
        cached = self.cache.get(str(pair_key))
        if cached is None:
            return None
        cached_at, row = cached
        age_seconds = (market_now() - cached_at).total_seconds()
        if age_seconds > STOCK_CACHE_MAX_AGE_SECONDS:
            return None
        out = dict(row)
        out["quote_ts"] = market_now().floor("s")
        out["volume"] = 0.0
        out["amount"] = 0.0
        out["volume_is_delta"] = True
        out["amount_is_delta"] = True
        out["quote_source"] = "stock_quote_cache"
        return out


def normalize_percent(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = pd.to_numeric(str(value).replace("%", "").strip(), errors="coerce")
    if pd.isna(result):
        return None
    return float(result)


def fetch_realtime_cb_metrics(watchlist: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    disable_proxy_if_needed()
    raw = ak.bond_zh_cov()
    if raw is None or raw.empty:
        return {}

    df = pd.DataFrame()
    df["bond_code"] = raw["债券代码"].apply(normalize_bond_code)
    df["premium_rate"] = raw["转股溢价率"].apply(normalize_percent)
    df["conversion_value"] = pd.to_numeric(raw["转股价值"], errors="coerce")
    df["conversion_price"] = pd.to_numeric(raw["转股价"], errors="coerce")
    wanted = set(watchlist["bond_code"].astype(str))
    df = df[df["bond_code"].isin(wanted)]

    return {
        str(row.bond_code): {
            "premium_rate": None if pd.isna(row.premium_rate) else float(row.premium_rate),
            "conversion_value": None if pd.isna(row.conversion_value) else float(row.conversion_value),
            "conversion_price": None if pd.isna(row.conversion_price) else float(row.conversion_price),
        }
        for row in df.itertuples(index=False)
    }


def bucket_start(ts: pd.Timestamp, bar_minutes: int) -> pd.Timestamp:
    minute = (ts.minute // bar_minutes) * bar_minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def live_row_timestamp(bar_start: pd.Timestamp, entity_key: str) -> pd.Timestamp:
    offset_ms = zlib.crc32(str(entity_key).encode("utf-8")) % 300000
    return bar_start + pd.Timedelta(milliseconds=offset_ms)


def merge_quote_source(current_source: Any, next_source: Any) -> str:
    current = str(current_source or "")
    next_value = str(next_source or "")
    if not current:
        return next_value[:32]
    if current == next_value:
        return current[:32]
    if current == "mixed_quote_sources":
        return current
    return "mixed_quote_sources"


class FiveMinuteAggregator:
    def __init__(
        self,
        bar_minutes: int,
        asset_type: str,
        code_attr: str,
        name_attr: str,
        code_key: str,
        name_key: str,
        identity_attr: str | None = None,
    ):
        self.bar_minutes = bar_minutes
        self.asset_type = asset_type
        self.code_attr = code_attr
        self.name_attr = name_attr
        self.code_key = code_key
        self.name_key = name_key
        self.identity_attr = identity_attr or code_attr
        self.current: dict[str, dict[str, Any]] = {}
        self.previous_cumulative: dict[str, tuple[float | None, float | None]] = {}

    def update(self, quotes: pd.DataFrame) -> list[dict[str, Any]]:
        completed = []
        for row in quotes.itertuples(index=False):
            price = row.trade
            if pd.isna(price) or price <= 0:
                continue

            quote_ts = pd.Timestamp(row.quote_ts).tz_localize(None)
            bucket = bucket_start(quote_ts, self.bar_minutes)
            code = str(getattr(row, self.code_attr))
            identity = str(getattr(row, self.identity_attr))

            prev_volume, prev_amount = self.previous_cumulative.get(identity, (None, None))
            volume_is_delta = bool(getattr(row, "volume_is_delta", False))
            amount_is_delta = bool(getattr(row, "amount_is_delta", False))
            volume_delta = 0.0
            amount_delta = 0.0
            if volume_is_delta and pd.notna(row.volume):
                volume_delta = float(row.volume)
            elif prev_volume is not None and pd.notna(row.volume) and row.volume >= prev_volume:
                volume_delta = float(row.volume - prev_volume)
            if amount_is_delta and pd.notna(row.amount):
                amount_delta = float(row.amount)
            elif prev_amount is not None and pd.notna(row.amount) and row.amount >= prev_amount:
                amount_delta = float(row.amount - prev_amount)
            self.previous_cumulative[identity] = (
                None if pd.isna(row.volume) else float(row.volume),
                None if pd.isna(row.amount) else float(row.amount),
            )

            state = self.current.get(identity)
            if state and state["ts"] != bucket:
                completed.append(state)
                state = None

            row_source = str(getattr(row, "quote_source", f"{SOURCE}_watchlist_realtime"))
            is_bar_quote = volume_is_delta or row_source.endswith("hist_min_em")
            row_open = pd.to_numeric(getattr(row, "open", None), errors="coerce")
            row_high = pd.to_numeric(getattr(row, "high", None), errors="coerce")
            row_low = pd.to_numeric(getattr(row, "low", None), errors="coerce")
            open_value = float(row_open) if is_bar_quote and pd.notna(row_open) and row_open > 0 else float(price)
            high_value = float(row_high) if is_bar_quote and pd.notna(row_high) and row_high > 0 else float(price)
            low_value = float(row_low) if is_bar_quote and pd.notna(row_low) and row_low > 0 else float(price)

            if state is None:
                state = {
                    "ts": bucket,
                    "asset_type": self.asset_type,
                    "watchlist_status": normalize_watchlist_status(getattr(row, "watchlist_status", WATCHLIST_STATUS_ACTIVE)),
                    "pool_state": normalize_watchlist_status(getattr(row, "pool_state", getattr(row, "watchlist_status", WATCHLIST_STATUS_ACTIVE))),
                    "bond_code": "",
                    "bond_name": "",
                    "stock_code": "",
                    "stock_name": "",
                    "market": str(getattr(row, "market", getattr(row, "symbol", "")))[:2].upper(),
                    "open": open_value,
                    "high": high_value,
                    "low": low_value,
                    "close": float(price),
                    "change_percent": None if pd.isna(row.changepercent) else float(row.changepercent),
                    "turnover_rate": None if not hasattr(row, "turnover_rate") or pd.isna(row.turnover_rate) else float(row.turnover_rate),
                    "volume": volume_delta,
                    "amount": amount_delta,
                    "prices": [float(price)],
                    "source": row_source,
                }
                state[self.code_key] = code
                state[self.name_key] = str(getattr(row, self.name_attr))
                if hasattr(row, "bond_code"):
                    state["bond_code"] = str(row.bond_code)
                if hasattr(row, "bond_name"):
                    state["bond_name"] = str(row.bond_name)
                if hasattr(row, "stock_code"):
                    state["stock_code"] = str(row.stock_code)
                if hasattr(row, "stock_name"):
                    state["stock_name"] = str(row.stock_name)
                self.current[identity] = state
            else:
                state["high"] = max(state["high"], high_value)
                state["low"] = min(state["low"], low_value)
                state["close"] = float(price)
                state["change_percent"] = None if pd.isna(row.changepercent) else float(row.changepercent)
                state["turnover_rate"] = None if not hasattr(row, "turnover_rate") or pd.isna(row.turnover_rate) else float(row.turnover_rate)
                state["volume"] += volume_delta
                state["amount"] += amount_delta
                state["prices"].append(float(price))
                state["source"] = merge_quote_source(state.get("source"), row_source)
                state["watchlist_status"] = normalize_watchlist_status(
                    getattr(row, "watchlist_status", state.get("watchlist_status", WATCHLIST_STATUS_ACTIVE))
                )
                state["pool_state"] = normalize_watchlist_status(
                    getattr(row, "pool_state", state.get("pool_state", state.get("watchlist_status", WATCHLIST_STATUS_ACTIVE)))
                )

        return completed

    def flush_completed(self, now_ts: pd.Timestamp) -> list[dict[str, Any]]:
        current_bucket = bucket_start(pd.Timestamp(now_ts).tz_localize(None), self.bar_minutes)
        completed = []
        for code, state in list(self.current.items()):
            if state["ts"] < current_bucket:
                completed.append(state)
                del self.current[code]
        return completed

    def flush_all(self) -> list[dict[str, Any]]:
        bars = list(self.current.values())
        self.current = {}
        return bars


def realized_volatility(prices: list[float]) -> float | None:
    if len(prices) < 2:
        return None
    returns = pd.Series(prices).pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std(ddof=0))


def insert_5m_bars(conn, bars: list[dict[str, Any]], metrics: dict[str, dict[str, float | None]]) -> int:
    if not bars:
        return 0

    create_live_bar_table(conn)
    values = []
    for bar in sorted(bars, key=lambda item: (str(item.get("asset_type", "CB")), str(item["bond_code"]), str(item.get("stock_code", "")))):
        bar_start = pd.Timestamp(bar["ts"]).tz_localize(None)
        asset_type = str(bar.get("asset_type", "CB"))
        watchlist_status = normalize_watchlist_status(bar.get("watchlist_status"))
        pool_state = normalize_watchlist_status(bar.get("pool_state", watchlist_status))
        instrument_code = str(bar["bond_code"] if asset_type == "CB" else bar.get("stock_code", ""))
        instrument_name = str(bar["bond_name"] if asset_type == "CB" else bar.get("stock_name", ""))
        row_ts = live_row_timestamp(bar_start, f"{asset_type}:{bar.get('bond_code', '')}:{instrument_code}")
        trade_date = bar_start.strftime("%Y-%m-%d")
        metric = metrics.get(str(bar["bond_code"]), {}) if asset_type == "CB" else {}
        premium_value = metric.get("premium_rate") if asset_type == "CB" else None
        turnover_value = bar.get("turnover_rate")
        turnover_available = turnover_value is not None and not pd.isna(turnover_value)
        premium_source = "vendor" if premium_value is not None and not pd.isna(premium_value) else "unavailable"
        values.append(
            "("
            f"{sql_string(row_ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])}, "
            f"{sql_string(bar_start.strftime('%Y-%m-%d %H:%M:%S'))}, "
            f"{sql_string(trade_date)}, "
            f"{sql_string(asset_type)}, "
            f"{sql_string(watchlist_status)}, "
            f"{sql_string(pool_state)}, "
            f"{sql_string(instrument_code)}, "
            f"{sql_string(instrument_name)}, "
            f"{sql_string(bar.get('bond_code'))}, "
            f"{sql_string(bar.get('bond_name'))}, "
            f"{sql_string(bar.get('stock_code'))}, "
            f"{sql_string(bar.get('stock_name'))}, "
            f"{sql_string(bar['market'])}, "
            f"{sql_number(bar['open'])}, "
            f"{sql_number(bar['high'])}, "
            f"{sql_number(bar['low'])}, "
            f"{sql_number(bar['close'])}, "
            f"{sql_number(bar.get('change_percent'))}, "
            f"{sql_number(premium_value)}, "
            "NULL, "
            f"{sql_number(metric.get('conversion_value'))}, "
            f"{sql_number(metric.get('conversion_price'))}, "
            f"{sql_number(bar['volume'])}, "
            f"{sql_number(bar['amount'])}, "
            f"{sql_number(bar.get('turnover_rate'))}, "
            "0, "
            f"{sql_string('unavailable')}, "
            f"{'1' if turnover_available else '0'}, "
            f"{sql_string('vendor' if turnover_available else 'unavailable')}, "
            f"{sql_string(premium_source)}, "
            f"{sql_number(realized_volatility(bar.get('prices', [])))}, "
            f"{len(bar.get('prices', []))}, "
            f"{sql_string(bar.get('source', SOURCE + '_watchlist_realtime'))}"
            ")"
        )

    conn.execute(
        f"""
        INSERT INTO {LIVE_BAR_TABLE} (
            ts, bar_start, trade_date,
            asset_type, watchlist_status, pool_state, instrument_code, instrument_name,
            bond_code, bond_name, stock_code, stock_name, market,
            open, high, low, close, change_percent,
            premium_rate, ytm, conversion_value, conversion_price,
            volume, amount, turnover_rate,
            ytm_available, ytm_source, turnover_available, turnover_source, premium_source,
            volatility, quote_count, source
        ) VALUES
        """
        + " ".join(values)
    )
    return len(values)


def run_realtime_monitor(
    conn,
    watchlist: pd.DataFrame,
    interval_seconds: float,
    bar_minutes: int,
    metrics_refresh_seconds: float,
    stop_time: str,
) -> None:
    if watchlist.empty:
        raise RuntimeError("No local watchlist available. Run daily screening and rebuild the watchlist before monitoring.")

    create_live_bar_table(conn)
    LOGGER.info(
        "Realtime monitor starts watchlist_size=%s active=%s observe=%s interval=%ss bar_minutes=%s stop_time=%s",
        len(watchlist),
        len(watchlist[watchlist["watchlist_status"] == WATCHLIST_STATUS_ACTIVE]) if "watchlist_status" in watchlist.columns else len(watchlist),
        len(watchlist[watchlist["watchlist_status"] == WATCHLIST_STATUS_OBSERVE]) if "watchlist_status" in watchlist.columns else 0,
        interval_seconds,
        bar_minutes,
        stop_time,
    )
    cb_aggregator = FiveMinuteAggregator(
        bar_minutes=bar_minutes,
        asset_type="CB",
        code_attr="bond_code",
        name_attr="bond_name",
        code_key="bond_code",
        name_key="bond_name",
    )
    stock_aggregator = FiveMinuteAggregator(
        bar_minutes=bar_minutes,
        asset_type="STOCK",
        code_attr="stock_code",
        name_attr="stock_name",
        code_key="stock_code",
        name_key="stock_name",
        identity_attr="pair_key",
    )
    stock_quote_provider = WatchlistStockQuoteProvider(bar_minutes=bar_minutes)
    metrics = {}
    last_metrics_refresh = 0.0
    last_wait_log_bucket = None

    while True:
        market_dt = datetime.now(MARKET_TZ)
        if is_after_stop_time(market_dt, stop_time):
            LOGGER.info("Stop time reached stop_time=%s", stop_time)
            break

        if not is_trading_session(market_dt):
            wait_bucket = int(time.monotonic() // 300)
            if wait_bucket != last_wait_log_bucket:
                LOGGER.info("Outside trading session, waiting now=%s", market_dt.strftime("%H:%M:%S"))
                last_wait_log_bucket = wait_bucket
            time.sleep(interval_seconds)
            continue

        now = time.monotonic()
        if not metrics or now - last_metrics_refresh >= metrics_refresh_seconds:
            try:
                metrics = fetch_realtime_cb_metrics(watchlist)
                last_metrics_refresh = now
                LOGGER.info("Refreshed realtime CB metrics rows=%s", len(metrics))
            except Exception as exc:
                LOGGER.warning("Failed to refresh realtime CB metrics: %s", exc)

        try:
            quotes = fetch_spot_quotes(watchlist)
        except Exception as exc:
            LOGGER.warning("Failed to fetch spot quotes: %s", exc)
            time.sleep(interval_seconds)
            continue

        LOGGER.info("Fetched spot quotes rows=%s", len(quotes))
        completed = cb_aggregator.flush_completed(market_now())
        if not quotes.empty:
            LOGGER.debug("\n%s", quotes.to_string(index=False))
            completed.extend(cb_aggregator.update(quotes))
        if completed:
            inserted = insert_5m_bars(conn, completed, metrics)
            LOGGER.info("Inserted completed CB %sm bars rows=%s", bar_minutes, inserted)

        try:
            stock_quotes = stock_quote_provider.fetch(watchlist)
        except Exception as exc:
            LOGGER.warning("Failed to fetch stock spot quotes: %s", exc)
            time.sleep(interval_seconds)
            continue

        LOGGER.info("Fetched stock spot quotes rows=%s", len(stock_quotes))
        stock_completed = stock_aggregator.flush_completed(market_now())
        if not stock_quotes.empty:
            LOGGER.debug("\n%s", stock_quotes.to_string(index=False))
            stock_completed.extend(stock_aggregator.update(stock_quotes))
        if stock_completed:
            inserted = insert_5m_bars(conn, stock_completed, metrics)
            LOGGER.info("Inserted completed STOCK %sm bars rows=%s", bar_minutes, inserted)

        time.sleep(interval_seconds)

    partial = cb_aggregator.flush_all() + stock_aggregator.flush_all()
    if partial:
        inserted = insert_5m_bars(conn, partial, metrics)
        LOGGER.info("Inserted closing partial bars rows=%s", inserted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the watchlist from local TDengine screening snapshots. "
            "Realtime monitoring will attach to this local watchlist."
        )
    )
    parser.add_argument("--premium-min", type=float, default=20.0)
    parser.add_argument("--premium-max", type=float, default=30.0)
    parser.add_argument("--max-debt-asset-ratio", type=float, default=70.0)
    parser.add_argument("--monitor", action="store_true", help="Run realtime spot polling and local 5-minute aggregation.")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--bar-minutes", type=int, default=5)
    parser.add_argument("--metrics-refresh-seconds", type=float, default=300.0)
    parser.add_argument("--reset-live-data", action="store_true", help="Drop and recreate the intraday live table before continuing.")
    parser.add_argument("--rebuild-watchlist", action="store_true", help="Build a fresh watchlist from the latest local screening snapshot before monitoring.")
    parser.add_argument("--retain-live-days", type=int, default=DEFAULT_LIVE_RETAIN_DAYS, help="Keep this many recent trade dates in the live 5-minute table.")
    parser.add_argument("--watchlist-rebalance-days", type=int, default=DEFAULT_WATCHLIST_REBALANCE_DAYS, help="Refresh the formal ACTIVE watchlist every N local trade days.")
    parser.add_argument("--observation-days", type=int, default=DEFAULT_OBSERVATION_DAYS, help="Keep new candidates in OBSERVE for at least N local trade days before promotion.")
    parser.add_argument("--history-trading-days", type=int, default=45, help="Required 5-minute trading-day coverage before OBSERVE can become ACTIVE.")
    parser.add_argument("--history-sleep-seconds", type=float, default=0.5, help="Cooldown between history backfill requests during promotion checks.")
    parser.add_argument("--prepare-history-on-promotion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-non-trading-day", action="store_true", help="Exit without monitoring when today is not a CN trading day.")
    parser.add_argument("--stop-time", default="15:05", help="Beijing time HH:MM when the monitor exits and flushes the final partial bar.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.skip_non_trading_day and not is_cn_trade_day():
        LOGGER.info("Skip realtime monitor because today is not a CN trading day")
        return

    conn = get_conn(use_db=True)
    try:
        if args.reset_live_data:
            reset_live_data(conn)
        else:
            purge_old_live_data(conn, retain_days=args.retain_live_days)

        if args.monitor:
            if args.reset_live_data or args.rebuild_watchlist:
                watchlist = build_watchlist_from_latest_screening(conn, args)
            else:
                watchlist = load_latest_watchlist(conn)
                if watchlist.empty:
                    watchlist = build_watchlist_from_latest_screening(conn, args)

            run_realtime_monitor(
                conn,
                watchlist=watchlist,
                interval_seconds=args.interval_seconds,
                bar_minutes=args.bar_minutes,
                metrics_refresh_seconds=args.metrics_refresh_seconds,
                stop_time=args.stop_time,
            )
            return

        build_watchlist_from_latest_screening(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
