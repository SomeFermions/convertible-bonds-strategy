import argparse
import csv
import logging
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd

from cb_universe import get_cb_universe, normalize_bond_code, to_ak_bond_symbol
from config import MAX_CONSECUTIVE_FAILURES, RETRY_TIMES, SLEEP_SECONDS, SOURCE, USE_PROXY_FOR_MARKET_DATA
from db import get_conn
from market_calendar import MARKET_TZ, is_cn_trade_day


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
REQUIRED_DAILY_COLUMNS = ["ts", "open", "high", "low", "close", "volume", "amount"]


def disable_proxy_if_needed() -> None:
    if USE_PROXY_FOR_MARKET_DATA:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)


def tdengine_table_name(bond_code: str) -> str:
    return f"cb_bar_1d_{bond_code}"


def sql_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    escaped = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def sql_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    return str(float(value))


def to_stock_secucode(stock_code: str) -> str | None:
    code = str(stock_code).strip().zfill(6)
    if code.startswith(("60", "68", "90")):
        return f"{code}.SH"
    if code.startswith(("00", "30", "20")):
        return f"{code}.SZ"
    return None


def normalize_percent(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("%", "").strip()
    result = pd.to_numeric(text, errors="coerce")
    if pd.isna(result):
        return None
    return float(result)


def parse_symbol_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def load_failed_symbols(path: str | None) -> set[str]:
    if not path:
        return set()

    failed_path = Path(path)
    if not failed_path.exists():
        raise FileNotFoundError(f"Failed-symbol file does not exist: {failed_path}")

    symbols = set()
    with failed_path.open(newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        if "," not in sample:
            for line in fh:
                item = line.strip()
                if item:
                    symbols.add(item)
            return symbols

        reader = csv.DictReader(fh)
        for row in reader:
            for key in ("bond_code", "ak_symbol"):
                value = (row.get(key) or "").strip()
                if value:
                    symbols.add(value)
    return symbols


def write_failed_rows(failed_rows: list[dict[str, Any]]) -> Path | None:
    if not failed_rows:
        return None

    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"failed_cb_1d_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["run_time_utc", "bond_code", "bond_name", "ak_symbol", "reason"],
        )
        writer.writeheader()
        writer.writerows(failed_rows)
    return path


def cooldown(base_seconds: float = SLEEP_SECONDS, multiplier: float = 1.0) -> None:
    delay = max(0, base_seconds * multiplier) + random.uniform(0.2, 0.8)
    time.sleep(delay)


def normalize_daily_df(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=REQUIRED_DAILY_COLUMNS)

    rename_map = {
        "date": "ts",
        "日期": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw_df.rename(columns=rename_map).copy()
    if "amount" not in df.columns:
        df["amount"] = None

    missing = [col for col in ["ts", "open", "high", "low", "close", "volume"] if col not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing required daily columns {missing}; raw columns={list(raw_df.columns)}")

    df = df[REQUIRED_DAILY_COLUMNS].copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    df["ts"] = df["ts"].dt.normalize()
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return df.reset_index(drop=True)


def fetch_cb_daily(symbol: str, start_date: str, end_date: str, retries: int = RETRY_TIMES) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            raw_df = ak.bond_zh_hs_cov_daily(symbol=symbol)
            bars = normalize_daily_df(raw_df, symbol)
            if bars.empty:
                return bars

            start_ts = pd.to_datetime(start_date, errors="coerce")
            end_ts = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(start_ts):
                bars = bars[bars["ts"] >= start_ts.normalize()]
            if pd.notna(end_ts):
                bars = bars[bars["ts"] <= end_ts.normalize()]
            return bars.reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            LOGGER.warning("%s daily fetch failed attempt %s/%s: %s", symbol, attempt, retries, exc)
            if attempt < retries:
                cooldown(multiplier=attempt)
    raise RuntimeError(f"{symbol} daily fetch failed after {retries} attempts: {last_error}")


def ensure_subtable(conn, bond_code: str, bond_name: str, market: str) -> str:
    table_name = tdengine_table_name(bond_code)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name}
        USING st_cb_bar_1d
        TAGS ({sql_string(bond_code)}, {sql_string(bond_name)}, {sql_string(market)})
        """
    )
    return table_name


def insert_bars(conn, table_name: str, bars: pd.DataFrame) -> int:
    if bars.empty:
        return 0

    values = []
    for row in bars.itertuples(index=False):
        ts = row.ts.strftime("%Y-%m-%d %H:%M:%S")
        amount = "NULL" if pd.isna(row.amount) else row.amount
        values.append(
            "("
            f"{sql_string(ts)}, "
            f"{row.open}, {row.high}, {row.low}, {row.close}, "
            f"{row.volume if pd.notna(row.volume) else 'NULL'}, "
            f"{amount}, "
            f"{sql_string(SOURCE + '_sina_daily')}"
            ")"
        )

    conn.execute(f"INSERT INTO {table_name} VALUES " + " ".join(values))
    return len(values)


def existing_row_count(conn, bond_code: str, start_date: str, end_date: str) -> int:
    table_name = tdengine_table_name(bond_code)
    where = []
    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(start_ts):
        where.append(f"ts >= {sql_string(start_ts.normalize().strftime('%Y-%m-%d %H:%M:%S'))}")
    if pd.notna(end_ts):
        where.append(f"ts <= {sql_string(end_ts.normalize().strftime('%Y-%m-%d %H:%M:%S'))}")

    sql = f"SELECT COUNT(*) FROM {table_name}"
    if where:
        sql += " WHERE " + " AND ".join(where)

    try:
        for row in conn.query(sql):
            return int(row[0])
    except Exception:
        return 0
    return 0


def collect_one_cb(conn, row: pd.Series, start_date: str, end_date: str) -> int:
    bond_code = row["bond_code"]
    bond_name = row.get("bond_name", "")
    market = row["market"]
    symbol = row["ak_symbol"]

    bars = fetch_cb_daily(symbol, start_date=start_date, end_date=end_date)
    if bars.empty:
        LOGGER.info("%s %s returned no daily bars", bond_code, symbol)
        return 0

    table_name = ensure_subtable(conn, bond_code, bond_name, market)
    inserted = insert_bars(conn, table_name, bars)
    LOGGER.info("%s %s inserted daily rows=%s latest=%s", bond_code, symbol, inserted, bars["ts"].max())
    return inserted


def collect_all_cb_daily(
    limit: int | None = None,
    start_date: str = "1990-01-01",
    end_date: str = "2222-01-01",
    sleep_seconds: float = SLEEP_SECONDS,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    symbols: set[str] | None = None,
    skip_existing: bool = False,
    min_existing_rows: int = 1,
) -> None:
    universe = get_cb_universe()
    if symbols:
        universe = universe[
            universe["bond_code"].isin(symbols)
            | universe["ak_symbol"].isin(symbols)
        ].copy()
    if limit is not None:
        universe = universe.head(limit)

    LOGGER.info(
        "Starting daily collection for %s bonds sleep_seconds=%s skip_existing=%s min_existing_rows=%s",
        len(universe),
        sleep_seconds,
        skip_existing,
        min_existing_rows,
    )
    success = 0
    failed = 0
    skipped = 0
    consecutive_failures = 0
    inserted_total = 0
    failed_rows = []

    conn = get_conn(use_db=True)
    try:
        for row in universe.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            if skip_existing:
                existing = existing_row_count(conn, row_series["bond_code"], start_date, end_date)
                if existing >= min_existing_rows:
                    skipped += 1
                    LOGGER.info(
                        "Skip existing daily bond_code=%s ak_symbol=%s rows=%s",
                        row_series["bond_code"],
                        row_series["ak_symbol"],
                        existing,
                    )
                    continue

            try:
                inserted_total += collect_one_cb(conn, row_series, start_date, end_date)
                success += 1
                consecutive_failures = 0
            except Exception as exc:
                failed += 1
                consecutive_failures += 1
                LOGGER.exception(
                    "Daily collect failed bond_code=%s ak_symbol=%s reason=%s",
                    row_series.get("bond_code"),
                    row_series.get("ak_symbol"),
                    exc,
                )
                failed_rows.append(
                    {
                        "run_time_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                        "bond_code": row_series.get("bond_code"),
                        "bond_name": row_series.get("bond_name"),
                        "ak_symbol": row_series.get("ak_symbol"),
                        "reason": str(exc),
                    }
                )
                if consecutive_failures >= max_consecutive_failures:
                    LOGGER.error(
                        "Stop daily collection after %s consecutive failures; data source may be throttling or unreachable",
                        consecutive_failures,
                    )
                    break
            cooldown(base_seconds=sleep_seconds)
    finally:
        conn.close()

    failed_path = write_failed_rows(failed_rows)
    if failed_path:
        LOGGER.info("Failed daily symbols written to %s", failed_path)
    LOGGER.info(
        "Daily collection done success=%s failed=%s skipped=%s inserted_total=%s",
        success,
        failed,
        skipped,
        inserted_total,
    )


def get_cb_screening_base(
    premium_min: float | None = None,
    premium_max: float | None = None,
    listed_only: bool = True,
) -> pd.DataFrame:
    disable_proxy_if_needed()
    df = ak.bond_zh_cov()
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["bond_code"] = df["债券代码"].apply(normalize_bond_code)
    out["bond_name"] = df["债券简称"].astype(str).str.strip()
    out["ak_symbol"] = out["bond_code"].apply(to_ak_bond_symbol)
    out["stock_code"] = df["正股代码"].astype(str).str.extract(r"(\d{6})", expand=False)
    out["stock_name"] = df["正股简称"].astype(str).str.strip()
    out["stock_secucode"] = out["stock_code"].apply(to_stock_secucode)
    out["bond_price"] = pd.to_numeric(df["债现价"], errors="coerce")
    out["stock_price"] = pd.to_numeric(df["正股价"], errors="coerce")
    out["conversion_price"] = pd.to_numeric(df["转股价"], errors="coerce")
    out["conversion_value"] = pd.to_numeric(df["转股价值"], errors="coerce")
    out["premium_rate"] = df["转股溢价率"].apply(normalize_percent)
    out["listed_date"] = pd.to_datetime(df["上市时间"], errors="coerce")

    try:
        ths = ak.bond_zh_cov_info_ths()
        if ths is not None and not ths.empty:
            maturity = pd.DataFrame()
            maturity["bond_code"] = ths["债券代码"].apply(normalize_bond_code)
            maturity["maturity_date"] = pd.to_datetime(ths["到期时间"], errors="coerce")
            maturity = maturity.dropna(subset=["bond_code"]).drop_duplicates(subset=["bond_code"])
            out = out.merge(maturity, on="bond_code", how="left")
    except Exception as exc:
        LOGGER.warning("failed to fetch THS maturity dates: %s", exc)
    if "maturity_date" not in out.columns:
        out["maturity_date"] = pd.NaT
    today = pd.Timestamp.today().normalize()
    out["days_to_maturity"] = (out["maturity_date"] - today).dt.days

    out = out.dropna(subset=["bond_code", "ak_symbol", "stock_code", "stock_secucode"])
    out = out.drop_duplicates(subset=["bond_code"])

    if listed_only:
        out = out[out["listed_date"].notna() & (out["listed_date"] <= today)]
    if premium_min is not None:
        out = out[out["premium_rate"] >= premium_min]
    if premium_max is not None:
        out = out[out["premium_rate"] <= premium_max]

    return out.reset_index(drop=True)


def get_latest_financial_row(stock_secucode: str, annual_only: bool = True) -> pd.Series | None:
    df = ak.stock_financial_analysis_indicator_em(symbol=stock_secucode, indicator="按报告期")
    if df is None or df.empty:
        return None

    df = df.copy()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
    df = df.dropna(subset=["REPORT_DATE"]).sort_values("REPORT_DATE", ascending=False)
    if annual_only and "REPORT_TYPE" in df.columns:
        annual = df[df["REPORT_TYPE"].astype(str).str.contains("年报", na=False)]
        if not annual.empty:
            df = annual
    if df.empty:
        return None
    return df.iloc[0]


def extract_financial_metrics(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {
            "report_date": pd.NaT,
            "debt_asset_ratio": None,
            "operating_cash_flow": None,
            "free_cash_flow": None,
        }

    operating_cash_flow = pd.to_numeric(row.get("MGJYXJJE"), errors="coerce")
    if pd.isna(operating_cash_flow):
        operating_cash_flow = pd.to_numeric(row.get("NCO_OP"), errors="coerce")

    free_cash_flow = pd.to_numeric(row.get("FCFF_BACK"), errors="coerce")
    if pd.isna(free_cash_flow):
        free_cash_flow = pd.to_numeric(row.get("FCFF_FORWARD"), errors="coerce")

    return {
        "report_date": row.get("REPORT_DATE"),
        "debt_asset_ratio": pd.to_numeric(row.get("ZCFZL"), errors="coerce"),
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
    }


def collect_screening_snapshot(
    conn,
    premium_min: float | None = 15.0,
    premium_max: float | None = 35.0,
    annual_only: bool = True,
    sleep_seconds: float = SLEEP_SECONDS,
    limit: int | None = None,
) -> int:
    base = get_cb_screening_base(
        premium_min=premium_min,
        premium_max=premium_max,
        listed_only=True,
    )
    if limit is not None:
        base = base.head(limit)

    LOGGER.info("Starting screening snapshot collection rows=%s", len(base))
    rows = []
    for item in base.itertuples(index=False):
        try:
            financial_row = get_latest_financial_row(item.stock_secucode, annual_only=annual_only)
            row = item._asdict()
            row.update(extract_financial_metrics(financial_row))
            rows.append(row)
            LOGGER.info(
                "screening %s %s premium=%s debt=%s ocf=%s fcf=%s",
                item.bond_code,
                item.bond_name,
                item.premium_rate,
                row.get("debt_asset_ratio"),
                row.get("operating_cash_flow"),
                row.get("free_cash_flow"),
            )
        except Exception as exc:
            LOGGER.warning("screening fetch failed bond_code=%s stock=%s reason=%s", item.bond_code, item.stock_secucode, exc)
        cooldown(base_seconds=sleep_seconds)

    if not rows:
        return 0

    snapshot_id = datetime.now(MARKET_TZ).strftime("%Y%m%d%H%M%S")
    base_ts = pd.Timestamp.now(tz=MARKET_TZ).floor("s").tz_localize(None)
    values = []
    for idx, row in enumerate(rows):
        row_ts = (base_ts + pd.Timedelta(milliseconds=idx)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        listed_date = row.get("listed_date")
        maturity_date = row.get("maturity_date")
        report_date = row.get("report_date")
        listed_date_sql = None if pd.isna(listed_date) else pd.Timestamp(listed_date).strftime("%Y-%m-%d %H:%M:%S")
        maturity_date_sql = None if pd.isna(maturity_date) else pd.Timestamp(maturity_date).strftime("%Y-%m-%d %H:%M:%S")
        report_date_sql = None if pd.isna(report_date) else pd.Timestamp(report_date).strftime("%Y-%m-%d %H:%M:%S")
        values.append(
            "("
            f"{sql_string(row_ts)}, {sql_string(snapshot_id)}, "
            f"{sql_string(row.get('bond_code'))}, {sql_string(row.get('bond_name'))}, {sql_string(row.get('ak_symbol'))}, "
            f"{sql_number(row.get('bond_price'))}, {sql_number(row.get('premium_rate'))}, "
            f"{sql_number(row.get('conversion_value'))}, {sql_number(row.get('conversion_price'))}, "
            f"{sql_string(row.get('stock_code'))}, {sql_string(row.get('stock_name'))}, {sql_number(row.get('stock_price'))}, "
            f"{sql_string(listed_date_sql)}, {sql_string(maturity_date_sql)}, "
            f"{int(row.get('days_to_maturity')) if pd.notna(row.get('days_to_maturity')) else 'NULL'}, "
            f"{sql_string(report_date_sql)}, "
            f"{sql_number(row.get('debt_asset_ratio'))}, {sql_number(row.get('operating_cash_flow'))}, "
            f"{sql_number(row.get('free_cash_flow'))}, {sql_string(SOURCE + '_screening')}"
            ")"
        )

    conn.execute(
        """
        INSERT INTO cb_screening_daily (
            ts, snapshot_id, bond_code, bond_name, ak_symbol,
            bond_price, premium_rate, conversion_value, conversion_price,
            stock_code, stock_name, stock_price, listed_date, maturity_date, days_to_maturity, report_date,
            debt_asset_ratio, operating_cash_flow, free_cash_flow, source
        ) VALUES
        """
        + " ".join(values)
    )
    LOGGER.info("Screening snapshot inserted rows=%s", len(values))
    return len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect full-market convertible bond daily bars into TDengine. "
            "The filename is kept for compatibility; this script now writes st_cb_bar_1d."
        )
    )
    parser.add_argument("--limit", type=int, default=None, help="Collect only the first N bonds for smoke testing.")
    parser.add_argument("--start-date", default="1990-01-01", help="Daily start date.")
    parser.add_argument("--end-date", default="2222-01-01", help="Daily end date.")
    parser.add_argument("--today", action="store_true", help="Collect only today's Beijing-date daily bar.")
    parser.add_argument("--skip-non-trading-day", action="store_true", help="Exit without collecting when the target date is not a CN trading day.")
    parser.add_argument("--symbols", default=None, help="Comma-separated bond_code or ak_symbol list to collect.")
    parser.add_argument("--failed-file", default=None, help="CSV or plain-text failed-symbol file to retry.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip symbols that already have enough rows in TDengine.")
    parser.add_argument("--min-existing-rows", type=int, default=1, help="Rows required for --skip-existing.")
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS, help="Base cooldown between symbols.")
    parser.add_argument("--skip-screening", action="store_true", help="Do not collect daily screening snapshot fields.")
    parser.add_argument("--screening-only", action="store_true", help="Only collect daily screening snapshot fields.")
    parser.add_argument("--screening-premium-min", type=float, default=15.0)
    parser.add_argument("--screening-premium-max", type=float, default=35.0)
    parser.add_argument("--screening-limit", type=int, default=None, help="Limit screening snapshot rows for smoke testing.")
    parser.add_argument("--latest-report", action="store_true", help="Use latest report instead of latest annual report for screening metrics.")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=MAX_CONSECUTIVE_FAILURES,
        help="Stop after this many consecutive symbol failures.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if args.today:
        today = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")
        args.start_date = today
        args.end_date = today
        LOGGER.info("Using today's Beijing date for daily collection: %s", today)

    if args.skip_non_trading_day and not is_cn_trade_day(args.end_date):
        LOGGER.info("Skip daily collection because %s is not a CN trading day", args.end_date)
        return

    symbols = parse_symbol_list(args.symbols) | load_failed_symbols(args.failed_file)
    if not args.screening_only:
        collect_all_cb_daily(
            limit=args.limit,
            start_date=args.start_date,
            end_date=args.end_date,
            sleep_seconds=args.sleep_seconds,
            max_consecutive_failures=args.max_consecutive_failures,
            symbols=symbols,
            skip_existing=args.skip_existing,
            min_existing_rows=args.min_existing_rows,
        )

    if not args.skip_screening:
        conn = get_conn(use_db=True)
        try:
            collect_screening_snapshot(
                conn,
                premium_min=args.screening_premium_min,
                premium_max=args.screening_premium_max,
                annual_only=not args.latest_report,
                sleep_seconds=args.sleep_seconds,
                limit=args.screening_limit,
            )
        finally:
            conn.close()


if __name__ == "__main__":
    main()
