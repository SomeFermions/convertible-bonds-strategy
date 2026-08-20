from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# TDengine's Python native client shifts TIMESTAMP values when it is loaded
# under Asia/Shanghai. Keep the process timezone UTC and use MARKET_TZ
# explicitly for market-time decisions.
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()

import pandas as pd

from intraday_factors.config import load_config, load_schema
from intraday_factors.data_source import (
    get_connection,
    insert_alert_rows,
    insert_factor_rows,
    insert_job_run,
    query_to_dataframe,
    sql_string,
)
from intraday_factors.engine import FactorEngine
from intraday_factors.history_tools import (
    DEFAULT_HISTORY_TRADING_DAYS,
    backfill_pool_history,
    diff_live_backfill,
    prepare_active_history,
    readiness_check,
    summarize_latest_diff_status,
)
from intraday_factors.trading_session import MARKET_TZ, completed_bar_cutoff, ensure_market_naive, is_session_bar


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCK_FILE = "/tmp/cb_intraday_factors.lock"
JOB_NAME = "intraday_factors_incremental"
SUMMARY_COLUMNS = [
    "bar_start",
    "bond_code",
    "stock_code",
    "pool_state",
    "alert_state",
    "signal_type",
    "setup_score",
    "trigger_score",
    "signal_state",
    "signal_score",
    "residual_z",
    "stock_mom_30m_z",
    "cb_mom_15m_z",
    "premium_slot_z",
    "factor_coverage",
    "session_bar_role",
    "is_actionable_bar",
    "volatility_gate_pass",
    "cooldown_suppressed",
]


def market_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None)


def ts_or_none(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_localize(None)


def ts_text(value: Any) -> str | None:
    ts = ts_or_none(value)
    return None if ts is None else ts.strftime("%Y-%m-%d %H:%M:%S")


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return ts_text(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def query_scalar_ts(conn, sql: str, column: str = "bar_start") -> pd.Timestamp | None:
    df = query_to_dataframe(conn, sql)
    if df.empty or column not in df.columns:
        return None
    return ts_or_none(df.loc[0, column])


def latest_live_bars(conn, schema: dict[str, Any]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    table = schema["tables"]["live_5m"]
    latest_cb = query_scalar_ts(
        conn,
        f"SELECT MAX(bar_start) AS bar_start FROM {table} WHERE asset_type = 'CB'",
    )
    latest_stock = query_scalar_ts(
        conn,
        f"SELECT MAX(bar_start) AS bar_start FROM {table} WHERE asset_type = 'STOCK'",
    )
    return latest_cb, latest_stock


def latest_written_bar(conn, table: str, factor_version: str) -> pd.Timestamp | None:
    return query_scalar_ts(
        conn,
        f"""
        SELECT MAX(bar_start) AS bar_start
        FROM {table}
        WHERE factor_version = {sql_string(factor_version)}
        """,
    )


def effective_completed_bar(
    requested_asof: pd.Timestamp,
    latest_cb_bar: pd.Timestamp | None,
    latest_stock_bar: pd.Timestamp | None,
    bar_minutes: int,
) -> pd.Timestamp | None:
    if latest_cb_bar is None or latest_stock_bar is None:
        return None
    cutoff = completed_bar_cutoff(requested_asof, bar_minutes)
    effective = min(cutoff, latest_cb_bar, latest_stock_bar)
    while effective is not None and not is_session_bar(effective, bar_minutes):
        effective = effective - pd.Timedelta(minutes=bar_minutes)
    return effective


def completed_until(latest_factor_bar: pd.Timestamp | None, latest_alert_bar: pd.Timestamp | None) -> pd.Timestamp | None:
    if latest_factor_bar is None or latest_alert_bar is None:
        return None
    return min(latest_factor_bar, latest_alert_bar)


def select_new_rows(result: pd.DataFrame, completed_through: pd.Timestamp | None, effective_asof: pd.Timestamp) -> pd.DataFrame:
    if result.empty:
        return result
    out = result.copy()
    out["bar_start"] = pd.to_datetime(out["bar_start"], errors="coerce")
    out = out[out["bar_start"].notna()].copy()
    out = out[out["bar_start"] <= effective_asof].copy()
    if completed_through is not None:
        out = out[out["bar_start"] > completed_through].copy()
    return out.sort_values(["bar_start", "bond_code"]).reset_index(drop=True)


def count_missing_stock_bonds(rows: pd.DataFrame) -> int:
    if rows.empty or "missing_stock_bar" not in rows.columns:
        return 0
    missing = rows[pd.Series(rows["missing_stock_bar"]).fillna(False).astype(bool)]
    return int(missing["bond_code"].dropna().astype(str).nunique()) if "bond_code" in missing.columns else 0


def write_factor_alert_rows(
    conn,
    rows: pd.DataFrame,
    schema: dict[str, Any],
    factor_version: str,
) -> tuple[int, int, int, int, str | None]:
    if rows.empty:
        return 0, 0, 0, 0, None

    factor_rows = 0
    alert_rows = 0
    skipped_bonds: set[str] = set()
    errors: list[str] = []

    try:
        factor_rows = insert_factor_rows(conn, rows, schema["tables"]["factor_snapshot"], factor_version)
    except Exception as exc:
        errors.append(f"factor bulk insert failed: {exc}")
        for bond_code, part in rows.groupby(rows["bond_code"].astype(str), dropna=False):
            try:
                factor_rows += insert_factor_rows(conn, part, schema["tables"]["factor_snapshot"], factor_version)
            except Exception as bond_exc:
                skipped_bonds.add(str(bond_code))
                errors.append(f"factor insert failed bond={bond_code}: {bond_exc}")

    try:
        alert_rows = insert_alert_rows(conn, rows, schema["tables"]["alert_table"], factor_version)
    except Exception as exc:
        errors.append(f"alert bulk insert failed: {exc}")
        for bond_code, part in rows.groupby(rows["bond_code"].astype(str), dropna=False):
            try:
                alert_rows += insert_alert_rows(conn, part, schema["tables"]["alert_table"], factor_version)
            except Exception as bond_exc:
                skipped_bonds.add(str(bond_code))
                errors.append(f"alert insert failed bond={bond_code}: {bond_exc}")

    return factor_rows, alert_rows, len(skipped_bonds), len(errors), "; ".join(errors)[:1000] if errors else None


def print_result_tail(result: pd.DataFrame) -> None:
    if result.empty:
        print("No factor rows")
        return
    cols = [col for col in SUMMARY_COLUMNS if col in result.columns]
    print(result[cols].tail(20).to_string(index=False))


def latest_job_run(conn) -> dict[str, Any]:
    try:
        df = query_to_dataframe(conn, "SELECT * FROM cb_intraday_job_runs ORDER BY ts DESC LIMIT 1")
    except Exception:
        return {}
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {key: (ts_text(value) if "time" in key or key.endswith("_at") or key.endswith("_bar") or key == "ts" else value) for key, value in row.items()}


def insert_run_record(row: dict[str, Any]) -> None:
    conn = get_connection()
    try:
        insert_job_run(conn, row)
    finally:
        conn.close()


def run_incremental(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    run_id = args.run_id or uuid.uuid4().hex
    started_at = market_now()
    requested_asof = ensure_market_naive(args.asof) if args.asof else market_now()
    factor_version = str(config["factor_version"])
    log_path = args.log_path or os.getenv("CB_JOB_LOG_PATH")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "job_name": args.job_name,
        "mode": "incremental",
        "factor_version": factor_version,
        "requested_asof": requested_asof,
        "effective_asof": None,
        "latest_cb_bar": None,
        "latest_stock_bar": None,
        "latest_factor_bar": None,
        "latest_alert_bar": None,
        "started_at": started_at,
        "finished_at": None,
        "status": "FAILED",
        "processed_bars": 0,
        "processed_bonds": 0,
        "factor_rows_upserted": 0,
        "alert_rows_upserted": 0,
        "skipped_bonds": 0,
        "error_count": 0,
        "error_message": None,
        "log_path": log_path,
        "pool_scope": "ACTIVE,ACTIVE_MANUAL",
        "history_coverage_min_days": None,
        "live_backfill_diff_status": None,
    }

    conn = get_connection()
    try:
        latest_cb_bar, latest_stock_bar = latest_live_bars(conn, schema)
        latest_factor_bar = latest_written_bar(conn, schema["tables"]["factor_snapshot"], factor_version)
        latest_alert_bar = latest_written_bar(conn, schema["tables"]["alert_table"], factor_version)
        effective_asof = effective_completed_bar(requested_asof, latest_cb_bar, latest_stock_bar, int(config["bar_minutes"]))
        summary.update(
            {
                "latest_cb_bar": latest_cb_bar,
                "latest_stock_bar": latest_stock_bar,
                "latest_factor_bar": latest_factor_bar,
                "latest_alert_bar": latest_alert_bar,
                "effective_asof": effective_asof,
            }
        )

        if effective_asof is None:
            summary["status"] = "FAILED"
            summary["error_count"] = 1
            summary["error_message"] = "No calculable completed bar because CB or STOCK live bars are missing"
            return 1

        done_through = completed_until(latest_factor_bar, latest_alert_bar)
        if done_through is not None and done_through >= effective_asof:
            summary["status"] = "NO_NEW_BAR"
            LOGGER.info("no_new_completed_bar latest_factor=%s latest_alert=%s effective_asof=%s", latest_factor_bar, latest_alert_bar, effective_asof)
            return 0

        start = effective_asof - pd.Timedelta(days=args.lookback_days)
        engine = FactorEngine(config, schema)
        result = engine.compute_from_db(start, effective_asof, asof=None, write=False)
        new_rows = select_new_rows(result, done_through, effective_asof)
        if not new_rows.empty:
            new_rows["effective_asof"] = effective_asof
        summary["processed_bars"] = int(new_rows["bar_start"].nunique()) if not new_rows.empty else 0
        summary["processed_bonds"] = int(new_rows["bond_code"].dropna().astype(str).nunique()) if not new_rows.empty else 0
        summary["skipped_bonds"] = count_missing_stock_bonds(new_rows)

        if new_rows.empty:
            summary["status"] = "NO_NEW_BAR"
            summary["error_message"] = "No factor rows computed for new completed bars"
            LOGGER.warning(summary["error_message"])
            return 0

        factor_count = alert_count = write_skipped = write_errors = 0
        write_error_message = None
        if args.write:
            factor_count, alert_count, write_skipped, write_errors, write_error_message = write_factor_alert_rows(
                conn,
                new_rows,
                schema,
                factor_version,
            )
        summary["factor_rows_upserted"] = factor_count
        summary["alert_rows_upserted"] = alert_count
        summary["skipped_bonds"] += write_skipped
        summary["error_count"] = write_errors
        summary["error_message"] = write_error_message
        summary["status"] = "PARTIAL_SUCCESS" if write_errors else "SUCCESS"

        LOGGER.info(
            "incremental_complete run_id=%s effective_asof=%s processed_bars=%s processed_bonds=%s factor_rows=%s alert_rows=%s status=%s",
            run_id,
            effective_asof,
            summary["processed_bars"],
            summary["processed_bonds"],
            factor_count,
            alert_count,
            summary["status"],
        )
        print_result_tail(new_rows[new_rows["bar_start"] == new_rows["bar_start"].max()].copy())
        return 0 if write_errors == 0 else 1
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["error_count"] = int(summary.get("error_count") or 0) + 1
        summary["error_message"] = f"{exc}\n{traceback.format_exc()}"[:1000]
        LOGGER.exception("incremental_failed run_id=%s", run_id)
        return 1
    finally:
        summary["finished_at"] = market_now()
        try:
            insert_job_run(conn, summary)
        except Exception:
            LOGGER.exception("failed_to_record_job_run run_id=%s", run_id)
        conn.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


def run_backfill(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    if not args.start or not args.end:
        raise SystemExit("--mode backfill requires --start and --end")
    engine = FactorEngine(config, schema)
    result = engine.compute_from_db(args.start, args.end, write=args.write)
    print_result_tail(result)
    return 0


def _record_aux_job(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    mode: str,
    status: str,
    started_at: pd.Timestamp,
    finished_at: pd.Timestamp,
    processed_bonds: int = 0,
    skipped_bonds: int = 0,
    error_count: int = 0,
    error_message: str | None = None,
    history_coverage_min_days: int | None = None,
    live_backfill_diff_status: str | None = None,
) -> None:
    row = {
        "run_id": args.run_id or uuid.uuid4().hex,
        "job_name": args.job_name,
        "mode": mode,
        "factor_version": str(config["factor_version"]),
        "requested_asof": ensure_market_naive(args.asof) if args.asof else market_now(),
        "effective_asof": None,
        "latest_cb_bar": None,
        "latest_stock_bar": None,
        "latest_factor_bar": None,
        "latest_alert_bar": None,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "processed_bars": 0,
        "processed_bonds": processed_bonds,
        "factor_rows_upserted": 0,
        "alert_rows_upserted": 0,
        "skipped_bonds": skipped_bonds,
        "error_count": error_count,
        "error_message": error_message,
        "log_path": args.log_path or os.getenv("CB_JOB_LOG_PATH"),
        "pool_scope": args.pool,
        "history_coverage_min_days": history_coverage_min_days,
        "live_backfill_diff_status": live_backfill_diff_status,
    }
    insert_run_record(row)


def run_backfill_5m_history(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    started_at = market_now()
    asof_date = args.asof_date or market_now().strftime("%Y-%m-%d")
    conn = get_connection()
    reports: list[dict[str, Any]] = []
    try:
        reports = backfill_pool_history(
            conn,
            pool=args.pool,
            trading_days=args.trading_days,
            asof_date=asof_date,
            live_table=schema["tables"]["live_5m"],
            sleep_seconds=args.history_sleep_seconds,
        )
    finally:
        conn.close()

    error_count = sum(1 for item in reports if item.get("error"))
    ready_reports = [item for item in reports if item.get("ready_for_active")]
    min_days = min((int(item.get("history_days_available") or 0) for item in reports), default=0)
    status = "SUCCESS" if error_count == 0 else ("PARTIAL_SUCCESS" if reports else "FAILED")
    _record_aux_job(
        args=args,
        config=config,
        mode="backfill-5m-history",
        status=status,
        started_at=started_at,
        finished_at=market_now(),
        processed_bonds=len(reports),
        skipped_bonds=len(reports) - len(ready_reports),
        error_count=error_count,
        error_message=None if error_count == 0 else "some symbols failed 5m history backfill",
        history_coverage_min_days=min_days,
    )
    output = {
        "mode": "backfill-5m-history",
        "pool": args.pool,
        "asof_date": str(asof_date),
        "trading_days": args.trading_days,
        "processed_bonds": len(reports),
        "error_count": error_count,
        "history_coverage_min_days": min_days,
        "coverage_report": reports,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=json_default))
    return 0 if error_count == 0 else 1


def run_prepare_active_history(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    if not args.bond_code:
        raise SystemExit("--mode prepare-active-history requires --bond-code")
    started_at = market_now()
    asof_date = args.asof_date or market_now().strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        report = prepare_active_history(
            conn,
            bond_code=args.bond_code,
            trading_days=args.trading_days,
            asof_date=asof_date,
            live_table=schema["tables"]["live_5m"],
            sleep_seconds=args.history_sleep_seconds,
        )
    finally:
        conn.close()
    status = "SUCCESS" if report.get("ready_for_active") else "PARTIAL_SUCCESS"
    _record_aux_job(
        args=args,
        config=config,
        mode="prepare-active-history",
        status=status,
        started_at=started_at,
        finished_at=market_now(),
        processed_bonds=1,
        skipped_bonds=0 if report.get("ready_for_active") else 1,
        error_count=1 if report.get("error") else 0,
        error_message=report.get("error") or report.get("failure_reason"),
        history_coverage_min_days=int(report.get("history_days_available") or 0),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_default))
    return 0 if report.get("ready_for_active") else 2


def run_readiness_check(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    check_date = args.date or args.asof_date or market_now().strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        output, code = readiness_check(
            conn,
            check_date=check_date,
            factor_version=str(config["factor_version"]),
            trading_days=args.trading_days,
            live_table=schema["tables"]["live_5m"],
        )
    finally:
        conn.close()
    print(json.dumps(output, ensure_ascii=False, indent=2, default=json_default))
    return code


def run_diff_live_backfill(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    if not args.date:
        raise SystemExit("--mode diff-live-backfill requires --date")
    started_at = market_now()
    factor_version = args.factor_version or str(config["factor_version"])
    config = dict(config)
    config["factor_version"] = factor_version
    conn = get_connection()
    try:
        summary, diff_rows = diff_live_backfill(
            conn,
            diff_date=args.date,
            factor_version=factor_version,
            config=config,
            schema=schema,
            write=args.write,
        )
    finally:
        conn.close()
    _record_aux_job(
        args=args,
        config=config,
        mode="diff-live-backfill",
        status="SUCCESS" if summary["status"] == "OK" else "PARTIAL_SUCCESS",
        started_at=started_at,
        finished_at=market_now(),
        processed_bonds=0,
        skipped_bonds=0,
        error_count=0 if summary["status"] == "OK" else int(summary["diff_rows"]),
        error_message=None if summary["status"] == "OK" else "live/backfill diff found",
        live_backfill_diff_status="OK" if summary["status"] == "OK" else "WARN",
    )
    print(json.dumps({"summary": summary, "diff_sample": diff_rows[:20]}, ensure_ascii=False, indent=2, default=json_default))
    return 0 if summary["status"] == "OK" else 1


def lock_status(lock_file: str) -> dict[str, Any]:
    path = Path(lock_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
            locked = False
        except BlockingIOError:
            locked = True
    return {"path": str(path), "exists": path.exists(), "locked": locked}


def cron_status(script_path: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(["crontab", "-l"], text=True, capture_output=True, check=False)
    except Exception as exc:
        return {"configured": False, "error": str(exc), "lines": []}
    lines = [line for line in proc.stdout.splitlines() if script_path in line and not line.strip().startswith("#")]
    return {"configured": bool(lines), "error": proc.stderr.strip() if proc.returncode else None, "lines": lines}


def recent_error_from_log(log_path: str | None) -> str | None:
    if not log_path or not Path(log_path).exists():
        return None
    lines = Path(log_path).read_text(errors="ignore").splitlines()[-400:]
    for line in reversed(lines):
        if any(token in line for token in ("ERROR", "Traceback", "FAILED", "Exception")):
            return line[-1000:]
    return None


def minutes_delay(live_bar: pd.Timestamp | None, downstream_bar: pd.Timestamp | None) -> float | None:
    if live_bar is None or downstream_bar is None:
        return None
    return float((live_bar - downstream_bar).total_seconds() / 60.0)


def run_healthcheck(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    factor_version = str(config["factor_version"])
    log_path = args.log_path or os.getenv("CB_JOB_LOG_PATH") or str(PROJECT_DIR / "logs" / f"intraday_factors_{market_now().date()}.log")
    conn = get_connection()
    try:
        latest_cb_bar, latest_stock_bar = latest_live_bars(conn, schema)
        latest_factor_bar = latest_written_bar(conn, schema["tables"]["factor_snapshot"], factor_version)
        latest_alert_bar = latest_written_bar(conn, schema["tables"]["alert_table"], factor_version)
        last_run = latest_job_run(conn)
        diff_status = summarize_latest_diff_status(conn, schema["tables"].get("live_backfill_diff", "cb_intraday_live_backfill_diff"))
        try:
            factor_cov = query_to_dataframe(
                conn,
                f"""
                SELECT AVG(factor_coverage) AS factor_coverage_mean
                FROM {schema["tables"]["factor_snapshot"]}
                WHERE factor_version = {sql_string(factor_version)}
                  AND bar_start = {sql_string(ts_text(latest_factor_bar))}
                """,
            )
            factor_coverage_mean = None if factor_cov.empty or pd.isna(factor_cov.loc[0, "factor_coverage_mean"]) else float(factor_cov.loc[0, "factor_coverage_mean"])
        except Exception:
            factor_coverage_mean = None
        try:
            ready_output, ready_code = readiness_check(
                conn,
                args.date or market_now().strftime("%Y-%m-%d"),
                factor_version=factor_version,
                trading_days=args.trading_days,
                live_table=schema["tables"]["live_5m"],
            )
        except Exception as exc:
            ready_output, ready_code = {"error": str(exc), "ready_for_open": False}, 2
    finally:
        conn.close()

    factor_delay = minutes_delay(latest_cb_bar, latest_factor_bar)
    alert_delay = minutes_delay(latest_cb_bar, latest_alert_bar)
    stale = (
        factor_delay is None
        or alert_delay is None
        or factor_delay > args.health_max_delay_minutes
        or alert_delay > args.health_max_delay_minutes
    )
    output = {
        "system_time_market": ts_text(market_now()),
        "system_time_db_process_tz": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": market_now().strftime("%Y-%m-%d"),
        "factor_version": factor_version,
        "latest_cb_live_bar": {"db_raw": ts_text(latest_cb_bar), "market_time": ts_text(latest_cb_bar)},
        "latest_stock_live_bar": {"db_raw": ts_text(latest_stock_bar), "market_time": ts_text(latest_stock_bar)},
        "latest_factor_snapshot_bar": {"db_raw": ts_text(latest_factor_bar), "market_time": ts_text(latest_factor_bar)},
        "latest_alert_bar": {"db_raw": ts_text(latest_alert_bar), "market_time": ts_text(latest_alert_bar)},
        "cb_live_to_factor_delay_minutes": factor_delay,
        "cb_live_to_alert_delay_minutes": alert_delay,
        "factor_coverage_mean_latest_bar": factor_coverage_mean,
        "history_coverage_ready": ready_code == 0,
        "readiness_summary": {
            "active_count": ready_output.get("active_count"),
            "observe_count": ready_output.get("observe_count"),
            "shadow_research_count": ready_output.get("shadow_research_count"),
            "active_pending_history_count": ready_output.get("active_pending_history_count"),
            "ready_for_open": ready_output.get("ready_for_open"),
            "active_failures": ready_output.get("active_failures"),
        },
        "live_backfill_diff_status": diff_status,
        "last_job_run": last_run,
        "recent_error_log": recent_error_from_log(log_path),
        "cron": cron_status(str(PROJECT_DIR / "scripts" / "run_intraday_factors_incremental.sh")),
        "lock": lock_status(args.lock_file),
        "stale_alert_pipeline": stale,
        "status": "FAILED" if stale else "OK",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=json_default))
    return 2 if stale else 0


def record_locked(args: argparse.Namespace, config: dict[str, Any], schema: dict[str, Any]) -> int:
    now = market_now()
    conn = get_connection()
    try:
        latest_cb_bar, latest_stock_bar = latest_live_bars(conn, schema)
        latest_factor_bar = latest_written_bar(conn, schema["tables"]["factor_snapshot"], str(config["factor_version"]))
        latest_alert_bar = latest_written_bar(conn, schema["tables"]["alert_table"], str(config["factor_version"]))
        row = {
            "run_id": args.run_id or uuid.uuid4().hex,
            "job_name": args.job_name,
            "mode": "incremental",
            "factor_version": str(config["factor_version"]),
            "requested_asof": now,
            "effective_asof": None,
            "latest_cb_bar": latest_cb_bar,
            "latest_stock_bar": latest_stock_bar,
            "latest_factor_bar": latest_factor_bar,
            "latest_alert_bar": latest_alert_bar,
            "started_at": now,
            "finished_at": now,
            "status": "LOCKED_SKIPPED",
            "processed_bars": 0,
            "processed_bonds": 0,
            "factor_rows_upserted": 0,
            "alert_rows_upserted": 0,
            "skipped_bonds": 0,
            "error_count": 0,
            "error_message": "Another intraday factor job is already running",
            "log_path": args.log_path or os.getenv("CB_JOB_LOG_PATH"),
            "pool_scope": "ACTIVE,ACTIVE_MANUAL",
            "history_coverage_min_days": None,
            "live_backfill_diff_status": None,
        }
        insert_job_run(conn, row)
    finally:
        conn.close()
    print("LOCKED_SKIPPED")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convertible bond intraday factor monitor V0.")
    parser.add_argument(
        "--mode",
        choices=[
            "backfill",
            "incremental",
            "healthcheck",
            "record-locked",
            "backfill-5m-history",
            "prepare-active-history",
            "readiness-check",
            "diff-live-backfill",
        ],
        required=True,
    )
    parser.add_argument("--start", default=None, help="Backfill start YYYY-MM-DD or timestamp.")
    parser.add_argument("--end", default=None, help="Backfill end YYYY-MM-DD or timestamp.")
    parser.add_argument("--asof", default=None, help="Incremental as-of timestamp. Defaults to Asia/Shanghai now.")
    parser.add_argument("--asof-date", default=None, help="History/readiness as-of date YYYY-MM-DD.")
    parser.add_argument("--date", default=None, help="Business date for readiness or live/backfill diff.")
    parser.add_argument("--bond-code", default=None, help="Single bond code for prepare-active-history.")
    parser.add_argument("--pool", default="active,observe", help="Pool scope for history backfill, e.g. active,observe,shadow_research.")
    parser.add_argument("--trading-days", type=int, default=DEFAULT_HISTORY_TRADING_DAYS)
    parser.add_argument("--history-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--factor-version", default=None, help="Override factor version for diff-live-backfill.")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--config", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--write", action="store_true", help="Write factor snapshot rows into TDengine.")
    parser.add_argument("--job-name", default=JOB_NAME)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--lock-file", default=DEFAULT_LOCK_FILE)
    parser.add_argument("--health-max-delay-minutes", type=float, default=10.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    config = load_config(args.config)
    schema = load_schema(args.schema)

    if args.mode == "backfill":
        return run_backfill(args, config, schema)
    if args.mode == "incremental":
        return run_incremental(args, config, schema)
    if args.mode == "healthcheck":
        return run_healthcheck(args, config, schema)
    if args.mode == "record-locked":
        return record_locked(args, config, schema)
    if args.mode == "backfill-5m-history":
        return run_backfill_5m_history(args, config, schema)
    if args.mode == "prepare-active-history":
        return run_prepare_active_history(args, config, schema)
    if args.mode == "readiness-check":
        return run_readiness_check(args, config, schema)
    if args.mode == "diff-live-backfill":
        return run_diff_live_backfill(args, config, schema)
    raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
