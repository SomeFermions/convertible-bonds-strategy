from __future__ import annotations

import argparse
import json
import logging
import sys

from db import get_conn
from training_data_v06.pipeline import (
    audit_jq_training_bars,
    bool_arg,
    build_jq_download_plan,
    build_local_training_windows,
    build_training_sample,
    canonicalize_jq_training_bars,
    collect_akshare_conversion_price_history,
    collect_jq_benchmarks,
    jq_smoke_test,
    report_jq_training_coverage,
    report_jq_benchmark_coverage,
    refresh_akshare_static_cache_from_local,
    refresh_cn_trade_calendar,
    repair_jq_raw_bar_semantics,
    run_jq_download_plan,
    run_wind_turnover_ytm_enrichment,
    wind_smoke_test,
)
from training_data_v06.config import load_config
from training_data_v06.schemas import ensure_training_tables


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V0.6 JQData/Wind historical training data cache pipeline.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "init-schema",
            "jq-smoke-test",
            "build-training-sample",
            "refresh-akshare-static-cache",
            "collect-akshare-conversion-price-history",
            "audit-jq-training-bars",
            "collect-jq-benchmarks",
            "report-jq-benchmark-coverage",
            "refresh-cn-trade-calendar",
            "build-jq-download-plan",
            "run-jq-download-plan",
            "canonicalize-jq-training-bars",
            "report-jq-training-coverage",
            "build-local-training-windows",
            "run-wind-turnover-ytm-enrichment",
            "wind-smoke-test",
            "repair-jq-bar-semantics",
        ],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-version", default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--available-range", default="trial_15m_to_3m")
    parser.add_argument("--frequency", default="5m")
    parser.add_argument("--fields", default="open,high,low,close,volume,money,paused")
    parser.add_argument("--bond-codes", default=None)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--dry-run", default="true")
    parser.add_argument("--confirm-large-download", action="store_true")
    parser.add_argument("--confirm-akshare-download", action="store_true")
    parser.add_argument("--confirm-wind-download", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-bonds", type=int, default=None)
    parser.add_argument("--max-rows-today", type=int, default=450000)
    parser.add_argument("--max-rows-week", type=int, default=45000)
    parser.add_argument("--max-wind-cells-run", type=int, default=None)
    parser.add_argument("--scope", default="active_like_only")
    parser.add_argument("--window-trading-days", type=int, default=45)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def print_json(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    config = load_config(args.config)
    sample_version = args.sample_version or config["default_sample_version"]
    dry_run = bool_arg(args.dry_run)

    if args.available_range != "trial_15m_to_3m":
        raise SystemExit("V0.6 only supports --available-range trial_15m_to_3m")

    if args.mode == "refresh-cn-trade-calendar":
        try:
            print_json(refresh_cn_trade_calendar(config))
            return 0
        except Exception as exc:
            logging.exception("training data v0.6 command failed")
            print_json({"mode": args.mode, "status": "FAILED", "error": str(exc)})
            return 1

    conn = get_conn()
    try:
        try:
            ensure_training_tables(conn)
            if args.mode == "init-schema":
                print_json({"mode": args.mode, "status": "OK"})
                return 0
            if args.mode == "jq-smoke-test":
                out = jq_smoke_test(
                    conn,
                    parse_csv(args.bond_codes),
                    args.days,
                    dry_run,
                    config,
                    sample_version,
                    end_date=args.end_date,
                )
                print_json(out)
                return 0
            if args.mode == "wind-smoke-test":
                out = wind_smoke_test(
                    conn,
                    parse_csv(args.bond_codes),
                    args.days,
                    dry_run,
                    args.confirm_wind_download,
                    sample_version,
                    config,
                )
                print_json(out)
                return 0
            if args.mode == "repair-jq-bar-semantics":
                out = repair_jq_raw_bar_semantics(conn, args.batch_id, config)
                print_json(out)
                return 0
            if args.mode == "build-training-sample":
                out = build_training_sample(conn, args.sample_size, sample_version, config)
                print_json(out)
                return 0
            if args.mode == "refresh-akshare-static-cache":
                out = refresh_akshare_static_cache_from_local(conn)
                print_json(out)
                return 0
            if args.mode == "collect-akshare-conversion-price-history":
                out = collect_akshare_conversion_price_history(
                    conn,
                    sample_version=sample_version,
                    config=config,
                    confirm_download=args.confirm_akshare_download,
                    refresh_cache=args.refresh_cache,
                    max_bonds=args.max_bonds,
                )
                print_json(out)
                return 0
            if args.mode == "audit-jq-training-bars":
                print_json(audit_jq_training_bars(conn, sample_version, config))
                return 0
            if args.mode == "collect-jq-benchmarks":
                out = collect_jq_benchmarks(
                    conn,
                    sample_version=sample_version,
                    dry_run=dry_run,
                    confirm_large_download=args.confirm_large_download,
                    max_rows_today=args.max_rows_today,
                    config=config,
                )
                print_json(out)
                return 0
            if args.mode == "report-jq-benchmark-coverage":
                print_json(report_jq_benchmark_coverage(conn, sample_version, config))
                return 0
            if args.mode == "build-jq-download-plan":
                out = build_jq_download_plan(
                    conn,
                    sample_version=sample_version,
                    frequency=args.frequency,
                    fields=parse_csv(args.fields),
                    dry_run=dry_run,
                    confirm_large_download=args.confirm_large_download,
                    config=config,
                )
                print_json(out)
                return 0
            if args.mode == "run-jq-download-plan":
                out = run_jq_download_plan(
                    conn,
                    sample_version=sample_version,
                    max_rows_today=args.max_rows_today,
                    confirm_large_download=args.confirm_large_download,
                    config=config,
                )
                print_json(out)
                return 0
            if args.mode == "canonicalize-jq-training-bars":
                out = canonicalize_jq_training_bars(conn, sample_version, config, batch_id=args.batch_id)
                print_json(out)
                return 0
            if args.mode == "report-jq-training-coverage":
                out = report_jq_training_coverage(conn, sample_version, config)
                print_json(out)
                return 0
            if args.mode == "build-local-training-windows":
                out = build_local_training_windows(conn, sample_version, args.window_trading_days, args.max_windows)
                print_json(out)
                return 0
            if args.mode == "run-wind-turnover-ytm-enrichment":
                out = run_wind_turnover_ytm_enrichment(
                    conn,
                    sample_version=sample_version,
                    scope=args.scope,
                    max_rows_week=args.max_rows_week,
                    confirm_wind_download=args.confirm_wind_download,
                    config=config,
                    max_cells_run=args.max_wind_cells_run,
                )
                print_json(out)
                return 0
        except Exception as exc:
            logging.exception("training data v0.6 command failed")
            print_json({"mode": args.mode, "status": "FAILED", "error": str(exc)})
            return 1
    finally:
        conn.close()
    raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
