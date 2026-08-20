from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cb_system_matplotlib")

from .config import load_config
from .pipeline import RunOptions, regenerate_report, run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V0.8 causal nonlinear CB-stock response and hidden-state research"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["build-dataset", "smoke-test", "backtest", "fit-latest", "report"],
    )
    parser.add_argument("--config", default="config/dynamic_response_v080.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--models", default=None, help="Comma-separated ridge,lightgbm,xgboost")
    parser.add_argument("--retune", action="store_true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", dest="use_cache", action="store_true")
    cache_group.add_argument(
        "--no-use-cache", "--no-cache", dest="use_cache", action="store_false"
    )
    parser.set_defaults(use_cache=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(args.config)
    if args.seed is not None:
        config["strategy"]["random_seed"] = int(args.seed)
    models = tuple(value.strip() for value in args.models.split(",") if value.strip()) if args.models else None
    options = RunOptions(
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        days=args.days,
        models=models,
        retune=bool(args.retune),
        use_cache=bool(args.use_cache),
        output_override=args.output_dir,
    )
    if args.mode == "report":
        result = {"report": str(regenerate_report(config, options))}
    else:
        result = run_backtest(config, options)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
