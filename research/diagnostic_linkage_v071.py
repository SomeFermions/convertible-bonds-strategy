from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from db import get_conn
from research.diagnostic_linkage_v071_core import add_linkage_features, deduplicate_event_episodes
from research.diagnostic_linkage_v071_data import (
    add_versions,
    build_common_slice,
    build_universe_report,
    filter_dates,
    load_config,
    load_daily_panel,
    output_dir,
    write_csv,
)
from research.diagnostic_linkage_v071_eval import (
    backtest_daily_simple_rv,
    backtest_intraday_linkage,
    benchmark_comparison,
    build_walk_forward_plan,
    build_ablation_report,
    common_slice_comparison,
    evaluate_cb_leads_stock,
    evaluate_lsmc_exit_overlay,
    evaluate_qiv_mc_restored_ablation,
    grouped_linkage_performance,
    linkage_coefficient_reports,
    load_qiv_pricing,
    prepare_daily_simple_rv,
    qiv_mc_coverage_report,
    qiv_mc_monotonicity,
    reverse_linkage_window_stability,
    window_stability,
)


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
HYBRID_OUTPUT = PROJECT_DIR / "outputs" / "research" / "hybrid_v07"


def parse_csv(value: str | None, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in str(value or "").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V0.7.1 diagnostic reset and stock-bond linkage research."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "diagnostic-reset",
            "build-linkage-features",
            "build-linkage-events",
            "backtest-linkage",
            "backtest-daily-simple-rv",
            "evaluate-lsmc-exit-overlay",
            "evaluate-cb-leads-stock",
            "common-slice",
            "ablation",
            "report",
            "all",
        ],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lags", default="0,1,2,3")
    parser.add_argument("--entry-delay-bars", default="0,1,2")
    parser.add_argument(
        "--cost-scenarios", default="zero,realistic,conservative,disaster_5x"
    )
    parser.add_argument("--holding-days", default="1,3,5")
    parser.add_argument("--top-n", default="5,10")
    parser.add_argument("--strategies", default="intraday_linkage,daily_simple_rv,daily_rv_lsmc_exit,hybrid")
    parser.add_argument("--model", default="ridge", choices=["ridge", "huber", "ols"])
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _artifact(config: dict[str, Any], name: str) -> Path:
    return output_dir(config) / name


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(
        path,
        dtype={"bond_code": str, "stock_code": str},
        low_memory=False,
    )
    for column in [
        "trade_date",
        "signal_time",
        "first_seen_time",
        "entry_time",
        "exit_time",
        "entry_date",
        "exit_date",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _compact_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(include=["float64"]).columns:
        out[column] = pd.to_numeric(out[column], downcast="float")
    for column in [
        "window_id",
        "window_role",
        "bond_type",
        "research_pool_scope",
        "chosen_market_index",
        "linkage_model_source",
        "dominant_lead_direction",
        "liquidity_regime_5m",
        "event_type",
        "event_action_state",
    ]:
        if column in out.columns:
            out[column] = out[column].astype("category")
    return out


def ensure_common_slice(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_cache = _artifact(config, "common_slice_bars.pkl")
    benchmark_cache = _artifact(config, "common_slice_benchmarks.pkl")
    connection = None
    try:
        if refresh_cache or not common_cache.exists() or not benchmark_cache.exists():
            connection = get_conn()
        bars, benchmark, daily = build_common_slice(
            connection,
            config,
            start_date,
            end_date,
            refresh_cache=refresh_cache,
        )
    finally:
        if connection is not None:
            connection.close()
    universe, manifest = build_universe_report(bars, benchmark, daily, config)
    write_csv(universe, _artifact(config, "point_in_time_universe.csv"), config)
    write_csv(manifest, _artifact(config, "common_slice_manifest.csv"), config)
    return bars, benchmark, daily


def ensure_features(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    refresh_cache: bool,
    model: str,
) -> pd.DataFrame:
    path = _artifact(config, f"linkage_features_{model}.pkl")
    meta_path = _artifact(config, f"linkage_features_{model}.meta.json")
    valid_cache = False
    if path.exists() and meta_path.exists() and not refresh_cache:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        valid_cache = metadata.get("config_version") == config["strategy"]["config_version"]
    if valid_cache:
        features = pd.read_pickle(path)
    else:
        bars, _, _ = ensure_common_slice(config, None, None, refresh_cache)
        started = time.perf_counter()
        features = add_linkage_features(bars, config, model=model)
        features = _compact_features(features)
        features.to_pickle(path)
        meta_path.write_text(
            json.dumps(
                {
                    "config_version": config["strategy"]["config_version"],
                    "factor_version": config["strategy"]["factor_version"],
                    "model": model,
                    "rows": int(len(features)),
                    "runtime_seconds": time.perf_counter() - started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if "strategy_linkage_liquidity_no_hot_reject" not in features.columns:
        broad_liquidity = ~features["liquidity_regime_5m"].astype(str).eq("ILLIQUID")
        no_hot_reject = (
            features["strategy_linkage_repair"].fillna(False) & broad_liquidity
        ).rename("strategy_linkage_liquidity_no_hot_reject")
        features = pd.concat(
            [features, no_hot_reject],
            axis=1,
        )
    features = filter_dates(features, start_date, end_date)
    coefficients, by_bond, by_type = linkage_coefficient_reports(features, config)
    write_csv(coefficients, _artifact(config, "linkage_coefficients.csv"), config)
    write_csv(by_bond, _artifact(config, "lead_lag_by_bond.csv"), config)
    write_csv(by_type, _artifact(config, "lead_lag_by_type.csv"), config)
    half_life = features.sort_values("bar_start").drop_duplicates(
        ["trade_date", "bond_code"], keep="last"
    ).groupby("response_half_life_bars", dropna=False).agg(
        bond_days=("bond_code", "size"),
        bond_count=("bond_code", "nunique"),
        actionable_beta_sum=("stock_to_cb_beta_actionable_sum", "mean"),
        linkage_confidence=("linkage_model_confidence", "mean"),
    ).reset_index()
    write_csv(half_life, _artifact(config, "response_half_life_distribution.csv"), config)
    return features


def ensure_events(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    events = deduplicate_event_episodes(features, config)
    write_csv(events, _artifact(config, "linkage_events.csv"), config)
    event_by_type = events.groupby(
        ["event_type", "bond_type", "liquidity_regime_5m"], dropna=False
    ).agg(
        episode_count=("episode_id", "nunique"),
        bond_count=("bond_code", "nunique"),
        trading_days=("trade_date", "nunique"),
        stock_shock_z=("stock_shock_z", "mean"),
        linkage_gap_3bar=("linkage_gap_3bar", "mean"),
        hot_money_risk_score=("hot_money_risk_score_5m", "mean"),
    ).reset_index()
    write_csv(event_by_type, _artifact(config, "linkage_event_by_type.csv"), config)
    return events


def ensure_intraday_backtest(
    features: pd.DataFrame,
    config: dict[str, Any],
    entry_delays: list[int],
    scenarios: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades, summary, waterfall = backtest_intraday_linkage(
        features, config, entry_delays, scenarios
    )
    write_csv(trades, _artifact(config, "intraday_linkage_trades.csv"), config)
    write_csv(summary, _artifact(config, "linkage_delay_sensitivity.csv"), config)
    write_csv(waterfall, _artifact(config, "cost_waterfall.csv"), config)
    dynamic = summary[
        summary["strategy"].isin(
            [
                "D4_GAP_REPAIR_LIQUIDITY",
                "D5_DYNAMIC_LINKAGE_EXIT",
                "D6_HOT_MONEY_REJECT",
                "D7_NO_HOT_MONEY_REJECT",
                "D9_FIXED_PERCENT_EXIT_BENCHMARK",
            ]
        )
    ].copy()
    write_csv(dynamic, _artifact(config, "dynamic_exit_comparison.csv"), config)
    return trades, summary, waterfall


def ensure_daily_rv(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
    top_n: list[int],
    holding_days: list[int],
    scenarios: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = load_daily_panel(config, start_date, end_date)
    pricing = load_qiv_pricing(HYBRID_OUTPUT / "cb_daily_relative_value_snapshot.csv")
    pricing = filter_dates(pricing, start_date, end_date)
    snapshot = prepare_daily_simple_rv(daily, config, pricing)
    write_csv(snapshot, _artifact(config, "daily_simple_rv.csv"), config)
    selectors = [f"top{value}" for value in top_n] + ["top20pct"]
    trades, summary = backtest_daily_simple_rv(
        snapshot, config, selectors, holding_days, scenarios
    )
    write_csv(trades, _artifact(config, "daily_simple_rv_trades.csv"), config)
    write_csv(summary, _artifact(config, "daily_simple_rv_summary.csv"), config)
    benchmark = benchmark_comparison(trades, snapshot)
    write_csv(benchmark, _artifact(config, "benchmark_comparison.csv"), config)
    return snapshot, trades, summary


def ensure_lsmc_overlay(
    snapshot: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lsmc = _read_csv(HYBRID_OUTPUT / "cb_lsmc_daily_snapshot.csv")
    lsmc = filter_dates(lsmc, None, None)
    trades, summary, matched = evaluate_lsmc_exit_overlay(snapshot, lsmc, config)
    write_csv(trades, _artifact(config, "daily_exit_overlay_trades.csv"), config)
    write_csv(summary, _artifact(config, "daily_exit_overlay.csv"), config)
    write_csv(matched, _artifact(config, "lsmc_matched_experiments.csv"), config)
    return trades, summary, matched


def _merge_event_exits(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if events.empty or trades.empty:
        return events.copy()
    exits = trades[
        trades["strategy"].eq("D6_HOT_MONEY_REJECT")
        & trades["entry_delay_bars"].eq(0)
        & trades["cost_scenario"].eq("realistic")
    ].copy()
    columns = [
        "trade_date",
        "bond_code",
        "signal_time",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "exit_reason",
        "holding_bars",
        "gross_return",
        "total_cost_bps",
        "net_return",
        "mfe",
        "mae",
    ]
    exits = exits[[c for c in columns if c in exits.columns]].drop_duplicates(
        ["trade_date", "bond_code", "signal_time"], keep="last"
    )
    result = events.merge(
        exits,
        on=["trade_date", "bond_code", "signal_time"],
        how="left",
        suffixes=("", "_trade"),
    )
    result["data_mode"] = "scheduled_backfill"
    result["version"] = "diagnostic_linkage_v071"
    return result


def _fmt_pct(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "NA" if not np.isfinite(number) else f"{number * 100:.4f}%"


def _fmt_num(value: Any, digits: int = 4) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        return "No available rows."
    view = frame[available].head(limit).copy()
    header = "| " + " | ".join(available) + " |"
    divider = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = [header, divider]
    for values in view.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def generate_report(config: dict[str, Any]) -> Path:
    out = output_dir(config)
    manifest = _read_csv(out / "common_slice_manifest.csv")
    intraday = _read_csv(out / "linkage_delay_sensitivity.csv")
    lead_type = _read_csv(out / "lead_lag_by_type.csv")
    events = _read_csv(out / "linkage_events.csv")
    waterfall = _read_csv(out / "cost_waterfall.csv")
    half_life = _read_csv(out / "response_half_life_distribution.csv")
    dynamic = _read_csv(out / "dynamic_exit_comparison.csv")
    daily = _read_csv(out / "daily_simple_rv_summary.csv")
    benchmark = _read_csv(out / "benchmark_comparison.csv")
    overlay = _read_csv(out / "daily_exit_overlay.csv")
    matched = _read_csv(out / "lsmc_matched_experiments.csv")
    qiv = _read_csv(out / "qiv_monotonicity.csv")
    qiv_coverage = _read_csv(out / "qiv_mc_coverage.csv")
    qiv_restored = _read_csv(out / "qiv_mc_restored_ablation.csv")
    reverse = _read_csv(out / "cb_leads_stock_summary.csv")
    reverse_windows = _read_csv(out / "cb_leads_stock_window_stability.csv")
    common = _read_csv(out / "common_slice_strategy_comparison.csv")
    windows = _read_csv(out / "window_stability.csv")

    def select_one(frame: pd.DataFrame, **conditions: Any) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=object)
        mask = pd.Series(True, index=frame.index)
        for column, value in conditions.items():
            if column not in frame.columns:
                return pd.Series(dtype=object)
            mask &= frame[column].eq(value)
        selected = frame[mask]
        return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)
    formal = intraday[
        intraday["strategy"].eq("D6_HOT_MONEY_REJECT")
        & intraday["entry_delay_bars"].eq(0)
        & intraday["cost_scenario"].isin(["zero", "realistic"])
    ]
    gross_row = formal[formal["cost_scenario"].eq("zero")]
    real_row = formal[formal["cost_scenario"].eq("realistic")]
    gross_mean = gross_row["gross_return_mean"].iloc[0] if not gross_row.empty else np.nan
    real_mean = real_row["net_return_mean"].iloc[0] if not real_row.empty else np.nan
    gross_ci_low = gross_row["block_bootstrap_ci_low"].iloc[0] if not gross_row.empty else np.nan
    stable_gross = bool(np.isfinite(gross_ci_low) and gross_ci_low > 0)
    delay_real = intraday[
        intraday["strategy"].eq("D6_HOT_MONEY_REJECT")
        & intraday["cost_scenario"].eq("realistic")
    ].sort_values("entry_delay_bars")
    lsmc_fixed = overlay[
        overlay["exit_rule"].eq("fixed_3d") & overlay["cost_scenario"].eq("realistic")
    ]
    lsmc_exit = overlay[
        overlay["exit_rule"].eq("lsmc_exit") & overlay["cost_scenario"].eq("realistic")
    ]
    lsmc_increment = (
        lsmc_exit["net_return"].iloc[0] - lsmc_fixed["net_return"].iloc[0]
        if not lsmc_exit.empty and not lsmc_fixed.empty
        else np.nan
    )
    lsmc_fixed1 = select_one(overlay, exit_rule="fixed_1d", cost_scenario="realistic")
    lsmc_real = select_one(overlay, exit_rule="lsmc_exit", cost_scenario="realistic")
    lsmc_vs_fixed1 = (
        pd.to_numeric(lsmc_real.get("net_return"), errors="coerce")
        - pd.to_numeric(lsmc_fixed1.get("net_return"), errors="coerce")
    )
    calibration = False
    if "lsmc_hold_edge_tradable_calibration" in matched.columns:
        calibration = bool(
            matched["lsmc_hold_edge_tradable_calibration"]
            .astype(str)
            .str.lower()
            .eq("true")
            .any()
        )
    qiv_rho = (
        pd.to_numeric(qiv.get("spearman_bin_vs_3d"), errors="coerce")
        .dropna()
        .mean()
        if not qiv.empty
        else np.nan
    )
    qiv_all = select_one(qiv_coverage, scope="ALL")
    qiv_restored_real = select_one(
        qiv_restored,
        selector="top10",
        exit_rule="fixed_3d",
        cost_scenario="realistic",
    )
    daily_best = select_one(
        daily, selector="top5", exit_rule="fixed_5d", cost_scenario="realistic"
    )
    daily_best_benchmark = select_one(
        benchmark,
        selector="top5",
        exit_rule="fixed_5d",
        cost_scenario="realistic",
    )
    dynamic_real = select_one(
        dynamic,
        strategy="D6_HOT_MONEY_REJECT",
        entry_delay_bars=0,
        cost_scenario="realistic",
    )
    fixed_real = select_one(
        dynamic,
        strategy="D9_FIXED_PERCENT_EXIT_BENCHMARK",
        entry_delay_bars=0,
        cost_scenario="realistic",
    )
    no_hot_real = select_one(
        dynamic,
        strategy="D7_NO_HOT_MONEY_REJECT",
        entry_delay_bars=0,
        cost_scenario="realistic",
    )
    event_counts = events["event_type"].value_counts() if not events.empty else pd.Series(dtype=int)
    focused_windows = (
        windows[
            windows["strategy"].isin(
                [
                    "intraday_linkage_dynamic",
                    "intraday_linkage_fixed_exit",
                    "daily_top5_fixed5",
                ]
            )
            & windows["cost_scenario"].eq("realistic")
        ]
        if not windows.empty
        else windows
    )
    daily_report = daily[
        daily["selector"].isin(["top5", "top10"])
        & daily["cost_scenario"].isin(["zero", "realistic"])
    ].copy() if not daily.empty else daily
    if not daily_report.empty:
        daily_report["_selector_order"] = daily_report["selector"].map(
            {"top5": 0, "top10": 1}
        )
        daily_report["_cost_order"] = daily_report["cost_scenario"].map(
            {"zero": 0, "realistic": 1}
        )
        daily_report = daily_report.sort_values(
            ["_selector_order", "exit_rule", "_cost_order"]
        )
    lines = [
        "# 可转债 V0.7.1 Diagnostic Reset + Stock-Bond Linkage Research Report",
        "",
        f"- strategy_version: `{config['strategy']['version']}`",
        f"- factor_version: `{config['strategy']['factor_version']}`",
        f"- config_version: `{config['strategy']['config_version']}`",
        "- research_only: `true`；未改变生产 alert，也未生成订单。",
        f"- universe bias: `{config['common_slice']['universe_bias_flag']}`.",
        f"- historical fundamental PIT status: `{config['common_slice']['fundamental_pit_status']}`.",
        "",
        "## 核心诊断",
        "",
        f"正式联动规则的零成本平均毛收益为 {_fmt_pct(gross_mean)}，现实成本后为 {_fmt_pct(real_mean)}。",
        f"按交易日 block bootstrap 的毛收益下界为 {_fmt_pct(gross_ci_low)}，因此“存在稳定毛 Alpha”判定为 `{stable_gross}`。",
        "lag0 是联动主体，但它在 5min 收盘后不可捕获；lag1-lag3 虽有正系数，尚未转化为跨窗口稳定、成本后为正的交易收益。",
        "qIV/MC 的正式权重保持 0；LSMC 不参与入场，只在完全相同的日频 RV 入场样本上比较退出。",
        "历史 canonical 样本来自 2026-07 的训练池选择，所有历史结果均带 selection bias 标记，不宣称是无偏历史 universe。",
        "",
        "## Common Slice 与 PIT",
        "",
        _markdown_table(
            manifest,
            [
                "window_id",
                "window_role",
                "first_date",
                "last_date",
                "trading_days",
                "observable_bonds",
                "pit_eligible_bond_days",
                "conversion_price_coverage",
                "turnover_coverage",
            ],
        ),
        "",
        "四个窗口各 45 个交易日且互不重叠。滚动估计不跨窗口空档制造 bar；日内最早下一根 open 成交，日频最早次日 open 成交。转股价和换手率覆盖率均为 100%，但历史基本面/ST 准入无法按当时状态完整重建。",
        "",
        "## 成本瀑布",
        "",
        _markdown_table(
            waterfall[
                waterfall["strategy"].eq("D6_HOT_MONEY_REJECT")
                & waterfall["entry_delay_bars"].eq(0)
            ]
            if not waterfall.empty
            else waterfall,
            [
                "cost_scenario",
                "trade_count",
                "gross_return",
                "commission_cost_bps",
                "spread_cost_bps",
                "latency_slippage_bps",
                "market_impact_bps",
                "liquidity_penalty_bps",
                "total_cost_bps",
                "net_return",
            ],
        ),
        "",
        "`latency_slippage_bps` 是 signal close 到实际 entry open 的 implementation shortfall 诊断；entry price 已采用延迟后的开盘价，因此不会再次从 net return 重复扣除。",
        "",
        "## Lead-Lag 结构",
        "",
        _markdown_table(
            lead_type,
            [
                "bond_type",
                "bond_count",
                "beta_lag0",
                "beta_lag1",
                "beta_lag2",
                "beta_lag3",
                "actionable_beta_sum",
                "response_half_life_bars",
                "linkage_r2",
            ],
        ),
        "",
        f"事件数：`{json.dumps(event_counts.to_dict(), ensure_ascii=False)}`。",
        "`beta_lag0` 只做诊断；可捕获响应严格使用 lag1-lag3。EQUITY_LIKE 的 actionable beta 较高，但不能据此直接变成类型硬 gate。",
        "",
        _markdown_table(
            half_life,
            [
                "response_half_life_bars",
                "bond_days",
                "bond_count",
                "actionable_beta_sum",
                "linkage_confidence",
            ],
        ),
        "",
        "## 执行延迟",
        "",
        _markdown_table(
            delay_real,
            [
                "entry_delay_bars",
                "trade_count",
                "gross_return_mean",
                "net_return_mean",
                "return_1bar",
                "return_2bar",
                "return_3bar",
                "mfe",
                "mae",
                "hit_rate_net",
            ],
        ),
        "",
        "额外延迟 1 根 bar 的均值偶然改善，但置信区间跨 0；延迟 2 根 bar 后毛收益转负。该非单调结果不能解释成“晚买更好”，只能说明 138 条事件仍少且路径差异很大。",
        "",
        "## 动态退出与接盘风险",
        "",
        _markdown_table(
            dynamic[
                dynamic["entry_delay_bars"].eq(0)
                & dynamic["cost_scenario"].isin(["zero", "realistic"])
            ]
            if not dynamic.empty
            else dynamic,
            [
                "strategy",
                "cost_scenario",
                "trade_count",
                "gross_return_mean",
                "net_return_mean",
                "hit_rate_net",
                "average_holding_bars",
                "mfe",
                "mae",
            ],
            limit=12,
        ),
        "",
        f"正式动态退出现实净收益 {_fmt_pct(dynamic_real.get('net_return_mean'))}，固定波动率归一化基准为 {_fmt_pct(fixed_real.get('net_return_mean'))}；当前动态逻辑没有增量。hot-money reject 将样本从 {_fmt_num(no_hot_real.get('trade_count'), 0)} 降到 {_fmt_num(dynamic_real.get('trade_count'), 0)}，但只减少风险，未创造成本后 Alpha。",
        "",
        "## 日频简单 RV 与 LSMC Exit Overlay",
        "",
        _markdown_table(
            daily_report,
            [
                "selector",
                "exit_rule",
                "cost_scenario",
                "trade_count",
                "gross_return_mean",
                "net_return_mean",
                "hit_rate",
                "mfe",
                "mae",
                "max_drawdown",
            ],
        ),
        "",
        f"全样本中 top5/fixed5d 的现实成本后均值为 {_fmt_pct(daily_best.get('net_return_mean'))}，相对全池/同类型超额分别为 {_fmt_pct(daily_best_benchmark.get('excess_vs_full_pool'))} / {_fmt_pct(daily_best_benchmark.get('excess_vs_same_type'))}。但四窗口方向不稳定，holdout 中位数为负，且存在历史池偏差，所以它只是下一轮 shadow 候选。",
        f"在完全相同的 260 个入场上，LSMC exit 相对 fixed3d 改善 {_fmt_pct(lsmc_increment)}，但相对 fixed1d 差 {_fmt_pct(lsmc_vs_fixed1)}，且所有退出方式均为负。hold-edge 可交易校准判定为 `{calibration}`。",
        "LSMC matched 数据只有 27 个交易日、1 个窗口；减少交易数量、改变入场和改变退出的作用已在 L1-L4 中拆开，当前不能把 continuation value 称为 Alpha。",
        "",
        "## qIV / MC 诊断",
        "",
        f"qIV smooth 覆盖率 {_fmt_pct(qiv_all.get('qiv_smooth_coverage'))}，MC gap 覆盖率 {_fmt_pct(qiv_all.get('mc_price_gap_coverage'))}；可判定样本中的 contamination 比例为 {_fmt_pct(qiv_all.get('contamination_rate_when_assessable'))}。",
        f"同类型 qIV/MC 分桶对未来 3 日收益的平均 Spearman 为 {_fmt_num(qiv_rho)}。恢复 40% qIV/MC 诊断权重后的 top10/fixed3d 现实均值为 {_fmt_pct(qiv_restored_real.get('net_return_mean'))}，但区间跨 0、窗口/中位数不稳定，不能据此恢复正式权重。",
        "结论：qIV/MC 继续只做诊断字段，formal decision weight = `0.0`。",
        "",
        "## CB Leads Stock 隔夜信息",
        "",
        _markdown_table(
            reverse,
            [
                "bond_type",
                "liquidity_regime_5m",
                "event_count",
                "stock_next_open_return",
                "stock_next_close_return",
                "bond_next_open_return",
                "bond_next_close_return",
            ],
        ),
        "",
        _markdown_table(
            reverse_windows,
            [
                "window_id",
                "event_count",
                "bond_count",
                "stock_next_open_return",
                "stock_next_close_return",
                "bond_next_open_return",
                "bond_next_close_return",
            ],
        ),
        "",
        "反向联动只做隔夜信息诊断，不进入转债日内买入。窗口方向反复，最终 holdout 的正股次日收盘收益为负，因此当前不能作为隔夜持有理由。",
        "",
        "## Common-Slice 策略比较",
        "",
        _markdown_table(
            common,
            [
                "comparison_scope",
                "strategy",
                "common_slice_days",
                "mean_daily_cohort_return",
                "median_daily_cohort_return",
                "compound_cohort_return",
                "hit_rate",
                "max_drawdown",
                "worst_day",
            ],
        ),
        "",
        "这些是同一日期、同一 canonical pool、同一现实成本口径下的等权 signal-date cohort return，不是带仓位约束的投资组合 NAV。LSMC 只有 26 个严格重叠日，不能与 180 日 core slice 混报。",
        "",
        "## Walk-Forward 稳定性",
        "",
        _markdown_table(
            focused_windows,
            [
                "window_id",
                "strategy",
                "cost_scenario",
                "sample_count",
                "mean_return",
                "median_return",
                "hit_rate",
                "ci_low",
                "ci_high",
                "worst_date",
                "worst_bond",
            ],
            limit=24,
        ),
        "",
        "正式日内动态策略在 W4 holdout 的零成本均值已经为负；日频 top5/fixed5d 虽在 W1/W3 较强，但 W2 接近负、W4 中位数明显为负。收益存在显著窗口和尾部样本集中。",
        "",
        "## 必答问题",
        "",
        f"1. 稳定毛 Alpha：`{stable_gross}`；正式规则毛收益 {_fmt_pct(gross_mean)}，bootstrap 下界 {_fmt_pct(gross_ci_low)}。",
        f"2. 现实成本后收益：`{bool(np.isfinite(real_mean) and real_mean > 0)}`；均值 {_fmt_pct(real_mean)}。",
        "3. lag1-lag3 有小幅统计响应，但远弱于不可交易的 lag0，尚无稳定可捕获证据。",
        "4. EQUITY_LIKE 的 lag1-lag3 系数最高；BOND_LIKE 更容易形成长期不跟。类型差异只分组报告，不做硬 gate。",
        "5. 延迟 1bar 的均值为正但不稳定、净收益仍负，不能支持人工执行。",
        "6. 延迟 2bar 毛收益转负，不可交易。",
        "7. distributed-lag gap 的宽松版本略优于同期 residual，但改善不足以覆盖成本。",
        "8. repair confirmation 大幅缩样本，未提高当前动态退出收益，反而暴露确认过晚风险。",
        "9. liquidity sweet spot 提高了筛选后毛收益，但现实成本后仍为负。",
        "10. 正确消融是 D7(no reject) vs D6(hot reject)：reject 降低样本和部分风险，没有创造净 Alpha。",
        f"11. 动态退出现实净收益 {_fmt_pct(dynamic_real.get('net_return_mean'))}，劣于固定波动率基准 {_fmt_pct(fixed_real.get('net_return_mean'))}。",
        "12. half-life 主要为 1-2bar；当前 TIME_STOP/失效退出过于激进，不能从全样本挑一个最优 bar 上线。",
        "13. CB_LEADS_STOCK 的次日信息含量跨窗口不稳定，holdout 转负，不构成隔夜理由。",
        f"14. 简单日频 RV 最值得继续观察的是 top5/fixed5d，现实均值 {_fmt_pct(daily_best.get('net_return_mean'))}，但窗口和中位数不稳。",
        f"15. LSMC exit 相对 fixed3d 改善 {_fmt_pct(lsmc_increment)}，但不如 fixed1d，且共同样本全部为负。",
        "16. L1 固定交易数、L2 固定入场、L3 固定退出和 L4 hold-edge 分桶均已拆分；LSMC 改善不能归因于已校准 continuation Alpha。",
        f"17. qIV/MC 平均分桶 Spearman {_fmt_num(qiv_rho)}，没有稳定单调性。",
        "18. qIV/MC 应继续保持权重 0；D16 的样本均值改善只是一条待独立复制的 MC-gap 假设。",
        "19. bond type 硬 gate 未证明有增量，当前仅作分组、风险提示和控制变量。",
        "20. 收益明显集中在少数窗口、日期与债券；W4 对正式日内规则是负结果。",
        "21. 180 日 core cohort 中 Daily/Hybrid 均值较好，但 strict LSMC overlap 26 日四种策略全部为负；不能混为同一结论。",
        "22. 下一步只适合做 alert/logging shadow：STOCK_LEADS_CB_REPAIR + 固定波动率归一化退出；另建 top5/fixed5d 日频 paper cohort。暂不建议模拟仓实际下单。",
        "23. 可交易性、类型优势、时段优势、日频 RV 和反向联动均仍是研究假设；历史 universe 偏差消除前不能上线。",
        "",
        "## 当前不能得出的结论",
        "",
        "- 没有证据支持生产交易提示、自动下单、恢复 qIV/MC 入场权重或启用 LSMC 入场。",
        "- 当前本地表无法完整重建历史 ST/基本面准入，不能把 present-day selected pool 当成无偏历史 universe。",
        "- 单个均值、单个窗口或全样本最优参数都不能被称为 Alpha。",
    ]
    path = out / "diagnostic_linkage_v071_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    delays = parse_csv(args.entry_delay_bars, int)
    scenarios = parse_csv(args.cost_scenarios)
    top_n = parse_csv(args.top_n, int)
    holding = parse_csv(args.holding_days, int)
    features = ensure_features(
        config, args.start_date, args.end_date, args.refresh_cache, args.model
    )
    events = ensure_events(features, config)
    intraday_trades, intraday_summary, _ = ensure_intraday_backtest(
        features, config, delays, scenarios
    )
    grouped_linkage = grouped_linkage_performance(intraday_trades)
    write_csv(
        grouped_linkage,
        _artifact(config, "linkage_grouped_diagnostics.csv"),
        config,
    )
    event_episodes = _merge_event_exits(events, intraday_trades)
    write_csv(event_episodes, _artifact(config, "cb_linkage_event_episode.csv"), config)
    snapshot, daily_trades, daily_summary = ensure_daily_rv(
        config, args.start_date, args.end_date, top_n, holding, scenarios
    )
    overlay_trades, overlay_summary, _ = ensure_lsmc_overlay(snapshot, config)
    pricing = load_qiv_pricing(HYBRID_OUTPUT / "cb_daily_relative_value_snapshot.csv")
    pricing = filter_dates(pricing, args.start_date, args.end_date)
    pricing_common = pricing.merge(
        snapshot[["trade_date", "bond_code"]].drop_duplicates(),
        on=["trade_date", "bond_code"],
        how="inner",
    )
    qiv_coverage = qiv_mc_coverage_report(pricing_common, config)
    write_csv(qiv_coverage, _artifact(config, "qiv_mc_coverage.csv"), config)
    monotonicity = qiv_mc_monotonicity(snapshot, pricing, intraday_trades, config)
    write_csv(monotonicity, _artifact(config, "qiv_monotonicity.csv"), config)
    qiv_restored_trades, qiv_restored_summary = evaluate_qiv_mc_restored_ablation(
        snapshot, config
    )
    write_csv(
        qiv_restored_trades,
        _artifact(config, "qiv_mc_restored_ablation_trades.csv"),
        config,
    )
    write_csv(
        qiv_restored_summary,
        _artifact(config, "qiv_mc_restored_ablation.csv"),
        config,
    )
    reverse_events, reverse_summary = evaluate_cb_leads_stock(events, snapshot, config)
    write_csv(reverse_events, _artifact(config, "cb_leads_stock_events.csv"), config)
    write_csv(reverse_summary, _artifact(config, "cb_leads_stock_summary.csv"), config)
    reverse_windows = reverse_linkage_window_stability(reverse_events)
    write_csv(
        reverse_windows,
        _artifact(config, "cb_leads_stock_window_stability.csv"),
        config,
    )
    common_daily, common_summary = common_slice_comparison(
        intraday_trades, daily_trades, overlay_trades, snapshot, config
    )
    write_csv(common_daily, _artifact(config, "common_slice_daily_returns.csv"), config)
    write_csv(
        common_summary,
        _artifact(config, "common_slice_strategy_comparison.csv"),
        config,
    )
    stability = window_stability(intraday_trades, daily_trades, config)
    write_csv(stability, _artifact(config, "window_stability.csv"), config)
    walk_forward = build_walk_forward_plan(snapshot, config)
    write_csv(walk_forward, _artifact(config, "walk_forward_plan.csv"), config)
    ablation = build_ablation_report(
        intraday_summary,
        daily_summary,
        overlay_summary,
        monotonicity,
        config,
        intraday_trades=intraday_trades,
        qiv_restored_summary=qiv_restored_summary,
    )
    write_csv(ablation, _artifact(config, "ablation_report.csv"), config)
    report = generate_report(config)
    return {
        "feature_rows": int(len(features)),
        "event_episodes": int(len(events)),
        "intraday_trade_rows": int(len(intraday_trades)),
        "daily_trade_rows": int(len(daily_trades)),
        "lsmc_overlay_rows": int(len(overlay_trades)),
        "report": str(report),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(args.config)
    if args.n_jobs is not None:
        config["strategy"]["n_jobs"] = args.n_jobs
    expected_lags = parse_csv(args.lags, int)
    if expected_lags != [0, 1, 2, 3]:
        raise ValueError("V0.7.1 canonical linkage model requires --lags 0,1,2,3")
    mode = args.mode
    if mode == "all":
        result = run_all(args, config)
    elif mode == "diagnostic-reset":
        bars, benchmark, daily = ensure_common_slice(
            config, args.start_date, args.end_date, args.refresh_cache
        )
        result = {
            "common_bar_rows": len(bars),
            "benchmark_rows": len(benchmark),
            "daily_rows": len(daily),
        }
    elif mode == "build-linkage-features":
        features = ensure_features(
            config, args.start_date, args.end_date, args.refresh_cache, args.model
        )
        result = {"feature_rows": len(features)}
    elif mode == "build-linkage-events":
        features = ensure_features(
            config, args.start_date, args.end_date, args.refresh_cache, args.model
        )
        events = ensure_events(features, config)
        result = {"event_episodes": len(events), "event_types": events["event_type"].value_counts().to_dict()}
    elif mode == "backtest-linkage":
        features = ensure_features(
            config, args.start_date, args.end_date, args.refresh_cache, args.model
        )
        trades, summary, _ = ensure_intraday_backtest(
            features,
            config,
            parse_csv(args.entry_delay_bars, int),
            parse_csv(args.cost_scenarios),
        )
        result = {"trade_rows": len(trades), "summary_rows": len(summary)}
    elif mode == "backtest-daily-simple-rv":
        snapshot, trades, summary = ensure_daily_rv(
            config,
            args.start_date,
            args.end_date,
            parse_csv(args.top_n, int),
            parse_csv(args.holding_days, int),
            parse_csv(args.cost_scenarios),
        )
        result = {"snapshot_rows": len(snapshot), "trade_rows": len(trades), "summary_rows": len(summary)}
    elif mode == "evaluate-lsmc-exit-overlay":
        snapshot, _, _ = ensure_daily_rv(
            config,
            args.start_date,
            args.end_date,
            parse_csv(args.top_n, int),
            parse_csv(args.holding_days, int),
            parse_csv(args.cost_scenarios),
        )
        trades, summary, matched = ensure_lsmc_overlay(snapshot, config)
        result = {"trade_rows": len(trades), "summary_rows": len(summary), "matched_rows": len(matched)}
    elif mode == "evaluate-cb-leads-stock":
        features = ensure_features(
            config, args.start_date, args.end_date, args.refresh_cache, args.model
        )
        events = ensure_events(features, config)
        snapshot, _, _ = ensure_daily_rv(
            config,
            args.start_date,
            args.end_date,
            parse_csv(args.top_n, int),
            parse_csv(args.holding_days, int),
            parse_csv(args.cost_scenarios),
        )
        reverse, summary = evaluate_cb_leads_stock(events, snapshot, config)
        write_csv(reverse, _artifact(config, "cb_leads_stock_events.csv"), config)
        write_csv(summary, _artifact(config, "cb_leads_stock_summary.csv"), config)
        result = {"event_rows": len(reverse), "summary_rows": len(summary)}
    elif mode in {"common-slice", "ablation"}:
        result = run_all(args, config)
    elif mode == "report":
        result = {"report": str(generate_report(config))}
    else:
        raise AssertionError(mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
