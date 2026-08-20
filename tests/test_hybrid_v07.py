from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from research.hybrid_v07_core import (
    add_daily_features,
    add_qiv_and_mc,
    classify_bonds,
    compute_liquidity_and_friction,
    load_hybrid_config,
    monte_carlo_price,
    proxy_convertible_price,
    run_lsmc_row,
    select_trade_horizon,
    solve_qiv,
)
from research.hybrid_v07_data import (
    attach_point_in_time_conversion_price,
    attach_point_in_time_static,
    derive_point_in_time_parity_and_premium,
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def synthetic_panel(days: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    rng = np.random.default_rng(7)
    stock_returns = rng.normal(0.0002, 0.012, days)
    bond_returns = 0.45 * stock_returns + rng.normal(0.0001, 0.004, days)
    stock = 20 * np.exp(np.cumsum(stock_returns))
    bond = 125 * np.exp(np.cumsum(bond_returns))
    amount = 8e7 * np.exp(rng.normal(0, 0.25, days))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "bond_code": "123001",
            "stock_code": "300001",
            "research_pool_scope": "ACTIVE_LIKE",
            "bond_open": bond * 0.999,
            "bond_high": bond * 1.01,
            "bond_low": bond * 0.99,
            "bond_close": bond,
            "bond_volume": amount / bond,
            "bond_amount": amount,
            "bond_valid_bars": 48,
            "bond_zero_bars": 0,
            "stock_open": stock * 0.999,
            "stock_high": stock * 1.01,
            "stock_low": stock * 0.99,
            "stock_close": stock,
            "stock_volume": amount / stock,
            "stock_amount": amount * 5,
            "stock_valid_bars": 48,
            "stock_zero_bars": 0,
            "daily_turnover": 2.0,
            "premium": np.nan,
            "parity": np.nan,
            "conversion_price": np.nan,
            "days_to_maturity": np.nan,
            "debt_ratio": np.nan,
            "operating_cash_flow": np.nan,
            "free_cash_flow": np.nan,
            "static_asof_valid": False,
        }
    )


def prepared_panel() -> tuple[pd.DataFrame, dict]:
    config = load_hybrid_config()
    panel = add_daily_features(synthetic_panel())
    panel = classify_bonds(panel, config)
    panel = compute_liquidity_and_friction(panel, config)
    for column, value in [
        ("qiv_microstructure_contaminated", False),
        ("qiv_relative_value_score", np.nan),
        ("mc_price_gap_pct", np.nan),
        ("qiv_smooth", np.nan),
    ]:
        panel[column] = value
    return panel, config


def test_point_in_time_static_rejects_future_snapshot() -> None:
    config = load_hybrid_config()
    hist = synthetic_panel(2)
    screening = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-02-01")],
            "bond_code": ["123001"],
            "conversion_price": [10.0],
            "premium": [25.0],
            "static_snapshot_time": [pd.Timestamp("2025-02-01 15:00:00")],
        }
    )
    result = attach_point_in_time_static(hist, screening, config)
    assert_true(result["conversion_price"].isna().all(), "future static snapshot leaked backward")
    assert_true(~result["static_asof_valid"].any(), "future snapshot marked point-in-time valid")


def test_conversion_price_interval_join_is_point_in_time() -> None:
    hist = synthetic_panel(3)
    hist["trade_date"] = pd.to_datetime(["2025-06-20", "2025-06-23", "2025-06-24"])
    history = pd.DataFrame(
        {
            "bond_code": ["123001", "123001"],
            "stock_code": ["300001", "300001"],
            "effective_start": pd.to_datetime(["2025-06-23", "2025-07-01"]),
            "effective_end_exclusive": pd.to_datetime(["2025-07-01", "2025-08-01"]),
            "conversion_price": [10.0, 9.5],
            "source_vendor": ["akshare+jqdata", "jqdata"],
            "price_source": ["pit_baseline", "jqdata_adjustment"],
            "confidence": ["HIGH", "HIGH"],
        }
    )
    result = attach_point_in_time_conversion_price(hist, history)
    assert_true(pd.isna(result.loc[0, "conversion_price"]), "future conversion price leaked backward")
    assert_true(result.loc[1, "conversion_price"] == 10.0, "effective interval was not joined")
    assert_true(result.loc[2, "conversion_price"] == 10.0, "active interval was not carried within its range")
    assert_true(not bool(result.loc[0, "conversion_price_asof_valid"]), "future interval marked valid")


def test_point_in_time_parity_and_premium_use_unadjusted_stock_price() -> None:
    frame = synthetic_panel(2)
    frame["conversion_price"] = [10.0, 10.0]
    frame["conversion_price_asof_valid"] = [True, True]
    frame["qiv_stock_close"] = [20.0, np.nan]
    frame["qiv_stock_close_source"] = ["jqdata_fq_none_daily", None]
    frame["stock_close"] = [200.0, 200.0]
    frame["bond_close"] = [210.0, 210.0]
    result = derive_point_in_time_parity_and_premium(frame)
    assert_true(np.isclose(result.loc[0, "parity"], 200.0), "parity did not use unadjusted stock close")
    assert_true(np.isclose(result.loc[0, "premium"], 5.0), "premium unit or parity basis is wrong")
    assert_true(pd.isna(result.loc[1, "parity"]), "adjusted stock close was used as pricing fallback")
    assert_true(not bool(result.loc[1, "pricing_inputs_asof_valid"]), "missing unadjusted price marked valid")

    screening = frame.iloc[:1].copy()
    screening["data_mode"] = "daily_screening_snapshot"
    screening["static_asof_valid"] = True
    screening["conversion_price_asof_valid"] = False
    screening["qiv_stock_close"] = np.nan
    screening["qiv_stock_close_source"] = None
    screening_result = derive_point_in_time_parity_and_premium(screening)
    assert_true(
        bool(screening_result.loc[screening_result.index[0], "pricing_inputs_asof_valid"]),
        "same-day screening conversion price was not accepted as PIT",
    )


def test_ma20_is_past_and_current_only() -> None:
    panel = add_daily_features(synthetic_panel(30))
    baseline = panel.loc[19, "bond_ma20d"]
    changed = synthetic_panel(30)
    changed.loc[29, "bond_close"] *= 10
    changed_result = add_daily_features(changed)
    assert_true(np.isclose(baseline, changed_result.loc[19, "bond_ma20d"]), "future price changed MA20")


def test_daily_signal_executes_next_open() -> None:
    panel = add_daily_features(synthetic_panel(5))
    expected = panel.loc[1, "bond_open"]
    assert_true(np.isclose(panel.loc[0, "next_open"], expected), "daily execution is not next open")
    assert_true(bool(panel.loc[0, "same_close_execution_forbidden"]), "same-close execution was not forbidden")


def test_qiv_and_mc_are_reproducible_without_ytm() -> None:
    config = load_hybrid_config()
    row = pd.Series(
        {
            "trade_date": pd.Timestamp("2026-01-05"),
            "bond_code": "123001",
            "bond_type": "BALANCED_CORE",
            "stock_close": 20.0,
            "conversion_price": 16.0,
            "days_to_maturity": 900,
            "stock_rv_20d": 0.35,
            "liquidity_regime": "LIQUIDITY_SWEET_SPOT",
            "distress_flag": False,
            "debt_ratio": 50,
            "operating_cash_flow": 1,
            "free_cash_flow": 1,
        }
    )
    spread = config["credit_spread_proxy"]["base_bps"]["BALANCED_CORE"] / 10000
    row["bond_close"] = float(
        proxy_convertible_price(
            row["stock_close"],
            row["conversion_price"],
            row["days_to_maturity"] / 365,
            config["lsmc"]["risk_free_rate"],
            spread,
            0.35,
        )
    )
    solved = solve_qiv(row, config)
    assert_true(solved["qiv_solve_success"], "qIV failed on an internally consistent proxy price")
    first = monte_carlo_price(row, config, paths=1000)
    second = monte_carlo_price(row, config, paths=1000)
    assert_true(first["mc_model_price"] == second["mc_model_price"], "fixed-seed MC is not reproducible")
    assert_true("ytm_unavailable" in first["model_assumption_flags"], "missing YTM was not documented")
    sensitivity = json.loads(first["mc_credit_spread_sensitivity"])
    assert_true(set(sensitivity) == {"-200", "-100", "0", "100", "200"}, "credit sensitivity grid changed")
    assert_true(sensitivity["-200"] > sensitivity["200"], "credit-spread sensitivity direction is wrong")


def test_qiv_prefers_unadjusted_stock_support_price() -> None:
    config = load_hybrid_config()
    row = pd.Series(
        {
            "trade_date": pd.Timestamp("2026-01-05"),
            "bond_code": "123001",
            "bond_type": "BALANCED_CORE",
            "stock_close": 200.0,
            "qiv_stock_close": 20.0,
            "conversion_price": 16.0,
            "days_to_maturity": 900,
            "stock_rv_20d": 0.35,
            "liquidity_regime": "LIQUIDITY_SWEET_SPOT",
            "distress_flag": False,
            "debt_ratio": 50,
            "operating_cash_flow": 1,
            "free_cash_flow": 1,
        }
    )
    spread = config["credit_spread_proxy"]["base_bps"]["BALANCED_CORE"] / 10000
    row["bond_close"] = float(
        proxy_convertible_price(
            row["qiv_stock_close"],
            row["conversion_price"],
            row["days_to_maturity"] / 365,
            config["lsmc"]["risk_free_rate"],
            spread,
            0.35,
        )
    )
    solved = solve_qiv(row, config)
    assumptions = json.loads(solved["qiv_model_assumptions"])
    assert_true(solved["qiv_solve_success"], "qIV ignored the unadjusted support price")
    assert_true(
        assumptions["stock_price_source"] == "jqdata_fq_none_daily",
        "qIV did not record the unadjusted stock price source",
    )


def test_qiv_does_not_use_future_row() -> None:
    config = load_hybrid_config()
    base = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
            "bond_code": ["123001", "123001"],
            "stock_code": ["300001", "300001"],
            "bond_type": ["BALANCED_CORE", "BALANCED_CORE"],
            "bond_close": [120.0, 120.0],
            "stock_close": [20.0, 200.0],
            "conversion_price": [18.0, 18.0],
            "days_to_maturity": [900, 899],
            "stock_rv_20d": [0.35, 0.35],
            "liquidity_regime": ["LIQUIDITY_SWEET_SPOT", "LIQUIDITY_SWEET_SPOT"],
            "distress_flag": [False, False],
            "debt_ratio": [50.0, 50.0],
            "operating_cash_flow": [1.0, 1.0],
            "free_cash_flow": [1.0, 1.0],
            "amount_z": [0.0, 0.0],
            "premium_change_z": [0.0, 0.0],
            "high_low_range_proxy_z": [0.0, 0.0],
            "close_location": [0.5, 0.5],
            "bond_return_1d": [0.0, 0.0],
            "stock_return_1d": [0.0, 0.0],
            "adv20_amount": [1e8, 1e8],
        }
    )
    first = add_qiv_and_mc(base, config, paths=100).loc[0, "qiv_raw"]
    changed = base.copy()
    changed.loc[1, "bond_close"] = 1000
    second = add_qiv_and_mc(changed, config, paths=100).loc[0, "qiv_raw"]
    assert_true((pd.isna(first) and pd.isna(second)) or np.isclose(first, second), "future row changed qIV")


def test_lsmc_reproducibility_and_no_future_training() -> None:
    panel, config = prepared_panel()
    row = panel.iloc[-1].copy()
    row["lsmc_eligible"] = True
    row["lsmc_exclusion_reason"] = ""
    history = panel.iloc[:-1].copy()
    first = run_lsmc_row(row, history, config, paths=256)
    second = run_lsmc_row(row, history, config, paths=256)
    assert_true(first.continuation_value == second.continuation_value, "LSMC fixed seed is not reproducible")
    expected_immediate = row["bond_close"] * (1 - row["total_friction_bps"] / 10000)
    assert_true(np.isclose(first.immediate_liquidation_value, expected_immediate), "liquidation friction missing")
    assert_true(
        np.isclose(first.hold_edge, first.continuation_value - first.immediate_liquidation_value),
        "hold edge arithmetic is wrong",
    )
    future = history.copy()
    future["trade_date"] = row["trade_date"] + pd.Timedelta(days=5)
    future["residual_daily"] = 5.0
    with_future = run_lsmc_row(row, pd.concat([history, future]), config, paths=256)
    assert_true(
        first.continuation_value == with_future.continuation_value,
        "future test-period rows leaked into LSMC paths",
    )
    assert_true(first.path_count == 512, "both configured path models were not used")


def test_lsmc_exclusions_and_experimental_distress() -> None:
    panel, config = prepared_panel()
    row = panel.iloc[-1].copy()
    row["bond_type"] = "CLAUSE_DRIVEN"
    row["lsmc_eligible"] = False
    row["lsmc_exclusion_reason"] = "CLAUSE_DRIVEN_EXCLUDED"
    result = run_lsmc_row(row, panel.iloc[:-1], config, paths=64)
    assert_true(result.path_count == 0 and result.recommended_action == "WATCH_ONLY", "clause bond entered LSMC")
    row["bond_type"] = "ILLIQUID"
    row["lsmc_exclusion_reason"] = "ILLIQUID_EXCLUDED"
    result = run_lsmc_row(row, panel.iloc[:-1], config, paths=64)
    assert_true(result.path_count == 0, "illiquid bond entered LSMC")
    row["bond_type"] = "DISTRESS_REVERSAL"
    row["lsmc_eligible"] = True
    row["lsmc_exclusion_reason"] = "DISTRESS_EXPERIMENTAL_ONLY"
    result = run_lsmc_row(row, panel.iloc[:-1], config, paths=64)
    assert_true("DISTRESS_EXPERIMENTAL_ONLY" in result.exclusion_reason, "distress mode lost experimental flag")


def selector_row(config: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bond_code": ["123001"],
            "stock_code": ["300001"],
            "bond_type": ["BALANCED_CORE"],
            "bond_close": [125.0],
            "residual_z_daily": [-2.0],
            "residual_std_20d": [0.01],
            "stock_return_1d": [0.01],
            "intraday_eligible": [True],
            "daily_holding_eligible": [True],
            "lsmc_eligible": [True],
            "liquidity_regime": ["LIQUIDITY_SWEET_SPOT"],
            "liquidity_sweet_spot_score": [0.9],
            "hot_money_risk_score": [0.1],
            "total_friction_bps": [5.0],
            "qiv_microstructure_contaminated": [False],
            "mc_price_gap_pct": [-0.02],
            "lsmc_hold_edge": [2.0],
            "lsmc_recommended_action": ["HOLD"],
        }
    )


def test_horizon_selector_gates_and_loss_prohibition() -> None:
    config = load_hybrid_config()
    config = deepcopy(config)
    config["intraday"]["min_net_edge_bps"] = 5
    frame = selector_row(config)
    selected = select_trade_horizon(frame, config)
    assert_true(selected.loc[0, "selected_horizon"] == "INTRADAY_T0", "valid intraday edge was not selected")
    weak = frame.copy()
    weak["total_friction_bps"] = 500
    weak_selected = select_trade_horizon(weak, config)
    assert_true(weak_selected.loc[0, "selected_horizon"] != "INTRADAY_T0", "edge below friction entered intraday")
    no_hold = frame.copy()
    no_hold["intraday_eligible"] = False
    no_hold["lsmc_hold_edge"] = -1
    no_hold_selected = select_trade_horizon(no_hold, config)
    assert_true(no_hold_selected.loc[0, "selected_horizon"] != "OVERNIGHT_DAILY", "negative hold edge selected HOLD")
    losing = frame.copy()
    losing["current_intraday_pnl"] = -0.01
    losing["lsmc_hold_edge"] = -1
    losing_selected = select_trade_horizon(losing, config)
    assert_true(
        losing_selected.loc[0, "carry_overnight_action"] == "FLATTEN_BEFORE_CLOSE",
        "losing intraday position was mechanically carried overnight",
    )
    assert_true(bool(losing_selected.loc[0, "loss_to_overnight_prohibition"]), "loss prohibition disabled")


def test_mc_price_gap_direction_in_intraday_edge() -> None:
    config = load_hybrid_config()
    cheap = selector_row(config)
    cheap["mc_price_gap_pct"] = 0.05
    cheap["residual_z_daily"] = -1.0
    cheap["residual_std_20d"] = 0.01
    rich = cheap.copy()
    rich["mc_price_gap_pct"] = -0.05
    cheap_edge = select_trade_horizon(cheap, config).loc[0, "expected_intraday_edge_bps"]
    rich_edge = select_trade_horizon(rich, config).loc[0, "expected_intraday_edge_bps"]
    assert_true(cheap_edge > rich_edge, "negative MC gap was treated as positive relative-value edge")


def test_momentum_and_ma_cannot_trigger_alone() -> None:
    config = load_hybrid_config()
    frame = selector_row(config)
    frame["residual_z_daily"] = 0.0
    frame["residual_std_20d"] = 0.0
    frame["mc_price_gap_pct"] = 0.0
    frame["lsmc_hold_edge"] = -1.0
    frame["stock_return_1d"] = 0.10
    frame["cb_ma20d_dev"] = 0.20
    frame["stock_ma20d_dev"] = 0.20
    selected = select_trade_horizon(frame, config)
    assert_true(
        selected.loc[0, "selected_horizon"] not in {"INTRADAY_T0", "OVERNIGHT_DAILY"},
        "momentum or MA triggered a standalone trade",
    )
    assert_true(config["intraday"]["momentum_weight"] == 0.05, "low momentum weight config changed")


def test_type_classification_and_grouping() -> None:
    config = load_hybrid_config()
    rows = pd.concat([synthetic_panel(25), synthetic_panel(25).assign(bond_code="123002")])
    rows = add_daily_features(rows)
    rows = compute_liquidity_and_friction(classify_bonds(rows, config), config)
    assert_true(rows["bond_type_version"].eq(config["bond_type"]["version"]).all(), "type version missing")
    illiquid = rows.iloc[-1:].copy()
    illiquid["adv20_amount"] = 1.0
    illiquid["active_bar_ratio"] = 0.1
    result = classify_bonds(illiquid, config)
    assert_true(result.iloc[0]["bond_type"] == "ILLIQUID", "illiquid classification failed")
    clause = illiquid.copy()
    clause["adv20_amount"] = 1e8
    clause["active_bar_ratio"] = 1.0
    clause["clause_driven_flag"] = True
    result = classify_bonds(clause, config)
    assert_true(result.iloc[0]["bond_type"] == "CLAUSE_DRIVEN", "clause classification failed")
    assert_true(not bool(result.iloc[0]["lsmc_eligible"]), "clause classification did not exclude LSMC")


def run_all() -> None:
    tests = [
        test_point_in_time_static_rejects_future_snapshot,
        test_conversion_price_interval_join_is_point_in_time,
        test_point_in_time_parity_and_premium_use_unadjusted_stock_price,
        test_ma20_is_past_and_current_only,
        test_daily_signal_executes_next_open,
        test_qiv_and_mc_are_reproducible_without_ytm,
        test_qiv_prefers_unadjusted_stock_support_price,
        test_qiv_does_not_use_future_row,
        test_lsmc_reproducibility_and_no_future_training,
        test_lsmc_exclusions_and_experimental_distress,
        test_horizon_selector_gates_and_loss_prohibition,
        test_mc_price_gap_direction_in_intraday_edge,
        test_momentum_and_ma_cannot_trigger_alone,
        test_type_classification_and_grouping,
    ]
    for test in tests:
        test()
    print(f"hybrid v0.7 tests passed: {len(tests)} groups")


if __name__ == "__main__":
    run_all()
