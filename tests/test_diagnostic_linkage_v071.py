from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from research.diagnostic_linkage_v071_core import (
    add_distributed_lag_for_bond,
    add_liquidity_and_events,
    bar_returns,
    choose_market_index_for_history,
    deduplicate_event_episodes,
    fit_regression,
    response_half_life,
)
from research.diagnostic_linkage_v071_data import (
    add_versions,
    assign_window,
    load_config,
    session_slot,
    valid_session_bar,
)
from research.diagnostic_linkage_v071_eval import (
    _daily_cost_components,
    _round_trip_cost_components,
    _simulate_daily_entry,
    _simulate_intraday_trade,
    block_bootstrap_interval,
    build_walk_forward_plan,
    common_slice_comparison,
    linkage_coefficient_reports,
    prepare_daily_simple_rv,
    qiv_mc_coverage_report,
    reverse_linkage_window_stability,
)
from research.hybrid_v07_data import attach_point_in_time_conversion_price


def config() -> dict:
    cfg = deepcopy(load_config())
    cfg["strategy"]["n_jobs"] = 1
    cfg["linkage"]["rolling_windows_days"] = [5]
    cfg["linkage"]["default_window_days"] = 5
    cfg["linkage"]["minimum_model_samples"] = 40
    cfg["linkage"]["model_confidence_full_samples"] = 100
    cfg["linkage"]["ridge_alpha"] = 0.001
    cfg["stock_shock"]["same_slot_min_days"] = 2
    cfg["stock_shock"]["same_slot_lookback_days"] = 10
    return cfg


def regression_frame(days: int = 14, bars: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(71)
    rows = []
    previous = [0.0, 0.0, 0.0]
    for trade_date in pd.bdate_range("2025-01-02", periods=days):
        previous = [0.0, 0.0, 0.0]
        for slot in range(bars):
            stock_idio = rng.normal(0, 0.004)
            market = rng.normal(0, 0.002)
            cb_return = (
                0.10 * stock_idio
                + 0.35 * previous[-1]
                + 0.20 * previous[-2]
                + 0.10 * previous[-3]
                + 0.40 * market
                + rng.normal(0, 0.00005)
            )
            rows.append(
                {
                    "trade_date": trade_date,
                    "bar_start": trade_date + pd.Timedelta(hours=9, minutes=30 + slot * 5),
                    "bar_slot": slot,
                    "bond_code": "123001",
                    "stock_code": "300001",
                    "stock_idiosyncratic_return": stock_idio,
                    "stock_return": stock_idio + 0.8 * market,
                    "cb_market_return": market,
                    "cb_return": cb_return,
                    "stock_index_model_r2": 0.5,
                }
            )
            previous.append(stock_idio)
            previous = previous[-3:]
    return pd.DataFrame(rows)


def event_frame(**overrides) -> pd.DataFrame:
    values = {
        "cb_amount_z_5m": 0.5,
        "high_low_range_proxy_z": 0.2,
        "premium_expansion_proxy_z": 0.0,
        "close_location": 0.6,
        "cb_vwap_dev": 0.0,
        "relative_amount": 1.0,
        "adv20_amount": 1e8,
        "active_bar_ratio": 1.0,
        "stock_shock_z": 1.5,
        "stock_return_z_unadjusted": 1.5,
        "stock_shock_amount_z": 0.5,
        "linkage_sample_count": 100,
        "unadjusted_linkage_sample_count": 100,
        "linkage_regression_r2": 0.05,
        "linkage_gap_3bar": 0.002,
        "unadjusted_linkage_gap_3bar": 0.002,
        "linkage_repair_started": True,
        "pit_universe_eligible": True,
        "linkage_residual_z": -1.0,
        "cb_shock_z": 0.0,
        "cb_to_stock_beta_actionable_sum": 0.0,
        "dominant_lead_direction": "STOCK_LEADS_CB",
    }
    values.update(overrides)
    return pd.DataFrame([values])


def intraday_path(prices: list[float] | None = None) -> pd.DataFrame:
    prices = prices or [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    date = pd.Timestamp("2026-03-03")
    rows = []
    for slot, price in enumerate(prices):
        rows.append(
            {
                "trade_date": date,
                "bar_start": date + pd.Timedelta(hours=9, minutes=30 + slot * 5),
                "bar_end": date + pd.Timedelta(hours=9, minutes=35 + slot * 5),
                "bar_slot": slot,
                "cb_open": price,
                "cb_high": price * 1.0001,
                "cb_low": price * 0.9999,
                "cb_close": price,
                "stock_idiosyncratic_return": 0.0,
                "stock_shock_z": 1.0,
                "linkage_gap_3bar": 0.001,
                "response_half_life_bars": 2.0,
                "high_low_range_proxy": 0.0002,
                "liquidity_regime_5m": "NORMAL",
                "close_location": 0.5,
            }
        )
    rows[0]["stock_idiosyncratic_return"] = 0.01
    return pd.DataFrame(rows)


def daily_frame(days: int = 12, bonds: tuple[str, ...] = ("123001", "123002")) -> pd.DataFrame:
    rows = []
    for bond_number, bond in enumerate(bonds):
        for index, date in enumerate(pd.bdate_range("2025-06-16", periods=days)):
            price = 110 + bond_number * 5 + index * 0.2
            rows.append(
                {
                    "trade_date": date,
                    "window_id": "W1_TRAIN",
                    "window_role": "train",
                    "bond_code": bond,
                    "stock_code": f"30000{bond_number + 1}",
                    "bond_type": "BALANCED_CORE",
                    "bond_open": price,
                    "bond_high": price * 1.01,
                    "bond_low": price * 0.99,
                    "bond_close": price + 0.1,
                    "stock_open": 10 + index * 0.01,
                    "stock_close": 10 + index * 0.01 + 0.02,
                    "conversion_price_asof_valid": True,
                    "daily_turnover": 2.0,
                    "premium": 22 + bond_number,
                    "adv20_amount": 1e8 + bond_number * 1e7,
                    "residual_z_daily": -1.0 + bond_number * 0.2,
                    "spread_proxy_bps": 2.0,
                    "market_impact_bps": 2.0,
                    "liquidity_risk_bps": 1.0,
                    "forward_return_1d": 0.001,
                    "forward_return_3d": 0.002,
                    "forward_return_5d": 0.003,
                    "mfe_5d": 0.01,
                    "mae_5d": -0.005,
                    "market_regime": "NEUTRAL",
                }
            )
    frame = pd.DataFrame(rows).sort_values(["bond_code", "trade_date"])
    grouped = frame.groupby("bond_code", sort=False)
    frame["next_open"] = grouped["bond_open"].shift(-1)
    frame["next_trade_date"] = grouped["trade_date"].shift(-1)
    return frame.reset_index(drop=True)


def test_01_stock_index_beta_uses_past_only() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.003, 500)
    history = pd.DataFrame(
        {
            "stock_return": 1.4 * x + rng.normal(0, 0.0001, 500),
            "index_CSI300_return": x,
            "index_CSI500_return": rng.normal(0, 0.003, 500),
        }
    )
    code, result = choose_market_index_for_history(history, ["CSI300", "CSI500"], 300)
    assert code == "CSI300"
    assert abs(result.coefficients[0] - 1.4) < 0.05


def test_02_market_index_selection_ignores_future_rows() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    base = pd.DataFrame(
        {
            "stock_return": x,
            "index_CSI300_return": x,
            "index_CSI500_return": rng.normal(size=400),
        }
    )
    changed = base.copy()
    changed.loc[350:, "index_CSI500_return"] = changed.loc[350:, "stock_return"]
    assert choose_market_index_for_history(base.iloc[:350], ["CSI300", "CSI500"], 100)[0] == choose_market_index_for_history(changed.iloc[:350], ["CSI300", "CSI500"], 100)[0]


def test_03_conversion_price_interval_is_point_in_time() -> None:
    panel = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2025-06-19", "2025-06-20"]), "bond_code": ["123001"] * 2}
    )
    history = pd.DataFrame(
        {
            "bond_code": ["123001"],
            "effective_start": pd.to_datetime(["2025-06-20"]),
            "effective_end_exclusive": pd.to_datetime(["2025-07-01"]),
            "conversion_price": [10.0],
            "source_vendor": ["local"],
            "price_source": ["pit"],
            "confidence": ["HIGH"],
        }
    )
    result = attach_point_in_time_conversion_price(panel, history)
    assert pd.isna(result.loc[0, "conversion_price"])
    assert result.loc[1, "conversion_price"] == 10.0


def test_04_historical_universe_is_explicitly_bias_flagged() -> None:
    assert config()["common_slice"]["universe_bias_flag"] == "FUTURE_SELECTED_TRAINING_SAMPLE_SOURCE"


def test_05_missing_bar_is_not_filled_as_zero_return() -> None:
    frame = pd.DataFrame(
        {
            "bond_code": ["1", "1"],
            "trade_date": pd.to_datetime(["2025-01-02"] * 2),
            "bar_slot": [0, 1],
            "close": [100.0, np.nan],
            "open": [100.0, np.nan],
        }
    )
    result = bar_returns(frame, "close", "open")
    assert pd.isna(result.iloc[1])


def test_06_lunch_timestamp_is_not_a_valid_bar() -> None:
    timestamps = pd.Series(pd.to_datetime(["2025-01-02 11:25", "2025-01-02 12:00", "2025-01-02 13:00"]))
    assert valid_session_bar(timestamps).tolist() == [True, False, True]
    assert session_slot(timestamps).iloc[2] == 24


def test_07_intraday_signal_enters_next_bar_open() -> None:
    result = _simulate_intraday_trade(intraday_path(), 0, 0, "fixed", config())
    assert result["entry_time"] == intraday_path().iloc[1]["bar_start"]


def test_08_daily_signal_enters_next_trading_day_open() -> None:
    frame = prepare_daily_simple_rv(daily_frame(bonds=("123001",)), config())
    result = _simulate_daily_entry(frame.reset_index(drop=True), 0, 1)
    assert result["entry_date"] == frame.iloc[1]["trade_date"]
    assert result["same_close_execution_forbidden"]


def test_09_distributed_lag_recovers_known_beta() -> None:
    result = add_distributed_lag_for_bond(regression_frame(), config())
    latest = result.dropna(subset=["stock_to_cb_beta_lag1"]).iloc[-1]
    assert abs(latest["stock_to_cb_beta_lag1"] - 0.35) < 0.08
    assert abs(latest["stock_to_cb_beta_lag2"] - 0.20) < 0.08


def test_10_lag0_is_not_in_actionable_sum() -> None:
    result = add_distributed_lag_for_bond(regression_frame(), config()).dropna(
        subset=["stock_to_cb_beta_lag1"]
    )
    row = result.iloc[-1]
    expected = row["stock_to_cb_beta_lag1"] + row["stock_to_cb_beta_lag2"] + row["stock_to_cb_beta_lag3"]
    assert np.isclose(row["stock_to_cb_beta_actionable_sum"], expected)


def test_11_lags_do_not_cross_trade_date() -> None:
    result = add_distributed_lag_for_bond(regression_frame(), config())
    first = result.groupby("trade_date").head(1)
    assert first["stock_idiosyncratic_return"].notna().all()


def test_12_market_component_regression_is_correct() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1000, 1))
    y = 0.001 + 1.2 * x[:, 0] + rng.normal(0, 0.01, 1000)
    fitted = fit_regression(x, y, "ridge", 0.001, 100)
    assert abs(fitted.coefficients[0] - 1.2) < 0.02


def test_13_linkage_coefficients_before_future_mutation_are_unchanged() -> None:
    base = regression_frame()
    first = add_distributed_lag_for_bond(base, config())
    changed = base.copy()
    changed.loc[changed["trade_date"] == changed["trade_date"].max(), "cb_return"] *= 10
    second = add_distributed_lag_for_bond(changed, config())
    previous_date = sorted(base["trade_date"].unique())[-2]
    columns = ["stock_to_cb_beta_lag1", "stock_to_cb_beta_lag2", "stock_to_cb_beta_lag3"]
    assert np.allclose(
        first.loc[first["trade_date"] == previous_date, columns],
        second.loc[second["trade_date"] == previous_date, columns],
        equal_nan=True,
    )


def test_14_response_half_life_is_correct() -> None:
    assert response_half_life(0.6, 0.3, 0.1) == 1
    assert response_half_life(0.1, 0.6, 0.3) == 2


def test_15_reverse_linkage_does_not_use_future_stock_return() -> None:
    base = regression_frame()
    first = add_distributed_lag_for_bond(base, config())
    changed = base.copy()
    changed.loc[changed.index[-1], "stock_idiosyncratic_return"] = 1.0
    second = add_distributed_lag_for_bond(changed, config())
    assert np.isclose(
        first.iloc[-2]["cb_to_stock_beta_lag1"], second.iloc[-2]["cb_to_stock_beta_lag1"], equal_nan=True
    )


def test_16_insufficient_history_is_unavailable() -> None:
    short = regression_frame(days=2, bars=5)
    result = add_distributed_lag_for_bond(short, config())
    assert result["linkage_model_source"].eq("unavailable_insufficient_history").all()


def test_17_stock_leads_cb_repair_event_conditions() -> None:
    result = add_liquidity_and_events(event_frame(), config())
    assert result.iloc[0]["event_type"] == "STOCK_LEADS_CB_REPAIR"


def test_18_no_response_is_not_a_buy_event() -> None:
    result = add_liquidity_and_events(
        event_frame(linkage_repair_started=False, relative_amount=0.3, cb_amount_z_5m=-1.0), config()
    )
    assert result.iloc[0]["event_type"] == "STOCK_LEADS_CB_NO_RESPONSE"
    assert result.iloc[0]["event_action_state"] == "REJECT_LAG"


def test_19_cb_leads_stock_never_becomes_intraday_buy() -> None:
    result = add_liquidity_and_events(
        event_frame(
            stock_shock_z=0.0,
            stock_return_z_unadjusted=0.0,
            cb_shock_z=2.0,
            cb_to_stock_beta_actionable_sum=0.5,
            dominant_lead_direction="CB_LEADS_STOCK",
        ),
        config(),
    )
    assert result.iloc[0]["event_type"] == "CB_LEADS_STOCK_INFORMATION"
    assert result.iloc[0]["event_action_state"] == "WATCH_ONLY"


def test_20_hot_money_overshoot_does_not_chase() -> None:
    result = add_liquidity_and_events(
        event_frame(linkage_gap_3bar=-0.001, cb_amount_z_5m=3.0, premium_expansion_proxy_z=2.0),
        config(),
    )
    assert result.iloc[0]["event_type"] == "CB_OVERSHOOT_HOT_MONEY"
    assert result.iloc[0]["event_action_state"] == "DO_NOT_CHASE"


def test_21_continuous_event_bars_merge_into_one_episode() -> None:
    date = pd.Timestamp("2025-01-02")
    rows = []
    for slot in [1, 2, 3]:
        row = event_frame().iloc[0].to_dict()
        row.update(
            {
                "trade_date": date,
                "bond_code": "123001",
                "stock_code": "300001",
                "bar_slot": slot,
                "bar_start": date + pd.Timedelta(hours=9, minutes=30 + slot * 5),
                "bar_end": date + pd.Timedelta(hours=9, minutes=35 + slot * 5),
                "event_type": "STOCK_LEADS_CB_REPAIR",
            }
        )
        rows.append(row)
    assert len(deduplicate_event_episodes(pd.DataFrame(rows), config())) == 1


def test_22_episode_does_not_cross_trade_date() -> None:
    rows = []
    for date in pd.to_datetime(["2025-01-02", "2025-01-03"]):
        row = event_frame().iloc[0].to_dict()
        row.update(
            {
                "trade_date": date,
                "bond_code": "123001",
                "stock_code": "300001",
                "bar_slot": 1,
                "bar_start": date + pd.Timedelta(hours=9, minutes=35),
                "bar_end": date + pd.Timedelta(hours=9, minutes=40),
                "event_type": "STOCK_LEADS_CB_REPAIR",
            }
        )
        rows.append(row)
    assert len(deduplicate_event_episodes(pd.DataFrame(rows), config())) == 2


def test_23_stock_reversal_stop() -> None:
    path = intraday_path()
    path.loc[1, "stock_idiosyncratic_return"] = -0.02
    result = _simulate_intraday_trade(path, 0, 0, "dynamic", config())
    assert result["exit_reason"] == "STOCK_REVERSAL_STOP"


def test_24_linkage_failure_stop() -> None:
    cfg = config()
    cfg["exits"]["stock_reversal_fraction"] = 10.0
    path = intraday_path()
    path.loc[0, "response_half_life_bars"] = 1
    result = _simulate_intraday_trade(path, 0, 0, "dynamic", cfg)
    assert result["exit_reason"] == "LINKAGE_FAILURE_STOP"


def test_25_time_stop_uses_response_half_life() -> None:
    cfg = config()
    cfg["exits"]["use_linkage_failure_stop"] = False
    cfg["exits"]["stock_reversal_fraction"] = 10.0
    path = intraday_path()
    path.loc[0, "response_half_life_bars"] = 2
    result = _simulate_intraday_trade(path, 0, 0, "dynamic", cfg)
    assert result["exit_reason"] == "TIME_STOP"
    assert result["holding_bars"] >= 2


def test_26_linkage_repaired_take_profit() -> None:
    prices = [100.0, 100.0, 100.08, 100.08, 100.08, 100.08]
    result = _simulate_intraday_trade(intraday_path(prices), 0, 0, "dynamic", config())
    assert result["exit_reason"] == "LINKAGE_REPAIRED"


def test_27_overshoot_exit() -> None:
    prices = [100.0, 100.0, 100.2, 100.2, 100.2, 100.2]
    result = _simulate_intraday_trade(intraday_path(prices), 0, 0, "dynamic", config())
    assert result["exit_reason"] == "OVERSHOOT_EXIT"


def test_28_fixed_percent_is_only_a_benchmark_exit() -> None:
    result = _simulate_intraday_trade(intraday_path(), 0, 0, "fixed_percent", config())
    assert result["exit_reason"].startswith("FIXED_")


def test_29_daily_rv_entry_does_not_depend_on_lsmc() -> None:
    frame = daily_frame()
    first = prepare_daily_simple_rv(frame, config())["daily_rv_score_simple"]
    frame["lsmc_hold_edge"] = np.arange(len(frame)) * 1000
    second = prepare_daily_simple_rv(frame, config())["daily_rv_score_simple"]
    assert np.allclose(first, second, equal_nan=True)


def test_30_lsmc_entry_is_disabled() -> None:
    assert config()["lsmc"]["entry_enabled"] is False
    assert config()["lsmc"]["exit_overlay_enabled"] is True


def test_31_exit_overlay_contract_requires_matched_entries() -> None:
    assert config()["lsmc"]["diagnostic_only"] is True
    assert config()["lsmc"]["overlay_max_holding_days"] == 5


def test_32_loss_to_overnight_prohibition_is_preserved() -> None:
    assert config()["lsmc"]["loss_to_overnight_prohibition"] is True


def test_33_intraday_loss_cannot_activate_lsmc_entry() -> None:
    result = add_liquidity_and_events(event_frame(), config())
    assert not bool(result.iloc[0]["lsmc_entry_enabled"])


def test_34_cost_waterfall_sums_exactly() -> None:
    signal = pd.Series(
        {
            "cb_close": 100.0,
            "cb_high": 100.2,
            "cb_low": 99.8,
            "cb_amount": 1e7,
            "adv20_amount": 1e8,
            "liquidity_sweet_spot_score_5m": 0.8,
        }
    )
    cost = _round_trip_cost_components(signal, 100.1, "realistic", config())
    trading_parts = sum(
        cost[key]
        for key in [
            "commission_cost_bps",
            "spread_cost_bps",
            "market_impact_bps",
            "liquidity_penalty_bps",
        ]
    )
    assert np.isclose(trading_parts, cost["total_cost_bps"])
    assert np.isclose(
        cost["total_cost_bps"] + cost["latency_slippage_bps"],
        cost["total_implementation_shortfall_bps"],
    )


def test_35_zero_cost_and_net_return_are_separate() -> None:
    signal = pd.Series({"cb_close": 100.0})
    cost = _round_trip_cost_components(signal, 100.0, "zero", config())
    assert cost["total_cost_bps"] == 0


def test_36_daily_type_rank_is_computed_within_type() -> None:
    result = prepare_daily_simple_rv(daily_frame(), config())
    assert result.groupby(["trade_date", "bond_type"])["premium_rank_within_type"].max().dropna().le(1).all()


def test_37_common_slice_uses_same_date_index() -> None:
    daily = prepare_daily_simple_rv(daily_frame(), config())
    empty_intraday = pd.DataFrame(
        columns=["strategy", "entry_delay_bars", "cost_scenario", "trade_date", "net_return"]
    )
    empty_daily = pd.DataFrame(
        columns=["selector", "exit_rule", "cost_scenario", "trade_date", "net_return"]
    )
    panel, _ = common_slice_comparison(empty_intraday, empty_daily, pd.DataFrame(), daily, config())
    assert panel["trade_date"].nunique() == daily["trade_date"].nunique()
    assert panel["common_slice_pool"].nunique() == 1


def test_38_walk_forward_is_not_random() -> None:
    frame = daily_frame(days=12)
    frame = assign_window(frame, config())
    plan = build_walk_forward_plan(frame, config())
    assert not plan["random_split"].any()


def test_39_purging_and_embargo_are_active() -> None:
    frame = daily_frame(days=12)
    frame = assign_window(frame, config())
    plan = build_walk_forward_plan(frame, config())
    later = plan.iloc[1:]
    assert later["purging_applied"].all()
    assert later["embargo_days"].eq(1).all()


def test_40_block_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        {"trade_date": pd.bdate_range("2025-01-01", periods=20), "value": np.arange(20) / 1000}
    )
    assert block_bootstrap_interval(frame, "value", 7, 100) == block_bootstrap_interval(frame, "value", 7, 100)


def test_41_all_outputs_can_record_versions() -> None:
    result = add_versions(pd.DataFrame({"x": [1]}), config())
    assert result.iloc[0]["strategy_version"] == "diagnostic_linkage_v071"
    assert result.iloc[0]["factor_version"] == "linkage_v071"


def test_42_qiv_and_mc_formal_weights_are_zero() -> None:
    result = prepare_daily_simple_rv(daily_frame(), config())
    assert result["qiv_formal_weight"].eq(0).all()
    assert result["mc_gap_formal_weight"].eq(0).all()


def test_43_momentum_is_not_in_daily_rv_score() -> None:
    frame = daily_frame()
    first = prepare_daily_simple_rv(frame, config())["daily_rv_score_simple"]
    frame["stock_return_1d"] = np.linspace(-10, 10, len(frame))
    second = prepare_daily_simple_rv(frame, config())["daily_rv_score_simple"]
    assert np.allclose(first, second, equal_nan=True)


def test_44_missing_conversion_price_is_daily_ineligible() -> None:
    frame = daily_frame()
    frame.loc[0, "conversion_price_asof_valid"] = False
    result = prepare_daily_simple_rv(frame, config())
    assert not bool(result.loc[0, "daily_rv_eligible"])


def test_45_qiv_contamination_cannot_reactivate_formal_weight() -> None:
    frame = daily_frame()
    pricing = frame[["trade_date", "bond_code"]].copy()
    pricing["qiv_smooth"] = 9.0
    pricing["qiv_microstructure_contaminated"] = True
    result = prepare_daily_simple_rv(frame, config(), pricing)
    assert result["qiv_formal_weight"].eq(0).all()


def test_46_qiv_coverage_keeps_formal_weight_zero() -> None:
    pricing = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "bond_code": ["123001", "123001"],
            "bond_type": ["BALANCED_CORE", "BALANCED_CORE"],
            "qiv_smooth": [0.30, np.nan],
            "qiv_solve_success": [True, False],
            "mc_price_gap_pct": [0.01, np.nan],
            "qiv_contamination_assessable": [True, False],
            "qiv_microstructure_contaminated": [True, False],
        }
    )
    result = qiv_mc_coverage_report(pricing, config())
    overall = result[result["scope"].eq("ALL")].iloc[0]
    assert np.isclose(overall["qiv_smooth_coverage"], 0.5)
    assert np.isclose(overall["contamination_rate_when_assessable"], 1.0)
    assert overall["formal_decision_weight"] == 0


def test_47_reverse_linkage_stability_is_window_separated() -> None:
    events = pd.DataFrame(
        {
            "window_id": ["W1", "W2"],
            "episode_id": ["e1", "e2"],
            "bond_code": ["123001", "123001"],
            "trade_date": pd.to_datetime(["2025-01-02", "2025-03-03"]),
            "stock_next_open_return": [0.01, -0.01],
            "stock_next_close_return": [0.02, -0.02],
            "bond_next_open_return": [0.01, -0.01],
            "bond_next_close_return": [0.02, -0.02],
            "forward_return_3d": [0.03, -0.03],
        }
    )
    result = reverse_linkage_window_stability(events)
    assert result["window_id"].tolist() == ["W1", "W2"]
    assert result["event_count"].eq(1).all()


def test_48_lead_lag_by_bond_remains_one_row_when_type_changes() -> None:
    frame = pd.DataFrame(
        {
            "bar_start": pd.to_datetime(["2025-01-02 14:55", "2025-01-03 14:55"]),
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "bond_code": ["123001", "123001"],
            "stock_code": ["300001", "300001"],
            "bond_type": ["BALANCED_CORE", "EQUITY_LIKE"],
            "stock_to_cb_beta_lag0": [0.2, 0.3],
            "stock_to_cb_beta_lag1": [0.1, 0.2],
            "stock_to_cb_beta_lag2": [0.0, 0.1],
            "stock_to_cb_beta_lag3": [0.0, 0.0],
            "stock_to_cb_beta_actionable_sum": [0.1, 0.3],
            "cb_to_stock_beta_actionable_sum": [0.0, 0.1],
            "linkage_regression_r2": [0.4, 0.5],
            "response_half_life_bars": [1.0, 2.0],
            "linkage_model_confidence": [0.8, 0.9],
        }
    )
    _, by_bond, _ = linkage_coefficient_reports(frame, config())
    assert len(by_bond) == 1
    assert by_bond.iloc[0]["bond_type_count"] == 2


def run_all() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"diagnostic linkage v0.7.1 tests passed: {len(tests)} groups")


if __name__ == "__main__":
    run_all()
