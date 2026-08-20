import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from intraday_factors.config import load_config, load_schema
import intraday_factors.data_source as data_source_module
from intraday_factors.data_source import IntradayDataAdapter, factor_row_timestamp
from intraday_factors.engine import FactorEngine
from intraday_factors.trading_session import is_session_bar, session_bar_role, slot_id
from intraday_factors.utils import point_in_time_asof_join, rolling_regression, same_slot_robust_z
from run_intraday_factors import effective_completed_bar, select_new_rows


def assert_true(value, message):
    if not bool(value):
        raise AssertionError(message)


def synthetic_panel(n=80, bonds=("111000", "127110")):
    times = pd.date_range("2026-06-25 09:30:00", periods=n, freq="5min")
    times = [ts for ts in times if is_session_bar(ts)]
    rows = []
    for bond_idx, bond_code in enumerate(bonds):
        stock_code = "600000" if bond_idx == 0 else "000001"
        for i, ts in enumerate(times):
            stock_close = 10 + bond_idx + 0.02 * i
            parity = 100 * stock_close / (10 + bond_idx)
            cb_close = 120 + bond_idx + 0.5 * (parity - 100) + 0.01 * i
            rows.append(
                {
                    "bar_start": ts,
                    "trade_date": ts.strftime("%Y-%m-%d"),
                    "slot_id": slot_id(ts),
                    "bond_code": bond_code,
                    "bond_name": f"bond{bond_idx}",
                    "stock_code": stock_code,
                    "stock_name": f"stock{bond_idx}",
                    "cb_open": cb_close - 0.02,
                    "cb_high": cb_close + 0.05,
                    "cb_low": cb_close - 0.05,
                    "cb_close": cb_close,
                    "cb_volume": 1000 + i,
                    "cb_amount": float(100000 + i * 100),
                    "cb_quote_count": 2,
                    "stock_open": stock_close - 0.01,
                    "stock_high": stock_close + 0.03,
                    "stock_low": stock_close - 0.03,
                    "stock_close": stock_close,
                    "stock_volume": 10000 + i,
                    "stock_amount": float(1000000 + i * 1000),
                    "stock_quote_count": 2,
                    "conversion_price": 10 + bond_idx,
                    "bar_conversion_price": 10 + bond_idx,
                    "conversion_price_source": "test",
                    "amount_used": float(100000 + i * 100),
                    "amount_source": "reported_amount",
                    "stock_amount_used": float(1000000 + i * 1000),
                }
            )
    return pd.DataFrame(rows)


def test_point_in_time_asof_join():
    left = pd.DataFrame({"bond_code": ["111000", "111000"], "bar_start": pd.to_datetime(["2026-06-01", "2026-07-01"])})
    right = pd.DataFrame(
        {
            "bond_code": ["111000", "111000"],
            "snapshot_ts": pd.to_datetime(["2026-06-15", "2026-07-02"]),
            "conversion_price": [11.0, 12.0],
        }
    )
    out = point_in_time_asof_join(left, right, "bond_code", "bar_start", "snapshot_ts", ["conversion_price"])
    assert_true(pd.isna(out.loc[0, "conversion_price"]), "Future conversion price leaked into historical bar")
    assert_true(out.loc[1, "conversion_price"] == 11.0, "As-of join did not use latest past conversion price")


def test_no_future_function():
    cfg = load_config()
    cfg["residual_model"]["min_observations"] = 8
    engine = FactorEngine(cfg, load_schema())
    panel = synthetic_panel()
    baseline = engine.compute_panel_factors(panel)
    changed = panel.copy()
    changed.loc[changed.index[-1], "cb_close"] *= 10
    rerun = engine.compute_panel_factors(changed)
    cutoff = baseline["bar_start"].sort_values().unique()[-5]
    left = baseline[baseline["bar_start"] < cutoff]["signal_score"].reset_index(drop=True)
    right = rerun[rerun["bar_start"] < cutoff]["signal_score"].reset_index(drop=True)
    assert_true(left.equals(right), "Changing future data changed past factor results")


def test_residual_model_known_beta():
    idx = pd.RangeIndex(60)
    x = pd.DataFrame({"r_stock": np.linspace(-0.01, 0.01, 60)})
    y = 2.0 * x["r_stock"] + 0.001
    params = rolling_regression(y, x, window=50, min_observations=20)
    beta = params["r_stock"].dropna().iloc[-1]
    assert_true(abs(beta - 2.0) < 0.05, f"Estimated beta too far from truth: {beta}")


def test_lunch_session_slots():
    assert_true(slot_id("2026-06-25 11:25:00") == 23, "Morning final slot mismatch")
    assert_true(slot_id("2026-06-25 11:30:00") is None, "Lunch boundary should not be a tradable slot")
    assert_true(slot_id("2026-06-25 13:00:00") == 24, "Afternoon first slot mismatch")


def test_missing_bar_no_fake_zero_return():
    panel = synthetic_panel(n=20, bonds=("111000",))
    panel = panel.drop(panel.index[5]).reset_index(drop=True)
    engine = FactorEngine(load_config(), load_schema())
    out = engine.compute_panel_factors(panel)
    assert_true(not (out["r_cb"] == 0).all(), "Missing bar produced all-zero returns")


def test_same_slot_standardization_past_only():
    rows = []
    for i, day in enumerate(pd.date_range("2026-06-01", periods=12, freq="D")):
        rows.append({"bond_code": "111000", "bar_start": day + pd.Timedelta(hours=9, minutes=30), "slot_id": 0, "x": float(i)})
    df = pd.DataFrame(rows)
    z1, _ = same_slot_robust_z(df, "x", ["bond_code"], "slot_id", 20, 10, 20)
    df2 = df.copy()
    df2.loc[11, "x"] = 1000.0
    z2, _ = same_slot_robust_z(df2, "x", ["bond_code"], "slot_id", 20, 10, 20)
    assert_true(z1.iloc[10] == z2.iloc[10], "Current/future same-slot value leaked into previous z-score")


def test_idempotent_row_timestamp():
    ts1 = factor_row_timestamp(pd.Timestamp("2026-06-25 10:00:00"), "111000", "v0.1")
    ts2 = factor_row_timestamp(pd.Timestamp("2026-06-25 10:00:00"), "111000", "v0.1")
    assert_true(ts1 == ts2, "Factor row timestamp is not deterministic")


def test_zero_mad_no_inf():
    df = pd.DataFrame(
        {
            "bond_code": ["111000"] * 12,
            "bar_start": pd.date_range("2026-06-01", periods=12, freq="D"),
            "slot_id": [0] * 12,
            "x": [1.0] * 12,
        }
    )
    z, _ = same_slot_robust_z(df, "x", ["bond_code"], "slot_id", 20, 10, 20)
    assert_true(not np.isinf(z.dropna()).any(), "Zero MAD produced infinite z-score")


def test_no_ytm_or_turnover_required():
    cfg = load_config()
    cfg["residual_model"]["min_observations"] = 8
    engine = FactorEngine(cfg, load_schema())
    out = engine.compute_panel_factors(synthetic_panel())
    assert_true(len(out) > 0, "Engine failed without YTM/live_turnover fields")


def test_amount_fallback_marker_survives():
    cfg = load_config()
    cfg["residual_model"]["min_observations"] = 8
    panel = synthetic_panel()
    panel.loc[:3, "amount_used"] = panel.loc[:3, "cb_volume"] * panel.loc[:3, "cb_close"]
    panel.loc[:3, "amount_source"] = "proxy_volume_close"
    out = FactorEngine(cfg, load_schema()).compute_panel_factors(panel)
    assert_true("proxy_volume_close" in set(out["amount_source"]), "amount proxy marker was lost")


def test_current_active_bonds_keep_observation_history():
    schema = load_schema()
    adapter = IntradayDataAdapter(schema)
    original_query = data_source_module.query_to_dataframe
    live_rows = pd.DataFrame(
        [
            {
                "bar_start": pd.Timestamp("2026-06-30 09:30:00"),
                "trade_date": "2026-06-30",
                "asset_type": "CB",
                "watchlist_status": "OBSERVE",
                "bond_code": "111000",
                "bond_name": "bond1",
                "stock_code": "600000",
                "stock_name": "stock1",
                "open": 120.0,
                "high": 121.0,
                "low": 119.0,
                "close": 120.5,
                "volume": 1000.0,
                "amount": 120500.0,
                "premium_rate": 25.0,
                "conversion_price": 10.0,
                "quote_count": 1,
            },
            {
                "bar_start": pd.Timestamp("2026-06-30 09:30:00"),
                "trade_date": "2026-06-30",
                "asset_type": "STOCK",
                "watchlist_status": "OBSERVE",
                "bond_code": "111000",
                "bond_name": "bond1",
                "stock_code": "600000",
                "stock_name": "stock1",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.05,
                "volume": 10000.0,
                "amount": 100500.0,
                "premium_rate": np.nan,
                "conversion_price": np.nan,
                "quote_count": 1,
            },
            {
                "bar_start": pd.Timestamp("2026-06-30 09:35:00"),
                "trade_date": "2026-06-30",
                "asset_type": "CB",
                "watchlist_status": "ACTIVE",
                "bond_code": "111000",
                "bond_name": "bond1",
                "stock_code": "600000",
                "stock_name": "stock1",
                "open": 120.5,
                "high": 121.5,
                "low": 120.0,
                "close": 121.0,
                "volume": 900.0,
                "amount": 108900.0,
                "premium_rate": 25.2,
                "conversion_price": 10.0,
                "quote_count": 1,
            },
            {
                "bar_start": pd.Timestamp("2026-06-30 09:35:00"),
                "trade_date": "2026-06-30",
                "asset_type": "STOCK",
                "watchlist_status": "ACTIVE",
                "bond_code": "111000",
                "bond_name": "bond1",
                "stock_code": "600000",
                "stock_name": "stock1",
                "open": 10.05,
                "high": 10.2,
                "low": 10.0,
                "close": 10.1,
                "volume": 11000.0,
                "amount": 111100.0,
                "premium_rate": np.nan,
                "conversion_price": np.nan,
                "quote_count": 1,
            },
            {
                "bar_start": pd.Timestamp("2026-06-30 09:30:00"),
                "trade_date": "2026-06-30",
                "asset_type": "CB",
                "watchlist_status": "OBSERVE",
                "bond_code": "222000",
                "bond_name": "bond2",
                "stock_code": "000001",
                "stock_name": "stock2",
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.2,
                "volume": 800.0,
                "amount": 80160.0,
                "premium_rate": 24.0,
                "conversion_price": 9.0,
                "quote_count": 1,
            },
        ]
    )

    def fake_query(_conn, sql):
        if "LAST(snapshot_id)" in sql:
            return pd.DataFrame({"snapshot_id": ["snap1"]})
        if "SELECT bond_code, watchlist_status" in sql:
            return pd.DataFrame({"bond_code": ["111000", "222000"], "watchlist_status": ["ACTIVE", "OBSERVE"]})
        if "SELECT * FROM" in sql and schema["tables"]["live_5m"] in sql:
            return live_rows.copy()
        if "SELECT ts, bond_code" in sql:
            return pd.DataFrame(columns=["ts", "bond_code", "stock_code", "stock_name", "conversion_price"])
        raise AssertionError(f"unexpected SQL: {sql}")

    data_source_module.query_to_dataframe = fake_query
    try:
        panel = adapter.build_panel(None, "2026-06-30 09:30:00", "2026-06-30 09:35:00")
    finally:
        data_source_module.query_to_dataframe = original_query

    assert_true(set(panel["bond_code"]) == {"111000"}, "OBSERVE-only bond leaked into factor panel")
    assert_true(
        pd.Timestamp("2026-06-30 09:30:00") in set(panel["bar_start"]),
        "Promoted active bond lost its observation-period history",
    )


def test_incremental_effective_asof_allows_live_skew():
    effective = effective_completed_bar(
        requested_asof=pd.Timestamp("2026-06-30 14:40:00"),
        latest_cb_bar=pd.Timestamp("2026-06-30 14:25:00"),
        latest_stock_bar=pd.Timestamp("2026-06-30 14:35:00"),
        bar_minutes=5,
    )
    assert_true(effective == pd.Timestamp("2026-06-30 14:25:00"), "CB/STOCK live skew stopped effective bar selection")


def test_incremental_no_new_bar_window_is_empty():
    result = pd.DataFrame(
        {
            "bar_start": pd.to_datetime(["2026-06-30 10:45:00", "2026-06-30 10:50:00"]),
            "bond_code": ["111000", "111000"],
        }
    )
    out = select_new_rows(
        result,
        completed_through=pd.Timestamp("2026-06-30 10:50:00"),
        effective_asof=pd.Timestamp("2026-06-30 10:50:00"),
    )
    assert_true(out.empty, "No-new-bar incremental window was not empty")


def test_incremental_after_lunch_can_select_completed_bar():
    effective = effective_completed_bar(
        requested_asof=pd.Timestamp("2026-06-30 13:10:00"),
        latest_cb_bar=pd.Timestamp("2026-06-30 13:05:00"),
        latest_stock_bar=pd.Timestamp("2026-06-30 13:05:00"),
        bar_minutes=5,
    )
    assert_true(effective == pd.Timestamp("2026-06-30 13:05:00"), "Incremental cutoff did not resume after lunch")


def test_incremental_close_boundary_uses_close_bar():
    effective = effective_completed_bar(
        requested_asof=pd.Timestamp("2026-06-30 15:10:00"),
        latest_cb_bar=pd.Timestamp("2026-06-30 15:00:00"),
        latest_stock_bar=pd.Timestamp("2026-06-30 15:00:00"),
        bar_minutes=5,
    )
    assert_true(effective == pd.Timestamp("2026-06-30 15:00:00"), "Close boundary 15:00 should be a factor snapshot bar")
    assert_true(session_bar_role("2026-06-30 15:00:00") == "CLOSE", "15:00 bar role should be CLOSE")


def signal_frame_for_alerts():
    rows = []
    for ts in pd.to_datetime(["2026-06-25 10:00:00", "2026-06-25 10:05:00", "2026-06-25 10:40:00", "2026-06-25 11:00:00"]):
        rows.append(
            {
                "bar_start": ts,
                "bond_code": "111000",
                "stock_code": "600000",
                "factor_residual_mr": 1.2,
                "factor_premium_mr": 0.5,
                "stock_momentum_score": 0.8,
                "factor_vol_response_gap": 0.2,
                "flow_confirmation": 0.2,
                "factor_vol_mr": np.nan,
                "stock_mom_30m_z": 0.8,
                "stock_mom_60m_z": 0.2,
                "cb_mom_15m_z": 0.1,
                "cb_flow_z": 0.1,
                "residual_z": -1.0,
                "premium_slot_z": 0.4,
                "relative_amount": 0.8,
                "residual_vol_ratio_z": np.nan,
                "high_low_range_proxy_z": np.nan,
                "noise_penalty": 0.0,
                "volatility_gate_pass": True,
                "data_quality_pass": True,
                "liquidity_pass": True,
            }
        )
    df = pd.DataFrame(rows)
    df.loc[2, ["stock_mom_30m_z", "cb_mom_15m_z", "cb_flow_z", "residual_z", "relative_amount"]] = [1.2, 0.7, 0.5, 0.0, 1.2]
    df.loc[3, ["stock_mom_30m_z", "stock_mom_60m_z", "residual_z", "premium_slot_z", "relative_amount"]] = [0.2, -0.1, -0.2, 1.0, 0.2]
    return df


def test_action_alert_cooldown_and_watch_downgrade():
    cfg = load_config()
    engine = FactorEngine(cfg, load_schema())
    df = signal_frame_for_alerts()
    engine._compose_signal(df)
    assert_true(df.loc[0, "signal_type"] == "LAG_REPAIR_LONG", "Lag repair signal did not trigger")
    assert_true(df.loc[0, "alert_state"] == "ACTION_LONG", "First action was not emitted")
    assert_true(bool(df.loc[1, "cooldown_suppressed"]), "Cooldown did not suppress repeated action")
    assert_true(df.loc[1, "alert_state"] != "ACTION_LONG", "Suppressed action still emitted ACTION_LONG")
    assert_true(df.loc[2, "signal_type"] == "MOMENTUM_BREAKOUT_LONG", "Momentum breakout signal did not trigger")
    assert_true(df.loc[3, "signal_type"] == "WATCH_LONG", "Watch-only row was not classified")
    assert_true(df.loc[3, "alert_state"] == "WATCH_ONLY", "WATCH_LONG should not be ACTION_LONG")


def test_close_bar_does_not_emit_new_action_long():
    cfg = load_config()
    engine = FactorEngine(cfg, load_schema())
    df = signal_frame_for_alerts().iloc[[0]].copy()
    df["bar_start"] = pd.Timestamp("2026-06-25 15:00:00")
    df["pool_state"] = "ACTIVE"
    engine._compose_signal(df)
    assert_true(df.iloc[0]["signal_type"] == "LAG_REPAIR_LONG", "Close bar should still compute the raw signal type")
    assert_true(df.iloc[0]["alert_state"] != "ACTION_LONG", "Close bar emitted a new ACTION_LONG")
    assert_true(not bool(df.iloc[0]["is_actionable_bar"]), "Close bar should not be actionable")


def test_observe_and_shadow_do_not_emit_action_long():
    cfg = load_config()
    engine = FactorEngine(cfg, load_schema())
    df = pd.concat([signal_frame_for_alerts().iloc[[0]], signal_frame_for_alerts().iloc[[0]]], ignore_index=True)
    df.loc[0, "pool_state"] = "OBSERVE"
    df.loc[1, "pool_state"] = "SHADOW_RESEARCH"
    engine._compose_signal(df)
    assert_true(not (df["alert_state"] == "ACTION_LONG").any(), "Non-active pool emitted ACTION_LONG")
    assert_true(set(df["suppression_reason"]) == {"pool_state_not_actionable"}, "Non-active pool suppression reason missing")


def test_action_rows_have_v03_episode_fields():
    cfg = load_config()
    engine = FactorEngine(cfg, load_schema())
    df = signal_frame_for_alerts().iloc[[0]].copy()
    df["pool_state"] = "ACTIVE"
    engine._compose_signal(df)
    for col in [
        "is_actionable_bar",
        "session_bar_role",
        "cooldown_suppressed",
        "suppression_reason",
        "no_new_action_after_suppressed",
        "watch_signal_type",
        "watch_missing_to_action",
        "exit_type",
    ]:
        assert_true(col in df.columns, f"missing V0.3 alert support field {col}")


def test_armed_long_is_not_default_action():
    cfg = load_config()
    engine = FactorEngine(cfg, load_schema())
    df = signal_frame_for_alerts().iloc[[0]].copy()
    df.loc[df.index[0], ["stock_mom_30m_z", "residual_z", "relative_amount", "flow_confirmation"]] = [0.5, -0.65, 0.45, -0.1]
    df["pool_state"] = "ACTIVE"
    engine._compose_signal(df)
    assert_true(df.iloc[0]["signal_type"] == "ARMED_LAG_LONG", "Near-lag setup did not become ARMED_LAG_LONG")
    assert_true(df.iloc[0]["alert_state"] == "ARMED_LONG", "ARMED_LAG_LONG should alert as ARMED_LONG")
    assert_true(df.iloc[0]["alert_state"] != "ACTION_LONG", "ARMED_LONG leaked into default action")


def test_watch_risky_subtype_and_gap_text():
    cfg = load_config()
    engine = FactorEngine(cfg, load_schema())
    df = signal_frame_for_alerts().iloc[[3]].copy()
    df.loc[df.index[0], ["premium_slot_z", "premium_change_10m_z", "residual_z", "high_low_range_proxy_z"]] = [1.8, 1.7, 0.9, 2.0]
    df["pool_state"] = "ACTIVE"
    engine._compose_signal(df)
    assert_true(df.iloc[0]["signal_type"] == "WATCH_LONG", "Risky watch should keep WATCH_LONG compatibility")
    assert_true(df.iloc[0]["watch_signal_type"] == "WATCH_RISKY", "Risky watch subtype missing")
    assert_true("风险" in df.iloc[0]["watch_missing_to_action"], "Risky watch did not explain risk")


def test_manual_alert_config_lists():
    cfg = load_config()
    assert_true("WATCH_LONG" not in set(cfg["manual_alert_mode"]["action_signal_types"]), "WATCH_LONG leaked into action signal types")
    assert_true("LAG_REPAIR_LONG" in set(cfg["manual_alert_mode"]["action_signal_types"]), "Lag repair is not configured as action")


if __name__ == "__main__":
    test_point_in_time_asof_join()
    test_no_future_function()
    test_residual_model_known_beta()
    test_lunch_session_slots()
    test_missing_bar_no_fake_zero_return()
    test_same_slot_standardization_past_only()
    test_idempotent_row_timestamp()
    test_zero_mad_no_inf()
    test_no_ytm_or_turnover_required()
    test_amount_fallback_marker_survives()
    test_current_active_bonds_keep_observation_history()
    test_incremental_effective_asof_allows_live_skew()
    test_incremental_no_new_bar_window_is_empty()
    test_incremental_after_lunch_can_select_completed_bar()
    test_incremental_close_boundary_uses_close_bar()
    test_action_alert_cooldown_and_watch_downgrade()
    test_close_bar_does_not_emit_new_action_long()
    test_observe_and_shadow_do_not_emit_action_long()
    test_action_rows_have_v03_episode_fields()
    test_armed_long_is_not_default_action()
    test_watch_risky_subtype_and_gap_text()
    test_manual_alert_config_lists()
    print("intraday factor tests passed")
