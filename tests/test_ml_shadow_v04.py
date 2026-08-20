import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from intraday_factors.config import load_config
from research.ml_shadow_v04 import (
    LABEL_COLUMNS,
    _training_frame,
    _feature_columns,
    build_predictions,
    build_ml_event_dataset_from_frames,
    build_walk_forward_splits,
)


def assert_true(value, message):
    if not bool(value):
        raise AssertionError(message)


def _factor_row(ts, signal_type="NONE", alert_state="INFO_ONLY", bond_code="111000", stock_code="600000"):
    i = int((pd.Timestamp(ts).hour * 60 + pd.Timestamp(ts).minute) / 5)
    return {
        "bar_start": pd.Timestamp(ts),
        "bond_code": bond_code,
        "stock_code": stock_code,
        "pool_state": "ACTIVE",
        "alert_state": alert_state,
        "signal_type": signal_type,
        "watch_signal_type": "WATCH_LAG" if signal_type == "WATCH_LONG" else "",
        "is_actionable_bar": True,
        "session_bar_role": "CONTINUOUS",
        "parity": 105.0 + 0.01 * i,
        "premium_raw": 0.25,
        "premium_slot_z": -0.2,
        "adv20_amount": 1000000.0,
        "relative_amount": 1.0,
        "residual_z": -1.0 + 0.05 * (i % 10),
        "residual_ewm": -0.001,
        "factor_residual_mr": 1.0,
        "residual_z_change_1bar": 0.1,
        "residual_z_change_2bar": 0.2,
        "residual_z_slope_3bar": 0.1,
        "residual_repair_started": True,
        "stock_mom_5m_z": 0.3,
        "stock_mom_10m_z": 0.4,
        "stock_mom_15m_z": 0.5,
        "stock_mom_30m_z": 1.0,
        "stock_mom_60m_z": 0.2,
        "stock_impulse_score": 0.6,
        "stock_amount_burst_z": 0.5,
        "cb_mom_5m_z": 0.1,
        "cb_mom_10m_z": 0.2,
        "cb_mom_15m_z": 0.3,
        "cb_mom_30m_z": 0.2,
        "premium_change_5m_z": 0.0,
        "premium_change_10m_z": 0.0,
        "premium_change_30m_z": 0.0,
        "cb_vwap_dev": 0.0,
        "stock_vwap_dev": 0.002,
        "vwap_gap": 0.002,
        "cb_flow_z": 0.2,
        "stock_flow_z": 0.1,
        "flow_confirmation": 0.15,
        "flow_lead": 0.1,
        "residual_vol_ratio_z": 0.1,
        "high_low_range_proxy_z": 0.1,
        "volatility_gate_pass": True,
        "noise_penalty": 0.0,
        "active_bar_ratio": 1.0,
        "cb_market_mom_30m": 0.001,
        "cb_market_breadth_30m": 0.6,
        "cb_market_breadth_60m": 0.6,
        "cb_market_amount_burst": 1.0,
        "market_regime_intraday": "RISK_ON",
        "amount_source": "reported_amount",
        "setup_score": 0.7,
        "trigger_score": 1.1,
        "signal_score": 0.5,
        "beta": 0.5,
        "gamma": 0.0,
    }


def _panel_row(ts, close, bond_code="111000", stock_code="600000"):
    return {
        "bar_start": pd.Timestamp(ts),
        "bond_code": bond_code,
        "stock_code": stock_code,
        "cb_open": close - 0.02,
        "cb_high": close + 0.05,
        "cb_low": close - 0.05,
        "cb_close": close,
        "stock_open": 10.0,
        "stock_high": 10.1,
        "stock_low": 9.9,
        "stock_close": 10.0 + close / 10000.0,
        "amount_used": 1000000.0,
        "stock_amount_used": 2000000.0,
    }


def synthetic_frames():
    times = [
        "2026-07-01 09:30:00",
        "2026-07-01 09:35:00",
        "2026-07-01 09:40:00",
        "2026-07-01 09:45:00",
        "2026-07-01 09:50:00",
        "2026-07-01 09:55:00",
        "2026-07-01 10:00:00",
        "2026-07-01 10:05:00",
        "2026-07-01 14:45:00",
        "2026-07-01 14:50:00",
        "2026-07-01 14:55:00",
        "2026-07-01 15:00:00",
        "2026-07-02 09:30:00",
        "2026-07-02 09:35:00",
    ]
    factors = []
    panel = []
    for idx, ts in enumerate(times):
        signal_type = "NONE"
        alert_state = "INFO_ONLY"
        if ts == "2026-07-01 09:30:00":
            signal_type = "ARMED_LAG_LONG"
            alert_state = "ARMED_LONG"
        if ts in {"2026-07-01 09:35:00", "2026-07-01 09:40:00"}:
            signal_type = "LAG_REPAIR_LONG"
            alert_state = "ACTION_LONG"
        if ts == "2026-07-01 14:45:00":
            signal_type = "MOMENTUM_BREAKOUT_LONG"
            alert_state = "ACTION_LONG"
        if ts == "2026-07-01 15:00:00":
            signal_type = "LAG_REPAIR_LONG"
            alert_state = "ACTION_LONG"
        if ts == "2026-07-02 09:30:00":
            signal_type = "WATCH_LONG"
            alert_state = "WATCH_ONLY"
        factors.append(_factor_row(ts, signal_type, alert_state))
        panel.append(_panel_row(ts, 120.0 + idx * 0.1))
    factors.append(_factor_row("2026-07-01 15:00:00", "LAG_REPAIR_LONG", "ACTION_LONG", bond_code="222000", stock_code="600001"))
    panel.append(_panel_row("2026-07-01 15:00:00", 130.0, bond_code="222000", stock_code="600001"))
    return pd.DataFrame(factors), pd.DataFrame(panel)


def test_t0_dataset_labels_and_dedupe():
    cfg = load_config()
    factors, panel = synthetic_frames()
    ds = build_ml_event_dataset_from_frames(factors, panel, cfg, event_scope="ARMED,ACTION,WATCH")
    assert_true(not ds.empty, "dataset should build without ytm/live_turnover fields")

    actions = ds[ds["event_type"] == "ACTION"]
    assert_true(len(actions[actions["signal_time"].dt.strftime("%H:%M:%S") == "09:35:00"]) == 1, "first ACTION missing")
    assert_true(len(actions[actions["signal_time"].dt.strftime("%H:%M:%S") == "09:40:00"]) == 0, "continuous ACTION was not deduped")

    armed = ds[ds["event_type"] == "ARMED"].iloc[0]
    assert_true(armed["label_armed_to_action_1bar"] == 1.0, "ARMED -> ACTION 1bar label failed")
    assert_true(armed["label_armed_to_action_3bar"] == 1.0, "ARMED -> ACTION 3bar label failed")

    late = ds[ds["signal_time"].dt.strftime("%H:%M:%S") == "14:45:00"].iloc[0]
    assert_true(bool(late["no_new_action_after_suppressed"]), "late signal was not marked no_new_action_after_suppressed")
    assert_true(not bool(late["eligible_for_action_training"]), "late ACTION entered default training sample")
    assert_true(pd.isna(late["return_30m"]), "tail return_30m should be unavailable")
    assert_true(bool(late["label_unavailable_due_to_eod"]), "tail unavailable flag missing")
    assert_true(pd.notna(late["return_to_eod"]), "return_to_eod should use same-day force flatten price")

    close_bar = ds[ds["signal_time"].dt.strftime("%H:%M:%S") == "15:00:00"].iloc[0]
    assert_true(not bool(close_bar["is_actionable_bar"]), "15:00 bar should not be actionable")
    assert_true(not bool(close_bar["eligible_for_action_training"]), "15:00 ACTION leaked into training")

    watch_train = _training_frame(ds, "label_action_good_30m", event_scope="WATCH,ACTION")
    assert_true("WATCH" not in set(watch_train["event_type"]), "WATCH leaked into default action training")

    assert_true((pd.to_datetime(ds["force_flatten_time"]) <= pd.to_datetime(ds["trade_date"] + " 14:55:00")).all(), "force flatten exceeded same day")
    assert_true((ds["first_seen_time"] == ds["signal_time"]).all(), "first_seen_time did not fallback to signal_time")
    assert_true(set(ds.loc[ds["eligible_for_action_training"].fillna(False), "pool_state"]).issubset({"ACTIVE", "ACTIVE_MANUAL"}), "non-official pool leaked into action training")


class ConstantModel:
    def predict_proba(self, frame):
        return np.column_stack([np.full(len(frame), 0.3), np.full(len(frame), 0.7)])


def test_ml_features_and_predictions_do_not_leak_or_override_alerts():
    cfg = load_config()
    factors, panel = synthetic_frames()
    ds = build_ml_event_dataset_from_frames(factors, panel, cfg, event_scope="ARMED,ACTION,WATCH")

    numeric, categorical = _feature_columns(ds)
    feature_set = set(numeric + categorical)
    assert_true(feature_set.isdisjoint(set(LABEL_COLUMNS)), "label column leaked into ML features")
    assert_true(not any(name.startswith(("return_", "mfe_", "mae_")) for name in feature_set), "future return/path label leaked into features")

    preds = build_predictions(
        ds,
        {
            "model": ConstantModel(),
            "model_version": "unit_constant",
            "target": "label_action_good_30m",
            "model_type": "unit",
        },
        cfg,
    )
    assert_true("alert_state" not in preds.columns, "prediction output should not overwrite rule alert_state")
    assert_true(set(preds["ml_action_hint"]).issubset({"BOOST", "NEUTRAL", "SUPPRESS", "WATCH_ONLY"}), "unexpected ML hint")
    assert_true("alert_state" in ds.columns, "source dataset lost original rule alert_state")


def test_walk_forward_split_is_time_ordered():
    cfg = load_config()
    rows = []
    for i, day in enumerate(pd.date_range("2026-06-01", periods=22, freq="D")):
        rows.append({"trade_date": day.strftime("%Y-%m-%d"), "label_action_good_30m": float(i % 2), "event_type": "ACTION"})
    df = pd.DataFrame(rows)
    splits = build_walk_forward_splits(df, cfg["ml_shadow_v04"])
    assert_true(len(splits) > 0, "walk-forward splits missing")
    for split in splits:
        assert_true(max(split["train_dates"]) < min(split["test_dates"]), "walk-forward split used future train dates")
        assert_true(split["purged"], "purging flag not propagated")
        assert_true(split["embargo_bars"] == cfg["ml_shadow_v04"]["validation"]["embargo_bars"], "embargo setting not propagated")


if __name__ == "__main__":
    test_t0_dataset_labels_and_dedupe()
    test_walk_forward_split_is_time_ordered()
    print("ml shadow v0.4 tests passed")
