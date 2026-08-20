import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cb_universe import normalize_bond_code, to_ak_bond_symbol
from collect_cb_daily import normalize_daily_df
from collect_watchlist_realtime import (
    WATCHLIST_STATUS_ACTIVE,
    WATCHLIST_STATUS_ACTIVE_PENDING_HISTORY,
    WATCHLIST_STATUS_OBSERVE,
    FiveMinuteAggregator,
    WatchlistStockQuoteProvider,
    build_monitoring_watchlist,
    live_retention_cutoff_date,
)


def assert_equal(actual, expected):
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def test_symbol_mapping():
    assert_equal(normalize_bond_code("113011.SH"), "113011")
    assert_equal(normalize_bond_code("127061.SZ"), "127061")
    assert_equal(normalize_bond_code(" 128123 "), "128123")
    assert_equal(normalize_bond_code(113011), "113011")
    assert_equal(normalize_bond_code(127061.0), "127061")
    assert_equal(normalize_bond_code(None), None)
    assert_equal(normalize_bond_code(""), None)
    assert_equal(to_ak_bond_symbol("113011"), "sh113011")
    assert_equal(to_ak_bond_symbol("118000"), "sh118000")
    assert_equal(to_ak_bond_symbol("127061"), "sz127061")
    assert_equal(to_ak_bond_symbol("128123"), "sz128123")
    assert_equal(to_ak_bond_symbol("810011"), None)


def test_daily_column_normalization():
    raw = pd.DataFrame(
        {
            "date": ["2026-06-22"],
            "open": ["100.1"],
            "high": ["101.2"],
            "low": ["99.8"],
            "close": ["100.5"],
            "volume": ["123"],
        }
    )
    out = normalize_daily_df(raw, "sz127061")
    assert_equal(len(out), 1)
    assert_equal(str(out.loc[0, "ts"]), "2026-06-22 00:00:00")
    assert_equal(float(out.loc[0, "close"]), 100.5)


def test_watchlist_stock_provider_groups_duplicate_underlying():
    watchlist = pd.DataFrame(
        [
            {"bond_code": "111111", "bond_name": "bond1", "stock_code": "600000", "stock_name": "stock"},
            {"bond_code": "222222", "bond_name": "bond2", "stock_code": "600000", "stock_name": "stock"},
        ]
    )
    provider = WatchlistStockQuoteProvider(bar_minutes=5)
    calls = []

    def fake_bid_ask(pair):
        calls.append(pair.stock_code)
        return {
            "quote_ts": pd.Timestamp("2026-06-29 10:01:00"),
            "pair_key": pair.pair_key,
            "bond_code": pair.bond_code,
            "bond_name": pair.bond_name,
            "stock_code": pair.stock_code,
            "stock_name": pair.stock_name,
            "market": "SH",
            "trade": 10.0,
            "open": 9.8,
            "high": 10.1,
            "low": 9.7,
            "volume": 1000.0,
            "amount": 10000.0,
            "changepercent": 1.1,
            "turnover_rate": 0.5,
            "volume_is_delta": False,
            "amount_is_delta": False,
            "quote_source": "stock_bid_ask_em",
        }

    provider._fetch_bid_ask_row = fake_bid_ask
    provider._fetch_hist_5m_row = lambda pair: None
    quotes = provider.fetch(watchlist)
    assert_equal(calls, ["600000"])
    assert_equal(len(quotes), 2)
    assert_equal(set(quotes["pair_key"]), {"111111:600000", "222222:600000"})


def test_stock_aggregator_keeps_bond_stock_pair_identity():
    quotes = pd.DataFrame(
        [
            {
                "quote_ts": pd.Timestamp("2026-06-29 10:01:00"),
                "pair_key": "111111:600000",
                "bond_code": "111111",
                "bond_name": "bond1",
                "stock_code": "600000",
                "stock_name": "stock",
                "market": "SH",
                "trade": 10.0,
                "open": 9.8,
                "high": 10.1,
                "low": 9.7,
                "volume": 1000.0,
                "amount": 10000.0,
                "changepercent": 1.1,
                "turnover_rate": 0.5,
                "volume_is_delta": False,
                "amount_is_delta": False,
                "quote_source": "stock_bid_ask_em",
            },
            {
                "quote_ts": pd.Timestamp("2026-06-29 10:01:00"),
                "pair_key": "222222:600000",
                "bond_code": "222222",
                "bond_name": "bond2",
                "stock_code": "600000",
                "stock_name": "stock",
                "market": "SH",
                "trade": 10.0,
                "open": 9.8,
                "high": 10.1,
                "low": 9.7,
                "volume": 1000.0,
                "amount": 10000.0,
                "changepercent": 1.1,
                "turnover_rate": 0.5,
                "volume_is_delta": False,
                "amount_is_delta": False,
                "quote_source": "stock_bid_ask_em",
            },
        ]
    )
    aggregator = FiveMinuteAggregator(
        bar_minutes=5,
        asset_type="STOCK",
        code_attr="stock_code",
        name_attr="stock_name",
        code_key="stock_code",
        name_key="stock_name",
        identity_attr="pair_key",
    )
    aggregator.update(quotes)
    bars = aggregator.flush_all()
    assert_equal(len(bars), 2)
    assert_equal({bar["bond_code"] for bar in bars}, {"111111", "222222"})


def test_live_retention_keeps_today_plus_recent_trade_dates():
    cutoff = live_retention_cutoff_date(
        ["2026-06-26", "2026-06-25", "2026-06-24"],
        today=pd.Timestamp("2026-06-29 09:00:00"),
        retain_days=3,
    )
    assert_equal(cutoff, "2026-06-25")


def test_watchlist_rebalance_keeps_active_between_windows():
    previous = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-06-30 09:00:00"),
                "bond_code": "111111",
                "watchlist_status": WATCHLIST_STATUS_ACTIVE,
                "candidate_since": pd.Timestamp("2026-06-30 09:00:00"),
                "active_since": pd.Timestamp("2026-06-30 09:00:00"),
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {"bond_code": "222222", "bond_name": "new", "stock_code": "600000", "stock_name": "stock"},
        ]
    )
    out = build_monitoring_watchlist(
        candidates,
        previous,
        today=pd.Timestamp("2026-07-01 09:00:00"),
        known_trade_dates={"2026-06-30", "2026-07-01"},
        rebalance_days=3,
        observation_days=3,
    )
    assert_equal(set(out["bond_code"]), {"111111", "222222"})
    assert_equal(out.set_index("bond_code").loc["111111", "watchlist_status"], WATCHLIST_STATUS_ACTIVE)
    assert_equal(out.set_index("bond_code").loc["222222", "watchlist_status"], WATCHLIST_STATUS_OBSERVE)


def test_watchlist_rebalance_promotes_observed_candidates_only():
    previous = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-06-29 09:00:00"),
                "bond_code": "111111",
                "watchlist_status": WATCHLIST_STATUS_ACTIVE,
                "candidate_since": pd.Timestamp("2026-06-29 09:00:00"),
                "active_since": pd.Timestamp("2026-06-29 09:00:00"),
            },
            {
                "ts": pd.Timestamp("2026-06-30 09:00:00"),
                "bond_code": "222222",
                "watchlist_status": WATCHLIST_STATUS_OBSERVE,
                "candidate_since": pd.Timestamp("2026-06-30 09:00:00"),
                "active_since": pd.NaT,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {"bond_code": "222222", "bond_name": "old_obs", "stock_code": "600000", "stock_name": "stock"},
            {"bond_code": "333333", "bond_name": "new_obs", "stock_code": "000001", "stock_name": "stock2"},
        ]
    )
    out = build_monitoring_watchlist(
        candidates,
        previous,
        today=pd.Timestamp("2026-07-02 09:00:00"),
        known_trade_dates={"2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"},
        rebalance_days=3,
        observation_days=1,
    )
    by_code = out.set_index("bond_code")
    assert_equal(set(out["bond_code"]), {"222222", "333333"})
    assert_equal(by_code.loc["222222", "watchlist_status"], WATCHLIST_STATUS_ACTIVE)
    assert_equal(by_code.loc["333333", "watchlist_status"], WATCHLIST_STATUS_OBSERVE)


def test_watchlist_rebalance_blocks_promotion_without_history():
    previous = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2026-06-30 09:00:00"),
                "bond_code": "222222",
                "watchlist_status": WATCHLIST_STATUS_OBSERVE,
                "candidate_since": pd.Timestamp("2026-06-30 09:00:00"),
                "active_since": pd.NaT,
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {"bond_code": "222222", "bond_name": "old_obs", "stock_code": "600000", "stock_name": "stock"},
        ]
    )
    out = build_monitoring_watchlist(
        candidates,
        previous,
        today=pd.Timestamp("2026-07-02 09:00:00"),
        known_trade_dates={"2026-06-30", "2026-07-01", "2026-07-02"},
        rebalance_days=3,
        observation_days=1,
        active_readiness={"222222": False},
    )
    assert_equal(out.set_index("bond_code").loc["222222", "watchlist_status"], WATCHLIST_STATUS_ACTIVE_PENDING_HISTORY)


if __name__ == "__main__":
    test_symbol_mapping()
    test_daily_column_normalization()
    test_watchlist_stock_provider_groups_duplicate_underlying()
    test_stock_aggregator_keeps_bond_stock_pair_identity()
    test_live_retention_keeps_today_plus_recent_trade_dates()
    test_watchlist_rebalance_keeps_active_between_windows()
    test_watchlist_rebalance_promotes_observed_candidates_only()
    test_watchlist_rebalance_blocks_promotion_without_history()
    print("smoke tests passed")
