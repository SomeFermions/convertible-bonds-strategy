import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training_data_v06.config import load_config
from training_data_v06.pipeline import (
    available_trial_range,
    bar_quality_flags,
    benchmark_rows_from_jq_frame,
    build_local_training_windows,
    build_wind_missing_requests,
    chunk_dates,
    classify_sample,
    contiguous_missing_date_ranges,
    estimate_rows,
    infer_conversion_price_intervals,
    jq_plan_can_run,
    permission_safe_trial_range,
    raw_chunk_complete_from_coverage,
    require_large_download_confirmation,
    require_smoke_scope,
    raw_rows_from_jq_frame,
    selected_training_dates,
    split_jq_fields,
    stock_daily_support_rows,
    summarize_training_bar_days,
    wind_raw_rows,
    wind_fields_for_scope,
    wind_scope_candidates,
)
from training_data_v06.providers import (
    JQDataProvider,
    WindPyProvider,
    build_jqdata_provider,
    build_wind_provider,
    jq_code_from_bond,
    jq_code_from_stock,
    jq_intraday_bounds,
    load_jqdata_credentials,
    read_secret,
    validate_jq_fields,
    validate_wind_fields,
    wind_code_from_bond,
)
from training_data_v06.schemas import (
    CANONICAL_TABLE,
    BENCHMARK_CANONICAL_TABLE,
    BENCHMARK_COVERAGE_TABLE,
    BENCHMARK_PLAN_TABLE,
    CONVERSION_PRICE_COVERAGE_TABLE,
    CONVERSION_PRICE_HISTORY_TABLE,
    COVERAGE_TABLE,
    JQ_CB_RAW_TABLE,
    JQ_BENCHMARK_RAW_TABLE,
    JQ_STOCK_DAILY_UNADJUSTED_TABLE,
    JQ_STOCK_RAW_TABLE,
    table_column_names,
)


def assert_true(value, message):
    if not bool(value):
        raise AssertionError(message)


def assert_raises(fn, message):
    try:
        fn()
    except Exception:
        return
    raise AssertionError(message)


def test_jq_smoke_scope_guard():
    cfg = load_config()
    require_smoke_scope(["113661", "127110"], 2, cfg)
    assert_raises(lambda: require_smoke_scope(["113661", "127110", "123124"], 2, cfg), "smoke test allowed >2 bonds")
    assert_raises(lambda: require_smoke_scope(["113661"], 3, cfg), "smoke test allowed >2 days")


def test_large_download_requires_confirm():
    cfg = load_config()
    require_large_download_confirmation(2, 100, True, cfg)
    assert_raises(
        lambda: require_large_download_confirmation(3, 100, False, cfg),
        "large instrument count did not require confirm",
    )
    assert_raises(
        lambda: require_large_download_confirmation(1, 10001, False, cfg),
        "large estimated rows did not require confirm",
    )


def test_jq_plan_status_guard():
    assert_true(jq_plan_can_run("READY", 0, 3), "READY plan with attempts left should run")
    assert_true(not jq_plan_can_run("DRY_RUN_ESTIMATED", 0, 3), "dry-run plan must not run")
    assert_true(not jq_plan_can_run("READY", 3, 3), "plan at max attempts must not run")


def test_jq_credentials_can_come_from_secret_files_without_logging_values():
    with tempfile.TemporaryDirectory() as tmp:
        username_file = Path(tmp) / "jq_user"
        password_file = Path(tmp) / "jq_password"
        username_file.write_text("user-from-file\n", encoding="utf-8")
        password_file.write_text("password-from-file\n", encoding="utf-8")
        provider = JQDataProvider(
            username_env="MISSING_JQ_USER_ENV",
            password_env="MISSING_JQ_PASSWORD_ENV",
            username_file_env="TEST_JQ_USER_FILE",
            password_file_env="TEST_JQ_PASSWORD_FILE",
        )
        import os

        old_user = os.environ.get("TEST_JQ_USER_FILE")
        old_password = os.environ.get("TEST_JQ_PASSWORD_FILE")
        try:
            os.environ["TEST_JQ_USER_FILE"] = str(username_file)
            os.environ["TEST_JQ_PASSWORD_FILE"] = str(password_file)
            assert_true(read_secret("MISSING_JQ_USER_ENV", "TEST_JQ_USER_FILE") == "user-from-file", "username file secret not read")
            assert_true(provider.account_id_hash() != "user-from-file", "account hash leaked raw username")
            assert_true(len(provider.account_id_hash()) == 32, "account hash length mismatch")
        finally:
            if old_user is None:
                os.environ.pop("TEST_JQ_USER_FILE", None)
            else:
                os.environ["TEST_JQ_USER_FILE"] = old_user
            if old_password is None:
                os.environ.pop("TEST_JQ_PASSWORD_FILE", None)
            else:
                os.environ["TEST_JQ_PASSWORD_FILE"] = old_password


def test_jq_credentials_can_come_from_local_toml_and_env_still_wins():
    import os

    with tempfile.TemporaryDirectory() as tmp:
        credentials_file = Path(tmp) / "jqdata_credentials.toml"
        credentials_file.write_text('[jqdata]\nusername = "file-user"\npassword = "file-password"\n', encoding="utf-8")
        assert_true(
            load_jqdata_credentials(credentials_file) == ("file-user", "file-password"),
            "JQData TOML credentials were not loaded",
        )
        provider = JQDataProvider(
            username_env="TEST_JQ_USERNAME",
            password_env="TEST_JQ_PASSWORD",
            username_file_env="MISSING_JQ_USERNAME_FILE",
            password_file_env="MISSING_JQ_PASSWORD_FILE",
            credentials_file=credentials_file,
        )
        old_user = os.environ.get("TEST_JQ_USERNAME")
        old_password = os.environ.get("TEST_JQ_PASSWORD")
        try:
            assert_true(provider.credentials() == ("file-user", "file-password"), "provider ignored local TOML credentials")
            os.environ["TEST_JQ_USERNAME"] = "env-user"
            os.environ["TEST_JQ_PASSWORD"] = "env-password"
            assert_true(provider.credentials() == ("env-user", "env-password"), "environment did not override TOML credentials")
            assert_true("file-password" not in repr(provider), "credential value leaked through provider repr")
        finally:
            if old_user is None:
                os.environ.pop("TEST_JQ_USERNAME", None)
            else:
                os.environ["TEST_JQ_USERNAME"] = old_user
            if old_password is None:
                os.environ.pop("TEST_JQ_PASSWORD", None)
            else:
                os.environ["TEST_JQ_PASSWORD"] = old_password


def test_pipeline_jq_provider_uses_configured_credentials_file():
    with tempfile.TemporaryDirectory() as tmp:
        credentials_file = Path(tmp) / "jqdata_credentials.toml"
        credentials_file.write_text('[jqdata]\nusername = "pipeline-user"\npassword = "pipeline-password"\n', encoding="utf-8")
        provider = build_jqdata_provider({"jqdata": {"credentials_file": str(credentials_file)}})
        assert_true(provider.credentials() == ("pipeline-user", "pipeline-password"), "pipeline provider did not use configured credentials")


def test_field_whitelists_exclude_terms_and_extra_market_data():
    assert_true(validate_jq_fields("CB", ["open", "high", "low", "close", "volume", "money", "paused"]), "valid JQ CB fields rejected")
    assert_raises(lambda: validate_jq_fields("CB", ["open", "ytm"]), "JQ CB ytm should be rejected")
    assert_true(validate_wind_fields(["live_turnover", "ytm"]) == ["live_turnover", "ytm"], "valid Wind fields rejected")
    assert_raises(lambda: validate_wind_fields(["live_turnover", "call_trigger_count"]), "Wind clause field should be rejected")
    assert_raises(lambda: validate_wind_fields(["open", "high", "low", "close"]), "Wind OHLC minute fields should be rejected")


def test_jq_field_split_keeps_stock_optional_fields_off_cb_requests():
    cfg = load_config()
    cb_fields, stock_fields = split_jq_fields(["open", "high", "low", "close", "volume", "money", "paused", "live_turnover"], cfg)
    assert_true("live_turnover" not in cb_fields, "CB plan should not request stock-only live_turnover")
    assert_true("live_turnover" in stock_fields, "STOCK plan should keep live_turnover")
    assert_true("paused" in cb_fields and "paused" in stock_fields, "common optional paused field missing")


def test_jq_provider_retries_without_optional_fields_when_vendor_rejects_them():
    calls = []

    class Provider(JQDataProvider):
        def _sdk(self):
            def fake_get_price(vendor_code, start_date=None, end_date=None, frequency="daily", fields=None, panel=True, **_):
                calls.append(list(fields or []))
                if "live_turnover" in fields:
                    raise ValueError("unknown field live_turnover")
                return pd.DataFrame(
                    {
                        "open": [1.0],
                        "high": [1.1],
                        "low": [0.9],
                        "close": [1.0],
                        "volume": [100.0],
                        "money": [1000.0],
                    },
                    index=[pd.Timestamp("2026-01-05 09:35:00")],
                )

            return fake_get_price

    frame = Provider().get_price_5m("000001.XSHE", "2026-01-05", "2026-01-05", ["open", "close", "live_turnover"])
    assert_true(calls == [["open", "close", "live_turnover"], ["open", "close"]], "JQ optional field fallback did not retry as expected")
    assert_true("live_turnover" in frame.columns and pd.isna(frame.loc[0, "live_turnover"]), "fallback column should be present and null")


def test_jq_provider_does_not_retry_network_errors_as_field_fallback():
    calls = []

    class Provider(JQDataProvider):
        def _sdk(self):
            def fake_get_price(*_, **kwargs):
                calls.append(list(kwargs.get("fields") or []))
                raise ConnectionError("connection reset by peer")

            return fake_get_price

    assert_raises(
        lambda: Provider().get_price_5m("000001.XSHE", "2026-01-05", "2026-01-05", ["close", "live_turnover"]),
        "network error should be returned to the plan retry layer",
    )
    assert_true(calls == [["close", "live_turnover"]], "network error caused an unnecessary optional-field retry")


def test_jq_provider_caches_rejected_optional_fields():
    calls = []

    class Provider(JQDataProvider):
        def _sdk(self):
            def fake_get_price(vendor_code, **kwargs):
                requested = list(kwargs.get("fields") or [])
                calls.append((vendor_code, requested))
                if "live_turnover" in requested:
                    raise ValueError("unknown field live_turnover")
                return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2026-01-05 09:35:00")])

            return fake_get_price

    provider = Provider()
    provider.get_price_5m("000001.XSHE", "2026-01-05", "2026-01-05", ["close", "live_turnover"])
    provider.get_price_5m("000002.XSHE", "2026-01-05", "2026-01-05", ["close", "live_turnover"])
    assert_true(
        calls == [
            ("000001.XSHE", ["close", "live_turnover"]),
            ("000001.XSHE", ["close"]),
            ("000002.XSHE", ["close"]),
        ],
        "rejected optional field was retried for the next instrument",
    )


def test_jq_code_mapping():
    assert_true(jq_code_from_bond("113661") == "113661.XSHG", "Shanghai bond JQ code mismatch")
    assert_true(jq_code_from_bond("123124") == "123124.XSHE", "Shenzhen bond JQ code mismatch")
    assert_true(jq_code_from_stock("688131") == "688131.XSHG", "Shanghai stock JQ code mismatch")
    assert_true(jq_code_from_stock("300655") == "300655.XSHE", "Shenzhen stock JQ code mismatch")
    assert_true(wind_code_from_bond("113661") == "113661.SH", "Shanghai bond Wind code mismatch")
    assert_true(wind_code_from_bond("123124") == "123124.SZ", "Shenzhen bond Wind code mismatch")


def test_jq_intraday_date_bounds_cover_the_market_session():
    start, end = jq_intraday_bounds("2026-04-10", "2026-04-10")
    assert_true(start == "2026-04-10 09:30:00", "JQ minute start did not enter the market session")
    assert_true(end == "2026-04-10 15:00:00", "JQ minute end did not include the close")
    explicit_start, explicit_end = jq_intraday_bounds("2026-04-10 10:00:00", "2026-04-10 10:30:00")
    assert_true(explicit_start.endswith("10:00:00") and explicit_end.endswith("10:30:00"), "explicit JQ times were changed")


def test_jq_vendor_timestamp_is_normalized_from_bar_end_to_bar_start():
    cfg = load_config()
    plan = pd.Series(
        {
            "batch_id": "test-batch",
            "instrument_code": "113661.XSHG",
            "instrument_slot": 7,
            "asset_type": "CB",
            "bond_code": "113661",
            "stock_code": "603806",
        }
    )
    frame = pd.DataFrame(
        [
            {"bar_start": pd.Timestamp("2026-04-10 09:35:00"), "open": 1, "high": 1, "low": 1, "close": 1},
            {"bar_start": pd.Timestamp("2026-04-10 11:30:00"), "open": 1, "high": 1, "low": 1, "close": 1},
            {"bar_start": pd.Timestamp("2026-04-10 15:00:00"), "open": 1, "high": 1, "low": 1, "close": 1},
        ]
    )
    rows = raw_rows_from_jq_frame(plan, frame, cfg)
    assert_true(len(rows) == 3, "valid JQ morning/close bars were filtered")
    assert_true(rows[0]["bar_start"] == pd.Timestamp("2026-04-10 09:30:00"), "JQ bar start was not shifted")
    assert_true(rows[1]["bar_start"] == pd.Timestamp("2026-04-10 11:25:00"), "JQ morning close bar was not normalized")
    assert_true(rows[2]["bar_end"] == pd.Timestamp("2026-04-10 15:00:00"), "JQ close bar end changed")
    assert_true(rows[0]["ts"] == pd.Timestamp("2026-04-10 09:30:00.007"), "JQ instrument slot was not used as the TDengine key")


def test_windpy_provider_uses_only_whitelisted_daily_fields_and_closes_own_session():
    class Response:
        ErrorCode = 0
        Fields = ["TURN", "YTM_B"]
        Times = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
        Data = [[1.25, 1.5], [2.1, 2.2]]

    class StartResponse:
        ErrorCode = 0

    class FakeWind:
        def __init__(self):
            self.connected = False
            self.calls = []
            self.closed = False

        def isconnected(self):
            return self.connected

        def start(self):
            self.connected = True
            return StartResponse()

        def wsd(self, code, fields, start_date, end_date, options):
            self.calls.append((code, fields, start_date, end_date, options))
            return Response()

        def close(self):
            self.closed = True
            self.connected = False

    wind = FakeWind()
    provider = WindPyProvider(_wind=wind)
    frame = provider.fetch_turnover_ytm(["113661"], "2026-01-05", "2026-01-06", ["live_turnover", "ytm"])
    assert_true(wind.calls == [("113661.SH", "turn,ytm_b", "2026-01-05", "2026-01-06", "Days=Trading")], "Wind wsd call scope changed")
    assert_true(wind.closed, "Wind session started by provider was not closed")
    assert_true(frame["bond_code"].tolist() == ["113661", "113661"], "Wind rows lost bond mapping")
    assert_true(frame["live_turnover"].tolist() == [1.25, 1.5], "Wind turnover normalization failed")
    assert_true(frame["ytm"].tolist() == [2.1, 2.2], "Wind ytm normalization failed")


def test_wind_provider_auto_detects_repository_remote_adapter():
    provider = build_wind_provider(load_config())
    assert_true(isinstance(provider, WindPyProvider), "auto provider did not detect remote_windpy_adapter")


def test_wind_raw_rows_preserve_optional_availability_and_stable_identity():
    cfg = load_config()
    frame = pd.DataFrame(
        [
            {
                "bond_code": "113661",
                "trade_date": "2026-01-05",
                "live_turnover": 1.25,
                "ytm": None,
            }
        ]
    )
    first = wind_raw_rows(frame, "batch-a", cfg)[0]
    second = wind_raw_rows(frame, "batch-b", cfg)[0]
    assert_true(first["ts"] == second["ts"], "Wind raw identity should not depend on download batch")
    assert_true(first["live_turnover_source"] == "wind", "available Wind turnover was not marked")
    assert_true(first["ytm_source"] == "unavailable", "missing Wind ytm should remain optional")


def test_wind_validation_scope_is_bounded_and_prioritized():
    cfg = load_config()
    assert_true(wind_fields_for_scope(cfg, "validation_sample") == ["live_turnover"], "Wind validation should not retry unavailable YTM")
    assert_true(wind_fields_for_scope(cfg, "active_like_only") == ["live_turnover", "ytm"], "Wind P1 field scope changed")
    sample = pd.DataFrame(
        [
            {"bond_code": "110001", "research_pool_scope": "ACTIVE_LIKE"},
            {"bond_code": "120001", "research_pool_scope": "BROAD_BUT_CLEAN"},
            {"bond_code": "130001", "research_pool_scope": "ELIGIBLE_SHADOW"},
            {"bond_code": "130002", "research_pool_scope": "ELIGIBLE_SHADOW"},
        ]
    )
    candidates = wind_scope_candidates(sample, "validation_sample")
    assert_true(candidates["bond_code"].tolist() == ["130001", "130002", "120001"], "Wind validation priority changed")
    assert_true(
        candidates["bond_code"].astype(str).str.zfill(6).drop_duplicates().tolist() == ["130001", "130002", "120001"],
        "Wind request list lost validation scope priority",
    )
    assert_raises(lambda: wind_scope_candidates(sample, "all"), "unbounded Wind scope should be rejected")

    selected, requests, cells = build_wind_missing_requests(
        candidates["bond_code"].tolist(),
        ["2026-01-05", "2026-01-06"],
        {("130001", "2026-01-05")},
        field_count=2,
        max_cells=5,
    )
    assert_true(selected == ["130001"], "Wind cell cap did not stop before the next whole bond")
    assert_true(requests == [("130001", "2026-01-06", "2026-01-06")] and cells == 2, "Wind missing range estimate changed")
    selected, requests, cells = build_wind_missing_requests(
        ["130001"],
        ["2026-01-05", "2026-01-06"],
        set(),
        field_count=2,
        max_cells=3,
    )
    assert_true(not selected and not requests and cells == 0, "Wind selected a partial bond above the cell cap")


def test_sample_classification_scopes():
    cfg = load_config()
    active = pd.Series(
        {
            "stock_code": "603806",
            "stock_name": "clean",
            "premium_rate": 25.0,
            "days_to_maturity": 500,
            "debt_asset_ratio": 60.0,
            "operating_cash_flow": 1.0,
            "free_cash_flow": 1.0,
        }
    )
    scope, reason, exclusion, _ = classify_sample(active, cfg)
    assert_true(scope == "ACTIVE_LIKE" and exclusion is None and "premium_20_30" in reason, "ACTIVE_LIKE classification failed")

    shadow = active.copy()
    shadow["premium_rate"] = 35.0
    scope, _, exclusion, _ = classify_sample(shadow, cfg)
    assert_true(scope == "ELIGIBLE_SHADOW" and exclusion is None, "ELIGIBLE_SHADOW classification failed")

    broad = active.copy()
    broad["premium_rate"] = 55.0
    scope, _, exclusion, _ = classify_sample(broad, cfg)
    assert_true(scope == "BROAD_BUT_CLEAN" and exclusion is None, "BROAD_BUT_CLEAN classification failed")

    st = active.copy()
    st["stock_name"] = "ST bad"
    scope, _, exclusion, _ = classify_sample(st, cfg)
    assert_true(scope is None and "stock_st" in exclusion, "ST sample was not excluded")


def test_bar_quality_does_not_cross_lunch_or_fake_missing_bars():
    valid = pd.Series({"bar_start": pd.Timestamp("2026-07-01 10:00:00"), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "paused": False})
    ok, flag = bar_quality_flags(valid)
    assert_true(ok and flag == "OK", "valid session bar failed QA")

    lunch = valid.copy()
    lunch["bar_start"] = pd.Timestamp("2026-07-01 12:00:00")
    ok, flag = bar_quality_flags(lunch)
    assert_true(not ok and "non_session_bar" in flag, "lunch bar should be rejected")

    bad = valid.copy()
    bad["high"] = 0.8
    ok, flag = bar_quality_flags(bad)
    assert_true(not ok and "invalid_ohlc" in flag, "invalid OHLC was not flagged")


def test_range_chunk_and_estimate_helpers():
    start, end = available_trial_range("2026-07-09")
    assert_true(start.strftime("%Y-%m-%d") == "2025-04-09", "trial start range mismatch")
    assert_true(end.strftime("%Y-%m-%d") == "2026-04-09", "trial end range mismatch")
    cfg = load_config()
    dynamic_cfg = deepcopy(cfg)
    dynamic_cfg["jqdata"]["available_start_date"] = None
    dynamic_cfg["jqdata"]["available_end_date"] = None
    dynamic_cfg["jqdata"]["trial_window_anchor_date"] = None
    dynamic_cfg["jqdata"]["trial_download_start_buffer_days"] = 14
    safe_start, safe_end = permission_safe_trial_range(dynamic_cfg, "2026-07-12")
    assert_true(safe_start.strftime("%Y-%m-%d") == "2025-04-28", "trial-safe start should roll forward after the 14-day buffer")
    assert_true(safe_end.strftime("%Y-%m-%d") == "2026-04-10", "trial-safe end should roll back to a business day")
    selected_start, selected_end, dates = selected_training_dates(dynamic_cfg, "2026-07-09")
    assert_true(len(dates) == 200, "training plan should cap to about 200 trading days")
    assert_true(selected_end.strftime("%Y-%m-%d") == "2026-04-09", "selected training end should stay at trial range end")
    assert_true(selected_start >= safe_start, "selected dates escaped the permission-safe lower boundary")
    _, configured_end, configured_dates = selected_training_dates(cfg, "2026-07-09")
    assert_true(configured_end.strftime("%Y-%m-%d") == "2026-04-09", "dynamic trial boundary mismatch")
    assert_true(len(configured_dates) == 200, "dynamic JQData range should still cap at 200 business days")

    anchored_cfg = deepcopy(dynamic_cfg)
    anchored_cfg["jqdata"]["trial_window_anchor_date"] = "2026-07-14"
    anchored_start, anchored_end = permission_safe_trial_range(anchored_cfg)
    assert_true(anchored_start.strftime("%Y-%m-%d") == "2025-04-28", "configured trial anchor was ignored")
    assert_true(anchored_end.strftime("%Y-%m-%d") == "2026-04-14", "configured trial anchor end mismatch")
    chunks = chunk_dates(["2026-01-01", "2026-01-02", "2026-01-05"], 2)
    assert_true(chunks == [("2026-01-01", "2026-01-02", 2), ("2026-01-05", "2026-01-05", 1)], "date chunking failed")
    assert_true(estimate_rows(400, 200, 48) == 3840000, "row estimate mismatch")
    coverage = {("113661.XSHG", "2026-04-09"): 48, ("113661.XSHG", "2026-04-10"): 48}
    assert_true(raw_chunk_complete_from_coverage(coverage, "113661.XSHG", "2026-04-09", "2026-04-10", 96), "complete raw chunk was not detected")
    assert_true(not raw_chunk_complete_from_coverage(coverage, "113661.XSHG", "2026-04-09", "2026-04-10", 97), "partial raw chunk was treated as complete")
    assert_true(
        contiguous_missing_date_ranges(
            ["2026-04-08", "2026-04-09", "2026-04-10", "2026-04-13"],
            ["2026-04-08", "2026-04-09", "2026-04-13"],
        ) == [("2026-04-08", "2026-04-09"), ("2026-04-13", "2026-04-13")],
        "missing Wind dates were not split into contiguous trading ranges",
    )


def test_training_tables_are_separate_from_akshare_live():
    assert_true(JQ_CB_RAW_TABLE != "cb_watchlist_bar_5m_live", "JQ CB raw table must not be live table")
    assert_true(JQ_STOCK_RAW_TABLE != "cb_watchlist_bar_5m_live", "JQ stock raw table must not be live table")
    assert_true(CANONICAL_TABLE != "cb_watchlist_bar_5m_live", "canonical training table must not be live table")
    assert_true("bar_time_semantics" in table_column_names(JQ_CB_RAW_TABLE), "CB raw table lacks timestamp semantics")
    assert_true("bar_time_semantics" in table_column_names(JQ_STOCK_RAW_TABLE), "stock raw table lacks timestamp semantics")
    assert_true("instrument_slot" in table_column_names(JQ_CB_RAW_TABLE), "CB raw table lacks collision-free instrument slot")
    assert_true("instrument_slot" in table_column_names(JQ_STOCK_RAW_TABLE), "stock raw table lacks collision-free instrument slot")
    assert_true(
        "fq_mode" in table_column_names(JQ_STOCK_DAILY_UNADJUSTED_TABLE),
        "qIV stock support table does not record adjustment mode",
    )
    assert_true("valid_cb_bars" in table_column_names(COVERAGE_TABLE), "coverage table lacks valid CB bar count")
    assert_true("row_coverage_ratio" in table_column_names(COVERAGE_TABLE), "coverage table conflates row and valid coverage")
    assert_true(
        "effective_end_exclusive" in table_column_names(CONVERSION_PRICE_HISTORY_TABLE),
        "conversion-price history lacks explicit interval semantics",
    )
    assert_true(
        "qa_mismatch_days" in table_column_names(CONVERSION_PRICE_COVERAGE_TABLE),
        "conversion-price coverage lacks source QA",
    )
    assert_true(BENCHMARK_CANONICAL_TABLE != CANONICAL_TABLE, "benchmark bars must not mix into bond-stock canonical")
    assert_true("benchmark_code" in table_column_names(JQ_BENCHMARK_RAW_TABLE), "benchmark raw lineage missing")
    assert_true("missing_dates_json" in table_column_names(BENCHMARK_PLAN_TABLE), "benchmark plan cannot resume missing dates")
    assert_true("coverage_ratio" in table_column_names(BENCHMARK_COVERAGE_TABLE), "benchmark coverage table missing ratio")


def test_conversion_price_intervals_are_point_in_time_and_event_driven():
    dates = list(pd.to_datetime(["2025-06-16", "2025-06-17", "2025-06-23", "2025-06-24"]))
    stock = pd.DataFrame(
        {
            "trade_date": dates,
            "close": [7.12, 7.20, 4.79, 4.80],
        }
    )
    values = pd.DataFrame(
        {
            "trade_date": dates,
            "conversion_value": [100.0, 101.12, 100.0, 100.21],
        }
    )
    adjustments = pd.DataFrame(
        {
            "effective_date": [pd.Timestamp("2025-06-23")],
            "announcement_date": [pd.Timestamp("2025-06-17")],
            "conversion_price": [4.79],
            "adjustment_reason": ["调整转股价"],
        }
    )
    intervals, report = infer_conversion_price_intervals(
        "sample",
        "110086",
        "600496",
        dates,
        stock,
        values,
        adjustments,
        pd.Timestamp("2026-07-23"),
    )
    assert_true(len(intervals) == 2, "adjustment event did not split the conversion-price interval")
    assert_true(intervals[0]["conversion_price"] == 7.12, "AkShare/JQ unadjusted baseline inference failed")
    assert_true(intervals[0]["effective_end_exclusive"] == "2025-06-23", "interval end is not exclusive")
    assert_true(intervals[1]["conversion_price"] == 4.79, "JQData adjustment price was not applied")
    assert_true(report["coverage_pass"], "fully covered conversion-price history failed coverage")


def test_conversion_price_history_does_not_backfill_without_pit_baseline():
    intervals, report = infer_conversion_price_intervals(
        "sample",
        "110086",
        "600496",
        list(pd.to_datetime(["2025-06-16", "2025-06-17"])),
        pd.DataFrame(columns=["trade_date", "close"]),
        pd.DataFrame(columns=["trade_date", "conversion_value"]),
        pd.DataFrame(
            {
                "effective_date": [pd.Timestamp("2025-06-23")],
                "announcement_date": [pd.Timestamp("2025-06-17")],
                "conversion_price": [4.79],
                "adjustment_reason": ["调整转股价"],
            }
        ),
        pd.Timestamp("2026-07-23"),
    )
    assert_true(not intervals, "future adjustment leaked backward as a baseline")
    assert_true(not report["coverage_pass"], "missing PIT baseline was marked covered")


def test_qiv_stock_support_marks_previous_close_fill():
    rows = stock_daily_support_rows(
        pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2025-08-13", "2025-08-14"]),
                "vendor_code": ["603058.XSHG", "603058.XSHG"],
                "stock_code": ["603058", "603058"],
                "close": [10.88, None],
            }
        ),
        "sample",
        "batch",
        pd.Timestamp("2026-07-23"),
    )
    assert_true(len(rows) == 2, "marked previous close did not preserve the support row")
    assert_true(rows[1]["close"] == 10.88, "previous unadjusted close was not carried")
    assert_true(rows[1]["vendor_close"] is None or pd.isna(rows[1]["vendor_close"]), "raw missing close was overwritten")
    assert_true(rows[1]["is_filled"], "carried support close was not marked")


def test_training_gap_audit_does_not_treat_prelisting_cb_rows_as_repairable():
    rows = pd.DataFrame(
        {
            "asset_type": ["CB", "CB", "CB", "CB", "STOCK"],
            "bond_code": ["123001"] * 5,
            "stock_code": ["300001"] * 5,
            "trade_date": pd.to_datetime(
                ["2025-06-16", "2025-06-17", "2025-06-18", "2025-06-19", "2025-06-16"]
            ),
            "row_count": [48, 48, 48, 48, 48],
            "valid_rows": [0, 48, 24, 48, 48],
            "null_volume": [48, 0, 24, 0, 0],
            "null_amount": [48, 0, 24, 0, 0],
        }
    )
    cb = summarize_training_bar_days(rows, "CB")
    stock = summarize_training_bar_days(rows, "STOCK")
    assert_true(cb["expected_unavailable_days"] == 1, "prelisting CB day was not classified")
    assert_true(cb["repairable_days"] == 1, "active-span CB gap was not repairable")
    assert_true(stock["repairable_days"] == 0, "complete stock day was flagged")


def test_benchmark_rows_normalize_jq_bar_end_and_keep_ohlcv():
    cfg = load_config()
    frame = pd.DataFrame(
        {
            "bar_start": pd.to_datetime(["2025-06-16 09:35:00", "2025-06-16 15:00:00"]),
            "open": [4000.0, 4010.0],
            "high": [4002.0, 4012.0],
            "low": [3999.0, 4008.0],
            "close": [4001.0, 4011.0],
            "volume": [100.0, 200.0],
            "money": [1e8, 2e8],
            "paused": [None, None],
        }
    )
    raw, canonical = benchmark_rows_from_jq_frame(
        frame,
        {"benchmark_code": "CSI300", "benchmark_name": "沪深300", "vendor_code": "000300.XSHG"},
        "sample",
        "batch",
        0,
        cfg,
    )
    assert_true(len(raw) == 2 and len(canonical) == 2, "benchmark bars were dropped")
    assert_true(raw[0]["bar_start"] == pd.Timestamp("2025-06-16 09:30:00"), "benchmark bar end was not normalized")
    assert_true(canonical[1]["volume"] == 200.0 and canonical[1]["amount"] == 2e8, "benchmark OHLCV/amount changed")
    assert_true(canonical[0]["is_valid_bar"], "valid benchmark bar failed QA")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("training data v0.6 tests passed")
