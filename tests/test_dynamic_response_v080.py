from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from research.dynamic_response_v080.config import load_config
from research.dynamic_response_v080.data import (
    FORBIDDEN_SOURCE_FEATURES,
    _normalize_source,
    build_walk_forward_folds,
    validate_source_schema,
)
from research.dynamic_response_v080.eval import (
    cost_sensitivity,
    heldout_ale_profiles,
    outcome_feature_sets,
    stratified_outcome_diagnostics,
    two_dimensional_response_surfaces,
)
from research.dynamic_response_v080.features import (
    STAGE_A_CATEGORICAL_FEATURES,
    STAGE_A_NUMERIC_FEATURES,
    add_forward_labels,
    add_oof_gap_and_closure_labels,
    build_feature_dataset,
)
from research.dynamic_response_v080.models import (
    enforce_quantile_order,
    fit_lightgbm_surface,
    fit_ridge_surface,
    fit_xgboost_surface,
)
from research.dynamic_response_v080.state import (
    align_states,
    causal_forward_filter,
)
from research.dynamic_response_v080.report import determine_stage_gate
from research.dynamic_response_v080.__main__ import parse_args


def config() -> dict:
    cfg = deepcopy(load_config())
    cfg["strategy"]["n_jobs"] = 1
    cfg["surface_models"]["lightgbm"]["n_estimators"] = 20
    cfg["surface_models"]["xgboost"]["n_estimators"] = 20
    cfg["surface_models"]["early_stopping_days"] = 2
    cfg["surface_models"]["training_row_cap"] = 5000
    cfg["hmm"]["n_iter"] = 10
    cfg["hmm"]["candidate_states"] = [3]
    cfg["hmm"]["covariance_types"] = ["diag"]
    return cfg


def slot_time(date: pd.Timestamp, slot: int) -> pd.Timestamp:
    if slot < 24:
        return date + pd.Timedelta(hours=9, minutes=30 + slot * 5)
    return date + pd.Timedelta(hours=13, minutes=(slot - 24) * 5)


def source_frame(days: int = 35, bonds: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(80)
    rows = []
    for bond_index in range(bonds):
        cb_price = 110.0 + bond_index
        stock_price = 10.0 + bond_index
        for day_index, date in enumerate(pd.bdate_range("2024-01-02", periods=days)):
            for slot in range(48):
                stock_return = rng.normal(0, 0.002)
                cb_return = 0.45 * stock_return + rng.normal(0, 0.001)
                cb_open = cb_price
                stock_open = stock_price
                cb_price *= 1.0 + cb_return
                stock_price *= 1.0 + stock_return
                bar_start = slot_time(date, slot)
                rows.append(
                    {
                        "trade_date": date,
                        "bar_start": bar_start,
                        "bar_end": bar_start + pd.Timedelta(minutes=5),
                        "bar_slot": slot,
                        "bond_code": f"123{bond_index + 1:03d}",
                        "stock_code": f"300{bond_index + 1:03d}",
                        "research_pool_scope": "ACTIVE",
                        "window_id": "SYNTHETIC",
                        "window_role": "test",
                        "pair_data_quality_pass": True,
                        "cb_open": cb_open,
                        "cb_high": max(cb_open, cb_price) * 1.0002,
                        "cb_low": min(cb_open, cb_price) * 0.9998,
                        "cb_close": cb_price,
                        "cb_amount": 2e6 + rng.uniform(0, 1e6),
                        "stock_open": stock_open,
                        "stock_high": max(stock_open, stock_price) * 1.0002,
                        "stock_low": min(stock_open, stock_price) * 0.9998,
                        "stock_close": stock_price,
                        "stock_amount": 5e6 + rng.uniform(0, 2e6),
                        "cb_return": cb_return,
                        "stock_return": stock_return,
                        "stock_idiosyncratic_return": stock_return * 0.9,
                        "conversion_price": 8.0 + (day_index >= 20),
                        "conversion_price_asof_valid": True,
                        "cb_market_return": cb_return * 0.2,
                        "chosen_market_index": "CSI500",
                        "stock_index_beta": 1.0,
                        "stock_shock_z": stock_return / 0.002,
                        "stock_shock_amount_z": 0.2,
                        "cb_amount_z_5m": 0.1,
                        "relative_amount": 1.0,
                        "high_low_range_proxy_z": 0.1,
                        "vwap_gap": 0.0,
                        "premium_expansion_proxy_z": 0.0,
                        "stock_to_cb_beta_lag0": 0.45,
                        "stock_to_cb_beta_lag1": 0.1,
                        "stock_to_cb_beta_lag2": 0.05,
                        "stock_to_cb_beta_lag3": 0.02,
                        "cb_market_gamma": 0.2,
                        "linkage_regression_intercept": 0.0,
                        "linkage_regression_r2": 0.3,
                        "linkage_model_confidence": 0.8,
                        "linkage_residual": cb_return - 0.45 * stock_return,
                        "linkage_residual_z": 0.0,
                        "linkage_gap_z": 0.0,
                        "liquidity_sweet_spot_score_5m": 0.8,
                        "hot_money_risk_score_5m": 0.1,
                        "volume_climax_score": 0.1,
                        "liquidity_regime_5m": "NORMAL",
                        "daily_turnover": 2.0 + day_index * 0.01,
                        "adv20_amount": 1e8,
                        "active_bar_ratio": 1.0,
                        "zero_bar_ratio": 0.0,
                        "amihud": 1e-9,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["session_part"] = np.where(frame["bar_slot"] < 24, "AM", "PM")
    frame["session_slot"] = np.where(frame["bar_slot"] < 24, frame["bar_slot"], frame["bar_slot"] - 24)
    return frame


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return build_feature_dataset(source_frame(), config())


def model_frame(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2025-01-02", periods=12)
    frame = pd.DataFrame(index=range(rows))
    frame["trade_date"] = np.resize(dates, rows)
    frame["bond_code"] = np.where(np.arange(rows) % 2, "123001", "123002")
    for column in STAGE_A_NUMERIC_FEATURES:
        frame[column] = rng.normal(size=rows)
    for column in STAGE_A_CATEGORICAL_FEATURES:
        frame[column] = np.where(np.arange(rows) % 2, "A", "B")
    frame.loc[3, STAGE_A_NUMERIC_FEATURES[3]] = np.nan
    frame["surface_target_cb_return"] = (
        0.002 * frame["stock_return_current"]
        + 0.001 * np.square(frame["premium_lag1"])
        + rng.normal(0, 0.0001, rows)
    )
    return frame


def fitted_hmm(seed: int = 80) -> tuple[GaussianHMM, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = np.r_[
        rng.normal(-1, 0.3, size=(50, 2)),
        rng.normal(1, 0.3, size=(50, 2)),
    ]
    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=30,
        random_state=seed,
    ).fit(matrix, [50, 50])
    return model, matrix


def test_01_required_schema_fails_loudly() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_source_schema(pd.DataFrame({"trade_date": [pd.Timestamp("2025-01-01")]}))


def test_02_duplicate_aligned_bar_fails_loudly() -> None:
    raw = source_frame(days=1, bonds=1)
    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_source(duplicated, config())


def test_03_session_has_no_lunch_bars() -> None:
    raw = source_frame(days=1, bonds=1)
    hours = raw["bar_start"].dt.hour * 60 + raw["bar_start"].dt.minute
    assert not hours.between(690, 775).any()
    assert set(raw.groupby("session_part")["session_slot"].max()) == {23}


def test_04_current_cb_return_is_not_surface_feature() -> None:
    assert "cb_return" not in STAGE_A_NUMERIC_FEATURES
    assert "surface_target_cb_return" not in STAGE_A_NUMERIC_FEATURES


def test_05_forbidden_fundamentals_are_explicit() -> None:
    assert {"debt_ratio", "operating_cash_flow", "free_cash_flow", "ytm"}.issubset(
        FORBIDDEN_SOURCE_FEATURES
    )


def test_06_feature_builder_preserves_row_coverage(dataset: pd.DataFrame) -> None:
    assert len(dataset) == len(source_frame())
    assert dataset["bond_code"].nunique() == 2


def test_07_conversion_price_is_point_in_time_row_value(dataset: pd.DataFrame) -> None:
    daily = dataset.drop_duplicates(["bond_code", "trade_date"])
    early = daily[daily["trade_date"] < pd.Timestamp("2024-01-30")]
    late = daily[daily["trade_date"] >= pd.Timestamp("2024-01-30")]
    assert early["conversion_price"].eq(8.0).all()
    assert late["conversion_price"].eq(9.0).all()


def test_08_forward_label_does_not_cross_lunch(dataset: pd.DataFrame) -> None:
    labeled, _ = add_forward_labels(dataset, config())
    assert labeled.loc[labeled["bar_slot"].isin([21, 22, 23]), "forward_cb_return_15m"].isna().all()


def test_09_forward_label_does_not_cross_overnight(dataset: pd.DataFrame) -> None:
    labeled, _ = add_forward_labels(dataset, config())
    assert labeled.loc[labeled["bar_slot"].eq(47), "forward_cb_return_15m"].isna().all()


def test_10_missing_bar_does_not_become_shorter_horizon() -> None:
    raw = source_frame(days=1, bonds=1)
    raw = raw[~raw["bar_slot"].eq(2)].reset_index(drop=True)
    built = build_feature_dataset(raw, config())
    labeled, _ = add_forward_labels(built, config())
    assert pd.isna(labeled.loc[labeled["bar_slot"].eq(0), "forward_cb_return_15m"]).all()


def test_11_maximum_bar_has_no_future_target(dataset: pd.DataFrame) -> None:
    labeled, _ = add_forward_labels(dataset, config())
    row = labeled[labeled["bar_slot"].eq(47)]
    assert row[["forward_cb_return_15m", "forward_cb_return_30m", "forward_cb_return_60m"]].isna().all().all()


def test_12_future_change_does_not_change_past_features() -> None:
    raw = source_frame(days=2, bonds=1)
    baseline = build_feature_dataset(raw, config())
    changed = raw.copy()
    changed.loc[changed.index[-1], ["cb_close", "cb_return"]] *= 2.0
    rebuilt = build_feature_dataset(changed, config())
    columns = STAGE_A_NUMERIC_FEATURES + STAGE_A_CATEGORICAL_FEATURES
    pd.testing.assert_series_equal(baseline.loc[10, columns], rebuilt.loc[10, columns])


def test_13_daily_aggregate_is_lagged_one_day(dataset: pd.DataFrame) -> None:
    first_day = dataset["trade_date"].min()
    first = dataset[dataset["trade_date"].eq(first_day)]
    assert first["daily_turnover_lag1d"].isna().all()


def test_14_walk_forward_is_chronological_and_purged(dataset: pd.DataFrame) -> None:
    folds = build_walk_forward_folds(dataset, config())
    assert len(folds) == 2
    for fold in folds:
        assert fold.train_end < fold.validation_start
        assert fold.purge_days >= 1


def test_15_ridge_scaler_does_not_see_future_validation() -> None:
    frame = model_frame()
    train = frame.iloc[:240].copy()
    model = fit_ridge_surface(
        train,
        "surface_target_cb_return",
        STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES,
        config(),
    )
    scaler_mean = model.preprocessor.named_transformers_["numeric"].named_steps["scaler"].mean_[0]
    assert scaler_mean == pytest.approx(train[STAGE_A_NUMERIC_FEATURES[0]].median(), abs=0.2)


def test_16_imputer_handles_missing_feature() -> None:
    frame = model_frame()
    model = fit_ridge_surface(
        frame.iloc[:240],
        "surface_target_cb_return",
        STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES,
        config(),
    )
    assert np.isfinite(model.predict(frame.iloc[240:])).all()


def test_17_quantile_crossing_is_explicitly_corrected() -> None:
    ordered, crossing = enforce_quantile_order(np.array([[0.3, 0.1, 0.2], [0.1, 0.2, 0.3]]))
    assert crossing.tolist() == [True, False]
    assert np.all(ordered[:, 0] <= ordered[:, 1])
    assert np.all(ordered[:, 1] <= ordered[:, 2])


def test_18_lightgbm_is_seed_deterministic() -> None:
    frame = model_frame()
    cfg = config()
    first = fit_lightgbm_surface(
        frame, "surface_target_cb_return", STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES, cfg, objective="quantile", alpha=0.5,
    )
    second = fit_lightgbm_surface(
        frame, "surface_target_cb_return", STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES, cfg, objective="quantile", alpha=0.5,
    )
    np.testing.assert_allclose(first.predict(frame), second.predict(frame), atol=1e-12)


def test_19_xgboost_is_seed_deterministic() -> None:
    frame = model_frame()
    cfg = config()
    first = fit_xgboost_surface(
        frame, "surface_target_cb_return", STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES, cfg,
    )
    second = fit_xgboost_surface(
        frame, "surface_target_cb_return", STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES, cfg,
    )
    np.testing.assert_allclose(first.predict(frame), second.predict(frame), atol=1e-12)


def test_20_model_serialization_round_trip(tmp_path: Path) -> None:
    frame = model_frame()
    model = fit_ridge_surface(
        frame, "surface_target_cb_return", STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES, config(),
    )
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(model.predict(frame), loaded.predict(frame))


def test_21_causal_hmm_probability_sums_to_one() -> None:
    model, matrix = fitted_hmm()
    probabilities = causal_forward_filter(model, matrix, [50, 50])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)


def test_22_modifying_future_does_not_change_past_filtered_probability() -> None:
    model, matrix = fitted_hmm()
    first = causal_forward_filter(model, matrix, [100])
    changed = matrix.copy()
    changed[60:] *= -5.0
    second = causal_forward_filter(model, changed, [100])
    np.testing.assert_allclose(first[:60], second[:60], atol=1e-12)


def test_23_hmm_sequence_reset_reuses_initial_distribution() -> None:
    model, matrix = fitted_hmm()
    sequence = matrix[:10]
    doubled = np.r_[sequence, sequence]
    probabilities = causal_forward_filter(model, doubled, [10, 10])
    np.testing.assert_allclose(probabilities[0], probabilities[10], atol=1e-12)


def test_24_hmm_filter_has_no_smoothing_call() -> None:
    source = inspect.getsource(causal_forward_filter)
    assert "predict_proba" not in source
    assert "_do_backward_pass" not in source


def test_25_state_alignment_handles_label_switching() -> None:
    reference = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]])
    candidate = reference[[2, 0, 1]]
    assert align_states(reference, candidate) == {0: 2, 1: 0, 2: 1}


def test_26_hmm_seed_reproducibility() -> None:
    first, matrix = fitted_hmm(81)
    second, _ = fitted_hmm(81)
    np.testing.assert_allclose(first.transmat_, second.transmat_)
    np.testing.assert_allclose(
        causal_forward_filter(first, matrix, [50, 50]),
        causal_forward_filter(second, matrix, [50, 50]),
    )


def test_27_transition_matrix_is_valid() -> None:
    model, _ = fitted_hmm()
    assert np.all(model.transmat_ >= 0)
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0)


def test_28_response_gap_uses_oof_prediction() -> None:
    frame = pd.DataFrame(
        {
            "bond_code": ["123001"], "stock_code": ["300001"],
            "trade_date": [pd.Timestamp("2025-01-02")], "session_part": ["AM"],
            "surface_target_cb_return": [0.001], "oof": [0.003],
            "dynamic_beta_lag0": [0.5],
            "forward_cb_return_15m": [0.001], "forward_stock_return_15m": [0.0],
        }
    )
    cfg = config()
    cfg["labels"]["horizons_bars"] = [3]
    result, _ = add_oof_gap_and_closure_labels(frame, "oof", cfg)
    assert result.loc[0, "response_gap"] == pytest.approx(-0.002)


def test_29_lagger_catchup_is_distinct_from_leader_reversal() -> None:
    cfg = config()
    cfg["labels"]["horizons_bars"] = [3]
    common = {
        "bond_code": "123001", "stock_code": "300001",
        "trade_date": pd.Timestamp("2025-01-02"), "session_part": "AM",
        "surface_target_cb_return": 0.001, "oof": 0.003,
        "dynamic_beta_lag0": 0.5,
    }
    lagger = pd.DataFrame([{**common, "forward_cb_return_15m": 0.001, "forward_stock_return_15m": 0.0}])
    leader = pd.DataFrame([{**common, "forward_cb_return_15m": 0.0, "forward_stock_return_15m": -0.002}])
    lagger_result, _ = add_oof_gap_and_closure_labels(lagger, "oof", cfg)
    leader_result, _ = add_oof_gap_and_closure_labels(leader, "oof", cfg)
    assert bool(lagger_result.loc[0, "lagger_catchup_flag_15m"])
    assert bool(leader_result.loc[0, "leader_reversal_flag_15m"])


def test_30_tiny_gap_ratio_is_unavailable() -> None:
    cfg = config()
    cfg["labels"]["horizons_bars"] = [3]
    frame = pd.DataFrame(
        {
            "bond_code": ["123001"], "stock_code": ["300001"],
            "trade_date": [pd.Timestamp("2025-01-02")], "session_part": ["AM"],
            "surface_target_cb_return": [0.00001], "oof": [0.0],
            "dynamic_beta_lag0": [0.5], "forward_cb_return_15m": [0.001],
            "forward_stock_return_15m": [0.0],
        }
    )
    result, _ = add_oof_gap_and_closure_labels(frame, "oof", cfg)
    assert pd.isna(result.loc[0, "gap_close_ratio_15m"])


def test_31_cost_deduction_is_separate_from_gross() -> None:
    predictions = pd.DataFrame(
        {
            "C_SURFACE_STATE_prediction_forward_cb_return_60m": [0.003, 0.0],
            "forward_cb_return_60m": [0.002, -0.001],
        }
    )
    result = cost_sensitivity(predictions, config())
    conservative = result[result["scenario"].eq("conservative")].iloc[0]
    assert conservative["net_mean_return"] < conservative["gross_mean_return"]


def test_32_outcome_model_c_adds_state_probabilities() -> None:
    frame = model_frame()
    frame["state_probability_0"] = 0.5
    frame["state_probability_1"] = 0.3
    frame["state_probability_2"] = 0.2
    frame["state_argmax_probability"] = 0.5
    frame["state_entropy"] = 1.0
    frame["state_expected_duration"] = 3.0
    sets = outcome_feature_sets(frame, 3)
    assert "state_probability_0" not in sets["B_SURFACE"][0]
    assert "state_probability_0" in sets["C_SURFACE_STATE"][0]


def test_33_empty_response_surface_bucket_does_not_error() -> None:
    assert two_dimensional_response_surfaces(pd.DataFrame(), config()).empty


def test_34_quantile_and_model_prediction_schema_handles_categories() -> None:
    frame = model_frame()
    model = fit_lightgbm_surface(
        frame,
        "surface_target_cb_return",
        STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES,
        config(),
        objective="quantile",
        alpha=0.5,
    )
    changed = frame.iloc[:3].copy()
    changed[STAGE_A_CATEGORICAL_FEATURES[0]] = "UNSEEN"
    assert model.predict(changed).shape == (3,)


def test_35_executable_return_enters_at_next_bar_open(dataset: pd.DataFrame) -> None:
    cfg = config()
    cfg["labels"]["horizons_bars"] = [3]
    working = dataset.copy()
    first_date = working["trade_date"].min()
    next_open = (
        working["bond_code"].eq("123001")
        & working["trade_date"].eq(first_date)
        & working["bar_slot"].eq(1)
    )
    working.loc[next_open, "cb_open"] *= 1.001
    labeled, _ = add_forward_labels(working, cfg)
    group = labeled[
        labeled["bond_code"].eq("123001")
        & labeled["trade_date"].eq(labeled["trade_date"].min())
        & labeled["session_part"].eq("AM")
    ].sort_values("bar_slot")
    signal = group.iloc[0]
    expected = group.iloc[3]["cb_close"] / group.iloc[1]["cb_open"] - 1.0
    assert signal["forward_executable_cb_return_15m"] == pytest.approx(expected)
    assert signal["forward_executable_cb_return_15m"] != pytest.approx(
        signal["forward_cb_return_15m"]
    )


def test_36_cost_evaluation_prefers_executable_return() -> None:
    predictions = pd.DataFrame(
        {
            "C_SURFACE_STATE_prediction_forward_cb_return_60m": [0.003],
            "forward_cb_return_60m": [0.10],
            "forward_executable_cb_return_60m": [0.002],
        }
    )
    result = cost_sensitivity(predictions, config())
    gross = result[result["scenario"].eq("gross")].iloc[0]
    assert gross["realized_return_column"] == "forward_executable_cb_return_60m"
    assert gross["gross_mean_return"] == pytest.approx(0.002)


def test_37_heldout_ale_uses_serialized_heldout_model(tmp_path: Path) -> None:
    cfg = config()
    cfg["strategy"]["output_dir"] = str(tmp_path)
    cfg["evaluation"]["ale_sample_rows"] = 300
    cfg["evaluation"]["ale_bins"] = 4
    cfg["evaluation"]["minimum_ale_bin_rows"] = 10
    feature = "stock_return_current"
    cfg["evaluation"]["ale_features"] = [feature]
    frame = model_frame()
    frame["fold_id"] = "HOLDOUT"
    model = fit_lightgbm_surface(
        frame,
        "surface_target_cb_return",
        STAGE_A_NUMERIC_FEATURES,
        STAGE_A_CATEGORICAL_FEATURES,
        cfg,
        objective="quantile",
        alpha=0.5,
        name="lightgbm_q50",
    )
    destination = tmp_path / "models" / "surface" / "HOLDOUT"
    destination.mkdir(parents=True)
    joblib.dump(model, destination / "lightgbm_q50.joblib")
    result = heldout_ale_profiles(frame, cfg, "HOLDOUT")
    assert not result.empty
    assert set(result["feature"]) == {feature}
    assert result["ale_value"].notna().all()
    assert set(result["data_scope"]) == {"strictly_heldout_validation"}


def test_38_stage_gate_rejects_fold_with_one_valid_hmm_seed() -> None:
    surface = pd.DataFrame(
        {
            "fold_id": ["F1", "F1", "F2", "F2", "F3", "F3"],
            "official": [True] * 6,
            "model": ["v071_fair_linear", "lightgbm_q50"] * 3,
            "rmse": [2.0, 1.0] * 3,
        }
    )
    outcome_rows = []
    for fold in ["F1", "F2", "F3"]:
        for horizon in ["30m", "60m"]:
            outcome_rows.extend(
                [
                    {"fold_id": fold, "horizon": horizon, "target": f"forward_cb_return_{horizon}", "model": "B_SURFACE", "rmse": 2.0},
                    {"fold_id": fold, "horizon": horizon, "target": f"forward_cb_return_{horizon}", "model": "C_SURFACE_STATE", "rmse": 1.0},
                ]
            )
    states = pd.DataFrame(
        {
            "fold_id": ["F1", "F1", "F2", "F2", "F3", "F3"],
            "canonical_state_id": [0, 1] * 3,
            "occupancy": [0.5] * 6,
            "semantic_label": ["A", "B"] * 3,
        }
    )
    stability = pd.DataFrame(
        {
            "fold_id": ["F1", "F1", "F2", "F2", "F3"],
            "seed_valid": [True, True, True, True, True],
            "adjusted_rand_vs_first_seed": [1.0, 0.8, 1.0, 0.8, 1.0],
        }
    )
    conclusion, details = determine_stage_gate(
        surface, pd.DataFrame(outcome_rows), states, stability
    )
    assert conclusion == "B_RESPONSE_SURFACE_ONLY"
    assert not details["state_seed_stability_pass"]
    assert details["minimum_valid_seed_count_per_fold"] == 1


def test_39_empty_stratified_diagnostics_does_not_error() -> None:
    assert stratified_outcome_diagnostics(pd.DataFrame(), pd.DataFrame(), config()).empty


def test_40_cli_accepts_no_cache_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["dynamic_response_v080", "--mode", "smoke-test", "--no-cache"])
    assert parse_args().use_cache is False
