# Dynamic Response V0.8.0 Research Report

## Research contract

- Strategy version: `dynamic_response_v080`
- Factor version: `response_surface_v080`
- Config version: `dynamic_response_v080_001`
- Research only; no production alert or order integration.
- Same-bar CB return is a target, never a response-surface feature.
- All response gaps are chronological OOF predictions.
- HMM probabilities use causal forward filtering, never future smoothing.
- Intraday rolling windows and labels do not cross lunch or overnight.

## Data coverage

- Range: 2025-06-16 to 2026-04-08
- Trading dates: 180
- Rows: 1,384,176
- Bonds: 170
- Feature count: 60
- Universe bias: `FUTURE_SELECTED_TRAINING_SAMPLE_SOURCE`
- Historical fundamental PIT status: `UNAVAILABLE_FOR_HISTORICAL_RECONSTRUCTION`

This historical sample remains a future-selected research universe. Results are therefore diagnostic and cannot be described as an unbiased historical production-universe backtest.

## Response surface

| model                     |   folds |        mae |       rmse |   spearman |
|:--------------------------|--------:|-----------:|-----------:|-----------:|
| lightgbm_mean             |       3 | 0.00128957 | 0.00235389 | 0.59522789 |
| lightgbm_q10              |       3 | 0.00201255 | 0.00316179 | 0.52567488 |
| lightgbm_q50              |       3 | 0.00128932 | 0.00235332 | 0.59539181 |
| lightgbm_q50_with_bond_id |       3 | 0.00128915 | 0.00235260 | 0.59565210 |
| lightgbm_q90              |       3 | 0.00204980 | 0.00315294 | 0.50731653 |
| ridge                     |       3 | 0.00136977 | 0.00242189 | 0.56541710 |
| v071_fair_linear          |       3 | 0.00118939 | 0.00206894 | 0.67366477 |
| v071_linear               |       3 | 0.00116416 | 0.00206773 | 0.67627956 |
| xgboost_mean              |       3 | 0.00128943 | 0.00227711 | 0.59373320 |

Mean q10-q90 OOF interval coverage: 74.452%.
Quantile crossing is explicitly detected and monotonically reordered; the raw crossing rate remains in `quantile_calibration.csv`.
Held-out first-order ALE is computed without refitting the model and is stored in `ale_profiles.csv`.

## Hidden state

- Selected states: 5
- Covariance: `diag`
- Maximum fold/state occupancy: 44.561%
- Median cross-seed ARI: 0.9846
- Daily reset: True; lunch reset: True

Semantic labels are assigned only where emission signatures are sufficiently distinctive; ambiguous states remain `STATE_K`. Future outcomes are not HMM inputs.

## Model A/B/C

| model           | horizon   | target                |    sample_count |        mae |       rmse |    spearman |   directional_accuracy |   positive_rate |   brier_score |   roc_auc |   pr_auc |   accuracy | strategy_version      | factor_version        | config_version            |   model_seed |
|:----------------|:----------|:----------------------|----------------:|-----------:|-----------:|------------:|-----------------------:|----------------:|--------------:|----------:|---------:|-----------:|:----------------------|:----------------------|:--------------------------|-------------:|
| A_BASELINE      | 15m       | forward_cb_return_15m | 282954.33333333 | 0.00274184 | 0.00443580 |  0.01855114 |             0.50205543 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| B_SURFACE       | 15m       | forward_cb_return_15m | 282954.33333333 | 0.00274872 | 0.00444251 |  0.03945395 |             0.51159472 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| C_SURFACE_STATE | 15m       | forward_cb_return_15m | 282954.33333333 | 0.00274905 | 0.00444266 |  0.03947298 |             0.51176786 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| A_BASELINE      | 30m       | forward_cb_return_30m | 242988.66666667 | 0.00386011 | 0.00618563 |  0.00807047 |             0.49949817 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| B_SURFACE       | 30m       | forward_cb_return_30m | 242988.66666667 | 0.00387485 | 0.00620085 |  0.02920701 |             0.50685854 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| C_SURFACE_STATE | 30m       | forward_cb_return_30m | 242988.66666667 | 0.00387604 | 0.00620152 |  0.02920439 |             0.50616247 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| A_BASELINE      | 60m       | forward_cb_return_60m | 162812.33333333 | 0.00542704 | 0.00857372 | -0.01336143 |             0.49387083 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| B_SURFACE       | 60m       | forward_cb_return_60m | 162812.33333333 | 0.00544765 | 0.00859618 |  0.01519832 |             0.49871188 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| C_SURFACE_STATE | 60m       | forward_cb_return_60m | 162812.33333333 | 0.00544970 | 0.00859705 |  0.01672081 |             0.49906636 |             nan |           nan |       nan |      nan |        nan | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |

Model A is the V0.7.1-style linear context, Model B adds the nonlinear OOF response surface and response gap, and Model C adds causal filtered state probabilities. All use identical official validation rows within each horizon.

## Stratified diagnostics

Descriptive OOF strata: bond, fold, hidden_state, liquidity, pool_scope, premium, source_vendor, stock_shock_direction, stock_shock_size, time_slot, trade_date, volatility.
Buckets are evaluation-only and are not recycled into training or parameter selection. Detailed fold, bond, date, shock, premium, volatility, liquidity, time-slot, state, source, and pool results are in `stratified_outcomes.csv`.

## Gap closure attribution

For gap `g_t`, required closure direction is `d=-sign(g_t)`. Lagger catch-up is `d * future_cb_return`; leader reversal is `-d * beta_t * future_stock_return`. This prevents a stock reversal from being mislabeled as CB catch-up merely because absolute gap falls.

## Cost proxy

| scenario     | horizon   | prediction_model                                 | realized_return_column           |   selected_sample_count |   selection_rate |   half_spread_bps |   slippage_bps |   impact_bps |   total_cost_bps |   gross_mean_return |   net_mean_return |   gross_hit_rate |   net_hit_rate | strategy_version      | factor_version        | config_version            |   model_seed |
|:-------------|:----------|:-------------------------------------------------|:---------------------------------|------------------------:|-----------------:|------------------:|---------------:|-------------:|-----------------:|--------------------:|------------------:|-----------------:|---------------:|:----------------------|:----------------------|:--------------------------|-------------:|
| gross        | 60m       | C_SURFACE_STATE_prediction_forward_cb_return_60m | forward_executable_cb_return_60m |                  319720 |       0.30249016 |        0.00000000 |     0.00000000 |   0.00000000 |       0.00000000 |          0.00002007 |        0.00002007 |       0.48944076 |     0.48944076 | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| conservative | 60m       | C_SURFACE_STATE_prediction_forward_cb_return_60m | forward_executable_cb_return_60m |                   93322 |       0.08829284 |        3.00000000 |     5.00000000 |   2.00000000 |      10.00000000 |          0.00006258 |       -0.00093742 |       0.49350635 |     0.41364309 | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |
| stress       | 60m       | C_SURFACE_STATE_prediction_forward_cb_return_60m | forward_executable_cb_return_60m |                    8869 |       0.00839105 |        6.00000000 |    10.00000000 |   5.00000000 |      21.00000000 |          0.00004473 |       -0.00205527 |       0.49971812 |     0.37647987 | dynamic_response_v080 | response_surface_v080 | dynamic_response_v080_001 |      2026080 |

Candidate returns use the next valid bar open as executable entry and the configured horizon close as exit. These are conservative proxy costs, not observed bid-ask execution. Gross and net estimates are kept separate.

## Stage gate

**Decision: `C_NO_STABLE_INCREMENT`**

- Surface wins: 0 / 3 official folds.
- State increment wins: 2 / 6 30m/60m comparisons.
- State occupancy pass: True.
- State seed stability pass: False.
- Minimum valid HMM seeds in any official fold: 1.
- Canonical-state semantic stability: 60.0% (pass=False).

Neither layer demonstrates stable majority-fold incremental value under the present contract. The negative result should redirect work toward data frequency, label quality, alignment, and observable microstructure rather than additional model complexity.

## Limitations

- The universe is future-selected and cannot support an unbiased historical deployment claim.
- Five-minute bars cannot identify responses completed inside the same bar.
- No true bid-ask, L2 flow, YTM, company fundamentals, credit model, or subjective clause labels enter V0.8.
- Current bar CB range/close-derived premium is excluded from Stage A to avoid target leakage; only lagged CB microstructure is used.
- HMM state semantics are descriptive, not causal economic proof.
- No model score is connected to production alerts or orders.

## Reproducibility

- Python: `3.13.13 | packaged by conda-forge | (main, Apr  8 2026, 02:00:33) [GCC 14.3.0]`
- Seed: `2026080`
- Exact package versions: `environment_manifest.json`
- Fold dates: `fold_definition.csv`
- Feature contract: `feature_schema.json`
