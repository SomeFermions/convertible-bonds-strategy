# Dynamic Response V0.8.0

V0.8 is a research-only nonlinear stock-bond response and latent-state layer. It
does not alter collectors, cron, TDengine schemas, watchlists, production factors,
alerts, or order handling.

## Scope

The response surface estimates the current five-minute convertible-bond return
after the current stock bar is observed. The current CB return is the target and
is never an input. CB path, premium, range, flow, and liquidity fields are lagged
by at least one bar. Same-day daily aggregates are replaced by prior-day values.
No fundamentals, YTM, credit labels, subjective clause labels, qIV, or MC fields
enter this model.

The primary causal response gap is:

```text
g_t = r_cb_t - OOF_prediction(r_cb_t | X_t)
```

Every prediction used to create `g_t` comes from a model trained on earlier
trade dates. Imputers, scalers, category mappings, feature selection, early
stopping, and optional tuning are fitted inside the training period.

## Labels

For horizon `h` in 3, 6, and 12 bars:

```text
forward_cb_return_h = cb_close(t+h) / cb_close(t) - 1
forward_stock_return_h = stock_close(t+h) / stock_close(t) - 1
forward_residual_return_h = forward_cb_return_h
                            - beta_lag0(t) * forward_stock_return_h
future_gap_h = g_t + forward_cb_return_h
                 - beta_lag0(t) * forward_stock_return_h
gap_abs_reduction_h = abs(g_t) - abs(future_gap_h)
gap_close_ratio_h = gap_abs_reduction_h / abs(g_t)
```

Gap close ratio is unavailable when `abs(g_t)` is below the configured one-basis-
point denominator floor. A clipped `[-5, 5]` version is used for stable outcome
regression; the raw value remains in the research artifact.

To distinguish CB catch-up from stock reversal, let `d = -sign(g_t)`:

```text
lagger_catchup_contribution = d * forward_cb_return_h
leader_reversal_contribution = -d * beta_lag0(t) * forward_stock_return_h
```

MFE and MAE use bars `t+1` through `t+h` relative to `cb_close(t)`. Labels require
an exact slot difference and never cross lunch or a trade date.

Potential execution diagnostics separately use `cb_open(t+1)` as the earliest
entry and `cb_close(t+h)` as exit. Cost sensitivity uses this executable return,
not the same-bar close-to-close research label.

## Models

- V0.7.1 dynamic-linear diagnostic baseline
- Pooled Ridge baseline
- LightGBM mean, q10, q50, and q90 heads
- XGBoost mean head
- Optional LightGBM q50 comparison with bond ID
- Gaussian HMM candidates with 3-6 states and diagonal/full covariance

LightGBM is the primary quantile model. XGBoost quantiles are not synthesized when
the installed API does not provide a stable equivalent. Quantile crossing is
recorded and corrected by row-wise monotonic ordering.

Optuna tuning is optional (`--retune`), bounded, and performed only inside the
first training period. It never sees the final validation or holdout periods.

## Causal Hidden States

`hmmlearn` estimates Gaussian emissions and transitions. Online state probability
uses a local log-space forward recursion:

```text
P(S_t | X_1, ..., X_t)
```

The implementation does not call `predict_proba` or forward-backward smoothing.
Sequences reset by bond and trade date; the default also treats AM and PM as
separate sequences. State IDs are aligned across folds by emission-center distance
with Hungarian matching. Economic labels are assigned conservatively after fit;
ambiguous states remain `STATE_K`.

## Model A/B/C

- **A_BASELINE**: V0.7.1-style linear context.
- **B_SURFACE**: A plus nonlinear OOF predictions, interval width, and response gap.
- **C_SURFACE_STATE**: B plus causal filtered state probabilities, entropy, and duration.

All three use the same date folds and horizon-specific validation rows. The report
requires at least three official folds before a stage can pass.

## Data Limitations

The audited V0.7.1 cache covers 180 trade dates and 170 bonds, but its historical
universe is explicitly tagged `FUTURE_SELECTED_TRAINING_SAMPLE_SOURCE`. V0.8 cannot
turn this into an unbiased point-in-time production universe. Results remain
diagnostic even when OOF modeling itself is leakage controlled.

Five-minute bars cannot capture an adjustment that completes inside the same bar.
There is no true bid-ask/L2 execution data, so net outcomes use documented cost
proxies and are always reported beside gross outcomes.

## Commands

Install research dependencies:

```bash
python -m pip install -r requirements-research-ml.txt
```

Run tests:

```bash
python -m pytest -q tests/test_dynamic_response_v080.py
```

Build only the dataset:

```bash
python -m research.dynamic_response_v080 \
  --config config/dynamic_response_v080.yaml \
  --mode build-dataset \
  --days 180
```

Run a 45-day end-to-end smoke test:

```bash
python -m research.dynamic_response_v080 \
  --config config/dynamic_response_v080.yaml \
  --mode smoke-test \
  --days 45 \
  --models ridge,lightgbm,xgboost \
  --no-cache
```

Run the complete 180-day backtest:

```bash
python -m research.dynamic_response_v080 \
  --config config/dynamic_response_v080.yaml \
  --mode backtest \
  --days 180 \
  --models ridge,lightgbm,xgboost \
  --use-cache
```

Explicit date ranges, tuning, seeds, and output roots are supported:

```bash
python -m research.dynamic_response_v080 \
  --config config/dynamic_response_v080.yaml \
  --mode backtest \
  --start-date 2025-06-16 \
  --end-date 2026-04-08 \
  --retune \
  --seed 2026080 \
  --output-dir outputs/research/dynamic_response_v080_tuned
```

Fit latest research artifacts without production integration:

```bash
python -m research.dynamic_response_v080 \
  --config config/dynamic_response_v080.yaml \
  --mode fit-latest \
  --days 180
```

Regenerate a report from existing artifacts:

```bash
python -m research.dynamic_response_v080 \
  --config config/dynamic_response_v080.yaml \
  --mode report
```

## Artifacts

The default root is `outputs/research/dynamic_response_v080/`. It contains the
dataset/environment manifests, feature schema, fold definitions, surface and
forward OOF predictions, filtered state probabilities, model comparison, state
diagnostics, feature importance, held-out SHAP and ALE summaries, quantile
calibration, executable-entry cost sensitivity, plots, serialized research
models, and the Markdown report. `fold_definition.csv` records fold-level row,
bond, feature-missingness, and target coverage; `model_parameter_manifest.json`
records exact shared parameters and fold-specific fitted iterations.
`stratified_outcomes.csv` reports fold/bond/date, shock direction and magnitude,
premium, volatility, liquidity, time slot, hidden state, source, and pool strata.
