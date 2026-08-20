# Convertible Bond Hybrid Strategy V0.7 Research Report

- Strategy version: `hybrid_v07`
- Config version: `hybrid_v07_pit_conversion_002`
- Requested range: `2026-03-02 .. 2026-04-10`
- Latest decision date in outputs: `2026-04-10`
- Scope: research only; no automatic orders.

## Data and point-in-time status

- Latest classified bonds: 170
- qIV solved rows: 4408 / 4910 (89.78%)
- Full local panel qIV/MC coverage: 31159 / 32539 (95.76%)
- qIV contamination assessable: 24253 / 31159 solved rows; unavailable rows are not called clean.
- LSMC evaluated rows: 3247 / 4910
- Reliable YTM coverage is zero. Discounting uses risk-free plus a documented credit-spread proxy.
- PIT conversion-price coverage: 32194 / 32539 (98.94%).
- Same-day unadjusted stock-price coverage: 32198 / 32539 (98.95%).
- Joint PIT pricing-input coverage: 32193 / 32539 (98.94%). Missing rows remain unavailable and are never filled with future conversion prices or adjusted stock prices.
- The 170-bond historical universe was selected in 2026-07, so historical results retain survivorship/selection bias.

## Bond type snapshot

| bond_type                |   count |
|:-------------------------|--------:|
| BALANCED_CORE            |      84 |
| EQUITY_LIKE              |      59 |
| SPECULATIVE_HIGH_PREMIUM |      15 |
| BOND_LIKE                |      12 |

## Horizon decisions

| selected_horizon   |   count |
|:-------------------|--------:|
| WATCH_ONLY         |    4371 |
| OVERNIGHT_DAILY    |     518 |
| INTRADAY_T0        |      21 |

- BALANCED_CORE HOLD examples: 64
- INTRADAY_T0 examples: 21
- CARRY_OVERNIGHT examples: 522
- CLAUSE_DRIVEN/ILLIQUID excluded examples: 0

## Portfolio comparison

| strategy   |   cost_bps_nominal |   cost_stress_multiplier |   trade_count |   average_holding_days |   overnight_exposure_count |   days |   total_return |   annualized_return |     sharpe |   max_drawdown |   hit_rate |   worst_day |   mean_daily_return |   turnover | strategy_version   | config_version                |
|:-----------|-------------------:|-------------------------:|--------------:|-----------------------:|---------------------------:|-------:|---------------:|--------------------:|-----------:|---------------:|-----------:|------------:|--------------------:|-----------:|:-------------------|:------------------------------|
| intraday   |                  5 |                        5 |            21 |                0       |                          0 |     14 |     -0.0091217 |           -0.152059 |  -0.687983 |     -0.0491095 |   0.357143 |  -0.0205057 |        -0.000573664 |   0.3      | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| daily_lsmc |                  5 |                        5 |           237 |                5.1308  |                        174 |     28 |     -0.0749582 |           -0.504034 |  -2.36077  |     -0.0819767 |   0.464286 |  -0.0457074 |        -0.00262789  |   1.46071  | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| hybrid     |                  5 |                        5 |           221 |                4.61991 |                        200 |     29 |     -0.0555568 |           -0.391464 |  -2.23846  |     -0.0777178 |   0.482759 |  -0.0327494 |        -0.00188284  |   0.832931 | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| intraday   |                 10 |                        5 |            21 |                0       |                          0 |     14 |     -0.0432691 |           -0.548959 |  -3.68618  |     -0.0685358 |   0.214286 |  -0.0230057 |        -0.00307366  |   0.3      | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| daily_lsmc |                 10 |                        5 |           237 |                5.1308  |                        174 |     28 |     -0.0972236 |           -0.601689 |  -3.11132  |     -0.0918301 |   0.357143 |  -0.0482074 |        -0.00349245  |   1.46071  | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| hybrid     |                 10 |                        5 |           221 |                4.61991 |                        200 |     29 |     -0.0817134 |           -0.523247 |  -3.35922  |     -0.0915686 |   0.37931  |  -0.0343744 |        -0.0028475   |   0.832931 | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| intraday   |                 20 |                        5 |            21 |                0       |                          0 |     14 |     -0.10831   |           -0.872987 |  -9.68258  |     -0.115004  |   0.214286 |  -0.0280057 |        -0.00807366  |   0.3      | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| daily_lsmc |                 20 |                        5 |           237 |                5.1308  |                        174 |     28 |     -0.140235  |           -0.743304 |  -4.56478  |     -0.116441  |   0.321429 |  -0.0532074 |        -0.00522156  |   1.46071  | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| hybrid     |                 20 |                        5 |           221 |                4.61991 |                        200 |     29 |     -0.131964  |           -0.707642 |  -5.52753  |     -0.120369  |   0.344828 |  -0.0376244 |        -0.00477682  |   0.832931 | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| intraday   |                 30 |                        5 |            21 |                0       |                          0 |     14 |     -0.169226  |           -0.964462 | -15.679    |     -0.17135   |   0.142857 |  -0.0330057 |        -0.0130737   |   0.3      | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| daily_lsmc |                 30 |                        5 |           237 |                5.1308  |                        174 |     28 |     -0.181293  |           -0.834745 |  -5.94752  |     -0.142476  |   0.321429 |  -0.0582074 |        -0.00695068  |   1.46071  | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| hybrid     |                 30 |                        5 |           221 |                4.61991 |                        200 |     29 |     -0.179579  |           -0.820935 |  -7.5769   |     -0.148326  |   0.275862 |  -0.0408744 |        -0.00670615  |   0.832931 | hybrid_v07         | hybrid_v07_pit_conversion_002 |

At 10 bps nominal cost, fees are stressed by 5x before liquidity impact. Results: intraday=-4.327% (21 trades), daily_lsmc=-9.722% (237 trades), hybrid=-8.171% (221 trades). These are research backtests, not deployment evidence.

## Existing 5-minute ACTION benchmark

| evidence_layer         |   cost_bps_nominal |   cost_stress_multiplier |   action_count |   label_available_count |   mean_return_30m_after_stressed_fee |   median_return_30m_after_stressed_fee |   hit_rate_30m_after_stressed_fee |   mfe_6bar_mean |   mae_6bar_mean | first_trade_date   | last_trade_date   | signal_type_counts                                    | data_mode_counts         | friction_scope                                             | comparable_to_daily_backtest   | strategy_version   | config_version                |
|:-----------------------|-------------------:|-------------------------:|---------------:|------------------------:|-------------------------------------:|---------------------------------------:|----------------------------------:|----------------:|----------------:|:-------------------|:------------------|:------------------------------------------------------|:-------------------------|:-----------------------------------------------------------|:-------------------------------|:-------------------|:------------------------------|
| v04_5min_action_events |                  5 |                        5 |            144 |                     142 |                          -0.00140843 |                             -0.0017672 |                         0.373239  |      0.00424261 |     -0.00310307 | 2026-05-19         | 2026-07-03        | {"LAG_REPAIR_LONG": 94, "MOMENTUM_BREAKOUT_LONG": 50} | {"manual_backfill": 144} | stressed_nominal_fee_only;daily_liquidity_model_not_joined | False                          | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| v04_5min_action_events |                 10 |                        5 |            144 |                     142 |                          -0.00390843 |                             -0.0042672 |                         0.204225  |      0.00424261 |     -0.00310307 | 2026-05-19         | 2026-07-03        | {"LAG_REPAIR_LONG": 94, "MOMENTUM_BREAKOUT_LONG": 50} | {"manual_backfill": 144} | stressed_nominal_fee_only;daily_liquidity_model_not_joined | False                          | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| v04_5min_action_events |                 20 |                        5 |            144 |                     142 |                          -0.00890843 |                             -0.0092672 |                         0.084507  |      0.00424261 |     -0.00310307 | 2026-05-19         | 2026-07-03        | {"LAG_REPAIR_LONG": 94, "MOMENTUM_BREAKOUT_LONG": 50} | {"manual_backfill": 144} | stressed_nominal_fee_only;daily_liquidity_model_not_joined | False                          | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| v04_5min_action_events |                 30 |                        5 |            144 |                     142 |                          -0.0139084  |                             -0.0142672 |                         0.0633803 |      0.00424261 |     -0.00310307 | 2026-05-19         | 2026-07-03        | {"LAG_REPAIR_LONG": 94, "MOMENTUM_BREAKOUT_LONG": 50} | {"manual_backfill": 144} | stressed_nominal_fee_only;daily_liquidity_model_not_joined | False                          | hybrid_v07         | hybrid_v07_pit_conversion_002 |

This event benchmark reuses official V0.4 ACTION episodes and is not merged into the daily/Hybrid equity curve because its calendar differs. It applies the 5x nominal fee stress but does not pretend that the daily liquidity-friction model was observed at each old event.

## Ablation

| ablation                | strategy   |   trade_count |   days |   total_return |   annualized_return |   sharpe |   max_drawdown |   hit_rate |   worst_day |   mean_daily_return |   turnover |   cost_bps_nominal |   cost_stress_multiplier | ablation_identified   | notes                                                                                                                                     | strategy_version   | config_version                |
|:------------------------|:-----------|--------------:|-------:|---------------:|--------------------:|---------:|---------------:|-----------:|------------:|--------------------:|-----------:|-------------------:|-------------------------:|:----------------------|:------------------------------------------------------------------------------------------------------------------------------------------|:-------------------|:------------------------------|
| A_NO_LSMC               | hybrid     |           586 |     29 |     -0.131792  |           -0.707139 | -7.79772 |     -0.1218    |   0.241379 |  -0.0245245 |         -0.00481486 |   1.2969   |                 10 |                        5 | True                  | Rule RV/residual daily baseline rebuilt without LSMC hold decisions.                                                                      | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| B_WITH_LSMC             | hybrid     |           221 |     29 |     -0.0817134 |           -0.523247 | -3.35922 |     -0.0915686 |   0.37931  |  -0.0343744 |         -0.0028475  |   0.832931 |                 10 |                        5 | True                  | nan                                                                                                                                       | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| C_NO_QIV_MC             | hybrid     |           229 |     29 |     -0.0553715 |           -0.390425 | -2.55447 |     -0.0661374 |   0.448276 |  -0.0288218 |         -0.00189538 |   0.711552 |                 10 |                        5 | True                  | Full research ablation uses empirical-bootstrap-only LSMC continuation, removes qIV/MC entry fields, and recomputes the horizon selector. | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| D_NO_LIQUIDITY_FRICTION | hybrid     |           221 |     29 |     -0.0374992 |           -0.2826   | -1.46815 |     -0.0668539 |   0.517241 |  -0.0312773 |         -0.0012314  |   0.832931 |                 10 |                        5 | True                  | nan                                                                                                                                       | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| E_NO_BOND_TYPE          | hybrid     |           251 |     29 |     -0.0775025 |           -0.503911 | -3.41385 |     -0.0796239 |   0.344828 |  -0.0313348 |         -0.00270146 |   0.943793 |                 10 |                        5 | True                  | Selector/type-gate ablation; continuation regressions remain type-conditioned.                                                            | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| F_INCLUDE_CLAUSE_DRIVEN | hybrid     |           221 |     29 |     -0.0817134 |           -0.523247 | -3.35922 |     -0.0915686 |   0.37931  |  -0.0343744 |         -0.0028475  |   0.832931 |                 10 |                        5 | False                 | No point-in-time CLAUSE_DRIVEN sample is available.                                                                                       | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| G_HIGH_MOMENTUM_WEIGHT  | hybrid     |          1053 |     29 |     -0.138449  |           -0.726085 | -4.52359 |     -0.121722  |   0.310345 |  -0.0446343 |         -0.00497757 |   1.32466  |                 10 |                        5 | True                  | Old-style monitoring baseline permits high momentum to dominate entry.                                                                    | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| H_LOW_MOMENTUM_WEIGHT   | hybrid     |           221 |     29 |     -0.0817134 |           -0.523247 | -3.35922 |     -0.0915686 |   0.37931  |  -0.0343744 |         -0.0028475  |   0.832931 |                 10 |                        5 | True                  | nan                                                                                                                                       | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| I_INTRADAY_ONLY         | intraday   |            21 |     14 |     -0.0432691 |           -0.548959 | -3.68618 |     -0.0685358 |   0.214286 |  -0.0230057 |         -0.00307366 |   0.3      |                 10 |                        5 | True                  | nan                                                                                                                                       | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| J_DAILY_ONLY            | daily_lsmc |           237 |     28 |     -0.0972236 |           -0.601689 | -3.11132 |     -0.0918301 |   0.357143 |  -0.0482074 |         -0.00349245 |   1.46071  |                 10 |                        5 | True                  | nan                                                                                                                                       | hybrid_v07         | hybrid_v07_pit_conversion_002 |
| K_HYBRID_HORIZON        | hybrid     |           221 |     29 |     -0.0817134 |           -0.523247 | -3.35922 |     -0.0915686 |   0.37931  |  -0.0343744 |         -0.0028475  |   0.832931 |                 10 |                        5 | True                  | nan                                                                                                                                       | hybrid_v07         | hybrid_v07_pit_conversion_002 |

Ablations are evaluated after stressed costs. Adding LSMC changes total return versus the rebuilt no-LSMC rule baseline by 5.008%. Adding qIV/MC structural mapping versus empirical-only LSMC changes total return by -2.634%; this ablation is identified with empirical-bootstrap-only LSMC.

## Research answers

1. BALANCED_CORE and selected EQUITY_LIKE rows are the intended intraday candidates, but only when residual/RV edge clears stressed friction. Momentum alone never triggers a trade.
2. BALANCED_CORE, BOND_LIKE, and selected EQUITY_LIKE rows are eligible for daily holding. BOND_LIKE is not treated as ordinary lag repair.
3. CLAUSE_DRIVEN, ILLIQUID, and extreme SPECULATIVE_HIGH_PREMIUM rows are excluded from ordinary LSMC. DISTRESS_REVERSAL remains experimental.
4. LSMC improves the no-LSMC prototype in this slice but does not produce positive cost-after performance. Positive simulated hold edge is therefore miscalibrated as a tradable forecast.
5. Hybrid is not proven superior. It loses less than Daily LSMC at the tested costs, while Intraday has too few observations for a fair comparison.
6. Type and per-date outputs should be checked for concentration; a small number of bonds/dates can dominate this limited sample.
7. qIV/MC now have historical point-in-time inputs for most canonical rows, but independence from momentum still requires the reported ablation and walk-forward stability rather than coverage alone.
8. qIV contamination is reported only among solved rows; unsolved rows are not silently treated as clean.
9. Wind daily turnover is included when available and directly enters liquidity/friction states.
10. Momentum weight is 0.05 and monitoring-only. The high-momentum ablation is worse in this slice, which supports the downgrade direction but not a final parameter.
11. Clause data are unavailable historically, so clause-pollution conclusions cannot be made from the current dataset.
12. DISTRESS_REVERSAL should remain a separate research framework because price dynamics and event information are not captured by ordinary residual models.
13. LSMC is most sensitive to path residuals, volatility, friction buffers, horizon, and credit spread.
14. Credit spread sensitivity is stored for -200/-100/0/+100/+200 bps, but it remains a proxy because YTM and complete terms are unavailable.
15. Intraday positions should flatten when the repair edge is gone, friction rises, liquidity deteriorates, or no independent daily hold case exists.
16. Carry overnight is allowed only when an independent LSMC hold edge clears rebalance, overnight-risk, and model-uncertainty buffers; current backtest calibration does not yet justify acting on it.
17. `loss_to_overnight_prohibition=true` prevents a losing intraday trade from becoming an overnight position solely to avoid realizing a loss.
18. Current data are sufficient to exercise the research pipeline, not to support deployment conclusions.
19. Bond-type thresholds, credit spreads, qIV structural assumptions, and LSMC path mappings remain research hypotheses.

## Current limitations

- No reliable YTM, coupon schedule, call/put/reset history, or rating history. Conversion-price history is reconstructed point-in-time from AkShare conversion-value anchors, JQData unadjusted closes, and JQData effective adjustment events.
- Structural MC is a discounted-par plus terminal-conversion proxy, not a complete convertible-bond pricer.
- Historical universe construction is not point-in-time and has survivorship bias.
- PIT conversion-price and unadjusted-stock support are not perfectly complete. Missing rows remain unavailable; qIV solve coverage is also constrained by the deliberately simplified proxy-pricer range.
- Daily holding episodes enter at the next trading-day open, mark to market each day, and exit at the next executable open after an LSMC/rule exit; same-close execution is forbidden.
- The common-slice `INTRADAY_T0` portfolio is an EOD-planned next-day T+0 proxy. Existing V0.4 five-minute ACTION events are reported separately because their dates do not overlap the daily training history.
- Intraday event and daily evidence cover different calendar samples. Direct hybrid claims require a common, point-in-time dataset.
