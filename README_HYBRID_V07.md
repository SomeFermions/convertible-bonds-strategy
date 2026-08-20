# Convertible Bond Hybrid Strategy V0.7

`research.hybrid_v07` is a research-only layer for comparing:

- intraday T+0 relative-value trades;
- multi-day daily holdings;
- a hybrid horizon selector;
- LSMC continuation value;
- qIV and Monte Carlo pricing proxies;
- stressed liquidity and execution friction.

It does not modify collection jobs, cron, TDengine retention, alerts, or order
interfaces. It never sends orders.

## Point-in-time rules

The historical 5-minute training table is aggregated to daily CB/stock pairs.
Price, volume, amount, and Wind daily turnover are used only on their own dates.
Static screening fields are joined only when their snapshot date is no later
than the decision date.

Historical conversion prices are joined as explicit point-in-time intervals:

```text
effective_start <= trade_date < effective_end_exclusive
```

The baseline is inferred from AkShare historical conversion value and JQData
unadjusted stock close, while JQData conversion-price adjustment events split
the intervals. Historical parity and premium are derived only when both that
PIT conversion price and a same-day unadjusted stock price are present. Missing
rows remain unavailable; adjusted stock prices and future conversion prices are
never used as pricing fallbacks.

Reliable YTM coverage is zero. Discounting uses:

```text
risk_free_rate + configured_credit_spread_proxy
```

The credit proxy records bond type, fundamental, distress, and liquidity
adjustments. Every MC row stores -200/-100/0/+100/+200 bps sensitivity. The
proxy is not described as observed YTM or a complete market valuation.

## Layers

1. `bond_type`: point-in-time classification into `BALANCED_CORE`,
   `EQUITY_LIKE`, `BOND_LIKE`, `SPECULATIVE_HIGH_PREMIUM`,
   `DISTRESS_REVERSAL`, `CLAUSE_DRIVEN`, or `ILLIQUID`.
2. `friction`: 5x fee stress plus spread, slippage, impact, and liquidity-risk
   proxies. Liquidity is split into illiquid, sweet-spot, normal, and climax.
3. `qIV/MC`: discounted-par plus conversion-option qIV and reproducible
   terminal-conversion Monte Carlo. Missing terms and clauses are flagged.
4. `LSMC`: GBM and empirical-bootstrap paths with backward Ridge/OLS/Huber
   continuation regression.
5. `trade_horizon_selector`: research decisions only. Momentum and moving
   averages cannot independently trigger a trade.

`factor_vol_response_gap` is compatibility-only and has zero V0.7 weight.

## LSMC assumptions

The structural mapping is a deliberately limited proxy:

```text
max(discounted par redemption, conversion ratio * terminal stock price)
```

It omits coupon schedules, call/put/reset clauses, default recovery, and
issuer-specific term history. The empirical mapping bootstraps historical
stock returns and CB residuals using only observations before the decision
date. Both mapping diagnostics are exposed.

For each simulated path, LSMC compares future net liquidation values with
estimated continuation values. A `HOLD` requires the hold edge to exceed:

```text
daily rebalance friction
+ overnight risk buffer
+ model uncertainty buffer
```

`loss_to_overnight_prohibition=true`: a losing intraday position cannot be
carried merely to avoid realizing a loss. It needs an independent daily LSMC
case.

## Execution and backtest

Daily close-generated decisions execute no earlier than the next trading-day
open. Same-close fills are forbidden. Holding episodes are marked to market
daily and exit at the next executable open after `SELL`, hold-edge failure, or
the configured holding limit.

Costs are tested at 5/10/20/30 bps nominal and multiplied by the configured 5x
stress factor before adding liquidity friction. Results are reported for all
three portfolios and every ablation, including losing variants.

The 200-day historical universe was selected in July 2026. Results therefore
retain survivorship and sample-selection bias and are not deployment evidence.

## Commands

Run from the repository root.

```bash
python -m research.hybrid_v07 \
  --mode classify-bonds \
  --start-date 2025-06-16 \
  --end-date 2026-07-22
```

```bash
python -m research.hybrid_v07 \
  --mode price-daily \
  --start-date 2025-06-16 \
  --end-date 2026-07-22 \
  --paths 5000
```

```bash
python -m research.hybrid_v07 \
  --mode build-lsmc-dataset \
  --start-date 2025-06-16 \
  --end-date 2026-07-22
```

```bash
python -m research.hybrid_v07 \
  --mode run-lsmc \
  --start-date 2026-03-02 \
  --end-date 2026-07-22 \
  --backtest-paths 256 \
  --path-model gbm,empirical_bootstrap \
  --regression-model ridge \
  --n-jobs -1
```

For a 5,000-path-per-model latest snapshot:

```bash
python -m research.hybrid_v07 \
  --mode run-lsmc \
  --start-date 2026-07-22 \
  --end-date 2026-07-22 \
  --paths 5000 \
  --path-model gbm,empirical_bootstrap \
  --regression-model ridge \
  --n-jobs -1
```

For the equal-total-path empirical-only no-qIV/MC ablation:

```bash
python -m research.hybrid_v07 \
  --mode run-lsmc-no-qiv-ablation \
  --start-date 2026-03-02 \
  --end-date 2026-04-10 \
  --paths 512 \
  --regression-model ridge \
  --n-jobs -1
```

```bash
python -m research.hybrid_v07 \
  --mode select-horizon \
  --start-date 2026-03-02 \
  --end-date 2026-07-22
```

```bash
python -m research.hybrid_v07 \
  --mode backtest \
  --start-date 2026-03-02 \
  --end-date 2026-04-10 \
  --strategies intraday,daily_lsmc,hybrid \
  --cost-bps 5,10,20,30
```

```bash
python -m research.hybrid_v07 \
  --mode ablation \
  --start-date 2026-03-02 \
  --end-date 2026-04-10
```

```bash
python -m research.hybrid_v07 \
  --mode report \
  --start-date 2026-03-02 \
  --end-date 2026-04-10
```

## Outputs

All outputs are file-backed research tables under
`outputs/research/hybrid_v07/`:

- `cb_bond_type_snapshot.csv`
- `conversion_price_acceptance.csv`
- `cb_daily_relative_value_snapshot.csv`
- `cb_lsmc_daily_snapshot.csv`
- `cb_hybrid_trade_decision.csv`
- `bond_type_report.csv`
- `qiv_rv_report.csv`
- `lsmc_dataset.csv`
- `lsmc_snapshot.csv`
- `lsmc_latest_5000_snapshot.csv`
- `lsmc_no_qiv_mc_ablation_snapshot.csv`
- `hybrid_trade_episodes.csv`
- `intraday_event_benchmark.csv`
- `intraday_vs_daily_vs_hybrid.csv`
- `ablation_report.csv`
- `cost_sensitivity.csv`
- `type_performance.csv`
- `lsmc_diagnostics.csv`
- `hybrid_v07_report.md`

These files are logical research tables. V0.7 intentionally does not alter the
production TDengine schema.
