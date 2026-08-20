# Convertible Bond Research System

Research and data infrastructure for intraday convertible-bond signals. The
project combines full-market daily screening, a smaller 5-minute monitoring
universe, point-in-time data contracts, event-based signal evaluation, and
chronological walk-forward experiments.

> Research status: **PAUSED** (decision date: 2026-08-10). The repository is a
> research record, not a production trading system, an automated-ordering
> authorization, or a claim of deployable returns.

## What Is Here

The project has two related tracks:

- **Data engineering:** daily universe construction, ACTIVE/OBSERVE pool
  management, CB/stock/index 5-minute alignment, TDengine persistence, quality
  flags, and incremental/backfill-compatible contracts.
- **Strategy research:** continuous factors, sparse trigger events, episode
  evaluation, ML shadow filters, hybrid horizons, distributed-lag diagnostics,
  nonlinear response surfaces, and causal hidden-state experiments.

The main research path is documented in [RESEARCH_MAP.md](RESEARCH_MAP.md). It
explains the evolution from V0.1 through V0.8, the evidence hierarchy, rejected
hypotheses, and the conditions required to restart the research.

## Current Conclusion

The latest 180-trading-day chronological walk-forward work found that:

- the composite `signal_score` is not a stable continuous ranking alpha;
- most stock-to-bond response occurs in the same 5-minute bar;
- executable lagged response is too weak to reliably cover delay and costs;
- qIV/Monte Carlo, LSMC, ML, nonlinear models, and hidden states did not add
  stable out-of-sample value to the base signal;
- missing bid/ask, depth, tick-level execution, and higher-frequency synchronized
  data remain material identification limits.

These are research conclusions under the repository's data and execution
contracts, not universal claims about convertible-bond strategies.

## Repository Layout

| Path | Purpose |
|---|---|
| `intraday_factors/` | Factor contracts, engine, alerts, evaluation, and storage |
| `research/` | V0.7-V0.8 research pipelines and diagnostics |
| `training_data_v06/` | Historical canonical dataset construction |
| `scripts/` | Operational collection and research entry points |
| `tests/` | Unit and contract tests |
| `config/` | Non-secret configuration templates |
| `outputs/research/` | Selected compact reports only |

## Environment and Verification

The operational stack expects Python, local market-data vendor access, and
TDengine. Vendor credentials and machine-specific environment files are not
versioned. Research ML dependencies are listed in
`requirements-research-ml.txt`; other dependencies depend on the data adapter
and runtime being used.

Run the test suite from the repository root:

```bash
python -m pytest -q
```

Generated market data, feature matrices, serialized models, logs, and most
research outputs are intentionally excluded from Git. The committed reports are
small audit artifacts; they are not sufficient to reproduce experiments without
the corresponding licensed/local datasets.

## Safety Boundary

The repository has no production order-routing contract. Any future deployment
requires a separate review of data licensing, point-in-time correctness,
execution assumptions, risk limits, credentials, and compliance requirements.
