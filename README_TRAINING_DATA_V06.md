# V0.6 Historical Training Data Cache

This pipeline keeps JQData training bars, Wind enrichment, AkShare static data,
and the existing AkShare live tables separate.

## Trial Window

- Trial anchor date: `2026-07-14`.
- Permission-safe interval: first CN trading day on or after
  `anchor - 15 months + 14 calendar days`, through the last CN trading day on
  or before `anchor - 3 months`.
- Account-reported fixed interval: `2025-04-05` to `2026-04-12`.
- Current permission-safe interval: `2025-04-28` to `2026-04-10`.
- The exact 200-trading-day interval is derived from the cached calendar and
  ends on `2026-04-10`.
- CN trading dates are read from the one-time AkShare cache at
  `outputs/cn_trade_calendar.csv`.

The anchor is fixed in `config/training_data_v06.yaml`. Resume, repair,
coverage, and canonicalization therefore use the same dates throughout the
14-day trial.

## Current Plan

- Sample version: `jq_cb_train_v1`.
- Clean sample: 176 bonds (`ACTIVE_LIKE=7`, `ELIGIBLE_SHADOW=46`,
  `BROAD_BUT_CLEAN=123`).
- Unique source instruments: 352 (176 bonds and 176 stocks).
- Download plan: 7,040 ten-trading-day chunks.
- Estimated JQData rows: 3,379,200.
- Current status: `DRY_RUN_ESTIMATED`; no formal download has started.

Every source instrument has a stable `instrument_slot`. Raw and canonical
TDengine keys use `bar_start + instrument_slot milliseconds`, while the
business timestamp remains in `bar_start`/`bar_end`. This prevents instruments
sharing a five-minute bar from overwriting each other.

## Commands

Refresh the low-frequency CN trading calendar only when necessary:

```bash
python run_training_data_v06.py --mode refresh-cn-trade-calendar
```

Run the bounded JQData smoke test:

```bash
python run_training_data_v06.py \
  --mode jq-smoke-test \
  --sample-version jq_smoke_v1 \
  --bond-codes 113661,123124 \
  --days 1 \
  --end-date 2026-04-10 \
  --dry-run false
```

Rebuild the formal plan without vendor traffic:

```bash
python run_training_data_v06.py \
  --mode build-jq-download-plan \
  --sample-version jq_cb_train_v1 \
  --frequency 5m \
  --fields open,high,low,close,volume,money,paused \
  --dry-run true
```

After reviewing the quota estimate, promote the existing plan to `READY`:

```bash
python run_training_data_v06.py \
  --mode build-jq-download-plan \
  --sample-version jq_cb_train_v1 \
  --frequency 5m \
  --fields open,high,low,close,volume,money,paused \
  --dry-run false \
  --confirm-large-download
```

Run at most the configured daily hard cap:

```bash
python run_training_data_v06.py \
  --mode run-jq-download-plan \
  --sample-version jq_cb_train_v1 \
  --max-rows-today 450000 \
  --confirm-large-download
```

Canonicalize, report coverage, and build local windows after downloads finish:

```bash
python run_training_data_v06.py --mode canonicalize-jq-training-bars --sample-version jq_cb_train_v1
python run_training_data_v06.py --mode report-jq-training-coverage --sample-version jq_cb_train_v1
python run_training_data_v06.py --mode build-local-training-windows --sample-version jq_cb_train_v1 --window-trading-days 45 --max-windows 4
```

## Point-in-time conversion prices

Historical qIV inputs use `cb_conversion_price_history_pit`, whose intervals
are `[effective_start, effective_end_exclusive)`. The window baseline is
inferred from AkShare Eastmoney historical conversion value and a small
JQData `fq=None` stock-daily support pull. Subsequent interval boundaries and
prices come from JQData's low-frequency conversion-price adjustment table.
The current AkShare snapshot is never copied backward into historical dates.
The same unadjusted closes are persisted in
`jq_stock_daily_unadjusted_support`; qIV/MC uses this field while return and
factor features continue to use the existing canonical panel. A missing
vendor close after a valid history may use the previous unadjusted close for
qIV spot only; such rows retain `vendor_close = NULL`, `is_filled = true`, and
`price_status = PREVIOUS_UNADJUSTED_CLOSE`.

The AkShare responses and the small JQ support frames are cached locally, so
the command is resumable and repeated runs do not consume vendor quota:

```bash
env -u http_proxy -u https_proxy -u all_proxy \
  python run_training_data_v06.py \
  --mode collect-akshare-conversion-price-history \
  --sample-version jq_cb_train_v1 \
  --confirm-akshare-download
```

Per-bond coverage and source mismatches are recorded in
`cb_conversion_price_coverage_report`. Research loaders join the interval
table by bond and trade date; future intervals are not eligible.

## 5-minute completeness audit and market benchmarks

Audit the four local windows before research. CB rows before the first valid
listing date are reported as expected-unavailable and are not treated as a
repair request; missing/invalid rows inside an active CB span are repairable.

```bash
python run_training_data_v06.py \
  --mode audit-jq-training-bars \
  --sample-version jq_cb_train_v1
```

The five research benchmarks are stored separately from bond-stock canonical
bars:

- `CSI300` / `000300.XSHG` / 沪深300
- `CSI500` / `000905.XSHG` / 中证500
- `CSI1000` / `000852.XSHG` / 中证1000
- `CHINEXT` / `399006.XSHE` / 创业板指
- `STAR50` / `000688.XSHG` / 科创50

Tables:

- `jq_benchmark_download_plan`
- `jq_benchmark_bar_5m_raw`
- `market_benchmark_bar_5m_canonical`
- `jq_benchmark_coverage_report`

Plan without vendor traffic, then execute explicitly:

```bash
python run_training_data_v06.py \
  --mode collect-jq-benchmarks \
  --sample-version jq_cb_train_v1 \
  --dry-run true

env -u http_proxy -u https_proxy -u all_proxy \
  python run_training_data_v06.py \
  --mode collect-jq-benchmarks \
  --sample-version jq_cb_train_v1 \
  --dry-run false \
  --confirm-large-download \
  --max-rows-today 450000

python run_training_data_v06.py \
  --mode report-jq-benchmark-coverage \
  --sample-version jq_cb_train_v1
```

Canonicalization discovers every JQ raw batch in the selected sample/date scope,
including smoke-test rows reused by an idempotent formal plan. Use `--batch-id`
to canonicalize only one newly discovered raw batch.

Coverage distinguishes downloaded rows from usable bars:

- `row_coverage_ratio` measures expected row presence, including vendor
  placeholders for not-yet-listed or paused instruments.
- `coverage_ratio` uses `is_valid_bar` and is the research filtering ratio.
- `valid_cb_bars`, `invalid_cb_bars`, `valid_stock_bars`, and
  `invalid_stock_bars` retain the underlying counts.
- `coverage_pass` requires at least 90% valid paired CB/stock coverage. Failed
  samples remain in canonical data and are never silently discarded.

Load the Wind gateway environment before an explicitly approved enrichment:

```bash
set -a
source .wind_env
set +a
python run_training_data_v06.py \
  --mode run-wind-turnover-ytm-enrichment \
  --sample-version jq_cb_train_v1 \
  --scope active_like_only \
  --max-rows-week 45000 \
  --confirm-wind-download
```

Run a bounded P2 validation batch without including `ACTIVE_LIKE` again:

```bash
python run_training_data_v06.py \
  --mode run-wind-turnover-ytm-enrichment \
  --sample-version jq_cb_train_v1 \
  --scope validation_sample \
  --max-wind-cells-run 7000 \
  --max-rows-week 45000 \
  --confirm-wind-download
```

`validation_sample` requests only `live_turnover`. The ACTIVE_LIKE probe found
`ytm_b` unavailable for this Wind account, so validation batches do not spend a
second cell per date repeatedly requesting an all-null optional field.

Wind remains limited to daily `live_turnover` and `ytm`. JQData credentials
remain in `config/jqdata_credentials.toml`; neither credential file is logged
or committed.
