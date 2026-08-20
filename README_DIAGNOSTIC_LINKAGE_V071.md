# V0.7.1 Diagnostic Reset + Stock-Bond Linkage

`diagnostic_linkage_v071` 是研究模块，不改变生产 alert，不连接自动交易接口，也不修改采集、TDengine、cron 或 shell 调度。

## 研究目标

本版本先回答三个基础问题：

1. 原有策略在零成本下是否存在稳定毛 Alpha，现实成本后是否仍可交易。
2. 正股剥离市场冲击后，转债在 lag1-lag3 是否存在 5min 频率可捕获的延迟响应。
3. 不使用 qIV/MC 或 LSMC 入场时，简单日频 relative-value 是否值得继续验证。

qIV/MC 的正式权重为 0，只保留覆盖率、污染率和横截面单调性诊断。LSMC entry 关闭，只在完全相同的简单 RV 入场上比较退出。

## 数据口径

模块只读取本地数据：

- `market_bar_5m_training_canonical`：转债与正股 5min canonical bar。
- `market_benchmark_bar_5m_canonical`：CSI300、CSI500、CSI1000、CHINEXT、STAR50。
- `cb_conversion_price_history_pit`：point-in-time 转股价。
- `wind_cb_ytm_turnover_raw`：仅使用日频换手率；不伪造或依赖 YTM。
- `jq_stock_daily_unadjusted_support`：未复权正股日频支持数据。
- V0.7 本地日频 panel、bond type、qIV/MC 和 LSMC 研究快照。

共同样本为四个互不重叠的 45 交易日窗口，共 180 个交易日。转股价和换手率在 common slice 中覆盖 100%。但是 canonical 历史样本来自 2026-07 的训练池选择，无法完整重建历史基本面/ST 准入，因此所有输出都带：

```text
FUTURE_SELECTED_TRAINING_SAMPLE_SOURCE
UNAVAILABLE_FOR_HISTORICAL_RECONSTRUCTION
```

这意味着回测可以诊断信号结构，不能声称是无 survivorship/selection bias 的历史策略收益。

## Linkage 定义

正股先按过去数据选择拟合最好的市场指数，并估计过去窗口 beta：

```text
stock_idiosyncratic_return = stock_return - beta * index_return
```

随后按债、按日滚动拟合 Ridge distributed-lag：

```text
cb_return = alpha
          + beta0 * stock_idio[t]
          + beta1 * stock_idio[t-1]
          + beta2 * stock_idio[t-2]
          + beta3 * stock_idio[t-3]
          + gamma * cb_market_return
          + error
```

每个决策日的参数只使用前一交易日及更早数据。`beta0` 只用于诊断；正式 actionable beta 为 `beta1 + beta2 + beta3`。当前 bar 收盘形成信号后，最早按下一根有效 bar open 成交。

事件包括：

- `STOCK_LEADS_CB_REPAIR`：正股冲击、linkage gap 为正且开始修复。
- `STOCK_LEADS_CB_NO_RESPONSE`：正股已动但转债持续不响应，只做拒绝/观察。
- `CB_LEADS_STOCK_INFORMATION`：只做次日信息研究，不触发转债日内追涨。
- `CB_OVERSHOOT_HOT_MONEY`：转债超调、溢价/成交高潮风险，不追涨。
- `LINKAGE_UNCERTAIN`：样本、模型或流动性不足。

## 成交与成本

日内信号以 signal bar close 形成，entry 使用下一根 open；`--entry-delay-bars 1` 和 `2` 分别再延迟一根、两根 bar。日频信号以收盘数据形成，最早次日 open 成交。

四套成本同时输出：

- `zero`：只看毛 Alpha。
- `realistic`：佣金、spread proxy、冲击和流动性惩罚。
- `conservative`：放大 spread 和冲击。
- `disaster_5x`：压力测试，不作为唯一主结果。

`latency_slippage_bps` 单独记录 signal close 到 entry open 的 implementation shortfall。由于回测入场价已经是延迟后的 open，它不会再次从净收益重复扣除。

## 统计与验证

- 四个固定 45 日窗口：train、validation、test、final holdout。
- expanding walk-forward，不随机切分。
- 日频 purge/embargo，日内 overlapping event 按交易日聚合。
- 置信区间使用按交易日 block bootstrap，并固定 seed。
- 同时报告最差窗口、最差日期、最差债券和类型/流动性分组。
- common-slice 使用相同日期、canonical pool、PIT 转股价和成本口径。

## 运行命令

全量构建并生成全部研究产物：

```bash
python -m research.diagnostic_linkage_v071 --mode all
```

分步骤运行：

```bash
python -m research.diagnostic_linkage_v071 --mode diagnostic-reset \
  --start-date 2025-06-16 --end-date 2026-04-08

python -m research.diagnostic_linkage_v071 --mode build-linkage-features \
  --start-date 2025-06-16 --end-date 2026-04-08 --lags 0,1,2,3

python -m research.diagnostic_linkage_v071 --mode build-linkage-events \
  --start-date 2025-06-16 --end-date 2026-04-08

python -m research.diagnostic_linkage_v071 --mode backtest-linkage \
  --start-date 2025-06-16 --end-date 2026-04-08 \
  --entry-delay-bars 0,1,2 \
  --cost-scenarios zero,realistic,conservative,disaster_5x

python -m research.diagnostic_linkage_v071 --mode backtest-daily-simple-rv \
  --start-date 2025-06-16 --end-date 2026-04-08 \
  --holding-days 1,3,5 --top-n 5,10

python -m research.diagnostic_linkage_v071 --mode evaluate-lsmc-exit-overlay \
  --start-date 2025-06-16 --end-date 2026-04-08

python -m research.diagnostic_linkage_v071 --mode evaluate-cb-leads-stock \
  --start-date 2025-06-16 --end-date 2026-04-08

python -m research.diagnostic_linkage_v071 --mode common-slice \
  --start-date 2025-06-16 --end-date 2026-04-08 \
  --strategies intraday_linkage,daily_simple_rv,daily_rv_lsmc_exit,hybrid

python -m research.diagnostic_linkage_v071 --mode ablation \
  --start-date 2025-06-16 --end-date 2026-04-08

python -m research.diagnostic_linkage_v071 --mode report \
  --start-date 2025-06-16 --end-date 2026-04-08
```

测试：

```bash
python tests/test_diagnostic_linkage_v071.py
python tests/test_hybrid_v07.py
```

仓库当前没有安装 `pytest`，两份测试文件均提供直接运行入口。

## 主要产物

输出目录为 `outputs/research/diagnostic_linkage_v071/`，核心文件包括：

- `point_in_time_universe.csv`
- `common_slice_manifest.csv`
- `cost_waterfall.csv`
- `benchmark_comparison.csv`
- `linkage_coefficients.csv`
- `lead_lag_by_bond.csv`
- `lead_lag_by_type.csv`
- `linkage_events.csv`
- `linkage_delay_sensitivity.csv`
- `dynamic_exit_comparison.csv`
- `daily_simple_rv.csv`
- `daily_exit_overlay.csv`
- `lsmc_matched_experiments.csv`
- `qiv_mc_coverage.csv`
- `qiv_monotonicity.csv`
- `cb_leads_stock_window_stability.csv`
- `common_slice_strategy_comparison.csv`
- `window_stability.csv`
- `ablation_report.csv`
- `diagnostic_linkage_v071_report.md`

## 当前研究结论

当前正式 linkage 动态策略没有跨窗口稳定的毛 Alpha，现实成本后为负，final holdout 也为负。动态退出不如固定波动率归一化退出。日频 top5/fixed5d 是更值得继续积累的 paper cohort，但窗口、中位数和历史 universe 偏差都不支持上线结论。

下一阶段只建议 alert/logging shadow：记录 `STOCK_LEADS_CB_REPAIR` 的 first seen、下一根 open、固定退出和动态失效路径；另行维护 top5/fixed5d 日频 paper cohort。现阶段不建议把这些结果转成模拟仓自动或人工下单规则。
