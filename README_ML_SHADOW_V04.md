# 可转债日内监控 V0.4 ML Shadow Meta-Filter

## 定位

V0.4 在现有 V0.3 规则触发系统上增加一层机器学习影子过滤器。它只用于研究、解释、排序和人工看盘辅助，不改变 `cb_intraday_alerts` 里的正式 `alert_state`，也不生成生产交易指令。

当前系统仍然是：

- 慢数据决定“谁值得被看”；
- 盘中 5 分钟快数据决定“当前 bar 是否值得盯盘 / 追 / 退出”；
- 交易由人工在同花顺模拟仓确认；
- 不做隔夜，不做全自动高频，不做全市场连续 ranking alpha。

ML shadow 的输入是规则系统已经产生的 WATCH / ARMED / ACTION / EXIT 事件或 episode，不直接训练“所有 5 分钟 bar 涨跌预测”。这样可以避免把规则外的噪声样本灌进模型，也符合人工半自动交易的使用场景。

## 为什么不是端到端涨跌预测

当前 27-30 个交易日样本里，规则 ACTION 约百余条，ARMED 数百条，仍然是小样本。随机切分或全 bar 预测会严重高估效果，因为同一天、同一只债、相邻 5 分钟 bar 高度相关。

V0.4 采用 meta-labeling：

1. 规则引擎先生成 `WATCH`、`ARMED`、`ACTION`、`EXIT`。
2. ML 只评价这些事件是否更值得人工执行、是否更像假突破、ARMED 是否更可能转 ACTION。
3. 模型输出 `p_*` 和 `ml_action_hint`，仅作为 shadow 字段展示。

## 日内 T+0 约束

所有标签和 episode 都遵守同日边界：

- 不跨交易日；
- 不使用第二天开盘价；
- 不计算隔夜收益；
- `force_flatten_time` 后不得继续持仓；
- `no_new_action_after` 后不得生成新开仓 ACTION 训练样本；
- `15:00` bar 可以进入 factor / label，但不能作为新开仓 ACTION。

默认配置：

```yaml
ml_shadow_v04:
  shadow_only: true
  no_overnight: true
  no_new_action_after: "14:30:00"
  exit_hint_after: "14:50:00"
  force_flatten_time: "14:55:00"
  allow_15_00_bar_for_label: true
  allow_15_00_bar_for_new_action: false
```

## 数据集单位

逻辑数据集名：

- `cb_intraday_ml_event_dataset`
- `cb_intraday_ml_predictions`
- `cb_intraday_ml_experiments`

当前实现默认写到 `outputs/ml_shadow/` 下的 CSV。若环境安装了 parquet engine，会同时导出 parquet；否则自动跳过 parquet，不影响研究流程。

每一行代表一个事件级 / episode anchor：

- `event_type`: `WATCH` / `ARMED` / `ACTION` / `EXIT`
- `signal_type`: `LAG_REPAIR_LONG`、`MOMENTUM_BREAKOUT_LONG`、`ARMED_LAG_LONG`、`ARMED_BREAKOUT_LONG` 等
- `signal_time`
- `first_seen_time`
- `alert_inserted_at`
- `effective_asof`
- `data_mode`
- `factor_version`
- `config_version`
- `feature_version`
- `label_version`
- `unreliable_live_day`
- `is_actionable_bar`
- `minutes_to_close`
- `no_new_action_after_suppressed`
- `force_flatten_time`

连续重复 ACTION 会在 episode 内去重，避免同一次机会被当成多条独立训练样本。

## 标签

固定 horizon 标签：

- `return_1bar`
- `return_2bar`
- `return_3bar`
- `return_30m`
- `return_60m`
- `return_to_eod`
- `residual_return_30m`
- `residual_return_60m`
- `residual_return_to_eod`

MFE / MAE 标签：

- `mfe_3bar`
- `mae_3bar`
- `mfe_6bar`
- `mae_6bar`
- `mfe_to_eod`
- `mae_to_eod`

Meta labels：

- `label_action_good_30m`
- `label_action_good_to_eod`
- `label_action_bad_3bar`
- `label_triple_barrier_6bar`
- `label_fake_breakout`
- `label_lag_repair_success`
- `label_lag_repair_fail`
- `label_armed_to_action_1bar`
- `label_armed_to_action_3bar`
- `label_armed_false_alarm`
- `label_unavailable_due_to_eod`

当尾盘不足 30m / 60m 时，固定 horizon 标签标记为不可用，不跨到次日。此时可以使用 `return_to_eod` 研究同日剩余可执行空间。

## 特征

特征来自 V0.3 factor / alert / episode 字段，不包含未来收益、未来 MFE、未来 MAE。

主要分组：

- 信号身份：`signal_type`、`event_type`、`pool_state`、`time_bucket`、`is_actionable_bar`、`minutes_to_close`
- 转债形态：`parity`、`premium_raw`、`premium_slot_z`、价格/溢价/流动性分桶
- 残差：`residual_z`、`residual_ewm`、`residual_z_change_1bar`、`residual_repair_started`
- 正股脉冲：`stock_mom_5m_z`、`stock_mom_10m_z`、`stock_mom_15m_z`、`stock_impulse_score`
- 转债动量：`cb_mom_5m_z`、`cb_mom_10m_z`、`cb_mom_15m_z`、`cb_mom_30m_z`
- 溢价变化：`premium_change_5m_z`、`premium_change_10m_z`、`premium_change_30m_z`
- VWAP：`cb_vwap_dev`、`stock_vwap_dev`、`vwap_gap`
- 资金流 proxy：`relative_amount`、`cb_flow_z`、`stock_flow_z`、`flow_confirmation`
- 噪声和可交易性：`residual_vol_ratio_z`、`high_low_range_proxy_z`、`volatility_gate_pass`、`noise_penalty`
- 市场环境：`cb_market_mom_30m`、`cb_market_breadth_30m`、`market_regime_intraday`
- 执行可见性：`first_seen_delay_seconds`、`live_pipeline_delay_seconds`、`data_mode`

## 模型

默认轻量模型：

- Rule baseline：使用规则分数作为基准；
- Logistic Regression：可解释方向和系数；
- HistGradientBoosting fallback：作为树模型 / boosting 近似；
- Calibrated classifier：对概率做 sigmoid 校准；
- Small MLP：接口保留，默认关闭。

如果环境没有 sklearn，会回退到 numpy 实现。当前环境已安装 `scikit-learn`，默认使用 sklearn。

不默认使用 CUDA / 深度学习，因为当前样本量不足以支撑复杂模型结论。

## 验证方法

不使用随机 train/test split。

默认使用 walk-forward by `trade_date`：

- 前 N 日训练；
- 下一日测试；
- 逐日滚动；
- 输出每个测试日和全周期指标；
- 记录 purging / embargo 设置。

分类指标：

- sample_count
- positive_rate
- AUC
- PR-AUC
- accuracy
- precision
- recall
- F1
- Brier score
- calibration bucket

交易指标：

- 按预测概率 top/bottom 分桶；
- 1/2/3 bar return；
- 30m / to_eod return；
- hit rate；
- MFE / MAE；
- worst day / worst bond；
- signal reduction ratio。

## 三层收益口径

V0.4 区分：

- `theoretical_signal_return`: signal bar close 后，按 next bar open 入场；
- `live_visible_return`: alert 可见后，按 first executable price 入场；
- `manual_realized_return`: 人工模拟仓真实成交和退出。

当前没有 manual log 时，只输出 theoretical / live-visible 需要的字段接口。后续接入人工日志后，可计算：

- `actual_decision_time`
- `actual_entry_price`
- `actual_exit_time`
- `actual_exit_price`
- `slippage_vs_next_bar_open`
- `slippage_vs_first_seen_price`
- `realized_return`

## 命令

构建数据集：

```bash
cd /path/to/convertible-bonds-strategy
python -m research.ml_shadow_v04 \
  --mode build-ml-dataset \
  --start-date 2026-05-18 \
  --end-date 2026-07-06 \
  --factor-version v0.3 \
  --config-version intraday_factors_v0.4 \
  --pool-scope ACTIVE_ONLY \
  --event-scope ARMED,ACTION \
  --output-table cb_intraday_ml_event_dataset \
  --live-lookback-days 70 \
  --cost-bps 10
```

训练 ACTION good shadow model：

```bash
cd /path/to/convertible-bonds-strategy
python -m research.ml_shadow_v04 \
  --mode train-ml-shadow \
  --dataset cb_intraday_ml_event_dataset \
  --target label_action_good_30m \
  --model-type logistic \
  --start-date 2026-05-18 \
  --end-date 2026-07-06 \
  --event-scope ACTION
```

训练 BREAKOUT 假突破模型：

```bash
python -m research.ml_shadow_v04 \
  --mode train-ml-shadow \
  --dataset cb_intraday_ml_event_dataset \
  --target label_fake_breakout \
  --model-type logistic \
  --event-scope ACTION \
  --signal-scope MOMENTUM_BREAKOUT_LONG
```

训练 ARMED -> ACTION 模型：

```bash
python -m research.ml_shadow_v04 \
  --mode train-ml-shadow \
  --dataset cb_intraday_ml_event_dataset \
  --target label_armed_to_action_3bar \
  --model-type logistic \
  --event-scope ARMED
```

生成 shadow prediction：

```bash
python -m research.ml_shadow_v04 \
  --mode predict-ml-shadow \
  --dataset cb_intraday_ml_event_dataset \
  --model-version <model_version> \
  --prediction-table cb_intraday_ml_predictions \
  --event-scope ARMED,ACTION
```

评估 prediction：

```bash
python -m research.ml_shadow_v04 \
  --mode evaluate-ml-shadow \
  --dataset cb_intraday_ml_event_dataset \
  --prediction-table cb_intraday_ml_predictions
```

查看实验记录：

```bash
python -m research.ml_shadow_v04 \
  --mode report-ml-shadow
```

## Alert Card Join 示例

ML prediction 不覆盖规则 alert，可在展示层 join：

```sql
SELECT
  a.signal_time,
  a.bond_code,
  a.stock_code,
  a.alert_state,
  a.signal_type,
  p.p_action_good_30m,
  p.p_fake_breakout,
  p.p_lag_repair_success,
  p.p_armed_to_action_3bar,
  p.confidence_bucket,
  p.ml_action_hint,
  p.ml_reason_text
FROM cb_intraday_alerts a
LEFT JOIN cb_intraday_ml_predictions p
  ON a.bond_code = p.bond_code
 AND a.signal_type = p.signal_type
 AND a.signal_time = p.signal_time;
```

## 结果解读

`ml_action_hint` 含义：

- `BOOST`: shadow 模型认为该规则事件相对更值得关注；
- `NEUTRAL`: 暂无强过滤或增强证据；
- `SUPPRESS`: shadow 模型认为短期坏结果 / 假突破风险较高；
- `WATCH_ONLY`: 只展示，不进入默认交易训练或执行判断。

这些字段不改变生产 ACTION。即使 `SUPPRESS`，规则系统的 ACTION 也仍然存在，只是给人工一个风险提示。

## 当前限制

- 当前样本约 27-30 个交易日，对 ML 仍然偏小；
- BREAKOUT 和 LAG 的子样本更小，模型指标不稳定；
- in-sample prediction 只能当展示和假设生成，不能当 out-of-sample 结论；
- CSV 已可用，parquet 需要额外安装 `pyarrow` 或 `fastparquet`；
- manual trade log 尚未接入，暂时不能验证人工真实滑点和实际成交收益；
- SHADOW_RESEARCH 池不进入正式交易指标，只能作为研究对照。

至少累计更多交易日，并区分真实 live incremental 与 backfill 后，才适合讨论 shadow bucket 是否能进入人工默认排序。
