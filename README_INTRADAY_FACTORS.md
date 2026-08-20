# 可转债 5 分钟级日内因子监控 V0.2

## 目标

本模块在当前 watchlist 的 5 分钟 bar 完成后计算日内监控因子，并把研究字段写入 `cb_intraday_factor_snapshot`，把人工看盘卡片写入 `cb_intraday_alerts`。它只输出监控信号，不连接交易接口，不生成订单。

V0.2 不再把 `signal_score` 当作默认交易排序 alpha。默认交易提醒是触发型：`setup_score` 判断是否值得盯盘，`trigger_score` 判断当前 bar 是否弹人工确认，`exit_state` 提示均值回归完成或动量衰减。

## 数据表和字段

默认配置位于 `config/data_schema.yaml`。

实时 5 分钟行情表：

- 表名：`cb_watchlist_bar_5m_live`
- 关键字段：`bar_start`, `asset_type`, `bond_code`, `stock_code`, `open`, `high`, `low`, `close`, `volume`, `amount`, `conversion_price`
- `asset_type=CB` 为转债 bar，`asset_type=STOCK` 为对应正股 bar
- `bar_start` 被视为 5 分钟 bar 的开始时间，因子只处理已完成 bar

watchlist 和转股价快照表：

- 表名：`cb_watchlist_daily`
- 关键字段：`ts`, `bond_code`, `stock_code`, `conversion_price`
- 转股价通过 `bond_code + ts <= bar_start` 做 point-in-time as-of join

因子结果表：

- 表名：`cb_intraday_factor_snapshot`
- 写入使用确定性 `ts = bar_start + hash(bond_code, factor_version)`，重复执行同一版本、同一 bar 不产生重复行

人工提醒表：

- 表名：`cb_intraday_alerts`
- 关键字段：`alert_state`, `signal_type`, `setup_score`, `trigger_score`, `action_reason_text`, `risk_hint_text`, `cooldown_suppressed`, `alert_expire_time`
- `ACTION_LONG` 只可能来自 `LAG_REPAIR_LONG` 或 `MOMENTUM_BREAKOUT_LONG`
- `WATCH_LONG` 只作为面板候选，不进入默认交易型回测

## 核心因子

- `stock_momentum_score`：正股日内动量综合分，默认组合 30m、1d、5d 的 past-only robust z-score。历史不足时可只由短周期动量驱动。
- `factor_residual_mr`：股债残差均值回归核心因子。滚动回归使用 `t-1` 及以前的 `r_stock` 和可用的 `r_cb_market` 估计 beta/gamma，当前 bar 只用于计算残差。
- `factor_premium_mr`：转股溢价率同 slot 偏离的均值回归因子。条件溢价模型接口保留，V0 默认使用 `premium_slot_z`。
- `factor_vol_mr`：残差波动率短期异常扩张时降低追单倾向。
- `factor_vol_response_gap`：经验敏感度推算的转债反应缺口，不是真实期权定价。
- `flow_confirmation`：转债和正股 bar 级 signed amount 的确认指标。
- `cb_mom_10m_z`, `cb_mom_15m_z`, `cb_mom_30m_z`：短周期转债动量，主要用于突破型触发。

## 触发型信号

- `LAG_REPAIR_LONG`：正股 30m/60m 动量转强，转债残差仍滞涨，成交额不死，溢价未明显过热。
- `MOMENTUM_BREAKOUT_LONG`：正股和转债短周期共振向上，成交额放大，残差尚未明显透支。
- `WATCH_LONG`：只进监控面板，不默认买入。
- `MEAN_REVERSION_COMPLETE`、`MOMENTUM_DECAY`、`PREMIUM_OVERHEAT`：作为 `EXIT_HINT` 的来源。

`alert_controls` 控制单债 cooldown、单债日内上限、全市场日内上限、14:50 后禁止新 ACTION，以及 ACTION 过期 bar 数。

低波动率不再直接作为 alpha。`volatility_gate_pass` 只作为可交易性 gate，`noise_penalty` 会降低 `setup_score`。240 根 primary 波动率历史不足时，会输出 fallback 字段，但默认不进入 ACTION 综合分。

## 数据质量和缺失值

每行输出：

- `data_quality_pass`
- `missing_cb_bar`
- `missing_stock_bar`
- `stale_price_flag`
- `insufficient_history_flag`
- `fallback_standardization_flag`
- `amount_source`
- `factor_version`

如果 `amount` 缺失，会使用 `volume * close` 构造 `amount_used`，并把 `amount_source` 标为 `proxy_volume_close`。这不是换手率，也不被称为真实成交额。

如果转股价缺失，`parity`、`premium_raw` 和溢价相关因子不可用，不使用默认值替代。

## 标准化和无未来约束

- 市场时区：`Asia/Shanghai`
- 午休不生成虚假 slot
- 同 slot 标准化按 `bond_code + slot_id` 使用历史样本，当前 bar 不进入自身 median/MAD
- rolling median、MAD、回归参数都只使用 `t-1` 及以前数据
- 信号在 t bar 收盘后形成，回测评价最早按 t+1 bar open/后续收益评价

## YTM、live_turnover 和 Greeks

V0 不依赖 YTM 或 live_turnover。当前数据源没有稳定提供这些字段，因此模块不会静默伪造它们。

`empirical_delta_proxy`、`elasticity_proxy`、`empirical_gamma_proxy` 都是经验型 proxy，不是真实 Greeks，不包含债底、YTM、评级或完整期权定价。

## 配置

主配置：`config/intraday_factors.yaml`

可配置项包括：

- residual rolling window 和 min observations
- same-slot lookback
- z-score clip
- liquidity thresholds
- legacy research `signal_score` weights
- manual alert thresholds
- alert cooldown controls
- volatility primary/fallback windows
- setup score weights

默认流动性阈值偏宽松，必须结合本地 TDengine 中成交额量纲和实际分布再校准。

## 运行命令

初始化表结构：

```bash
cd /path/to/convertible-bonds-strategy
python init_tdengine.py
```

历史回填：

```bash
cd /path/to/convertible-bonds-strategy
python run_intraday_factors.py \
  --mode backfill \
  --start "2026-06-25 09:30:00" \
  --end "2026-06-25 15:00:00" \
  --write
```

增量计算：

```bash
cd /path/to/convertible-bonds-strategy
python run_intraday_factors.py \
  --mode incremental \
  --asof "2026-06-25 10:35:00" \
  --write
```

示例 cron 调用：

```cron
*/5 1-7 * * 1-5 /path/to/convertible-bonds-strategy/scripts/run_intraday_factors_incremental.sh
```

这只是示例，当前实现没有自动修改已有行情入库 cron。

## 评价器

```bash
cd /path/to/convertible-bonds-strategy
python evaluate_intraday_factors.py \
  --start "2026-06-25 09:30:00" \
  --end "2026-06-25 15:00:00" \
  --cost-bps 10
```

输出包括 pooled panel IC、单债时间序列 IC、LONG_CANDIDATE 命中率、平均未来收益和按债/月分组表现。

V0.2 默认交易型回测只统计：

- `alert_state = ACTION_LONG`
- `signal_type in [LAG_REPAIR_LONG, MOMENTUM_BREAKOUT_LONG]`

输出包括每类 signal 的样本数、30m/60m 均值和中位收益、残差收益、hit rate、MFE、MAE、触发后 1/2/3 根 bar 平均收益、cooldown 压制次数，以及 14:50 后被过滤的触发次数。

## Warm-up 要求

V0 可以在历史不足时运行，但会设置 `insufficient_history_flag` 或 `fallback_standardization_flag`。残差模型、同 slot 标准化和 20 日成交额代理需要多个交易日后才更稳定。

## 已知限制

- 当前生产 live 表是日内 RAM 式表，默认不保存多日 5 分钟历史；因此 20 日同 slot 标准化在生产 live 表上会经常回退到滚动历史。
- primary 波动率窗口为 240 根，单日 live 数据下常常只能看到 fallback 展示字段。
- 正股和转债 5 分钟行情来自 AkShare spot 聚合，不是交易所逐笔或 Wind L2。
- 条件溢价残差模型、经验 gamma 的稳定研究字段已预留，但 V0 默认不纳入综合分数。
- 流动性过滤使用成交额、活跃 bar、Amihud 和高低价差代理，不使用 live_turnover。
