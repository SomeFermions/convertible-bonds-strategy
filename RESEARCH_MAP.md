# 可转债盘中策略研究地图（V0.1-V0.8）

> 状态：`PAUSED`<br>
> 决策日期：2026-08-10<br>
> 用途：研究复盘、内部汇报、未来重启评审<br>
> 不代表：生产交易规则、自动下单授权或可上线收益声明

## 1. 执行摘要

本项目从“全市场可转债数据采集”出发，依次研究了连续因子、触发型事件、人工预警、ML meta-filter、混合持有期限、股债 distributed-lag、非线性响应面和隐状态模型。研究复杂度逐步提高，证据口径也从单日/小样本理论收益，升级到 180 个交易日、chronological walk-forward、下一根可执行 bar 和保守成本。

截至 V0.8，最重要的结论不是“还差一个因子”，而是：

1. `signal_score` 没有表现为稳定的连续 ranking alpha。
2. 早期触发型 ACTION 在小样本、未充分计入执行磨损时出现过正收益，但在更长样本和真实可执行口径下没有保住。
3. 正股与转债的主要响应集中在同一根 5 分钟 bar；可执行的 lag1-lag3 响应过弱，难以覆盖人工延迟和成本。
4. qIV/MC、LSMC、ML shadow、非线性响应面和 HMM 都没有为基础信号提供稳定增量。
5. 缺少真实 intraday turnover、bid/ask spread、盘口深度、逐笔成交及更高频同步数据，是当前研究路线的核心识别限制。
6. 继续增加规则、z-score 或模型复杂度，边际研究价值已经很低。当前正确动作是暂停策略研究、保留资产、低成本积累数据，等待数据条件发生实质变化。

**当前阶段判定：暂停，而非失败清零。** 工程数据链、PIT 数据合同、walk-forward/evaluator、成本模型和负结果本身，都是以后汇报与重启时可复用的研究资产。

## 2. 一页研究路径

```text
数据工程基础
  全市场日频 + ACTIVE/OBSERVE 5min + CB/STOCK 对齐 + TDengine
        |
        v
V0.1 连续复合因子
  signal_score / residual / premium / momentum / flow
        |  pooled/cross-sectional IC 接近 0
        v
V0.2 触发型人工监控
  WATCH_LONG + LAG_REPAIR_LONG + MOMENTUM_BREAKOUT_LONG
        |  单日只有 1 条 ACTION，无法定参
        v
V0.3 ARMED / WATCH 细分 / episode evaluator
  27 日小样本出现正的理论 ACTION 收益
        |  尚未充分回答执行、成本和多日稳定性
        v
V0.4 ML Shadow Meta-Filter
  ACTION / LAG / BREAKOUT / ARMED 的事件级分类
        |  样本小、校准弱、无可靠模型增量
        v
V0.5 研究契约整合（非独立冻结策略）
  episode、first_seen、live/backfill/manual、可靠性标记
        |
        v
V0.6 历史训练数据底座
  5min canonical + benchmark + PIT 转股价 + 日频换手率
        |  将研究升级为 180 日 walk-forward，但暴露 future-selected universe
        v
V0.7 Hybrid Horizon
  bond type + friction + qIV/MC + LSMC + 日内/日频选择
        |  成本后整体为负；复杂度只减少交易，未创造 Alpha
        v
V0.7.1 Diagnostic Reset + Linkage
  distributed-lag + common slice + 简单日频 RV + LSMC 匹配实验
        |  lag0 主导，lag1-lag3 不足以交易；日频 RV 不稳定
        v
V0.8 非线性响应面 + causal HMM
  LightGBM/XGBoost quantile + OOF gap + filtered hidden state
        |  非线性 0/3 fold 胜出；状态增量 2/6；成本后为负
        v
研究暂停（2026-08-10）
  不再继续堆因子/模型；等待微观结构与更高频数据
```

## 3. 证据等级

不同版本的数字不能直接横向比较。项目采用以下证据等级解释历史结果：

| 等级 | 口径 | 能回答的问题 | 不能回答的问题 |
|---|---|---|---|
| E0 | 逻辑与工程 smoke test | 公式、时间边界、数据链能否运行 | 是否存在 Alpha |
| E1 | 单日或极小事件样本 | 信号形态是否符合直觉 | 稳定性、参数有效性 |
| E2 | 约 27 日 backfill/episode | 是否值得继续研究 | 跨市场状态稳定性、真实执行 |
| E3 | 180 日 chronological walk-forward | 样本外方向、fold 稳定性 | 无偏生产收益，若 universe 有偏 |
| E4 | 下一根可执行价格 + 成本/冲击 proxy | 理论收益能否覆盖磨损 | 真实盘口成交与人工滑点 |
| E5 | 真实 live first-seen + 人工模拟成交 | 人工能否兑现信号 | 当前项目尚未形成足量样本 |

早期 V0.3 的正收益主要属于 E2；V0.7.1/V0.8 的否定证据已经达到 E3-E4，因此在决策权重上更高。

## 4. 工程与研究两条线

### 4.1 数据工程线

这条线解决“能否持续、准确地得到本地数据”，不是 Alpha 本身：

- 盘后慢线：全市场日频行情、PIT 转股价、到期日、日频换手率及筛选快照。
- 盘前池管理：ACTIVE / OBSERVE，正式池低频调整，新标的先积累历史。
- 盘中快线：只采池内可转债和对应正股 5 分钟量价数据。
- 因子链：相同数据合同支持 incremental 与 backfill，记录 factor/alert/job 时间和数据模式。
- 历史研究底座：canonical CB/STOCK/index 5 分钟表、PIT 结构变量和 walk-forward 切片。

暂停的是新增策略和因子研究，不是否定这条工程线。若维护成本可控，继续被动积累原始数据仍有价值。

### 4.2 策略研究线

这条线始终围绕同一个问题迭代：正股已经变化后，转债是否存在尚未完成且可由人工捕获的响应。研究从线性残差逐步扩展到事件、期限选择、distributed-lag、非线性响应面和隐状态。最终发现，响应关系本身存在，但大部分发生在同一根 5 分钟 bar 内；剩余可执行边际不足以覆盖交易磨损。

## 5. 版本复盘

### 5.1 前置阶段：数据采集与 watchlist 架构

**原始假设**

全市场分钟数据可以直接支撑实时因子扫描。

**实际演化**

- AkShare/东方财富接口在全市场逐债轮询下速度慢且容易断连。
- 架构改为“全市场日频筛选 + 小池盘中 5 分钟监控”。
- CB 与 STOCK 放在同一 live 数据合同中按 `bond_code + stock_code + bar_start` 对齐。
- ACTIVE/OBSERVE 机制用于解决新入池标的冷启动。

**保留资产**

数据链、池状态、交易时段处理、质量标记和本地 TDengine 结构。它们为后续所有研究提供了基础。

### 5.2 V0.1：连续复合因子

**假设**

将残差均值回归、溢价均值回归、正股动量、资金流、相对成交额和低波动特征合成为 `signal_score`，可对转债下一阶段收益进行连续排序。

**实现**

- rolling/dynamic residual；
- premium residual；
- stock momentum；
- flow 与 relative amount；
- volatility response；
- pooled panel IC、cross-section IC 和 forward return。

**证据与结论**

早期 pooled panel IC 为负或接近 0，cross-section IC 接近 0。少量 `LONG_CANDIDATE` 在 30 分钟收益和 residual return 上看起来较好，但不足以支持连续排序。

**决策**

`signal_score` 保留为研究字段，退出正式排序 Alpha。研究范式转向稀疏触发事件。

### 5.3 V0.2：触发型人工监控

**假设**

连续 IC 可以接近 0，但条件尾部事件仍可能具有交易价值。

**实现**

- `setup_score`：是否值得盯盘；
- `trigger_score`：当前 bar 是否值得人工确认；
- `exit_state`：修复、衰减或退出提示；
- `LAG_REPAIR_LONG`：正股先动、转债滞后、成交不死、溢价不热；
- `MOMENTUM_BREAKOUT_LONG`：股债共振但尚未明显透支；
- `WATCH_LONG`：只进面板，不进默认交易回测；
- cooldown、每日频率限制、尾盘过滤和 tradeability gate。

**单日诊断（2026-06-30）**

- factor/alert rows：546；
- pooled panel IC：0.0053；
- cross-section IC：-0.0094；
- ACTION：1 条，`LAG_REPAIR_LONG`，14:15，债券 123124；
- 后 1 bar：-0.1317%；后 2 bar：-0.0834%；30/60 分钟标签当时不可用。

**决策**

事件结构符合设计，但 1 条 ACTION 既不能证明有效，也不能证明无效；禁止据此调参。

### 5.4 V0.3：ARMED、WATCH 细分与 episode evaluator

**假设**

ACTION 可能出现得太晚；提前的 ARMED 和更细的 WATCH 状态可以改善人工可执行性，同时 episode 评价能避免连续 bar 重复计数。

**实现**

- WATCH：`WATCH_SETUP`、`WATCH_LAG`、`WATCH_MOMENTUM`、`WATCH_RISKY`；
- ARMED：`ARMED_LAG_LONG`、`ARMED_BREAKOUT_LONG`；
- 新字段：stock impulse、短周期 CB momentum、residual slope/repair、VWAP gap、premium change、market regime；
- EXIT：residual repaired、momentum decay、premium overheat、time stop、noise spike、end of day；
- bar-level 与 episode-level 双 evaluator；
- theoretical / live-visible / manual 三层收益合同。

**27 日阶段性结果**

- factor rows：15,660；factor coverage：0.974；
- pooled IC：0.0085；cross-section IC：0.0192；
- bar-level ACTION：165；ARMED：523；
- ACTION 30m 平均约 +0.189%，60m 约 +0.146%；
- LAG_REPAIR 30m 约 +0.232%；
- BREAKOUT 30m 约 +0.098%，60m 转负。

**解释**

这是项目最积极的一组早期结果，但它仍属于短样本、理论/回填口径。后续事件去重、可执行入场和成本模型会减少样本并降低收益，不能把这里的数字当作可交易业绩。

**决策**

保留“触发器而非 ranking alpha”的定位；重点研究 LAG_REPAIR，降低 BREAKOUT 置信度；继续扩充历史和执行口径。

### 5.5 V0.4：ML Shadow Meta-Filter

**假设**

规则先产生 ACTION/ARMED，轻量 ML 只判断事件质量，可能过滤假突破和无效滞涨，而不替代规则引擎。

**数据集与任务**

- event/episode rows：464，覆盖 27 个交易日；
- ARMED：315；ACTION：149；
- `ARMED_BREAKOUT_LONG` 198、`ARMED_LAG_LONG` 117、`LAG_REPAIR_LONG` 99、`MOMENTUM_BREAKOUT_LONG` 50；
- 数据集字段 `label_action_good_30m` 可用样本 408，正例率 35.29%（该可用数覆盖事件数据集，不等同于 149 条正式 ACTION 数）；
- fake breakout 50 条，正例率 60%；
- lag repair success 92 条，正例率 38.04%；
- ARMED 到 ACTION：1 bar 6.67%，3 bar 14.92%。

**模型结果摘要**

- ACTION-good 的 logistic/calibrated AUC 约 0.41，GBDT AUC 约 0.50；
- ARMED -> ACTION 3bar logistic AUC 约 0.60，但 PR-AUC 约 0.24；
- fake-breakout 与 lag-repair 子任务样本更小，AUC 未显示可靠增量；
- 所有主要任务均有 small-sample warning。

**决策**

ML 只保留 shadow 研究身份。没有证据支持用其过滤生产 ACTION，也没有理由引入更复杂 DL/CUDA 模型。

### 5.6 V0.5：研究契约整合阶段

仓库中没有独立冻结的 V0.5 策略包或正式 V0.5 报告。这个编号主要代表 V0.4 到 V0.6 之间的研究契约成熟，而不是一套新的 Alpha：

- 从 bar alert 转为 event/episode 单位；
- 明确 `signal_time`、`first_seen_time`、`actual_decision_time`；
- 区分 live incremental、manual backfill、scheduled backfill；
- 增加 unreliable live day、执行价和滑点字段；
- WATCH/ARMED 始终不作为默认买入信号。

汇报时应直接说明“V0.5 是整合阶段，没有独立绩效结论”，避免人为补造版本成果。

### 5.7 V0.6：历史训练数据底座

**目标**

把单日和 27 日研究升级为可进行 walk-forward 的本地历史训练集。

**实现**

- canonical 可转债/正股/指数 5 分钟数据；
- PIT 转股价与转股价值；
- 日频换手率；
- 交易日、午休、停牌、缺 bar 和 source-vendor 数据合同；
- 约 200 个交易日目标数据，后续研究采用 180 日共同切片。

**关键限制**

- 研究池由较晚时点选出，历史样本带有 `FUTURE_SELECTED_TRAINING_SAMPLE_SOURCE` 偏差；
- 历史基本面/ST 无法完整 PIT 重建；
- 日频换手率已经补充，但仍没有 intraday turnover；
- 没有真实 bid/ask、盘口深度和逐笔成交。

**决策**

V0.6 是高价值研究基础设施，不是策略版本。它让早期假设接受更严格检验，也暴露了历史偏差和微观结构缺口。

### 5.8 V0.7：Hybrid Horizon、qIV/MC 与 LSMC

**假设**

若日内边际不足，可按 bond type、friction、qIV/MC 相对价值和 LSMC continuation value，在日内 T+0 与多日持有之间切换。

**实现**

- bond type；
- 三段式流动性与摩擦模型；
- qIV/MC proxy 定价；
- LSMC continuation value；
- intraday/daily/hybrid 期限选择；
- 成本敏感性和 ablation。

**覆盖与决策分布**

- qIV solve：4,408 / 4,910（89.78%）；
- 全本地面板 qIV/MC：31,159 / 32,539（95.76%）；
- `WATCH_ONLY` 4,371、`OVERNIGHT_DAILY` 518、`INTRADAY_T0` 21。

**10 bps 名义成本、手续费 5 倍压力结果**

- Intraday：-4.327%，21 笔；
- Daily LSMC：-9.722%，237 笔；
- Hybrid：-8.171%，221 笔。

**关键消融**

- 加 LSMC 后比无 LSMC 少亏，但主要来自减少交易/更保守，不代表 continuation value 是 Alpha；
- 去掉 qIV/MC 后结果改善，说明其当前为负增量；
- 去掉流动性冲击后亏损明显缩小，证明摩擦是一阶变量；
- 高动量权重显著更差，支持继续将动量降级为确认和风险字段。

**决策**

qIV/MC 降级为诊断，LSMC 不再负责入场，Hybrid 未获可交易证据。

### 5.9 V0.7.1：Diagnostic Reset 与股债联动

**假设**

真正值得检验的问题不是“如何让 Hybrid 变正”，而是正股冲击后，转债在 lag1-lag3 是否存在可捕获的延迟响应。

**研究合同**

- 180 个共同交易日、170 只债、4 个 45 日窗口；
- 同日、同池、同 PIT 转股价与日频换手率；
- qIV/MC 正式权重为 0；
- LSMC 只作为固定入场样本的退出 overlay；
- distributed-lag 剥离指数共同冲击；
- next-bar open、延迟 0/1/2 bar 和成本瀑布。

**事件结构**

- `CB_LEADS_STOCK_INFORMATION`：23,773；
- `CB_OVERSHOOT_HOT_MONEY`：16,627；
- `STOCK_LEADS_CB_NO_RESPONSE`：5,294；
- `STOCK_LEADS_CB_REPAIR`：仅 138。

**主要结果**

- 联动主要由 lag0 驱动，可执行 lag1-lag3 beta 很小；
- 正式 138 条事件零成本均值约 +0.0020%，现实成本后 -0.0999%；
- 延迟 2 根 bar 后毛收益转负；
- 动态退出净收益 -0.0999%，劣于固定波动率基准 -0.0068%；
- hot-money reject 能减少风险样本，但未创造成本后 Alpha；
- 简单日频 RV `top5/fixed5d` 全样本现实均值 +0.5698%，但窗口方向不稳、holdout 中位数为负，且受历史池偏差影响；
- LSMC matched exit 没有稳定优于固定 1/3/5 日退出；
- qIV/MC 3 日 bucket 平均 Spearman 约 -0.073，继续保持正式权重为 0；
- CB leads stock 的次日信息方向不稳定。

**决策**

五分钟频率下没有找到稳定、可人工捕获且覆盖成本的 stock-leads-CB Alpha。日频简单 RV 仅保留为低优先级 shadow 假设。

### 5.10 V0.8：非线性动态响应面与因果隐状态

**假设**

固定 beta 和手工阈值可能漏掉非线性交互；causal OOF response gap 与 filtered HMM state 可能区分 underreaction、catch-up、co-move、noise 和 breakdown。

**实现**

- 180 日、1,384,176 rows、170 只债、60 个特征；
- Ridge、V0.7.1 fair linear、LightGBM、XGBoost；
- q10/q50/q90 条件分位数；
- chronological OOF response gap；
- 5-state diagonal Gaussian HMM；
- causal forward filter，禁止 future smoothing；
- Model A（线性）、B（响应面）、C（响应面 + state）；
- next-valid-bar-open 与 0/10/21 bps 成本评价。

**响应面结果**

| 模型 | OOF RMSE | OOF Spearman |
|---|---:|---:|
| V0.7.1 fair linear | 0.002069 | 0.6737 |
| XGBoost mean | 0.002277 | 0.5937 |
| LightGBM q50 | 0.002353 | 0.5954 |
| Ridge | 0.002422 | 0.5654 |

- 非线性模型相对 fair linear：0 / 3 个正式 fold 胜出；
- q10-q90 名义 80% 区间实际覆盖 74.45%；
- B/C 的 rank correlation 略有上升，但 RMSE/MAE 更差，directional accuracy 约 50%。

**隐状态结果**

- 选中 5 state、diag covariance；最大 state occupancy 44.56%，没有直接 collapse；
- seed/fold 有效性不稳，最差正式 fold 仅 1 个有效 seed；
- canonical state 语义稳定性 60%，未通过门槛；
- state increment 在 30/60m 比较中仅 2 / 6 胜出。

**可执行与成本结果**

- gross 候选均值约 +0.20 bps；
- 10 bps 保守成本后约 -9.37 bps；
- 21 bps stress 后约 -20.55 bps。

**阶段门槛**

`C_NO_STABLE_INCREMENT`：当前数据和标签下，非线性响应面与隐状态均未证明稳定增量。继续增加模型复杂度不合理。

## 6. 关键证据总表

| 阶段 | 样本/口径 | 最积极观察 | 更高等级证据 | 最终判定 |
|---|---|---|---|---|
| V0.1 | 早期 bar panel | 少量候选残差收益较好 | IC 接近 0 | 连续排序退出主线 |
| V0.2 | 单日 546 rows | 触发结构可解释 | 仅 1 ACTION，短线为负 | 只能继续收样本 |
| V0.3 | 27 日、15,660 rows | ACTION 30m +0.189% | 短样本、理论口径 | 保留事件假设 |
| V0.4 | 464 episodes | ARMED 有少量转化信息 | 多任务 AUC/PR-AUC 弱 | ML 仅 shadow |
| V0.7 | 29 日混合回测 + 成本压力 | LSMC 比无 LSMC 少亏 | 所有主组合成本后为负 | Hybrid/qIV/LSMC 不成立 |
| V0.7.1 | 180 日 common slice | 简单日频 RV 有局部窗口 | 联动现实净收益 -0.0999%，窗口不稳 | 五分钟 lag Alpha 不成立 |
| V0.8 | 180 日、138 万 rows、OOF | 非线性略增 rank correlation | 0/3 fold 胜线性，成本后明显负 | 模型复杂化暂停 |

## 7. 研究假设决策账本

| 组件/假设 | 当前状态 | 依据 | 未来处理 |
|---|---|---|---|
| `signal_score` 连续排序 | 退役 | pooled/cross-section IC 接近 0 | 只作兼容研究字段 |
| WATCH 面板 | 保留概念 | 有助于人工解释，不等于收益 | 暂停实时策略化 |
| ARMED 提前预警 | 未证实 | 转化率低，人工增量未验证 | 有真实执行日志后再评估 |
| LAG_REPAIR | 暂停 | 早期小样本较好，180 日成本后不成立 | 更高频/L2 数据下重检 |
| MOMENTUM_BREAKOUT | 退居对照 | 60m 转负、fake breakout 高 | 不进入主策略 |
| 动量主 Alpha | 否定 | 高动量权重显著更差 | 仅用于确认/退出/背景 |
| volatility/noise gate | 风险字段 | 能描述 tradeability，但不创造收益 | 保留数据合同 |
| bond type 硬 gate | 未证实 | 分组有解释力，硬 gate 增量不足 | 仅描述/分层 |
| qIV/MC | 诊断专用 | 消融为负增量、rank 不单调 | 正式权重保持 0 |
| LSMC 入场 | 停用 | 未证明 continuation forecast | 不再决定新入场 |
| LSMC 退出 overlay | 未证实 | matched exits 未稳定胜简单退出 | 归档 |
| distributed-lag gap | 否定于 5min | lag0 主导，lag1-lag3 太弱 | 更细频率下可重启 |
| CB leads stock | 未证实 | 次日方向跨窗口不稳 | 不作为隔夜理由 |
| 简单日频 RV | 低优先级候选 | 全样本局部正，holdout/偏差不稳 | 仅在无偏 PIT 池重检 |
| 非线性响应面 | 未证实 | 0/3 folds 胜线性 | 暂停 |
| HMM 隐状态 | 未证实 | 状态语义/seed 稳定性和增量不足 | 暂停，不推进 V0.9 |

## 8. 看似矛盾的结果如何解释

### 8.1 V0.3 ACTION 为正，为什么后面判定无 Alpha

V0.3 使用较短日期、bar-level/回填和更理论化的收益口径；V0.7.1/V0.8 使用 episode 去重、next-bar open、180 日 walk-forward 和成本。后者更接近实际可执行问题，证据等级更高。二者不是同一数据合同下的直接复现。

### 8.2 LSMC 让 Hybrid 少亏，为什么不能称为有效

LSMC 同时减少了交易数并改变了入场样本。匹配实验显示 continuation value 没有稳定单调性，固定入场下也没有稳定优于简单退出。因此“少亏”更像保守过滤或降换手，而不是预测 Alpha。

### 8.3 日频 RV 全样本为正，为什么没有继续上线

收益跨 45 日窗口不稳定，holdout 中位数为负，同时 170 只债是 future-selected universe。全样本平均值不能覆盖时间稳定性和选择偏差。

### 8.4 HMM 没有 collapse，为什么仍判失败

状态占比合理只说明模型能分群，不说明状态能预测收益。其跨 seed/fold 语义稳定性不足，加入 filtered state 后仅 2/6 个主要比较改善，无法通过增量价值门槛。

### 8.5 线性模型相关性不低，为什么不能交易

V0.8 的 fair linear 对同 bar 转债响应拟合较好，说明股债联动存在；但同 bar 结果在 bar 收盘后已无法捕获。真正要交易的是 next-bar 以后剩余响应，而该边际接近 0 且被成本吞没。

## 9. 当前数据瓶颈

### 9.1 直接限制 Alpha 识别的缺口

1. **Intraday turnover**：只有日频换手率，无法判断单个 bar 的真实换手和筹码交换强度。
2. **真实 bid/ask spread**：当前成本只能用 high-low、流动性等 proxy，无法还原入场时可成交报价。
3. **盘口深度与订单簿**：无法区分真实追单、被动成交、撤单和脆弱流动性。
4. **逐笔成交/更高频时间戳**：五分钟内完成的 stock-to-CB 传导被压缩成 lag0，研究者看得到关系但交易者抓不到。
5. **CB/STOCK 严格同步**：供应商时间口径、bar close 与实际可见时点仍会影响 lead-lag 判断。
6. **真实执行日志**：缺少足量 `first_seen -> decision -> entry -> exit` 样本，无法估计人工反应损耗。

### 9.2 影响历史结论外推的缺口

- future-selected 170 债历史池；
- 历史基本面/ST/准入状态无法完整 PIT 重建；
- 当前数据区间不能覆盖足够多市场制度和极端流动性状态；
- 无真实 YTM、信用和条款事件数据，但这些不是当前五分钟联动主线的首要缺口。

## 10. 暂停决策

### 10.1 立即暂停

- 新增或优化日内因子；
- 新的阈值 grid search；
- ML/DL/HMM/LSMC 模型扩展；
- V0.9 latent-factor 或更复杂 regime 模型；
- 基于当前结果接入模拟下单或生产 alert；
- 将局部正样本包装成可上线结论。

### 10.2 继续保留

- 原始日频与 5 分钟数据的低成本积累；
- PIT 转股价、代码映射、交易日和质量标记；
- 现有工程健康检查；
- V0.1-V0.8 代码、配置、测试、报告和模型 manifest；
- evaluator、walk-forward、purging、成本瀑布和 next-bar execution 合同；
- 所有负结果，防止未来重复走同一条路。

### 10.3 当前不做生产声明

- 没有稳定毛 Alpha；
- 没有真实成本后 Alpha；
- 没有可证明的人工可执行 edge；
- 没有达到模拟仓 shadow 交易门槛；
- 没有任何模型被授权修改生产信号或自动下单。

## 11. 研究重启门槛

至少满足以下一项关键数据升级，并通过数据验收后，才值得重启：

1. 可验证的 1 分钟或更细 CB/STOCK 同步数据；
2. intraday turnover 或逐 bar 的可靠换手字段；
3. bid/ask、spread、盘口深度或逐笔成交；
4. Wind L2 或等价微观结构数据；
5. 无偏 point-in-time 历史 universe、ST 和准入快照；
6. 足量真实 live first-seen 与人工模拟成交日志。

建议的数据量门槛：

- 新数据至少覆盖 120-250 个交易日；
- 覆盖多个市场和流动性 regime；
- 预先冻结测试窗口和假设；
- 先验证 next-executable-price 的 gross edge，再谈复杂模型。

## 12. 重启顺序

未来不要从 V0.9 或更复杂模型直接开始。建议顺序：

1. 验收新数据的时间戳、同步、缺失和可执行报价。
2. 复跑最简单的 V0.7.1 distributed-lag baseline。
3. 将 bar 缩短到 1 分钟或 event-time，检查 lag0 是否分解为可执行 lag。
4. 先证明 gross edge 在 walk-forward 多数窗口为正。
5. 再验证 spread、冲击和人工延迟后仍为正。
6. 最后才考虑事件阈值、meta-filter 或隐状态。

建议的阶段门槛：

- 多数 fold 的 gross edge 同向；
- bootstrap 置信区间不再稳定覆盖 0；
- 延迟 1 个实际执行单位后仍有边际；
- realistic cost 后仍为正；
- 收益不集中于单日、单债或单一市场状态。

## 13. 汇报话术

### 13.1 30 秒版本

我们先搭建了全市场日频筛选和 watchlist 级五分钟实时数据链。早期连续因子 IC 接近零，因此转向稀疏触发事件；27 日小样本一度显示 LAG_REPAIR 有正的理论收益。随后我们把研究升级到 180 日 walk-forward、下一根可执行价格和成本模型，发现股债联动主要在同一根五分钟 bar 内完成，可执行的后续响应不足以覆盖磨损。qIV、LSMC、ML、非线性模型和 HMM 都未提供稳定增量，因此暂停继续堆模型，等待 intraday turnover、L2 和更高频同步数据。

### 13.2 五分钟版本的主线

1. **先解决数据**：将全市场慢筛选与小池快监控拆开，建立 CB/STOCK 本地对齐数据。
2. **推翻连续排序**：signal_score 的 pooled/cross-sectional IC 接近 0。
3. **转向事件**：WATCH/ARMED/ACTION 更符合人工半自动交易，但早期证据样本很小。
4. **扩大样本**：V0.6 把研究提升到 180 日 canonical 数据和 walk-forward。
5. **加入可执行性**：next-bar open 与成本后，早期正收益消失。
6. **拆解复杂模型**：qIV/MC 为负增量，LSMC 更像降换手过滤器，ML meta-filter 无稳定校准。
7. **直接研究 lead-lag**：lag0 强，lag1-lag3 弱，五分钟级人工套利空间不足。
8. **最后验证非线性和状态**：LightGBM/XGBoost 未胜线性，HMM 无稳定 forward 增量。
9. **研究决策**：暂停不是因为模型没做够，而是当前观测频率和微观结构数据无法识别可交易边际。

### 13.3 不应对外声称

- 不应把 V0.3 的 +0.189% 称为策略历史收益；
- 不应把 LSMC 的“少亏”称为 continuation Alpha；
- 不应把日频 RV 全样本正均值称为稳定超额；
- 不应把 HMM 的高 seed ARI 称为状态具有交易价值；
- 不应把同 bar 高相关性称为可捕获 lead-lag；
- 不应把 proxy 成本结果称为真实成交回测。

## 14. 核心产物索引

### 汇报总表

- [版本研究地图 CSV](outputs/research/research_map_versions.csv)

### 因子与事件

- [日内因子说明](README_INTRADAY_FACTORS.md)
- [日内因子配置](config/intraday_factors.yaml)
- [因子引擎](intraday_factors/engine.py)
- [研究 evaluator](research/evaluator.py)

### ML Shadow

- [V0.4 README](README_ML_SHADOW_V04.md)
- [V0.4 实现](research/ml_shadow_v04.py)
- [事件数据集](outputs/ml_shadow/cb_intraday_ml_event_dataset.csv)
- [实验记录](outputs/ml_shadow/cb_intraday_ml_experiments.jsonl)

### 历史训练数据

- [V0.6 README](README_TRAINING_DATA_V06.md)
- [V0.6 pipeline](training_data_v06/pipeline.py)
- [V0.6 配置](config/training_data_v06.yaml)

### Hybrid V0.7

- [V0.7 README](README_HYBRID_V07.md)
- [V0.7 配置](config/hybrid_strategy_v07.yaml)
- [V0.7 报告](outputs/research/hybrid_v07/hybrid_v07_report.md)
- [V0.7 消融](outputs/research/hybrid_v07/ablation_report.csv)

### Diagnostic Linkage V0.7.1

- [V0.7.1 README](README_DIAGNOSTIC_LINKAGE_V071.md)
- [V0.7.1 配置](config/diagnostic_linkage_v071.yaml)
- [V0.7.1 报告](outputs/research/diagnostic_linkage_v071/diagnostic_linkage_v071_report.md)
- [联动事件](outputs/research/diagnostic_linkage_v071/linkage_events.csv)
- [窗口稳定性](outputs/research/diagnostic_linkage_v071/window_stability.csv)

### Dynamic Response V0.8

- [V0.8 README](README_DYNAMIC_RESPONSE_V080.md)
- [V0.8 配置](config/dynamic_response_v080.yaml)
- [V0.8 报告](outputs/research/dynamic_response_v080/dynamic_response_v080_report.md)
- [模型比较](outputs/research/dynamic_response_v080/model_comparison.csv)
- [阶段门槛](outputs/research/dynamic_response_v080/stage_gate.json)
- [环境清单](outputs/research/dynamic_response_v080/environment_manifest.json)
- [OOF response surface](outputs/research/dynamic_response_v080/oof_surface_predictions.parquet)
- [OOF hidden state](outputs/research/dynamic_response_v080/oof_state_probabilities.parquet)

### 测试

- [日内因子测试](tests/test_intraday_factors.py)
- [ML shadow 测试](tests/test_ml_shadow_v04.py)
- [训练数据测试](tests/test_training_data_v06.py)
- [Hybrid 测试](tests/test_hybrid_v07.py)
- [Diagnostic Linkage 测试](tests/test_diagnostic_linkage_v071.py)
- [Dynamic Response 测试](tests/test_dynamic_response_v080.py)

## 15. 最终研究结论

项目已经完成了一条相对完整的研究证伪路径：从简单因子到事件，从小样本到 180 日，从理论价格到可执行价格，从规则到 ML/非线性/隐状态。每次增加复杂度都带着一个明确问题进入，而不是为了把回测调成正收益。

当前最稳健的结论是：**在现有五分钟数据、缺少 intraday turnover 和盘口微观结构的条件下，没有证据表明正股领先转债的剩余响应能稳定覆盖人工执行与交易成本；复杂模型也没有修复这一点。**

因此，研究于 2026-08-10 暂停。未来重启的前提是信息集发生变化，而不是继续在同一数据上寻找新的参数组合。
