# RQ1--RQ5 指标与证据采集说明

本文把研究所需信息按其产生时机分为三类：运行时必须保留的原始证据、可在运行结束后由原始证据或官方元数据补充的派生指标，以及必须由人工阅读轨迹后完成的标注。原则是：运行时保存事实和证据，不让 pipeline 自动推断失败机制；事后计算统计量；人工负责机制解释和失败边界。

## 一、运行时必须捕获的指标

这一部分的信息若没有在运行时保存，实验结束后通常无法可靠恢复。它是 RQ1--RQ5 的原始证据层。

### 1.1 每条运行的身份、配置与可复现信息

每个 `(mode, task_id, run_name)` 记录：

| 指标 | 用途 |
|---|---|
| `mode`、`mode_label`、`run_name`、`task_id`、`task_key` | 将一条运行准确关联到架构和任务 |
| benchmark、task seed、`fixed_task_seed`、任务组合数 | 保证任务实例可复现 |
| 请求模型名、provider 返回的真实模型快照、max tokens | 区分配置中的 alias 与服务端实际执行的模型版本 |
| action space、观察模态、设备/AVD、AndroidWorld 配置 | 说明执行环境和接口 |
| `configured_max_steps`、timeout、action retry 上限 | 说明运行预算和停止边界 |
| started/finished timestamp、运行目录、manifest 版本、有效配置 SHA-256、源码指纹/Git commit | 审计与复现；无 Git 元数据时以源码指纹作为代码版本 |

正式的四架构实验应统一：

```text
configured_max_steps = 30
```

并同时保存实际执行的 `step_count`。`max_steps` 是运行时配置事实，不能在运行结束后补成另一种实验设置。

### 1.2 每条运行的最终原始结果与成本

每个任务运行结束后，必须保存：

| 指标 | RQ 用途 |
|---|---|
| evaluator 原始 success/failure、原始评测输出 | RQ1 的原始终点；供后续人工复核 |
| `step_count` | RQ1、RQ3 temporal、RQ4、RQ5 |
| total runtime、started/finished time | RQ4、RQ5 |
| prompt / completion / total tokens | RQ4、RQ5 |
| 各 LLM stage 的 token（若有角色） | RQ4；解释多角色成本 |
| 原始 exception、timeout、设备/环境错误信息 | RQ2 人工判别和运行审计 |

这里的原始结果只描述系统事实，例如 success、timeout、exception 或已到达步数上限；不要求 pipeline 判断“为什么失败”。

### 1.3 每个 decision step 的原始轨迹证据

每一步都应能让后续人员回答：模型当时看到了什么、想做什么、实际执行了什么、执行后页面变成什么。

| 指标 | 最低要求 |
|---|---|
| `step_index` | 在同一 run 内连续、唯一 |
| action 前截图 | 保存当前决策页面 |
| action 后截图 | 保存执行后的页面；若架构没有显式 after 文件，至少要能与下一步 before 对齐 |
| action 前 XML | 保存当前页面结构；允许为空，但必须显式标记 |
| action 后 XML | 同上 |
| 原始 prompt、原始 response、每次 LLM call 的时间与 token | 支持 RQ2 的人工阅读和 RQ4/RQ5 的成本核算 |
| 每次调用的 `requested_model_name`、`resolved_model_name` 与缺失原因 | 保存 provider `response.model` 原始事实，不用请求 alias 冒充真实快照 |
| agent 原始 action 与参数 | 如点击坐标、文本、滑动方向、目标元素 |
| action 执行记录、返回值、异常 | 支持操作失败和重复无效动作的判读 |
| step duration | 支持运行时间和长尾分析 |
| 当前 package / 页面状态 / 原始错误 | 支持环境问题与状态变化审计 |

四个架构不需要采用相同的内部 action 格式，也不需要统一 prompt、角色或 planner/reflector 结构；必须保留各自原始日志。需要统一的是外层索引和文件定位方式，例如：

```text
task_reports/<task_id>/steps/step_001/
  step.json
  screenshots/
  xml/
  llm_calls/
```

这样人工可以按同一条路径找到证据，而不会抹掉架构自身的实现差异。

### 1.4 当前实验 pipeline 的关键补齐项

1. **P1 必须额外捕获 XML。** P1 仍保持 screenshot-only 的模型输入；XML 只作为离线证据保存，不进入 P1 prompt。新运行应同时保存每步 before/after screenshot 与 before/after XML。
2. **四架构统一为 30 steps。** P1/P2/P3/P4 的正式配置都使用相同 `configured_max_steps=30`。
3. **每步 token 与 duration 要可直接读取。** 即使原始 LLM call 已保存，也应在 step 汇总中写出该步聚合的 prompt/completion/total tokens 与 duration，避免 RQ4/RQ5 依赖不一致的日志解析。
4. **保留原始系统结束事件。** 可以记录 `agent_done`、`max_steps_reached`、`timeout`、`exception` 等原始事件，但不要把它们自动解释为 failure mechanism。
5. **记录候选 run 与选定 run 的关系。** 每个规范 run 应保留候选数、选择规则、选择原因和被放弃 run 的标识，避免 491 条原始记录到 464 条规范记录的选择过程不可审计。
6. **P2 报告以 checkpoint step ledger 为规范调用账本。** `appagent_log.jsonl` 只在 checkpoint 没有调用记录时回退使用；AppAgent 原始目录优先按 checkpoint 中的截图/XML 路径精确绑定，不能仅凭任务文本模糊匹配。
7. **每个 run 保存版本来源。** `run_manifest.json` 的 provenance 同时记录 defaults 文件哈希、脱敏后的有效配置哈希、源码 path+content 哈希和可选 Git 状态；工作区不属于 Git 仓库时必须显式记录原因，不能留空或伪造 commit。

## 二、事后补充的指标

这一部分不需要 agent 在运行时直接生成；只要第一部分的原始证据和官方任务元数据保存完整，就可以在运行结束后计算、连接或导出。

| 指标 | 来源 / 计算方式 | 主要 RQ |
|---|---|---|
| 官方任务难度、任务 tags、`optimal_steps` | AndroidWorld 官方任务定义或固定 metadata 表 | RQ1、RQ3、RQ4 |
| `success_corrected` | 原始 evaluator 结果与人工复核结果合并 | RQ1、RQ3、RQ4、RQ5 |
| 成功率、Wilson interval、难度分层成功率 | 464 条规范 run 的最终标签 | RQ1 |
| completion spectrum、unique completion、set difference | 四架构的任务级成功向量 | RQ1 |
| Cochran's Q、McNemar、Holm 校正 | 配对任务级成功向量 | RQ1 |
| 页面特征：structural load、pixel clutter、confusability 等 | XML、截图和 step 对齐后离线计算 | RQ3 |
| 页面风险集、trap 前/后步数、条件成功曲线 | step index、最终标签与人工 trap 边界合并 | RQ3、RQ5 |
| token / runtime / step 的中位数、CoF、amortized cost | run-level token、runtime、success 标签 | RQ4 |
| survival curve、wasted compute、阈值停止反事实 | step_count、token、最终标签和人工 failure step | RQ5 |
| artifact completeness、XML/截图质量、路径一致性 | 原始 task report 与 step 文件审计 | RQ3 |

事后补充的目的，是把原始事实变成可比较的统计量；它不能代替运行时缺失的截图、XML、prompt、response、action 或 token 记录。

## 三、人工分析标注的指标

这一部分必须由研究者阅读轨迹、页面、prompt、response 和执行证据后作出判断。pipeline 只保存原始证据，不应自动替代人工判断。

| 标注指标 | 说明 | 主要 RQ |
|---|---|---|
| evaluator 误判 / `ORACLE-ERR` | 实际完成但自动评测器给出失败时，人工复核并修正标签 | RQ1、RQ2、RQ3、RQ4、RQ5 |
| capability failure / `X-ENV` / `X-ART` | 区分 agent 能力问题、环境阻塞和工具/解析故障 | RQ2、RQ3、RQ5 |
| failure dimension 与 primary / secondary root cause | Intelligence、Command、Operations、Audit 及具体类别 | RQ2 |
| false finish、状态误读、规划错误、grounding miss、无效重复等 | 对失败机制的解释性归类 | RQ2、RQ4、RQ5 |
| `failure_step` / trap step | 人工判断首次明确进入失败或 trap 的步骤；必须带来源和解析状态 | RQ3、RQ5 |
| trap onset | RQ3 主页面模型中的 event boundary | RQ3 |
| C5-only、X-ENV、X-ART 的纳入/排除决定 | 依据预先写明的分类规则 | RQ3 |
| fast failure / flail | 在 trap 后步数确定后做行为分类 | RQ3、RQ4 |
| handoff target | 应验证、状态检查、路线修复、局部 UI 帮助或停止/重校准 | RQ5 |

人工标注应保留：标注者、标注依据、原始证据路径、是否存在第二类别、争议处理结果和最终 adjudication。`failure_step` 缺失时，应标记为缺失；不能为了页面 onset 分析而在 RQ3 中自动把它补成终末步。RQ5 若采用终末步回退来计算保守的反事实成本，也必须单独记录该回退来源，而不能把它误写为人工观察到的 failure step。
