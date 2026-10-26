# 数据字段 / Data fields

`execution_results.xlsx`、`rq2_annotations.xlsx`、`rq3_annotations.xlsx` 是最终汇总与标注的唯一输入。修改最终 Excel 后重新运行分析及网站构建即可同步展示。

## 执行汇总

| 字段 | 含义 |
|---|---|
| `model`, `mode`, `task_id` | 模型族、P1–P4 实现、任务实例；共同构成唯一执行键 |
| `task_difficulty` | easy / medium / hard |
| `optimal_steps` | AndroidWorld 参考步数 Lₜ |
| `step_count` | 作者核验的最终交互步数 |
| `success_corrected` | 作者核验的最终终态，布尔值 |
| `runtime_seconds` | 汇总运行时间，单位秒 |
| `total_tokens`, `prompt_tokens`, `completion_tokens` | 保留记录的 provider token 计数 |
| `task_description`, `task_tags` | 任务要求与任务类型 |

每个系统 116 个任务，其中 easy=61、medium=36、hard=19；共 770 次成功、622 次未成功。交互步不是 GUI 动作数量。

## RQ2 最终标注

| 字段 | 含义 |
|---|---|
| `execution_id` | `model/Pn/task_id` |
| `dominant_code` | 每条诊断失败的最终主导原因 |
| `cooccurring_code` | 最终伴随原因；为空表示未记录额外原因 |

主分析为 572 条诊断失败；其余 50 条未成功执行未进入这一诊断子集。族别从主导码首字母派生：R=Representation、D=Decision、E=Execution、M=Monitoring/recovery。R1–R3、D1–D7、E1–E3、M1–M2 共 15 个代码；网站提供名称。每条最多一个伴随原因，共现按无序原因对计算。百分比的分母是所选系统的诊断失败数，不能替换为 116 或该系统全部失败数。

## RQ3 最终标注与 GUI 特征

| 字段 | 含义 |
|---|---|
| `trajectory_id` | 与执行键相同 |
| `source_step_index` | 从 1 开始的原始交互步骤位置 |
| `label` | E=error，N=non-error，U=uncertain |

1,304 条轨迹，E=4,194、N=13,380、U=84。标签键与 `gui_features/rq3_steps.csv` 一一对应。特征表仅包含两列键与四项原始测量：`visual_stagnation`（相邻交互前屏幕的 SSIM）、`median_candidate_area`（可操作元素相对面积中位数）、`spatial_crowding`（归一化空间拥挤度）、`edge_density_clutter`（Canny 边缘像素比例）。空值表示未获得对应测量，不填成零。

## 逐步资源贡献

`resources/` 中两张表按执行键与 `continuation_step` 连接。其 `task_id` 为无 `_0` 后缀的任务模板名；执行键仍保留任务实例名。

`initial_denominator_contribution` 表示固定起点队列成员；`new_completion_contribution` 表示是否恰在该新增步完成。`observed_step` 与 `structural_zero_after_termination` 区分实测资源与已知终止后的结构性零。运行时间单位为秒；其他资源由 `metric` 和 `resource_unit` 指定。有效 N 从相应贡献表计算，不以全部执行数代替。

`cases/` 为精选轨迹，图像与步骤记录采用包内相对路径。分析支持最终标签/保存特征到统计结果的重算，不声称包含全部原始截图/XML的重新提取。
