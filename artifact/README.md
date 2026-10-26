# 最终数据与论文分析 / Final data and analysis

本目录可独立使用，无需 `pipline`。Python 3.11；GUI 模型 bootstrap 另需 C++17 编译器。标注只保留最终版本。

| 文件 | 内容 | 记录数 |
|---|---|---:|
| `data/execution_results.xlsx` | 最终终态、交互步数、资源汇总及任务信息 | 1,392 |
| `data/rq2_annotations.xlsx` | 执行键、最终主导原因、可选伴随原因 | 572 |
| `data/rq3_annotations.xlsx` | 轨迹键、原始步骤序号、最终 E/N/U 标签 | 17,658 |
| `data/gui_features/rq3_steps.csv` | 步骤键与四项 GUI 原始特征 | 17,658 |
| `data/resources/runtime_contributions.csv` | 同样本继续执行运行时间贡献 | 7,060 |
| `data/resources/selected_resource_contributions.csv` | 按 metric、模型区分的资源贡献 | 按有效记录 |

数据字段说明见 `data/README.md`。Excel 没有人工待审核状态、历史暂定标签或 A/B 标注流程字段。U 是最终不确定标签，不能转成 N。精选证据位于 `cases/`。

## 离线分析

从顶层目录运行：

```sh
python -m pip install -r artifact/requirements.lock.txt
python artifact/analysis/verify.py
python artifact/analysis/run.py basic
python artifact/analysis/rq3/run.py --bootstrap 1000
```

也可用 `python artifact/analysis/run.py all` 一次计算四个 RQ。文件写入系统临时目录，程序打印实际位置；`--output-dir PATH` 可以指定自己的位置。分析生成物不随输入包分发。

`analysis/reproduce.py` 读取三份 Excel、GUI 特征和资源贡献。RQ1/RQ2 计算完成率、任务覆盖、原因数量和共现；RQ3 从最终标签重建错误历史与序列指标，拟合 H0–H3 历史模型及八项按轨迹分层的条件 Logistic 模型；RQ4 计算固定支持任务的继续窗口、完成收益及资源成本。

The canonical inputs are the three Excel workbooks. Derived histories, predictor standardization and model eligibility are recomputed from final labels; they are not separately maintained in the data files. The numerical kernels are in `analysis/rq3/`.

## 模型口径

RQ3 严格分析包含 16,250 个目标步。U 或缺口打断精确邻接与连续错误段；E→N 表示局部错误中断。历史模型控制任务、基础模型、实现与步骤位置。GUI 模型区分 onset 与 persistence，GUI 特征在同系统完整队列的有效 E/N 位置标准化；每项模型使用 1,000 次任务 bootstrap，seed=20261022。运行时间区间使用 5,000 次任务 bootstrap，seed=20260924。关联不解释为因果效应。

作者核验的终态和步数用于主分析。历史换算不生成缺失的逐步映射；资源缺失不补零。RQ4 固定起点队列与资源完整子集的分母分别报告，provider 的 token 口径不用于跨模型效率排名。标注 κ 与工时沿用论文报告值，不从最终标签反算。

## 交互展示

```sh
python artifact/analysis/build_site.py
python -m http.server 8765 --directory artifact
```

打开 `http://localhost:8765/`。网站与分析读取同一份最终 Excel，提供任务矩阵、失败数量/百分比切换、错误时间轴、回归关联、继续执行收益成本及三份 Excel 下载。原始模型文字与截图保留原语言，外围界面中英双语。

The website builder uses computed analyses in the temporary workspace and creates `website/data.js`. If the computations are absent, it runs the offline analyses first. It does not call a model API.

## 论文主图展示 / Paper figure presentation

`website/index.html` 展示当前定稿的七张主图与四个 RQ 的核心发现；`website/explorer.html` 提供补充交互数据与轨迹浏览。主图的原始 PDF 和高分辨率 PNG 均位于 `website/figures/`。

The public homepage presents seven current-manuscript figures with bilingual findings. The supplementary explorer provides filters and selected trajectories. Original PDFs and rendered PNGs are bundled under `website/figures/`.

GitHub Pages deploys this directory as its static root. After editing a canonical workbook, rerun the analyses and `analysis/build_site.py`, verify the new data and findings, then commit the updated files.
