# GUI Trajectory Analysis

《How Autonomous Can GUI Agents Be? Understanding the Limitations of LLM-Based GUI Agents》的实验代码与论文分析材料。

Research code and final analysis data for the GUI agent autonomy study.

```text
gui-trajectory-analysis/
├── pipline/                 # 四个 Agent 与独立实验运行程序
│   ├── agents/              # P1–P4 接口与实现适配
│   ├── runner/              # AndroidWorld 运行、记录与汇总
│   ├── vendor/              # 必需的上游源码与许可证
│   ├── configs/             # 三个模型预设与 116 个任务
│   ├── run.py
│   └── README.md
└── artifact/                # 最终 Excel、分析代码及交互展示
    ├── data/
    │   ├── execution_results.xlsx
    │   ├── rq2_annotations.xlsx
    │   ├── rq3_annotations.xlsx
    │   ├── gui_features/
    │   └── resources/
    ├── analysis/
    │   └── rq3/             # Logistic 数值内核与回归入口
    ├── cases/
    ├── website/
    └── README.md
```

两部分独立使用、分别安装依赖。`pipline` 不读取 `artifact`；离线分析不导入 Agent 代码，也不调用模型 API。

The two components have separate dependencies. Experiments do not read the analysis dataset; offline analysis does not import agent code or call model APIs.

## 复现论文分析 / Reproduce the analyses

Python 3.11；完整 RQ3 重拟合需要 C++17 编译器。

```sh
python -m pip install -r artifact/requirements.lock.txt
python artifact/analysis/verify.py
python artifact/analysis/run.py all
python artifact/analysis/build_site.py
python -m http.server 8765 --directory artifact
```

打开 `http://localhost:8765/`。程序会打印生成文件所在的系统临时目录；可用 `--output-dir PATH` 自选位置。构建网站时可用 `--analysis-dir PATH` 读取该次计算。

Open `http://localhost:8765/`. Generated analyses are written to the system temporary directory and its path is printed. Use `--output-dir PATH` to choose another location; the website builder accepts `--analysis-dir PATH`.

## 运行 Agent / Run the agents

安装和设备准备见 `pipline/README.md`。仅检查配置可以运行：

```sh
python -m pip install -r pipline/requirements-cli.txt
python pipline/run.py --model gemini --mode p2 --dry-run
```

`--dry-run` 不调用模型或设备。实际实验另需完整运行依赖、Android SDK/ADB、已初始化的 AndroidWorld 模拟器与环境变量中的 API 密钥。

Dry runs do not call a model or device. Actual experiments additionally require the runtime dependencies, Android SDK/ADB, initialized AndroidWorld devices and environment-held API credentials.

## 数据口径 / Data scope

- 12 个系统、116 个任务，共 1,392 次执行，770 次成功。
- RQ2：572 条诊断失败，最终主导原因及可选伴随原因。
- RQ3：1,304 条轨迹、17,658 步；E=4,194、N=13,380、U=84。
- Excel 为最终标注与执行汇总的唯一入口；不同时维护另一套标注 CSV。
- GUI 特征与逐步资源贡献用于离线重算；原始截图和运行记录仅提供 16 条精选轨迹。

The Excel workbooks are the canonical final inputs. Final outcomes and interaction lengths are retained; missing step mappings and measurements are not imputed. U is a final uncertainty category. Agreement statistics and annotation effort remain manuscript-reported. The experiment code is a portable source snapshot; no new experiments were run during packaging. Upstream licenses are retained within `pipline/vendor/`.

## 在线展示 / Online presentation

- [论文主图与核心发现 / Paper figures and findings](https://likelikec.github.io/gui-trajectory-analysis/website/)
- [交互数据与案例 / Interactive data and cases](https://likelikec.github.io/gui-trajectory-analysis/website/explorer.html)
- [GitHub 发布包 / Versioned downloads](https://github.com/likelikec/gui-trajectory-analysis/releases/tag/v0.2.0)
- Google Sites: https://sites.google.com/view/gui-agent

主图来自当前定稿 PDF。网站解释图表单位和分母；RQ2 原稿图注中的 count/percentage 不一致在对应图下明确说明。图形展示不更改最终数据。

The main figures are copied from the current manuscript PDFs. Units and denominators are explained on the page, including the RQ2 heatmap caption discrepancy. Figure presentation does not change final data.
