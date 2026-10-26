# 四个 Agent 与实验运行 / Agents and experiment runner

本目录可单独复制使用，Python 3.11。所有项目源码引用均位于本目录内。

| 实现 | 论文名称 | 本地代码 |
|---|---|---|
| P1 | Visual ReAct | `agents/guiagent/p1_reactive/`、`vendor/mobile-use/mobile_use/agents/agent_qwen.py` |
| P2 | XML-Assisted ReAct | `agents/guiagent/p2_appagent/`、`vendor/AppAgent/scripts/` |
| P3 | Planner–Executor–Reflector | `agents/guiagent/p3_mobileagentv3/`、`vendor/MobileAgent/Mobile-Agent-v3/android_world_v3/android_world/agents/` |
| P4 | Hierarchical Reflection | `agents/guiagent/p4_mobileuse/`、`vendor/mobile-use/mobile_use/agents/multi_agent.py` |

`runner/study/` 保存实际运行、观测与动作适配、逐步指标记录以及汇总程序。P1、P4 共用 MobileUse 基础组件；各模式的控制流程由配置工厂分别构造。使用的是 Mobile-Agent-v3 随附的 AndroidWorld fork。

## 安装与配置

```sh
python -m venv .venv-pipline
source .venv-pipline/bin/activate
python -m pip install -r pipline/requirements.txt
python pipline/run.py --help
```

Windows 使用对应虚拟环境激活命令。仅检查配置时，安装 `pipline/requirements-cli.txt` 即可。

模型预设位于 `configs/qwen.yaml`、`gemini.yaml`、`gpt.yaml`；全部预设为 30 个 interaction steps，`configs/tasks.json` 列出论文的 116 个任务。预设保留三种模型的协议及坐标设置。服务商、endpoint 或模型可用性发生变化时，使用 `--defaults` 指定自己的配置；不要把新的运行当作已发布执行的替代。

The presets cover Qwen3-VL-235B-A22B-Instruct, Gemini-3-Flash-Preview and GPT-5.6-Luna. They are portable rerun settings, not proof of an exact historical checkout. Provider availability is external to this package.

```sh
python pipline/run.py --model qwen --mode p1 --dry-run
python pipline/run.py --model gemini --mode p2 --dry-run
python pipline/run.py --model gpt --mode p3 --dry-run
python pipline/run.py --model gemini --mode p4 --dry-run
```

`--dry-run` 只生成配置预览，不需要 API 密钥，不操作模拟器。

## 实际运行

1. 按 `vendor/MobileAgent/Mobile-Agent-v3/android_world_v3/README.md` 中的 SDK／AVD 部分准备 Android SDK、ADB、模拟器、应用资源及 AndroidWorld 初始状态。运行本研究请使用本页安装与启动命令，上游 README 的通用启动命令不作为本包入口。应用安装包和模拟器镜像不随源码分发。
2. 设置 `ANDROID_SDK_ROOT`，将 SDK 的 `platform-tools`、`emulator` 加入 PATH。启动所需设备，使用 `adb devices` 确认连接。
3. 将密钥放入 `VLM_API_KEY` 或对应 provider 的环境变量；配置文件中的密钥保持为空。
4. 指定设备、输出目录并运行。例如：

```sh
python pipline/run.py --model gemini --mode p2 \
  --serial-no emulator-5554 --console-port 5554 --grpc-port 8554 \
  --tasks AudioRecorderRecordAudio \
  --output-dir ./pipline/outputs --run-name gemini_p2_example
```

不传 `--tasks` 时使用全部 116 个任务。多设备并行应为每个运行指定独立设备、端口和输出目录。详细参数通过 `python pipline/run.py --model gemini --mode p2 --help` 及 `runner/study/run_mode.py` 查看。

输出包含逐步动作、可用截图/XML、模型响应、运行时间和 token 记录。`runner/study/reporting.py` 提供运行结果汇总；记录指标定义见 `METRICS.md`。不要使用新运行覆盖 `artifact` 中人工核验的最终 Excel。

运行完成后，可用保留的汇总程序导出本次记录，不调用额外翻译模型：

```sh
python pipline/summarize.py --output-root ./pipline/outputs \
  --output ./pipline/outputs/rerun_summary.xlsx
```

默认包含所选输出根目录下全部已记录运行；`--run-names` 可以指定逗号分隔的运行名。此导出不自动挑选补跑结果或应用人工修正，也不覆盖论文最终数据。

This packaging was checked with configuration previews and imports; it did not execute Android tasks or make model API calls. Upstream code and licenses are bundled in `vendor/`.
