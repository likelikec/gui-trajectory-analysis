"""Chinese localization helpers for study outputs."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from openai import OpenAI

from study.bootstrap import PROJECT_ROOT
from study.config import fill_model_credentials, load_defaults


EXCEL_HEADERS_ZH = {
    "record_id": "记录编号",
    "benchmark": "基准",
    "mode": "模式",
    "mode_label": "模式说明",
    "run_name": "运行名称",
    "task_id": "任务编号",
    "task": "任务",
    "execution_steps": "执行步骤",
    "final_result": "最终结果",
    "success": "是否成功",
    "step_count": "步数",
    "prompt_tokens": "输入令牌",
    "completion_tokens": "输出令牌",
    "total_tokens": "总令牌",
    "runtime_seconds": "运行时长(秒)",
    "avg_step_seconds": "平均步时(秒)",
    "started_at": "开始时间",
    "finished_at": "结束时间",
    "task_report_dir": "任务报告目录",
    "run_output_dir": "运行目录",
    "notes": "备注",
}


EXACT_TRANSLATIONS = {
    "Reactive Single-Agent": "响应式单代理",
    "Explore-then-Deploy": "先探索后执行",
    "Planner-Decision-Reflection": "规划-决策-反思",
    "Hierarchical Reflection": "分层反思",
    "androidworld": "安卓世界",
    "androidlab": "安卓实验室",
    "success": "成功",
    "failed": "失败",
    "incomplete": "未完成",
    "agent_finished_but_metric_failed": "代理已结束但未通过评测",
    "terminal_log_missing": "终端日志缺失",
    "token_usage_missing": "令牌统计缺失",
    "runtime_agent_log_missing": "运行时代理日志缺失",
    "llm_log_missing": "大模型交互日志缺失",
    "screenshots_partial_or_missing": "截图不完整或缺失",
    "screenshots_missing": "截图缺失",
    "True": "是",
    "False": "否",
    "Swipe": "滑动",
    "Tap": "点击",
    "Type": "输入",
    "Open": "打开",
    "Launch": "启动",
    "Wait": "等待",
    "Back": "返回",
    "Home": "主页",
    "Finish": "完成",
    "FINISH": "完成",
    "Done": "完成",
    "Use Clock to set an alarm for 3PM with the label \"meeting\".": "使用时钟设置一个下午3点的闹钟，标签为“会议”。",
    "Create a timer with 0 hours, 16 minutes, and 35 seconds. Do not start the timer.": "创建一个时长为0小时16分钟35秒的计时器，但不要启动它。",
    "Add John as a contact and set his mobile phone number to be 12345678": "添加约翰为联系人，并将其手机号码设置为12345678。",
    "Add John as a contact and set his mobile phone number to be 12345678.": "添加约翰为联系人，并将其手机号码设置为12345678。",
    "Add John as a contacts and set his mobile phone number to be 12345678": "添加约翰为联系人，并将其手机号码设置为12345678。",
    "Add John as a contacts and set his mobile phone number to be 12345678.": "添加约翰为联系人，并将其手机号码设置为12345678。",
    "Add Hugo Pereira as a contact and set his mobile phone number to be 12345678": "添加雨果·佩雷拉为联系人，并将其手机号码设置为12345678。",
    "Add Hugo Pereira as a contact and set his mobile phone number to be 12345678.": "添加雨果·佩雷拉为联系人，并将其手机号码设置为12345678。",
}


COMMON_TERM_REPLACEMENTS = {
    r"(?<![A-Za-z])Clock(?![A-Za-z])": "时钟",
    r"(?<![A-Za-z])Alarm(?![A-Za-z])": "闹钟",
    r"(?<![A-Za-z])Timer(?![A-Za-z])": "计时器",
    r"(?<![A-Za-z])Stopwatch(?![A-Za-z])": "秒表",
    r"(?<![A-Za-z])Contacts(?![A-Za-z])": "联系人",
    r"(?<![A-Za-z])Contact(?![A-Za-z])": "联系人",
    r"(?<![A-Za-z])Phone(?![A-Za-z])": "电话",
    r"(?<![A-Za-z])Google(?![A-Za-z])": "谷歌",
    r"(?<![A-Za-z])YouTube(?![A-Za-z])": "优兔",
    r"(?<![A-Za-z])meeting(?![A-Za-z])": "会议",
    r"(?<![A-Za-z])John(?![A-Za-z])": "约翰",
    r"(?<![A-Za-z])Hugo(?![A-Za-z])": "雨果",
    r"(?<![A-Za-z])Pereira(?![A-Za-z])": "佩雷拉",
    r"(?<![A-Za-z])Hugo\s+Pereira(?![A-Za-z])": "雨果·佩雷拉",
    r"(?<![A-Za-z])First\s+name(?![A-Za-z])": "名字",
    r"(?<![A-Za-z])Last\s+name(?![A-Za-z])": "姓氏",
    r"(?<![A-Za-z])mobile\s+phone\s+number(?![A-Za-z])": "手机号码",
    r"(?<![A-Za-z])phone\s+number(?![A-Za-z])": "电话号码",
    r"(?<![A-Za-z])Save(?![A-Za-z])": "保存",
    r"(?<![A-Za-z])Add\s+label(?![A-Za-z])": "添加标签",
    r"(?<![A-Za-z])OK(?![A-Za-z])": "确定",
    r"(?<![A-Za-z])open(?![A-Za-z])": "打开",
    r"(?<![A-Za-z])FINISH(?![A-Za-z])": "完成",
    r"(?<![A-Za-z])grid(?![A-Za-z])": "网格",
}


TRANSLATABLE_KEYS = {
    "mode_label",
    "task",
    "execution_steps",
    "final_result",
    "notes",
    "prompt",
    "response",
    "thought",
    "action_desc",
    "summary",
    "plan",
    "sub_goal",
    "progress",
    "reflection_error",
    "trajectory_reflection_error",
    "evaluation_result",
    "evaluation_reason",
    "memory_response",
    "insight",
    "reflect_response",
    "planning_response",
    "error",
    "message",
    "goal",
    "target",
    "content",
    "task_instruction",
    "last_act",
    "reason",
}


PRESERVE_VALUE_KEYS = {
    "record_id",
    "mode",
    "run_name",
    "task_id",
    "task_report_dir",
    "run_output_dir",
    "checkpoint_path",
    "agent_log_dir",
    "task_dir",
    "trace_id",
    "xml",
    "ac_xml",
    "image",
    "screenshot",
    "before_screenshot",
    "after_screenshot",
    "before_xml",
    "after_xml",
    "path",
    "base_url",
    "api_url",
    "api_key",
    "model_name",
    "serial_no",
    "host",
    "package",
    "package_name",
    "current_activity",
    "metric_type",
    "stage",
    "name",
    "operation",
    "action",
    "benchmark",
    "started_at",
    "finished_at",
    "terminal_log",
    "explored_knowledge_path",
}


def _contains_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _looks_like_path_or_identifier(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in EXACT_TRANSLATIONS:
        return False
    if stripped.startswith(("http://", "https://", "file://")):
        return True
    if re.match(r"^[A-Za-z]:\\", stripped):
        return True
    if ("/" in stripped or "\\" in stripped) and "\n" not in stripped:
        return True
    if re.match(r"^[A-Za-z0-9_.|:-]+$", stripped) and " " not in stripped:
        return True
    if re.match(r"^(com|org|io|net)\.[A-Za-z0-9_.-]+$", stripped):
        return True
    return False


def _replace_common_terms(text: str) -> str:
    result = text
    for pattern, replacement in COMMON_TERM_REPLACEMENTS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _chunk_text(text: str, max_chars: int = 2200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for piece in re.split(r"(\n\n+|\n)", text):
        if not piece:
            continue
        if len(current) + len(piece) <= max_chars:
            current += piece
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(piece) <= max_chars:
            current = piece
            continue
        start = 0
        while start < len(piece):
            chunks.append(piece[start : start + max_chars])
            start += max_chars
    if current:
        chunks.append(current)
    return chunks


def _heuristic_translate(text: str) -> str:
    stripped = text.strip()
    if stripped in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[stripped]

    translated = _replace_common_terms(text)
    translated = re.sub(
        r"(?is)^Create a new\s+联系人\s+for\s+(.+?)\.\s*Their number is\s+([+\d]+)\.?$",
        r"创建一个新联系人：\1，电话号码为\2。",
        translated,
    )
    translated = re.sub(
        r"(?is)^Add\s+(.+?)\s+as a\s+联系人\s+and set\s+(?:his|her|their)\s+mobile\s+电话\s+number to be\s+([+\d]+)\.?$",
        r"添加\1为联系人，并将其手机号码设置为\2。",
        translated,
    )
    translated = re.sub(
        r"(?is)^Add\s+(.+?)\s+as a\s+contacts?\s+and set\s+(?:his|her|their)\s+mobile\s+phone\s+number to be\s+([+\d]+)\.?$",
        r"添加\1为联系人，并将其手机号码设置为\2。",
        translated,
    )
    translated = re.sub(r"(?im)^\s*(\d+)\.\s*Swipe\s*$", r"\1. 滑动", translated)
    translated = re.sub(r"(?im)^\s*(\d+)\.\s*Tap\s*$", r"\1. 点击", translated)
    translated = re.sub(r"(?im)^\s*(\d+)\.\s*Type\s*$", r"\1. 输入", translated)
    translated = re.sub(r"(?im)^\s*(\d+)\.\s*Open\s*$", r"\1. 打开", translated)
    translated = re.sub(r"(?im)^\s*(\d+)\.\s*(Finish|FINISH|Done)\s*$", r"\1. 完成", translated)
    translated = re.sub(r"(?i)\bAdd\b", "添加", translated)
    translated = re.sub(r"(?i)\bas a contacts?\b", "为联系人", translated)
    translated = re.sub(r"(?i)\bset his mobile phone number to be\b", "并将其手机号码设置为", translated)
    translated = re.sub(r"(?i)\bset her mobile phone number to be\b", "并将其手机号码设置为", translated)
    translated = re.sub(r"(?i)\bset the mobile phone number to be\b", "并将手机号码设置为", translated)
    return translated


class ChineseTranslator:
    def __init__(self, cache_path: str | Path | None = None):
        cfg = fill_model_credentials(
            load_defaults(
                PROJECT_ROOT / "configs" / "qwen.yaml"
            ),
            None,
            None,
            None,
        )
        self.client = None
        if cfg.model.api_key and cfg.model.base_url:
            self.client = OpenAI(api_key=cfg.model.api_key, base_url=cfg.model.base_url)
        self.model_name = cfg.model.model_name
        self.cache_path = Path(cache_path or (PROJECT_ROOT / ".cache" / "study_translation_cache.json"))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError, UnicodeDecodeError):
            broken_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.broken")
            try:
                if broken_path.exists():
                    broken_path.unlink()
                self.cache_path.replace(broken_path)
            except OSError:
                pass
            return {}
        return data if isinstance(data, dict) else {}

    def _save_cache(self) -> None:
        tmp_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.cache_path)

    def _translate_chunk(self, text: str) -> str:
        if text in EXACT_TRANSLATIONS:
            return EXACT_TRANSLATIONS[text]
        if not self.client:
            return _heuristic_translate(text)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=4096,
                timeout=90,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个专业翻译引擎。把用户给出的英文翻译成自然、准确的简体中文。"
                            "普通自然语言中不要保留英文字母。"
                            "仅保留路径、URL、文件名、设备序列号、变量名、JSON键名、动作枚举等机器标识。"
                            "只返回JSON对象，格式为{\"translation\":\"...\"}。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            )
            content = response.choices[0].message.content or "{}"
            try:
                parsed = json.loads(content)
                candidate = str(parsed.get("translation", text))
            except JSONDecodeError:
                candidate = content
            return _replace_common_terms(candidate)
        except Exception:
            return _heuristic_translate(text)

    def translate_text(self, text: str) -> str:
        if text in EXACT_TRANSLATIONS:
            return EXACT_TRANSLATIONS[text]
        if not _contains_english(text):
            return text
        if _looks_like_path_or_identifier(text):
            return text
        cached = self.cache.get(text)
        if isinstance(cached, str) and cached and not _contains_english(cached):
            return cached

        translated = "".join(
            self._translate_chunk(chunk) if _contains_english(chunk) else chunk
            for chunk in _chunk_text(text)
        )
        translated = _replace_common_terms(translated)
        self.cache[text] = translated
        try:
            self._save_cache()
        except OSError:
            pass
        return translated


def localize_string(value: str, translator: ChineseTranslator, *, key: str | None = None) -> str:
    if key in PRESERVE_VALUE_KEYS:
        return value
    if value in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[value]
    value = _replace_common_terms(value)
    if key not in TRANSLATABLE_KEYS and _looks_like_path_or_identifier(value):
        return value
    if not _contains_english(value):
        return value
    return translator.translate_text(value)


def localize_data(value: Any, translator: ChineseTranslator, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: localize_data(v, translator, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [localize_data(item, translator, key=key) for item in value]
    if isinstance(value, str):
        return localize_string(value, translator, key=key)
    return value


def localize_record(record: dict[str, Any], translator: ChineseTranslator) -> dict[str, Any]:
    localized = dict(record)
    localized["benchmark"] = EXACT_TRANSLATIONS.get(str(record.get("benchmark", "")), str(record.get("benchmark", "")))
    localized["mode_label"] = localize_string(str(record.get("mode_label", "")), translator, key="mode_label")
    localized["task"] = localize_string(str(record.get("task", "")), translator, key="task")
    localized["execution_steps"] = localize_string(str(record.get("execution_steps", "")), translator, key="execution_steps")
    localized["final_result"] = localize_string(str(record.get("final_result", "")), translator, key="final_result")
    localized["notes"] = "，".join(
        localize_string(part.strip(), translator, key="notes")
        for part in str(record.get("notes", "")).split(",")
        if part.strip()
    )
    localized["success"] = "是" if bool(record.get("success")) else "否"
    return localized


def localize_json_file(path: Path, translator: ChineseTranslator) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError, UnicodeDecodeError):
            return
        localized = localize_data(data, translator)
        path.write_text(json.dumps(localized, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(localize_data(json.loads(stripped), translator))
                except JSONDecodeError:
                    continue
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if suffix in {".log", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        path.write_text(localize_string(text, translator), encoding="utf-8")


def localize_run_output_files(run_output_dir: str | Path, translator: ChineseTranslator | None = None) -> None:
    translator = translator or ChineseTranslator()
    run_output_dir = Path(run_output_dir).resolve()
    for path in sorted(run_output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".log", ".txt"}:
            continue
        localize_json_file(path, translator)


def localize_outputs_tree(outputs_root: str | Path, translator: ChineseTranslator | None = None) -> None:
    translator = translator or ChineseTranslator()
    outputs_root = Path(outputs_root).resolve()
    for path in sorted(outputs_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".log", ".txt"}:
            continue
        localize_json_file(path, translator)
