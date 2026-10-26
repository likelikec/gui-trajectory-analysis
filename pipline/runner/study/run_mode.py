"""Unified CLI: run one mode on one benchmark with one command."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.bootstrap import PROJECT_ROOT
from study.config import ensure_model_ready, fill_model_credentials, get_mode_model_config, load_defaults
from study.knowledge import resolve_default_knowledge_path
from study.provenance import build_run_provenance, collect_model_snapshots
from guiagent.mode_factory import MODE_LABELS, build_mode_preview


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or self.encoding or "utf-8"
                sanitized = data.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(sanitized)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def close(self):
        self.flush()


def _env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value:
            return int(value)
    return default


def _redact_secrets(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key.lower() == "api_key" and item:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _requested_models_for_mode(cfg, mode: str) -> dict[str, dict[str, object]]:
    roles_by_mode = {
        "p1": ("p1",),
        "p2": ("p2",),
        "p3": ("p3_decision", "p3_planning", "p3_caption"),
        "p4": ("p4",),
    }
    requested: dict[str, dict[str, object]] = {}
    for role in roles_by_mode[mode]:
        model = get_mode_model_config(cfg, role)
        requested[role] = {
            "model_name": model.model_name,
            "provider": model.provider,
            "adapter_name": model.adapter_name,
        }
    return requested


def _sync_final_manifest_to_task_reports(run_output_dir: Path, manifest: dict) -> None:
    """Keep each task's raw manifest identical to the finalized run manifest."""
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    for task_manifest in run_output_dir.glob("task_reports/*/raw/run_manifest.json"):
        task_manifest.write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one study mode on one benchmark.")
    parser.add_argument("--mode", choices=sorted(MODE_LABELS.keys()), required=True)
    parser.add_argument("--benchmark", choices=["androidworld"], required=True)
    parser.add_argument("--defaults", default=str(PROJECT_ROOT / "configs" / "qwen.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--provider",
        choices=["openrouter", "poixe", "silra"],
        default=None,
        help="Credential label and default endpoint provider.",
    )
    parser.add_argument("--serial-no", default=os.getenv("ANDROID_SERIAL", "emulator-5554"))
    parser.add_argument("--host", default=os.getenv("ANDROID_ADB_HOST", "127.0.0.1"))
    parser.add_argument(
        "--adb-server-port",
        type=int,
        default=_env_int("ANDROID_ADB_SERVER_PORT", "ADB_SERVER_PORT", default=5037),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-action-retry", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--task-random-seed", type=int, default=None)
    parser.add_argument("--n-task-combinations", type=int, default=None)
    parser.add_argument("--request-interval", type=float, default=None)
    parser.add_argument("--wait-after-action-seconds", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--num-histories", type=int, default=None)
    parser.add_argument("--num-latest-screenshots", type=int, default=None)
    parser.add_argument("--include-device-time", action="store_true")
    parser.add_argument("--include-a11y-tree", action="store_true")
    parser.add_argument("--include-tips", action="store_true")
    parser.add_argument("--reflect-on-demand", action="store_true")
    parser.add_argument("--explored-knowledge-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--selected-for-export",
        action="store_true",
        help="Mark this run as an explicitly selected input for the per-model workbook.",
    )
    parser.add_argument(
        "--export-group",
        default=None,
        help="Explicit selection group used to keep smoke/formal/retry workbooks separate.",
    )

    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))

    parser.add_argument("--adb-path", default=os.getenv("ADB_PATH"))
    parser.add_argument("--console-port", type=int, default=_env_int("CONSOLE_PORT", default=5554))
    parser.add_argument("--grpc-port", type=int, default=_env_int("GRPC_PORT", default=8554))
    parser.add_argument("--perform-emulator-setup", action="store_true")
    parser.add_argument(
        "--androidworld-avd-name",
        default=None,
        help="Override the AndroidWorld AVD name for the selected mode.",
    )
    parser.add_argument(
        "--androidworld-launch-emulator",
        dest="androidworld_launch_emulator",
        action="store_true",
        default=None,
        help="Override AndroidWorld emulator launching to enabled.",
    )
    parser.add_argument(
        "--no-androidworld-launch-emulator",
        dest="androidworld_launch_emulator",
        action="store_false",
        help="Override AndroidWorld emulator launching to disabled.",
    )
    parser.add_argument(
        "--androidworld-wipe-data-on-launch",
        dest="androidworld_wipe_data_on_launch",
        action="store_true",
        default=None,
        help="Override AndroidWorld emulator launch wiping to enabled.",
    )
    parser.add_argument(
        "--no-androidworld-wipe-data-on-launch",
        dest="androidworld_wipe_data_on_launch",
        action="store_false",
        help="Override AndroidWorld emulator launch wiping to disabled.",
    )
    parser.add_argument("--suite-family", default=os.getenv("SUITE_FAMILY"))
    parser.add_argument("--tasks", default=None, help="Comma-separated AndroidWorld task names.")
    parser.add_argument("--fixed-task-seed", action="store_true")

    parser.add_argument("--task-configs", default=None, help="Comma-separated Android-Lab yaml config paths.")
    parser.add_argument("--task-ids", default=None, help="Comma-separated Android-Lab task ids.")
    parser.add_argument("--apps", default=None, help="Comma-separated Android-Lab app names.")
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--avd-name", default=os.getenv("AVD_NAME", "AndroidLabAvd"))
    parser.add_argument(
        "--avd-log-dir",
        default=os.getenv("AVD_LOG_DIR", str(PROJECT_ROOT / "outputs" / "androidlab_avd_logs")),
    )
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--docker-image-name", default=os.getenv("DOCKER_IMAGE_NAME", "android_eval:latest"))
    parser.add_argument("--docker-port", type=int, default=_env_int("DOCKER_PORT", default=6060))
    return parser


def apply_cli_overrides(cfg, args) -> None:
    if args.max_steps is not None:
        cfg.control.max_steps = args.max_steps
        cfg.benchmark.androidlab_max_rounds = args.max_steps
    if args.max_action_retry is not None:
        cfg.control.max_action_retry = args.max_action_retry
    if args.max_tokens is not None:
        cfg.model.max_tokens = args.max_tokens
        for role in ("p1", "p2", "p3_decision", "p3_planning", "p3_caption", "p4"):
            getattr(cfg.mode_models, role).max_tokens = args.max_tokens
    if args.request_interval is not None:
        cfg.executor.request_interval = args.request_interval
    if args.wait_after_action_seconds is not None:
        cfg.executor.wait_after_action_seconds = args.wait_after_action_seconds
    if args.timeout_seconds is not None:
        cfg.executor.timeout_seconds = args.timeout_seconds
    if args.max_rounds is not None:
        cfg.benchmark.androidlab_max_rounds = args.max_rounds
    if args.num_histories is not None:
        cfg.observation.num_histories = args.num_histories
    if args.num_latest_screenshots is not None:
        cfg.observation.num_latest_screenshots = args.num_latest_screenshots
    if args.include_device_time:
        cfg.observation.include_device_time = True
    if args.include_a11y_tree:
        cfg.observation.include_a11y_tree = True
    if args.include_tips:
        cfg.control.include_tips = True
    if args.reflect_on_demand:
        cfg.control.reflect_on_demand = True
    if args.androidworld_launch_emulator is not None:
        cfg.benchmark.androidworld_launch_emulator = args.androidworld_launch_emulator
    if args.androidworld_wipe_data_on_launch is not None:
        cfg.benchmark.androidworld_wipe_data_on_launch = args.androidworld_wipe_data_on_launch
    if args.androidworld_avd_name:
        mode_avd_field = f"androidworld_{args.mode}_avd_name"
        if hasattr(cfg.benchmark, mode_avd_field):
            setattr(cfg.benchmark, mode_avd_field, args.androidworld_avd_name)


def apply_mode_defaults(cfg, *, mode: str, explicit_timeout_seconds: int | None) -> None:
    if (
        mode == "p3"
        and explicit_timeout_seconds is None
        and cfg.executor.timeout_seconds is not None
        and cfg.benchmark.p3_timeout_seconds is not None
    ):
        cfg.executor.timeout_seconds = max(
            cfg.executor.timeout_seconds,
            cfg.benchmark.p3_timeout_seconds,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_defaults(args.defaults)
    cfg = fill_model_credentials(
        cfg,
        args.api_key,
        args.base_url,
        args.model_name,
        provider=args.provider,
    )
    apply_cli_overrides(cfg, args)
    apply_mode_defaults(cfg, mode=args.mode, explicit_timeout_seconds=args.timeout_seconds)
    if not args.dry_run:
        ensure_model_ready(cfg, mode=args.mode)

    output_root = Path(args.output_dir).resolve()
    if output_root.parent.name == "output-all" and cfg.control.max_steps != 30:
        raise ValueError(
            "Supplemental runs under output-all must use configured_max_steps=30; "
            "pass --max-steps 30 or update the model defaults file."
        )
    benchmark_output_dir = output_root / args.benchmark / args.mode
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.mode}_{args.benchmark}"
    run_output_dir = benchmark_output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    agent_log_dir = run_output_dir / "agent_logs"
    agent_log_dir.mkdir(parents=True, exist_ok=True)
    explored_knowledge_path = resolve_default_knowledge_path(
        mode=args.mode,
        run_output_dir=run_output_dir,
        provided_path=args.explored_knowledge_path,
    )
    if args.mode == "p4" and not cfg.benchmark.androidworld_p4_proactive_exploration:
        explored_knowledge_path = None

    manifest_defaults = _redact_secrets(cfg.to_dict())
    requested_tasks = _split_csv(args.tasks) if args.benchmark == "androidworld" else _split_csv(args.task_ids)
    if args.benchmark == "androidworld":
        avd_name = getattr(cfg.benchmark, f"androidworld_{args.mode}_avd_name", None)
    else:
        avd_name = args.avd_name
    selection_provenance = {
        "candidate_set_id": args.export_group,
        "selected_for_export": bool(args.selected_for_export),
        "selection_rule": "explicit_cli_flag",
        "selection_reason": (
            "run_marked_selected_for_export_by_launcher"
            if args.selected_for_export
            else "run_not_marked_selected_for_export_by_launcher"
        ),
        "candidate_count": None,
        "candidate_count_missing_reason": "resolved_at_export_time_across_model_root",
        "discarded_run_ids": None,
        "discarded_run_ids_missing_reason": "resolved_at_export_time_across_model_root",
    }
    manifest = {
        "runtime_metrics_version": 4,
        "mode": args.mode,
        "mode_label": MODE_LABELS[args.mode],
        "benchmark": args.benchmark,
        "run_name": run_name,
        "run_output_dir": str(run_output_dir),
        "defaults": manifest_defaults,
        "serial_no": args.serial_no,
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "a11y_host_port_base": (
            20000 + ((args.console_port - 5554) // 2) * 100
            if args.benchmark == "androidworld"
            else None
        ),
        "adb_path": args.adb_path,
        "avd_name": avd_name,
        "host": args.host,
        "adb_server_port": args.adb_server_port,
        "explored_knowledge_path": explored_knowledge_path,
        "configured_max_steps": cfg.control.max_steps,
        "timeout_seconds": cfg.executor.timeout_seconds,
        "max_action_retry": cfg.control.max_action_retry,
        "request_interval": cfg.executor.request_interval,
        "wait_after_action_seconds": cfg.executor.wait_after_action_seconds,
        "task_random_seed": cfg.control.task_random_seed if args.task_random_seed is None else args.task_random_seed,
        "fixed_task_seed": cfg.control.fixed_task_seed or args.fixed_task_seed,
        "n_task_combinations": (
            args.n_task_combinations
            if args.n_task_combinations is not None
            else cfg.control.n_task_combinations
        ),
        "task_names_requested": requested_tasks,
        "selected_for_export": bool(args.selected_for_export),
        "export_group": args.export_group,
        "selection_provenance": selection_provenance,
        "model_identity": {
            "requested_by_role": _requested_models_for_mode(cfg, args.mode),
            "runtime_snapshots": None,
            "runtime_snapshots_missing_reason": "run_not_completed",
        },
    }
    manifest["provenance"] = build_run_provenance(
        project_root=PROJECT_ROOT,
        defaults_path=args.defaults,
        effective_config=manifest_defaults,
    )
    agent_preview = build_mode_preview(
        mode=args.mode,
        cfg=cfg,
        serial_no=args.serial_no,
        host=args.host,
        port=args.adb_server_port,
        log_dir=agent_log_dir,
        explored_knowledge_path=explored_knowledge_path,
    )
    manifest["agent_config_preview"] = _redact_secrets(agent_preview)
    logs_dir = run_output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_path = logs_dir / "terminal.log"
    manifest["terminal_log"] = str(terminal_log_path)
    manifest["status"] = "prepared"
    manifest["started_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest_path = run_output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        manifest["model_identity"]["runtime_snapshots_missing_reason"] = "dry_run_no_provider_response"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint_dir = None
    actual_run_name = None
    records = []
    try:
        with terminal_log_path.open("a", encoding="utf-8") as log_file:
            tee_stdout = _TeeStream(sys.stdout, log_file)
            tee_stderr = _TeeStream(sys.stderr, log_file)
            with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                if args.benchmark == "androidworld":
                    from study.benchmarks.androidworld.runner import run_androidworld

                    checkpoint_dir = run_androidworld(
                        mode=args.mode,
                        defaults=cfg,
                        serial_no=args.serial_no,
                        host=args.host,
                        adb_server_port=args.adb_server_port,
                        adb_path=args.adb_path,
                        console_port=args.console_port,
                        grpc_port=args.grpc_port,
                        perform_emulator_setup=args.perform_emulator_setup,
                        suite_family=args.suite_family or cfg.benchmark.androidworld_suite_family,
                        tasks=_split_csv(args.tasks),
                        task_random_seed=args.task_random_seed,
                        n_task_combinations=args.n_task_combinations,
                        fixed_task_seed=args.fixed_task_seed,
                        run_output_dir=run_output_dir / "runs",
                        agent_log_dir=agent_log_dir,
                        explored_knowledge_path=explored_knowledge_path,
                    )
                else:
                    from study.benchmarks.androidlab.runner import run_androidlab

                    actual_run_name = run_androidlab(
                        mode=args.mode,
                        defaults=cfg,
                        serial_no=args.serial_no,
                        host=args.host,
                        adb_server_port=args.adb_server_port,
                        run_name=run_name,
                        save_dir=run_output_dir / "evaluation",
                        avd_name=args.avd_name,
                        avd_log_dir=args.avd_log_dir,
                        docker=args.docker,
                        docker_image_name=args.docker_image_name,
                        docker_port=args.docker_port,
                        task_config_paths=_split_csv(args.task_configs),
                        task_ids=_split_csv(args.task_ids),
                        apps=_split_csv(args.apps),
                        parallel=args.parallel,
                        explored_knowledge_path=explored_knowledge_path,
                    )

        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if checkpoint_dir is not None:
            manifest["checkpoint_dir"] = str(checkpoint_dir)
        if actual_run_name is not None:
            manifest["actual_run_name"] = actual_run_name
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            # Global AndroidWorld selected-index/export is rebuilt by dedicated
            # reporting scripts. Do not append per-run localized records into
            # the shared report index here, or we can overwrite the selected
            # index with a single run's records.
            from study.reporting import benchmark_report_paths
            from study.reporting import organize_run_output

            records = organize_run_output(run_output_dir, append_excel=False)
            manifest["report_record_count"] = len(records)
            manifest["report_excel"] = str(benchmark_report_paths(output_root, args.benchmark)["excel"])
            snapshots = collect_model_snapshots(run_output_dir)
            manifest["model_identity"]["runtime_snapshots"] = snapshots
            manifest["model_identity"]["runtime_snapshots_missing_reason"] = (
                snapshots["resolved_model_missing_reason"]
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            _sync_final_manifest_to_task_reports(run_output_dir, manifest)
        except Exception as report_exc:
            manifest["report_error"] = str(report_exc)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.benchmark == "androidworld":
        print(f"AndroidWorld run finished. Checkpoints: {checkpoint_dir}")
    else:
        print(f"AndroidLab run finished. Saved name: {actual_run_name}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
