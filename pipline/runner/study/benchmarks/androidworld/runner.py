"""AndroidWorld runner with unified mode config."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
import logging
from pathlib import Path
import os
import re
import subprocess
import sqlite3
import tempfile
import time
import xml.etree.ElementTree as ET
from unittest import mock

from guiagent.p2_appagent import AppAgentSession
from study.bootstrap import ensure_import_paths
from study.config import DefaultsConfig, get_mode_model_config
from study.modes.mobileuse_factory import build_mobileuse_agent_config


ensure_import_paths()

import mobile_use
from android_env import loader as android_env_loader
from android_env.components import adb_controller as adb_controller_lib
from android_env.components import adb_call_parser as adb_call_parser_lib
from android_env.components import adb_log_stream as adb_log_stream_lib
from android_env.components import action_fns as android_env_action_fns
from android_env.components import action_type as android_env_action_type
from android_env.components import config_classes as android_env_config_classes
from android_env.components import errors as android_env_errors
from android_env.proto import adb_pb2
from android_env.wrappers import a11y_grpc_wrapper as a11y_grpc_wrapper_lib
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.env import adb_utils as aw_adb_utils
from android_world.env import android_world_controller as aw_controller_lib
from android_world.env import env_launcher
from android_world.env.setup_device import apps as aw_setup_apps
from android_world.task_evals.information_retrieval import joplin_app_utils as aw_joplin_app_utils
from android_world.task_evals.single import retro_music as aw_retro_music
from android_world.task_evals.common_validators import sms_validators as aw_sms_validators
from android_world.task_evals.utils import sqlite_schema_utils as aw_sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils as aw_sqlite_utils
from android_world.utils import file_utils as aw_file_utils

from study.benchmarks.androidworld.androidworld_native import (
    AndroidWorldBackedAppAgentController,
    AndroidWorldBackedMobileUseEnvironment,
    AndroidWorldGuardedEnv,
)
from study.benchmarks.androidworld.mobile_agent_v3_output import OutputRecordingMobileAgentV3
from study.benchmarks.androidworld.mobile_use_agent import StudyMobileUseAgent
from study.benchmarks.androidworld.runtime_agent import StudyRuntimeAgent
from study.app_packages import resolve_package_name
from study.knowledge import ensure_mobileuse_explored_knowledge


_A11Y_HOST_PORT_BASE_ENV = "ANDROIDWORLD_A11Y_HOST_PORT_BASE"


def androidworld_a11y_host_port_base(console_port: int) -> int:
    """Return a disjoint host callback-port range for one emulator lane."""
    if console_port < 5554 or console_port % 2:
        raise ValueError(f"Unsupported Android emulator console port: {console_port}")
    lane = (console_port - 5554) // 2
    return 20000 + lane * 100


def _is_idempotent_empty_glob_remove_error(exc: BaseException) -> bool:
    """Return whether clear_directory lost an already-empty glob target."""
    if not isinstance(exc, android_env_errors.AdbControllerError):
        return False
    message = str(exc)
    return (
        "No such file or directory" in message
        and " shell rm -r " in message
        and "/*]" in message
    )


def _patch_androidworld_idempotent_clear_directory() -> None:
    """Treat only a vanished clear-directory glob as an idempotent success."""
    original_clear_directory = aw_file_utils.clear_directory
    protected_roots = {
        "/data",
        "/data/data",
        "/data/data/android_world",
        "/data/data/android_world/snapshots",
    }

    def _clear_directory_idempotent(directory_path, env) -> None:
        normalized_path = os.path.normpath(str(directory_path))
        if normalized_path in protected_roots:
            raise ValueError(
                f"Refusing to clear protected Android data root: {normalized_path}"
            )
        try:
            return original_clear_directory(directory_path, env)
        except android_env_errors.AdbControllerError as exc:
            if not _is_idempotent_empty_glob_remove_error(exc):
                raise
            return None

    aw_file_utils.clear_directory = _clear_directory_idempotent


def _patch_androidworld_sms_android_time() -> None:
    """Ignore native gRPC diagnostics accidentally mixed into ADB stdout."""
    original_get_android_time = aw_sms_validators.SimpleSMSSendSms.get_android_time
    original_decode_messages = aw_sms_validators._decode_messages_from_response

    def _get_android_time_robust(self, env) -> int:
        try:
            return original_get_android_time(self, env)
        except ValueError:
            adb_output = aw_adb_utils.issue_generic_request(
                ["shell", "date", "+%s"], env
            )
            output = adb_output.generic.output
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            timestamps = re.findall(r"(?m)^\s*(\d{9,12})\s*$", output)
            if not timestamps:
                raise
            return int(timestamps[-1]) * 1000

    aw_sms_validators.SimpleSMSSendSms.get_android_time = _get_android_time_robust

    def _decode_messages_robust(response):
        output = response.generic.output
        clean_lines = [
            line
            for line in output.splitlines()
            if line.startswith(b"Row:") or line.strip() == b"No result found."
        ]
        if clean_lines and b"\n".join(clean_lines) != output.strip():
            clean_response = adb_pb2.AdbResponse()
            clean_response.CopyFrom(response)
            clean_response.generic.output = b"\n".join(clean_lines)
            response = clean_response
        return original_decode_messages(response)

    aw_sms_validators._decode_messages_from_response = _decode_messages_robust


def _patch_android_env_windows_temp_apk_install() -> None:
    original_install_apk = adb_call_parser_lib.AdbCallParser._install_apk

    def _install_apk_windows_safe(self, request, timeout=None):
        install_apk = request.install_apk
        location_type = install_apk.WhichOneof("location")

        if os.name != "nt" or location_type != "blob":
            return original_install_apk(self, request, timeout)

        response = adb_pb2.AdbResponse()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as temp_file:
                temp_file.write(install_apk.blob.contents)
                temp_file.flush()
                temp_path = temp_file.name

            install_timeout = max(float(timeout or 0), 180.0)
            response, _ = self._execute_command(
                ["install", "--no-streaming", "-r", "-t", "-g", temp_path],
                timeout=install_timeout,
            )
            return response
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    adb_call_parser_lib.AdbCallParser._install_apk = _install_apk_windows_safe


def _patch_android_env_shared_adb_recovery() -> None:
    """Recover one device without killing the ADB server shared by all lanes."""

    def _restart_device_connection(self, timeout=None) -> None:
        recovery_timeout = max(float(timeout or 0), 15.0)
        server_prefix = self.command_prefix(include_device_name=False)
        device_prefix = self.command_prefix(include_device_name=True)

        # `start-server` is idempotent.  In contrast, AndroidEnv's default
        # recovery calls `kill-server`, which disconnects every concurrent
        # emulator when just one lane has a transient command failure.
        subprocess.run(
            [*server_prefix, "start-server"],
            capture_output=True,
            timeout=recovery_timeout,
            check=False,
        )
        subprocess.run(
            [*device_prefix, "reconnect"],
            capture_output=True,
            timeout=recovery_timeout,
            check=False,
        )
        subprocess.run(
            [*device_prefix, "wait-for-device"],
            capture_output=True,
            timeout=recovery_timeout,
            check=False,
        )
        time.sleep(0.5)

    adb_controller_lib.AdbController._restart_server = _restart_device_connection


def _patch_android_env_windows_start_activity() -> None:
    original_start_activity = adb_call_parser_lib.AdbCallParser._start_activity

    def _start_activity_windows_safe(self, request, timeout=None):
        if os.name != "nt":
            # AndroidEnv's caller commonly passes five seconds. On macOS, four
            # concurrent emulators can launch the activity successfully but
            # `am start -W` may need longer to return its completion record.
            return original_start_activity(
                self,
                request,
                max(float(timeout or 0), 30.0),
            )

        activity = request.start_activity.full_activity
        if not activity:
            return adb_pb2.AdbResponse(
                status=adb_pb2.AdbResponse.Status.FAILED_PRECONDITION,
                error_message="`start_activity.full_activity` cannot be empty.",
            )

        command = ["shell", "am", "start"]
        if request.start_activity.force_stop:
            command.append("-S")
        command.extend(["-W", "-n", activity])
        command.extend(list(request.start_activity.extra_args or []))

        start_timeout = max(float(timeout or 0), 30.0)
        response, command_output = self._execute_command(command, timeout=start_timeout)
        response.start_activity.full_activity = activity
        response.start_activity.output = command_output
        return response

    adb_call_parser_lib.AdbCallParser._start_activity = _start_activity_windows_safe


def _patch_android_env_a11y_forwarder_install() -> None:
    original_install = a11y_grpc_wrapper_lib.A11yGrpcWrapper._install_a11y_forwarding_apk

    def _install_a11y_forwarding_apk_cached(self) -> None:
        package_response = self._env.execute_adb_call(
            adb_pb2.AdbRequest(
                package_manager=adb_pb2.AdbRequest.PackageManagerRequest(
                    list=adb_pb2.AdbRequest.PackageManagerRequest.List(
                        packages=adb_pb2.AdbRequest.PackageManagerRequest.List.Packages()
                    )
                )
            )
        )
        if package_response.status == adb_pb2.AdbResponse.Status.OK:
            installed_packages = set(package_response.package_manager.list.items)
            if "com.google.androidenv.accessibilityforwarder" in installed_packages:
                return
        return original_install(self)

    a11y_grpc_wrapper_lib.A11yGrpcWrapper._install_a11y_forwarding_apk = _install_a11y_forwarding_apk_cached


def _patch_android_env_logcat_reconnect() -> None:
    """Keep AndroidEnv log readers alive across a short concurrent ADB reconnect."""
    original_get_stream_output = adb_log_stream_lib.AdbLogStream._get_stream_output

    def _get_stream_output_with_retry(self):
        last_error = None
        for attempt in range(1, 9):
            try:
                return original_get_stream_output(self)
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                logging.warning(
                    "AndroidEnv logcat setup failed (attempt %d/8); retrying: %s",
                    attempt,
                    exc,
                )
                time.sleep(min(3.0, 0.5 * attempt))
        # A log stream is diagnostic only. Do not turn a temporary device
        # reconnect into an uncaught exception in AndroidEnv's daemon thread.
        logging.error("AndroidEnv logcat disabled after retries: %s", last_error)
        return iter(())

    adb_log_stream_lib.AdbLogStream._get_stream_output = _get_stream_output_with_retry


def _patch_android_env_a11y_host_port_range() -> None:
    """Prevent concurrent lanes from racing for the same callback port."""
    original_init = a11y_grpc_wrapper_lib.A11yGrpcWrapper.__init__

    def _init_with_lane_port(self, *args, **kwargs):
        raw_base = os.getenv(_A11Y_HOST_PORT_BASE_ENV)
        if not raw_base:
            return original_init(self, *args, **kwargs)
        base = int(raw_base)

        def _pick_lane_port() -> int:
            for port in range(base, base + 32):
                if a11y_grpc_wrapper_lib.portpicker.is_port_free(port):
                    return port
            raise RuntimeError(
                f"No free AndroidWorld a11y callback port in [{base}, {base + 31}]"
            )

        with mock.patch.object(
            a11y_grpc_wrapper_lib.portpicker,
            "pick_unused_port",
            side_effect=_pick_lane_port,
        ):
            return original_init(self, *args, **kwargs)

    a11y_grpc_wrapper_lib.A11yGrpcWrapper.__init__ = _init_with_lane_port


def _patch_androidworld_uiautomator_dump_retry() -> None:
    original_dump = aw_adb_utils.uiautomator_dump

    def _host_adb_uiautomator_dump(env, timeout_sec: float) -> str:
        """Read a UI dump through host ADB when AndroidEnv's RPC path is stale.

        AndroidEnv can occasionally return its own diagnostic text instead of
        the device command output. The emulator's ADB endpoint remains usable
        in that state, so use it as a narrow recovery path for the hierarchy
        read only.
        """
        config = env._coordinator._simulator._config  # pylint: disable=protected-access
        adb_config = config.adb_controller
        launcher_config = config.emulator_launcher
        adb_path = str(adb_config.adb_path)
        adb_server_port = int(adb_config.adb_server_port)
        host = getattr(adb_config, "adb_server_host", None) or os.getenv(
            "ANDROID_ADB_SERVER_HOST", ""
        )
        serial = f"emulator-{int(launcher_config.emulator_console_port)}"
        command = [adb_path]
        if host:
            command.extend(["-H", host])
        command.extend(["-P", str(adb_server_port), "-s", serial, "shell"])

        dump = subprocess.run(
            [*command, "uiautomator", "dump", "/sdcard/window_dump.xml"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(float(timeout_sec), 10.0),
            check=False,
        )
        read = subprocess.run(
            [*command, "cat", "/sdcard/window_dump.xml"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(float(timeout_sec), 10.0),
            check=False,
        )
        xml_text = read.stdout
        if dump.returncode != 0 or read.returncode != 0:
            raise RuntimeError(
                "Host ADB uiautomator dump failed: "
                f"dump={dump.stderr.strip() or dump.stdout.strip()} "
                f"read={read.stderr.strip() or read.stdout.strip()}"
            )
        ET.fromstring(xml_text)
        return xml_text

    def _uiautomator_dump_retry(env, timeout_sec=30):
        latest_error = None
        for attempt in range(1, 6):
            try:
                xml_text = original_dump(env, timeout_sec=timeout_sec)
                # ADB can return the command's human-readable failure text or
                # a partially written dump while AccessibilityManager is
                # restarting. Validate the payload before handing it to the
                # XML parser, then issue a fresh dump on the next attempt.
                ET.fromstring(xml_text)
                return xml_text
            except ET.ParseError as exc:
                latest_error = exc
            except Exception as exc:
                latest_error = exc
            try:
                return _host_adb_uiautomator_dump(env, timeout_sec)
            except Exception as exc:
                latest_error = exc
            if attempt == 5:
                break
            time.sleep(1.5)
        raise latest_error

    aw_adb_utils.uiautomator_dump = _uiautomator_dump_retry


def _patch_androidworld_slow_apk_install() -> None:
    def _wait_for_package_service(env, attempts: int = 6) -> None:
        last_output = ""
        for attempt in range(1, attempts + 1):
            response = aw_adb_utils.issue_generic_request(
                ["shell", "cmd", "package", "list", "packages", "android"],
                env,
                timeout_sec=30.0,
            )
            last_output = (response.generic.output or b"").decode("utf-8", errors="replace")
            if response.status == adb_pb2.AdbResponse.Status.OK and "package:android" in last_output:
                return
            time.sleep(min(20.0, 2.0 * attempt))
        raise RuntimeError(f"Android package service did not become ready: {last_output.strip()}")

    def _install_apk_with_long_timeout(apk_location: str, env) -> None:
        if not os.path.exists(apk_location):
            raise ValueError("APK does not exist.")
        last_output = ""
        for attempt in range(1, 6):
            _wait_for_package_service(env)
            response = aw_adb_utils.issue_generic_request(
                ["install", "--no-streaming", "-r", "-t", "-g", apk_location],
                env,
                timeout_sec=240.0,
            )
            last_output = (response.generic.output or b"").decode("utf-8", errors="replace")
            if response.status == adb_pb2.AdbResponse.Status.OK:
                return
            time.sleep(min(60.0, 8.0 * attempt))
        raise RuntimeError(f"Failed to install APK after retries: {apk_location}\n{last_output.strip()}")

    aw_adb_utils.install_apk = _install_apk_with_long_timeout


def _patch_androidworld_slow_close_app() -> None:
    original_close_app = aw_adb_utils.close_app

    def _close_app_with_long_timeout(app_name: str, env, timeout_sec=None):
        close_timeout = max(float(timeout_sec or 0), 60.0)
        return original_close_app(app_name, env, timeout_sec=close_timeout)

    aw_adb_utils.close_app = _close_app_with_long_timeout


def _patch_android_env_lift_reset_touch() -> None:
    original_send_action = android_env_action_fns.send_action_to_simulator

    def _scalar_int(value) -> int:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    def _is_lift_only(action: dict) -> bool:
        action_type_keys = [key for key in action if str(key).startswith("action_type")]
        if not action_type_keys:
            return False
        return all(
            _scalar_int(action[key]) == int(android_env_action_type.ActionType.LIFT)
            for key in action_type_keys
        )

    def _send_action_without_reset_lift(action, simulator, screen_width, screen_height, num_fingers):
        if _is_lift_only(action):
            return True
        return original_send_action(action, simulator, screen_width, screen_height, num_fingers)

    android_env_action_fns.send_action_to_simulator = _send_action_without_reset_lift


def _sqlite_type_for_field(field: dataclasses.Field) -> str:
    type_name = getattr(field.type, "__name__", str(field.type)).lower()
    if "bool" in type_name or "int" in type_name:
        return "INTEGER"
    if "float" in type_name:
        return "REAL"
    return "TEXT"


def _create_table_for_row_type(cursor: sqlite3.Cursor, table_name: str, row_type: type, exclude_key: str | None = None) -> None:
    columns = []
    for field in dataclasses.fields(row_type):
        if exclude_key is not None and field.name == exclude_key:
            continue
        columns.append(f'"{field.name}" {_sqlite_type_for_field(field)}')
    if not columns:
        raise sqlite3.OperationalError(f"table {table_name} has no columns")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})")


def _joplin_row_type_for_table(table_name: str) -> type | None:
    return {
        "folders": aw_sqlite_schema_utils.JoplinFolder,
        "notes": aw_sqlite_schema_utils.JoplinNote,
        "notes_normalized": aw_sqlite_schema_utils.JoplinNormalizedNote,
    }.get(table_name)


def _sqlite_row_type_for_table(table_name: str) -> type | None:
    joplin_type = _joplin_row_type_for_table(table_name)
    if joplin_type is not None:
        return joplin_type
    return {
        "events": aw_sqlite_schema_utils.CalendarEvent,
        "expense": aw_sqlite_schema_utils.Expense,
        "map_markers": aw_sqlite_schema_utils.OsmAndMapMarker,
        "recipes": aw_sqlite_schema_utils.Recipe,
        "track": aw_sqlite_schema_utils.SportsActivity,
    }.get(table_name)


def _ensure_remote_sqlite_table(
    *,
    env,
    remote_db_file_path: str,
    table_name: str,
    row_type: type,
    app_name: str,
    exclude_key: str | None = None,
    timeout_sec=None,
) -> None:
    with env.controller.pull_file(remote_db_file_path, timeout_sec) as local_db_directory:
        local_db_path = aw_file_utils.convert_to_posix_path(
            local_db_directory, os.path.split(remote_db_file_path)[1]
        )
        conn = sqlite3.connect(local_db_path)
        try:
            cursor = conn.cursor()
            _create_table_for_row_type(cursor, table_name, row_type, exclude_key)
            conn.commit()
        finally:
            conn.close()
        env.controller.push_file(local_db_path, remote_db_file_path, timeout_sec)
        aw_adb_utils.close_app(app_name, env.controller)


def _ensure_remote_expense_table(
    *,
    env,
    remote_db_file_path: str,
    app_name: str,
    timeout_sec=None,
) -> None:
    with env.controller.pull_file(remote_db_file_path, timeout_sec) as local_db_directory:
        local_db_path = aw_file_utils.convert_to_posix_path(
            local_db_directory, os.path.split(remote_db_file_path)[1]
        )
        conn = sqlite3.connect(local_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS expense (
                    expense_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    amount INTEGER,
                    category INTEGER,
                    note TEXT,
                    created_date INTEGER,
                    modified_date INTEGER
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        env.controller.push_file(local_db_path, remote_db_file_path, timeout_sec)
        aw_adb_utils.close_app(app_name, env.controller)


def _patch_androidworld_schema_aware_sqlite_insert() -> None:
    def _insert_rows_schema_aware(
        rows,
        exclude_key,
        table_name,
        remote_db_file_path,
        app_name,
        env,
        timeout_sec=None,
    ) -> None:
        with env.controller.pull_file(
            remote_db_file_path, timeout_sec
        ) as local_db_directory:
            local_db_path = aw_file_utils.convert_to_posix_path(
                local_db_directory, os.path.split(remote_db_file_path)[1]
            )

            conn = sqlite3.connect(local_db_path)
            try:
                cursor = conn.cursor()
                table_columns = {
                    row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")
                }
                if not table_columns:
                    if table_name == "expense":
                        cursor.execute(
                            """
                            CREATE TABLE IF NOT EXISTS expense (
                                expense_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name TEXT,
                                amount INTEGER,
                                category INTEGER,
                                note TEXT,
                                created_date INTEGER,
                                modified_date INTEGER
                            )
                            """
                        )
                        table_columns = {
                            row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")
                        }
                    elif rows:
                        _create_table_for_row_type(cursor, table_name, rows[0].__class__, exclude_key)
                        table_columns = {
                            row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")
                        }
                    if not table_columns:
                        raise sqlite3.OperationalError(f"table {table_name} has no columns")

                for row in rows:
                    fields = [
                        field
                        for field in dataclasses.fields(row)
                        if field.name != exclude_key and field.name in table_columns
                    ]
                    if not fields:
                        continue
                    column_names = ", ".join(f'"{field.name}"' for field in fields)
                    placeholders = ", ".join("?" * len(fields))
                    insert_command = (
                        f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
                    )
                    values = tuple(getattr(row, field.name) for field in fields)
                    cursor.execute(insert_command, values)
                conn.commit()
            finally:
                conn.close()

            env.controller.push_file(local_db_path, remote_db_file_path, timeout_sec)
            aw_adb_utils.close_app(app_name, env.controller)

    aw_sqlite_utils.insert_rows_to_remote_db = _insert_rows_schema_aware


def _patch_androidworld_lenient_sqlite_delete() -> None:
    original_delete = aw_sqlite_utils.delete_all_rows_from_table

    def _delete_all_rows_schema_aware(
        table_name: str,
        remote_db_file_path: str,
        env,
        app_name: str,
        timeout_sec=None,
    ) -> None:
        try:
            return original_delete(table_name, remote_db_file_path, env, app_name, timeout_sec)
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).lower():
                raise
            if table_name == "expense":
                _ensure_remote_expense_table(
                    env=env,
                    remote_db_file_path=remote_db_file_path,
                    app_name=app_name,
                    timeout_sec=timeout_sec,
                )
                return original_delete(table_name, remote_db_file_path, env, app_name, timeout_sec)
            row_type = _sqlite_row_type_for_table(table_name)
            if row_type is None:
                raise
            _ensure_remote_sqlite_table(
                env=env,
                remote_db_file_path=remote_db_file_path,
                table_name=table_name,
                row_type=row_type,
                app_name=app_name,
                timeout_sec=timeout_sec,
            )
            return original_delete(table_name, remote_db_file_path, env, app_name, timeout_sec)

    aw_sqlite_utils.delete_all_rows_from_table = _delete_all_rows_schema_aware


def _patch_androidworld_lenient_joplin_folder_lookup() -> None:
    original_get_folder_to_id = aw_joplin_app_utils._get_folder_to_id

    def _get_folder_to_id_lenient(env) -> dict[str, str]:
        try:
            return original_get_folder_to_id(env)
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).lower():
                raise
            _ensure_remote_sqlite_table(
                env=env,
                remote_db_file_path=aw_joplin_app_utils._DB_PATH,
                table_name=aw_joplin_app_utils._FOLDER_TABLE,
                row_type=aw_sqlite_schema_utils.JoplinFolder,
                app_name=aw_joplin_app_utils._APP_NAME,
                exclude_key=aw_joplin_app_utils._EXCLUDE_FIELD,
            )
            return {}

    aw_joplin_app_utils._get_folder_to_id = _get_folder_to_id_lenient


def _patch_androidworld_lenient_retro_playing_queue() -> None:
    """Treat an uncreated Retro Music queue table as an empty queue."""
    original_get_playing_queue = aw_retro_music._get_playing_queue

    def _get_playing_queue_lenient(env) -> list[str]:
        try:
            return original_get_playing_queue(env)
        except sqlite3.OperationalError as exc:
            if "no such table: playing_queue" not in str(exc).lower():
                raise
            return []

    aw_retro_music._get_playing_queue = _get_playing_queue_lenient


def _patch_androidworld_lenient_expense_setup() -> None:
    def _expense_setup_lenient(cls, env) -> None:
        aw_setup_apps.AppSetup.setup.__func__(cls, env)
        aw_adb_utils.launch_app(cls.app_name, env.controller)
        try:
            time.sleep(5.0)
            controller = aw_setup_apps.tools.AndroidToolController(env=env.controller)
            for text in ("NEXT", "CONTINUE"):
                try:
                    controller.click_element(text)
                    time.sleep(2.0)
                except Exception:
                    pass
        finally:
            aw_adb_utils.close_app(cls.app_name, env.controller)

    aw_setup_apps.ExpenseApp.setup = classmethod(_expense_setup_lenient)


_patch_android_env_windows_temp_apk_install()
_patch_android_env_shared_adb_recovery()
_patch_android_env_windows_start_activity()
_patch_android_env_a11y_forwarder_install()
_patch_android_env_logcat_reconnect()
_patch_android_env_a11y_host_port_range()
_patch_androidworld_uiautomator_dump_retry()
_patch_androidworld_slow_apk_install()
_patch_androidworld_slow_close_app()
_patch_android_env_lift_reset_touch()
_patch_androidworld_idempotent_clear_directory()
_patch_androidworld_sms_android_time()
_patch_androidworld_schema_aware_sqlite_insert()
_patch_androidworld_lenient_sqlite_delete()
_patch_androidworld_lenient_joplin_folder_lookup()
_patch_androidworld_lenient_retro_playing_queue()
_patch_androidworld_lenient_expense_setup()


def find_adb_path() -> str:
    potential_paths = []
    sdk_root = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
    if sdk_root:
        potential_paths.extend(
            [
                os.path.join(sdk_root, "platform-tools", "adb.exe"),
                os.path.join(sdk_root, "platform-tools", "adb"),
            ]
        )
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        potential_paths.append(
            os.path.join(local_app_data, "Android", "Sdk", "platform-tools", "adb.exe")
        )
    potential_paths.extend(
        [
            os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
            "/root/android/android_sdk/platform-tools/adb",
        ]
    )
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        "adb not found in common Android SDK paths. Pass --adb-path or set ANDROID_SDK_ROOT/ANDROID_HOME."
    )


def find_emulator_path() -> str:
    potential_paths = []
    sdk_root = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
    if sdk_root:
        potential_paths.extend(
            [
                os.path.join(sdk_root, "emulator", "emulator.exe"),
                os.path.join(sdk_root, "emulator", "emulator"),
            ]
        )
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        potential_paths.append(
            os.path.join(local_app_data, "Android", "Sdk", "emulator", "emulator.exe")
        )
    potential_paths.extend(
        [
            os.path.expanduser("~/Library/Android/sdk/emulator/emulator"),
            os.path.expanduser("~/Android/Sdk/emulator/emulator"),
            "/root/android/android_sdk/emulator/emulator",
        ]
    )
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        "emulator not found in common Android SDK paths. Set ANDROID_SDK_ROOT/ANDROID_HOME."
    )


def ensure_adb_server(adb_path: str, adb_server_port: int) -> None:
    start_command = [adb_path, "-P", str(adb_server_port), "start-server"]
    result = subprocess.run(start_command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"Failed to start adb server on dedicated port {adb_server_port}.\n"
        "The runner will not call `adb kill-server` because that could "
        "disconnect unrelated experiments.\n"
        f"stdout: {result.stdout.strip()}\n"
        f"stderr: {result.stderr.strip()}"
    )


def _adb_base_command(adb_path: str, host: str, adb_server_port: int) -> list[str]:
    command = [adb_path]
    if host:
        command.extend(["-H", host])
    command.extend(["-P", str(adb_server_port)])
    return command


def _list_adb_devices(*, adb_path: str, host: str, adb_server_port: int) -> dict[str, str]:
    devices = subprocess.run(
        [*_adb_base_command(adb_path, host, adb_server_port), "devices"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    result: dict[str, str] = {}
    if devices.returncode != 0:
        return result
    for line in devices.stdout.splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[0].startswith("emulator-"):
            result[columns[0]] = columns[1]
    return result


def _get_emulator_avd_name(*, adb_path: str, host: str, adb_server_port: int, serial_no: str) -> str | None:
    response = subprocess.run(
        [*_adb_base_command(adb_path, host, adb_server_port), "-s", serial_no, "emu", "avd", "name"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if response.returncode != 0:
        return None
    lines = [line.strip() for line in response.stdout.splitlines() if line.strip()]
    for line in lines[::-1]:
        if line.lower() != "ok":
            return line
    return None


def _wait_for_device_disconnect(
    *,
    adb_path: str,
    host: str,
    adb_server_port: int,
    serial_no: str,
    timeout_seconds: int = 90,
    poll_seconds: float = 2.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        devices = _list_adb_devices(adb_path=adb_path, host=host, adb_server_port=adb_server_port)
        if serial_no not in devices:
            return
        time.sleep(poll_seconds)


def _select_androidworld_avd_name(mode: str, defaults: DefaultsConfig) -> str:
    per_mode = {
        "p1": defaults.benchmark.androidworld_p1_avd_name,
        "p2": defaults.benchmark.androidworld_p2_avd_name,
        "p3": defaults.benchmark.androidworld_p3_avd_name,
        "p4": defaults.benchmark.androidworld_p4_avd_name,
    }.get(mode)
    if per_mode:
        return per_mode
    return defaults.benchmark.androidworld_appagent_avd_name


def ensure_androidworld_emulator(
    *,
    mode: str,
    defaults: DefaultsConfig,
    adb_path: str,
    host: str,
    adb_server_port: int,
    serial_no: str,
    console_port: int,
    grpc_port: int,
) -> subprocess.Popen[str] | None:
    if not defaults.benchmark.androidworld_launch_emulator:
        return None

    target_avd_name = _select_androidworld_avd_name(mode, defaults)
    devices = _list_adb_devices(adb_path=adb_path, host=host, adb_server_port=adb_server_port)
    if devices.get(serial_no) == "device":
        current_avd_name = _get_emulator_avd_name(
            adb_path=adb_path,
            host=host,
            adb_server_port=adb_server_port,
            serial_no=serial_no,
        )
        if current_avd_name == target_avd_name and not defaults.benchmark.androidworld_wipe_data_on_launch:
            return None
        subprocess.run(
            [*_adb_base_command(adb_path, host, adb_server_port), "-s", serial_no, "emu", "kill"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        _wait_for_device_disconnect(
            adb_path=adb_path,
            host=host,
            adb_server_port=adb_server_port,
            serial_no=serial_no,
        )

    emulator_path = find_emulator_path()
    emulator_command = [
        emulator_path,
        "-avd",
        target_avd_name,
        "-port",
        str(console_port),
        "-grpc",
        str(grpc_port),
        "-no-audio",
        "-no-snapshot-save",
    ]
    if not defaults.benchmark.emulator_show_window:
        emulator_command.append("-no-window")
    if defaults.benchmark.emulator_gpu:
        emulator_command.extend(["-gpu", defaults.benchmark.emulator_gpu])
    if defaults.benchmark.emulator_memory_mb:
        emulator_command.extend(["-memory", str(defaults.benchmark.emulator_memory_mb)])
    if defaults.benchmark.androidworld_wipe_data_on_launch:
        emulator_command.append("-wipe-data")
    return subprocess.Popen(
        emulator_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            **os.environ,
            "ANDROID_ADB_SERVER_PORT": str(adb_server_port),
            **({"ANDROID_ADB_SERVER_HOST": host} if host else {}),
        },
    )


def stop_androidworld_emulator(
    *,
    adb_path: str,
    host: str,
    adb_server_port: int,
    serial_no: str,
    process: subprocess.Popen[str] | None,
) -> None:
    if process is None:
        return
    try:
        subprocess.run(
            [*_adb_base_command(adb_path, host, adb_server_port), "-s", serial_no, "emu", "kill"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except Exception:
        pass
    _wait_for_device_disconnect(
        adb_path=adb_path,
        host=host,
        adb_server_port=adb_server_port,
        serial_no=serial_no,
        timeout_seconds=30,
    )
    try:
        process.wait(timeout=30)
    except Exception:
        try:
            process.terminate()
            process.wait(timeout=15)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def wait_for_adb_device_ready(
    *,
    adb_path: str,
    host: str,
    adb_server_port: int,
    serial_no: str,
    timeout_seconds: int = 120,
    poll_seconds: float = 2.0,
) -> None:
    deadline = time.time() + timeout_seconds
    base_command = _adb_base_command(adb_path, host, adb_server_port)
    last_reason = f"Device {serial_no} is not listed by adb yet."

    while time.time() < deadline:
        try:
            devices = subprocess.run(
                [*base_command, "devices"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_reason = f"`adb devices` probe failed for {serial_no}: {exc}"
            time.sleep(poll_seconds)
            continue
        online = False
        if devices.returncode == 0:
            for line in devices.stdout.splitlines():
                columns = line.split()
                if len(columns) >= 2 and columns[0] == serial_no and columns[1] == "device":
                    online = True
                    break
            if not online:
                devices_stdout = devices.stdout or ""
                devices_stderr = devices.stderr or ""
                last_reason = (
                    f"Device {serial_no} is not connected. "
                    f"adb devices output:\n{devices_stdout.strip()}"
                )
        else:
            devices_stdout = devices.stdout or ""
            devices_stderr = devices.stderr or ""
            last_reason = f"`adb devices` failed: {devices_stderr.strip() or devices_stdout.strip()}"

        if online:
            try:
                boot = subprocess.run(
                    [*base_command, "-s", serial_no, "shell", "getprop", "sys.boot_completed"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_reason = f"Device {serial_no} is online but boot probe failed: {exc}"
                time.sleep(poll_seconds)
                continue
            boot_stdout = boot.stdout or ""
            if boot.returncode == 0 and boot_stdout.strip() == "1":
                return
            last_reason = (
                f"Device {serial_no} is connected but not fully booted yet. "
                f"sys.boot_completed={boot_stdout.strip()!r}"
            )

        time.sleep(poll_seconds)

    raise RuntimeError(
        "AndroidWorld could not find a ready emulator before environment setup.\n"
        f"Expected adb serial: {serial_no}\n"
        f"adb host/port: {host}:{adb_server_port}\n"
        f"Last observed state: {last_reason}\n"
        "Start the AndroidWorld AVD first, wait for boot completion, then rerun."
    )


def wait_for_android_services_ready(
    *,
    adb_path: str,
    host: str,
    adb_server_port: int,
    serial_no: str,
    timeout_seconds: int = 240,
    poll_seconds: float = 5.0,
    settle_seconds: float = 15.0,
    require_uiautomator: bool = True,
) -> None:
    deadline = time.time() + timeout_seconds
    base_command = _adb_base_command(adb_path, host, adb_server_port)
    last_reason = "Android framework services have not been probed yet."
    stable_count = 0

    probes = [
        ("package", ["shell", "cmd", "package", "list", "packages", "android"]),
        ("activity", ["shell", "cmd", "activity", "get-current-user"]),
    ]
    if require_uiautomator:
        probes.extend(
            [
                ("uiautomator_dump", ["shell", "uiautomator", "dump", "/sdcard/window_dump.xml"]),
                ("window_dump", ["shell", "cat", "/sdcard/window_dump.xml"]),
            ]
        )

    while time.time() < deadline:
        failed = False
        for probe_name, probe_args in probes:
            try:
                result = subprocess.run(
                    [*base_command, "-s", serial_no, *probe_args],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failed = True
                last_reason = f"{probe_name} probe raised {exc}"
                break
            if result.returncode != 0:
                failed = True
                stdout = (result.stdout or "").strip()
                stderr = (result.stderr or "").strip()
                last_reason = f"{probe_name} probe failed: {stderr or stdout}"
                break

        if not failed:
            stable_count += 1
            if stable_count >= 2:
                time.sleep(settle_seconds)
                return
        else:
            stable_count = 0
            time.sleep(poll_seconds)

    raise RuntimeError(
        "AndroidWorld emulator booted but Android framework services were not ready "
        "before environment setup.\n"
        f"Expected adb serial: {serial_no}\n"
        f"adb host/port: {host}:{adb_server_port}\n"
        f"Last observed state: {last_reason}"
    )


def _restart_androidworld_a11y_forwarder(
    *,
    adb_path: str,
    host: str,
    adb_server_port: int,
    serial_no: str,
    timeout_seconds: int = 60,
) -> None:
    """Force a clean accessibility-service bind instead of trusting a stale PID."""
    base_command = _adb_base_command(adb_path, host, adb_server_port)
    package = "com.google.androidenv.accessibilityforwarder"
    component = f"{package}/{package}.AccessibilityForwarder"

    def _adb(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*base_command, "-s", serial_no, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    deadline = time.time() + timeout_seconds
    last_state = "not started"
    first_cycle = True
    while time.time() < deadline:
        # A PID can survive a dead AccessibilityService registration. Rebind
        # in short cycles so a failed P4 lane gets several fresh starts within
        # the overall readiness budget instead of waiting on one stale process.
        if not first_cycle:
            _adb("shell", "am", "force-stop", package)
            _adb("shell", "settings", "put", "secure", "accessibility_enabled", "0")
            _adb("shell", "settings", "delete", "secure", "enabled_accessibility_services")
            time.sleep(0.5)
        for key, value in (
            ("enabled_accessibility_services", component),
            ("accessibility_enabled", "1"),
        ):
            response = _adb("shell", "settings", "put", "secure", key, value)
            if response.returncode != 0:
                last_state = f"settings {key}: {response.stderr.strip() or response.stdout.strip()}"
                break
        else:
            stable_pid = 0
            cycle_deadline = min(deadline, time.time() + 12.0)
            while time.time() < cycle_deadline:
                pid = _adb("shell", "pidof", package)
                services = _adb("shell", "dumpsys", "accessibility")
                configured = _adb("shell", "settings", "get", "secure", "enabled_accessibility_services")
                pid_ok = pid.returncode == 0 and bool(pid.stdout.strip())
                service_text = services.stdout or ""
                configured_text = configured.stdout or ""
                bound_ok = component in service_text or component in configured_text
                if pid_ok and bound_ok:
                    stable_pid += 1
                    if stable_pid >= 2:
                        time.sleep(1.0)
                        return
                else:
                    stable_pid = 0
                    last_state = (
                        f"pid={pid.stdout.strip()!r} accessibility="
                        f"{service_text[:160].strip()!r}"
                    )
                time.sleep(1.0)
        first_cycle = False
        time.sleep(0.5)
    raise RuntimeError(
        f"AndroidEnv accessibility forwarder did not stably bind on {serial_no} "
        f"within {timeout_seconds} seconds; last state: {last_state}"
    )


def ensure_androidworld_a11y_forwarder_ready(
    *,
    adb_path: str,
    host: str,
    adb_server_port: int,
    serial_no: str,
    timeout_seconds: int = 60,
) -> None:
    """Ensure the installed AndroidEnv accessibility service is freshly bound."""
    base_command = _adb_base_command(adb_path, host, adb_server_port)
    package = "com.google.androidenv.accessibilityforwarder"
    component = f"{package}/{package}.AccessibilityForwarder"

    def _adb(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*base_command, "-s", serial_no, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    package_path = _adb("shell", "pm", "path", package)
    if package_path.returncode != 0 or not package_path.stdout.strip().startswith("package:"):
        # Fresh AVD snapshots may not contain the APK yet. AndroidEnv's
        # A11yGrpcWrapper installs it during environment construction when
        # `install_a11y_forwarding` is enabled, so do not reject the lane
        # before that installation path can run.
        print(
            f"[androidworld] accessibility forwarder APK absent on {serial_no}; "
            "deferring install to AndroidEnv",
            flush=True,
        )
        return

    _restart_androidworld_a11y_forwarder(
        adb_path=adb_path,
        host=host,
        adb_server_port=adb_server_port,
        serial_no=serial_no,
        timeout_seconds=timeout_seconds,
    )


def _patch_androidworld_a11y_tree_recovery() -> None:
    """Retry empty a11y streams by rebinding the service and gRPC callback."""
    original_get_a11y_forest = aw_controller_lib.AndroidWorldController.get_a11y_forest

    def _get_a11y_forest_with_recovery(self):
        last_error: RuntimeError | None = None
        for attempt in range(1, 4):
            try:
                return original_get_a11y_forest(self)
            except RuntimeError as exc:
                last_error = exc
                if attempt == 3:
                    break
                try:
                    # The wrapper exposes the underlying AndroidEnv interface;
                    # use its existing ADB path rather than a second adb client.
                    env = self.env
                    aw_adb_utils.issue_generic_request(
                        ["shell", "am", "force-stop", "com.google.androidenv.accessibilityforwarder"],
                        env,
                        timeout_sec=30.0,
                    )
                    aw_adb_utils.issue_generic_request(
                        ["shell", "settings", "put", "secure", "accessibility_enabled", "0"],
                        env,
                        timeout_sec=30.0,
                    )
                    aw_adb_utils.issue_generic_request(
                        ["shell", "settings", "delete", "secure", "enabled_accessibility_services"],
                        env,
                        timeout_sec=30.0,
                    )
                    time.sleep(1.0)
                    aw_adb_utils.issue_generic_request(
                        [
                            "shell",
                            "settings",
                            "put",
                            "secure",
                            "enabled_accessibility_services",
                            "com.google.androidenv.accessibilityforwarder/com.google.androidenv.accessibilityforwarder.AccessibilityForwarder",
                        ],
                        env,
                        timeout_sec=30.0,
                    )
                    aw_adb_utils.issue_generic_request(
                        ["shell", "settings", "put", "secure", "accessibility_enabled", "1"],
                        env,
                        timeout_sec=30.0,
                    )
                    time.sleep(3.0)
                    self.refresh_env()
                    # refresh_env creates a new wrapper. Explicitly rebroadcast
                    # its selected callback port even when relaunch_count did
                    # not change, which is the common stale-service case.
                    if hasattr(self.env, "_configure_grpc"):
                        self.env._configure_grpc()
                except Exception as recovery_error:
                    print(
                        f"a11y recovery attempt {attempt} failed: {recovery_error}",
                        flush=True,
                    )
                else:
                    print(
                        f"a11y tree unavailable; rebound forwarder (attempt {attempt}/2)",
                        flush=True,
                    )
        assert last_error is not None
        raise last_error

    aw_controller_lib.AndroidWorldController.get_a11y_forest = _get_a11y_forest_with_recovery


_patch_androidworld_a11y_tree_recovery()


def configure_androidworld_a11y_method(a11y_method: str, adb_server_port: int) -> None:
    normalized = (a11y_method or "a11y_forwarder_app").strip().lower()
    if normalized not in {"a11y_forwarder_app", "uiautomator"}:
        raise ValueError(
            "androidworld_a11y_method must be 'a11y_forwarder_app' or 'uiautomator', "
            f"got {a11y_method!r}."
        )

    method = (
        aw_controller_lib.A11yMethod.UIAUTOMATOR
        if normalized == "uiautomator"
        else aw_controller_lib.A11yMethod.A11Y_FORWARDER_APP
    )

    def _get_controller_with_port(
        console_port: int = 5554,
        adb_path: str = aw_controller_lib.DEFAULT_ADB_PATH,
        grpc_port: int = 8554,
    ) -> aw_controller_lib.AndroidWorldController:
        config = android_env_config_classes.AndroidEnvConfig(
            task=android_env_config_classes.FilesystemTaskConfig(
                path=aw_controller_lib._write_default_task_proto()
            ),
            simulator=android_env_config_classes.EmulatorConfig(
                emulator_launcher=android_env_config_classes.EmulatorLauncherConfig(
                    emulator_console_port=console_port,
                    adb_port=console_port + 1,
                    grpc_port=grpc_port,
                ),
                adb_controller=android_env_config_classes.AdbControllerConfig(
                    adb_path=adb_path,
                    adb_server_port=adb_server_port,
                ),
            ),
        )
        android_env_instance = android_env_loader.load(config)
        return aw_controller_lib.AndroidWorldController(
            android_env_instance,
            a11y_method=method,
        )

    aw_controller_lib.get_controller = _get_controller_with_port


def androidworld_a11y_method_for_mode(defaults: DefaultsConfig, mode: str) -> str:
    """Resolve a mode-specific a11y method without changing other lanes."""
    override = getattr(defaults.benchmark, f"androidworld_{mode}_a11y_method", None)
    return (override or defaults.benchmark.androidworld_a11y_method).strip().lower()


def _create_androidworld_suite(
    *,
    suite_family: str,
    task_random_seed: int | None,
    n_task_combinations: int,
    tasks: Sequence[str] | None,
    fixed_task_seed: bool,
):
    task_registry = registry.TaskRegistry()
    suite = suite_utils.create_suite(
        task_registry.get_registry(family=suite_family),
        n_task_combinations=n_task_combinations,
        seed=task_random_seed,
        tasks=list(tasks) if tasks else None,
        use_identical_params=fixed_task_seed,
    )
    suite.suite_family = suite_family
    return suite


def _suite_goal_to_apps(suite) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    goal_to_apps: dict[str, tuple[str, ...]] = {}
    app_names: list[str] = []
    for task_instances in suite.values():
        for task in task_instances:
            goal_to_apps[task.goal] = tuple(getattr(task, "app_names", ()) or ())
            for app_name in goal_to_apps[task.goal]:
                if app_name and app_name not in app_names:
                    app_names.append(app_name)
    return goal_to_apps, app_names


def _build_mobileuse_agent_with_androidworld_env(
    *,
    agent_params: dict,
    androidworld_env_wrapper: AndroidWorldBackedMobileUseEnvironment,
):
    import mobile_use.agents.base as mobile_use_base

    with mock.patch.object(
        mobile_use_base,
        "Environment",
        new=lambda **kwargs: androidworld_env_wrapper,
    ):
        agent_core = mobile_use.Agent.from_params(agent_params)
    agent_core.env = androidworld_env_wrapper
    return agent_core


def run_androidworld(
    *,
    mode: str,
    defaults: DefaultsConfig,
    serial_no: str | None,
    host: str,
    adb_server_port: int,
    adb_path: str | None,
    console_port: int,
    grpc_port: int,
    perform_emulator_setup: bool,
    suite_family: str,
    tasks: Sequence[str] | None,
    task_random_seed: int | None,
    n_task_combinations: int | None,
    fixed_task_seed: bool | None,
    run_output_dir: str | Path,
    agent_log_dir: str | Path,
    explored_knowledge_path: str | None = None,
) -> Path:
    task_random_seed = defaults.control.task_random_seed if task_random_seed is None else task_random_seed
    n_task_combinations = defaults.control.n_task_combinations if n_task_combinations is None else n_task_combinations
    fixed_task_seed = defaults.control.fixed_task_seed if fixed_task_seed is None else fixed_task_seed
    suite_family = suite_family or defaults.benchmark.androidworld_suite_family

    adb_path = adb_path or find_adb_path()
    # AndroidEnv has a few fallback paths that invoke adb by name. Keep those
    # subprocesses on the same per-lane server as the explicit controller.
    os.environ["ANDROID_ADB_SERVER_PORT"] = str(adb_server_port)
    if host:
        os.environ["ANDROID_ADB_SERVER_HOST"] = host
    else:
        os.environ.pop("ANDROID_ADB_SERVER_HOST", None)
    a11y_host_port_base = androidworld_a11y_host_port_base(console_port)
    os.environ[_A11Y_HOST_PORT_BASE_ENV] = str(a11y_host_port_base)
    print(f"[androidworld] adb_path={adb_path}", flush=True)
    ensure_adb_server(adb_path, adb_server_port)
    a11y_method = androidworld_a11y_method_for_mode(defaults, mode)
    configure_androidworld_a11y_method(a11y_method, adb_server_port)
    run_max_steps = defaults.control.max_steps if defaults.control.max_steps and defaults.control.max_steps > 0 else None
    print(
        "[androidworld] "
        f"a11y_method={a11y_method} "
        f"a11y_host_port_range={a11y_host_port_base}-{a11y_host_port_base + 31} "
        f"freeze_datetime={defaults.benchmark.androidworld_freeze_datetime} "
        f"max_steps={run_max_steps}",
        flush=True,
    )
    run_output_dir = Path(run_output_dir)
    agent_log_dir = Path(agent_log_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    agent_log_dir.mkdir(parents=True, exist_ok=True)

    task_ids = list(tasks) if tasks else None
    serial_no = serial_no or f"emulator-{console_port}"
    print(f"[androidworld] ensuring emulator serial={serial_no} mode={mode}", flush=True)
    launched_emulator = ensure_androidworld_emulator(
        mode=mode,
        defaults=defaults,
        adb_path=adb_path,
        host=host,
        adb_server_port=adb_server_port,
        serial_no=serial_no,
        console_port=console_port,
        grpc_port=grpc_port,
    )
    print(f"[androidworld] emulator_launched={launched_emulator is not None}", flush=True)
    wait_for_adb_device_ready(
        adb_path=adb_path,
        host=host,
        adb_server_port=adb_server_port,
        serial_no=serial_no,
        timeout_seconds=defaults.benchmark.androidworld_device_ready_timeout_seconds,
    )
    print("[androidworld] device_ready", flush=True)
    wait_for_android_services_ready(
        adb_path=adb_path,
        host=host,
        adb_server_port=adb_server_port,
        serial_no=serial_no,
        timeout_seconds=max(defaults.benchmark.androidworld_device_ready_timeout_seconds, 240),
        require_uiautomator=a11y_method == "uiautomator",
    )
    print("[androidworld] services_ready", flush=True)
    if a11y_method == "a11y_forwarder_app":
        ensure_androidworld_a11y_forwarder_ready(
            adb_path=adb_path,
            host=host,
            adb_server_port=adb_server_port,
            serial_no=serial_no,
        )
        print("[androidworld] a11y_forwarder_ready", flush=True)
    print(
        "[androidworld] loading_env "
        f"perform_setup={perform_emulator_setup} "
        "task_setup_count=default "
        f"task_ids={task_ids or 'all'}",
        flush=True,
    )
    raw_env = env_launcher.load_and_setup_env(
        console_port=console_port,
        emulator_setup=perform_emulator_setup,
        freeze_datetime=defaults.benchmark.androidworld_freeze_datetime,
        adb_path=adb_path,
        grpc_port=grpc_port,
    )
    # Keep the evaluator on a single settled post-agent state. Coordinate
    # repair remains disabled here: P1/P3/P4 are coordinate-sensitive study
    # conditions and must expose the model's actual Android-pixel action.
    env = AndroidWorldGuardedEnv(raw_env, repair_coordinates=False)
    print("[androidworld] env_loaded", flush=True)

    print("[androidworld] creating_suite", flush=True)
    suite = _create_androidworld_suite(
        suite_family=suite_family,
        task_random_seed=task_random_seed,
        n_task_combinations=n_task_combinations,
        tasks=task_ids,
        fixed_task_seed=fixed_task_seed,
    )
    goal_to_apps, knowledge_app_names = _suite_goal_to_apps(suite)
    suite_instance_count = sum(len(task_instances) for task_instances in suite.values())
    print(
        f"[androidworld] suite_ready tasks={len(goal_to_apps)} apps={knowledge_app_names}",
        flush=True,
    )
    if suite_instance_count != len(goal_to_apps):
        print(
            "[androidworld] suite_task_count "
            f"instances={suite_instance_count} unique_goals={len(goal_to_apps)}",
            flush=True,
        )
    if (
        mode == "p4"
        and defaults.benchmark.androidworld_p4_proactive_exploration
        and explored_knowledge_path
        and knowledge_app_names
    ):
        print(
            f"[androidworld] p4_proactive_exploration_start apps={knowledge_app_names}",
            flush=True,
        )
        exploration_controller = AndroidWorldBackedAppAgentController(
            benchmark_env=env,
            wait_after_action_seconds=defaults.executor.wait_after_action_seconds,
        )
        package_names = {
            app_name: package_name
            for app_name in knowledge_app_names
            if (
                package_name := resolve_package_name(
                    benchmark="androidworld",
                    app_name=app_name,
                )
            )
        }
        ensure_mobileuse_explored_knowledge(
            controller=exploration_controller,
            app_names=knowledge_app_names,
            knowledge_path=explored_knowledge_path,
            package_names=package_names,
            artifact_root=agent_log_dir / "_knowledge_artifacts",
        )
        print(
            f"[androidworld] p4_proactive_exploration_done path={explored_knowledge_path}",
            flush=True,
        )
    elif mode == "p4":
        print("[androidworld] p4_proactive_exploration_disabled", flush=True)

    def _build_androidworld_backed_mobileuse(mode_name: str, agent_name: str) -> StudyMobileUseAgent:
        agent_params = build_mobileuse_agent_config(
            mode=mode_name,
            cfg=defaults,
            serial_no=serial_no,
            host=host,
            port=adb_server_port,
            log_dir=agent_log_dir,
            explored_knowledge_path=explored_knowledge_path,
        )
        androidworld_env_wrapper = AndroidWorldBackedMobileUseEnvironment(
            benchmark_env=env,
            serial_no=serial_no,
            host=host,
            port=adb_server_port,
            wait_after_action_seconds=defaults.executor.wait_after_action_seconds,
            include_a11y_tree=defaults.observation.include_a11y_tree,
            # P1 remains screenshot-only for model input.  The DOM/XML is
            # retained on the state object solely for offline evidence export.
            include_dom_tree=(mode_name in {"p1", "p4"}),
            include_device_time=defaults.observation.include_device_time,
        )
        agent_core = _build_mobileuse_agent_with_androidworld_env(
            agent_params=dict(agent_params),
            androidworld_env_wrapper=androidworld_env_wrapper,
        )
        print(f"[androidworld] mobileuse_agent_ready mode={mode_name}", flush=True)
        return StudyMobileUseAgent(
            env,
            agent_core,
            name=agent_name,
            timeout_seconds=defaults.executor.timeout_seconds,
        )

    try:
        if mode == "p1":
            print("[androidworld] building_p1_reactive_single_agent", flush=True)
            agent = _build_androidworld_backed_mobileuse(
                "p1",
                "study_p1_reactive_androidworld",
            )
        elif mode == "p4":
            print("[androidworld] building_p4_multiagent_reflection", flush=True)
            agent = _build_androidworld_backed_mobileuse(
                "p4",
                "study_p4_multiagent_androidworld",
            )
        elif mode == "p2":
            print("[androidworld] building_p2_appagent_deploy_agent", flush=True)
            appagent_controller = AndroidWorldBackedAppAgentController(
                benchmark_env=env,
                wait_after_action_seconds=defaults.executor.wait_after_action_seconds,
            )

            def _session_factory(goal: str):
                app_names_for_goal = goal_to_apps.get(goal, ())
                app_name = app_names_for_goal[0] if app_names_for_goal else None
                session = AppAgentSession(
                    controller=appagent_controller,
                    cfg=defaults,
                    task_root_dir=agent_log_dir,
                    explored_knowledge_path=explored_knowledge_path,
                    app_name=app_name,
                    package_name=None,
                )
                session.reset(goal)
                return session

            agent = StudyRuntimeAgent(
                env,
                _session_factory,
                name="study_p2_appagent_androidworld",
                timeout_seconds=defaults.executor.timeout_seconds,
            )
        elif mode == "p3":
            from android_world.agents import infer_ma3
            from android_world.agents import mobile_agent_v3
            from study.gpt5_1_adapter import (
                GPTAbsoluteAdapter,
                GPTAbsoluteMobileAgentV3Wrapper,
                GPTAdapter,
                GPTMobileAgentV3Wrapper,
            )
            from study.gpt5_6_adapter import (
                GPT56AbsoluteAdapter,
                GPT56MobileAgentV3Wrapper,
            )
            from study.model_adapters import (
                GeminiAdapter,
                GeminiMobileAgentV3Wrapper,
            )
            from study.glm_adapter import GLMAdapter, GLMMobileAgentV3Wrapper
            from study.claude_adapter import ClaudeAbsoluteAdapter, ClaudeMobileAgentV3Wrapper

            coor_type = defaults.benchmark.androidworld_p3_coor_type or "qwen-vl"
            print(
                f"[androidworld] building_p3_mobileagent_v3 coor_type={coor_type}",
                flush=True,
            )
            model_cfg = get_mode_model_config(defaults, "p3_decision")
            vllm = infer_ma3.GUIOwlWrapper(
                model_cfg.api_key,
                model_cfg.base_url,
                model_cfg.model_name,
                max_retry=model_cfg.max_retry,
                temperature=model_cfg.temperature,
                retry_waiting_seconds=model_cfg.retry_waiting_seconds,
            )
            # GUIOwlWrapper has a conservative constructor default; the study
            # config is authoritative for every provider, not only GPT.
            vllm.max_tokens = model_cfg.max_tokens
            if model_cfg.adapter_name.lower() == "gemini":
                vllm = GeminiMobileAgentV3Wrapper(vllm, GeminiAdapter())
            elif model_cfg.adapter_name.lower() == "glm":
                vllm = GLMMobileAgentV3Wrapper(vllm, GLMAdapter())
            elif model_cfg.adapter_name.lower() == "gpt":
                vllm.max_tokens_parameter = "max_completion_tokens"
                vllm.max_tokens = model_cfg.max_tokens
                vllm = GPTMobileAgentV3Wrapper(vllm, GPTAdapter())
            elif model_cfg.adapter_name.lower() == "gpt-absolute":
                vllm.max_tokens_parameter = "max_completion_tokens"
                vllm.max_tokens = model_cfg.max_tokens
                vllm = GPTAbsoluteMobileAgentV3Wrapper(
                    vllm, GPTAbsoluteAdapter()
                )
            elif model_cfg.adapter_name.lower() == "gpt-5.6-absolute":
                vllm.max_tokens_parameter = "max_completion_tokens"
                vllm.max_tokens = model_cfg.max_tokens
                vllm = GPT56MobileAgentV3Wrapper(
                    vllm, GPT56AbsoluteAdapter()
                )
            elif model_cfg.adapter_name.lower() == "claude-absolute":
                vllm = ClaudeMobileAgentV3Wrapper(
                    vllm, ClaudeAbsoluteAdapter()
                )
            raw_p3_agent = mobile_agent_v3.MobileAgentV3_M3A(
                env,
                vllm,
                name="study_p3_mobileagentv3_androidworld",
                wait_after_action_seconds=defaults.executor.wait_after_action_seconds,
                output_path=str(agent_log_dir / "mobileagentv3_traj"),
                coor_type=coor_type,
                max_action_retry=defaults.control.max_action_retry,
            )
            if hasattr(raw_p3_agent, "get_task_name"):
                raw_p3_agent.get_task_name(suite)
            raw_p3_agent.transition_pause = None
            agent = OutputRecordingMobileAgentV3(
                raw_p3_agent,
                name="study_p3_mobileagentv3_androidworld",
            )
        else:
            raise ValueError(f"Unsupported mode={mode!r} for AndroidWorld runner.")

        if not suite_family.startswith("miniwob"):
            agent.transition_pause = None

        checkpoint_dir = checkpointer_lib.create_run_directory(str(run_output_dir))
        print(f"[androidworld] checkpoint_dir={checkpoint_dir}", flush=True)
        suite_utils.run(
            suite,
            agent,
            checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
            demo_mode=False,
            max_steps=run_max_steps,
        )
        print("[androidworld] suite_run_done", flush=True)
        return Path(checkpoint_dir)
    finally:
        try:
            env.close()
        finally:
            stop_androidworld_emulator(
                adb_path=adb_path,
                host=host,
                adb_server_port=adb_server_port,
                serial_no=serial_no,
                process=launched_emulator,
            )
