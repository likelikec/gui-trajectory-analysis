"""ADB-backed device bridge used outside benchmark-native executors."""

from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import time

import cv2


class AdbDeviceBridge:
    """Thin controller wrapper around `adb -s <serial>` commands."""

    def __init__(
        self,
        serial_no: str,
        adb_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        screenshot_dir: str = "/sdcard",
        xml_dir: str = "/sdcard",
    ):
        self.device = serial_no
        self.adb_path = adb_path or "adb"
        self.host = host
        self.port = port
        self.screenshot_dir = screenshot_dir
        self.xml_dir = xml_dir
        self.backslash = "\\"
        self.width, self.height = self.get_device_size()
        self.viewport_size = (self.width, self.height)

    def _adb_prefix(self) -> str:
        parts = [f'"{self.adb_path}"']
        if self.host:
            parts.extend(["-H", self.host])
        if self.port:
            parts.extend(["-P", str(self.port)])
        parts.extend(["-s", self.device])
        return " ".join(parts)

    def execute_adb(self, adb_command: str, output: bool = True) -> str:
        result = subprocess.run(
            adb_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        if output:
            print(result.stderr.strip())
        return "ERROR"

    def run_command(self, command: str) -> str:
        if command.lstrip().startswith("adb "):
            command = command.replace("adb", self._adb_prefix(), 1)
        return self.execute_adb(command)

    def get_device_size(self) -> tuple[int, int]:
        output = self.execute_adb(f"{self._adb_prefix()} shell wm size")
        if output == "ERROR":
            return 0, 0
        resolution = output.split(":")[-1].strip()
        width, height = resolution.split("x")
        return int(width), int(height)

    def get_screenshot(self, prefix: str, save_dir: str | Path) -> str:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        remote_path = os.path.join(self.screenshot_dir, prefix + ".png").replace(self.backslash, "/")
        local_path = save_dir / f"{prefix}.png"
        cap_command = f"{self._adb_prefix()} shell screencap -p {remote_path}"
        pull_command = f'{self._adb_prefix()} pull {remote_path} "{local_path}"'
        for _ in range(5):
            if self.execute_adb(cap_command, output=False) == "ERROR":
                time.sleep(1)
                continue
            if self.execute_adb(pull_command, output=False) == "ERROR":
                time.sleep(1)
                continue
            if local_path.exists() and local_path.stat().st_size > 8:
                with local_path.open("rb") as handle:
                    has_png_header = handle.read(8) == b"\x89PNG\r\n\x1a\n"
                if has_png_header and cv2.imread(str(local_path)) is not None:
                    return str(local_path)
            time.sleep(1)
        return "ERROR"

    def save_screenshot(self, save_path: str | Path) -> str:
        save_path = Path(save_path)
        return self.get_screenshot(save_path.stem, save_path.parent)

    def get_xml(self, prefix: str, save_dir: str | Path) -> str:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        remote_path = os.path.join(self.xml_dir, prefix + ".xml").replace(self.backslash, "/")
        local_path = save_dir / f"{prefix}.xml"
        dump_command = f"{self._adb_prefix()} shell uiautomator dump {remote_path}"
        pull_command = f'{self._adb_prefix()} pull {remote_path} "{local_path}"'
        for _ in range(5):
            if self.execute_adb(dump_command, output=False) == "ERROR":
                time.sleep(1)
                continue
            if self.execute_adb(pull_command, output=False) == "ERROR":
                time.sleep(1)
                continue
            if local_path.exists() and local_path.stat().st_size > 0:
                return str(local_path)
            time.sleep(1)
        return "ERROR"

    def get_current_activity(self) -> str:
        output = self.execute_adb(f"{self._adb_prefix()} shell dumpsys window windows", output=False)
        if output == "ERROR":
            return ""
        match = re.search(r"mCurrentFocus.+? ([^ ]+/[^ }]+)", output)
        if match:
            return match.group(1)
        return ""

    def tap(self, x: int | float, y: int | float) -> str:
        return self.execute_adb(f"{self._adb_prefix()} shell input tap {int(x)} {int(y)}")

    def text(self, input_str: str) -> str:
        escaped = input_str.replace(" ", "%s").replace('"', "").replace("'", "")
        return self.execute_adb(f"{self._adb_prefix()} shell input text {escaped}")

    def long_press(self, x: int | float, y: int | float, duration: int = 1000) -> str:
        return self.execute_adb(
            f"{self._adb_prefix()} shell input swipe {int(x)} {int(y)} {int(x)} {int(y)} {duration}"
        )

    def swipe(self, x: int | float, y: int | float, direction: str, dist: str | int = "medium", quick: bool = False) -> str:
        if isinstance(dist, str):
            unit_dist = int(self.width / 10)
            if dist == "long":
                unit_dist *= 3
            elif dist == "medium":
                unit_dist *= 2
        else:
            unit_dist = int(dist)
        offsets = {
            "up": (0, -2 * unit_dist),
            "down": (0, 2 * unit_dist),
            "left": (-1 * unit_dist, 0),
            "right": (unit_dist, 0),
        }
        if direction not in offsets:
            return "ERROR"
        dx, dy = offsets[direction]
        duration = 100 if quick else 400
        return self.execute_adb(
            f"{self._adb_prefix()} shell input swipe {int(x)} {int(y)} {int(x + dx)} {int(y + dy)} {duration}"
        )

    def swipe_precise(self, start: tuple[int, int], end: tuple[int, int], duration: int = 500) -> str:
        start_x, start_y = start
        end_x, end_y = end
        return self.execute_adb(
            f"{self._adb_prefix()} shell input swipe {int(start_x)} {int(start_y)} {int(end_x)} {int(end_y)} {duration}"
        )

    def back(self) -> str:
        return self.execute_adb(f"{self._adb_prefix()} shell input keyevent KEYCODE_BACK")

    def home(self) -> str:
        return self.execute_adb(f"{self._adb_prefix()} shell input keyevent KEYCODE_HOME")

    def enter(self) -> str:
        return self.execute_adb(f"{self._adb_prefix()} shell input keyevent KEYCODE_ENTER")

    def launch_app(self, package_name: str) -> str:
        return self.execute_adb(
            f"{self._adb_prefix()} shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
        )

    def launch(self, package_name: str) -> str:
        return self.launch_app(package_name)
