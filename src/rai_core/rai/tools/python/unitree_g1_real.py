# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import pty
import select
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from rai.tools.python.unitree_g1_sim import (
    ADJUST_ACTIONS,
    COMPLIANCE_ACTIONS,
    DEFAULT_REPLAY_ALIASES,
    INTERFACE_KEYS,
    KEYBOARD_ACTIONS,
    CONTROL_MODES,
    TOOL_STATE_RULES,
    ToolStateError,
    UnitreeG1SimReplayPlayer,
    UnitreeG1RuntimeState,
    _as_bool,
    _configured_skill_names,
    _launch_terminal_tail,
    _normalize_name,
    _normalize_skill_name,
    _resolve_replay_path,
    _skill_catalog,
    _tool_state_rules_text,
)


DEFAULT_UNITREE_G1_REAL_TOOLS = [
    "confirm_deployment",
    "perform_skill",
    "perform_replay",
    "list_skills",
    "list_replays",
    "start_control",
    "switch_zmq",
    "switch_keyboard",
    "toggle_zmq_streaming",
    "toggle_planner",
    "keyboard",
    "select_mode",
    "adjust",
    "compliance",
]

DEFAULT_UNITREE_G1_REAL_SKILLS = [
    {
        "name": "wave left hand",
        "source": "replay",
        "file": "/home/zj/Agent/data/2026-05-23-13-59-19-filtered/data/chunk-000/episode_000000.parquet",
        "aliases": ["wave", "left hand wave", "wave_left_hand"],
        "description": "Wave the left hand.",
    },
    {
        "name": "wave left hand with vla",
        "source": "vla",
        "prompt": "Walk forward, grab the cola and throw into the trash bin",
        "server_root": "/home/zj/Isaac-GR00T",
        "model_path": (
            "/home/zj/Isaac-GR00T/test_ckpt/"
            "gr00t-freeze-vlm-half-expert-param-cola-644-binary-hand-bs96-0705/"
            "checkpoint-60000"
        ),
        "aliases": [],
        "description": "Run the configured VLA policy through GR00T server and WBC inference.",
    },
]


def _default_log_dir() -> Path:
    return Path(os.getenv("UNITREE_G1_REAL_LOG_DIR", "logs/unitree_g1_real"))


def _resolve_replay_file(replay_dir: str, action: str) -> Path:
    replay_root = Path(replay_dir or "replays/unitree_g1_sim").expanduser().resolve()
    normalized = _normalize_name(action)
    filename = DEFAULT_REPLAY_ALIASES.get(normalized, action)
    path = Path(filename)
    if path.is_absolute():
        default_path = replay_root / path.name
        path = default_path if default_path.exists() else path.expanduser()
    else:
        path = replay_root / path
    if path.suffix not in {".npy", ".parquet"}:
        path = path.with_suffix(".npy")
    return path


class UnitreeG1RealManager:
    """Owns the GR00T WBC real-robot manager process and sends terminal keys."""

    def __init__(
        self,
        deploy_dir: str = "",
        *,
        terminal_viewer: bool = False,
        log_dir: str | None = None,
    ):
        self.deploy_dir = deploy_dir
        self.terminal_viewer = terminal_viewer
        self.log_dir = Path(log_dir).expanduser() if log_dir else _default_log_dir()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._logs: deque[str] = deque(maxlen=240)
        self._started_at: float | None = None
        self._log_path: Path | None = None
        self.state = UnitreeG1RuntimeState("real")

    def configure(self, *, terminal_viewer: bool, log_dir: str | None = None) -> None:
        self.terminal_viewer = terminal_viewer
        if log_dir:
            self.log_dir = Path(log_dir).expanduser()

    def start(self, deploy_dir: str = "", extra_args: str = "") -> str:
        with self._lock:
            if self.is_running():
                return self.status()

            cwd = Path(
                deploy_dir
                or self.deploy_dir
                or os.getenv("UNITREE_G1_REAL_DEPLOY_DIR", "")
            )
            if not str(cwd):
                raise RuntimeError(
                    "Missing GR00T deployment directory. Set UNITREE_G1_REAL_DEPLOY_DIR "
                    "or pass deploy_dir to the start tool."
                )
            cwd = cwd.expanduser().resolve()
            deploy_script = cwd / "deploy.sh"
            if not deploy_script.exists():
                raise RuntimeError(f"deploy.sh was not found in {cwd}")
            setup_script = cwd / "scripts" / "setup_env.sh"
            if not setup_script.exists():
                raise RuntimeError(f"scripts/setup_env.sh was not found in {cwd}")

            deploy_command = [
                "./deploy.sh",
                "--input-type",
                "manager",
                "--zmq-host",
                "localhost",
                "--hand-type",
                "inspire",
            ]
            if extra_args.strip():
                deploy_command.extend(shlex.split(extra_args))
            deploy_command.append("real")
            command = [
                "bash",
                "-lc",
                "source scripts/setup_env.sh && " + shlex.join(deploy_command),
            ]

            master_fd, slave_fd = pty.openpty()
            self._logs.clear()
            self._log_path = self.log_dir / "manager.log"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text("", encoding="utf-8")
            self._append_log(f"$ cd {cwd}")
            self._append_log("$ " + command[-1])
            terminal_message = ""
            if self.terminal_viewer:
                try:
                    terminal_message = "\n" + _launch_terminal_tail(
                        self._log_path,
                        "Unitree G1 Real Manager",
                    )
                except Exception as exc:
                    terminal_message = f"\nCould not open terminal viewer: {exc}"

            self._process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
            )
            os.close(slave_fd)
            self._master_fd = master_fd
            self._started_at = time.time()
            self._reader_thread = threading.Thread(
                target=self._read_pty_output,
                name="unitree-g1-real-manager-reader",
                daemon=True,
            )
            self._reader_thread.start()
            self.state.reset_for_start()
            return (
                "Started GR00T WBC real manager: "
                f"pid={self._process.pid}, command={command[-1]}."
                f"{terminal_message}"
            )

    def ensure_started(
        self,
        deploy_dir: str = "",
        extra_args: str = "",
        confirm_deployment: bool = False,
        start_control: bool = False,
        settle_seconds: float = 2.0,
        init_done_timeout: float = 60.0,
    ) -> str:
        messages = [self.start(deploy_dir=deploy_dir, extra_args=extra_args)]
        if settle_seconds > 0:
            time.sleep(min(float(settle_seconds), 10.0))
        if confirm_deployment:
            self._require_running()
            self.state.update(deployment="waiting_init_done", last_error="")
            self._send_key("y")
            self._send_key("\n")
            messages.append("Sent deployment confirmation: Y")
            try:
                messages.append(self.wait_for_output("Init Done", timeout=init_done_timeout))
                self.state.update(deployment="ready", last_error="")
            except Exception as exc:
                self.state.update(deployment="failed", last_error=str(exc))
                raise
        if start_control:
            messages.append(self.send_key("]", "start_control"))
            self.state.update(control_mode="keyboard_normal")
        return "\n".join(messages)

    def stop(self, graceful: bool = True, timeout: float = 5.0) -> str:
        with self._lock:
            process = self._process
            if process is None:
                return "GR00T WBC real manager is not running."

            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2.0)

            return_code = process.returncode
            self._close_master_fd()
            self._process = None
            self._started_at = None
            return f"Stopped GR00T WBC real manager. Return code: {return_code}"

    def status(self, tail_lines: int = 20) -> str:
        with self._lock:
            running = self.is_running()
            pid = self._process.pid if self._process is not None else None
            uptime = time.time() - self._started_at if running and self._started_at else 0.0
            logs = self.tail_logs(tail_lines)
            return (
                f"running={running}, pid={pid}, uptime_seconds={uptime:.1f}, "
                f"deploy_dir={self.deploy_dir or os.getenv('UNITREE_G1_REAL_DEPLOY_DIR', '')}\n"
                f"Recent logs:\n{logs}"
            )

    def send_key(self, key: str, label: str, repeats: int = 1, interval: float = 0.15) -> str:
        safe_repeats = max(1, min(int(repeats), 20))
        safe_interval = max(0.0, min(float(interval), 2.0))
        with self._lock:
            self._require_running()
            for index in range(safe_repeats):
                self._send_key(key)
                if index < safe_repeats - 1:
                    time.sleep(safe_interval)
            return f"Sent {label}: key={key!r}, repeats={safe_repeats}"

    def confirm_deployment(self, timeout: float = 60.0) -> str:
        with self._lock:
            self._require_running()
            self.state.update(deployment="waiting_init_done", last_error="")
            self._send_key("y")
            self._send_key("\n")
        try:
            detected = self.wait_for_output("Init Done", timeout=timeout)
            self.state.update(deployment="ready", last_error="")
            return "\n".join(
                [
                    "Sent deployment confirmation: Y",
                    detected,
                    "Deployment完成. Current control_mode remains pre_control.",
                    "Next step before replay or VLA skills: call unitree_g1_real_start_control to send ']'.",
                    self.state.summary(),
                ]
            )
        except Exception as exc:
            self.state.update(deployment="failed", last_error=str(exc))
            raise

    def wait_for_output(self, text: str, timeout: float = 60.0) -> str:
        deadline = time.time() + max(1.0, float(timeout))
        while time.time() < deadline:
            with self._lock:
                self._require_running()
                if any(text in line for line in self._logs):
                    return f"Detected manager output: {text!r}"
            time.sleep(0.2)
        raise RuntimeError(
            f"Timed out waiting for manager output {text!r}. Recent logs:\n"
            f"{self.tail_logs()}"
        )

    def prepare_replay_streaming(self) -> str:
        with self._lock:
            self._require_running()
            self._send_key("]")
            time.sleep(1.0)
            self._send_key("#")
            time.sleep(0.2)
            self._send_key("\n")
            self.state.update(control_mode="zmq_streaming", last_error="")
            return (
                "Prepared replay streaming: sent ']', waited 1.0s, "
                "then sent '#' and ENTER."
            )

    def prepare_zmq_streaming(self) -> str:
        with self._lock:
            self._require_running()
            self._send_key("#")
            time.sleep(0.2)
            self._send_key("\n")
            self.state.update(control_mode="zmq_streaming", last_error="")
            return "Switched manager to ZMQ interface and sent ENTER to toggle streaming."

    def switch_to_zmq(self) -> str:
        message = self.send_key("#", "switch_interface:zmq")
        self.state.update(control_mode="zmq", last_error="")
        return f"{message}\n{self.state.summary()}"

    def toggle_zmq_streaming(self) -> str:
        message = self.send_key("\n", "toggle_zmq_streaming")
        current = self.state.snapshot()["control_mode"]
        next_mode = "zmq" if current == "zmq_streaming" else "zmq_streaming"
        self.state.update(control_mode=next_mode, last_error="")
        status = "DISABLED" if next_mode == "zmq" else "ENABLED"
        return (
            f"{message}\n"
            f"ZMQ STREAMING MODE: {status}\n"
            f"{self.state.summary()}"
        )

    def switch_to_keyboard(self) -> str:
        message = self.send_key("!", "switch_interface:keyboard")
        self.state.update(control_mode="keyboard_normal", last_error="")
        return f"{message}\n{self.state.summary()}"

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def tail_logs(self, lines: int = 20) -> str:
        safe_lines = max(1, min(int(lines), 80))
        return "\n".join(list(self._logs)[-safe_lines:]) or "(no output captured yet)"

    def _send_key(self, key: str) -> None:
        payload = b"\r" if key == "\n" else key.encode("utf-8")
        if self._master_fd is None:
            raise RuntimeError("GR00T WBC real manager PTY is not open.")
        os.write(self._master_fd, payload)
        self._append_log(f"[rai sent key] {key!r}")

    def _require_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(
                "GR00T WBC real manager is not running. Call unitree_g1_real_start_manager first."
            )

    def _read_pty_output(self) -> None:
        while True:
            with self._lock:
                fd = self._master_fd
                process = self._process
            if fd is None or process is None:
                return
            try:
                readable, _, _ = select.select([fd], [], [], 0.2)
                if not readable:
                    if process.poll() is not None:
                        return
                    continue
                data = os.read(fd, 4096)
            except OSError:
                return
            if not data:
                return
            text = data.decode("utf-8", errors="replace")
            for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
                if line.strip():
                    self._append_log(line)

    def _append_log(self, line: str) -> None:
        self._logs.append(line[-500:])
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def _close_master_fd(self) -> None:
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None


class _LoggedShellProcess:
    def __init__(
        self,
        *,
        name: str,
        log_dir: Path,
        log_name: str,
        terminal_viewer: bool,
    ) -> None:
        self.name = name
        self.log_dir = log_dir
        self.log_name = log_name
        self.terminal_viewer = terminal_viewer
        self._process: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._logs: deque[str] = deque(maxlen=240)
        self._log_path: Path | None = None

    def configure(self, *, log_dir: Path, terminal_viewer: bool) -> None:
        self.log_dir = log_dir
        self.terminal_viewer = terminal_viewer

    def start(self, *, cwd: Path, command: str, title: str) -> str:
        self.stop()
        master_fd, slave_fd = pty.openpty()
        self._logs.clear()
        self._log_path = self.log_dir / self.log_name
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text("", encoding="utf-8")
        self._append_log(f"$ cd {cwd}")
        self._append_log("$ " + command)
        terminal_message = ""
        if self.terminal_viewer:
            try:
                terminal_message = "\n" + _launch_terminal_tail(
                    self._log_path,
                    title,
                )
            except Exception as exc:
                terminal_message = f"\nCould not open terminal viewer: {exc}"
        self._process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(cwd),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
        )
        os.close(slave_fd)
        self._master_fd = master_fd
        self._reader_thread = threading.Thread(
            target=self._read_pty_output,
            name=f"{self.name}-reader",
            daemon=True,
        )
        self._reader_thread.start()
        return f"Started {self.name}: pid={self._process.pid}.{terminal_message}"

    def stop(self) -> str:
        process = self._process
        if process is None:
            return f"{self.name} is not running."
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2.0)
        return_code = process.returncode
        self._close_master_fd()
        self._process = None
        return f"Stopped {self.name}. Return code: {return_code}"

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def tail_logs(self, lines: int = 20) -> str:
        safe_lines = max(1, min(int(lines), 80))
        return "\n".join(list(self._logs)[-safe_lines:]) or "(no output captured yet)"

    def _read_pty_output(self) -> None:
        while True:
            fd = self._master_fd
            process = self._process
            if fd is None or process is None:
                return
            try:
                readable, _, _ = select.select([fd], [], [], 0.2)
                if not readable:
                    if process.poll() is not None:
                        return
                    continue
                data = os.read(fd, 4096)
            except OSError:
                return
            if not data:
                return
            text = data.decode("utf-8", errors="replace")
            for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
                if line.strip():
                    self._append_log(line)

    def _append_log(self, line: str) -> None:
        self._logs.append(line[-500:])
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def _close_master_fd(self) -> None:
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None


class UnitreeG1RealVLARunner:
    def __init__(
        self,
        *,
        terminal_viewer: bool = False,
        log_dir: str | None = None,
    ) -> None:
        self.terminal_viewer = terminal_viewer
        self.log_dir = Path(log_dir).expanduser() if log_dir else _default_log_dir()
        self._lock = threading.RLock()
        self.server = _LoggedShellProcess(
            name="VLA server",
            log_dir=self.log_dir,
            log_name="vla_server.log",
            terminal_viewer=terminal_viewer,
        )
        self.inference = _LoggedShellProcess(
            name="VLA inference",
            log_dir=self.log_dir,
            log_name="vla_inference.log",
            terminal_viewer=terminal_viewer,
        )

    def configure(self, *, terminal_viewer: bool, log_dir: str | None = None) -> None:
        self.terminal_viewer = terminal_viewer
        if log_dir:
            self.log_dir = Path(log_dir).expanduser()
        self.server.configure(log_dir=self.log_dir, terminal_viewer=terminal_viewer)
        self.inference.configure(log_dir=self.log_dir, terminal_viewer=terminal_viewer)

    def start(
        self,
        *,
        model_path: str,
        prompt: str,
        wbc_root: str,
        server_root: str = "",
        embodiment_tag: str = "NEW_EMBODIMENT",
        device: str = "cuda:0",
        port: int = 5550,
        camera_host: str = "192.168.123.164",
        camera_port: int = 5555,
        action_publish_rate: int = 50,
        action_horizon: int = 50,
        server_startup_seconds: float = 8.0,
    ) -> str:
        if not model_path:
            raise RuntimeError("Missing VLA model_path in the selected skill.")
        if not prompt:
            raise RuntimeError("Missing VLA prompt in the selected skill.")
        server_cwd = Path(
            server_root or os.getenv("ISAAC_GROOT_ROOT", "/home/zj/Isaac-GR00T")
        ).expanduser().resolve()
        wbc_cwd = Path(
            wbc_root or os.getenv("GR00T_WBC_ROOT", "/home/zj/GR00T-WholeBodyControl")
        ).expanduser().resolve()
        if not server_cwd.exists():
            raise RuntimeError(f"Isaac-GR00T root does not exist: {server_cwd}")
        if not wbc_cwd.exists():
            raise RuntimeError(f"GR00T-WholeBodyControl root does not exist: {wbc_cwd}")
        server_script = server_cwd / "gr00t" / "eval" / "run_gr00t_server.py"
        inference_script = wbc_cwd / "gear_sonic" / "scripts" / "run_vla_inference.py"
        if not server_script.exists():
            raise RuntimeError(f"VLA server script does not exist: {server_script}")
        if not inference_script.exists():
            raise RuntimeError(f"VLA inference script does not exist: {inference_script}")

        server_command = (
            "source .venv/bin/activate && "
            "export HF_HUB_OFFLINE=1 && "
            "export TRANSFORMERS_OFFLINE=1 && "
            "export NO_ALBUMENTATIONS_UPDATE=1 && "
            "python gr00t/eval/run_gr00t_server.py "
            f"--model-path {shlex.quote(model_path)} "
            f"--embodiment-tag {shlex.quote(embodiment_tag)} "
            f"--device {shlex.quote(device)} "
            f"--port {int(port)}"
        )
        inference_command = (
            "source .venv_inference/bin/activate && "
            "python gear_sonic/scripts/run_vla_inference.py "
            "--host localhost "
            f"--port {int(port)} "
            f"--embodiment-tag {shlex.quote(embodiment_tag)} "
            f"--prompt {shlex.quote(prompt)} "
            f"--action-publish-rate {int(action_publish_rate)} "
            f"--action-horizon {int(action_horizon)} "
            f"--camera-host {shlex.quote(camera_host)} "
            f"--camera-port {int(camera_port)}"
        )
        with self._lock:
            messages = [
                self.server.start(
                    cwd=server_cwd,
                    command=server_command,
                    title="Unitree G1 VLA Server",
                )
            ]
            time.sleep(max(0.0, min(float(server_startup_seconds), 60.0)))
            messages.append(
                self.inference.start(
                    cwd=wbc_cwd,
                    command=inference_command,
                    title="Unitree G1 VLA Inference",
                )
            )
            messages.append(f"VLA prompt: {prompt}")
            return "\n".join(messages)

    def status(self) -> str:
        return "\n".join(
            [
                f"VLA server running: {self.server.is_running()}",
                f"VLA inference running: {self.inference.is_running()}",
                "Recent VLA server logs:",
                self.server.tail_logs(),
                "Recent VLA inference logs:",
                self.inference.tail_logs(),
            ]
        )


_MANAGERS: dict[str, UnitreeG1RealManager] = {}
_MANAGERS_LOCK = threading.Lock()
_PLAYERS: dict[str, UnitreeG1SimReplayPlayer] = {}
_PLAYERS_LOCK = threading.Lock()
_VLA_RUNNERS: dict[str, UnitreeG1RealVLARunner] = {}
_VLA_RUNNERS_LOCK = threading.Lock()


def _get_manager(
    deploy_dir: str,
    *,
    terminal_viewer: bool | None = None,
    log_dir: str | None = None,
) -> UnitreeG1RealManager:
    key = deploy_dir or os.getenv("UNITREE_G1_REAL_DEPLOY_DIR", "")
    viewer = _as_bool(os.getenv("UNITREE_G1_REAL_TERMINAL_VIEWER"), False)
    if terminal_viewer is not None:
        viewer = terminal_viewer
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = UnitreeG1RealManager(
                key,
                terminal_viewer=viewer,
                log_dir=log_dir,
            )
            _MANAGERS[key] = manager
        else:
            manager.configure(terminal_viewer=viewer, log_dir=log_dir)
        return manager


def _get_player(
    gr00t_root: str,
    *,
    terminal_viewer: bool | None = None,
    log_dir: str | None = None,
) -> UnitreeG1SimReplayPlayer:
    key = gr00t_root or os.getenv("GR00T_WBC_ROOT", "")
    viewer = _as_bool(os.getenv("UNITREE_G1_REAL_TERMINAL_VIEWER"), False)
    if terminal_viewer is not None:
        viewer = terminal_viewer
    with _PLAYERS_LOCK:
        player = _PLAYERS.get(key)
        if player is None:
            player = UnitreeG1SimReplayPlayer(
                key,
                terminal_viewer=viewer,
                log_dir=log_dir or str(_default_log_dir()),
            )
            _PLAYERS[key] = player
        else:
            player.configure(
                terminal_viewer=viewer,
                log_dir=log_dir or str(_default_log_dir()),
            )
        return player


def _get_vla_runner(
    *,
    terminal_viewer: bool | None = None,
    log_dir: str | None = None,
) -> UnitreeG1RealVLARunner:
    key = str(Path(log_dir).expanduser() if log_dir else _default_log_dir())
    viewer = _as_bool(os.getenv("UNITREE_G1_REAL_TERMINAL_VIEWER"), False)
    if terminal_viewer is not None:
        viewer = terminal_viewer
    with _VLA_RUNNERS_LOCK:
        runner = _VLA_RUNNERS.get(key)
        if runner is None:
            runner = UnitreeG1RealVLARunner(
                terminal_viewer=viewer,
                log_dir=log_dir or str(_default_log_dir()),
            )
            _VLA_RUNNERS[key] = runner
        else:
            runner.configure(
                terminal_viewer=viewer,
                log_dir=log_dir or str(_default_log_dir()),
            )
        return runner


def start_unitree_g1_real_manager(
    *,
    deploy_dir: str | None = None,
    extra_args: str = "",
    confirm_deployment: bool = False,
    start_control: bool = False,
    settle_seconds: float = 2.0,
    init_done_timeout: float = 60.0,
    terminal_viewer: bool | None = None,
    log_dir: str | None = None,
) -> str:
    manager = _get_manager(
        deploy_dir or "",
        terminal_viewer=terminal_viewer,
        log_dir=log_dir,
    )
    return manager.ensure_started(
        deploy_dir=deploy_dir or "",
        extra_args=extra_args,
        confirm_deployment=confirm_deployment,
        start_control=start_control,
        settle_seconds=settle_seconds,
        init_done_timeout=init_done_timeout,
    )


def stop_unitree_g1_real_manager(*, deploy_dir: str | None = None) -> str:
    manager = _get_manager(deploy_dir or "")
    return manager.stop(graceful=True)


def get_unitree_g1_real_runtime_prompt(
    *,
    deploy_dir: str | None = None,
    replay_dir: str | None = None,
    skills: list[dict[str, Any]] | None = None,
    enabled_tools: list[str] | None = None,
) -> str:
    configured_replay_dir = replay_dir or os.getenv(
        "UNITREE_G1_REAL_REPLAY_DIR",
        "replays/unitree_g1_sim",
    )
    manager = _get_manager(deploy_dir or "")
    skill_names = _configured_skill_names(
        replay_dir=configured_replay_dir,
        skills=skills or DEFAULT_UNITREE_G1_REAL_SKILLS,
    )
    return "\n\n".join(
        [
            manager.state.summary(skills=skill_names),
            _tool_state_rules_text(enabled_tools),
        ]
    )


def get_unitree_g1_real_tools(
    *,
    deploy_dir: str | None = None,
    gr00t_root: str | None = None,
    replay_dir: str | None = None,
    enabled_tools: list[str] | None = None,
    skills: list[dict[str, Any]] | None = None,
    terminal_viewer: bool | None = None,
    log_dir: str | None = None,
) -> list[BaseTool]:
    configured_deploy_dir = deploy_dir or os.getenv("UNITREE_G1_REAL_DEPLOY_DIR", "")
    configured_gr00t_root = gr00t_root or os.getenv("GR00T_WBC_ROOT", "")
    configured_replay_dir = replay_dir or os.getenv(
        "UNITREE_G1_REAL_REPLAY_DIR",
        "replays/unitree_g1_sim",
    )
    enabled_tool_names = set(enabled_tools or DEFAULT_UNITREE_G1_REAL_TOOLS)
    configured_skills = skills or DEFAULT_UNITREE_G1_REAL_SKILLS
    active_deploy_dir = {"value": configured_deploy_dir}
    skill_names = _configured_skill_names(
        replay_dir=configured_replay_dir,
        skills=configured_skills,
    )

    def manager() -> UnitreeG1RealManager:
        return _get_manager(
            active_deploy_dir["value"],
            terminal_viewer=terminal_viewer,
            log_dir=log_dir,
        )

    def player() -> UnitreeG1SimReplayPlayer:
        return _get_player(
            configured_gr00t_root,
            terminal_viewer=terminal_viewer,
            log_dir=log_dir,
        )

    def vla_runner() -> UnitreeG1RealVLARunner:
        return _get_vla_runner(
            terminal_viewer=terminal_viewer,
            log_dir=log_dir,
        )

    def runtime_state() -> str:
        return manager().state.summary(skills=skill_names)

    def require_tool_state(tool_name: str) -> None:
        rule = TOOL_STATE_RULES[tool_name]
        manager().state.require(
            tool_name,
            deployment=rule["deployment"],
            control_mode=rule["control_mode"],
        )

    @tool
    def unitree_g1_real_start_manager(
        deploy_dir: str = "",
        extra_args: str = "",
    ) -> str:
        """Start GR00T WholeBodyControl manager for the real Unitree G1.

        Runs `source scripts/setup_env.sh && ./deploy.sh --input-type manager
        --zmq-host localhost --hand-type inspire real` from the configured deploy directory.
        `extra_args` are inserted before the final `real` argument.
        """
        target_deploy_dir = deploy_dir or configured_deploy_dir
        active_deploy_dir["value"] = target_deploy_dir
        target_manager = _get_manager(
            target_deploy_dir,
            terminal_viewer=terminal_viewer,
            log_dir=log_dir,
        )
        return target_manager.start(deploy_dir=deploy_dir, extra_args=extra_args)

    @tool
    def unitree_g1_real_stop_manager(graceful: bool = True) -> str:
        """Stop the backend-owned manager process without sending `O`.

        For robot safety, `O` emergency stop is reserved for the human operator
        typing in the visible control terminal.
        """
        return manager().stop(graceful=graceful)

    @tool
    def unitree_g1_real_status(tail_lines: int = 20) -> str:
        """Return real manager process status and recent terminal output."""
        return manager().status(tail_lines=tail_lines)

    @tool
    def unitree_g1_real_confirm_deployment(timeout: float = 60.0) -> str:
        """Confirm `Proceed with deployment` by sending `Y` and ENTER.

        Waits until the manager prints `Init Done`. This sets deployment to
        ready but leaves control_mode as pre_control; call
        unitree_g1_real_start_control next to send `]`.
        """
        require_tool_state("confirm_deployment")
        return manager().confirm_deployment(timeout=timeout)

    @tool
    def unitree_g1_real_list_replays() -> str:
        """List configured high-level replay actions and whether their files exist."""
        require_tool_state("list_replays")
        replay_root = Path(configured_replay_dir).expanduser().resolve()
        lines = [f"Replay directory: {replay_root}"]
        for action_name, filename in sorted(DEFAULT_REPLAY_ALIASES.items()):
            if action_name != Path(filename).stem:
                continue
            path = replay_root / filename
            status = "ready" if path.exists() else "missing"
            lines.append(f"- {action_name}: {path} ({status})")
        return "\n".join([*lines, runtime_state()])

    @tool
    def unitree_g1_real_list_skills() -> str:
        """List configured Unitree G1 real skills exposed to the agent."""
        require_tool_state("list_skills")
        catalog = _skill_catalog(
            replay_dir=configured_replay_dir,
            skills=configured_skills,
        )
        seen: set[str] = set()
        lines = ["Configured Unitree G1 real skills:"]
        for skill in catalog.values():
            name = skill["name"]
            if name in seen:
                continue
            seen.add(name)
            if skill["source"] == "replay":
                path = _resolve_replay_path(configured_replay_dir, skill["file"])
                status = "ready" if path.exists() else "missing"
                lines.append(f"- {name}: replay {path} ({status})")
            elif skill["source"] == "vla":
                model_path = skill.get("model_path", "")
                model_status = "ready" if model_path and Path(model_path).exists() else "missing"
                lines.append(
                    f"- {name}: VLA prompt {skill['prompt']!r}, "
                    f"model_path={model_path!r} ({model_status})"
                )
            else:
                lines.append(f"- {name}: {skill['source']}")
        return "\n".join([*lines, runtime_state()])

    def perform_replay_file(
        replay_file: Path,
        *,
        preparation: str,
        wait: bool,
        timeout: float,
        return_to_keyboard: bool,
    ) -> str:
        real_manager = manager()
        replay_player = player()
        if preparation == "keyboard_to_zmq_streaming":
            manager_message = real_manager.prepare_zmq_streaming()
        elif preparation == "toggle_zmq_streaming":
            manager_message = real_manager.toggle_zmq_streaming()
        else:
            raise ValueError(f"Unsupported replay preparation: {preparation!r}")
        player_message = replay_player.play(
            replay_file,
            gr00t_root=configured_gr00t_root,
            wait=False,
        )
        messages = [manager_message, player_message]
        if wait:
            messages.append(replay_player.wait_for_stopped_and_stop(timeout=timeout))
            if return_to_keyboard and not replay_player.is_running():
                messages.append(real_manager.switch_to_keyboard())
            elif return_to_keyboard:
                messages.append(
                    "Replay player is still running; keyboard mode was not restored yet."
                )
        elif return_to_keyboard:
            def restore_keyboard_after_replay() -> None:
                replay_player.wait_for_stopped_and_stop(timeout=timeout)
                if not replay_player.is_running():
                    real_manager.switch_to_keyboard()

            threading.Thread(
                target=restore_keyboard_after_replay,
                name="unitree-g1-real-return-keyboard",
                daemon=True,
            ).start()
            messages.append("Keyboard mode will be restored after the replay exits.")
        return "\n".join([*messages, runtime_state()])

    @tool
    def unitree_g1_real_perform_skill(
        skill: str,
        wait: bool = True,
        timeout: float = 60.0,
        return_to_keyboard: bool = True,
    ) -> str:
        """Perform a configured real-robot skill by name.

        Replay skills switch to ZMQ (`#`), send ENTER to open ZMQ streaming, then
        run the configured replay player. VLA skills start the configured
        GR00T server and WBC VLA inference client.
        """
        require_tool_state("perform_skill")
        catalog = _skill_catalog(
            replay_dir=configured_replay_dir,
            skills=configured_skills,
        )
        selected = catalog.get(_normalize_skill_name(skill))
        if selected is None:
            supported = ", ".join(sorted({item["name"] for item in catalog.values()}))
            raise ValueError(f"Unsupported real skill {skill!r}. Supported: {supported}")
        if selected["source"] == "vla":
            prompt = selected["prompt"] or selected["name"]
            try:
                real_manager = manager()
                real_manager.state.update(last_skill=selected["name"], last_error="")
                manager_message = real_manager.prepare_zmq_streaming()
                runner_message = vla_runner().start(
                    model_path=selected.get("model_path", ""),
                    prompt=prompt,
                    wbc_root=selected.get("wbc_root", "") or configured_gr00t_root,
                    server_root=selected.get("server_root", ""),
                    embodiment_tag=selected.get("embodiment_tag", "") or "NEW_EMBODIMENT",
                    device=selected.get("device", "") or "cuda:0",
                    port=int(selected.get("server_port", "") or 5550),
                    camera_host=selected.get("camera_host", "") or "192.168.123.164",
                    camera_port=int(selected.get("camera_port", "") or 5555),
                    action_publish_rate=int(
                        selected.get("action_publish_rate", "") or 50
                    ),
                    action_horizon=int(selected.get("action_horizon", "") or 50),
                )
                return "\n".join([manager_message, runner_message, runtime_state()])
            except Exception as exc:
                manager().state.update(last_error=str(exc))
                raise
        if selected["source"] != "replay":
            raise ValueError(f"Unsupported real skill source: {selected['source']!r}")
        replay_file = _resolve_replay_path(configured_replay_dir, selected["file"])
        try:
            manager().state.update(last_skill=selected["name"], last_error="")
            result = perform_replay_file(
                replay_file,
                preparation="keyboard_to_zmq_streaming",
                wait=wait,
                timeout=timeout,
                return_to_keyboard=return_to_keyboard,
            )
            return result
        except Exception as exc:
            manager().state.update(last_error=str(exc))
            raise

    @tool
    def unitree_g1_real_perform_replay(
        action: str,
        wait: bool = True,
        timeout: float = 60.0,
        return_to_keyboard: bool = True,
    ) -> str:
        """Perform a high-level replay action from ZMQ mode.

        The tool sends ENTER to enable ZMQ streaming, then runs the replay player
        in a separate shell.
        """
        require_tool_state("perform_replay")
        replay_file = _resolve_replay_file(configured_replay_dir, action)
        try:
            manager().state.update(last_skill=Path(replay_file).stem, last_error="")
            result = perform_replay_file(
                replay_file,
                preparation="toggle_zmq_streaming",
                wait=wait,
                timeout=timeout,
                return_to_keyboard=return_to_keyboard,
            )
            return result
        except Exception as exc:
            manager().state.update(last_error=str(exc))
            raise

    @tool
    def unitree_g1_real_switch_interface(interface: str) -> str:
        """Switch active manager interface: keyboard, gamepad, zmq, or ros2."""
        normalized = _normalize_name(interface)
        key = INTERFACE_KEYS.get(normalized)
        if key is None:
            supported = ", ".join(INTERFACE_KEYS)
            raise ValueError(f"Unsupported interface {interface!r}. Supported: {supported}")
        require_tool_state("switch_interface")
        current_mode = manager().state.snapshot()["control_mode"]
        if normalized == "zmq" and current_mode not in {
            "keyboard_normal",
            "keyboard_planner",
        }:
            raise ToolStateError(
                "Tool was not executed because switching to ZMQ is only allowed "
                f"from keyboard_normal or keyboard_planner. Current control_mode: {current_mode}."
            )
        if normalized == "keyboard" and current_mode not in {"zmq", "zmq_streaming"}:
            raise ToolStateError(
                "Tool was not executed because switching to keyboard is only allowed "
                f"from zmq or zmq_streaming. Current control_mode: {current_mode}."
            )
        message = manager().send_key(key, f"switch_interface:{normalized}")
        if normalized == "keyboard":
            mode = "keyboard_normal"
        else:
            mode = normalized if normalized in CONTROL_MODES else manager().state.control_mode
        manager().state.update(control_mode=mode, last_error="")
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_real_start_control() -> str:
        """Start the control system by sending `]` in manager keyboard mode."""
        require_tool_state("start_control")
        message = manager().send_key("]", "start_control")
        manager().state.update(control_mode="keyboard_normal", last_error="")
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_real_switch_zmq() -> str:
        """Switch manager to ZMQ interface by sending `#`."""
        require_tool_state("switch_zmq")
        return manager().switch_to_zmq()

    @tool
    def unitree_g1_real_switch_keyboard() -> str:
        """Switch manager to keyboard interface by sending `!`."""
        require_tool_state("switch_keyboard")
        return manager().switch_to_keyboard()

    @tool
    def unitree_g1_real_toggle_zmq_streaming() -> str:
        """Press ENTER in ZMQ mode to toggle ZMQ streaming enabled/disabled."""
        require_tool_state("toggle_zmq_streaming")
        return manager().toggle_zmq_streaming()

    @tool
    def unitree_g1_real_toggle_planner() -> str:
        """Press ENTER in keyboard mode to toggle keyboard normal/planner."""
        require_tool_state("toggle_planner")
        message = manager().send_key("\n", "toggle_planner")
        current = manager().state.snapshot()["control_mode"]
        next_mode = (
            "keyboard_normal"
            if current == "keyboard_planner"
            else "keyboard_planner"
        )
        manager().state.update(control_mode=next_mode, last_error="")
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_real_keyboard(
        action: str,
        repeats: int = 1,
        interval: float = 0.15,
    ) -> str:
        """Send a named keyboard control action to the real manager."""
        normalized = _normalize_name(action)
        key = KEYBOARD_ACTIONS.get(normalized)
        if key is None:
            supported = ", ".join(sorted(KEYBOARD_ACTIONS))
            raise ValueError(f"Unsupported keyboard action {action!r}. Supported: {supported}")
        require_tool_state("keyboard")
        message = manager().send_key(key, f"keyboard:{normalized}", repeats, interval)
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_real_select_mode(mode: int) -> str:
        """Select planner mode 1-8 within the current motion set."""
        safe_mode = int(mode)
        if safe_mode < 1 or safe_mode > 8:
            raise ValueError("Planner mode must be in the range 1-8.")
        require_tool_state("select_mode")
        message = manager().send_key(str(safe_mode), f"select_mode:{safe_mode}")
        manager().state.update(control_mode="keyboard_planner", last_error="")
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_real_adjust(
        action: str,
        repeats: int = 1,
        interval: float = 0.15,
    ) -> str:
        """Adjust planner speed or height."""
        normalized = _normalize_name(action)
        key = ADJUST_ACTIONS.get(normalized)
        if key is None:
            supported = ", ".join(sorted(ADJUST_ACTIONS))
            raise ValueError(f"Unsupported adjust action {action!r}. Supported: {supported}")
        require_tool_state("adjust")
        message = manager().send_key(key, f"adjust:{normalized}", repeats, interval)
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_real_compliance(
        action: str,
        repeats: int = 1,
        interval: float = 0.15,
    ) -> str:
        """Adjust global hand compliance and max hand close ratio."""
        normalized = _normalize_name(action)
        key = COMPLIANCE_ACTIONS.get(normalized)
        if key is None:
            supported = ", ".join(sorted(COMPLIANCE_ACTIONS))
            raise ValueError(f"Unsupported compliance action {action!r}. Supported: {supported}")
        require_tool_state("compliance")
        message = manager().send_key(key, f"compliance:{normalized}", repeats, interval)
        return f"{message}\n{runtime_state()}"

    available_tools = {
        "confirm_deployment": unitree_g1_real_confirm_deployment,
        "perform_skill": unitree_g1_real_perform_skill,
        "perform_replay": unitree_g1_real_perform_replay,
        "list_skills": unitree_g1_real_list_skills,
        "list_replays": unitree_g1_real_list_replays,
        "start_control": unitree_g1_real_start_control,
        "switch_zmq": unitree_g1_real_switch_zmq,
        "switch_keyboard": unitree_g1_real_switch_keyboard,
        "toggle_zmq_streaming": unitree_g1_real_toggle_zmq_streaming,
        "toggle_planner": unitree_g1_real_toggle_planner,
        "keyboard": unitree_g1_real_keyboard,
        "select_mode": unitree_g1_real_select_mode,
        "adjust": unitree_g1_real_adjust,
        "compliance": unitree_g1_real_compliance,
    }
    return [
        tool
        for tool_name, tool in available_tools.items()
        if tool_name in enabled_tool_names
    ]
