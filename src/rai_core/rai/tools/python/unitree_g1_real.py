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

from langchain_core.tools import BaseTool, tool

from rai.tools.python.unitree_g1_sim import (
    ADJUST_ACTIONS,
    COMPLIANCE_ACTIONS,
    DEFAULT_REPLAY_ALIASES,
    INTERFACE_KEYS,
    KEYBOARD_ACTIONS,
    UnitreeG1SimReplayPlayer,
    _as_bool,
    _launch_terminal_tail,
    _normalize_name,
)


DEFAULT_UNITREE_G1_REAL_TOOLS = [
    "start",
    "stop",
    "status",
    "perform_replay",
    "list_replays",
    "switch_interface",
    "start_control",
    "toggle_planner",
    "keyboard",
    "select_mode",
    "adjust",
    "compliance",
]


def _default_log_dir() -> Path:
    return Path(os.getenv("UNITREE_G1_REAL_LOG_DIR", "logs/unitree_g1_real"))


def _resolve_replay_file(replay_dir: str, action: str) -> Path:
    replay_root = Path(replay_dir or "replays/unitree_g1_sim").expanduser().resolve()
    normalized = _normalize_name(action)
    filename = DEFAULT_REPLAY_ALIASES.get(normalized, action)
    path = Path(filename)
    if not path.is_absolute():
        path = replay_root / path
    if path.suffix != ".npy":
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
            return (
                "Started GR00T WBC real manager: "
                f"pid={self._process.pid}, command={command[-1]}."
                f"{terminal_message}"
            )

    def ensure_started(
        self,
        deploy_dir: str = "",
        extra_args: str = "",
        confirm_deployment: bool = True,
        start_control: bool = True,
        settle_seconds: float = 2.0,
        init_done_timeout: float = 60.0,
    ) -> str:
        messages = [self.start(deploy_dir=deploy_dir, extra_args=extra_args)]
        if settle_seconds > 0:
            time.sleep(min(float(settle_seconds), 10.0))
        if confirm_deployment:
            self._require_running()
            self._send_key("y")
            self._send_key("\n")
            messages.append("Sent deployment confirmation: Y")
            messages.append(self.wait_for_output("Init Done", timeout=init_done_timeout))
        if start_control:
            messages.append(self.send_key("]", "start_control"))
        return "\n".join(messages)

    def stop(self, graceful: bool = True, timeout: float = 5.0) -> str:
        with self._lock:
            process = self._process
            if process is None:
                return "GR00T WBC real manager is not running."

            if process.poll() is None and graceful:
                self._send_key("o")
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass

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
            return (
                "Prepared replay streaming: sent ']', waited 1.0s, "
                "then sent '#' and ENTER."
            )

    def switch_to_keyboard(self) -> str:
        return self.send_key("!", "switch_interface:keyboard")

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


_MANAGERS: dict[str, UnitreeG1RealManager] = {}
_MANAGERS_LOCK = threading.Lock()
_PLAYERS: dict[str, UnitreeG1SimReplayPlayer] = {}
_PLAYERS_LOCK = threading.Lock()


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


def start_unitree_g1_real_manager(
    *,
    deploy_dir: str | None = None,
    extra_args: str = "",
    confirm_deployment: bool = True,
    start_control: bool = True,
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


def get_unitree_g1_real_tools(
    *,
    deploy_dir: str | None = None,
    gr00t_root: str | None = None,
    replay_dir: str | None = None,
    enabled_tools: list[str] | None = None,
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
    active_deploy_dir = {"value": configured_deploy_dir}

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
        """Stop GR00T real manager. Graceful mode sends `O` emergency stop first."""
        return manager().stop(graceful=graceful)

    @tool
    def unitree_g1_real_status(tail_lines: int = 20) -> str:
        """Return real manager process status and recent terminal output."""
        return manager().status(tail_lines=tail_lines)

    @tool
    def unitree_g1_real_list_replays() -> str:
        """List configured high-level replay actions and whether their .npy files exist."""
        replay_root = Path(configured_replay_dir).expanduser().resolve()
        lines = [f"Replay directory: {replay_root}"]
        for action_name, filename in sorted(DEFAULT_REPLAY_ALIASES.items()):
            if action_name != Path(filename).stem:
                continue
            path = replay_root / filename
            status = "ready" if path.exists() else "missing"
            lines.append(f"- {action_name}: {path} ({status})")
        return "\n".join(lines)

    @tool
    def unitree_g1_real_perform_replay(
        action: str,
        wait: bool = True,
        timeout: float = 60.0,
        return_to_keyboard: bool = True,
    ) -> str:
        """Perform a high-level replay action on the real manager via ZMQ."""
        replay_file = _resolve_replay_file(configured_replay_dir, action)
        real_manager = manager()
        replay_player = player()
        manager_message = real_manager.prepare_replay_streaming()
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
        return "\n".join(messages)

    @tool
    def unitree_g1_real_switch_interface(interface: str) -> str:
        """Switch active manager interface: keyboard, gamepad, zmq, or ros2."""
        normalized = _normalize_name(interface)
        key = INTERFACE_KEYS.get(normalized)
        if key is None:
            supported = ", ".join(INTERFACE_KEYS)
            raise ValueError(f"Unsupported interface {interface!r}. Supported: {supported}")
        return manager().send_key(key, f"switch_interface:{normalized}")

    @tool
    def unitree_g1_real_start_control() -> str:
        """Start the control system by sending `]` in manager keyboard mode."""
        return manager().send_key("]", "start_control")

    @tool
    def unitree_g1_real_toggle_planner() -> str:
        """Toggle between Normal mode and Planner mode by sending ENTER."""
        return manager().send_key("\n", "toggle_planner")

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
        return manager().send_key(key, f"keyboard:{normalized}", repeats, interval)

    @tool
    def unitree_g1_real_select_mode(mode: int) -> str:
        """Select planner mode 1-8 within the current motion set."""
        safe_mode = int(mode)
        if safe_mode < 1 or safe_mode > 8:
            raise ValueError("Planner mode must be in the range 1-8.")
        return manager().send_key(str(safe_mode), f"select_mode:{safe_mode}")

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
        return manager().send_key(key, f"adjust:{normalized}", repeats, interval)

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
        return manager().send_key(key, f"compliance:{normalized}", repeats, interval)

    available_tools = {
        "start": unitree_g1_real_start_manager,
        "stop": unitree_g1_real_stop_manager,
        "status": unitree_g1_real_status,
        "perform_replay": unitree_g1_real_perform_replay,
        "list_replays": unitree_g1_real_list_replays,
        "switch_interface": unitree_g1_real_switch_interface,
        "start_control": unitree_g1_real_start_control,
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
