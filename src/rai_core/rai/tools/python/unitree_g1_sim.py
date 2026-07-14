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

import json
import os
import platform
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


DEFAULT_UNITREE_G1_SIM_TOOLS = [
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

DEFAULT_REPLAY_ALIASES = {
    "wave_left_hand": "wave_left_hand.npy",
    "left_hand_wave": "wave_left_hand.npy",
    "wave": "wave_left_hand.npy",
    "run": "run.npy",
    "running": "run.npy",
    "squat_stand": "squat_stand.npy",
    "squat_to_stand": "squat_stand.npy",
    "stand_from_squat": "squat_stand.npy",
    "蹲起": "squat_stand.npy",
}

DEFAULT_UNITREE_G1_SIM_SKILLS = [
    {
        "name": "wave left hand",
        "source": "replay",
        "file": "wave_left_hand.npy",
        "aliases": ["wave", "left hand wave", "wave_left_hand"],
        "description": "Wave the left hand.",
    },
    {
        "name": "run",
        "source": "replay",
        "file": "run.npy",
        "aliases": ["running"],
        "description": "Run replay motion.",
    },
    {
        "name": "squat stand",
        "source": "replay",
        "file": "squat_stand.npy",
        "aliases": ["squat_to_stand", "蹲起"],
        "description": "Squat and stand replay motion.",
    },
]

DEPLOYMENT_STATES = {
    "not_confirmed",
    "waiting_init_done",
    "ready",
    "failed",
}
CONTROL_MODES = {
    "pre_control",
    "keyboard_normal",
    "keyboard_planner",
    "zmq",
    "zmq_streaming",
    "gamepad",
    "ros2",
}


class ToolStateError(RuntimeError):
    """Raised before a robot tool runs when the runtime state is invalid."""


TOOL_STATE_RULES = {
    "confirm_deployment": {
        "deployment": {"not_confirmed", "failed"},
        "control_mode": CONTROL_MODES,
        "description": "Use after the manager terminal asks to proceed.",
    },
    "start_control": {
        "deployment": {"ready"},
        "control_mode": {"pre_control", "keyboard_normal"},
        "description": "Use after deployment is ready, before motion/control tools.",
    },
    "perform_skill": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner"},
        "description": "Use from keyboard mode. It switches to ZMQ, enables streaming, then plays the replay file.",
    },
    "perform_replay": {
        "deployment": {"ready"},
        "control_mode": {"zmq"},
        "description": "Use from ZMQ mode. It presses ENTER to enable ZMQ streaming, then plays the replay file.",
    },
    "list_skills": {
        "deployment": DEPLOYMENT_STATES,
        "control_mode": CONTROL_MODES,
        "description": "Can be used anytime.",
    },
    "list_replays": {
        "deployment": DEPLOYMENT_STATES,
        "control_mode": CONTROL_MODES,
        "description": "Can be used anytime.",
    },
    "switch_interface": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner", "zmq", "zmq_streaming", "gamepad", "ros2"},
        "description": "Use only after deployment is ready. Target keyboard/zmq transitions still follow their specific state rules.",
    },
    "switch_zmq": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner"},
        "description": "Use only from keyboard_normal or keyboard_planner.",
    },
    "switch_keyboard": {
        "deployment": {"ready"},
        "control_mode": {"zmq", "zmq_streaming"},
        "description": "Use only from zmq or zmq_streaming.",
    },
    "toggle_zmq_streaming": {
        "deployment": {"ready"},
        "control_mode": {"zmq", "zmq_streaming"},
        "description": "Use ENTER in ZMQ mode to toggle streaming enabled/disabled.",
    },
    "toggle_planner": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner"},
        "description": "Use ENTER in keyboard mode to toggle normal/planner.",
    },
    "keyboard": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner"},
        "description": "Use after deployment is ready and keyboard control is active.",
    },
    "select_mode": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner"},
        "description": "Use after deployment is ready and keyboard control is active.",
    },
    "adjust": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner"},
        "description": "Use after deployment is ready and keyboard control is active.",
    },
    "compliance": {
        "deployment": {"ready"},
        "control_mode": {"keyboard_normal", "keyboard_planner"},
        "description": "Use after deployment is ready and keyboard control is active.",
    },
}

INTERFACE_KEYS = {
    "keyboard": "!",
    "gamepad": "@",
    "zmq": "#",
    "ros2": "$",
}

KEYBOARD_ACTIONS = {
    "play_motion": "t",
    "play": "t",
    "restart_motion": "r",
    "reset_motion": "r",
    "next_motion": "n",
    "next": "n",
    "previous_motion": "p",
    "previous": "p",
    "turn_left": "q",
    "turn_right": "e",
    "policy_heading_left": "j",
    "policy_heading_right": "l",
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "strafe_left": ",",
    "strafe_right": ".",
    "instant_stop": "r",
    "reinitialize_heading": "i",
    "toggle_encoder": "z",
    "temperature_report": "f",
}

ADJUST_ACTIONS = {
    "speed_up": "0",
    "speed_down": "9",
    "height_up": "=",
    "height_down": "-",
}

COMPLIANCE_ACTIONS = {
    "left_hand_more": "g",
    "left_hand_less": "h",
    "right_hand_more": "b",
    "right_hand_less": "v",
    "max_close_more": "x",
    "max_close_less": "c",
}


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_skill_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _default_log_dir() -> Path:
    return Path(os.getenv("UNITREE_G1_SIM_LOG_DIR", "logs/unitree_g1_sim"))


class UnitreeG1RuntimeState:
    """Small LLM-facing robot state, intentionally separate from process status."""

    def __init__(self, target: str):
        self.target = target
        self._lock = threading.RLock()
        self.deployment = "not_confirmed"
        self.control_mode = "pre_control"
        self.last_skill = ""
        self.last_error = ""

    def reset_for_start(self) -> None:
        self.update(
            deployment="not_confirmed",
            control_mode="pre_control",
            last_skill="",
            last_error="",
        )

    def update(
        self,
        *,
        deployment: str | None = None,
        control_mode: str | None = None,
        last_skill: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self._lock:
            if deployment is not None:
                self.deployment = deployment
            if control_mode is not None:
                self.control_mode = control_mode
            if last_skill is not None:
                self.last_skill = last_skill
            if last_error is not None:
                self.last_error = last_error

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {
                "target": self.target,
                "deployment": self.deployment,
                "control_mode": self.control_mode,
                "last_skill": self.last_skill or "none",
                "last_error": self.last_error or "none",
            }

    def require(
        self,
        tool_name: str,
        *,
        deployment: set[str],
        control_mode: set[str],
    ) -> None:
        state = self.snapshot()
        if (
            state["deployment"] not in deployment
            or state["control_mode"] not in control_mode
        ):
            allowed_deployment = ", ".join(sorted(deployment))
            allowed_modes = ", ".join(sorted(control_mode))
            raise ToolStateError(
                "\n".join(
                    [
                        "Tool was not executed because the Unitree G1 runtime state "
                        "does not satisfy this tool's preconditions.",
                        f"tool: {tool_name}",
                        f"current deployment: {state['deployment']}",
                        f"current control_mode: {state['control_mode']}",
                        f"allowed deployment: {allowed_deployment}",
                        f"allowed control_mode: {allowed_modes}",
                        "Use the current state and allowed states to choose the next "
                        "valid tool. Do not claim the rejected tool was executed.",
                    ]
                )
            )

    def summary(self, *, skills: list[str] | None = None) -> str:
        state = self.snapshot()
        lines = [
            "Current Unitree G1 runtime state:",
            f"- target: {state['target']}",
            f"- deployment: {state['deployment']}",
            f"- control_mode: {state['control_mode']}",
            f"- last_skill: {state['last_skill']}",
            f"- last_error: {state['last_error']}",
        ]
        if skills:
            lines.append("- available_skills: " + ", ".join(skills))
        return "\n".join(lines)


def _configured_skill_names(
    *,
    replay_dir: str,
    skills: list[dict[str, Any]] | None,
) -> list[str]:
    catalog = _skill_catalog(replay_dir=replay_dir, skills=skills)
    return sorted({skill["name"] for skill in catalog.values()})


def _tool_state_rules_text(enabled_tools: list[str] | None = None) -> str:
    tool_names = enabled_tools or DEFAULT_UNITREE_G1_SIM_TOOLS
    lines = ["Unitree G1 tool state rules:"]
    for tool_name in tool_names:
        rule = TOOL_STATE_RULES.get(tool_name)
        if rule is None:
            continue
        deployments = ", ".join(sorted(rule["deployment"]))
        modes = ", ".join(sorted(rule["control_mode"]))
        lines.append(
            f"- {tool_name}: deployment=[{deployments}], "
            f"control_mode=[{modes}]. {rule['description']}"
        )
    return "\n".join(lines)


def _launch_terminal_tail(log_path: Path, title: str) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    tail_command = f"tail -n 200 -f {shlex.quote(str(log_path))}"
    system = platform.system().lower()

    if system == "darwin":
        script = (
            'tell application "Terminal"\n'
            "  activate\n"
            f"  do script {json.dumps(tail_command)}\n"
            "end tell"
        )
        subprocess.Popen(["osascript", "-e", script])
        return f"Opened Terminal tailing {log_path}"

    candidates = [
        ["gnome-terminal", "--title", title, "--", "bash", "-lc", tail_command],
        ["konsole", "--new-tab", "-p", f"tabtitle={title}", "-e", "bash", "-lc", tail_command],
        ["xfce4-terminal", "--title", title, "--command", f"bash -lc {shlex.quote(tail_command)}"],
        ["xterm", "-T", title, "-e", "bash", "-lc", tail_command],
    ]
    for command in candidates:
        try:
            subprocess.Popen(command)
            return f"Opened terminal tailing {log_path}"
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "No supported terminal emulator found. Install gnome-terminal, konsole, "
        "xfce4-terminal, or xterm, or tail the log manually: "
        f"tail -f {log_path}"
    )


class UnitreeG1SimManager:
    """Owns the GR00T WBC manager process and sends terminal key presses."""

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
        self._terminal_started = False
        self.state = UnitreeG1RuntimeState("sim")

    def configure(self, *, terminal_viewer: bool, log_dir: str | None = None) -> None:
        self.terminal_viewer = terminal_viewer
        if log_dir:
            self.log_dir = Path(log_dir).expanduser()

    def start(self, deploy_dir: str = "", extra_args: str = "") -> str:
        with self._lock:
            if self.is_running():
                return self.status()

            cwd = Path(deploy_dir or self.deploy_dir or os.getenv("UNITREE_G1_SIM_DEPLOY_DIR", ""))
            if not str(cwd):
                raise RuntimeError(
                    "Missing GR00T deployment directory. Set [unitree_g1_sim].deploy_dir, "
                    "UNITREE_G1_SIM_DEPLOY_DIR, or pass deploy_dir to the start tool."
                )
            cwd = cwd.expanduser().resolve()
            deploy_script = cwd / "deploy.sh"
            if not deploy_script.exists():
                raise RuntimeError(f"deploy.sh was not found in {cwd}")
            setup_script = cwd / "scripts" / "setup_env.sh"
            if not setup_script.exists():
                raise RuntimeError(f"scripts/setup_env.sh was not found in {cwd}")

            master_fd, slave_fd = pty.openpty()
            deploy_command = ["bash", "deploy.sh", "--input-type", "manager"]
            if extra_args.strip():
                deploy_command.extend(shlex.split(extra_args))
            deploy_command.append("sim")
            command = [
                "bash",
                "-lc",
                "source scripts/setup_env.sh && " + shlex.join(deploy_command),
            ]

            self._logs.clear()
            self._terminal_started = False
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
                        "Unitree G1 Sim Manager",
                    )
                    self._terminal_started = True
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
                name="unitree-g1-sim-manager-reader",
                daemon=True,
            )
            self._reader_thread.start()
            self.state.reset_for_start()
            return (
                "Started GR00T WBC manager sim: "
                f"pid={self._process.pid}, command={command[-1]}. "
                "Use unitree_g1_sim_start_control to send ']' before motion commands."
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
                messages.append(
                    self.wait_for_output("Init Done", timeout=init_done_timeout)
                )
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
                return "GR00T WBC manager sim is not running."

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
            return f"Stopped GR00T WBC manager sim. Return code: {return_code}"

    def status(self, tail_lines: int = 20) -> str:
        with self._lock:
            running = self.is_running()
            pid = self._process.pid if self._process is not None else None
            uptime = time.time() - self._started_at if running and self._started_at else 0.0
            logs = self.tail_logs(tail_lines)
            return (
                f"running={running}, pid={pid}, uptime_seconds={uptime:.1f}, "
                f"deploy_dir={self.deploy_dir or os.getenv('UNITREE_G1_SIM_DEPLOY_DIR', '')}\n"
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
                    "Next step before motion tools: call unitree_g1_sim_start_control to send ']'.",
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
        message = self.send_key("!", "switch_interface:keyboard")
        self.state.update(control_mode="keyboard_normal", last_error="")
        return f"{message}\n{self.state.summary()}"

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def tail_logs(self, lines: int = 20) -> str:
        safe_lines = max(1, min(int(lines), 80))
        return "\n".join(list(self._logs)[-safe_lines:]) or "(no output captured yet)"

    def _send_key(self, key: str) -> None:
        if key == "\n":
            payload = b"\r"
        else:
            payload = key.encode("utf-8")
        if self._master_fd is None:
            raise RuntimeError("GR00T WBC manager sim PTY is not open.")
        os.write(self._master_fd, payload)
        self._append_log(f"[rai sent key] {key!r}")

    def _require_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(
                "GR00T WBC manager sim is not running. Call unitree_g1_sim_start_manager first."
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


class UnitreeG1SimReplayPlayer:
    """Runs a SONIC replay player in a separate shell process."""

    def __init__(
        self,
        gr00t_root: str = "",
        *,
        terminal_viewer: bool = False,
        log_dir: str | None = None,
    ):
        self.gr00t_root = gr00t_root
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

    def play(
        self,
        replay_file: Path,
        gr00t_root: str = "",
        wait: bool = True,
        timeout: float = 60.0,
    ) -> str:
        with self._lock:
            self.stop()
            cwd = Path(gr00t_root or self.gr00t_root or os.getenv("GR00T_WBC_ROOT", ""))
            if not str(cwd):
                raise RuntimeError(
                    "Missing GR00T-WholeBodyControl root. Set "
                    "[unitree_g1_sim].gr00t_root, GR00T_WBC_ROOT, or pass gr00t_root."
                )
            cwd = cwd.expanduser().resolve()
            is_parquet = replay_file.suffix == ".parquet"
            script_name = (
                "sonic_encoder_input_player_with_hand.py"
                if is_parquet
                else "sonic_encoder_input_player.py"
            )
            script_path = cwd / "gear_sonic" / "scripts" / script_name
            if not script_path.exists():
                raise RuntimeError(
                    f"{script_name} was not found under "
                    f"{cwd}/gear_sonic/scripts"
                )
            if not replay_file.exists():
                raise RuntimeError(f"Replay file does not exist: {replay_file}")

            input_arg = "--parquet-file" if is_parquet else "--latent-input-file"
            command = (
                "source .venv_teleop/bin/activate && "
                f"python gear_sonic/scripts/{script_name} "
                f"{input_arg} {shlex.quote(str(replay_file))}"
            )
            master_fd, slave_fd = pty.openpty()
            self._logs.clear()
            self._log_path = self.log_dir / "replay_player.log"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text("", encoding="utf-8")
            self._append_log(f"$ cd {cwd}")
            self._append_log("$ " + command)
            terminal_message = ""
            if self.terminal_viewer:
                try:
                    terminal_message = "\n" + _launch_terminal_tail(
                        self._log_path,
                        "Unitree G1 Replay Player",
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
            self._started_at = time.time()
            self._reader_thread = threading.Thread(
                target=self._read_pty_output,
                name="unitree-g1-sim-replay-reader",
                daemon=True,
            )
            self._reader_thread.start()
            started = (
                f"Started replay player: pid={self._process.pid}, "
                f"replay_file={replay_file}"
                f"{terminal_message}"
            )
        if not wait:
            return started
        return f"{started}\n{self.wait(timeout=timeout)}"

    def wait(self, timeout: float = 60.0) -> str:
        process = self._process
        if process is None:
            return "Replay player is not running."
        try:
            return_code = process.wait(timeout=max(1.0, float(timeout)))
        except subprocess.TimeoutExpired:
            return (
                "Replay player is still running after timeout. Recent logs:\n"
                f"{self.tail_logs()}"
            )
        return f"Replay player exited with code {return_code}. Recent logs:\n{self.tail_logs()}"

    def wait_for_output(self, text: str, timeout: float = 60.0) -> str:
        deadline = time.time() + max(1.0, float(timeout))
        while time.time() < deadline:
            process = self._process
            if process is None:
                return "Replay player is not running."
            if any(text in line for line in self._logs):
                return f"Detected replay player output: {text!r}"
            if process.poll() is not None:
                return (
                    f"Replay player exited before output {text!r} was detected. "
                    f"Recent logs:\n{self.tail_logs()}"
                )
            time.sleep(0.2)
        raise RuntimeError(
            f"Timed out waiting for replay player output {text!r}. Recent logs:\n"
            f"{self.tail_logs()}"
        )

    def wait_for_stopped_and_stop(self, timeout: float = 60.0) -> str:
        detected = self.wait_for_output("[EncoderInputPlayer] Stopped", timeout=timeout)
        stopped = self.stop()
        return f"{detected}\n{stopped}\nRecent logs:\n{self.tail_logs()}"

    def stop(self) -> str:
        process = self._process
        if process is None:
            return "Replay player is not running."
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
        return f"Stopped replay player. Return code: {return_code}"

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


_MANAGERS: dict[str, UnitreeG1SimManager] = {}
_MANAGERS_LOCK = threading.Lock()
_PLAYERS: dict[str, UnitreeG1SimReplayPlayer] = {}
_PLAYERS_LOCK = threading.Lock()


def _get_manager(
    deploy_dir: str,
    *,
    terminal_viewer: bool | None = None,
    log_dir: str | None = None,
) -> UnitreeG1SimManager:
    key = deploy_dir or os.getenv("UNITREE_G1_SIM_DEPLOY_DIR", "")
    viewer = _as_bool(os.getenv("UNITREE_G1_SIM_TERMINAL_VIEWER"), False)
    if terminal_viewer is not None:
        viewer = terminal_viewer
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = UnitreeG1SimManager(
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
    viewer = _as_bool(os.getenv("UNITREE_G1_SIM_TERMINAL_VIEWER"), False)
    if terminal_viewer is not None:
        viewer = terminal_viewer
    with _PLAYERS_LOCK:
        player = _PLAYERS.get(key)
        if player is None:
            player = UnitreeG1SimReplayPlayer(
                key,
                terminal_viewer=viewer,
                log_dir=log_dir,
            )
            _PLAYERS[key] = player
        else:
            player.configure(terminal_viewer=viewer, log_dir=log_dir)
        return player


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


def _resolve_replay_path(replay_dir: str, filename: str) -> Path:
    replay_root = Path(replay_dir or "replays/unitree_g1_sim").expanduser().resolve()
    path = Path(filename)
    if path.is_absolute():
        default_path = replay_root / path.name
        path = default_path if default_path.exists() else path.expanduser()
    else:
        path = replay_root / path
    if path.suffix not in {".npy", ".parquet"}:
        path = path.with_suffix(".npy")
    return path


def _skill_catalog(
    *,
    replay_dir: str,
    skills: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for raw_skill in skills or DEFAULT_UNITREE_G1_SIM_SKILLS:
        name = str(raw_skill.get("name", "")).strip()
        if not name:
            continue
        source = str(raw_skill.get("source", "replay")).strip().lower()
        skill = {
            "name": name,
            "source": source,
            "description": str(raw_skill.get("description", "")).strip(),
            "file": str(raw_skill.get("file", "")).strip(),
            "prompt": str(raw_skill.get("prompt", "")).strip(),
            "model_path": str(raw_skill.get("model_path", "")).strip(),
            "server_root": str(raw_skill.get("server_root", "")).strip(),
            "wbc_root": str(raw_skill.get("wbc_root", "")).strip(),
            "server_port": str(raw_skill.get("server_port", "")).strip(),
            "embodiment_tag": str(raw_skill.get("embodiment_tag", "")).strip(),
            "device": str(raw_skill.get("device", "")).strip(),
            "camera_host": str(raw_skill.get("camera_host", "")).strip(),
            "camera_port": str(raw_skill.get("camera_port", "")).strip(),
            "action_publish_rate": str(raw_skill.get("action_publish_rate", "")).strip(),
            "action_horizon": str(raw_skill.get("action_horizon", "")).strip(),
        }
        aliases = raw_skill.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [item.strip() for item in aliases.split(",")]
        for alias in [name, *list(aliases)]:
            if str(alias).strip():
                catalog[_normalize_skill_name(str(alias))] = skill
    for alias, filename in DEFAULT_REPLAY_ALIASES.items():
        key = _normalize_skill_name(alias)
        catalog.setdefault(
            key,
            {
                "name": alias,
                "source": "replay",
                "description": "",
                "file": filename,
                "prompt": "",
            },
        )
    return catalog


def start_unitree_g1_sim_manager(
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


def stop_unitree_g1_sim_manager(*, deploy_dir: str | None = None) -> str:
    manager = _get_manager(deploy_dir or "")
    return manager.stop(graceful=True)


def get_unitree_g1_sim_runtime_prompt(
    *,
    deploy_dir: str | None = None,
    replay_dir: str | None = None,
    skills: list[dict[str, Any]] | None = None,
    enabled_tools: list[str] | None = None,
) -> str:
    configured_replay_dir = replay_dir or os.getenv(
        "UNITREE_G1_SIM_REPLAY_DIR",
        "replays/unitree_g1_sim",
    )
    manager = _get_manager(deploy_dir or "")
    skill_names = _configured_skill_names(
        replay_dir=configured_replay_dir,
        skills=skills or DEFAULT_UNITREE_G1_SIM_SKILLS,
    )
    return "\n\n".join(
        [
            manager.state.summary(skills=skill_names),
            _tool_state_rules_text(enabled_tools),
        ]
    )


def get_unitree_g1_sim_tools(
    *,
    deploy_dir: str | None = None,
    gr00t_root: str | None = None,
    replay_dir: str | None = None,
    enabled_tools: list[str] | None = None,
    skills: list[dict[str, Any]] | None = None,
    terminal_viewer: bool | None = None,
    log_dir: str | None = None,
) -> list[BaseTool]:
    configured_deploy_dir = deploy_dir or os.getenv("UNITREE_G1_SIM_DEPLOY_DIR", "")
    configured_gr00t_root = gr00t_root or os.getenv("GR00T_WBC_ROOT", "")
    configured_replay_dir = replay_dir or os.getenv(
        "UNITREE_G1_SIM_REPLAY_DIR",
        "replays/unitree_g1_sim",
    )
    enabled_tool_names = set(enabled_tools or DEFAULT_UNITREE_G1_SIM_TOOLS)
    configured_skills = skills or DEFAULT_UNITREE_G1_SIM_SKILLS
    active_deploy_dir = {"value": configured_deploy_dir}
    skill_names = _configured_skill_names(
        replay_dir=configured_replay_dir,
        skills=configured_skills,
    )

    def manager() -> UnitreeG1SimManager:
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
    def unitree_g1_sim_start_manager(
        deploy_dir: str = "",
        extra_args: str = "",
    ) -> str:
        """Start GR00T WholeBodyControl manager sim.

        Runs `source scripts/setup_env.sh && bash deploy.sh --input-type
        manager sim` from the configured `gear_sonic_deploy` directory.
        `extra_args` may contain optional deploy flags such as `--zmq-host
        127.0.0.1 --zmq-port 5556 --zmq-topic pose`; they are inserted before
        the final `sim` argument.
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
    def unitree_g1_sim_stop_manager(graceful: bool = True) -> str:
        """Stop the backend-owned manager process without sending `O`.

        For robot safety, `O` emergency stop is reserved for the human operator
        typing in the visible control terminal.
        """
        return manager().stop(graceful=graceful)

    @tool
    def unitree_g1_sim_status(tail_lines: int = 20) -> str:
        """Return manager process status and recent terminal output."""
        return manager().status(tail_lines=tail_lines)

    @tool
    def unitree_g1_sim_confirm_deployment(timeout: float = 60.0) -> str:
        """Confirm `Proceed with deployment` by sending `Y` and ENTER.

        Waits until the manager prints `Init Done`. This sets deployment to
        ready but leaves control_mode as pre_control; call
        unitree_g1_sim_start_control next to send `]`.
        """
        require_tool_state("confirm_deployment")
        return manager().confirm_deployment(timeout=timeout)

    @tool
    def unitree_g1_sim_list_replays() -> str:
        """List configured high-level replay actions and whether their .npy files exist."""
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
    def unitree_g1_sim_list_skills() -> str:
        """List configured Unitree G1 sim skills exposed to the agent."""
        require_tool_state("list_skills")
        catalog = _skill_catalog(
            replay_dir=configured_replay_dir,
            skills=configured_skills,
        )
        seen: set[str] = set()
        lines = ["Configured Unitree G1 sim skills:"]
        for skill in catalog.values():
            name = skill["name"]
            if name in seen:
                continue
            seen.add(name)
            if skill["source"] == "replay":
                path = _resolve_replay_path(configured_replay_dir, skill["file"])
                status = "ready" if path.exists() else "missing"
                lines.append(f"- {name}: replay {path} ({status})")
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
        sim_manager = manager()
        replay_player = player()
        if preparation == "keyboard_to_zmq_streaming":
            manager_message = sim_manager.prepare_zmq_streaming()
        elif preparation == "toggle_zmq_streaming":
            manager_message = sim_manager.toggle_zmq_streaming()
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
                messages.append(sim_manager.switch_to_keyboard())
            elif return_to_keyboard:
                messages.append(
                    "Replay player is still running; keyboard mode was not restored yet."
                )
        elif return_to_keyboard:
            def restore_keyboard_after_replay() -> None:
                replay_player.wait_for_stopped_and_stop(timeout=timeout)
                if not replay_player.is_running():
                    sim_manager.switch_to_keyboard()

            threading.Thread(
                target=restore_keyboard_after_replay,
                name="unitree-g1-sim-return-keyboard",
                daemon=True,
            ).start()
            messages.append("Keyboard mode will be restored after the replay exits.")
        return "\n".join([*messages, runtime_state()])

    @tool
    def unitree_g1_sim_perform_skill(
        skill: str,
        wait: bool = True,
        timeout: float = 60.0,
        return_to_keyboard: bool = True,
    ) -> str:
        """Perform a configured sim skill by name.

        Replay skills switch to ZMQ (`#`), send ENTER to open ZMQ streaming, then
        run `sonic_encoder_input_player.py --latent-input-file <file>` in the
        replay player shell.
        """
        require_tool_state("perform_skill")
        catalog = _skill_catalog(
            replay_dir=configured_replay_dir,
            skills=configured_skills,
        )
        selected = catalog.get(_normalize_skill_name(skill))
        if selected is None:
            supported = ", ".join(sorted({item["name"] for item in catalog.values()}))
            raise ValueError(f"Unsupported sim skill {skill!r}. Supported: {supported}")
        if selected["source"] != "replay":
            raise ValueError(f"Unsupported sim skill source: {selected['source']!r}")
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
    def unitree_g1_sim_perform_replay(
        action: str,
        wait: bool = True,
        timeout: float = 60.0,
        return_to_keyboard: bool = True,
    ) -> str:
        """Perform a high-level replay action from a latent .npy file.

        The tool must be called from ZMQ mode. It sends ENTER to enable ZMQ
        streaming, then runs
        `sonic_encoder_input_player.py --latent-input-file <file>` from the
        configured GR00T-WholeBodyControl root. Supported action aliases include
        wave_left_hand, run, and squat_stand/蹲起. A direct .npy path is also
        accepted. By default, the replay player shell is closed after it prints
        `[EncoderInputPlayer] Stopped`, then the manager switches back to
        keyboard mode.
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
    def unitree_g1_sim_switch_interface(interface: str) -> str:
        """Switch active manager interface.

        Supported interfaces: keyboard (`!`), gamepad (`@`), zmq (`#`), ros2 (`$`).
        Switching triggers the manager safety reset described in the GR00T WBC docs.
        """
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
    def unitree_g1_sim_start_control() -> str:
        """Start the control system by sending `]` in manager keyboard mode."""
        require_tool_state("start_control")
        message = manager().send_key("]", "start_control")
        manager().state.update(control_mode="keyboard_normal", last_error="")
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_sim_switch_zmq() -> str:
        """Switch manager to ZMQ interface by sending `#`."""
        require_tool_state("switch_zmq")
        return manager().switch_to_zmq()

    @tool
    def unitree_g1_sim_switch_keyboard() -> str:
        """Switch manager to keyboard interface by sending `!`."""
        require_tool_state("switch_keyboard")
        return manager().switch_to_keyboard()

    @tool
    def unitree_g1_sim_toggle_zmq_streaming() -> str:
        """Press ENTER in ZMQ mode to toggle ZMQ streaming enabled/disabled."""
        require_tool_state("toggle_zmq_streaming")
        return manager().toggle_zmq_streaming()

    @tool
    def unitree_g1_sim_toggle_planner() -> str:
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
    def unitree_g1_sim_keyboard(
        action: str,
        repeats: int = 1,
        interval: float = 0.15,
    ) -> str:
        """Send a named keyboard control action to the active manager interface.

        Supported actions include: play_motion, restart_motion, next_motion,
        previous_motion, turn_left, turn_right, forward, backward, left, right,
        strafe_left, strafe_right, instant_stop, reinitialize_heading,
        toggle_encoder, temperature_report, policy_heading_left,
        policy_heading_right.
        """
        normalized = _normalize_name(action)
        key = KEYBOARD_ACTIONS.get(normalized)
        if key is None:
            supported = ", ".join(sorted(KEYBOARD_ACTIONS))
            raise ValueError(f"Unsupported keyboard action {action!r}. Supported: {supported}")
        require_tool_state("keyboard")
        message = manager().send_key(key, f"keyboard:{normalized}", repeats, interval)
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_sim_select_mode(mode: int) -> str:
        """Select planner mode 1-8 within the current motion set."""
        safe_mode = int(mode)
        if safe_mode < 1 or safe_mode > 8:
            raise ValueError("Planner mode must be in the range 1-8.")
        require_tool_state("select_mode")
        message = manager().send_key(str(safe_mode), f"select_mode:{safe_mode}")
        manager().state.update(control_mode="keyboard_planner", last_error="")
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_sim_adjust(
        action: str,
        repeats: int = 1,
        interval: float = 0.15,
    ) -> str:
        """Adjust planner speed or height.

        Supported actions: speed_up, speed_down, height_up, height_down.
        """
        normalized = _normalize_name(action)
        key = ADJUST_ACTIONS.get(normalized)
        if key is None:
            supported = ", ".join(sorted(ADJUST_ACTIONS))
            raise ValueError(f"Unsupported adjust action {action!r}. Supported: {supported}")
        require_tool_state("adjust")
        message = manager().send_key(key, f"adjust:{normalized}", repeats, interval)
        return f"{message}\n{runtime_state()}"

    @tool
    def unitree_g1_sim_compliance(
        action: str,
        repeats: int = 1,
        interval: float = 0.15,
    ) -> str:
        """Adjust global hand compliance and max hand close ratio.

        Supported actions: left_hand_more, left_hand_less, right_hand_more,
        right_hand_less, max_close_more, max_close_less.
        """
        normalized = _normalize_name(action)
        key = COMPLIANCE_ACTIONS.get(normalized)
        if key is None:
            supported = ", ".join(sorted(COMPLIANCE_ACTIONS))
            raise ValueError(f"Unsupported compliance action {action!r}. Supported: {supported}")
        require_tool_state("compliance")
        message = manager().send_key(key, f"compliance:{normalized}", repeats, interval)
        return f"{message}\n{runtime_state()}"

    available_tools = {
        "confirm_deployment": unitree_g1_sim_confirm_deployment,
        "perform_skill": unitree_g1_sim_perform_skill,
        "perform_replay": unitree_g1_sim_perform_replay,
        "list_skills": unitree_g1_sim_list_skills,
        "list_replays": unitree_g1_sim_list_replays,
        "start_control": unitree_g1_sim_start_control,
        "switch_zmq": unitree_g1_sim_switch_zmq,
        "switch_keyboard": unitree_g1_sim_switch_keyboard,
        "toggle_zmq_streaming": unitree_g1_sim_toggle_zmq_streaming,
        "toggle_planner": unitree_g1_sim_toggle_planner,
        "keyboard": unitree_g1_sim_keyboard,
        "select_mode": unitree_g1_sim_select_mode,
        "adjust": unitree_g1_sim_adjust,
        "compliance": unitree_g1_sim_compliance,
    }
    return [
        tool
        for tool_name, tool in available_tools.items()
        if tool_name in enabled_tool_names
    ]
