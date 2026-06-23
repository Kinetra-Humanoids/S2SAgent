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


DEFAULT_UNITREE_G1_SIM_TOOLS = [
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


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


class UnitreeG1SimManager:
    """Owns the GR00T WBC manager process and sends terminal key presses."""

    def __init__(self, deploy_dir: str = ""):
        self.deploy_dir = deploy_dir
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._logs: deque[str] = deque(maxlen=240)
        self._started_at: float | None = None

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

            master_fd, slave_fd = pty.openpty()
            command = ["bash", "deploy.sh", "--input-type", "manager"]
            if extra_args.strip():
                command.extend(shlex.split(extra_args))
            command.append("sim")

            self._logs.clear()
            self._append_log(f"$ cd {cwd}")
            self._append_log("$ " + shlex.join(command))
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
            return (
                "Started GR00T WBC manager sim: "
                f"pid={self._process.pid}, command={shlex.join(command)}. "
                "Use unitree_g1_sim_start_control to send ']' before motion commands."
            )

    def ensure_started(
        self,
        deploy_dir: str = "",
        extra_args: str = "",
        start_control: bool = True,
        settle_seconds: float = 2.0,
    ) -> str:
        messages = [self.start(deploy_dir=deploy_dir, extra_args=extra_args)]
        if settle_seconds > 0:
            time.sleep(min(float(settle_seconds), 10.0))
        if start_control:
            messages.append(self.send_key("]", "start_control"))
        return "\n".join(messages)

    def stop(self, graceful: bool = True, timeout: float = 5.0) -> str:
        with self._lock:
            process = self._process
            if process is None:
                return "GR00T WBC manager sim is not running."

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

    def prepare_zmq_streaming(self) -> str:
        with self._lock:
            self._require_running()
            self._send_key("#")
            time.sleep(0.2)
            self._send_key("\n")
            return "Switched manager to ZMQ interface and sent ENTER to toggle streaming."

    def switch_to_keyboard(self) -> str:
        return self.send_key("!", "switch_interface:keyboard")

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

    def _close_master_fd(self) -> None:
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None


class UnitreeG1SimReplayPlayer:
    """Runs sonic_encoder_input_player.py in a separate shell process."""

    def __init__(self, gr00t_root: str = ""):
        self.gr00t_root = gr00t_root
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._logs: deque[str] = deque(maxlen=240)
        self._started_at: float | None = None

    def play(
        self,
        latent_input_file: Path,
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
            if not (cwd / "gear_sonic" / "scripts" / "sonic_encoder_input_player.py").exists():
                raise RuntimeError(
                    "sonic_encoder_input_player.py was not found under "
                    f"{cwd}/gear_sonic/scripts"
                )
            if not latent_input_file.exists():
                raise RuntimeError(f"Replay file does not exist: {latent_input_file}")

            command = (
                "source .venv_teleop/bin/activate && "
                "python gear_sonic/scripts/sonic_encoder_input_player.py "
                f"--latent-input-file {shlex.quote(str(latent_input_file))}"
            )
            master_fd, slave_fd = pty.openpty()
            self._logs.clear()
            self._append_log(f"$ cd {cwd}")
            self._append_log("$ " + command)
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
                f"latent_input_file={latent_input_file}"
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


def _get_manager(deploy_dir: str) -> UnitreeG1SimManager:
    key = deploy_dir or os.getenv("UNITREE_G1_SIM_DEPLOY_DIR", "")
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = UnitreeG1SimManager(key)
            _MANAGERS[key] = manager
        return manager


def _get_player(gr00t_root: str) -> UnitreeG1SimReplayPlayer:
    key = gr00t_root or os.getenv("GR00T_WBC_ROOT", "")
    with _PLAYERS_LOCK:
        player = _PLAYERS.get(key)
        if player is None:
            player = UnitreeG1SimReplayPlayer(key)
            _PLAYERS[key] = player
        return player


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


def start_unitree_g1_sim_manager(
    *,
    deploy_dir: str | None = None,
    extra_args: str = "",
    start_control: bool = True,
    settle_seconds: float = 2.0,
) -> str:
    manager = _get_manager(deploy_dir or "")
    return manager.ensure_started(
        deploy_dir=deploy_dir or "",
        extra_args=extra_args,
        start_control=start_control,
        settle_seconds=settle_seconds,
    )


def stop_unitree_g1_sim_manager(*, deploy_dir: str | None = None) -> str:
    manager = _get_manager(deploy_dir or "")
    return manager.stop(graceful=True)


def get_unitree_g1_sim_tools(
    *,
    deploy_dir: str | None = None,
    gr00t_root: str | None = None,
    replay_dir: str | None = None,
    enabled_tools: list[str] | None = None,
) -> list[BaseTool]:
    configured_deploy_dir = deploy_dir or os.getenv("UNITREE_G1_SIM_DEPLOY_DIR", "")
    configured_gr00t_root = gr00t_root or os.getenv("GR00T_WBC_ROOT", "")
    configured_replay_dir = replay_dir or os.getenv(
        "UNITREE_G1_SIM_REPLAY_DIR",
        "replays/unitree_g1_sim",
    )
    enabled_tool_names = set(enabled_tools or DEFAULT_UNITREE_G1_SIM_TOOLS)
    active_deploy_dir = {"value": configured_deploy_dir}

    def manager() -> UnitreeG1SimManager:
        return _get_manager(active_deploy_dir["value"])

    def player() -> UnitreeG1SimReplayPlayer:
        return _get_player(configured_gr00t_root)

    @tool
    def unitree_g1_sim_start_manager(
        deploy_dir: str = "",
        extra_args: str = "",
    ) -> str:
        """Start GR00T WholeBodyControl manager sim.

        Runs `bash deploy.sh --input-type manager sim` from the configured
        `gear_sonic_deploy` directory. `extra_args` may contain optional deploy
        flags such as `--zmq-host 127.0.0.1 --zmq-port 5556 --zmq-topic pose`;
        they are inserted before the final `sim` argument.
        """
        target_deploy_dir = deploy_dir or configured_deploy_dir
        active_deploy_dir["value"] = target_deploy_dir
        target_manager = _get_manager(target_deploy_dir)
        return target_manager.start(deploy_dir=deploy_dir, extra_args=extra_args)

    @tool
    def unitree_g1_sim_stop_manager(graceful: bool = True) -> str:
        """Stop GR00T manager sim. Graceful mode sends `O` emergency stop first."""
        return manager().stop(graceful=graceful)

    @tool
    def unitree_g1_sim_status(tail_lines: int = 20) -> str:
        """Return manager process status and recent terminal output."""
        return manager().status(tail_lines=tail_lines)

    @tool
    def unitree_g1_sim_list_replays() -> str:
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
    def unitree_g1_sim_perform_replay(
        action: str,
        wait: bool = True,
        timeout: float = 60.0,
        return_to_keyboard: bool = True,
    ) -> str:
        """Perform a high-level replay action from a latent .npy file.

        The tool switches manager to ZMQ (`#`), sends ENTER to enable streaming,
        then runs `sonic_encoder_input_player.py --latent-input-file <file>` from
        the configured GR00T-WholeBodyControl root. Supported action aliases
        include wave_left_hand, run, and squat_stand/蹲起. A direct .npy path is
        also accepted. By default, the manager switches back to keyboard mode
        after the replay player exits.
        """
        replay_file = _resolve_replay_file(configured_replay_dir, action)
        sim_manager = manager()
        replay_player = player()
        manager_message = sim_manager.prepare_zmq_streaming()
        player_message = replay_player.play(
            replay_file,
            gr00t_root=configured_gr00t_root,
            wait=False,
        )
        messages = [manager_message, player_message]
        if wait:
            messages.append(replay_player.wait(timeout=timeout))
            if return_to_keyboard and not replay_player.is_running():
                messages.append(sim_manager.switch_to_keyboard())
            elif return_to_keyboard:
                messages.append(
                    "Replay player is still running; keyboard mode was not restored yet."
                )
        elif return_to_keyboard:
            def restore_keyboard_after_replay() -> None:
                replay_player.wait(timeout=timeout)
                if not replay_player.is_running():
                    sim_manager.switch_to_keyboard()

            threading.Thread(
                target=restore_keyboard_after_replay,
                name="unitree-g1-sim-return-keyboard",
                daemon=True,
            ).start()
            messages.append("Keyboard mode will be restored after the replay exits.")
        return "\n".join(messages)

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
        return manager().send_key(key, f"switch_interface:{normalized}")

    @tool
    def unitree_g1_sim_start_control() -> str:
        """Start the control system by sending `]` in manager keyboard mode."""
        return manager().send_key("]", "start_control")

    @tool
    def unitree_g1_sim_toggle_planner() -> str:
        """Toggle between Normal mode and Planner mode by sending ENTER."""
        return manager().send_key("\n", "toggle_planner")

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
        return manager().send_key(key, f"keyboard:{normalized}", repeats, interval)

    @tool
    def unitree_g1_sim_select_mode(mode: int) -> str:
        """Select planner mode 1-8 within the current motion set."""
        safe_mode = int(mode)
        if safe_mode < 1 or safe_mode > 8:
            raise ValueError("Planner mode must be in the range 1-8.")
        return manager().send_key(str(safe_mode), f"select_mode:{safe_mode}")

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
        return manager().send_key(key, f"adjust:{normalized}", repeats, interval)

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
        return manager().send_key(key, f"compliance:{normalized}", repeats, interval)

    available_tools = {
        "start": unitree_g1_sim_start_manager,
        "stop": unitree_g1_sim_stop_manager,
        "status": unitree_g1_sim_status,
        "perform_replay": unitree_g1_sim_perform_replay,
        "list_replays": unitree_g1_sim_list_replays,
        "switch_interface": unitree_g1_sim_switch_interface,
        "start_control": unitree_g1_sim_start_control,
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
