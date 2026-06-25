"""Smoke test for Unitree G1 sim replay actions.

Run from ``dist/s2s_agent_bundle``:

    uv run python app/test_unitree_g1_sim_replay.py --action wave_left_hand

This starts the GR00T manager sim in the current Python process, sends ``]`` to
enter control mode, calls ``unitree_g1_sim_perform_replay``, and switches back
to keyboard mode after the replay exits.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

import tomli

from rai.tools.python.unitree_g1_sim import _resolve_replay_file
from rai.tools.python import (
    get_unitree_g1_sim_tools,
    start_unitree_g1_sim_manager,
    stop_unitree_g1_sim_manager,
)


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    return key.strip(), value.strip().strip('"').strip("'")


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is not None:
            key, value = parsed
            os.environ.setdefault(key, value)


def load_raw_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return tomli.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Unitree G1 sim replay playback."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--action",
        default="wave_left_hand",
        help="Replay action name or .npy path. Default: wave_left_hand.",
    )
    parser.add_argument(
        "--deploy-dir",
        default=None,
        help="gear_sonic_deploy directory containing deploy.sh.",
    )
    parser.add_argument(
        "--gr00t-root",
        default=None,
        help="GR00T-WholeBodyControl repo root.",
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Directory containing replay .npy files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the replay player to exit.",
    )
    parser.add_argument(
        "--startup-settle-seconds",
        type=float,
        default=None,
        help="Delay between manager sim start and sending `]`.",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Skip manager startup. Use only if this same process already started it.",
    )
    parser.add_argument(
        "--no-confirm-deployment",
        action="store_true",
        help="Do not send Y/ENTER after starting deploy.sh.",
    )
    parser.add_argument(
        "--no-return-to-keyboard",
        action="store_true",
        help="Do not switch back to keyboard mode after replay exits.",
    )
    parser.add_argument(
        "--stop-after",
        action="store_true",
        help="Stop the manager sim after the replay test.",
    )
    return parser.parse_args()


def find_tool(tools, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    raise RuntimeError(f"Tool not found: {name}")


def print_shell_command(label: str, cwd: str | Path, command: str) -> None:
    print(f"[Sim Replay Test] {label}:")
    print(f"  cd {Path(cwd).expanduser().resolve()}")
    print(f"  {command}")


def build_manager_command() -> str:
    return "source scripts/setup_env.sh && bash deploy.sh --input-type manager sim"


def build_replay_command(latent_input_file: Path) -> str:
    return (
        "source .venv_teleop/bin/activate && "
        "python gear_sonic/scripts/sonic_encoder_input_player.py "
        f"--latent-input-file {shlex.quote(str(latent_input_file))}"
    )


def invoke_tool(tool, args: dict[str, Any]) -> Any:
    print(f"[Sim Replay Test] Tool call: {tool.name}")
    print(f"  args: {args}")
    return tool.invoke(
        {
            "name": tool.name,
            "args": args,
            "id": f"test_unitree_g1_sim_{uuid4().hex}",
            "type": "tool_call",
        }
    )


def main() -> int:
    args = parse_args()
    load_env_file()
    raw_config = load_raw_config(args.config)
    sim_config = raw_config.get("unitree_g1_sim", {})

    deploy_dir = (
        args.deploy_dir
        or sim_config.get("deploy_dir")
        or os.getenv("UNITREE_G1_SIM_DEPLOY_DIR")
        or ""
    )
    gr00t_root = (
        args.gr00t_root
        or sim_config.get("gr00t_root")
        or os.getenv("GR00T_WBC_ROOT")
        or ""
    )
    replay_dir = (
        args.replay_dir
        or sim_config.get("replay_dir")
        or os.getenv("UNITREE_G1_SIM_REPLAY_DIR")
        or "replays/unitree_g1_sim"
    )
    startup_settle_seconds = (
        args.startup_settle_seconds
        if args.startup_settle_seconds is not None
        else sim_config.get("startup_settle_seconds", 2.0)
    )
    confirm_deployment = bool(sim_config.get("confirm_deployment", True)) and not (
        args.no_confirm_deployment
    )

    print("[Sim Replay Test] Configuration:")
    print(f"  action: {args.action}")
    print(f"  deploy_dir: {deploy_dir}")
    print(f"  gr00t_root: {gr00t_root}")
    print(f"  replay_dir: {replay_dir}")

    if not args.no_auto_start:
        print_shell_command(
            "Manager command executed by startup/tool backend",
            deploy_dir,
            build_manager_command(),
        )
        print("[Sim Replay Test] Starting manager sim and entering control mode...")
        if confirm_deployment:
            print("[Sim Replay Test] Deployment confirmation after startup: 'Y' then ENTER")
            print("[Sim Replay Test] Waiting for manager output: Init Done")
        print("[Sim Replay Test] Manager hotkey after Init Done: ']'")
        startup_message = start_unitree_g1_sim_manager(
            deploy_dir=deploy_dir,
            confirm_deployment=confirm_deployment,
            start_control=True,
            settle_seconds=float(startup_settle_seconds),
        )
        print(startup_message)

    tools = get_unitree_g1_sim_tools(
        deploy_dir=deploy_dir,
        gr00t_root=gr00t_root,
        replay_dir=replay_dir,
        enabled_tools=["perform_replay", "list_replays", "status"],
    )

    list_replays_tool = find_tool(tools, "unitree_g1_sim_list_replays")
    print("[Sim Replay Test] Available replay files:")
    print(invoke_tool(list_replays_tool, {}))

    replay_tool = find_tool(tools, "unitree_g1_sim_perform_replay")
    print(f"[Sim Replay Test] Performing replay: {args.action}")
    replay_file = _resolve_replay_file(replay_dir, args.action)
    print("[Sim Replay Test] Replay tool hotkeys:")
    print("  before replay: ']', wait 1s, '#' then ENTER")
    print("  replay shell: close after output contains 'End'")
    if not args.no_return_to_keyboard:
        print("  after replay: '!'")
    print_shell_command(
        "Replay command executed by unitree_g1_sim_perform_replay",
        gr00t_root,
        build_replay_command(replay_file),
    )
    result = invoke_tool(
        replay_tool,
        {
            "action": args.action,
            "wait": True,
            "timeout": args.timeout,
            "return_to_keyboard": not args.no_return_to_keyboard,
        },
    )
    print(result)

    if args.stop_after:
        print("[Sim Replay Test] Stopping manager sim...")
        print(stop_unitree_g1_sim_manager(deploy_dir=deploy_dir))

    print("[Sim Replay Test] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
