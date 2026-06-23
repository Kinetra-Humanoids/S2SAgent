"""Minimal Unitree G1 high-level motion smoke test.

Run from ``dist/s2s_agent_bundle`` on the computer connected to G1:

    uv run python app/test_unitree_g1_motion.py --config config.toml --action status

Small movement test, only when the robot is already safely standing and the area
is clear:

    uv run python app/test_unitree_g1_motion.py --config config.toml \\
        --action move_forward --confirm-motion

Only public high-level ``unitree_sdk2py.g1.loco.LocoClient`` methods are used
for actions: ``Start``, ``Damp``, ``BalanceStand``, ``Squat2StandUp``,
``Move``, ``StopMove``, and ``WaveHand``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import tomli


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
    parser = argparse.ArgumentParser(description="Test whether Unitree G1 can move.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--network-interface",
        default=None,
        help="Network interface connected to G1, e.g. eth0.",
    )
    parser.add_argument(
        "--select-mode",
        default=None,
        choices=["ai", "normal", "advanced", "ai-w"],
        help=(
            "Optionally select robot motion mode through MotionSwitcherClient "
            "before running the high-level loco action. G1 examples commonly use ai."
        ),
    )
    parser.add_argument(
        "--action",
        choices=[
            "status",
            "stop",
            "damp",
            "start",
            "balance",
            "squat_to_stand",
            "official_stand",
            "stand_up",
            "move_forward",
            "turn_left",
            "wave_hand",
        ],
        default="status",
    )
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help="Required for any action that can move the robot.",
    )
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=0.5)
    return parser.parse_args()


def require_motion_confirmation(action: str, confirmed: bool) -> None:
    if action in {"status", "stop"}:
        return
    if not confirmed:
        raise SystemExit(
            f"Action {action!r} may move the robot. Re-run with --confirm-motion "
            "after confirming the robot is safely supported/standing and the area is clear."
        )


def initialize_channel(network_interface: str) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    print(f"[G1 Test] Initializing DDS channel on interface: {network_interface}")
    ChannelFactoryInitialize(0, network_interface)


def initialize_client():
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()
    print("[G1 Test] LocoClient initialized.")
    return client


def select_motion_mode(mode: str) -> None:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )

    print(f"[G1 Test] Selecting motion mode through MotionSwitcherClient: {mode}")
    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()
    try:
        print(f"[G1 Test] CheckMode before SelectMode: {client.CheckMode()!r}")
    except Exception as exc:
        print(f"[G1 Test] CheckMode before SelectMode failed: {exc}")
    result = client.SelectMode(mode)
    print(f"[G1 Test] SelectMode({mode!r}) return: {result!r}")
    time.sleep(1.0)
    try:
        print(f"[G1 Test] CheckMode after SelectMode: {client.CheckMode()!r}")
    except Exception as exc:
        print(f"[G1 Test] CheckMode after SelectMode failed: {exc}")


def call_and_print(label: str, func, *args):
    print(f"[G1 Test] Calling {label}({', '.join(map(repr, args))})")
    result = func(*args)
    print(f"[G1 Test] {label} return: {result!r}")
    return result


def run_action(client, args: argparse.Namespace) -> None:
    action = args.action
    if action == "status":
        print("[G1 Test] High-level LocoClient connection/init succeeded.")
        return
    if action == "stop":
        call_and_print("StopMove", client.StopMove)
        return
    if action == "damp":
        call_and_print("Damp", client.Damp)
    elif action == "start":
        call_and_print("Start", client.Start)
    elif action == "balance":
        call_and_print("BalanceStand", client.BalanceStand, 0)
    elif action == "squat_to_stand":
        call_and_print("Squat2StandUp", client.Squat2StandUp)
    elif action == "official_stand":
        print("[G1 Test] Official G1 sequence: Damp(); sleep(0.5); Squat2StandUp()")
        call_and_print("Damp", client.Damp)
        time.sleep(0.5)
        call_and_print("Squat2StandUp", client.Squat2StandUp)
    elif action == "stand_up":
        call_and_print("Start", client.Start)
    elif action == "move_forward":
        duration = max(0.1, min(float(args.duration), 1.0))
        vx = max(-0.2, min(float(args.vx), 0.2))
        print("[G1 Test] Sending small forward Move, then StopMove.")
        try:
            call_and_print("Move", client.Move, vx, 0.0, 0.0)
            time.sleep(duration + 0.2)
        finally:
            call_and_print("StopMove", client.StopMove)
    elif action == "turn_left":
        duration = max(0.1, min(float(args.duration), 1.0))
        vyaw = max(-0.4, min(float(args.vyaw), 0.4))
        print("[G1 Test] Sending small yaw Move, then StopMove.")
        try:
            call_and_print("Move", client.Move, 0.0, 0.0, vyaw)
            time.sleep(duration + 0.2)
        finally:
            call_and_print("StopMove", client.StopMove)
    elif action == "wave_hand":
        call_and_print("WaveHand", client.WaveHand)
    else:
        raise ValueError(f"Unsupported action: {action}")


def main() -> int:
    args = parse_args()
    load_env_file()
    raw_config = load_raw_config(args.config)
    interface = (
        args.network_interface
        or raw_config.get("unitree_g1", {}).get("network_interface")
        or os.getenv("UNITREE_G1_NETWORK_INTERFACE")
        or ""
    )
    if not interface:
        raise SystemExit(
            "Missing network interface. Pass --network-interface eth0 or set "
            "[unitree_g1].network_interface / UNITREE_G1_NETWORK_INTERFACE."
        )

    require_motion_confirmation(args.action, args.confirm_motion)
    print("[G1 Test] WARNING: Keep the emergency stop/control method ready.")
    print("[G1 Test] Make sure the robot is safely supported or already standing.")
    initialize_channel(interface)
    if args.select_mode is not None:
        select_motion_mode(args.select_mode)
    client = initialize_client()
    run_action(client, args)
    print("[G1 Test] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
