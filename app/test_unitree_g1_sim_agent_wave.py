"""Agent-level smoke test for Unitree G1 sim wave_left_hand replay.

Run from ``dist/s2s_agent_bundle``:

    uv run python app/test_unitree_g1_sim_agent_wave.py

This test feeds a natural-language instruction to a ReActAgent and uses a
deterministic fake chat model that calls ``unitree_g1_sim_perform_replay`` with
``action=wave_left_hand``. It exercises the agent -> tool path without relying
on an external LLM API.
"""

from __future__ import annotations

import argparse
import os
import shlex
import time
from pathlib import Path
from typing import Any, Sequence

import tomli
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from rai.agents.langchain import ReActAgent
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.tools.python import (
    get_unitree_g1_sim_tools,
    start_unitree_g1_sim_manager,
    stop_unitree_g1_sim_manager,
)
from rai.tools.python.unitree_g1_sim import _resolve_replay_file


DEFAULT_INSTRUCTION = "请让仿真里的 Unitree G1 向我挥左手。"


class ToolCallingFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict[str, Any]],
        **kwargs: Any,
    ) -> Runnable:
        return self


class CaptureConnector(HRIConnector[HRIMessage]):
    def __init__(self):
        super().__init__()
        self.messages: list[HRIMessage] = []

    def send_message(self, message: HRIMessage, target: str, **kwargs: Any) -> None:
        self.messages.append(message)
        print(f"[Agent Wave Test] Agent response to {target}: {message.text}")

    def receive_message(
        self,
        source: str,
        timeout_sec: float,
        **kwargs: Any,
    ) -> HRIMessage:
        raise NotImplementedError("CaptureConnector is send-only for this test.")


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
        description="Test that the text agent uses sim tools for wave_left_hand."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Natural-language instruction sent to the agent.",
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
        help="Seconds to wait for the replay player to stop.",
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the agent graph to finish.",
    )
    parser.add_argument(
        "--startup-settle-seconds",
        type=float,
        default=None,
        help="Delay between manager sim start and sending confirmation/control keys.",
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
        help="Stop the manager sim after the agent test.",
    )
    return parser.parse_args()


def build_manager_command() -> str:
    return "source scripts/setup_env.sh && bash deploy.sh --input-type manager sim"


def build_replay_command(latent_input_file: Path) -> str:
    return (
        "source .venv_teleop/bin/activate && "
        "python gear_sonic/scripts/sonic_encoder_input_player.py "
        f"--latent-input-file {shlex.quote(str(latent_input_file))}"
    )


def print_shell_command(label: str, cwd: str | Path, command: str) -> None:
    print(f"[Agent Wave Test] {label}:")
    print(f"  cd {Path(cwd).expanduser().resolve()}")
    print(f"  {command}")


def wait_for_agent(agent: ReActAgent, timeout: float) -> None:
    deadline = time.time() + max(1.0, float(timeout))
    while time.time() < deadline:
        if agent.ready():
            return
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for the agent to finish.")


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
    return_to_keyboard = not args.no_return_to_keyboard

    print("[Agent Wave Test] Configuration:")
    print(f"  instruction: {args.instruction}")
    print("  expected tool: unitree_g1_sim_perform_replay")
    print("  expected action: wave_left_hand")
    print(f"  deploy_dir: {deploy_dir}")
    print(f"  gr00t_root: {gr00t_root}")
    print(f"  replay_dir: {replay_dir}")

    replay_file = _resolve_replay_file(replay_dir, "wave_left_hand")
    print_shell_command(
        "Replay command expected from tool",
        gr00t_root,
        build_replay_command(replay_file),
    )

    if not args.no_auto_start:
        print_shell_command(
            "Manager command executed before agent instruction",
            deploy_dir,
            build_manager_command(),
        )
        if confirm_deployment:
            print("[Agent Wave Test] Startup sends 'Y' + ENTER, waits for Init Done.")
        print("[Agent Wave Test] Startup sends ']' after Init Done.")
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

    tool_call = {
        "name": "unitree_g1_sim_perform_replay",
        "args": {
            "action": "wave_left_hand",
            "wait": True,
            "timeout": args.timeout,
            "return_to_keyboard": return_to_keyboard,
        },
        "id": "call_wave_left_hand",
    }
    fake_llm = ToolCallingFakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="已调用仿真工具完成挥左手动作。"),
        ]
    )
    connector = CaptureConnector()
    agent = ReActAgent(
        target_connectors={"to_human": connector},
        llm=fake_llm,
        tools=tools,
        system_prompt=(
            "You are a robot control agent. Use the available Unitree G1 sim "
            "tools when the user asks for a robot motion."
        ),
        stream_response=False,
    )

    print("[Agent Wave Test] Sending instruction to agent:")
    print(f"  {args.instruction}")
    agent.run()
    try:
        agent(
            HRIMessage(
                text=args.instruction,
                message_author="human",
            )
        )
        wait_for_agent(agent, timeout=args.agent_timeout)
    finally:
        agent.stop()

    called_tools = [
        call["name"]
        for message in agent.state["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    print(f"[Agent Wave Test] Agent tool calls: {called_tools}")
    if "unitree_g1_sim_perform_replay" not in called_tools:
        raise RuntimeError("Agent did not call unitree_g1_sim_perform_replay.")

    if connector.messages:
        print("[Agent Wave Test] Final captured response:")
        print(connector.messages[-1].text)

    if args.stop_after:
        print("[Agent Wave Test] Stopping manager sim...")
        print(stop_unitree_g1_sim_manager(deploy_dir=deploy_dir))

    print("[Agent Wave Test] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
