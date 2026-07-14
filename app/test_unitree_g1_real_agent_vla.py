"""Agent-level smoke test for the Unitree G1 real VLA skill flow.

Run from ``dist/s2s_agent_bundle``:

    uv run python app/test_unitree_g1_real_agent_vla.py

The test uses a deterministic fake chat model and fake Unitree/VLA process
runners. It does not start the real robot manager, GR00T server, or VLA
inference process. It verifies the agent/tool flow:

    confirm_deployment -> start_control -> perform_skill(VLA)

and confirms that a VLA skill is rejected before ``]`` enters keyboard_normal.
"""

from __future__ import annotations

import argparse
import os
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
from rai.tools.python import unitree_g1_real as real_tools
from rai.tools.python.unitree_g1_sim import ToolStateError, UnitreeG1RuntimeState


DEFAULT_INSTRUCTION = "请让真实 Unitree G1 执行 VLA 技能。"
DEFAULT_VLA_SKILL = "wave left hand with vla"
DEFAULT_VLA_SERVER_ROOT = "/home/zj/Isaac-GR00T"
DEFAULT_WBC_ROOT = "/home/zj/GR00T-WholeBodyControl"


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
        print(f"[Real VLA Agent Test] Agent response to {target}: {message.text}")

    def receive_message(
        self,
        source: str,
        timeout_sec: float,
        **kwargs: Any,
    ) -> HRIMessage:
        raise NotImplementedError("CaptureConnector is send-only for this test.")


class FakeRealManager:
    def __init__(self):
        self.state = UnitreeG1RuntimeState("real")
        self.events: list[str] = []

    def confirm_deployment(self, timeout: float = 60.0) -> str:
        self.events.append("confirm_deployment")
        self.state.update(deployment="ready", last_error="")
        return "\n".join(
            [
                "Fake deployment confirmation: sent Y and ENTER.",
                "Deployment完成. Current control_mode remains pre_control.",
                self.state.summary(),
            ]
        )

    def send_key(
        self,
        key: str,
        label: str,
        repeats: int = 1,
        interval: float = 0.15,
    ) -> str:
        self.events.append(f"send_key:{label}:{key}")
        return f"Fake sent {label}: key={key!r}, repeats={repeats}"

    def prepare_zmq_streaming(self) -> str:
        state = self.state.snapshot()
        if state["control_mode"] not in {"keyboard_normal", "keyboard_planner"}:
            raise ToolStateError(
                f"Cannot prepare VLA streaming from {state['control_mode']}"
            )
        self.events.append("prepare_zmq_streaming")
        self.state.update(control_mode="zmq_streaming", last_error="")
        return "Fake switched manager to ZMQ interface and enabled streaming."

    def status(self, tail_lines: int = 20) -> str:
        return "Fake real manager running."


class FakeVLARunner:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def start(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return "\n".join(
            [
                "Fake started VLA server.",
                "Fake started VLA inference.",
                f"VLA prompt: {kwargs.get('prompt', '')}",
            ]
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
        description="Test that the agent follows the real VLA skill state flow."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--skill", default=DEFAULT_VLA_SKILL)
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the agent graph to finish.",
    )
    parser.add_argument(
        "--skip-invalid-order-check",
        action="store_true",
        help="Do not first assert that VLA is rejected from pre_control.",
    )
    return parser.parse_args()


def find_tool(tools: Sequence[BaseTool], name: str) -> BaseTool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise RuntimeError(f"Tool not found: {name}")


def wait_for_agent(agent: ReActAgent, timeout: float) -> None:
    deadline = time.time() + max(1.0, float(timeout))
    while time.time() < deadline:
        if agent.ready():
            return
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for the agent to finish.")


def first_vla_skill(skills: list[dict[str, Any]], skill_name: str) -> dict[str, Any]:
    normalized = skill_name.strip().lower()
    for skill in skills:
        if (
            str(skill.get("name", "")).strip().lower() == normalized
            and str(skill.get("source", "")).strip().lower() == "vla"
        ):
            return skill
    raise RuntimeError(f"VLA skill not found in config: {skill_name!r}")


def apply_vla_server_root(
    skills: list[dict[str, Any]],
    server_root: str,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for raw_skill in skills:
        skill = dict(raw_skill)
        if str(skill.get("source", "")).strip().lower() == "vla":
            skill["server_root"] = server_root
            skill.pop("wbc_root", None)
        updated.append(skill)
    return updated


def invoke_tool(tool: BaseTool, args: dict[str, Any]) -> Any:
    print(f"[Real VLA Agent Test] Tool call: {tool.name}")
    print(f"  args: {args}")
    return tool.invoke(args)


def main() -> int:
    args = parse_args()
    load_env_file()
    raw_config = load_raw_config(args.config)
    real_config = raw_config.get("unitree_g1_real", {})

    gr00t_root = (
        real_config.get("gr00t_root")
        or os.getenv("GR00T_WBC_ROOT")
        or DEFAULT_WBC_ROOT
    )
    vla_server_root = (
        real_config.get("vla_server_root")
        or os.getenv("ISAAC_GROOT_ROOT")
        or DEFAULT_VLA_SERVER_ROOT
    )
    skills = real_config.get("skills") or real_tools.DEFAULT_UNITREE_G1_REAL_SKILLS
    skills = apply_vla_server_root(skills, vla_server_root)
    selected_skill = first_vla_skill(skills, args.skill)

    print("[Real VLA Agent Test] Configuration:")
    print(f"  instruction: {args.instruction}")
    print(f"  expected skill: {args.skill}")
    print(f"  vla_server_root: {vla_server_root}")
    print(f"  gr00t_root / VLA inference root: {gr00t_root}")
    print(f"  model_path: {selected_skill.get('model_path', '')}")
    print(f"  prompt: {selected_skill.get('prompt', '')}")

    fake_manager = FakeRealManager()
    fake_vla_runner = FakeVLARunner()
    original_get_manager = real_tools._get_manager
    original_get_vla_runner = real_tools._get_vla_runner
    real_tools._get_manager = lambda *call_args, **call_kwargs: fake_manager
    real_tools._get_vla_runner = lambda *call_args, **call_kwargs: fake_vla_runner

    try:
        tools = real_tools.get_unitree_g1_real_tools(
            deploy_dir=real_config.get("deploy_dir", ""),
            gr00t_root=gr00t_root,
            replay_dir=real_config.get("replay_dir", "replays/unitree_g1_sim"),
            enabled_tools=[
                "confirm_deployment",
                "start_control",
                "perform_skill",
                "status",
                "list_skills",
            ],
            skills=skills,
        )

        perform_skill_tool = find_tool(tools, "unitree_g1_real_perform_skill")
        if not args.skip_invalid_order_check:
            print("[Real VLA Agent Test] Checking invalid pre_control -> VLA rejection...")
            try:
                invoke_tool(perform_skill_tool, {"skill": args.skill})
            except ToolStateError as exc:
                print("[Real VLA Agent Test] Rejected as expected:")
                print(exc)
            else:
                raise RuntimeError("VLA skill unexpectedly succeeded from pre_control.")

        tool_calls = [
            {
                "name": "unitree_g1_real_confirm_deployment",
                "args": {"timeout": 60.0},
                "id": "call_confirm_deployment",
            },
            {
                "name": "unitree_g1_real_start_control",
                "args": {},
                "id": "call_start_control",
            },
            {
                "name": "unitree_g1_real_perform_skill",
                "args": {
                    "skill": args.skill,
                    "wait": True,
                    "timeout": 60.0,
                    "return_to_keyboard": True,
                },
                "id": "call_perform_vla_skill",
            },
        ]
        fake_llm = ToolCallingFakeChatModel(
            responses=[
                AIMessage(content="", tool_calls=tool_calls),
                AIMessage(content="已按流程启动 VLA 技能。"),
            ]
        )
        connector = CaptureConnector()
        agent = ReActAgent(
            target_connectors={"to_human": connector},
            llm=fake_llm,
            tools=tools,
            system_prompt=(
                "You are a robot control agent. To play a Unitree G1 real VLA "
                "skill, confirm deployment, send ] via start_control, then call "
                "the configured VLA skill."
            ),
            stream_response=False,
        )

        print("[Real VLA Agent Test] Sending instruction to agent:")
        print(f"  {args.instruction}")
        agent.run()
        try:
            agent(HRIMessage(text=args.instruction, message_author="human"))
            wait_for_agent(agent, timeout=args.agent_timeout)
        finally:
            agent.stop()

        called_tools = [
            call["name"]
            for message in agent.state["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        expected_tools = [
            "unitree_g1_real_confirm_deployment",
            "unitree_g1_real_start_control",
            "unitree_g1_real_perform_skill",
        ]
        print(f"[Real VLA Agent Test] Agent tool calls: {called_tools}")
        if called_tools[:3] != expected_tools:
            raise RuntimeError(
                f"Unexpected tool order. Expected {expected_tools}, got {called_tools}"
            )

        final_state = fake_manager.state.snapshot()
        print(f"[Real VLA Agent Test] Final fake runtime state: {final_state}")
        if final_state["deployment"] != "ready":
            raise RuntimeError("Deployment did not reach ready.")
        if final_state["control_mode"] != "zmq_streaming":
            raise RuntimeError("VLA flow did not end in zmq_streaming.")
        if final_state["last_skill"] != args.skill:
            raise RuntimeError("Runtime state did not record the selected VLA skill.")
        if not fake_vla_runner.calls:
            raise RuntimeError("VLA runner was not started.")

        vla_call = fake_vla_runner.calls[-1]
        print("[Real VLA Agent Test] Captured VLA runner call:")
        for key in [
            "server_root",
            "wbc_root",
            "model_path",
            "prompt",
            "embodiment_tag",
            "device",
            "port",
            "camera_host",
            "camera_port",
        ]:
            print(f"  {key}: {vla_call.get(key)}")
        if vla_call.get("server_root") != vla_server_root:
            raise RuntimeError("VLA server_root was not propagated.")
        if vla_call.get("wbc_root") != gr00t_root:
            raise RuntimeError("VLA inference root did not match real gr00t_root.")
        if vla_call.get("model_path") != selected_skill.get("model_path", ""):
            raise RuntimeError("VLA model_path was not propagated.")
        if vla_call.get("prompt") != selected_skill.get("prompt", args.skill):
            raise RuntimeError("VLA prompt was not propagated.")

        if connector.messages:
            print("[Real VLA Agent Test] Final captured response:")
            print(connector.messages[-1].text)

        print("[Real VLA Agent Test] Done.")
        return 0
    finally:
        real_tools._get_manager = original_get_manager
        real_tools._get_vla_runner = original_get_vla_runner


if __name__ == "__main__":
    raise SystemExit(main())
