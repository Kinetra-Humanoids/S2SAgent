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
from functools import lru_cache

from langchain_core.tools import BaseTool, tool


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@lru_cache(maxsize=4)
def _get_loco_client(network_interface: str):
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    except ImportError as e:
        raise RuntimeError(
            "Unitree SDK is not installed. Install unitree_sdk2_python and its "
            "CycloneDDS dependency before enabling Unitree G1 tools."
        ) from e

    if not network_interface:
        raise RuntimeError(
            "Missing Unitree network interface. Set UNITREE_G1_NETWORK_INTERFACE "
            "or configure it in the Streamlit configurator."
        )

    ChannelFactoryInitialize(0, network_interface)
    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()
    return client


def _control_enabled(configured_enabled: bool) -> bool:
    env_value = os.getenv("UNITREE_G1_ENABLE_CONTROL", "")
    return configured_enabled or env_value.lower() in {"1", "true", "yes", "on"}


def get_unitree_g1_tools(
    *,
    network_interface: str | None = None,
    control_enabled: bool = False,
    enabled_tools: list[str] | None = None,
) -> list[BaseTool]:
    interface = network_interface or os.getenv("UNITREE_G1_NETWORK_INTERFACE", "")
    enabled_tool_names = set(enabled_tools or ["stop", "move", "posture", "gesture"])

    def client():
        return _get_loco_client(interface)

    def require_control_enabled() -> None:
        if not _control_enabled(control_enabled):
            raise RuntimeError(
                "Unitree G1 control is disabled. Enable it in the configurator or set "
                "UNITREE_G1_ENABLE_CONTROL=true after confirming the robot area is safe."
            )

    @tool
    def unitree_g1_stop() -> str:
        """Stop Unitree G1 locomotion by sending zero velocity."""
        require_control_enabled()
        code = client().StopMove()
        return f"StopMove sent. SDK return code: {code}"

    @tool
    def unitree_g1_move(
        vx: float,
        vy: float,
        vyaw: float,
        duration: float = 1.0,
    ) -> str:
        """Move Unitree G1 with bounded velocity for a short duration.

        Args:
            vx: Forward velocity in m/s. Positive moves forward. Clamped to [-0.4, 0.4].
            vy: Lateral velocity in m/s. Positive moves left. Clamped to [-0.3, 0.3].
            vyaw: Yaw velocity in rad/s. Positive turns left. Clamped to [-0.6, 0.6].
            duration: Command duration in seconds. Clamped to [0.1, 2.0].
        """
        require_control_enabled()
        safe_vx = _clamp(float(vx), -0.4, 0.4)
        safe_vy = _clamp(float(vy), -0.3, 0.3)
        safe_vyaw = _clamp(float(vyaw), -0.6, 0.6)
        safe_duration = _clamp(float(duration), 0.1, 2.0)
        code = client().SetVelocity(safe_vx, safe_vy, safe_vyaw, safe_duration)
        return (
            "SetVelocity sent: "
            f"vx={safe_vx}, vy={safe_vy}, vyaw={safe_vyaw}, duration={safe_duration}. "
            f"SDK return code: {code}"
        )

    @tool
    def unitree_g1_posture(action: str) -> str:
        """Run a Unitree G1 posture command.

        Supported actions: damp, start, sit, squat_to_stand, lie_to_stand,
        stand_to_squat, high_stand, low_stand, zero_torque.
        """
        require_control_enabled()
        actions = {
            "damp": "Damp",
            "start": "Start",
            "sit": "Sit",
            "squat_to_stand": "Squat2StandUp",
            "lie_to_stand": "Lie2StandUp",
            "stand_to_squat": "StandUp2Squat",
            "high_stand": "HighStand",
            "low_stand": "LowStand",
            "zero_torque": "ZeroTorque",
        }
        method_name = actions.get(action)
        if method_name is None:
            raise ValueError(f"Unsupported G1 posture action: {action}")
        code = getattr(client(), method_name)()
        return f"{method_name} sent. SDK return code: {code}"

    @tool
    def unitree_g1_gesture(name: str) -> str:
        """Run a Unitree G1 gesture. Supported names: wave_hand, wave_hand_turn, shake_hand."""
        require_control_enabled()
        if name == "wave_hand":
            code = client().WaveHand(False)
        elif name == "wave_hand_turn":
            code = client().WaveHand(True)
        elif name == "shake_hand":
            code = client().ShakeHand()
        else:
            raise ValueError(f"Unsupported G1 gesture: {name}")
        return f"{name} sent. SDK return code: {code}"

    available_tools = {
        "stop": unitree_g1_stop,
        "move": unitree_g1_move,
        "posture": unitree_g1_posture,
        "gesture": unitree_g1_gesture,
    }
    return [
        tool
        for tool_name, tool in available_tools.items()
        if tool_name in enabled_tool_names
    ]
