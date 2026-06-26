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
import time
from functools import lru_cache

from langchain_core.tools import BaseTool, tool


DEFAULT_UNITREE_G1_TOOLS = ["stop", "move", "posture", "gesture", "arm_action"]
ARM_ACTION_NAMES = [
    "release arm",
    "two-hand kiss",
    "left kiss",
    "right kiss",
    "hands up",
    "clap",
    "high five",
    "hug",
    "heart",
    "right heart",
    "reject",
    "right hand up",
    "x-ray",
    "face wave",
    "high wave",
    "shake hand",
]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@lru_cache(maxsize=4)
def _initialize_channel(network_interface: str) -> None:
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
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


@lru_cache(maxsize=4)
def _get_loco_client(network_interface: str):
    try:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    except ImportError as e:
        raise RuntimeError(
            "Unitree SDK is not installed. Install unitree_sdk2_python and its "
            "CycloneDDS dependency before enabling Unitree G1 tools."
        ) from e

    _initialize_channel(network_interface)
    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()
    return client


@lru_cache(maxsize=4)
def _get_arm_action_client(network_interface: str):
    try:
        from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
    except ImportError as e:
        raise RuntimeError(
            "Unitree SDK is not installed. Install unitree_sdk2_python and its "
            "CycloneDDS dependency before enabling Unitree G1 arm actions."
        ) from e

    _initialize_channel(network_interface)
    client = G1ArmActionClient()
    client.SetTimeout(10.0)
    client.Init()
    return client


def _control_enabled(configured_enabled: bool) -> bool:
    env_value = os.getenv("UNITREE_G1_ENABLE_CONTROL", "")
    return configured_enabled or env_value.lower() in {"1", "true", "yes", "on"}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _call_loco(client, method_name: str, *args):
    method = getattr(client, method_name, None)
    if method is None:
        raise RuntimeError(f"Installed Unitree G1 LocoClient has no {method_name} method.")
    return method(*args)


def get_unitree_g1_tools(
    *,
    network_interface: str | None = None,
    control_enabled: bool = False,
    enabled_tools: list[str] | None = None,
) -> list[BaseTool]:
    interface = network_interface or os.getenv("UNITREE_G1_NETWORK_INTERFACE", "")
    enabled_tool_names = set(enabled_tools or DEFAULT_UNITREE_G1_TOOLS)

    def client():
        return _get_loco_client(interface)

    def arm_client():
        return _get_arm_action_client(interface)

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

        Supported actions: damp, start, sit, squat, stand_up, squat_to_stand,
        lie_to_stand, stand_to_squat, high_stand, low_stand, balance_stand,
        zero_torque.
        """
        require_control_enabled()
        normalized_action = _normalize_name(action).replace(" ", "_")
        actions = {
            "damp": "Damp",
            "start": "Start",
            "sit": "Sit",
            "squat": "Squat",
            "stand_up": "StandUp",
            "squat_to_stand": "Squat2StandUp",
            "lie_to_stand": "Lie2StandUp",
            "stand_to_squat": "StandUp2Squat",
            "high_stand": "HighStand",
            "low_stand": "LowStand",
            "balance_stand": "BalanceStand",
            "zero_torque": "ZeroTorque",
        }
        method_name = actions.get(normalized_action)
        if method_name is None:
            raise ValueError(f"Unsupported G1 posture action: {action}")
        loco_client = client()
        if method_name == "Squat" and not hasattr(loco_client, method_name):
            code = loco_client.SetFsmId(2)
        elif method_name == "StandUp" and not hasattr(loco_client, method_name):
            code = loco_client.SetFsmId(4)
        elif method_name == "BalanceStand":
            code = _call_loco(loco_client, method_name, 0)
        else:
            code = _call_loco(loco_client, method_name)
        return f"{method_name} sent. SDK return code: {code}"

    @tool
    def unitree_g1_gesture(name: str) -> str:
        """Run a Unitree G1 loco gesture.

        Supported names: wave_hand, wave_hand_turn, shake_hand, shake_hand_start,
        shake_hand_finish.
        """
        require_control_enabled()
        normalized_name = _normalize_name(name).replace(" ", "_")
        if normalized_name == "wave_hand":
            code = client().WaveHand(False)
        elif normalized_name in {"wave_hand_turn", "wave_hand_with_turn"}:
            code = client().WaveHand(True)
        elif normalized_name == "shake_hand":
            code = client().ShakeHand()
        elif normalized_name == "shake_hand_start":
            code = client().ShakeHand(0)
        elif normalized_name in {"shake_hand_finish", "shake_hand_end"}:
            code = client().ShakeHand(1)
        else:
            raise ValueError(f"Unsupported G1 gesture: {name}")
        return f"{name} sent. SDK return code: {code}"

    @tool
    def unitree_g1_arm_action(
        action: str,
        release_after_seconds: float = 0.0,
    ) -> str:
        """Run a built-in Unitree G1 arm action.

        Supported actions: release arm, two-hand kiss, left kiss, right kiss,
        hands up, clap, high five, hug, heart, right heart, reject, right hand up,
        x-ray, face wave, high wave, shake hand.

        Args:
            action: Arm action name. Spaces, underscores, and hyphens are accepted.
            release_after_seconds: Optionally release the arms after this delay. Clamped
                to [0.0, 5.0] seconds. Use 0 to leave the action running naturally.
        """
        require_control_enabled()
        try:
            from unitree_sdk2py.g1.arm.g1_arm_action_client import action_map
        except ImportError as e:
            raise RuntimeError(
                "Unitree G1 arm action support is unavailable in this SDK install."
            ) from e

        normalized_actions = {
            _normalize_name(action_name): action_id
            for action_name, action_id in action_map.items()
        }
        normalized_action = _normalize_name(action)
        action_id = normalized_actions.get(normalized_action)
        if action_id is None:
            supported = ", ".join(ARM_ACTION_NAMES)
            raise ValueError(f"Unsupported G1 arm action: {action}. Supported: {supported}")

        g1_arm_client = arm_client()
        code = g1_arm_client.ExecuteAction(action_id)
        safe_delay = _clamp(float(release_after_seconds), 0.0, 5.0)
        release_code = None
        if safe_delay > 0.0 and normalized_action != "release arm":
            time.sleep(safe_delay)
            release_code = g1_arm_client.ExecuteAction(normalized_actions["release arm"])

        result = f"{action} arm action sent. SDK return code: {code}"
        if release_code is not None:
            result += f"; release arm return code: {release_code}"
        return result

    available_tools = {
        "stop": unitree_g1_stop,
        "move": unitree_g1_move,
        "posture": unitree_g1_posture,
        "gesture": unitree_g1_gesture,
        "arm_action": unitree_g1_arm_action,
    }
    return [
        tool
        for tool_name, tool in available_tools.items()
        if tool_name in enabled_tool_names
    ]
