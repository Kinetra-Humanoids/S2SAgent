"""Pure-Python Speech-to-Speech demo (no ROS2).

This runs the pipeline in-process:
  microphone -> VAD -> ASR -> LLM -> TTS -> speaker

Notes:
- Requires audio devices and `sounddevice`.
- Choose ASR/TTS backends based on what you have installed.
  - ASR: FasterWhisper/LocalWhisper run locally; OpenAIWhisper requires `OPENAI_API_KEY`.
  - TTS: KokoroTTS downloads an onnx model on first run.

Run:
  uv run python examples/s2s/s2s_no_ros.py --asr fasterwhisper --tts kokoro
  uv run python examples/s2s/s2s_no_ros.py --ros2-tools --nav2-tools --language zh

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any, Optional

import sounddevice as sd
import tomli
from langchain_core.tools import BaseTool

from rai.agents.langchain import ReActAgent
from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.initialization.model_initialization import get_llm_model, load_config
from rai.tools.python import (
    get_basic_tools,
    get_sensor_tools,
    get_unitree_g1_real_runtime_prompt,
    get_unitree_g1_real_tools,
    get_unitree_g1_sim_runtime_prompt,
    get_unitree_g1_sim_tools,
    get_unitree_g1_tools,
    start_unitree_g1_real_manager,
    start_unitree_g1_sim_manager,
    stop_unitree_g1_sim_manager,
)

from rai_s2s.asr.models import (
    BaseTranscriptionModel,
    DoubaoASR,
    OpenAIWhisper,
    SileroVAD,
)
from rai_s2s.s2s.agents import SpeechToSpeechAgent
from rai_s2s.sound_device import SoundDeviceConfig, SoundDeviceMessage
from rai_s2s.sound_device.unitree_g1_audio import UnitreeG1AudioPlayer
from rai_s2s.tts.models import DoubaoTTS, TTSModel


S2S_BASE_SYSTEM_PROMPT = """You are a concise voice assistant for a robot.
Reply in the same language as the user, using short, natural sentences suitable
for speech.

When the user asks about the robot's surroundings, what the robot can see, or a
visual inspection task, use the sensor camera tool if it is available.

For action requests, use tools only when the user has clearly asked the robot to
do something. Never claim an action was performed unless the corresponding tool
completed successfully. If a required tool is unavailable or fails, clearly say
that the action was not performed. Ask for clarification before acting when a
command is ambiguous. For non-action requests, answer normally without calling a
robot control tool.

If a tool returns an error observation, treat it as authoritative. If the error
says a tool precondition failed, that tool was not executed; use the reported
current state and allowed states to choose a valid next tool, or tell the user
what state transition is needed.
"""


POLICY_DELEGATE_SYSTEM_PROMPT = """You are the robot's visual and voice interaction
agent. Speak as the robot in the first person. A separate low-level policy agent
executes navigation, manipulation, grasping, and other physical policies; you do
not execute those policies yourself.

Reply in the same language as the user, using short, natural sentences suitable
for speech.

When a request depends on the robot's surroundings, visible objects, people,
locations, or an operation target, use the sensor camera tool to inspect the
current scene before choosing a target. Base visual claims only on the returned
camera observation. Identify targets precisely with useful attributes such as
object type, color, and relative position. If multiple plausible targets are
visible, ask the user which one they mean. If the target is not visible, say so
and ask for clarification rather than inventing it.

For a clear action request, describe the robot's accepted intent in the first
person. For example: "我将抓取桌面右侧的红色可乐罐。" or "I will pick up the
red cola can on the right side of the table." Do not say "the robot will", "I
suggest that you", or "wait for a human to execute it". In this interaction,
"I will ..." means that the request and target have been identified for the
separate policy agent; it does not mean the physical action has completed.

Do not call robot motion, navigation, manipulation, replay, VLA, or other
physical-control tools. Do not mention internal agents, policies, tools, models,
or system architecture to the user.

Keep execution states distinct:
- Before execution confirmation, say "I will ..." or "I am ready to ...".
- Say "I am ..." only after an execution-start status has been received.
- Say "I have ..." or that the task is complete only after an explicit success
  status has been received from the low-level policy agent.
- If an explicit failure status is received, truthfully say that the action
  failed and briefly state the reported reason when useful.

Normally respond in no more than two short sentences: first identify the target
when needed, then state what you will do in the first person.
"""


UNITREE_G1_SDK_TOOLS_PROMPT = """Unitree G1 SDK tool policy:
- These tools control a physical Unitree G1. Treat commands such as "stop",
  "停止", and "停下" as urgent and call the stop tool immediately.
- When the user explicitly asks the robot to move, stop, change posture, or
  perform a gesture, use the matching Unitree G1 tool if it is available.
- Refuse unsafe physical actions, but do not add unnecessary warnings to
  ordinary, clearly specified commands.
"""


UNITREE_G1_SIM_TOOLS_PROMPT = """Unitree G1 simulation tool policy:
- The robot is a GR00T WholeBodyControl Unitree G1 simulation controlled through
  the manager process. The runtime auto-starts `bash deploy.sh --input-type
  manager sim` and opens a control terminal when sim tools are enabled.
- To start deployment, call `unitree_g1_sim_confirm_deployment`; it sends `Y`
  and ENTER, waits for `Init Done`, then reports deployment completion. After
  deployment is ready, call `unitree_g1_sim_start_control` to send `]` and enter
  `keyboard_normal` before motion, replay, or VLA-style control.
- Prefer `unitree_g1_sim_perform_skill` for named motions and demonstrations,
  such as "wave left hand". If unsure which skills are configured, call
  `unitree_g1_sim_list_skills` first.
- `unitree_g1_sim_perform_skill` handles the full replay skill workflow: it
  is valid only from `keyboard_normal` or `keyboard_planner`; it switches to
  ZMQ mode with `#`, sends ENTER to enable streaming, runs the latent `.npy`
  replay player in a separate shell, and returns to keyboard mode after the
  replay finishes.
- `unitree_g1_sim_perform_replay` is a lower-level compatibility tool. It also
  is valid only from `zmq`; it sends ENTER to enable ZMQ streaming, runs the
  latent `.npy` replay player in a separate shell, and returns to keyboard mode
  after the replay finishes.
- Use lower-level keyboard tools only for simple manual controls such as
  forward/backward, turning, strafing, keyboard planner toggle, mode selection,
  speed, or height adjustment.
- Use `unitree_g1_sim_toggle_planner` only in keyboard mode. Use
  `unitree_g1_sim_toggle_zmq_streaming` only in ZMQ mode; it presses ENTER to
  toggle ZMQ streaming enabled/disabled. In `zmq_streaming`, pausing or
  disabling streaming is `unitree_g1_sim_toggle_zmq_streaming`.
- Treat "stop", "停止", and "停下" as urgent. Do not stop the manager terminal
  yourself. Use the keyboard `instant_stop` action only when the user
  specifically wants to halt planner momentum without exiting the manager.
  Never send `O`; that is reserved for the human operator in the terminal.
- If a replay `.npy` file is missing, say which replay file is missing and ask
  the user to add it to the configured replay directory.
"""


UNITREE_G1_REAL_TOOLS_PROMPT = """Unitree G1 real robot manager tool policy:
- These tools control a physical Unitree G1 through the GR00T WholeBodyControl
  manager process. Treat commands such as "stop", "停止", and "停下" as urgent,
  but do not stop the manager terminal yourself. Tell the user to use the
  visible terminal/operator controls for safety-critical stop actions.
- Prefer `unitree_g1_real_perform_skill` for clear, safe named motion requests.
  A real skill may be backed by a replay file or by a VLA model/prompt pair.
  Never claim a motion ran unless the tool completed successfully. Replay-backed skills are
  valid only from `keyboard_normal` or `keyboard_planner`; they switch to ZMQ
  with `#`, send ENTER to enable streaming, and run the replay player in a
  separate shell.
- VLA-backed skills are valid only from `keyboard_normal` or `keyboard_planner`;
  they switch to ZMQ streaming, start the configured GR00T server, then start
  the WBC VLA inference client with the configured prompt.
- `unitree_g1_real_perform_replay` is valid only from `zmq`; it sends ENTER to
  enable ZMQ streaming, then runs the replay player in a separate shell.
- The runtime can auto-start `source scripts/setup_env.sh && ./deploy.sh
  --input-type manager --zmq-host localhost --hand-type inspire real` and opens
  a control terminal when real tools are enabled.
- To start deployment, call `unitree_g1_real_confirm_deployment`; it sends `Y`
  and ENTER, waits for `Init Done`, then reports deployment completion. After
  deployment is ready, call `unitree_g1_real_start_control` to send `]` and enter
  `keyboard_normal` before replay or VLA skills.
- If unsure which real skills are configured, call
  `unitree_g1_real_list_skills` first.
- Use `unitree_g1_real_toggle_planner` only in keyboard mode. Use
  `unitree_g1_real_toggle_zmq_streaming` only in ZMQ mode; it presses ENTER to
  toggle ZMQ streaming enabled/disabled. In `zmq_streaming`, pausing or
  disabling streaming is `unitree_g1_real_toggle_zmq_streaming`.
- Never send `O`; that is reserved for the human operator in the terminal.
- If a replay file is missing, say which replay file is missing and ask the
  user to add it to the configured replay path.
"""


def build_system_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "agent_mode", "standard") == "policy_delegate":
        return POLICY_DELEGATE_SYSTEM_PROMPT

    prompt_parts = [S2S_BASE_SYSTEM_PROMPT]
    if args.unitree_g1_real_tools:
        prompt_parts.append(UNITREE_G1_REAL_TOOLS_PROMPT)
        prompt_parts.append(
            get_unitree_g1_real_runtime_prompt(
                deploy_dir=args.unitree_g1_real_deploy_dir,
                replay_dir=args.unitree_g1_real_replay_dir,
                skills=_parse_skills(args.unitree_g1_real_skills),
                enabled_tools=_parse_csv(args.unitree_g1_real_enabled_tools),
            )
        )
    elif args.unitree_g1_sim_tools:
        prompt_parts.append(UNITREE_G1_SIM_TOOLS_PROMPT)
        prompt_parts.append(
            get_unitree_g1_sim_runtime_prompt(
                deploy_dir=args.unitree_g1_sim_deploy_dir,
                replay_dir=args.unitree_g1_sim_replay_dir,
                skills=_parse_skills(args.unitree_g1_sim_skills),
                enabled_tools=_parse_csv(args.unitree_g1_sim_enabled_tools),
            )
        )
    elif args.unitree_g1_tools:
        prompt_parts.append(UNITREE_G1_SDK_TOOLS_PROMPT)
    return "\n\n".join(prompt_parts)


def _parse_csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _parse_skills(value: Any) -> Optional[list[dict[str, Any]]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("Skill JSON must be a list of skill objects.")
        return parsed
    raise TypeError(f"Unsupported skill config type: {type(value).__name__}")


def load_raw_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return tomli.load(file)


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


def config_get(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)


def apply_vla_server_root_to_skills(
    skills: Any,
    server_root: str,
) -> Any:
    if not isinstance(skills, list) or not server_root:
        return skills
    updated: list[dict[str, Any]] = []
    for raw_skill in skills:
        if not isinstance(raw_skill, dict):
            updated.append(raw_skill)
            continue
        skill = dict(raw_skill)
        if str(skill.get("source", "")).strip().lower() == "vla":
            skill["server_root"] = server_root
            skill.pop("wbc_root", None)
        updated.append(skill)
    return updated


def resolve_args(args: argparse.Namespace, raw_config: dict[str, Any]) -> argparse.Namespace:
    s2s = raw_config.get("s2s", {})
    tools = raw_config.get("tools", {})
    ros2 = raw_config.get("ros2", {})
    unitree_g1 = raw_config.get("unitree_g1", {})
    unitree_g1_sim = raw_config.get("unitree_g1_sim", {})
    unitree_g1_real = raw_config.get("unitree_g1_real", {})
    unitree_g1_audio = raw_config.get("unitree_g1_audio", {})
    sensor_tool = raw_config.get("sensor_tool", {})
    openai = raw_config.get("openai", {})
    tts = raw_config.get("tts", {})
    doubao = raw_config.get("doubao_speech", {})

    unitree_g1_real_skills = apply_vla_server_root_to_skills(
        unitree_g1_real.get("skills"),
        unitree_g1_real.get("vla_server_root", ""),
    )

    defaults = {
        "agent_mode": s2s.get("agent_mode", "standard"),
        "mic_device": s2s.get("mic_device", "default"),
        "speaker_device": s2s.get("speaker_device", "default"),
        "speaker_backend": s2s.get("speaker_backend", "sounddevice"),
        "asr": s2s.get("asr", "fasterwhisper"),
        "tts": s2s.get("tts", "kokoro"),
        "kokoro_voice": tts.get("voice", "af_sarah"),
        "openai_whisper_model": s2s.get("openai_whisper_model", "whisper-1"),
        "whisper_model": s2s.get("whisper_model", "tiny"),
        "openai_base_url": s2s.get("openai_base_url") or openai.get("base_url"),
        "doubao_app_id": doubao.get("app_id", ""),
        "doubao_token": doubao.get("token", ""),
        "doubao_asr_url": doubao.get("asr_url", ""),
        "doubao_asr_model": doubao.get("asr_model", "bigmodel"),
        "doubao_asr_auth_mode": doubao.get("asr_auth_mode", "auto"),
        "doubao_asr_resource_id": doubao.get("asr_resource_id", "volc.bigasr.auc_turbo"),
        "doubao_asr_app_key": doubao.get("asr_app_key", ""),
        "doubao_asr_access_key": doubao.get("asr_access_key", ""),
        "doubao_asr_api_key": doubao.get("asr_api_key", ""),
        "doubao_tts_url": doubao.get("tts_url", ""),
        "doubao_tts_cluster": doubao.get("tts_cluster", "volcano_tts"),
        "doubao_tts_voice_type": doubao.get("tts_voice_type", ""),
        "doubao_tts_encoding": doubao.get("tts_encoding", "wav"),
        "doubao_tts_sample_rate": doubao.get("tts_sample_rate", 24000),
        "doubao_tts_speed_ratio": doubao.get("tts_speed_ratio", 1.0),
        "language": s2s.get("language", "en"),
        "sample_rate": s2s.get("sample_rate", 16000),
        "vad_threshold": s2s.get("vad_threshold", 0.5),
        "block_size": s2s.get("block_size", 1280),
        "grace_period": s2s.get("grace_period", 0.8),
        "stream_response": s2s.get("stream_response", False),
        "python_tools": tools.get("python_tools", False),
        "unitree_g1_tools": unitree_g1.get("tools_enabled", False),
        "unitree_g1_enabled_tools": ",".join(
            unitree_g1.get(
                "enabled_tools",
                ["stop", "move", "posture", "gesture", "arm_action"],
            )
        ),
        "unitree_g1_network_interface": unitree_g1.get("network_interface", ""),
        "unitree_g1_control_enabled": unitree_g1.get("control_enabled", False),
        "unitree_g1_sim_tools": unitree_g1_sim.get("tools_enabled", False),
        "unitree_g1_sim_deploy_dir": unitree_g1_sim.get("deploy_dir", ""),
        "unitree_g1_sim_gr00t_root": unitree_g1_sim.get("gr00t_root", ""),
        "unitree_g1_sim_replay_dir": unitree_g1_sim.get(
            "replay_dir",
            "replays/unitree_g1_sim",
        ),
        "unitree_g1_sim_auto_start": unitree_g1_sim.get("auto_start", True),
        "unitree_g1_sim_confirm_deployment": unitree_g1_sim.get(
            "confirm_deployment",
            True,
        ),
        "unitree_g1_sim_start_control": unitree_g1_sim.get("start_control", True),
        "unitree_g1_sim_startup_settle_seconds": unitree_g1_sim.get(
            "startup_settle_seconds",
            2.0,
        ),
        "unitree_g1_sim_terminal_viewer": unitree_g1_sim.get(
            "terminal_viewer",
            True,
        ),
        "unitree_g1_sim_log_dir": unitree_g1_sim.get(
            "log_dir",
            "logs/unitree_g1_sim",
        ),
        "unitree_g1_sim_enabled_tools": ",".join(
            unitree_g1_sim.get(
                "enabled_tools",
                [
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
                ],
            )
        ),
        "unitree_g1_sim_skills": unitree_g1_sim.get("skills"),
        "unitree_g1_real_tools": unitree_g1_real.get("tools_enabled", False),
        "unitree_g1_real_deploy_dir": unitree_g1_real.get("deploy_dir", ""),
        "unitree_g1_real_gr00t_root": unitree_g1_real.get("gr00t_root", ""),
        "unitree_g1_real_replay_dir": unitree_g1_real.get(
            "replay_dir",
            "replays/unitree_g1_sim",
        ),
        "unitree_g1_real_auto_start": unitree_g1_real.get("auto_start", True),
        "unitree_g1_real_confirm_deployment": unitree_g1_real.get(
            "confirm_deployment",
            True,
        ),
        "unitree_g1_real_start_control": unitree_g1_real.get("start_control", True),
        "unitree_g1_real_startup_settle_seconds": unitree_g1_real.get(
            "startup_settle_seconds",
            2.0,
        ),
        "unitree_g1_real_terminal_viewer": unitree_g1_real.get(
            "terminal_viewer",
            True,
        ),
        "unitree_g1_real_log_dir": unitree_g1_real.get(
            "log_dir",
            "logs/unitree_g1_real",
        ),
        "unitree_g1_real_enabled_tools": ",".join(
            unitree_g1_real.get(
                "enabled_tools",
                [
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
                ],
            )
        ),
        "unitree_g1_real_skills": unitree_g1_real_skills,
        "unitree_g1_audio_enabled": unitree_g1_audio.get("enabled", False),
        "unitree_g1_audio_network_interface": unitree_g1_audio.get("network_interface", ""),
        "unitree_g1_audio_app_name": unitree_g1_audio.get("app_name", "rai_s2s"),
        "unitree_g1_audio_chunk_size": unitree_g1_audio.get("chunk_size", 96000),
        "unitree_g1_audio_stop_after_play": unitree_g1_audio.get("stop_after_play", False),
        "sensor_tools": sensor_tool.get("enabled", False),
        "sensor_transport": sensor_tool.get("transport", "zmq"),
        "sensor_server_ip": sensor_tool.get("server_ip", "localhost"),
        "sensor_server_port": sensor_tool.get("port", 5555),
        "sensor_http_base_url": sensor_tool.get("http_base_url", ""),
        "sensor_default_camera": sensor_tool.get("default_camera", ""),
        "sensor_blocking": sensor_tool.get("blocking", True),
        "ros2_tools": ros2.get("ros2_tools", False),
        "nav2_tools": ros2.get("nav2_tools", False),
        "ros2_readable": ros2.get("ros2_readable") or None,
        "ros2_writable": ros2.get("ros2_writable") or None,
        "ros2_forbidden": ros2.get("ros2_forbidden") or None,
        "ros2_node_name": ros2.get("ros2_node_name", "rai_s2s_tools"),
        "use_sim_time": ros2.get("use_sim_time", False),
        "nav2_frame_id": ros2.get("nav2_frame_id", "map"),
        "nav2_action_name": ros2.get("nav2_action_name", "navigate_to_pose"),
    }

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


class InProcessHRIConnector(HRIConnector[HRIMessage]):
    """In-memory pub/sub connector that satisfies the HRIConnector interface."""

    def __init__(self):
        super().__init__()

    def send_message(self, message: HRIMessage, target: str, **kwargs: Optional[Any]) -> None:
        # Deliver to callbacks registered for `target`.
        self.general_callback(target, message)

    def receive_message(self, source: str, timeout_sec: float, **kwargs: Optional[Any]) -> HRIMessage:
        raise NotImplementedError("InProcessHRIConnector is push-based only")

    def general_callback_preprocessor(self, message: Any) -> HRIMessage:
        if isinstance(message, HRIMessage):
            return message
        return HRIMessage(payload=message)


class LocalS2SAgent(SpeechToSpeechAgent):
    def __init__(
        self,
        *,
        from_human_topic: str,
        to_human_topic: str,
        connector: InProcessHRIConnector,
        text_agent: ReActAgent,
        microphone_config: SoundDeviceConfig,
        speaker_config: SoundDeviceConfig | None,
        vad: SileroVAD,
        asr: BaseTranscriptionModel,
        tts: TTSModel,
        unitree_audio_player: UnitreeG1AudioPlayer | None = None,
        grace_period: float = 0.8,
    ):
        self._connector = connector
        self._text_agent = text_agent
        self._unitree_audio_player = unitree_audio_player
        self._suppress_microphone_until = 0.0
        self._robot_audio_playing = threading.Event()
        super().__init__(
            from_human_topic,
            to_human_topic,
            microphone_config=microphone_config,
            speaker_config=speaker_config,
            transcription_model=asr,
            vad=vad,
            tts=tts,
            grace_period=grace_period,
            logger=logging.getLogger(__name__),
        )
        if self._unitree_audio_player is not None:
            self.tts_model.sample_rate = self._unitree_audio_player.sample_rate
            self.tts_model.channels = self._unitree_audio_player.channels

    def _setup_hri_connector(self) -> HRIConnector:
        self._connector.register_callback(self.to_human_topic, self._on_to_human_message)
        return self._connector

    def _send_from_human_message(self, data: str):
        # Feed transcription directly into the text agent.
        self._text_agent(HRIMessage(text=data))

    def _audio_gen_thread(self):
        if self._unitree_audio_player is None:
            return super()._audio_gen_thread()

        while not self.terminate_agent.wait(timeout=0.01):
            if self.current_transcription_id not in self.text_queues:
                continue
            try:
                data = self.text_queues[self.current_transcription_id].get(block=False)
            except Empty:
                continue
            if data.strip() == "":
                self.logger.debug("Skipping empty Unitree G1 TTS input.")
                continue
            try:
                self.logger.info("Generating Unitree G1 speech: %s", data)
                print(f"[Unitree Audio] TTS input: {data}", flush=True)
                audio = self.tts_model.get_speech(data)
            except Exception as exc:
                self.logger.exception("Failed to generate TTS audio for Unitree G1")
                print(f"[Unitree Audio Error] TTS failed: {exc}", flush=True)
                continue
            try:
                self.logger.info("Playing speech through Unitree G1 speaker")
                self._robot_audio_playing.set()
                self._unitree_audio_player.play(audio)
                print("[Unitree Audio] Playback completed", flush=True)
            except Exception as exc:
                self.logger.exception("Failed to play audio through Unitree G1")
                print(f"[Unitree Audio Error] Playback failed: {exc}", flush=True)
            finally:
                self._robot_audio_playing.clear()
                self._suppress_microphone_until = time.time() + 0.8
                self.playback_data.playing = False

    def run(self):
        if self._unitree_audio_player is None:
            return super().run()

        self.running = True
        self.logger.info("Starting SpeechToSpeechAgent with Unitree G1 speaker...")
        msg = SoundDeviceMessage(read=True)
        self.listener_handle = self.sound_connector.start_action(
            action_data=msg,
            target="microphone",
            on_feedback=self._on_microphone_sample,
            on_done=lambda: None,
        )
        self.logger.info("SpeechToSpeechAgent Started!")

    def _on_microphone_sample(self, indata, status_flags):
        if self._unitree_audio_player is not None and (
            self._robot_audio_playing.is_set()
            or time.time() < self._suppress_microphone_until
        ):
            with self.sample_buffer_lock:
                self.sample_buffer = []
            self.recording_started = False
            self.grace_period_start = 0
            return
        return super()._on_microphone_sample(indata, status_flags)

    def set_playback_state(self, state):
        super().set_playback_state(state)
        if state == "stop" and self._unitree_audio_player is not None:
            try:
                self._unitree_audio_player.stop()
            except Exception as exc:
                self.logger.warning("Failed to stop Unitree G1 audio: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speech-to-speech without ROS2",
        allow_abbrev=True,
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to config.toml (default: config.toml)",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["standard", "policy_delegate"],
        default=None,
        help="Agent behavior mode (default: [s2s].agent_mode)",
    )
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Microphone device name (default: [s2s].mic_device)",
    )
    parser.add_argument(
        "--speaker-device",
        default=None,
        help="Speaker device name (default: [s2s].speaker_device)",
    )
    parser.add_argument(
        "--speaker-backend",
        choices=["sounddevice", "unitree_g1"],
        default=None,
        help="Speaker backend (default: [s2s].speaker_backend)",
    )
    parser.add_argument(
        "--asr",
        choices=["fasterwhisper", "local", "openai", "doubao"],
        default=None,
        help="ASR backend (default: [s2s].asr)",
    )
    parser.add_argument(
        "--tts",
        choices=["kokoro", "doubao"],
        default=None,
        help="TTS backend (default: [s2s].tts)",
    )
    parser.add_argument(
        "--kokoro-voice",
        default=None,
        help="Kokoro voice name (default: [tts].voice)",
    )
    parser.add_argument(
        "--openai-whisper-model",
        default=None,
        help="OpenAI Whisper model name (default: [s2s].openai_whisper_model)",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Local/FasterWhisper model name or path (default: [s2s].whisper_model)",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible API base URL for ASR (default: [openai].base_url from config.toml)",
    )
    parser.add_argument("--doubao-app-id", default=None, help="Doubao app id")
    parser.add_argument("--doubao-token", default=None, help="Doubao token/access token")
    parser.add_argument("--doubao-asr-url", default=None, help="Doubao ASR endpoint URL")
    parser.add_argument("--doubao-asr-model", default=None, help="Doubao ASR model name")
    parser.add_argument(
        "--doubao-asr-auth-mode",
        choices=["auto", "new", "old"],
        default=None,
        help="Doubao ASR auth mode: new uses X-Api-Key, old uses X-Api-App-Key/X-Api-Access-Key",
    )
    parser.add_argument("--doubao-asr-resource-id", default=None, help="Doubao ASR resource id")
    parser.add_argument("--doubao-asr-app-key", default=None, help="Doubao ASR app key")
    parser.add_argument("--doubao-asr-access-key", default=None, help="Doubao ASR access key")
    parser.add_argument("--doubao-asr-api-key", default=None, help="Doubao ASR API key")
    parser.add_argument("--doubao-tts-url", default=None, help="Doubao TTS endpoint URL")
    parser.add_argument("--doubao-tts-cluster", default=None, help="Doubao TTS cluster")
    parser.add_argument("--doubao-tts-voice-type", default=None, help="Doubao TTS voice type")
    parser.add_argument("--doubao-tts-encoding", default=None, help="Doubao TTS audio encoding")
    parser.add_argument("--doubao-tts-sample-rate", type=int, default=None, help="Doubao TTS sample rate")
    parser.add_argument("--doubao-tts-speed-ratio", type=float, default=None, help="Doubao TTS speed ratio")
    parser.add_argument(
        "--language",
        default=None,
        help="ASR language (default: [s2s].language)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Microphone sampling rate for VAD/ASR (default: [s2s].sample_rate)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Silero VAD threshold (default: [s2s].vad_threshold)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Audio block size (default: [s2s].block_size)",
    )
    parser.add_argument(
        "--grace-period",
        type=float,
        default=None,
        help="Silence grace period before transcription (default: [s2s].grace_period)",
    )
    parser.add_argument(
        "--stream-response",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stream LLM response to TTS (default: [s2s].stream_response)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--python-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable non-ROS Python tools for the text agent",
    )
    parser.add_argument(
        "--unitree-g1-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Unitree G1 Python tools for the text agent",
    )
    parser.add_argument(
        "--unitree-g1-network-interface",
        default=None,
        help="Network interface connected to Unitree G1, e.g. en0 or eth0",
    )
    parser.add_argument(
        "--unitree-g1-enabled-tools",
        default=None,
        help=(
            "Comma-separated Unitree G1 tools to enable: "
            "stop,move,posture,gesture,arm_action"
        ),
    )
    parser.add_argument(
        "--unitree-g1-control-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow Unitree G1 tools to send movement/posture commands",
    )
    parser.add_argument(
        "--unitree-g1-sim-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable GR00T WholeBodyControl Unitree G1 sim manager tools",
    )
    parser.add_argument(
        "--unitree-g1-sim-deploy-dir",
        default=None,
        help="Path to GR00T gear_sonic_deploy directory containing deploy.sh",
    )
    parser.add_argument(
        "--unitree-g1-sim-gr00t-root",
        default=None,
        help="Path to GR00T-WholeBodyControl root for replay player",
    )
    parser.add_argument(
        "--unitree-g1-sim-replay-dir",
        default=None,
        help="Directory containing Unitree G1 sim replay .npy files",
    )
    parser.add_argument(
        "--unitree-g1-sim-auto-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-run deploy.sh manager sim when the agent starts",
    )
    parser.add_argument(
        "--unitree-g1-sim-confirm-deployment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send `Y` after starting deploy.sh to confirm deployment",
    )
    parser.add_argument(
        "--unitree-g1-sim-start-control",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="After auto-starting manager sim, send `]` to enter control mode",
    )
    parser.add_argument(
        "--unitree-g1-sim-startup-settle-seconds",
        type=float,
        default=None,
        help="Delay between manager sim start and sending `]`",
    )
    parser.add_argument(
        "--unitree-g1-sim-terminal-viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open a terminal window that tails Unitree G1 sim tool logs",
    )
    parser.add_argument(
        "--unitree-g1-sim-log-dir",
        default=None,
        help="Directory for Unitree G1 sim manager/replay terminal logs",
    )
    parser.add_argument(
        "--unitree-g1-sim-enabled-tools",
        default=None,
        help=(
            "Comma-separated sim manager tools to enable: "
            "confirm_deployment,perform_skill,perform_replay,"
            "list_skills,list_replays,start_control,switch_zmq,switch_keyboard,"
            "toggle_zmq_streaming,toggle_planner,keyboard,select_mode,adjust,"
            "compliance"
        ),
    )
    parser.add_argument(
        "--unitree-g1-sim-skills-json",
        dest="unitree_g1_sim_skills",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unitree-g1-real-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable GR00T WholeBodyControl Unitree G1 real manager tools",
    )
    parser.add_argument(
        "--unitree-g1-real-deploy-dir",
        default=None,
        help="Path to GR00T real deploy directory containing deploy.sh",
    )
    parser.add_argument(
        "--unitree-g1-real-gr00t-root",
        default=None,
        help="Path to GR00T-WholeBodyControl root for replay player",
    )
    parser.add_argument(
        "--unitree-g1-real-replay-dir",
        default=None,
        help="Directory containing Unitree G1 real replay files",
    )
    parser.add_argument(
        "--unitree-g1-real-auto-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-run deploy.sh manager real when the agent starts",
    )
    parser.add_argument(
        "--unitree-g1-real-confirm-deployment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send `Y` after starting real deploy.sh to confirm deployment",
    )
    parser.add_argument(
        "--unitree-g1-real-start-control",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="After auto-starting real manager, send `]` to enter control mode",
    )
    parser.add_argument(
        "--unitree-g1-real-startup-settle-seconds",
        type=float,
        default=None,
        help="Delay between real manager start and confirmation/control keys",
    )
    parser.add_argument(
        "--unitree-g1-real-terminal-viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open a terminal window that tails Unitree G1 real tool logs",
    )
    parser.add_argument(
        "--unitree-g1-real-log-dir",
        default=None,
        help="Directory for Unitree G1 real manager/replay terminal logs",
    )
    parser.add_argument(
        "--unitree-g1-real-enabled-tools",
        default=None,
        help=(
            "Comma-separated real manager tools to enable: "
            "confirm_deployment,perform_skill,perform_replay,"
            "list_skills,list_replays,start_control,switch_zmq,switch_keyboard,"
            "toggle_zmq_streaming,toggle_planner,keyboard,select_mode,adjust,"
            "compliance"
        ),
    )
    parser.add_argument(
        "--unitree-g1-real-skills-json",
        dest="unitree_g1_real_skills",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unitree-g1-audio-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Unitree G1 speaker output through AudioClient",
    )
    parser.add_argument(
        "--unitree-g1-audio-network-interface",
        default=None,
        help="Network interface connected to Unitree G1 speaker audio",
    )
    parser.add_argument(
        "--unitree-g1-audio-app-name",
        default=None,
        help="Unitree G1 audio app name",
    )
    parser.add_argument(
        "--unitree-g1-audio-chunk-size",
        type=int,
        default=None,
        help="Unitree G1 audio PCM chunk size",
    )
    parser.add_argument(
        "--unitree-g1-audio-stop-after-play",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Call PlayStop after each generated utterance",
    )
    parser.add_argument(
        "--sensor-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable camera sensor tools for the text agent",
    )
    parser.add_argument(
        "--sensor-transport",
        choices=["zmq", "http"],
        default=None,
        help="Camera sensor transport (default: [sensor_tool].transport)",
    )
    parser.add_argument(
        "--sensor-server-ip",
        default=None,
        help="ZMQ camera sensor server IP (default: [sensor_tool].server_ip)",
    )
    parser.add_argument(
        "--sensor-server-port",
        type=int,
        default=None,
        help="ZMQ camera sensor server port (default: [sensor_tool].port)",
    )
    parser.add_argument(
        "--sensor-http-base-url",
        default=None,
        help="HTTP MJPEG camera base URL, e.g. http://robot:8000",
    )
    parser.add_argument(
        "--sensor-default-camera",
        default=None,
        help="Default camera name for sensor tools, e.g. ego_view",
    )
    parser.add_argument(
        "--sensor-blocking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Wait for a fresh camera frame when using sensor tools",
    )
    parser.add_argument(
        "--ros2-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable generic ROS2 topic/service/action tools for the text agent",
    )
    parser.add_argument(
        "--nav2-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Nav2 navigation tools for the text agent",
    )
    parser.add_argument(
        "--ros2-readable",
        default=None,
        help="Comma-separated ROS2 names the agent may read (default: all when ROS2 tools are enabled)",
    )
    parser.add_argument(
        "--ros2-writable",
        default=None,
        help="Comma-separated ROS2 names the agent may write/call/start (default: all when ROS2 tools are enabled)",
    )
    parser.add_argument(
        "--ros2-forbidden",
        default=None,
        help="Comma-separated ROS2 names the agent must not access",
    )
    parser.add_argument(
        "--ros2-node-name",
        default=None,
        help="ROS2 node name for tool access (default: rai_s2s_tools)",
    )
    parser.add_argument(
        "--use-sim-time",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use ROS2 simulated time for tool connector",
    )
    parser.add_argument(
        "--nav2-frame-id",
        default=None,
        help="Nav2 frame id (default: [ros2].nav2_frame_id)",
    )
    parser.add_argument(
        "--nav2-action-name",
        default=None,
        help="Nav2 NavigateToPose action name (default: [ros2].nav2_action_name)",
    )
    return parser.parse_args()


def build_tools(args: argparse.Namespace) -> tuple[list[BaseTool], Any]:
    tools: list[BaseTool] = []
    if args.python_tools:
        tools.extend(get_basic_tools())
    if args.unitree_g1_tools:
        tools.extend(
            get_unitree_g1_tools(
                network_interface=args.unitree_g1_network_interface,
                control_enabled=args.unitree_g1_control_enabled,
                enabled_tools=_parse_csv(args.unitree_g1_enabled_tools),
            )
        )
    if args.unitree_g1_sim_tools:
        tools.extend(
            get_unitree_g1_sim_tools(
                deploy_dir=args.unitree_g1_sim_deploy_dir,
                gr00t_root=args.unitree_g1_sim_gr00t_root,
                replay_dir=args.unitree_g1_sim_replay_dir,
                enabled_tools=_parse_csv(args.unitree_g1_sim_enabled_tools),
                skills=_parse_skills(args.unitree_g1_sim_skills),
                terminal_viewer=args.unitree_g1_sim_terminal_viewer,
                log_dir=args.unitree_g1_sim_log_dir,
            )
        )
    if args.unitree_g1_real_tools:
        tools.extend(
            get_unitree_g1_real_tools(
                deploy_dir=args.unitree_g1_real_deploy_dir,
                gr00t_root=args.unitree_g1_real_gr00t_root,
                replay_dir=args.unitree_g1_real_replay_dir,
                enabled_tools=_parse_csv(args.unitree_g1_real_enabled_tools),
                skills=_parse_skills(args.unitree_g1_real_skills),
                terminal_viewer=args.unitree_g1_real_terminal_viewer,
                log_dir=args.unitree_g1_real_log_dir,
            )
        )
    if args.sensor_tools:
        tools.extend(
            get_sensor_tools(
                transport=args.sensor_transport,
                server_ip=args.sensor_server_ip,
                port=args.sensor_server_port,
                http_base_url=args.sensor_http_base_url,
                default_camera=args.sensor_default_camera,
                blocking=args.sensor_blocking,
            )
        )

    if not args.ros2_tools and not args.nav2_tools:
        return tools, None

    # Lazy imports keep the default audio-only demo usable without ROS2 sourced.
    from rai.communication.ros2 import ROS2Connector
    from rai.tools.ros2 import Nav2Toolkit, ROS2Toolkit

    connector = ROS2Connector(
        node_name=args.ros2_node_name,
        executor_type="multi_threaded",
        use_sim_time=args.use_sim_time,
    )
    readable = _parse_csv(args.ros2_readable)
    writable = _parse_csv(args.ros2_writable)
    forbidden = _parse_csv(args.ros2_forbidden)

    if args.ros2_tools:
        tools.extend(
            ROS2Toolkit(
                connector=connector,
                readable=readable,
                writable=writable,
                forbidden=forbidden,
            ).get_tools()
        )
    if args.nav2_tools:
        tools.extend(
            Nav2Toolkit(
                connector=connector,
                frame_id=args.nav2_frame_id,
                action_name=args.nav2_action_name,
            ).get_tools()
        )

    return tools, connector


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    load_env_file()
    args = parse_args()
    raw_config = load_raw_config(args.config)
    args = resolve_args(args, raw_config)

    if args.list_devices:
        print(sd.query_devices())
        print(f"Default input/output device ids: {sd.default.device}")
        return 0

    connector = InProcessHRIConnector()
    config = load_config(args.config)
    openai_base_url = args.openai_base_url or config.openai.base_url
    tools, ros2_connector = build_tools(args)
    if args.unitree_g1_sim_tools and args.unitree_g1_sim_auto_start:
        logging.info("Auto-starting Unitree G1 sim manager...")
        try:
            startup_message = start_unitree_g1_sim_manager(
                deploy_dir=args.unitree_g1_sim_deploy_dir,
                confirm_deployment=False,
                start_control=False,
                settle_seconds=args.unitree_g1_sim_startup_settle_seconds,
                terminal_viewer=True,
                log_dir=args.unitree_g1_sim_log_dir,
            )
            logging.info("Unitree G1 sim manager startup:\n%s", startup_message)
        except Exception:
            logging.exception("Failed to auto-start Unitree G1 sim manager")
            raise
    if args.unitree_g1_real_tools and args.unitree_g1_real_auto_start:
        logging.info("Auto-starting Unitree G1 real manager...")
        try:
            startup_message = start_unitree_g1_real_manager(
                deploy_dir=args.unitree_g1_real_deploy_dir,
                confirm_deployment=False,
                start_control=False,
                settle_seconds=args.unitree_g1_real_startup_settle_seconds,
                terminal_viewer=True,
                log_dir=args.unitree_g1_real_log_dir,
            )
            logging.info("Unitree G1 real manager startup:\n%s", startup_message)
        except Exception:
            logging.exception("Failed to auto-start Unitree G1 real manager")
            raise
    llm = get_llm_model(
        "complex_model",
        config_path=args.config,
        streaming=args.stream_response,
    )

    # Text agent: ReAct + configured vendor from config.toml
    # NOTE: This requires API key(s) according to the selected vendor in config.toml.

    system_prompt = build_system_prompt(args)
    text_agent = ReActAgent(
        target_connectors={"to_human": connector},
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        stream_response=args.stream_response,
    )

    mic_cfg = SoundDeviceConfig(
        stream=True,
        channels=1,
        device_name=args.mic_device,
        block_size=args.block_size,
        consumer_sampling_rate=args.sample_rate,
        dtype="int16",
        device_number=None,
        is_input=True,
        is_output=False,
    )
    use_unitree_speaker = (
        args.speaker_backend == "unitree_g1" or args.unitree_g1_audio_enabled
    )
    speaker_cfg = None
    if not use_unitree_speaker:
        speaker_cfg = SoundDeviceConfig(
            stream=True,
            is_output=True,
            device_name=args.speaker_device,
        )
    unitree_audio_player = None
    if use_unitree_speaker:
        unitree_audio_player = UnitreeG1AudioPlayer(
            network_interface=(
                args.unitree_g1_audio_network_interface
                or args.unitree_g1_network_interface
                or None
            ),
            app_name=args.unitree_g1_audio_app_name,
            chunk_size=args.unitree_g1_audio_chunk_size,
            stop_after_play=args.unitree_g1_audio_stop_after_play,
        )

    vad = SileroVAD(args.sample_rate, args.vad_threshold)
    match args.asr:
        case "fasterwhisper":
            from rai_s2s.asr.models import FasterWhisper

            asr = FasterWhisper(
                args.whisper_model,
                args.sample_rate,
                language=args.language,
            )
        case "local":
            from rai_s2s.asr.models import LocalWhisper

            asr = LocalWhisper(
                args.whisper_model,
                args.sample_rate,
                language=args.language,
            )
        case "openai":
            asr = OpenAIWhisper(
                args.openai_whisper_model,
                args.sample_rate,
                language=args.language,
                base_url=openai_base_url,
            )
        case "doubao":
            asr = DoubaoASR(
                args.doubao_asr_model,
                args.sample_rate,
                language=args.language,
                url=args.doubao_asr_url or None,
                app_key=args.doubao_asr_app_key or args.doubao_app_id or None,
                access_key=args.doubao_asr_access_key or args.doubao_token or None,
                api_key=args.doubao_asr_api_key or None,
                auth_mode=args.doubao_asr_auth_mode,
                resource_id=args.doubao_asr_resource_id,
            )
        case _:
            raise ValueError(f"Unknown ASR backend: {args.asr}")
    match args.tts:
        case "kokoro":
            from rai_s2s.tts.models import KokoroTTS

            tts = KokoroTTS(voice=args.kokoro_voice)
        case "doubao":
            tts = DoubaoTTS(
                app_id=args.doubao_app_id or None,
                token=args.doubao_token or None,
                cluster=args.doubao_tts_cluster,
                voice_type=args.doubao_tts_voice_type or None,
                url=args.doubao_tts_url or None,
                encoding=args.doubao_tts_encoding,
                audio_rate=args.doubao_tts_sample_rate,
                speed_ratio=args.doubao_tts_speed_ratio,
            )
        case _:
            raise ValueError(f"Unknown TTS backend: {args.tts}")

    s2s = LocalS2SAgent(
        from_human_topic="from_human",
        to_human_topic="to_human",
        connector=connector,
        text_agent=text_agent,
        microphone_config=mic_cfg,
        speaker_config=speaker_cfg,
        vad=vad,
        asr=asr,
        tts=tts,
        unitree_audio_player=unitree_audio_player,
        grace_period=args.grace_period,
    )

    text_agent.run()
    s2s.run()

    cleanup_started = threading.Event()

    def run_cleanup_step(name: str, func, timeout_sec: float = 5.0) -> None:
        error: list[BaseException] = []

        def target():
            try:
                func()
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=target, name=f"cleanup_{name}", daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)
        if thread.is_alive():
            logging.warning(
                "%s did not stop within %.1f seconds; continuing shutdown.",
                name,
                timeout_sec,
            )
            return
        if error:
            logging.warning("Error while stopping %s: %s", name, error[0])

    def cleanup(_signum=None, _frame=None):
        if cleanup_started.is_set():
            logging.warning("Shutdown already in progress; forcing exit.")
            os._exit(1)
        cleanup_started.set()
        logging.info("Stopping...")
        run_cleanup_step("speech-to-speech agent", s2s.stop, timeout_sec=5.0)
        run_cleanup_step("text agent", text_agent.stop, timeout_sec=5.0)
        if args.unitree_g1_sim_tools and args.unitree_g1_sim_auto_start:
            run_cleanup_step(
                "Unitree G1 sim manager",
                lambda: stop_unitree_g1_sim_manager(
                    deploy_dir=args.unitree_g1_sim_deploy_dir
                ),
                timeout_sec=8.0,
            )
        if ros2_connector is not None:
            run_cleanup_step("ROS2 connector", ros2_connector.shutdown, timeout_sec=5.0)
        logging.info("Stopped.")
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Keep process alive.
    while True:
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
