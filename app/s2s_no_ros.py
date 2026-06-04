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
import logging
import os
import signal
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
from rai.tools.python import get_basic_tools, get_unitree_g1_tools

from rai_s2s.asr.models import (
    BaseTranscriptionModel,
    DoubaoASR,
    FasterWhisper,
    LocalWhisper,
    OpenAIWhisper,
    SileroVAD,
)
from rai_s2s.s2s.agents import SpeechToSpeechAgent
from rai_s2s.sound_device import SoundDeviceConfig, SoundDeviceMessage
from rai_s2s.sound_device.unitree_g1_audio import UnitreeG1AudioPlayer
from rai_s2s.tts.models import DoubaoTTS, KokoroTTS, TTSModel


def _parse_csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


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


def resolve_args(args: argparse.Namespace, raw_config: dict[str, Any]) -> argparse.Namespace:
    s2s = raw_config.get("s2s", {})
    tools = raw_config.get("tools", {})
    ros2 = raw_config.get("ros2", {})
    unitree_g1 = raw_config.get("unitree_g1", {})
    unitree_g1_audio = raw_config.get("unitree_g1_audio", {})
    openai = raw_config.get("openai", {})
    tts = raw_config.get("tts", {})
    doubao = raw_config.get("doubao_speech", {})

    defaults = {
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
            unitree_g1.get("enabled_tools", ["stop", "move", "posture", "gesture"])
        ),
        "unitree_g1_network_interface": unitree_g1.get("network_interface", ""),
        "unitree_g1_control_enabled": unitree_g1.get("control_enabled", False),
        "unitree_g1_audio_enabled": unitree_g1_audio.get("enabled", False),
        "unitree_g1_audio_network_interface": unitree_g1_audio.get("network_interface", ""),
        "unitree_g1_audio_app_name": unitree_g1_audio.get("app_name", "rai_s2s"),
        "unitree_g1_audio_chunk_size": unitree_g1_audio.get("chunk_size", 96000),
        "unitree_g1_audio_stop_after_play": unitree_g1_audio.get("stop_after_play", False),
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
            audio = self.tts_model.get_speech(data)
            self._unitree_audio_player.play(audio)

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
        help="Comma-separated Unitree G1 tools to enable: stop,move,posture,gesture",
    )
    parser.add_argument(
        "--unitree-g1-control-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow Unitree G1 tools to send movement/posture commands",
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
    llm = get_llm_model(
        "complex_model",
        config_path=args.config,
        streaming=args.stream_response,
    )

    # Text agent: ReAct + configured vendor from config.toml
    # NOTE: This requires API key(s) according to the selected vendor in config.toml.
    text_agent = ReActAgent(
        target_connectors={"to_human": connector},
        llm=llm,
        tools=tools,
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
            asr = FasterWhisper(
                args.whisper_model,
                args.sample_rate,
                language=args.language,
            )
        case "local":
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

    def cleanup(_signum=None, _frame=None):
        logging.info("Stopping...")
        try:
            s2s.stop()
        finally:
            text_agent.stop()
            if ros2_connector is not None:
                ros2_connector.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Keep process alive.
    while True:
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
