"""Test sensor camera -> vision LLM -> TTS playback.

Run from ``dist/s2s_agent_bundle``:

    uv run python app/test_sensor_llm_tts.py --config config.toml

This bypasses ASR and the full speech agent. It directly calls the sensor tool,
sends the returned image artifact to the configured LLM, then speaks the LLM
reply using either the configured Unitree G1 speaker or local sounddevice output.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import tomli
from langchain_core.messages import SystemMessage
from pydub import AudioSegment

from rai.initialization.model_initialization import get_llm_model
from rai.messages import HumanMultimodalMessage
from rai.tools.python import get_sensor_tools
from rai_s2s.sound_device.unitree_g1_audio import UnitreeG1AudioPlayer
from rai_s2s.tts.models import DoubaoTTS, TTSModel


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
    with Path(config_path).open("rb") as file:
        return tomli.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test sensor camera -> LLM vision -> TTS playback."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--question",
        default="请简短描述机器人当前看到的画面。",
        help="Prompt sent to the vision LLM with the camera frame.",
    )
    parser.add_argument("--camera-name", default=None, help="Optional camera name.")
    parser.add_argument("--sensor-transport", choices=["zmq", "http"], default=None)
    parser.add_argument("--sensor-server-ip", default=None)
    parser.add_argument("--sensor-server-port", type=int, default=None)
    parser.add_argument("--sensor-http-base-url", default=None)
    parser.add_argument("--no-blocking", action="store_true")
    parser.add_argument(
        "--playback",
        choices=["config", "unitree_g1", "sounddevice", "none"],
        default="config",
        help="Where to play the LLM reply.",
    )
    return parser.parse_args()


def _as_tool_result(value: Any) -> tuple[str, dict[str, list[str]]]:
    if isinstance(value, tuple) and len(value) == 2:
        content, artifact = value
        return str(content), artifact
    artifact = getattr(value, "artifact", None)
    content = getattr(value, "content", value)
    if artifact is None:
        raise RuntimeError(
            "Sensor tool did not return an image artifact. "
            f"Raw result type: {type(value).__name__}"
        )
    return str(content), artifact


def call_sensor_tool(args: argparse.Namespace, raw_config: dict[str, Any]):
    sensor_cfg = raw_config.get("sensor_tool", {})
    tool = get_sensor_tools(
        transport=args.sensor_transport or sensor_cfg.get("transport", "zmq"),
        server_ip=args.sensor_server_ip or sensor_cfg.get("server_ip", "localhost"),
        port=args.sensor_server_port or sensor_cfg.get("port", 5555),
        http_base_url=args.sensor_http_base_url or sensor_cfg.get("http_base_url", ""),
        default_camera=args.camera_name or sensor_cfg.get("default_camera", ""),
        blocking=not args.no_blocking and sensor_cfg.get("blocking", True),
    )[0]
    print(f"[Test] Calling sensor tool: {tool.name}", flush=True)
    result = tool.invoke(
        {
            "camera_name": args.camera_name or "",
            "blocking": not args.no_blocking and sensor_cfg.get("blocking", True),
        }
    )
    content, artifact = _as_tool_result(result)
    images = artifact.get("images", [])
    if not images:
        raise RuntimeError(f"Sensor tool returned no images. Content: {content}")
    print(f"[Test] Sensor content:\n{content}", flush=True)
    print(f"[Test] Sensor returned {len(images)} image(s)", flush=True)
    return content, images


def ask_llm(config_path: str, question: str, sensor_content: str, images: list[str]) -> str:
    llm = get_llm_model("complex_model", config_path=config_path, streaming=False)
    messages = [
        SystemMessage(
            content=(
                "You are testing a robot vision pipeline. Answer concisely in the "
                "same language as the user."
            )
        ),
        HumanMultimodalMessage(
            content=f"{question}\n\nSensor tool result:\n{sensor_content}",
            images=images,
        ),
    ]
    print("[Test] Sending sensor image(s) to LLM...", flush=True)
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ]
        return "".join(text_parts).strip()
    return str(content).strip()


def build_tts(raw_config: dict[str, Any]) -> TTSModel:
    s2s = raw_config.get("s2s", {})
    doubao = raw_config.get("doubao_speech", {})
    tts_name = s2s.get("tts", "doubao")
    if tts_name == "doubao":
        return DoubaoTTS(
            app_id=doubao.get("app_id") or None,
            token=doubao.get("token") or None,
            cluster=doubao.get("tts_cluster", "volcano_tts"),
            voice_type=doubao.get("tts_voice_type") or None,
            url=doubao.get("tts_url") or None,
            encoding=doubao.get("tts_encoding", "wav"),
            audio_rate=doubao.get("tts_sample_rate", 24000),
            speed_ratio=doubao.get("tts_speed_ratio", 1.0),
        )
    if tts_name == "kokoro":
        from rai_s2s.tts.models import KokoroTTS

        return KokoroTTS(voice=raw_config.get("tts", {}).get("voice", "af_sarah"))
    raise ValueError(f"Unsupported TTS backend for test: {tts_name}")


def play_sounddevice(audio: AudioSegment, device_name: str | None = None) -> None:
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    samples = samples.astype(np.float32) / float(1 << (8 * audio.sample_width - 1))
    print(
        f"[Test] Playing with sounddevice: rate={audio.frame_rate}, "
        f"channels={audio.channels}, device={device_name or 'default'}",
        flush=True,
    )
    sd.play(samples, samplerate=audio.frame_rate, device=device_name)
    sd.wait()


def play_reply(
    reply: str,
    args: argparse.Namespace,
    raw_config: dict[str, Any],
) -> None:
    if args.playback == "none":
        print("[Test] Playback disabled.", flush=True)
        return

    s2s = raw_config.get("s2s", {})
    unitree_audio = raw_config.get("unitree_g1_audio", {})
    use_unitree = args.playback == "unitree_g1" or (
        args.playback == "config"
        and (s2s.get("speaker_backend") == "unitree_g1" or unitree_audio.get("enabled"))
    )

    tts = build_tts(raw_config)
    if use_unitree:
        player = UnitreeG1AudioPlayer(
            network_interface=unitree_audio.get("network_interface")
            or raw_config.get("unitree_g1", {}).get("network_interface")
            or None,
            app_name=unitree_audio.get("app_name", "rai_s2s"),
            chunk_size=unitree_audio.get("chunk_size", 96000),
            stop_after_play=unitree_audio.get("stop_after_play", False),
        )
        tts.sample_rate = player.sample_rate
        tts.channels = player.channels
        print("[Test] Generating TTS for Unitree G1...", flush=True)
        audio = tts.get_speech(reply)
        player.play(audio)
        return

    print("[Test] Generating TTS for local speaker...", flush=True)
    audio = tts.get_speech(reply)
    play_sounddevice(audio, s2s.get("speaker_device"))


def main() -> int:
    args = parse_args()
    load_env_file()
    raw_config = load_raw_config(args.config)

    sensor_content, images = call_sensor_tool(args, raw_config)
    reply = ask_llm(args.config, args.question, sensor_content, images)
    print(f"[Test] LLM reply:\n{reply}", flush=True)
    play_reply(reply, args, raw_config)
    print("[Test] Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
