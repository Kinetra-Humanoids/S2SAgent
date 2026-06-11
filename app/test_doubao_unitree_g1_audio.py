"""Generate speech with Doubao TTS and play it on a Unitree G1 speaker.

Run from dist/s2s_agent_bundle:
  uv run python app/test_doubao_unitree_g1_audio.py "你好，我是 RAI。"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
for package_root in (BUNDLE_ROOT / "src" / "rai_s2s", BUNDLE_ROOT / "src" / "rai_core"):
    sys.path.insert(0, str(package_root))

from rai_s2s.sound_device.unitree_g1_audio import (  # noqa: E402
    UnitreeG1AudioError,
    UnitreeG1AudioPlayer,
)
from rai_s2s.tts.models import TTSModelError  # noqa: E402
from rai_s2s.tts.models.doubao_tts import DoubaoTTS  # noqa: E402


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    return key.strip(), value.strip().strip('"').strip("'")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return tomllib.load(file)


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Doubao TTS playback through Unitree G1 AudioClient.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="你好，我是 RAI。豆包语音合成和宇树 G1 播放链路正在测试。",
        help="Text to synthesize and play.",
    )
    parser.add_argument(
        "--config",
        default=str(BUNDLE_ROOT / "config.toml"),
        help="Path to config.toml.",
    )
    parser.add_argument(
        "--env-file",
        default=str(BUNDLE_ROOT / ".env"),
        help="Path to .env containing Doubao credentials.",
    )
    parser.add_argument("--doubao-app-id", default=None, help="Doubao TTS app id.")
    parser.add_argument("--doubao-token", default=None, help="Doubao TTS token.")
    parser.add_argument("--doubao-tts-url", default=None, help="Doubao TTS endpoint URL.")
    parser.add_argument("--doubao-tts-cluster", default=None, help="Doubao TTS cluster.")
    parser.add_argument("--doubao-tts-voice-type", default=None, help="Doubao TTS voice type.")
    parser.add_argument("--doubao-tts-encoding", default=None, help="Doubao TTS encoding.")
    parser.add_argument(
        "--doubao-tts-sample-rate",
        type=int,
        default=None,
        help="Doubao TTS sample rate.",
    )
    parser.add_argument(
        "--doubao-tts-speed-ratio",
        type=float,
        default=None,
        help="Doubao TTS speed ratio.",
    )
    parser.add_argument(
        "--unitree-g1-audio-network-interface",
        default=None,
        help="Network interface connected to Unitree G1, for example en0 or eth0.",
    )
    parser.add_argument(
        "--unitree-g1-audio-app-name",
        default=None,
        help="Unitree G1 AudioClient app name.",
    )
    parser.add_argument(
        "--unitree-g1-audio-chunk-size",
        type=int,
        default=None,
        help="PCM chunk size sent to AudioClient.PlayStream.",
    )
    parser.add_argument(
        "--unitree-g1-audio-stop-after-play",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Call AudioClient.PlayStop after playback.",
    )
    parser.add_argument(
        "--output",
        default=str(BUNDLE_ROOT / "doubao_unitree_g1_test.wav"),
        help="Save synthesized audio to this wav file before playback.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Only generate the wav file; do not call Unitree AudioClient.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    config = load_config(Path(args.config))
    doubao = config.get("doubao_speech", {})
    unitree_audio = config.get("unitree_g1_audio", {})
    unitree_g1 = config.get("unitree_g1", {})

    tts = DoubaoTTS(
        app_id=first_non_empty(
            args.doubao_app_id,
            doubao.get("app_id"),
            os.getenv("DOUBAO_TTS_APP_ID"),
            os.getenv("DOUBAO_APP_ID"),
        ),
        token=first_non_empty(
            args.doubao_token,
            doubao.get("token"),
            os.getenv("DOUBAO_TTS_TOKEN"),
            os.getenv("DOUBAO_TOKEN"),
        ),
        cluster=first_non_empty(
            args.doubao_tts_cluster,
            doubao.get("tts_cluster"),
            os.getenv("DOUBAO_TTS_CLUSTER"),
            "volcano_tts",
        ),
        voice_type=first_non_empty(
            args.doubao_tts_voice_type,
            doubao.get("tts_voice_type"),
            os.getenv("DOUBAO_TTS_VOICE_TYPE"),
        ),
        url=first_non_empty(
            args.doubao_tts_url,
            doubao.get("tts_url"),
            os.getenv("DOUBAO_TTS_URL"),
            "https://openspeech.bytedance.com/api/v1/tts",
        ),
        encoding=first_non_empty(
            args.doubao_tts_encoding,
            doubao.get("tts_encoding"),
            os.getenv("DOUBAO_TTS_ENCODING"),
            "wav",
        ),
        audio_rate=int(
            first_non_empty(
                args.doubao_tts_sample_rate,
                doubao.get("tts_sample_rate"),
                os.getenv("DOUBAO_TTS_SAMPLE_RATE"),
                24000,
            )
        ),
        speed_ratio=float(
            first_non_empty(
                args.doubao_tts_speed_ratio,
                doubao.get("tts_speed_ratio"),
                os.getenv("DOUBAO_TTS_SPEED_RATIO"),
                1.0,
            )
        ),
    )

    print("Generating Doubao TTS audio...")
    audio = tts.get_speech(args.text)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_path, format="wav")
    print(
        "Saved wav:"
        f" {output_path} ({audio.frame_rate} Hz, {audio.channels} ch,"
        f" {len(audio)} ms)"
    )

    if args.no_play:
        print("Skipping Unitree G1 playback because --no-play was set.")
        return 0

    player = UnitreeG1AudioPlayer(
        network_interface=first_non_empty(
            args.unitree_g1_audio_network_interface,
            unitree_audio.get("network_interface"),
            unitree_g1.get("network_interface"),
            os.getenv("UNITREE_G1_AUDIO_NETWORK_INTERFACE"),
            os.getenv("UNITREE_G1_NETWORK_INTERFACE"),
        ),
        app_name=first_non_empty(
            args.unitree_g1_audio_app_name,
            unitree_audio.get("app_name"),
            os.getenv("UNITREE_G1_AUDIO_APP_NAME"),
            "rai_s2s_test",
        ),
        chunk_size=int(
            first_non_empty(
                args.unitree_g1_audio_chunk_size,
                unitree_audio.get("chunk_size"),
                os.getenv("UNITREE_G1_AUDIO_CHUNK_SIZE"),
                96000,
            )
        ),
        stop_after_play=bool(
            first_non_empty(
                args.unitree_g1_audio_stop_after_play,
                unitree_audio.get("stop_after_play"),
                False,
            )
        ),
    )
    print("Playing audio through Unitree G1 AudioClient...")
    player.play(audio)
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TTSModelError, UnitreeG1AudioError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
