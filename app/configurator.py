"""Streamlit configurator for the standalone S2S agent bundle."""

from __future__ import annotations

import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any

import streamlit as st
import tomli
import tomli_w

CONFIG_PATH = Path("config.toml")
ENV_PATH = Path(".env")
LANGUAGE_OPTIONS = [
    "zh",
    "en",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "ru",
]
WHISPER_MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "distil-large-v3",
]
KOKORO_VOICE_OPTIONS = [
    "af_sarah",
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]
DOUBAO_TTS_ENCODING_OPTIONS = ["wav", "mp3", "ogg_opus", "pcm"]
DOUBAO_TTS_SAMPLE_RATE_OPTIONS = [16000, 24000, 32000, 48000]
DOUBAO_ASR_RESOURCE_OPTIONS = [
    "volc.bigasr.auc_turbo",
    "volc.bigasr.auc",
]
DEFAULT_UNITREE_G1_REAL_REPLAY_FILE = (
    "/home/zj/Agent/data/2026-05-23-13-59-19-filtered/data/chunk-000/"
    "episode_000000.parquet"
)
DEFAULT_UNITREE_G1_VLA_MODEL_PATH = (
    "/home/zj/Isaac-GR00T/test_ckpt/"
    "gr00t-freeze-vlm-half-expert-param-cola-644-binary-hand-bs96-0705/"
    "checkpoint-60000"
)
DEFAULT_UNITREE_G1_VLA_PROMPT = (
    "Walk forward, grab the cola and throw into the trash bin"
)
DEFAULT_UNITREE_G1_VLA_SERVER_ROOT = "/home/zj/Isaac-GR00T"
DEFAULT_UNITREE_G1_SIM_SKILLS = [
    {
        "name": "wave left hand",
        "source": "replay",
        "file": "wave_left_hand.npy",
        "aliases": ["wave", "left hand wave", "wave_left_hand"],
        "description": "Wave the left hand using a replay trajectory.",
    },
    {
        "name": "run",
        "source": "replay",
        "file": "run.npy",
        "aliases": ["running"],
        "description": "Run using a replay trajectory.",
    },
    {
        "name": "squat stand",
        "source": "replay",
        "file": "squat_stand.npy",
        "aliases": ["squat_to_stand", "蹲起"],
        "description": "Squat down and stand up using a replay trajectory.",
    },
]
DEFAULT_UNITREE_G1_REAL_SKILLS = [
    {
        "name": "wave left hand",
        "source": "replay",
        "file": DEFAULT_UNITREE_G1_REAL_REPLAY_FILE,
        "aliases": ["wave", "left hand wave", "wave_left_hand"],
        "description": "Wave the left hand using a replay trajectory.",
    },
    {
        "name": "wave left hand with vla",
        "source": "vla",
        "prompt": DEFAULT_UNITREE_G1_VLA_PROMPT,
        "server_root": DEFAULT_UNITREE_G1_VLA_SERVER_ROOT,
        "model_path": DEFAULT_UNITREE_G1_VLA_MODEL_PATH,
        "aliases": [],
        "description": "Run the configured VLA policy through GR00T server and WBC inference.",
    },
]

DEFAULT_CONFIG: dict[str, Any] = {
    "vendor": {
        "simple_model": "openai",
        "complex_model": "openai",
        "embeddings_model": "openai",
    },
    "openai": {
        "simple_model": "gpt-4o",
        "complex_model": "gpt-4o",
        "embeddings_model": "text-embedding-ada-002",
        "base_url": "https://api.openai.com/v1/",
    },
    "aws": {
        "simple_model": "anthropic.claude-3-haiku-20240307-v1:0",
        "complex_model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "embeddings_model": "amazon.titan-embed-text-v1",
        "region_name": "us-east-1",
    },
    "ollama": {
        "simple_model": "llama3.2",
        "complex_model": "llama3.1:70b",
        "embeddings_model": "llama3.2",
        "base_url": "http://localhost:11434",
    },
    "google": {
        "simple_model": "gemini-3-flash",
        "complex_model": "gemini-3-pro",
        "embeddings_model": "text-embedding-004",
    },
    "tracing": {
        "project": "rai",
        "langfuse": {"use_langfuse": False, "host": "http://localhost:3000"},
        "langsmith": {"use_langsmith": False, "host": "https://api.smith.langchain.com"},
    },
    "s2s": {
        "agent_mode": "standard",
        "mic_device": "default",
        "speaker_device": "default",
        "speaker_backend": "sounddevice",
        "asr": "fasterwhisper",
        "tts": "kokoro",
        "openai_whisper_model": "whisper-1",
        "whisper_model": "tiny",
        "openai_base_url": "",
        "language": "zh",
        "sample_rate": 16000,
        "vad_threshold": 0.5,
        "block_size": 1280,
        "grace_period": 0.8,
        "stream_response": False,
    },
    "tools": {
        "python_tools": True,
    },
    "unitree_g1": {
        "tools_enabled": False,
        "enabled_tools": [
            "stop",
            "move",
            "posture",
            "gesture",
            "arm_action",
        ],
        "network_interface": "",
        "control_enabled": False,
    },
    "unitree_g1_audio": {
        "enabled": False,
        "network_interface": "",
        "app_name": "rai_s2s",
        "chunk_size": 96000,
        "stop_after_play": False,
    },
    "unitree_g1_sim": {
        "tools_enabled": False,
        "deploy_dir": "",
        "gr00t_root": "",
        "replay_dir": "replays/unitree_g1_sim",
        "auto_start": True,
        "confirm_deployment": True,
        "start_control": True,
        "startup_settle_seconds": 2.0,
        "terminal_viewer": True,
        "log_dir": "logs/unitree_g1_sim",
        "enabled_tools": [
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
        "skills": deepcopy(DEFAULT_UNITREE_G1_SIM_SKILLS),
    },
    "unitree_g1_real": {
        "tools_enabled": False,
        "deploy_dir": "",
        "gr00t_root": "",
        "vla_server_root": DEFAULT_UNITREE_G1_VLA_SERVER_ROOT,
        "replay_dir": "replays/unitree_g1_sim",
        "auto_start": True,
        "confirm_deployment": True,
        "start_control": True,
        "startup_settle_seconds": 2.0,
        "terminal_viewer": True,
        "log_dir": "logs/unitree_g1_real",
        "enabled_tools": [
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
        "skills": deepcopy(DEFAULT_UNITREE_G1_REAL_SKILLS),
    },
    "sensor_tool": {
        "enabled": False,
        "transport": "zmq",
        "server_ip": "localhost",
        "port": 5555,
        "http_base_url": "",
        "default_camera": "",
        "blocking": True,
    },
    "ros2": {
        "ros2_tools": False,
        "nav2_tools": False,
        "ros2_readable": "",
        "ros2_writable": "",
        "ros2_forbidden": "",
        "ros2_node_name": "rai_s2s_tools",
        "use_sim_time": False,
        "nav2_frame_id": "map",
        "nav2_action_name": "navigate_to_pose",
    },
    "asr": {
        "recording_device_name": "default",
        "transcription_model": "LocalWhisper",
        "language": "zh",
        "vad_model": "SileroVAD",
        "silence_grace_period": 0.8,
        "vad_threshold": 0.5,
        "use_wake_word": False,
        "wake_word_model": "",
        "wake_word_threshold": 0.5,
        "wake_word_model_name": "",
        "transcription_model_name": "tiny",
    },
    "tts": {
        "vendor": "KokoroTTS",
        "voice": "af_sarah",
        "speaker_device_name": "default",
    },
    "doubao_speech": {
        "app_id": "",
        "asr_url": "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
        "asr_model": "bigmodel",
        "asr_auth_mode": "auto",
        "asr_resource_id": "volc.bigasr.auc_turbo",
        "asr_app_key": "",
        "asr_access_key": "",
        "asr_api_key": "",
        "tts_url": "https://openspeech.bytedance.com/api/v1/tts",
        "tts_cluster": "volcano_tts",
        "tts_voice_type": "",
        "tts_encoding": "wav",
        "tts_sample_rate": 24000,
        "tts_speed_ratio": 1.0,
    },
}


def merge_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return deepcopy(DEFAULT_CONFIG)
    with CONFIG_PATH.open("rb") as file:
        return merge_defaults(tomli.load(file), DEFAULT_CONFIG)


def save_config(config: dict[str, Any]) -> None:
    with CONFIG_PATH.open("wb") as file:
        tomli_w.dump(config, file)


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    value = value.strip().strip('"').strip("'")
    return key.strip(), value


def load_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is not None:
            key, value = parsed
            values[key] = value
    return values


def save_env(values: dict[str, str]) -> None:
    existing = load_env()
    existing.update({key: value for key, value in values.items() if value is not None})
    lines = [
        "# Generated by app/configurator.py",
        f'OPENAI_API_KEY="{existing.get("OPENAI_API_KEY", "")}"',
        f'OPENAI_BASE_URL="{existing.get("OPENAI_BASE_URL", "")}"',
        f'LANGCHAIN_API_KEY="{existing.get("LANGCHAIN_API_KEY", "")}"',
        f'LANGFUSE_PUBLIC_KEY="{existing.get("LANGFUSE_PUBLIC_KEY", "")}"',
        f'LANGFUSE_SECRET_KEY="{existing.get("LANGFUSE_SECRET_KEY", "")}"',
        f'DOUBAO_APP_ID="{existing.get("DOUBAO_APP_ID", "")}"',
        f'DOUBAO_TOKEN="{existing.get("DOUBAO_TOKEN", "")}"',
        f'DOUBAO_ASR_AUTH_MODE="{existing.get("DOUBAO_ASR_AUTH_MODE", "")}"',
        f'DOUBAO_ASR_APP_KEY="{existing.get("DOUBAO_ASR_APP_KEY", "")}"',
        f'DOUBAO_ASR_ACCESS_KEY="{existing.get("DOUBAO_ASR_ACCESS_KEY", "")}"',
        f'DOUBAO_ASR_API_KEY="{existing.get("DOUBAO_ASR_API_KEY", "")}"',
        f'DOUBAO_TTS_VOICE_TYPE="{existing.get("DOUBAO_TTS_VOICE_TYPE", "")}"',
        f'UNITREE_G1_NETWORK_INTERFACE="{existing.get("UNITREE_G1_NETWORK_INTERFACE", "")}"',
        f'UNITREE_G1_ENABLE_CONTROL="{existing.get("UNITREE_G1_ENABLE_CONTROL", "false")}"',
        f'UNITREE_G1_AUDIO_APP_NAME="{existing.get("UNITREE_G1_AUDIO_APP_NAME", "rai_s2s")}"',
        f'UNITREE_G1_SIM_DEPLOY_DIR="{existing.get("UNITREE_G1_SIM_DEPLOY_DIR", "")}"',
        f'GR00T_WBC_ROOT="{existing.get("GR00T_WBC_ROOT", "")}"',
        "UNITREE_G1_SIM_REPLAY_DIR="
        f'"{existing.get("UNITREE_G1_SIM_REPLAY_DIR", "replays/unitree_g1_sim")}"',
        "UNITREE_G1_SIM_TERMINAL_VIEWER="
        f'"{existing.get("UNITREE_G1_SIM_TERMINAL_VIEWER", "true")}"',
        f'UNITREE_G1_SIM_LOG_DIR="{existing.get("UNITREE_G1_SIM_LOG_DIR", "logs/unitree_g1_sim")}"',
        f'UNITREE_G1_REAL_DEPLOY_DIR="{existing.get("UNITREE_G1_REAL_DEPLOY_DIR", "")}"',
        "UNITREE_G1_REAL_REPLAY_DIR="
        f'"{existing.get("UNITREE_G1_REAL_REPLAY_DIR", "replays/unitree_g1_sim")}"',
        "UNITREE_G1_REAL_TERMINAL_VIEWER="
        f'"{existing.get("UNITREE_G1_REAL_TERMINAL_VIEWER", "true")}"',
        f'UNITREE_G1_REAL_LOG_DIR="{existing.get("UNITREE_G1_REAL_LOG_DIR", "logs/unitree_g1_real")}"',
        "",
    ]
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")


def options_with_current(options: list[str], current: str) -> list[str]:
    if current and current not in options:
        return [current, *options]
    return options


def select_option(label: str, options: list[str], current: str) -> str:
    all_options = options_with_current(options, current)
    index = all_options.index(current) if current in all_options else 0
    return st.selectbox(label, all_options, index=index)


def resolve_replay_file(replay_dir: str, replay_file: str) -> Path:
    path = Path(replay_file)
    replay_root = Path(replay_dir or "replays/unitree_g1_sim").expanduser().resolve()
    if path.is_absolute():
        default_path = replay_root / path.name
        return default_path if default_path.exists() else path.expanduser()
    return replay_root / path


def replay_command_for_file(replay_file: Path) -> list[str]:
    if replay_file.suffix == ".parquet":
        return [
            "python gear_sonic/scripts/sonic_encoder_input_player_with_hand.py \\",
            f"  --parquet-file {shlex.quote(str(replay_file))}",
        ]
    return [
        "python gear_sonic/scripts/sonic_encoder_input_player.py \\",
        f"  --latent-input-file {shlex.quote(str(replay_file))}",
    ]


def get_replay_skill_file(
    skills: list[dict[str, Any]],
    skill_name: str,
    default: str,
) -> str:
    normalized_name = skill_name.strip().lower()
    for skill in skills:
        if (
            str(skill.get("name", "")).strip().lower() == normalized_name
            and str(skill.get("source", "replay")).strip().lower() == "replay"
        ):
            return str(skill.get("file", "") or default)
    return default


def set_replay_skill_file(
    skills: list[dict[str, Any]],
    skill_name: str,
    replay_file: str,
) -> list[dict[str, Any]]:
    normalized_name = skill_name.strip().lower()
    updated = deepcopy(skills)
    for skill in updated:
        if str(skill.get("name", "")).strip().lower() == normalized_name:
            skill["source"] = "replay"
            skill["file"] = replay_file
            return updated
    updated.append(
        {
            "name": skill_name,
            "source": "replay",
            "file": replay_file,
            "aliases": ["wave", "left hand wave", "wave_left_hand"],
            "description": "Wave the left hand using a replay trajectory.",
        }
    )
    return updated


def first_vla_skill(skills: list[dict[str, Any]]) -> dict[str, Any] | None:
    for skill in skills:
        if str(skill.get("source", "")).strip().lower() == "vla":
            return skill
    return None


def apply_vla_server_root(
    skills: list[dict[str, Any]],
    server_root: str,
) -> list[dict[str, Any]]:
    updated = deepcopy(skills)
    for skill in updated:
        if str(skill.get("source", "")).strip().lower() == "vla":
            if server_root:
                skill["server_root"] = server_root
            skill.pop("wbc_root", None)
    return updated


def get_sound_devices(output: bool = False) -> list[str]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception:
        return ["default"]

    channel_key = "max_output_channels" if output else "max_input_channels"
    names = [
        str(device["name"])
        for device in devices
        if int(device.get(channel_key, 0)) > 0
    ]
    return ["default", *names] if "default" not in names else names


def device_select(label: str, value: str, output: bool) -> str:
    devices = get_sound_devices(output=output)
    index = devices.index(value) if value in devices else 0
    return st.selectbox(label, devices, index=index)


def build_unitree_g1_sim_terminal_commands(unitree_g1_sim: dict[str, Any]) -> str:
    deploy_dir = unitree_g1_sim.get("deploy_dir", "")
    gr00t_root = unitree_g1_sim.get("gr00t_root", "")
    replay_dir = unitree_g1_sim.get("replay_dir", "replays/unitree_g1_sim")
    log_dir = unitree_g1_sim.get("log_dir", "logs/unitree_g1_sim")
    lines = [
        "# 1. Start the MuJoCo sim loop separately from the GR00T repo root:",
    ]
    if gr00t_root:
        lines.append(f"cd {shlex.quote(gr00t_root)}")
    lines.extend(
        [
            "source .venv_sim/bin/activate",
            "python gear_sonic/scripts/run_sim_loop.py",
            "",
            "# 2. The S2S tool backend starts the GR00T manager with:",
        ]
    )
    if deploy_dir:
        lines.append(f"cd {shlex.quote(deploy_dir)}")
    lines.extend(
        [
            "source scripts/setup_env.sh",
            "bash deploy.sh --input-type manager sim",
            "",
            "# 3. For replay actions, the backend runs:",
        ]
    )
    if gr00t_root:
        lines.append(f"cd {shlex.quote(gr00t_root)}")
    lines.extend(
        [
            "source .venv_teleop/bin/activate",
            "python gear_sonic/scripts/sonic_encoder_input_player.py \\",
            f"  --latent-input-file {shlex.quote(str(Path(replay_dir) / 'wave_left_hand.npy'))}",
            "",
            "# 4. To watch backend terminal output manually:",
            f"tail -f {shlex.quote(str(Path(log_dir) / 'manager.log'))}",
            f"tail -f {shlex.quote(str(Path(log_dir) / 'replay_player.log'))}",
        ]
    )
    return "\n".join(lines)


def build_unitree_g1_real_terminal_commands(unitree_g1_real: dict[str, Any]) -> str:
    deploy_dir = unitree_g1_real.get("deploy_dir", "")
    gr00t_root = unitree_g1_real.get("gr00t_root", "")
    replay_dir = unitree_g1_real.get("replay_dir", "replays/unitree_g1_sim")
    log_dir = unitree_g1_real.get("log_dir", "logs/unitree_g1_real")
    skills = unitree_g1_real.get("skills", DEFAULT_UNITREE_G1_REAL_SKILLS)
    vla_skill = first_vla_skill(skills)
    replay_file = resolve_replay_file(
        replay_dir,
        get_replay_skill_file(
            skills,
            "wave left hand",
            DEFAULT_UNITREE_G1_REAL_REPLAY_FILE,
        ),
    )
    lines = [
        "# 1. The S2S tool backend starts the real Unitree G1 manager with:",
    ]
    if deploy_dir:
        lines.append(f"cd {shlex.quote(deploy_dir)}")
    lines.extend(
        [
            "source scripts/setup_env.sh",
            "./deploy.sh --input-type manager --zmq-host localhost --hand-type inspire real",
            "",
            "# 2. For replay actions, the backend runs:",
        ]
    )
    if gr00t_root:
        lines.append(f"cd {shlex.quote(gr00t_root)}")
    lines.extend(
        [
            "source .venv_teleop/bin/activate",
            *replay_command_for_file(replay_file),
            "",
            "# 3. To watch backend terminal output manually:",
            f"tail -f {shlex.quote(str(Path(log_dir) / 'manager.log'))}",
            f"tail -f {shlex.quote(str(Path(log_dir) / 'replay_player.log'))}",
        ]
    )
    if vla_skill is not None:
        vla_server_root = str(
            unitree_g1_real.get("vla_server_root", DEFAULT_UNITREE_G1_VLA_SERVER_ROOT)
            or DEFAULT_UNITREE_G1_VLA_SERVER_ROOT
        )
        vla_wbc_root = str(gr00t_root or "/home/zj/GR00T-WholeBodyControl")
        lines.extend(
            [
                "",
                "# 4. For VLA skills, the backend starts the GR00T server with:",
                f"cd {shlex.quote(vla_server_root)}",
                "source .venv/bin/activate",
                "export HF_HUB_OFFLINE=1",
                "export TRANSFORMERS_OFFLINE=1",
                "export NO_ALBUMENTATIONS_UPDATE=1",
                "python gr00t/eval/run_gr00t_server.py \\",
                f"  --model-path {shlex.quote(str(vla_skill.get('model_path', DEFAULT_UNITREE_G1_VLA_MODEL_PATH)))} \\",
                "  --embodiment-tag NEW_EMBODIMENT \\",
                "  --device cuda:0 \\",
                "  --port 5550",
                "",
                "# 5. Then it starts WBC VLA inference with:",
            ]
        )
        lines.append(f"cd {shlex.quote(vla_wbc_root)}")
        lines.extend(
            [
                "source .venv_inference/bin/activate",
                "python gear_sonic/scripts/run_vla_inference.py \\",
                "  --host localhost \\",
                "  --port 5550 \\",
                "  --embodiment-tag NEW_EMBODIMENT \\",
                f"  --prompt {shlex.quote(str(vla_skill.get('prompt', DEFAULT_UNITREE_G1_VLA_PROMPT)))} \\",
                "  --action-publish-rate 50 \\",
                "  --action-horizon 50 \\",
                "  --camera-host 192.168.123.164 \\",
                "  --camera-port 5555",
                "",
                "# 6. To watch VLA output manually:",
                f"tail -f {shlex.quote(str(Path(log_dir) / 'vla_server.log'))}",
                f"tail -f {shlex.quote(str(Path(log_dir) / 'vla_inference.log'))}",
            ]
        )
    return "\n".join(lines)


def disable_sim_tools_when_sdk_enabled() -> None:
    if st.session_state.get("unitree_g1_tools_enabled", False):
        st.session_state.unitree_g1_sim_tools_enabled = False
        st.session_state.unitree_g1_real_tools_enabled = False


def disable_sdk_tools_when_sim_enabled() -> None:
    if st.session_state.get("unitree_g1_sim_tools_enabled", False):
        st.session_state.unitree_g1_tools_enabled = False
        st.session_state.unitree_g1_real_tools_enabled = False


def disable_sdk_sim_tools_when_real_enabled() -> None:
    if st.session_state.get("unitree_g1_real_tools_enabled", False):
        st.session_state.unitree_g1_tools_enabled = False
        st.session_state.unitree_g1_sim_tools_enabled = False


def normalize_skill_rows(rows: Any, *, allow_vla: bool) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    skills: list[dict[str, Any]] = []
    allowed_sources = {"replay", "vla"} if allow_vla else {"replay"}
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        source = str(row.get("source", "replay") or "replay").strip().lower()
        if source not in allowed_sources:
            source = "replay"
        aliases_value = row.get("aliases", "")
        if isinstance(aliases_value, list):
            aliases = [str(alias).strip() for alias in aliases_value]
        else:
            aliases = [alias.strip() for alias in str(aliases_value).split(",")]
        skill: dict[str, Any] = {
            "name": name,
            "source": source,
            "aliases": [alias for alias in aliases if alias],
        }
        replay_file = str(row.get("file", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        model_path = str(row.get("model_path", "")).strip()
        description = str(row.get("description", "")).strip()
        if replay_file:
            skill["file"] = replay_file
        if prompt:
            skill["prompt"] = prompt
        if model_path:
            skill["model_path"] = model_path
        if description:
            skill["description"] = description
        skills.append(skill)
    return skills


def unitree_skill_editor(
    label: str,
    skills: list[dict[str, Any]],
    *,
    key: str,
    allow_vla: bool,
) -> list[dict[str, Any]]:
    rows = []
    for skill in skills:
        aliases = skill.get("aliases", [])
        rows.append(
            {
                "name": skill.get("name", ""),
                "source": skill.get("source", "replay"),
                "file": skill.get("file", ""),
                "prompt": skill.get("prompt", ""),
                "model_path": skill.get("model_path", ""),
                "aliases": ", ".join(aliases) if isinstance(aliases, list) else aliases,
                "description": skill.get("description", ""),
            }
        )
    edited_rows = st.data_editor(
        rows,
        key=key,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "source": st.column_config.SelectboxColumn(
                "source",
                options=["replay", "vla"] if allow_vla else ["replay"],
                required=True,
            ),
            "name": st.column_config.TextColumn("name", required=True),
            "file": st.column_config.TextColumn("file"),
            "prompt": st.column_config.TextColumn("prompt"),
            "model_path": st.column_config.TextColumn("model_path"),
            "aliases": st.column_config.TextColumn("aliases"),
            "description": st.column_config.TextColumn("description"),
        },
    )
    st.caption(label)
    return normalize_skill_rows(edited_rows, allow_vla=allow_vla)


def model_tab(config: dict[str, Any]) -> None:
    st.subheader("Model Provider")
    env = st.session_state.env
    vendors = ["openai", "aws", "ollama", "google"]
    cols = st.columns(3)
    with cols[0]:
        config["vendor"]["simple_model"] = st.selectbox(
            "Simple model vendor",
            vendors,
            index=vendors.index(config["vendor"]["simple_model"]),
        )
    with cols[1]:
        config["vendor"]["complex_model"] = st.selectbox(
            "Complex model vendor",
            vendors,
            index=vendors.index(config["vendor"]["complex_model"]),
        )
    with cols[2]:
        config["vendor"]["embeddings_model"] = st.selectbox(
            "Embeddings vendor",
            vendors,
            index=vendors.index(config["vendor"]["embeddings_model"]),
        )

    with st.expander("OpenAI / OpenAI-compatible", expanded=True):
        env["OPENAI_API_KEY"] = st.text_input(
            "OPENAI_API_KEY",
            value=env.get("OPENAI_API_KEY", ""),
            type="password",
            help="Saved to .env, not config.toml.",
        )
        config["openai"]["simple_model"] = st.text_input(
            "OpenAI simple model", config["openai"]["simple_model"]
        )
        config["openai"]["complex_model"] = st.text_input(
            "OpenAI complex model", config["openai"]["complex_model"]
        )
        config["openai"]["embeddings_model"] = st.text_input(
            "OpenAI embeddings model", config["openai"]["embeddings_model"]
        )
        config["openai"]["base_url"] = st.text_input(
            "OpenAI-compatible base URL", config["openai"]["base_url"]
        )
        env["OPENAI_BASE_URL"] = st.text_input(
            "OPENAI_BASE_URL",
            value=env.get("OPENAI_BASE_URL", config["openai"]["base_url"]),
            help="Saved to .env for SDK clients that read environment variables.",
        )

    with st.expander("Ollama"):
        config["ollama"]["base_url"] = st.text_input(
            "Ollama base URL", config["ollama"]["base_url"]
        )
        config["ollama"]["simple_model"] = st.text_input(
            "Ollama simple model", config["ollama"]["simple_model"]
        )
        config["ollama"]["complex_model"] = st.text_input(
            "Ollama complex model", config["ollama"]["complex_model"]
        )
        config["ollama"]["embeddings_model"] = st.text_input(
            "Ollama embeddings model", config["ollama"]["embeddings_model"]
        )

    with st.expander("AWS / Google"):
        config["aws"]["region_name"] = st.text_input(
            "AWS region", config["aws"]["region_name"]
        )
        config["aws"]["simple_model"] = st.text_input(
            "AWS simple model", config["aws"]["simple_model"]
        )
        config["aws"]["complex_model"] = st.text_input(
            "AWS complex model", config["aws"]["complex_model"]
        )
        config["aws"]["embeddings_model"] = st.text_input(
            "AWS embeddings model", config["aws"]["embeddings_model"]
        )
        config["google"]["simple_model"] = st.text_input(
            "Google simple model", config["google"]["simple_model"]
        )
        config["google"]["complex_model"] = st.text_input(
            "Google complex model", config["google"]["complex_model"]
        )
        config["google"]["embeddings_model"] = st.text_input(
            "Google embeddings model", config["google"]["embeddings_model"]
        )


def s2s_tab(config: dict[str, Any]) -> None:
    st.subheader("Speech-to-Speech Runtime")
    s2s = config["s2s"]
    asr = config["asr"]
    tts = config["tts"]
    doubao = config["doubao_speech"]
    unitree_g1_audio = config["unitree_g1_audio"]
    env = st.session_state.env

    agent_mode_options = ["standard", "policy_delegate"]
    current_agent_mode = s2s.get("agent_mode", "standard")
    s2s["agent_mode"] = st.selectbox(
        "Agent mode",
        agent_mode_options,
        index=(
            agent_mode_options.index(current_agent_mode)
            if current_agent_mode in agent_mode_options
            else 0
        ),
        help=(
            "policy_delegate makes the S2S agent identify visual targets and speak "
            "as the robot in the first person while a separate low-level agent "
            "executes physical policies."
        ),
    )

    cols = st.columns(2)
    with cols[0]:
        s2s["mic_device"] = device_select("Microphone", s2s["mic_device"], output=False)
        asr["recording_device_name"] = s2s["mic_device"]
        asr_backend_options = options_with_current(
            ["fasterwhisper", "local", "openai", "doubao"], s2s["asr"]
        )
        s2s["asr"] = st.selectbox(
            "ASR backend",
            asr_backend_options,
            index=asr_backend_options.index(s2s["asr"]),
        )
        s2s["language"] = select_option(
            "Language code",
            LANGUAGE_OPTIONS,
            s2s["language"],
        )
        asr["language"] = s2s["language"]

    with cols[1]:
        speaker_backend_options = ["sounddevice", "unitree_g1"]
        s2s["speaker_backend"] = st.selectbox(
            "Speaker backend",
            speaker_backend_options,
            index=speaker_backend_options.index(
                s2s.get("speaker_backend", "sounddevice")
            )
            if s2s.get("speaker_backend", "sounddevice") in speaker_backend_options
            else 0,
        )
        unitree_g1_audio["enabled"] = s2s["speaker_backend"] == "unitree_g1"
        if s2s["speaker_backend"] == "sounddevice":
            s2s["speaker_device"] = device_select("Speaker", s2s["speaker_device"], output=True)
        else:
            s2s["speaker_device"] = "default"
            st.caption("TTS audio will be sent to the robot speaker through Unitree AudioClient.")
            unitree_g1_audio["network_interface"] = st.text_input(
                "Unitree speaker network interface",
                value=unitree_g1_audio.get("network_interface", "")
                or env.get("UNITREE_G1_NETWORK_INTERFACE", ""),
                help="Network interface connected to G1, for example en0, en7, eth0.",
            )
            unitree_g1_audio["app_name"] = st.text_input(
                "Unitree speaker app name",
                value=unitree_g1_audio.get("app_name", "rai_s2s")
                or env.get("UNITREE_G1_AUDIO_APP_NAME", "rai_s2s"),
            )
            unitree_g1_audio["chunk_size"] = st.number_input(
                "Unitree speaker chunk size",
                min_value=16000,
                max_value=320000,
                value=int(unitree_g1_audio.get("chunk_size", 96000)),
                step=16000,
            )
            unitree_g1_audio["stop_after_play"] = st.checkbox(
                "Call PlayStop after each utterance",
                value=bool(unitree_g1_audio.get("stop_after_play", False)),
            )
            env["UNITREE_G1_NETWORK_INTERFACE"] = unitree_g1_audio["network_interface"]
            env["UNITREE_G1_AUDIO_APP_NAME"] = unitree_g1_audio["app_name"]
        tts["speaker_device_name"] = s2s["speaker_device"]
        tts_options = options_with_current(["kokoro", "doubao"], s2s["tts"])
        s2s["tts"] = st.selectbox(
            "TTS backend",
            tts_options,
            index=tts_options.index(s2s["tts"]),
        )
        tts["vendor"] = "DoubaoTTS" if s2s["tts"] == "doubao" else "KokoroTTS"
        s2s["stream_response"] = st.checkbox(
            "Stream LLM response to TTS", value=bool(s2s["stream_response"])
        )

    with st.expander("Advanced speech settings"):
        advanced_cols = st.columns(2)
        with advanced_cols[0]:
            s2s["sample_rate"] = st.number_input(
                "Sample rate",
                min_value=8000,
                max_value=48000,
                value=int(s2s["sample_rate"]),
                step=1000,
            )
            s2s["block_size"] = st.number_input(
                "Block size",
                min_value=160,
                max_value=8192,
                value=int(s2s["block_size"]),
                step=160,
            )
            s2s["vad_threshold"] = st.slider(
                "VAD threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(s2s["vad_threshold"]),
            )
            asr["vad_threshold"] = s2s["vad_threshold"]
            s2s["grace_period"] = st.number_input(
                "Silence grace period",
                min_value=0.1,
                max_value=5.0,
                value=float(s2s["grace_period"]),
                step=0.1,
            )
            asr["silence_grace_period"] = s2s["grace_period"]

        with advanced_cols[1]:
            tts["voice"] = select_option(
                "Kokoro voice",
                KOKORO_VOICE_OPTIONS,
                tts.get("voice", "af_sarah"),
            )
            s2s["whisper_model"] = select_option(
                "Local/FasterWhisper model",
                WHISPER_MODEL_OPTIONS,
                s2s["whisper_model"],
            )
            asr["transcription_model_name"] = s2s["whisper_model"]
            s2s["openai_whisper_model"] = st.text_input(
                "OpenAI ASR model", s2s["openai_whisper_model"]
            )
            s2s["openai_base_url"] = st.text_input(
                "OpenAI ASR base URL override", s2s.get("openai_base_url", "")
            )

    with st.expander("Doubao Speech API", expanded=s2s["asr"] == "doubao" or s2s["tts"] == "doubao"):
        doubao["app_id"] = st.text_input(
            "Doubao App ID",
            value=doubao.get("app_id", "") or env.get("DOUBAO_APP_ID", ""),
        )
        env["DOUBAO_APP_ID"] = doubao["app_id"]
        env["DOUBAO_TOKEN"] = st.text_input(
            "Doubao Token / Access Token",
            value=env.get("DOUBAO_TOKEN", ""),
            type="password",
            help="Saved to .env.",
        )
        st.caption("ASR settings")
        doubao["asr_url"] = st.text_input("ASR URL", doubao.get("asr_url", ""))
        doubao["asr_model"] = st.text_input("ASR model", doubao.get("asr_model", "bigmodel"))
        doubao["asr_auth_mode"] = st.selectbox(
            "ASR auth mode",
            ["old", "new", "auto"],
            index=["old", "new", "auto"].index(doubao.get("asr_auth_mode", "old"))
            if doubao.get("asr_auth_mode", "old") in ["old", "new", "auto"]
            else 0,
            help="Old sends X-Api-App-Key and X-Api-Access-Key. New sends X-Api-Key.",
        )
        env["DOUBAO_ASR_AUTH_MODE"] = doubao["asr_auth_mode"]
        doubao["asr_resource_id"] = select_option(
            "ASR resource id",
            DOUBAO_ASR_RESOURCE_OPTIONS,
            doubao.get("asr_resource_id", "volc.bigasr.auc_turbo"),
        )
        doubao["asr_app_key"] = st.text_input(
            "ASR app key / API key",
            value=doubao.get("asr_app_key", "") or env.get("DOUBAO_ASR_APP_KEY", ""),
            help="New Doubao Speech console can use this alone as X-Api-Key.",
        )
        env["DOUBAO_ASR_APP_KEY"] = doubao["asr_app_key"]
        env["DOUBAO_ASR_ACCESS_KEY"] = st.text_input(
            "ASR access key",
            value=env.get("DOUBAO_ASR_ACCESS_KEY", ""),
            type="password",
            help="Old Doubao Speech console only. Saved to .env.",
        )
        env["DOUBAO_ASR_API_KEY"] = st.text_input(
            "ASR API key",
            value=env.get("DOUBAO_ASR_API_KEY", ""),
            type="password",
            help="New Doubao Speech console. If set, this is sent as X-Api-Key.",
        )
        st.caption("TTS settings")
        doubao["tts_url"] = st.text_input("TTS URL", doubao.get("tts_url", ""))
        doubao["tts_cluster"] = st.text_input("TTS cluster", doubao.get("tts_cluster", "volcano_tts"))
        doubao["tts_voice_type"] = st.text_input(
            "TTS voice type",
            value=doubao.get("tts_voice_type", "") or env.get("DOUBAO_TTS_VOICE_TYPE", ""),
        )
        env["DOUBAO_TTS_VOICE_TYPE"] = doubao["tts_voice_type"]
        doubao["tts_encoding"] = select_option(
            "TTS encoding",
            DOUBAO_TTS_ENCODING_OPTIONS,
            doubao.get("tts_encoding", "wav"),
        )
        sample_rates = [str(value) for value in DOUBAO_TTS_SAMPLE_RATE_OPTIONS]
        doubao["tts_sample_rate"] = int(
            select_option(
                "TTS sample rate",
                sample_rates,
                str(doubao.get("tts_sample_rate", 24000)),
            )
        )
        doubao["tts_speed_ratio"] = st.slider(
            "TTS speed ratio",
            min_value=0.5,
            max_value=2.0,
            value=float(doubao.get("tts_speed_ratio", 1.0)),
            step=0.1,
        )

    asr_model_map = {
        "fasterwhisper": "FasterWhisper",
        "local": "LocalWhisper",
        "openai": "OpenAI",
        "doubao": "DoubaoASR",
    }
    asr["transcription_model"] = asr_model_map[s2s["asr"]]


def tools_tab(config: dict[str, Any]) -> None:
    st.subheader("Tools")
    tools = config["tools"]
    unitree_g1 = config["unitree_g1"]
    unitree_g1_sim = config["unitree_g1_sim"]
    unitree_g1_real = config["unitree_g1_real"]
    sensor_tool = config["sensor_tool"]
    ros2 = config["ros2"]
    env = st.session_state.env
    tools["python_tools"] = st.checkbox(
        "Enable Python tools", value=bool(tools["python_tools"])
    )
    enabled_unitree_manager_count = sum(
        bool(section.get("tools_enabled"))
        for section in (unitree_g1, unitree_g1_sim, unitree_g1_real)
    )
    if enabled_unitree_manager_count > 1:
        unitree_g1_sim["tools_enabled"] = False
        unitree_g1_real["tools_enabled"] = False
        st.warning("Unitree G1 SDK, Sim, and Real tools are mutually exclusive.")

    st.divider()
    st.subheader("Unitree G1 Tools")
    unitree_g1["tools_enabled"] = st.checkbox(
        "Enable Unitree G1 SDK tools",
        value=bool(unitree_g1["tools_enabled"]),
        key="unitree_g1_tools_enabled",
        on_change=disable_sim_tools_when_sdk_enabled,
        help="Mutually exclusive with Unitree G1 sim manager tools.",
    )
    if unitree_g1["tools_enabled"]:
        unitree_g1_sim["tools_enabled"] = False
        unitree_g1_real["tools_enabled"] = False
    enabled_unitree_tools = set(
        unitree_g1.get(
            "enabled_tools",
            ["stop", "move", "posture", "gesture", "arm_action"],
        )
    )
    st.caption("Choose which Unitree tools are exposed to the agent.")
    tool_cols = st.columns(5)
    unitree_tool_options = [
        ("stop", "Stop"),
        ("move", "Move"),
        ("posture", "Posture"),
        ("gesture", "Gesture"),
        ("arm_action", "Arm actions"),
    ]
    selected_unitree_tools: list[str] = []
    for col, (tool_name, label) in zip(tool_cols, unitree_tool_options):
        with col:
            enabled = st.checkbox(
                label,
                value=tool_name in enabled_unitree_tools,
                key=f"unitree_g1_tool_{tool_name}",
            )
            if enabled:
                selected_unitree_tools.append(tool_name)
    unitree_g1["enabled_tools"] = selected_unitree_tools
    unitree_g1["network_interface"] = st.text_input(
        "Unitree network interface",
        value=unitree_g1.get("network_interface", "")
        or env.get("UNITREE_G1_NETWORK_INTERFACE", ""),
        help="Network interface connected to G1, for example en0, en7, eth0.",
    )
    unitree_g1["control_enabled"] = st.checkbox(
        "Allow movement/posture commands",
        value=bool(unitree_g1["control_enabled"]),
        help="Leave disabled while testing. Enable only when the robot area is safe.",
    )
    env["UNITREE_G1_NETWORK_INTERFACE"] = unitree_g1["network_interface"]
    env["UNITREE_G1_ENABLE_CONTROL"] = str(unitree_g1["control_enabled"]).lower()

    st.divider()
    st.subheader("Unitree G1 Sim Manager Tools")
    unitree_g1_sim["tools_enabled"] = st.checkbox(
        "Enable Unitree G1 sim manager tools",
        value=bool(unitree_g1_sim.get("tools_enabled", False)),
        key="unitree_g1_sim_tools_enabled",
        on_change=disable_sdk_tools_when_sim_enabled,
        help=(
            "Expose tools that run `bash deploy.sh --input-type manager sim` "
            "and send manager hotkeys. Mutually exclusive with SDK tools."
        ),
    )
    if unitree_g1_sim["tools_enabled"]:
        unitree_g1["tools_enabled"] = False
        unitree_g1_real["tools_enabled"] = False
    unitree_g1_sim["deploy_dir"] = st.text_input(
        "GR00T deploy directory",
        value=unitree_g1_sim.get("deploy_dir", "")
        or env.get("UNITREE_G1_SIM_DEPLOY_DIR", ""),
        help="Path to the gear_sonic_deploy directory that contains deploy.sh.",
    )
    unitree_g1_sim["gr00t_root"] = st.text_input(
        "GR00T-WholeBodyControl root",
        value=unitree_g1_sim.get("gr00t_root", "")
        or env.get("GR00T_WBC_ROOT", ""),
        help="Path used to run gear_sonic/scripts/sonic_encoder_input_player.py.",
    )
    unitree_g1_sim["replay_dir"] = st.text_input(
        "Replay .npy directory",
        value=unitree_g1_sim.get("replay_dir", "replays/unitree_g1_sim")
        or env.get("UNITREE_G1_SIM_REPLAY_DIR", "replays/unitree_g1_sim"),
    )
    sim_start_cols = st.columns(2)
    with sim_start_cols[0]:
        unitree_g1_sim["auto_start"] = st.checkbox(
            "Auto-start manager",
            value=bool(unitree_g1_sim.get("auto_start", True)),
        )
    with sim_start_cols[1]:
        unitree_g1_sim["startup_settle_seconds"] = st.number_input(
            "Startup settle seconds",
            min_value=0.0,
            max_value=30.0,
            value=float(unitree_g1_sim.get("startup_settle_seconds", 2.0)),
            step=0.5,
        )
    sim_terminal_cols = st.columns(2)
    with sim_terminal_cols[0]:
        unitree_g1_sim["terminal_viewer"] = st.checkbox(
            "Open terminal viewer",
            value=bool(unitree_g1_sim.get("terminal_viewer", True)),
            help="Open a terminal window when the manager starts.",
        )
    with sim_terminal_cols[1]:
        unitree_g1_sim["log_dir"] = st.text_input(
            "Terminal log directory",
            value=unitree_g1_sim.get("log_dir", "logs/unitree_g1_sim")
            or env.get("UNITREE_G1_SIM_LOG_DIR", "logs/unitree_g1_sim"),
        )
    enabled_sim_tools = set(
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
    )
    st.caption("Choose which GR00T manager sim tools are exposed to the agent.")
    sim_tool_options = [
        ("confirm_deployment", "confirm_deployment"),
        ("perform_skill", "perform_skill"),
        ("perform_replay", "perform_replay"),
        ("list_skills", "list_skills"),
        ("list_replays", "list_replays"),
        ("start_control", "start_control"),
        ("switch_zmq", "switch_zmq"),
        ("switch_keyboard", "switch_keyboard"),
        ("toggle_zmq_streaming", "toggle_zmq_streaming"),
        ("toggle_planner", "toggle_planner"),
        ("keyboard", "keyboard"),
        ("select_mode", "select_mode"),
        ("adjust", "adjust"),
        ("compliance", "compliance"),
    ]
    selected_sim_tools: list[str] = []
    for row_start in range(0, len(sim_tool_options), 5):
        sim_cols = st.columns(5)
        for col, (tool_name, label) in zip(
            sim_cols, sim_tool_options[row_start : row_start + 5]
        ):
            with col:
                enabled = st.checkbox(
                    label,
                    value=tool_name in enabled_sim_tools,
                    key=f"unitree_g1_sim_tool_{tool_name}",
                )
                if enabled:
                    selected_sim_tools.append(tool_name)
    unitree_g1_sim["enabled_tools"] = selected_sim_tools
    unitree_g1_sim["skills"] = unitree_skill_editor(
        "Add replay-backed skills for the sim agent. For example, name `wave left hand` with file `wave_left_hand.npy`.",
        unitree_g1_sim.get("skills", deepcopy(DEFAULT_UNITREE_G1_SIM_SKILLS)),
        key="unitree_g1_sim_skills",
        allow_vla=False,
    )
    env["UNITREE_G1_SIM_DEPLOY_DIR"] = unitree_g1_sim["deploy_dir"]
    env["GR00T_WBC_ROOT"] = unitree_g1_sim["gr00t_root"]
    env["UNITREE_G1_SIM_REPLAY_DIR"] = unitree_g1_sim["replay_dir"]
    env["UNITREE_G1_SIM_TERMINAL_VIEWER"] = str(
        unitree_g1_sim["terminal_viewer"]
    ).lower()
    env["UNITREE_G1_SIM_LOG_DIR"] = unitree_g1_sim["log_dir"]
    with st.expander(
        "Terminal commands for Unitree G1 sim",
        expanded=bool(unitree_g1_sim.get("tools_enabled", False)),
    ):
        st.code(
            build_unitree_g1_sim_terminal_commands(unitree_g1_sim),
            language="bash",
        )

    st.divider()
    st.subheader("Unitree G1 Real Manager Tools")
    unitree_g1_real["tools_enabled"] = st.checkbox(
        "Enable Unitree G1 real manager tools",
        value=bool(unitree_g1_real.get("tools_enabled", False)),
        key="unitree_g1_real_tools_enabled",
        on_change=disable_sdk_sim_tools_when_real_enabled,
        help=(
            "Expose tools that run `source scripts/setup_env.sh && ./deploy.sh "
            "--input-type manager --zmq-host localhost --hand-type inspire real` "
            "and send manager hotkeys. "
            "Mutually exclusive with SDK and sim tools."
        ),
    )
    if unitree_g1_real["tools_enabled"]:
        unitree_g1["tools_enabled"] = False
        unitree_g1_sim["tools_enabled"] = False
    real_path_cols = st.columns(2)
    with real_path_cols[0]:
        unitree_g1_real["deploy_dir"] = st.text_input(
            "Real GR00T deploy directory",
            value=unitree_g1_real.get("deploy_dir", "")
            or env.get("UNITREE_G1_REAL_DEPLOY_DIR", ""),
            help="Path to the real deploy directory that contains deploy.sh.",
        )
        unitree_g1_real["vla_server_root"] = st.text_input(
            "VLA server root",
            value=unitree_g1_real.get("vla_server_root", "")
            or DEFAULT_UNITREE_G1_VLA_SERVER_ROOT,
            help="Path to Isaac-GR00T used to start gr00t/eval/run_gr00t_server.py.",
        )
    with real_path_cols[1]:
        unitree_g1_real["gr00t_root"] = st.text_input(
            "Real GR00T-WholeBodyControl root",
            value=unitree_g1_real.get("gr00t_root", "")
            or env.get("GR00T_WBC_ROOT", ""),
            help="Path used to run gear_sonic replay and VLA inference scripts.",
        )
        unitree_g1_real["replay_dir"] = st.text_input(
            "Real replay directory",
            value=unitree_g1_real.get("replay_dir", "replays/unitree_g1_sim")
            or env.get("UNITREE_G1_REAL_REPLAY_DIR", "replays/unitree_g1_sim"),
        )
    real_start_cols = st.columns(2)
    with real_start_cols[0]:
        unitree_g1_real["auto_start"] = st.checkbox(
            "Auto-start real manager",
            value=bool(unitree_g1_real.get("auto_start", True)),
        )
    with real_start_cols[1]:
        unitree_g1_real["startup_settle_seconds"] = st.number_input(
            "Real startup settle seconds",
            min_value=0.0,
            max_value=30.0,
            value=float(unitree_g1_real.get("startup_settle_seconds", 2.0)),
            step=0.5,
        )
    real_terminal_cols = st.columns(2)
    with real_terminal_cols[0]:
        unitree_g1_real["terminal_viewer"] = st.checkbox(
            "Open real terminal viewer",
            value=bool(unitree_g1_real.get("terminal_viewer", True)),
            help="Open a terminal window when the real manager starts.",
        )
    with real_terminal_cols[1]:
        unitree_g1_real["log_dir"] = st.text_input(
            "Real terminal log directory",
            value=unitree_g1_real.get("log_dir", "logs/unitree_g1_real")
            or env.get("UNITREE_G1_REAL_LOG_DIR", "logs/unitree_g1_real"),
        )
    enabled_real_tools = set(
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
    )
    st.caption("Choose which GR00T real manager tools are exposed to the agent.")
    selected_real_tools: list[str] = []
    for row_start in range(0, len(sim_tool_options), 5):
        real_cols = st.columns(5)
        for col, (tool_name, label) in zip(
            real_cols, sim_tool_options[row_start : row_start + 5]
        ):
            with col:
                enabled = st.checkbox(
                    label,
                    value=tool_name in enabled_real_tools,
                    key=f"unitree_g1_real_tool_{tool_name}",
                )
                if enabled:
                    selected_real_tools.append(tool_name)
    unitree_g1_real["enabled_tools"] = selected_real_tools
    real_skills = unitree_g1_real.get(
        "skills",
        deepcopy(DEFAULT_UNITREE_G1_REAL_SKILLS),
    )
    wave_left_hand_file = st.text_input(
        "Wave left hand replay file",
        value=get_replay_skill_file(
            real_skills,
            "wave left hand",
            DEFAULT_UNITREE_G1_REAL_REPLAY_FILE,
        ),
        help=(
            "Absolute .parquet files run sonic_encoder_input_player_with_hand.py "
            "with --parquet-file. If an absolute path is configured, the backend "
            "first checks the real replay directory for a file with the same name, "
            "then falls back to the absolute path."
        ),
    )
    real_skills = set_replay_skill_file(
        real_skills,
        "wave left hand",
        wave_left_hand_file,
    )
    unitree_g1_real["skills"] = unitree_skill_editor(
        "Add real robot skills. Replay skills can use `.npy` or `.parquet` files; VLA skills use the prompt field as the skill input.",
        real_skills,
        key="unitree_g1_real_skills",
        allow_vla=True,
    )
    unitree_g1_real["skills"] = apply_vla_server_root(
        unitree_g1_real["skills"],
        unitree_g1_real.get("vla_server_root", ""),
    )
    env["UNITREE_G1_REAL_DEPLOY_DIR"] = unitree_g1_real["deploy_dir"]
    env["GR00T_WBC_ROOT"] = unitree_g1_real["gr00t_root"]
    env["UNITREE_G1_REAL_REPLAY_DIR"] = unitree_g1_real["replay_dir"]
    env["UNITREE_G1_REAL_TERMINAL_VIEWER"] = str(
        unitree_g1_real["terminal_viewer"]
    ).lower()
    env["UNITREE_G1_REAL_LOG_DIR"] = unitree_g1_real["log_dir"]
    with st.expander(
        "Terminal commands for Unitree G1 real",
        expanded=bool(unitree_g1_real.get("tools_enabled", False)),
    ):
        st.code(
            build_unitree_g1_real_terminal_commands(unitree_g1_real),
            language="bash",
        )

    st.divider()
    st.subheader("Sensor Tools")
    sensor_tool["enabled"] = st.checkbox(
        "Enable camera sensor tools",
        value=bool(sensor_tool.get("enabled", False)),
    )
    sensor_tool["transport"] = st.selectbox(
        "Camera transport",
        ["zmq", "http"],
        index=["zmq", "http"].index(sensor_tool.get("transport", "zmq")),
    )
    if sensor_tool["transport"] == "zmq":
        sensor_tool["server_ip"] = st.text_input(
            "Sensor server IP",
            sensor_tool.get("server_ip", "localhost"),
        )
        sensor_tool["port"] = int(
            st.number_input(
                "Sensor server port",
                min_value=1,
                max_value=65535,
                value=int(sensor_tool.get("port", 5555)),
                step=1,
            )
        )
    else:
        sensor_tool["http_base_url"] = st.text_input(
            "HTTP camera base URL",
            sensor_tool.get("http_base_url", ""),
            help="Example: http://robot:8000",
        )
    sensor_tool["default_camera"] = st.text_input(
        "Default camera",
        sensor_tool.get("default_camera", ""),
        help="Optional: ego_view, head, left_wrist, or right_wrist.",
    )
    sensor_tool["blocking"] = st.checkbox(
        "Wait for fresh frames",
        value=bool(sensor_tool.get("blocking", True)),
    )
    env["SENSOR_SERVER_IP"] = sensor_tool.get("server_ip", "")
    env["SENSOR_SERVER_PORT"] = str(sensor_tool.get("port", 5555))
    env["SENSOR_HTTP_BASE_URL"] = sensor_tool.get("http_base_url", "")
    env["SENSOR_DEFAULT_CAMERA"] = sensor_tool.get("default_camera", "")

    st.divider()
    st.subheader("ROS2 Tools")
    ros2["ros2_tools"] = st.checkbox(
        "Enable generic ROS2 tools", value=bool(ros2["ros2_tools"])
    )
    ros2["nav2_tools"] = st.checkbox(
        "Enable Nav2 tools", value=bool(ros2["nav2_tools"])
    )
    ros2["ros2_node_name"] = st.text_input("ROS2 node name", ros2["ros2_node_name"])
    ros2["use_sim_time"] = st.checkbox("Use sim time", value=bool(ros2["use_sim_time"]))
    ros2["ros2_readable"] = st.text_input("Readable allowlist", ros2["ros2_readable"])
    ros2["ros2_writable"] = st.text_input("Writable allowlist", ros2["ros2_writable"])
    ros2["ros2_forbidden"] = st.text_input("Forbidden names", ros2["ros2_forbidden"])
    ros2["nav2_frame_id"] = st.text_input("Nav2 frame id", ros2["nav2_frame_id"])
    ros2["nav2_action_name"] = st.text_input("Nav2 action name", ros2["nav2_action_name"])


def tracing_tab(config: dict[str, Any]) -> None:
    st.subheader("Tracing")
    config["tracing"]["project"] = st.text_input(
        "Tracing project", config["tracing"]["project"]
    )
    config["tracing"]["langfuse"]["use_langfuse"] = st.checkbox(
        "Enable Langfuse", value=bool(config["tracing"]["langfuse"]["use_langfuse"])
    )
    config["tracing"]["langfuse"]["host"] = st.text_input(
        "Langfuse host", config["tracing"]["langfuse"]["host"]
    )
    config["tracing"]["langsmith"]["use_langsmith"] = st.checkbox(
        "Enable LangSmith", value=bool(config["tracing"]["langsmith"]["use_langsmith"])
    )
    config["tracing"]["langsmith"]["host"] = st.text_input(
        "LangSmith host", config["tracing"]["langsmith"]["host"]
    )


def preview_tab(config: dict[str, Any]) -> None:
    st.subheader("Preview")
    st.code(tomli_w.dumps(config), language="toml")
    st.caption("Secrets such as OPENAI_API_KEY are saved to .env and are not shown here.")
    st.subheader("Generated command")
    command = [
        "uv run python app/s2s_no_ros.py",
        f"--agent-mode {config['s2s'].get('agent_mode', 'standard')}",
        "--python-tools" if config["tools"]["python_tools"] else "--no-python-tools",
        f"--language {config['s2s']['language']}",
        f"--asr {config['s2s']['asr']}",
        f"--tts {config['s2s']['tts']}",
        f"--whisper-model {config['s2s']['whisper_model']}",
    ]
    if config["ros2"]["ros2_tools"]:
        command.append("--ros2-tools")
    if config["ros2"]["nav2_tools"]:
        command.append("--nav2-tools")
    if config.get("unitree_g1", {}).get("tools_enabled"):
        command.append("--unitree-g1-tools")
        enabled_tools = ",".join(config["unitree_g1"].get("enabled_tools", []))
        if enabled_tools:
            command.append(f"--unitree-g1-enabled-tools {enabled_tools}")
    if config.get("unitree_g1_sim", {}).get("tools_enabled"):
        command.append("--unitree-g1-sim-tools")
        if config["unitree_g1_sim"].get("deploy_dir"):
            deploy_dir = shlex.quote(config["unitree_g1_sim"]["deploy_dir"])
            command.append(
                f"--unitree-g1-sim-deploy-dir {deploy_dir}"
            )
        if config["unitree_g1_sim"].get("gr00t_root"):
            gr00t_root = shlex.quote(config["unitree_g1_sim"]["gr00t_root"])
            command.append(f"--unitree-g1-sim-gr00t-root {gr00t_root}")
        if config["unitree_g1_sim"].get("replay_dir"):
            replay_dir = shlex.quote(config["unitree_g1_sim"]["replay_dir"])
            command.append(f"--unitree-g1-sim-replay-dir {replay_dir}")
        if not config["unitree_g1_sim"].get("auto_start", True):
            command.append("--no-unitree-g1-sim-auto-start")
        settle_seconds = config["unitree_g1_sim"].get("startup_settle_seconds", 2.0)
        command.append(f"--unitree-g1-sim-startup-settle-seconds {settle_seconds}")
        if config["unitree_g1_sim"].get("terminal_viewer", False):
            command.append("--unitree-g1-sim-terminal-viewer")
        if config["unitree_g1_sim"].get("log_dir"):
            log_dir = shlex.quote(config["unitree_g1_sim"]["log_dir"])
            command.append(f"--unitree-g1-sim-log-dir {log_dir}")
        enabled_sim_tools = ",".join(
            config["unitree_g1_sim"].get("enabled_tools", [])
        )
        if enabled_sim_tools:
            command.append(f"--unitree-g1-sim-enabled-tools {enabled_sim_tools}")
    if config.get("unitree_g1_real", {}).get("tools_enabled"):
        command.append("--unitree-g1-real-tools")
        if config["unitree_g1_real"].get("deploy_dir"):
            deploy_dir = shlex.quote(config["unitree_g1_real"]["deploy_dir"])
            command.append(f"--unitree-g1-real-deploy-dir {deploy_dir}")
        if config["unitree_g1_real"].get("gr00t_root"):
            gr00t_root = shlex.quote(config["unitree_g1_real"]["gr00t_root"])
            command.append(f"--unitree-g1-real-gr00t-root {gr00t_root}")
        if config["unitree_g1_real"].get("replay_dir"):
            replay_dir = shlex.quote(config["unitree_g1_real"]["replay_dir"])
            command.append(f"--unitree-g1-real-replay-dir {replay_dir}")
        if not config["unitree_g1_real"].get("auto_start", True):
            command.append("--no-unitree-g1-real-auto-start")
        settle_seconds = config["unitree_g1_real"].get("startup_settle_seconds", 2.0)
        command.append(f"--unitree-g1-real-startup-settle-seconds {settle_seconds}")
        if config["unitree_g1_real"].get("terminal_viewer", False):
            command.append("--unitree-g1-real-terminal-viewer")
        if config["unitree_g1_real"].get("log_dir"):
            log_dir = shlex.quote(config["unitree_g1_real"]["log_dir"])
            command.append(f"--unitree-g1-real-log-dir {log_dir}")
        enabled_real_tools = ",".join(
            config["unitree_g1_real"].get("enabled_tools", [])
        )
        if enabled_real_tools:
            command.append(f"--unitree-g1-real-enabled-tools {enabled_real_tools}")
    if config.get("sensor_tool", {}).get("enabled"):
        command.append("--sensor-tools")
        command.append(f"--sensor-transport {config['sensor_tool']['transport']}")
        if config["sensor_tool"]["transport"] == "zmq":
            command.append(f"--sensor-server-ip {config['sensor_tool']['server_ip']}")
            command.append(f"--sensor-server-port {config['sensor_tool']['port']}")
        else:
            command.append(f"--sensor-http-base-url {config['sensor_tool']['http_base_url']}")
        if config["sensor_tool"].get("default_camera"):
            command.append(f"--sensor-default-camera {config['sensor_tool']['default_camera']}")
    st.code(" ".join(command), language="bash")


def main() -> None:
    st.set_page_config(page_title="S2S Agent Configurator", layout="wide")
    st.title("S2S Agent Configurator")

    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "env" not in st.session_state:
        st.session_state.env = load_env()

    config = st.session_state.config
    tabs = st.tabs(["Models", "S2S", "Tools", "Tracing", "Preview"])
    with tabs[0]:
        model_tab(config)
    with tabs[1]:
        s2s_tab(config)
    with tabs[2]:
        tools_tab(config)
    with tabs[3]:
        tracing_tab(config)
    with tabs[4]:
        preview_tab(config)

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Save config", type="primary"):
            save_config(config)
            save_env(st.session_state.env)
            st.success(f"Saved {CONFIG_PATH} and {ENV_PATH}")
    with col2:
        if st.button("Reload from disk"):
            st.session_state.config = load_config()
            st.session_state.env = load_env()
            st.rerun()


if __name__ == "__main__":
    main()
