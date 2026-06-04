# RAI S2S Agent Bundle

This bundle contains the source needed to run the no-ROS speech-to-speech agent:

- `src/rai_core`: core agent, communication, initialization, and tool code
- `src/rai_s2s`: ASR, VAD, TTS, sound device, and S2S agent code
- `app/s2s_no_ros.py`: standalone entry file
- `config.toml`: model/provider configuration

## Setup

```bash
cp .env.example .env
# edit .env if you use OpenAI-compatible cloud models
./setup_env.sh
```

## Run

Open the configurator:

```bash
uv run streamlit run app/configurator.py
```

Pure local ASR with Python tools:

```bash
uv run python app/s2s_no_ros.py --python-tools --language zh
```

After saving config in the configurator, the same defaults can be used directly:

```bash
uv run python app/s2s_no_ros.py
```

List audio devices:

```bash
uv run python app/s2s_no_ros.py --list-devices
```

## Optional Doubao Speech

Open the configurator and choose `doubao` for ASR and/or TTS:

```bash
uv run streamlit run app/configurator.py
```

Fill the Doubao Speech API section. Secrets are saved to `.env`:

```bash
DOUBAO_APP_ID=
DOUBAO_TOKEN=
DOUBAO_ASR_APP_KEY=
DOUBAO_ASR_ACCESS_KEY=
DOUBAO_ASR_API_KEY=
DOUBAO_TTS_VOICE_TYPE=
```

Then run with the configured defaults:

```bash
uv run python app/s2s_no_ros.py --config config.toml
```

You can also override from the command line:

```bash
uv run python app/s2s_no_ros.py \
  --asr doubao \
  --tts doubao \
  --doubao-tts-voice-type YOUR_VOICE_TYPE \
  --language zh
```

Use explicit audio device names:

```bash
uv run python app/s2s_no_ros.py \
  --mic-device "MacBook Pro Microphone" \
  --speaker-device "MacBook Pro Speakers" \
  --python-tools \
  --language zh
```

## Optional ROS2 Tools

The entry file still supports ROS2 tools, but ROS2 is not bundled. Before using
`--ros2-tools` or `--nav2-tools`, install and source ROS2 in the runtime
environment.

```bash
source /opt/ros/jazzy/setup.bash
uv run python app/s2s_no_ros.py --nav2-tools --language zh
```

## Optional Unitree G1 Tools

The Unitree SDK is intentionally not included in the default bundle
dependencies, so macOS-only testing will not download or build it.

On a robot/Ubuntu machine, install the SDK manually before enabling these
tools:

```bash
uv pip install "unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python.git"
```

Open the configurator and enable `Unitree G1 Tools`:

```bash
uv run streamlit run app/configurator.py
```

Set the network interface connected to the robot, for example `en0`, `en7`, or
`eth0`. Keep `Allow movement/posture commands` disabled while testing tool
calling. Enable it only after confirming the robot area is safe.

Run with the configured defaults:

```bash
uv run python app/s2s_no_ros.py
```

Or override from the command line:

```bash
uv run python app/s2s_no_ros.py \
  --unitree-g1-tools \
  --unitree-g1-network-interface en0 \
  --unitree-g1-control-enabled
```

## Optional Unitree G1 Speaker

For a standalone host controlling G1 over the network, the agent can send TTS
audio to the robot speaker through the Unitree `AudioClient`. Install the
Unitree SDK on that host first, then set the network interface connected to G1.

In the configurator, enable `Use Unitree G1 speaker` under `Tools`.

Command-line example:

```bash
uv run python app/s2s_no_ros.py \
  --speaker-backend unitree_g1 \
  --unitree-g1-audio-network-interface en0
```

The speaker backend converts generated TTS audio to 16 kHz mono 16-bit PCM
before calling `AudioClient.PlayStream`.
