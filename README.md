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

The default environment installs the configured OpenAI-compatible LLM and
Doubao speech path. Install local speech models only when needed:

```bash
uv sync --extra local-asr
uv sync --extra local-tts
uv sync --extra all-local-models
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

Quick standalone test for Doubao TTS -> Unitree G1 speaker:

```bash
uv run python app/test_doubao_unitree_g1_audio.py \
  "你好，我是 RAI。正在测试豆包语音合成和宇树 G1 播放。" \
  --unitree-g1-audio-network-interface en0
```

Use `--no-play` to only verify Doubao synthesis and save
`doubao_unitree_g1_test.wav`.

## Optional Unitree G1 Sim Manager Tools

The bundle can also expose GR00T WholeBodyControl manager controls as tools for
the Unitree G1 MuJoCo sim. This toolset owns the C++ deployment terminal process
and sends the same hotkeys documented by the GR00T manager tutorial.

Start the MuJoCo sim loop separately from the GR00T repo root:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Then enable `Unitree G1 Sim Manager Tools` in the configurator and set:

- `GR00T deploy directory`: the `gear_sonic_deploy` directory containing
  `deploy.sh`
- `GR00T-WholeBodyControl root`: the repo root used to run
  `gear_sonic/scripts/sonic_encoder_input_player.py`
- `Replay .npy directory`: a directory containing latent replay files

When the S2S agent starts with `--unitree-g1-sim-tools`, it automatically starts
the deployment process and sends `]` to enter control mode:

```bash
bash deploy.sh --input-type manager sim
```

Command-line example:

```bash
uv run python app/s2s_no_ros.py \
  --unitree-g1-sim-tools \
  --unitree-g1-sim-deploy-dir /home/ljc/GR00T-WholeBodyControl/gear_sonic_deploy \
  --unitree-g1-sim-gr00t-root /home/ljc/GR00T-WholeBodyControl \
  --unitree-g1-sim-replay-dir replays/unitree_g1_sim
```

For named motions, put real latent `.npy` files in
`replays/unitree_g1_sim/` or point `--unitree-g1-sim-replay-dir` to your own
folder. Starter names are declared in
`replays/unitree_g1_sim/replay_manifest.toml`:

- `wave_left_hand.npy`
- `run.npy`
- `squat_stand.npy`

The high-level `unitree_g1_sim_perform_replay` tool maps requests such as
"wave left hand", "run", or "蹲起" to those files. It switches the manager to
ZMQ mode (`#`), sends ENTER, then starts a separate shell process equivalent to:

```bash
cd /home/ljc/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/sonic_encoder_input_player.py \
  --latent-input-file /path/to/replays/unitree_g1_sim/wave_left_hand.npy
```

Lower-level tool groups are still available for process lifecycle (`start`,
`stop`, `status`), manager interface switching (`keyboard`, `gamepad`, `zmq`,
`ros2`), control startup (`]`), planner toggle (ENTER), keyboard motion
commands (`WASD`, `T/R/P/N`, `Q/E`, `,/.`), planner mode selection (`1`-`8`),
speed/height adjustment, and hand compliance controls. Say "stop" or "停止" to
trigger the sim manager stop tool; it sends `O` before terminating the process.
