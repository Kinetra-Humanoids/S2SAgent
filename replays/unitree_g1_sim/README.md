# Unitree G1 Sim Replays

Place GR00T WholeBodyControl latent replay files (`.npy`) in this directory.
The S2S agent exposes them through `unitree_g1_sim_perform_replay`.

Expected starter filenames:

- `wave_left_hand.npy`
- `run.npy`
- `squat_stand.npy`

These files are not bundled because they are model/generated motion assets. Copy
the real files here, or point `[unitree_g1_sim].replay_dir` at an existing replay
directory.

Quick test after `wave_left_hand.npy` is available:

```bash
uv run python app/test_unitree_g1_sim_replay.py --action wave_left_hand
```
