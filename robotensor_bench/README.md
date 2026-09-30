# robotensor_bench

Benchmark harness for [Robotensor Vector](https://github.com/robotensor/vector-orchestrator),
built into this fork of [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). It runs
from the checkout (not installed) under the simulator's Python, and the orchestrator drives it
only through its CLI.

## Commands

Run from the repository root:

```bash
python -m robotensor_bench.cli <command>
```

| Command | |
|---|---|
| `demo make` | Record an expert demonstration on the first candidate seed the expert solves; with `--eval-seed-candidates`, also pick the scoring scene |
| `demo verify` | Check a recorded prompt against its unit, without the simulator |
| `eval run` | Roll out a served policy (`--policy ADDR`) or a stub on the scoring scene; `--step-ratio R` caps steps at R × the expert's |
| `tasks list` | Task table (`--json` for the orchestrator) |
| `config check` | Print the harness version, source digest, robots, cameras and limits |
| `selftest` | `demo make` → `demo verify` → `eval run` with the replay stub |

Exit codes: `0` result written (including fail/void), `1` selftest failed, `2` bad input,
`3` simulator/assets/GPU unavailable, `4` prompt failed verification.

`--expect-source-sha256` refuses to run if the harness source changed, so a checkout edited
mid-duel cannot score a unit.

## Changes to upstream

Forked from upstream `6dde571`. The only change to upstream files is a difficulty-level gate:

- `envs/_levels.py`, `env_cfg/task_config/level{1..4}.yml`: randomization sources are enabled
  per level (1–4); one seed gives the same layout at every level.
- `envs/_base_task.py` and the 16 Vector task files route their random draws through the gate.
  `level` defaults to 4, so upstream configs behave as before.
- `envs/robot/robot.py`: rebuild planners on reset after a failed build.
- `envs/utils/pkl2hdf5.py`: store per-episode level and seed info.

## Setup and tests

```bash
bash robotensor_bench/scripts/install_robotwin.sh   # simulator environment

pip install -r robotensor_bench/requirements.txt pytest
PYTHONPATH=$PWD pytest -c robotensor_bench/pytest.ini robotensor_bench/tests -m "not sim"
python -m robotensor_bench.cli selftest             # needs the simulator and a GPU
```

## Troubleshooting

- **`cannot import name 'CuroboPlanner'`**: CuRobo is missing or failed to build. Rerun
  `install_robotwin.sh` and read the CuRobo build output.
- **`no kernel image is available` / `Unknown CUDA arch (12.0)`**: a Blackwell GPU on the
  reference torch. Rebuild the env with `ROBOTWIN_GPU_PATH=blackwell`.
- **`failed to find a rendering device`** or a missing `egl_vendor.d`: the container lacks
  `libegl1` or the NVIDIA EGL/Vulkan manifests. Rerun the installer; `scripts/test_render.py`
  must print "Render Well".
