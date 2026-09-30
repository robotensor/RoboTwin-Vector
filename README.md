# RoboTwin-Vector

The benchmark for **[Robotensor Vector](https://competition.robotensor.ai/docs/vector)**, a
King-of-the-Hill competition for one-demonstration imitation. This is a fork of
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) with a benchmark harness,
[`robotensor_bench`](robotensor_bench/README.md), built in. The Vector duel engine,
[vector-orchestrator](https://github.com/robotensor/vector-orchestrator), runs it from this
checkout (branch `vector`) to play every unit of every duel.

## Where it fits

| Repo | Role |
|---|---|
| [robotensor-subnet](https://github.com/robotensor/robotensor-subnet) | The validator: reads miners' commitments from chain, queues them, sets weights |
| [vector-orchestrator](https://github.com/robotensor/vector-orchestrator) | Runs a duel between the challenger and the king, scores it, publishes the record |
| **RoboTwin-Vector** (this repo) | The simulator and the harness: records demonstrations and rolls out policies |

The orchestrator never imports this repo. It calls `python -m robotensor_bench.cli` under the
simulator's Python, and passes `--expect-source-sha256` so a checkout edited in the middle of a duel
can't score a unit.

## What a duel plays

The contract is the orchestrator's
[`spec.json`](https://github.com/robotensor/vector-orchestrator/blob/main/spec.json). At the time
of writing it pins:

- **Robot:** `franka-panda` (two arms), three cameras (head and both wrists).
- **Level:** 1 (`env_cfg/task_config/level1.yml`). Between the demonstration's scene and the
  policy's, only the objects' position and yaw change, plus which arm does the job when the task
  allows either.
- **16 tasks:** `beat_block_hammer`, `move_stapler_pad`, `place_bread_skillet`,
  `place_burger_fries`, `place_can_basket`, `place_cans_plasticbox`, `place_container_plate`,
  `place_dual_shoes`, `place_empty_cup`, `place_fan`, `place_mouse_pad`, `place_object_basket`,
  `place_object_scale`, `place_object_stand`, `place_phone_stand`, `place_shoe`.
- **Units:** each unit is one task, a demonstration recorded by the expert in one scene, and a
  different scene the policy must do the task in. A rollout is capped at 2× the expert's steps.
- **Seeds:** duel scenes are drawn from `[2,000,000, 2^31 - 1)`, above the public seed ranges
  `[0, 2M)`, so a policy can't have seen them.

The competition rules, the model architecture (`vector_v1.1`) and how the crown moves are in the
[Vector docs](https://competition.robotensor.ai/docs/vector).

## Quick start

```bash
git clone -b vector https://github.com/robotensor/RoboTwin-Vector.git
cd RoboTwin-Vector

# Simulator environment: a conda env named `robotwin` (python 3.10), CuRobo, rendering, assets.
# Picks the torch/CUDA build from your GPU; ROBOTWIN_GPU_PATH=blackwell forces the RTX 50xx path.
bash robotensor_bench/scripts/install_robotwin.sh

# Demonstration -> verify -> rollout with the replay stub. Needs a GPU.
conda activate robotwin
python -m robotensor_bench.cli selftest
```

The harness commands (`demo make`, `demo verify`, `eval run`, `tasks list`, `config check`),
exit codes, tests and troubleshooting are in [`robotensor_bench/README.md`](robotensor_bench/README.md).

## Changes to upstream

Forked from upstream `6dde571`. Apart from adding `robotensor_bench/`, the fork changes upstream
files only to add a difficulty-level gate: `envs/_levels.py` and
`env_cfg/task_config/level{1..4}.yml`, with the task files routing their random draws through it.
`level` defaults to 4, so upstream configs behave as before. The full list is in
[`robotensor_bench/README.md`](robotensor_bench/README.md#changes-to-upstream).

## Citation

This benchmark is built on RoboTwin. If you use it, please cite RoboTwin 2.0:

```bibtex
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

Earlier RoboTwin papers are listed in the
[upstream README](https://github.com/RoboTwin-Platform/RoboTwin#-citations).

## License

RoboTwin is released under the MIT license ([LICENSE](LICENSE)). `robotensor_bench/` is released
under Apache-2.0 ([robotensor_bench/LICENSE](robotensor_bench/LICENSE)).
