"""Checking a made demonstration against the unit it was made for: `rt-bench demo verify`.

A duel hands both sides the same prompt, and a published record carries its bytes, so anyone
holding a prompt must be able to confirm it is the one its unit asked for. Pure: it reads the file,
never unpickles it, and needs no simulator, assets or GPU.

The unit is what `demo make` was told: the task on its robot, under a task config and frame spacing,
with the candidate seeds tried in order. In the cross-scene setting the demonstration is recorded on
one of `seed_candidates` and the unit is scored on one of `eval_seed_candidates`; otherwise both
happen on one of `seed_candidates`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import prompt as prompt_
from .records import CROSS_SCENE_LEVEL, SAME_SCENE
from .scene import SceneFingerprint, digest

#: `qpos` width per robot: joints then a gripper per arm, six joints an arm on aloha-agilex and
#: seven on a Franka. An `ee` action is 16 on either (`robotwin.EE_ACTION_DIM`).
QPOS_DIMS = {"aloha-agilex": 14, "franka-panda": 16}

#: The cameras a policy observes under every task config a unit is made with (`demo_clean` and the
#: level configs collect the head camera and both wrist cameras), named as `envs/camera/camera.py`
#: names them. A prompt and an observation hold `frames_<camera>` of each.
CAMERAS = ("head_camera", "left_camera", "right_camera")


@dataclass(frozen=True)
class Unit:
    """What `demo make` was asked for, which a prompt is held to."""

    task: str
    embodiment: str
    task_config: str
    save_freq: int
    seed_candidates: tuple[int, ...]
    eval_seed_candidates: tuple[int, ...] | None = None
    #: The task config's difficulty level, which a cross-scene prompt must record.
    level: int | None = None

    @property
    def cross_scene(self) -> bool:
        return self.eval_seed_candidates is not None

    @property
    def scored_on(self) -> list[int]:
        """The candidates the unit is scored on, in the order they are tried."""
        return list(self.eval_seed_candidates if self.cross_scene else self.seed_candidates)

    def scene_config(self) -> dict[str, Any]:
        """The scene config `demo make` records in a prompt's meta and `eval run` rebuilds from:
        the task config and frame spacing, no head camera or override, and the setting."""
        return {
            "task_config": self.task_config,
            "save_freq": self.save_freq,
            "head_camera": None,
            "overrides": {},
            "evaluation_setting": CROSS_SCENE_LEVEL if self.cross_scene else SAME_SCENE,
        }


def sha256_of(path: Path) -> str:
    digest_ = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest_.update(block)
    return digest_.hexdigest()


def expert_problems(expert: Any, seed: int, candidates: Sequence[int]) -> list[str]:
    """What is wrong with a prompt's expert record for a unit whose candidates hold its seed.

    `demo make` tries the candidates in order and keeps the first the expert solves, so the record
    must name the unit's candidates, have tried a prefix of them ending at the scene seed, and
    reject every one before it: a scene seed that is merely a candidate could have been picked.
    """
    candidates = list(candidates)
    expert = expert if isinstance(expert, dict) else {}
    if expert.get("scene_seeds") != candidates:
        return [
            f"the prompt's expert was given candidates {expert.get('scene_seeds')!r}; the unit's "
            f"are {candidates}"
        ]
    attempts = expert.get("attempts")
    if not isinstance(attempts, list) or not all(isinstance(a, dict) for a in attempts):
        return [f"the prompt's expert record has no attempts, but {attempts!r}"]
    tried = [a.get("seed") for a in attempts]
    if not tried or tried != candidates[: len(tried)]:
        return [f"the prompt's expert tried seeds {tried}, not the unit's candidates in order"]
    problems = []
    if tried[-1] != seed:
        problems.append(
            f"the prompt's expert last tried seed {tried[-1]}, not its scene seed {seed}"
        )
    elif attempts[-1].get("rejection") is not None:
        problems.append("the prompt's expert record rejects the seed the prompt was built on")
    kept = [a.get("seed") for a in attempts[:-1] if not a.get("rejection")]
    if kept:
        problems.append(
            f"the prompt's expert record keeps no rejection for candidate(s) {kept} tried before "
            "its scene seed"
        )
    return problems


def verify_prompt(path: str | Path, unit: Unit) -> dict[str, Any]:
    """Whether the prompt at `path` is the one `unit` asked for; every problem, not the first.

    Reads the file itself, never a manifest, and never unpickles it: its `meta` must name the
    unit's task, one of the candidate scene seeds it is scored on (found by trying them in order)
    and its robot, as that robot was chosen, and the scene config every unit is built with, which
    `eval run` rebuilds the scene from; every array must belong to the published channel map and
    together hold a demonstration of the robot's width, from no camera but `CAMERAS`; and the
    recorded scene fingerprint must digest to the digest beside it. Returns `{"ok", "sha256",
    "problems"}`, where `sha256` is the file's bytes' (None if it cannot be read), plus the `task`,
    `scene_seed` and `embodiment` it found.
    """
    file = Path(path)
    problems: list[str] = []
    found: dict[str, Any] = {"task": None, "scene_seed": None, "embodiment": None}

    def verdict(sha256: str | None) -> dict[str, Any]:
        return {"ok": not problems, "sha256": sha256, "problems": problems, **found}

    try:
        sha256 = sha256_of(file)
    except OSError as exc:
        problems.append(f"cannot read {file}: {exc}")
        return verdict(None)
    try:
        arrays, meta = prompt_.read_raw(file)
    except prompt_.PromptError as exc:
        problems.append(str(exc))
        return verdict(sha256)

    embodiment = meta.get("embodiment") if isinstance(meta.get("embodiment"), dict) else {}
    found.update(
        task=meta.get("task"), scene_seed=meta.get("scene_seed"), embodiment=embodiment.get("name")
    )

    if meta.get("schema") != prompt_.PROMPT_SCHEMA:
        problems.append(f"meta schema is {meta.get('schema')!r}, not {prompt_.PROMPT_SCHEMA}")
    if meta.get("task") != unit.task:
        problems.append(
            f"the prompt is of task {meta.get('task')!r}; the unit asks for {unit.task!r}"
        )

    seed, candidates = meta.get("scene_seed"), unit.scored_on
    if not candidates:
        problems.append("the unit names no candidate scene seeds")
    elif isinstance(seed, bool) or not isinstance(seed, int) or seed not in candidates:
        problems.append(
            f"the prompt's scene seed {seed!r} is not one of the unit's candidates {candidates}"
        )
    else:
        problems.extend(expert_problems(meta.get("expert"), seed, candidates))

    if unit.cross_scene:
        problems.extend(_demonstration_problems(meta, unit, seed))
    elif meta.get("demonstration_scene") is not None:
        problems.append("a Same Scene prompt names a demonstration scene of its own")

    name, choice = embodiment.get("name"), embodiment.get("choice")
    if name != unit.embodiment or choice != unit.embodiment:
        problems.append(
            f"the prompt was built on {name!r} (chosen as {choice!r}); the unit asks for "
            f"{unit.embodiment!r}"
        )

    # eval run rebuilds and scores the scene from exactly these, so a prompt built under another
    # config is another unit however its task, seed and robot read.
    for key, value in unit.scene_config().items():
        if key not in meta or meta[key] != value or type(meta[key]) is not type(value):
            problems.append(
                f"the prompt's {key} is {meta.get(key)!r}; every unit is built with {value!r}"
            )

    unclaimed = sorted(a for a in arrays if prompt_.channel_of(a) in (None, "privileged"))
    if unclaimed:
        problems.append(f"arrays in no published channel: {', '.join(unclaimed)}")
    try:
        demonstration = prompt_.demonstration_from(arrays)
    except prompt_.PromptError as exc:
        problems.append(f"the arrays do not hold a demonstration: {exc}")
    else:
        wanted = unit.embodiment
        if wanted in QPOS_DIMS and demonstration.qpos_dim != QPOS_DIMS[wanted]:
            problems.append(
                f"qpos is {demonstration.qpos_dim} wide; {wanted} is {QPOS_DIMS[wanted]}"
            )
        if meta.get("frames") != len(demonstration):
            problems.append(
                f"meta says {meta.get('frames')!r} frames; the arrays hold {len(demonstration)}"
            )
        if meta.get("cameras") != list(demonstration.cameras):
            problems.append(
                f"meta names cameras {meta.get('cameras')!r}; the arrays hold "
                f"{list(demonstration.cameras)}"
            )
        # eval run hands the policy every camera the prompt holds, so one no unit collects would
        # reach the policy as if the benchmark observed it.
        unobserved = sorted(set(demonstration.cameras) - set(CAMERAS))
        if unobserved:
            problems.append(
                f"frames of camera(s) {', '.join(unobserved)}, which no unit observes; the "
                f"benchmark's cameras are {', '.join(CAMERAS)}"
            )

    scene = meta.get("scene")
    try:
        recorded = digest(SceneFingerprint.from_json(scene["fingerprint"]))
        if recorded != scene["sha256"]:
            problems.append(
                f"the recorded scene digests to {recorded[:12]}, meta says "
                f"{str(scene['sha256'])[:12]}: meta was edited after it was written"
            )
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"meta has no readable scene fingerprint: {type(exc).__name__}: {exc}")
    return verdict(sha256)


def _demonstration_problems(meta: Mapping[str, Any], unit: Unit, scene_seed: Any) -> list[str]:
    """What is wrong with a cross-scene prompt's demonstration scene for its unit: it must be one
    of the unit's `seed_candidates`, found by trying them in order (`expert_problems`), never the
    scene the unit is scored in, at the task config's level."""
    problems = []
    if meta.get("level") != unit.level:
        problems.append(
            f"the prompt is of level {meta.get('level')!r}; {unit.task_config} is level "
            f"{unit.level!r}"
        )
    scene = meta.get("demonstration_scene")
    candidates = list(unit.seed_candidates)
    if not isinstance(scene, Mapping):
        return [*problems, "a cross-scene prompt names no demonstration scene"]
    seed = scene.get("seed")
    if not candidates:
        problems.append("the unit names no candidate demonstration seeds")
    elif isinstance(seed, bool) or not isinstance(seed, int) or seed not in candidates:
        problems.append(
            f"the prompt's demonstration seed {seed!r} is not one of the unit's seed candidates "
            f"{candidates}"
        )
    else:
        problems.extend(
            f"demonstration: {problem}"
            for problem in expert_problems(scene.get("expert"), seed, candidates)
        )
    if seed == scene_seed:
        problems.append("the demonstration was recorded in the scene the unit is scored in")
    if not isinstance(scene.get("sha256"), str) or not scene.get("sha256"):
        problems.append("the demonstration scene has no digest")
    return problems
