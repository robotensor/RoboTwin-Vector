"""`demo verify`'s check: a made prompt is held to the unit it was made for, read from the file.

The prompt goes through the harness's own writers - `unit.build_meta` and `prompt.write_prompt` - so
what `check.verify_prompt` reads is laid out exactly as `demo make` writes it.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robotensor_bench import check, prompt, unit
from robotensor_bench.demo import Demonstration, Frame
from robotensor_bench.robotwin import EMBODIMENTS
from robotensor_bench.scene import SceneFingerprint

CANDIDATES = [5, 11, 17, 23]


def demonstration(qpos_dim, frames=4):
    endpose = {
        "left_endpose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "left_gripper": 1.0,
        "right_endpose": [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
        "right_gripper": 0.0,
    }
    return Demonstration(
        frames=tuple(
            Frame(
                index=i,
                images={"head_camera": np.full((8, 10, 3), i, dtype=np.uint8)},
                qpos=np.full(qpos_dim, i / 10.0),
                endpose=endpose,
                time_s=i / 15.0,
            )
            for i in range(frames)
        ),
        frequency=250 / 15,
    )


@pytest.fixture
def make_prompt(tmp_path):
    """`make_prompt(task=, scene_seed=, embodiment=, qpos_dim=)`: the path of a prompt.npz."""

    def make(task="click_bell", scene_seed=11, embodiment="franka-panda", qpos_dim=None, name="p"):
        qpos_dim = qpos_dim or check.QPOS_DIMS[embodiment]
        demo = demonstration(qpos_dim)
        initial = SceneFingerprint(
            actors={"cube": np.array([0.1, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])},
            articulations={},
            articulation_roots={},
            cameras={"head_camera": np.eye(4)[:3]},
            robot_qpos=np.zeros(qpos_dim),
        )
        config = SimpleNamespace(
            embodiment=embodiment,
            task_config="demo_clean",
            save_freq=15,
            head_camera=None,
            overrides=None,
        )
        args = {"embodiment_name": embodiment, "embodiment": list(EMBODIMENTS[embodiment])}
        # As demo make records it: the unit's candidates tried in order up to the one kept, every
        # earlier one rejected. A seed that is no candidate keeps build_meta's one-seed record.
        expert = None
        if scene_seed in CANDIDATES:
            tried = CANDIDATES[: CANDIDATES.index(scene_seed) + 1]
            attempts = [{"seed": s, "rejection": "unstable", "detail": "moved"} for s in tried[:-1]]
            attempts.append({"seed": scene_seed, "rejection": None, "detail": ""})
            rejected = {"unstable": len(tried) - 1} if len(tried) > 1 else {}
            expert = {"scene_seeds": list(CANDIDATES), "attempts": attempts, "rejections": rejected}
        meta = unit.build_meta(task, scene_seed, config, args, demo, initial, expert)
        path = tmp_path / name / "prompt.npz"
        prompt.write_prompt(path, demo, meta)
        return path

    return make


#: A Same Scene unit on two Franka arms whose candidates hold the prompt `make_prompt` writes.
FRANKA = check.Unit(
    task="click_bell",
    embodiment="franka-panda",
    task_config="demo_clean",
    save_freq=15,
    seed_candidates=tuple(CANDIDATES),
)


def test_the_prompt_a_unit_asked_for_is_verified(make_prompt):
    path = make_prompt()
    verdict = check.verify_prompt(path, FRANKA)
    assert verdict["ok"] is True and verdict["problems"] == []
    assert verdict["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (verdict["task"], verdict["scene_seed"], verdict["embodiment"]) == (
        "click_bell",
        11,
        "franka-panda",
    )


@pytest.mark.parametrize(
    ("made", "reason"),
    [
        ({"task": "click_alarmclock"}, "the prompt is of task 'click_alarmclock'"),
        ({"scene_seed": 12}, "scene seed 12 is not one of the unit's candidates"),
        ({"embodiment": "aloha-agilex"}, "built on 'aloha-agilex'"),
        ({"qpos_dim": 14}, "qpos is 14 wide; franka-panda is 16"),
    ],
)
def test_a_prompt_that_is_not_the_units_is_refused(make_prompt, made, reason):
    verdict = check.verify_prompt(make_prompt(**made), FRANKA)
    assert verdict["ok"] is False and any(reason in p for p in verdict["problems"])
    assert len(verdict["sha256"]) == 64


def _rewrite(path, arrays=None, meta=None):
    old_arrays, old_meta = prompt.read_raw(path)
    arrays = old_arrays if arrays is None else arrays(old_arrays)
    meta = old_meta if meta is None else meta(old_meta)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays, meta=json.dumps(meta))


def test_a_pickled_npz_is_refused_without_being_unpickled(make_prompt, tmp_path):
    path = make_prompt()
    arrays, meta = prompt.read_raw(path)
    pickled = tmp_path / "pickled.npz"
    with open(pickled, "wb") as handle:
        np.savez(handle, **arrays, meta=json.dumps(meta), extra=np.array([{"a": 1}], dtype=object))
    verdict = check.verify_prompt(pickled, FRANKA)
    assert verdict["ok"] is False and "allow_pickle=False" in " ".join(verdict["problems"])


@pytest.mark.parametrize(
    ("arrays", "meta", "reason"),
    [
        (lambda a: {**a, "notes": np.zeros(3)}, None, "arrays in no published channel: notes"),
        (lambda a: {k: v for k, v in a.items() if k != "actions"}, None, "no 'actions' array"),
        (lambda a: {**a, "qpos": a["qpos"].astype(np.float32)}, None, "qpos is float32"),
        (None, lambda m: {**m, "frames": 99}, "meta says 99 frames"),
        (None, lambda m: {**m, "scene": {**m["scene"], "sha256": "0" * 64}}, "was edited"),
        (None, lambda m: {k: v for k, v in m.items() if k != "scene"}, "no readable scene"),
        (
            None,
            lambda m: {**m, "embodiment": {**m["embodiment"], "choice": None}},
            "chosen as None",
        ),
    ],
)
def test_a_prompt_whose_arrays_or_meta_do_not_hold_up_is_refused(make_prompt, arrays, meta, reason):
    path = make_prompt()
    _rewrite(path, arrays, meta)
    verdict = check.verify_prompt(path, FRANKA)
    assert verdict["ok"] is False and any(reason in p for p in verdict["problems"]), verdict


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("task_config", "demo_randomized"),
        ("save_freq", 5),
        ("save_freq", 15.0),
        ("overrides", {"step_lim": 100000}),
        ("head_camera", "D435"),
        ("evaluation_setting", "different_object_pose"),
        ("task_config", None),
    ],
)
def test_a_prompt_built_under_another_scene_config_is_refused(make_prompt, key, value):
    # eval run rebuilds and scores the scene from these: its task, seed and robot can all be the
    # unit's and it still be another unit.
    path = make_prompt()
    _rewrite(path, meta=lambda m: {**m, key: value})
    verdict = check.verify_prompt(path, FRANKA)
    assert verdict["ok"] is False
    assert any(f"the prompt's {key} is {value!r}" in p for p in verdict["problems"]), verdict


def _expert(**changes):
    """A meta edit replacing fields of the expert record `make_prompt` wrote."""
    return lambda m: {**m, "expert": {**m["expert"], **changes}}


KEPT_11 = {"seed": 11, "rejection": None, "detail": ""}


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (_expert(scene_seeds=[11]), "was given candidates [11]; the unit's are [5, 11, 17, 23]"),
        (
            # A later candidate cherry-picked: the first was never tried.
            _expert(attempts=[KEPT_11]),
            "tried seeds [11], not the unit's candidates in order",
        ),
        (
            _expert(attempts=[{"seed": 5, "rejection": None, "detail": ""}, KEPT_11]),
            "keeps no rejection for candidate(s) [5]",
        ),
        (
            _expert(attempts=[{"seed": 5, "rejection": "unstable", "detail": ""}]),
            "last tried seed 5, not its scene seed 11",
        ),
        (
            _expert(
                attempts=[
                    {"seed": 5, "rejection": "unstable", "detail": ""},
                    {**KEPT_11, "rejection": "unstable"},
                ]
            ),
            "rejects the seed the prompt was built on",
        ),
        (lambda m: {k: v for k, v in m.items() if k != "expert"}, "was given candidates None"),
    ],
)
def test_a_prompt_whose_expert_record_is_not_the_units_is_refused(make_prompt, edit, reason):
    # The scene seed alone being a candidate says nothing of how it was chosen.
    path = make_prompt()
    _rewrite(path, meta=edit)
    verdict = check.verify_prompt(path, FRANKA)
    assert verdict["ok"] is False and any(reason in p for p in verdict["problems"]), verdict


def test_a_prompt_holding_a_camera_no_unit_observes_is_refused(make_prompt):
    # eval run hands the policy every camera a prompt holds; meta edited to match hides nothing.
    path = make_prompt()

    def extra_camera(arrays):
        return {**arrays, "frames_extra": np.zeros_like(arrays["frames_head_camera"])}

    _rewrite(path, extra_camera, lambda m: {**m, "cameras": ["extra", "head_camera"]})
    verdict = check.verify_prompt(path, FRANKA)
    assert verdict["ok"] is False
    assert any("camera(s) extra, which no unit observes" in p for p in verdict["problems"])
    assert len(verdict["problems"]) == 1, verdict  # the arrays and meta agree with each other


def test_a_prompt_whose_meta_leaves_out_its_scene_config_is_refused(make_prompt):
    path = make_prompt()
    _rewrite(path, meta=lambda m: {k: v for k, v in m.items() if k != "overrides"})
    verdict = check.verify_prompt(path, FRANKA)
    assert verdict["ok"] is False and any("overrides is None" in p for p in verdict["problems"])


def test_an_unreadable_prompt_is_refused(tmp_path):
    missing = check.verify_prompt(tmp_path / "missing.npz", FRANKA)
    assert missing["ok"] is False and missing["sha256"] is None
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    junk = check.verify_prompt(tmp_path / "junk.npz", FRANKA)
    assert junk["ok"] is False and "cannot read prompt" in junk["problems"][0]


#: A cross-scene unit: demonstrated on one of 7 and 8, scored on one of 5 and 6, at level 1.
CROSS = check.Unit(
    task="click_bell",
    embodiment="franka-panda",
    task_config="level1",
    save_freq=15,
    seed_candidates=(7, 8),
    eval_seed_candidates=(5, 6),
    level=1,
)


def test_a_cross_scene_unit_is_scored_on_its_eval_candidates():
    assert CROSS.cross_scene and CROSS.scored_on == [5, 6]
    assert CROSS.scene_config()["evaluation_setting"] == "cross_scene_level"
    assert not FRANKA.cross_scene and FRANKA.scored_on == CANDIDATES
    assert FRANKA.scene_config()["evaluation_setting"] == "same_scene"


def test_a_cross_scene_prompt_is_held_to_its_demonstration_scene():
    good = {
        "level": 1,
        "demonstration_scene": {
            "seed": 8,
            "sha256": "ab",
            "expert": {
                "scene_seeds": [7, 8],
                "attempts": [
                    {"seed": 7, "rejection": "unstable", "detail": ""},
                    {"seed": 8, "rejection": None, "detail": ""},
                ],
                "rejections": {"unstable": 1},
            },
        },
    }
    assert check._demonstration_problems(good, CROSS, 5) == []
    wrong_seed = {**good, "demonstration_scene": {**good["demonstration_scene"], "seed": 9}}
    assert (
        "not one of the unit's seed candidates"
        in check._demonstration_problems(wrong_seed, CROSS, 5)[0]
    )
    assert check._demonstration_problems({**good, "level": 2}, CROSS, 5)
    assert "recorded in the scene the unit is scored in" in " ".join(
        check._demonstration_problems(good, CROSS, 8)
    )
    assert check._demonstration_problems({"level": 1}, CROSS, 5) == [
        "a cross-scene prompt names no demonstration scene"
    ]
