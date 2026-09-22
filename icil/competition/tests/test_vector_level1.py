"""The Vector competition's suite: level-1 cross-scene units on two Franka arms."""

import json

import pytest

from icil_benchmark_robotwin import BENCHMARK, catalogue, commands, units
from robotwin_icil.records import CROSS_SCENE_LEVEL

SUITE = catalogue.VECTOR_LEVEL1


def test_the_suite_is_the_level_benchmarks_sixteen_tasks_on_two_frankas():
    assert len(catalogue.SUITES[SUITE]) == 16
    profile = catalogue.profile(SUITE)
    assert profile.embodiment == "franka-panda" and profile.task_config == "level1"
    assert profile.evaluation_setting == CROSS_SCENE_LEVEL and profile.cross_scene
    assert profile.legal_joints and profile.level == 1
    info = BENCHMARK.info()
    assert info["suite_profiles"][SUITE]["cross_scene"] is True
    assert info["embodiment_of_suite"][SUITE] == "franka-panda"


def test_units_carry_disjoint_demo_and_scene_seeds_outside_every_public_range():
    drawn = units.derive_units(seed_material="duel|block|hash|skill", count=48, suite=SUITE)
    assert len(drawn) == 48
    counts = {}
    for unit in drawn:
        params = unit["instance_params"]
        scene, demo = params["scene_seeds"], params["demo_seeds"]
        assert len(scene) == len(demo) == units.SCENE_SEED_CANDIDATES
        assert not set(scene) & set(demo)
        # Never a seed the level benchmark's train [0, 1M) or eval [1M, 2M) split published.
        assert all(catalogue.PUBLIC_SEEDS_END <= s < catalogue.SEED_BOUND for s in scene + demo)
        assert params["embodiment"] == "franka-panda" and params["scene_seed"] is None
        assert params["task_config"] == "level1" and params["level"] == 1
        counts[unit["task"]] = counts.get(unit["task"], 0) + 1
    # 48 units over 16 tasks: three each.
    assert set(counts.values()) == {3} and len(counts) == 16


def test_units_are_a_pure_function_of_the_seed_material_and_prefix_stable():
    a = units.derive_units(seed_material="m", count=16, suite=SUITE)
    b = units.derive_units(seed_material="m", count=160, suite=SUITE)
    assert json.dumps(a) == json.dumps(b[:16])
    assert a != units.derive_units(seed_material="m2", count=16, suite=SUITE)


def test_the_suite_draws_under_its_own_label_so_franka_units_do_not_move():
    franka = units.derive_units(seed_material="m", count=4, suite=catalogue.FRANKA_1ARM)
    assert "demo_seeds" not in franka[0]["instance_params"]
    assert units.DERIVATION == "robotwin-icil-competition/units/2"
    assert catalogue.profile(SUITE).derivation != units.DERIVATION


def test_materialize_records_on_demo_seeds_and_scores_on_scene_seeds(tmp_path):
    unit = units.derive_units(seed_material="m", count=1, suite=SUITE)[0]
    argv = commands.materialize_command(unit=unit, out_dir=str(tmp_path))
    params = unit["instance_params"]
    scene_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "--scene-seed"]
    eval_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "--eval-seed"]
    assert scene_flags == [str(s) for s in params["demo_seeds"]]
    assert eval_flags == [str(s) for s in params["scene_seeds"]]
    assert "--legal-joints" in argv
    assert argv[argv.index("--task-config") + 1] == "level1"
    assert argv[argv.index("--embodiment") + 1] == "franka-panda"


def test_a_unit_whose_demo_and_scene_seeds_overlap_is_refused():
    unit = units.derive_units(seed_material="m", count=1, suite=SUITE)[0]
    unit["instance_params"]["demo_seeds"][0] = unit["instance_params"]["scene_seeds"][0]
    with pytest.raises(ValueError, match="share a seed"):
        commands.materialize_command(unit=unit, out_dir="/tmp/x")


def test_a_cross_scene_prompt_is_held_to_its_demonstration_scene():
    from icil_benchmark_robotwin import prompts

    params = {"scene_seeds": [5, 6], "demo_seeds": [7, 8]}
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
    profile = catalogue.profile(SUITE)
    assert prompts._demonstration_problems(good, params, 5, profile) == []
    wrong_seed = {**good, "demonstration_scene": {**good["demonstration_scene"], "seed": 9}}
    assert (
        "not one of the unit's demo_seeds"
        in prompts._demonstration_problems(wrong_seed, params, 5, profile)[0]
    )
    assert prompts._demonstration_problems({**good, "level": 2}, params, 5, profile)
    assert prompts._demonstration_problems({"level": 1}, params, 5, profile) == [
        "a cross-scene prompt names no demonstration scene"
    ]
