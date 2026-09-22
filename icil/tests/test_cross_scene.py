"""A cross-scene unit: the demonstration is recorded in one scene and the unit is scored in another."""

import pytest

from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
from robotwin_icil import prompt, robotwin, scene, unit
from robotwin_icil.policy import ReplayPolicy
from robotwin_icil.records import CROSS_SCENE_LEVEL, SAME_SCENE

pytest.importorskip("imageio_ffmpeg")

TASK = "click_bell"


@pytest.fixture(autouse=True)
def _fake_sim(monkeypatch):
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)


def test_the_demonstration_comes_from_one_scene_and_the_unit_is_scored_in_another(tmp_path):
    env = FakeTaskEnv(unstable_seeds={3}, expert_misses_on={21})
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, [3, 8], FakeConfig(), out, eval_seeds=[21, 34], task_env=env)
    assert done.ok
    # The demonstration's candidates first, then the evaluation's; each list stops at its first
    # success, so neither 13 nor a third eval seed is built.
    assert env.setups == [3, 8, 21, 34]

    result = unit.read_result(out)
    assert result["evaluation_setting"] == CROSS_SCENE_LEVEL
    assert result["demo_seed"] == 8 and result["scene_seed"] == 34
    assert result["eval_seeds"] == [21, 34]
    assert [a["seed"] for a in result["eval_attempts"]] == [21, 34]
    assert result["eval_rejections"] == {"expert_failed": 1}
    assert result["demo_scene_sha256"] != result["scene_sha256"]

    demonstration, meta = prompt.read_prompt(out / "prompt.npz")
    assert meta["evaluation_setting"] == CROSS_SCENE_LEVEL
    assert meta["scene_seed"] == 34 and meta["demonstration_scene"]["seed"] == 8
    assert meta["expert"]["scene_seeds"] == [21, 34]
    assert meta["demonstration_scene"]["expert"]["scene_seeds"] == [3, 8]
    # The scene meta names is the evaluation scene, rebuilt here from its seed alone.
    rebuilt = FakeTaskEnv()
    rebuilt.setup_demo(seed=34)
    assert scene.digest(robotwin.fingerprint(rebuilt)) == meta["scene"]["sha256"]

    # run-unit rebuilds the evaluation scene and hands the policy the other scene's demonstration.
    run_env = FakeTaskEnv()
    run = unit.run_unit(out / "prompt.npz", ReplayPolicy(), tmp_path / "run", task_env=run_env)
    assert run_env.setups == [34]
    assert run["void"] is False and run["scene_seed"] == 34 and run["demo_seed"] == 8
    assert run["evaluation_setting"] == CROSS_SCENE_LEVEL
    # Replaying the demonstration's joints does not reach another scene's target.
    assert run["success"] is False


def test_the_evaluation_scene_is_verified_without_rendering_a_camera(tmp_path):
    env = FakeTaskEnv()
    unit.materialize(TASK, [8], FakeConfig(), tmp_path / "p", eval_seeds=[34], task_env=env)
    demo_only = FakeTaskEnv()
    unit.materialize(TASK, [8], FakeConfig(), tmp_path / "q", task_env=demo_only)
    # Verifying the evaluation scene added builds, never a camera read.
    assert env.get_obs_calls == demo_only.get_obs_calls


def test_a_same_scene_unit_records_its_setting_and_no_demonstration_scene(tmp_path):
    out = tmp_path / "prompt"
    unit.materialize(TASK, [8], FakeConfig(), out, task_env=FakeTaskEnv())
    _, meta = prompt.read_raw(out / "prompt.npz")
    assert meta["evaluation_setting"] == SAME_SCENE and meta["demonstration_scene"] is None
    result = unit.read_result(out)
    assert result["evaluation_setting"] == SAME_SCENE and result["demo_seed"] == 8
    assert result["scene_seed"] == 8 and "eval_seeds" not in result


def test_a_seed_in_both_lists_is_refused_before_any_scene(tmp_path):
    env = FakeTaskEnv()
    with pytest.raises(unit.UnitError, match="both a demonstration and an evaluation candidate"):
        unit.materialize(TASK, [3, 8], FakeConfig(), tmp_path / "p", eval_seeds=[8], task_env=env)
    assert env.setups == []


def test_every_evaluation_candidate_rejected_is_a_void_harness_result(tmp_path):
    env = FakeTaskEnv(unstable_seeds={21}, plan_fails_on={34})
    out = tmp_path / "prompt"
    done = unit.materialize(TASK, [8], FakeConfig(), out, eval_seeds=[21, 34], task_env=env)
    assert not done.ok
    assert {p.name for p in out.iterdir()} == {"result.json"}
    result = unit.read_result(out)
    assert result["void"] is True and result["void_cause"] == "harness"
    assert result["success"] is None and result["scene_seed"] is None and result["demo_seed"] == 8
    assert result["error"] == (
        "expert rejected all 2 evaluation candidate seeds: "
        "seed 21: unstable: objects unstable in seed 21; seed 34: plan_failed"
    )


def test_a_rejected_demonstration_never_builds_an_evaluation_scene(tmp_path):
    env = FakeTaskEnv(expert_misses_on={8})
    done = unit.materialize(TASK, [8], FakeConfig(), tmp_path / "p", eval_seeds=[34], task_env=env)
    assert not done.ok and env.setups == [8]
    result = unit.read_result(tmp_path / "p")
    assert result["error"] == "expert rejected the demonstration seed: expert_failed"
    assert result["eval_attempts"] == []


def test_legal_joints_rejects_an_expert_that_planned_past_a_joint_limit(tmp_path):
    env = FakeTaskEnv()
    env.planned_joints_legal = lambda: (env.seed != 8, 0.25 if env.seed == 8 else 0.0)
    done = unit.materialize(
        TASK,
        [8, 13],
        FakeConfig(),
        tmp_path / "p",
        eval_seeds=[34],
        legal_joints=True,
        task_env=env,
    )
    assert done.ok
    result = unit.read_result(tmp_path / "p")
    assert result["demo_seed"] == 13
    assert result["attempts"][0] == {
        "seed": 8,
        "rejection": "illegal_joints",
        "detail": "planned joints exceed limits by 0.250 rad",
    }


def test_the_cli_takes_eval_seeds_and_legal_joints():
    from robotwin_icil import cli

    args = cli.build_parser().parse_args(
        [
            "materialize",
            "--task",
            TASK,
            "--scene-seed",
            "8",
            "--eval-seed",
            "21",
            "--eval-seed",
            "34",
            "--legal-joints",
            "--task-config",
            "level1",
            "--out",
            "x",
        ]
    )
    assert args.eval_seeds == [21, 34] and args.legal_joints is True
    assert args.task_config == "level1"
    plain = cli.build_parser().parse_args(
        ["materialize", "--task", TASK, "--scene-seed", "8", "--out", "x"]
    )
    assert plain.eval_seeds is None and plain.legal_joints is False
