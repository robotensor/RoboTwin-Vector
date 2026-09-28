import json
import os
from pathlib import Path

import pytest

import robotensor_bench
from robotensor_bench import CLI_API, cli, tasks

#: One unit's flags, as the competition builds them: the fake simulator's robot has 14 joints.
UNIT = ["--task", "click_bell", "--task-config", "demo_clean", "--embodiment", "aloha-agilex"]
MAKE = ["demo", "make", *UNIT, "--seed-candidates", "11"]
VERIFY = ["demo", "verify", *UNIT, "--seed-candidates", "11"]


def listed(out: str) -> list[str]:
    return [line.split()[0] for line in out.splitlines() if line.startswith("  ")]


def test_tasks_list_prints_categories_and_arms(capsys):
    assert cli.main(["tasks", "list"]) == 0
    out = capsys.readouterr().out
    assert "Pick and Place (pick_and_place)" in out
    assert "click_bell  (one arm)\n" in out
    assert "stack_bowls_two  (switching arms)\n" in out
    assert "lift_pot  (two arms)" in out
    assert listed(out) == list(tasks.table().tasks)


def test_tasks_list_with_arms_1_lists_exactly_the_one_arm_tasks(capsys):
    assert cli.main(["tasks", "list", "--arms", "1"]) == 0
    out = capsys.readouterr().out
    one_arm = [task.name for task in tasks.table().tasks.values() if task.arms == "1"]
    assert listed(out) == one_arm and len(one_arm) == 26
    assert "Bimanual Manipulation" not in out  # no one-arm task there, so no empty heading


def test_tasks_list_json_is_the_table_the_competition_deals_from(capsys):
    assert cli.main(["tasks", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    table = tasks.table()
    assert listing["categories"] == dict(table.categories)
    assert listing["tasks"] == {
        name: {"category": task.category, "arms": task.arms} for name, task in table.tasks.items()
    }
    assert cli.main(["tasks", "list", "--json", "--category", "press_push"]) == 0
    pressed = json.loads(capsys.readouterr().out)["tasks"]
    assert pressed and {entry["category"] for entry in pressed.values()} == {"press_push"}
    assert cli.main(["tasks", "list", "--category", "juggling"]) == 2
    assert "unknown category 'juggling'" in capsys.readouterr().err


def test_arms_accepts_only_1_or_2():
    with pytest.raises(SystemExit):
        cli.main(["tasks", "list", "--arms", "switching"])


@pytest.fixture
def fake_sim(monkeypatch):
    """Route demo make and eval run to fake envs; `envs` is what a test wants load_task to give."""
    from fake_robotwin import FakeConfig, FakeTaskEnv, FakeUnstable
    from robotensor_bench import robotwin

    envs = {"next": lambda name: FakeTaskEnv()}
    monkeypatch.setattr(robotwin, "unstable_error", lambda: FakeUnstable)
    monkeypatch.setattr(robotwin, "SceneConfig", FakeConfig)
    monkeypatch.setattr(robotwin, "load_task", lambda name: envs["next"](name))
    return envs


def test_demo_make_then_eval_run_through_the_cli(tmp_path, fake_sim, capsys):
    pytest.importorskip("imageio_ffmpeg")
    out, run = tmp_path / "prompt", tmp_path / "run"
    assert cli.main([*MAKE, "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["scene_seed"] == 11
    assert {p.name for p in out.iterdir()} == {"prompt.npz", "demonstration.mp4", "result.json"}

    assert (
        cli.main(
            ["eval", "run", "--demo", str(out), "--stub-policy", "replay"] + ["--out", str(run)]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["success"] is True and printed["void"] is False
    assert json.loads((run / "result.json").read_text()) == printed
    assert (run / "evaluation.mp4").is_file()


def test_a_rejected_seed_exits_0_with_its_result(tmp_path, fake_sim, capsys):
    # The competition reads result.json whatever the exit; a non-zero exit for an outcome it
    # expects would have it void the unit on the log tail and lose the rejection's reason.
    from fake_robotwin import FakeTaskEnv

    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={11})
    assert cli.main([*MAKE, "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["rejection"] == "unstable"
    assert json.loads((tmp_path / "result.json").read_text())["ok"] is False
    assert not (tmp_path / "prompt.npz").exists()


def test_demo_make_takes_every_candidate_seed_in_order(tmp_path, fake_sim, capsys):
    from fake_robotwin import FakeTaskEnv

    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={11, 12})
    argv = ["demo", "make", *UNIT, "--seed-candidates", "11,12,13", "--out", str(tmp_path)]
    assert cli.build_parser().parse_args(argv).seed_candidates == [11, 12, 13]
    assert cli.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["scene_seed"] == 13
    assert [a["seed"] for a in printed["attempts"]] == [11, 12, 13]

    # Every candidate rejected is still a result, and still exit 0.
    fake_sim["next"] = lambda name: FakeTaskEnv(unstable_seeds={11, 12, 13})
    assert cli.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["void"] is True and printed["void_cause"] == "harness"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


@pytest.mark.parametrize("seeds", ["11,11", "", "x", "-1", str(2**32)])
def test_seed_candidates_are_distinct_seeds(seeds, capsys):
    with pytest.raises(SystemExit) as refused:
        cli.main(["demo", "make", *UNIT, "--seed-candidates", seeds, "--out", "x"])
    assert refused.value.code == 2 and "--seed-candidates" in capsys.readouterr().err


def test_a_command_built_for_other_harness_source_refuses_to_run(tmp_path, fake_sim, capsys):
    # The competition builds its command lines for the source `config check` reported; a checkout
    # changed since may read the same flags differently.
    other = "0" * 64
    out = tmp_path / "p"
    assert cli.main([*MAKE, "--out", str(out), "--expect-source-sha256", other]) == 2
    err = capsys.readouterr().err
    assert f"digests to {robotensor_bench.source_sha256()}, not the {other}" in err
    assert not out.exists()

    argv = ["eval", "run", "--demo", str(out), "--stub-policy", "replay"]
    assert cli.main([*argv, "--out", str(tmp_path / "r"), "--expect-source-sha256", other]) == 2
    assert "other robotensor_bench code" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()

    # Its own digest runs, and the result says which source wrote it.
    mine = ["--expect-source-sha256", robotensor_bench.source_sha256()]
    assert cli.main([*MAKE, "--out", str(out), *mine]) == 0
    assert json.loads(capsys.readouterr().out)["source_sha256"] == robotensor_bench.source_sha256()
    assert cli.main([*argv, "--out", str(tmp_path / "r"), *mine]) == 0
    assert json.loads(capsys.readouterr().out)["source_sha256"] == robotensor_bench.source_sha256()


def test_the_denoiser_is_a_flag_that_reaches_the_simulator_seam(tmp_path, fake_sim, monkeypatch):
    # The competition hands a harness subprocess only allow-listed variables, so the override for
    # a denoiser that hangs camera reads travels as an argument.
    from fake_robotwin import FakeTaskEnv

    monkeypatch.setenv("ROBOTWIN_BENCH_DENOISER", "")
    seen = []

    def load_task(name):
        seen.append(os.environ.get("ROBOTWIN_BENCH_DENOISER"))
        return FakeTaskEnv()

    fake_sim["next"] = load_task
    assert cli.main([*MAKE, "--out", str(tmp_path / "p"), "--denoiser", "none"]) == 0
    argv = ["eval", "run", "--demo", str(tmp_path / "p"), "--stub-policy", "replay"]
    assert cli.main([*argv, "--out", str(tmp_path / "r"), "--denoiser", "oidn"]) == 0
    assert seen == ["none", "oidn"]
    with pytest.raises(SystemExit) as refused:
        cli.main([*MAKE, "--out", str(tmp_path / "q"), "--denoiser", "fast"])
    assert refused.value.code == 2


def test_a_task_the_harness_does_not_have_is_refused_and_leaves_no_earlier_prompt(
    tmp_path, fake_sim, capsys
):
    _stale(tmp_path, "prompt.npz", "demonstration.mp4", "result.json")
    argv = ["demo", "make", "--task", "nope", *UNIT[2:], "--seed-candidates", "1"]
    assert cli.main([*argv, "--out", str(tmp_path)]) == 2
    assert "unknown task 'nope'" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_eval_run_exits_0_on_a_failed_policy_and_3_on_a_harness_error(tmp_path, fake_sim, capsys):
    from robotensor_bench import robotwin

    assert cli.main([*MAKE, "--out", str(tmp_path / "p")]) == 0
    argv = ["eval", "run", "--demo", str(tmp_path / "p"), "--stub-policy", "zero"]
    assert cli.main([*argv, "--out", str(tmp_path / "r")]) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "r" / "result.json").read_text())["success"] is False

    def broken(name):
        raise robotwin.RoboTwinError("no RoboTwin checkout")

    fake_sim["next"] = broken
    assert cli.main([*argv, "--out", str(tmp_path / "r2")]) == 3
    assert "the harness is unavailable: no RoboTwin checkout" in capsys.readouterr().err


def test_a_harness_fault_mid_unit_is_a_void_result_and_a_logged_traceback(
    tmp_path, fake_sim, monkeypatch, capsys
):
    # The competition reads result.json; a command that died would leave it a log tail to read.
    from robotensor_bench import robotwin

    assert cli.main([*MAKE, "--out", str(tmp_path / "p")]) == 0
    capsys.readouterr()

    def broken(env):
        raise KeyError("head_camera")

    monkeypatch.setattr(robotwin, "fingerprint", broken)
    argv = ["eval", "run", "--demo", str(tmp_path / "p"), "--stub-policy", "replay"]
    assert cli.main([*argv, "--out", str(tmp_path / "r")]) == 0
    printed, err = capsys.readouterr()
    result = json.loads(printed)
    assert result["void"] is True and "KeyError: 'head_camera'" in result["error"]
    assert json.loads((tmp_path / "r" / "result.json").read_text()) == result
    assert "Traceback" in err and "KeyError: 'head_camera'" in err


def test_demo_make_records_the_robot_it_was_given(tmp_path, fake_sim, capsys):
    from robotensor_bench import prompt

    assert cli.main([*MAKE, "--out", str(tmp_path)]) == 0
    _, meta = prompt.read_raw(tmp_path / "prompt.npz")
    assert meta["embodiment"]["name"] == meta["embodiment"]["choice"] == "aloha-agilex"
    assert json.loads(capsys.readouterr().out)["embodiment"] == "aloha-agilex"
    # A unit names its robot: without one there is no unit to hold a prompt to.
    with pytest.raises(SystemExit):
        cli.main(
            ["demo", "make", "--task", "click_bell", "--task-config", "demo_clean"]
            + ["--seed-candidates", "1", "--out", "x"]
        )


def _stale(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text('{"success": true, "void": false, "ok": true}')


def test_a_served_policy_without_its_key_is_refused_and_leaves_no_earlier_result(
    tmp_path, fake_sim, monkeypatch, capsys
):
    # A key that is not there is the caller's configuration, not the unit's outcome: exit 2, and
    # no result.json the competition could take for one, not even an earlier run's.
    monkeypatch.delenv("NO_SUCH_POLICY_KEY", raising=False)
    assert cli.main([*MAKE, "--out", str(tmp_path / "p")]) == 0
    capsys.readouterr()
    out = tmp_path / "r"
    _stale(out, "result.json", "evaluation.mp4")
    argv = ["eval", "run", "--demo", str(tmp_path / "p"), "--out", str(out)]
    argv += ["--policy", "/nonexistent/policy.sock", "--authkey-env", "NO_SUCH_POLICY_KEY"]
    assert cli.main(argv) == 2
    assert "NO_SUCH_POLICY_KEY: the environment variable holds no key" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == []


def test_eval_run_refuses_to_write_over_the_demonstrations_own_result(tmp_path, fake_sim, capsys):
    out = tmp_path / "p"
    assert cli.main([*MAKE, "--out", str(out)]) == 0
    capsys.readouterr()
    before = (out / "result.json").read_text()
    argv = ["eval", "run", "--demo", str(out), "--stub-policy", "replay"]
    assert cli.main([*argv, "--out", str(out)]) == 2
    assert "holds the prompt" in capsys.readouterr().err
    assert (out / "result.json").read_text() == before
    assert sorted(p.name for p in out.iterdir()) == [
        "demonstration.mp4",
        "prompt.npz",
        "result.json",
    ]


EVAL = ["eval", "run", "--demo", "p"]


def test_eval_run_takes_a_served_policy_or_a_stub_never_both(tmp_path, capsys):
    base = [*EVAL, "--out", str(tmp_path / "r")]
    with pytest.raises(SystemExit) as refused:
        cli.main([*base, "--stub-policy", "replay", "--policy", "/tmp/policy.sock"])
    assert refused.value.code == 2 and "not allowed with argument" in capsys.readouterr().err
    with pytest.raises(SystemExit) as refused:
        cli.main(base)
    assert refused.value.code == 2 and "--policy --stub-policy" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--act-timeout-s", "5", "--policy-log", "x.log"],
        ["--policy-budget-s", "5"],
        ["--unit-timeout-s", "600"],
    ],
)
def test_a_stub_refuses_the_flags_of_a_served_policy(tmp_path, capsys, flags):
    assert cli.main([*EVAL, "--out", str(tmp_path / "r"), "--stub-policy", "replay", *flags]) == 2
    assert f"{flags[0]}" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()  # refused before anything was cleared or written


@pytest.mark.parametrize("flag", ["--act-timeout-s", "--policy-budget-s", "--unit-timeout-s"])
@pytest.mark.parametrize("seconds", ["0", "-1", "inf", "soon"])
def test_a_served_policys_limits_are_positive_numbers_of_seconds(flag, seconds, capsys):
    flags = ["--policy", "/tmp/p.sock", flag, seconds]
    with pytest.raises(SystemExit) as refused:
        cli.main([*EVAL, "--out", "r", *flags])
    assert refused.value.code == 2 and "positive number of seconds" in capsys.readouterr().err


# -- demo verify ------------------------------------------------------------------------------


def test_demo_verify_holds_a_made_demonstration_to_its_unit(tmp_path, fake_sim, capsys):
    assert cli.main([*MAKE, "--out", str(tmp_path)]) == 0
    made = json.loads(capsys.readouterr().out)
    assert cli.main([*VERIFY, "--demo", str(tmp_path)]) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["ok"] is True and verdict["problems"] == []
    assert verdict["sha256"] == made["prompt_sha256"] and verdict["scene_seed"] == 11

    # Another unit's flags: a prompt of this task and robot is still not that unit's.
    other = ["demo", "verify", *UNIT, "--seed-candidates", "12,13", "--demo", str(tmp_path)]
    assert cli.main(other) == 4
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["ok"] is False
    assert any("not one of the unit's candidates [12, 13]" in p for p in verdict["problems"])


def test_demo_verify_of_nothing_exits_4_and_needs_no_simulator(tmp_path, capsys):
    assert cli.main([*VERIFY, "--demo", str(tmp_path)]) == 4
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["ok"] is False and verdict["sha256"] is None


def test_demo_verify_reads_a_cross_scene_units_level_from_its_task_config(tmp_path, capsys):
    flags = ["--task", "click_bell", "--task-config", "level1", "--embodiment", "franka-panda"]
    argv = ["demo", "verify", *flags, "--seed-candidates", "7,8", "--eval-seed-candidates", "5,6"]
    assert cli.main([*argv, "--demo", str(tmp_path)]) == 4
    capsys.readouterr()
    assert cli._unit(cli.build_parser().parse_args([*argv, "--demo", "x"])).level == 1
    shared = ["demo", "verify", *flags, "--seed-candidates", "5,8", "--eval-seed-candidates", "5,6"]
    assert cli.main([*shared, "--demo", str(tmp_path)]) == 2
    assert "share a seed" in capsys.readouterr().err
    unknown = ["demo", "verify", *flags[:2], "--task-config", "nope", *flags[4:]]
    assert (
        cli.main(
            [*unknown, "--seed-candidates", "7", "--eval-seed-candidates", "5"] + ["--demo", "x"]
        )
        == 2
    )
    assert "no task config 'nope'" in capsys.readouterr().err


# -- config check -----------------------------------------------------------------------------


def test_config_check_prints_the_config_and_what_the_harness_is(fake_sim, capsys):
    assert (
        cli.main(["config", "check", "--task-config", "level1", "--embodiment", "franka-panda"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["task_config"] == "level1" and report["level"] == 1
    assert report["cameras"] == ["head_camera", "left_camera", "right_camera"]
    harness = report["harness"]
    assert harness["cli"] == "robotensor_bench.cli" and harness["api"] == CLI_API
    assert harness["source_sha256"] == robotensor_bench.source_sha256()
    assert harness["limits"]["result_reserve_s"] == 30.0
    assert harness["embodiments"]["franka-panda"]["action_dims"] == {"qpos": 16, "ee": 16}
    assert cli.main(["config", "check", "--task-config", "nope"]) == 2
    assert "no task config 'nope'" in capsys.readouterr().err


# -- selftest ---------------------------------------------------------------------------------


def test_selftest_makes_verifies_and_replays_one_unit(tmp_path, fake_sim, capsys):
    pytest.importorskip("imageio_ffmpeg")
    argv = ["selftest", "--embodiment", "aloha-agilex", "--seed", "11", "--out", str(tmp_path)]
    assert cli.main(argv) == 0
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["success"] is True and last["task"] == "click_bell"
    assert (tmp_path / "unit" / "prompt.npz").is_file()
    assert (tmp_path / "result" / "result.json").is_file()


def test_selftest_fails_when_its_replay_does_not_succeed(tmp_path, fake_sim, monkeypatch, capsys):
    pytest.importorskip("imageio_ffmpeg")
    from robotensor_bench import policy

    monkeypatch.setattr(policy.ReplayPolicy, "_act", policy.DummyPolicy._act)
    argv = ["selftest", "--embodiment", "aloha-agilex", "--seed", "11", "--out", str(tmp_path)]
    assert cli.main(argv) == 1
    assert "did not end success" in capsys.readouterr().err


def test_selftest_writes_under_the_harness_by_default():
    out = Path(cli.build_parser().parse_args(["selftest"]).out)
    assert out == Path(robotensor_bench.__file__).parent / "runs" / "selftest"
