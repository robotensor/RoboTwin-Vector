"""`rt-bench`: make a demonstration, verify it, and evaluate a served policy against it.

    rt-bench demo make --task place_container_plate --task-config level1 \
        --embodiment franka-panda --seed-candidates 2000017,2000123,2000456,2000789 \
        --eval-seed-candidates 2001001,2001002,2001003,2001004 --legal-joints --out pool/vl-000
    rt-bench demo verify --demo pool/vl-000 (the same unit flags)
    rt-bench eval run --demo pool/vl-000 --policy /tmp/policy.sock \
        --authkey-env VECTOR_AUTHKEY --out results/challenger/vl-000
    rt-bench eval run --demo pool/vl-000 --stub-policy replay --out runs/replay
    rt-bench tasks list --json
    rt-bench config check --task-config level1 --embodiment franka-panda
    rt-bench selftest

Run from the repository root, never installed: `python -m robotensor_bench.cli ...`.

Exit codes:

- 0: a result was written; a failure and a void are results. `demo make` whose every candidate
  the expert rejected writes a void `result.json` and exits 0; `demo verify` exits 0 when the
  prompt is the unit's;
- 1: `selftest` only - it ran to the end and its own replay did not end `success`;
- 2: input this harness will not take: an argument, a task, a task config, a command line built
  for other source (`--expect-source-sha256`);
- 3: the harness is unavailable - the simulator, its assets, the GPU - before a result was
  written;
- 4: `demo verify` found a prompt it cannot read or that is not the one the unit asked for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import CLI_API, __version__, source_sha256
from . import tasks as tasks_
from .arms import LABELS, ONE, TWO
from .demo import DemonstrationError
from .policy import PolicyError
from .prompt import PROMPT_FILE, PromptError
from .records import RecordError
from .remote import ACT_TIMEOUT_S, POLICY_BUDGET_S, RESULT_RESERVE_S
from .robotwin import BENCH_ROOT, DENOISER_ENV, DENOISERS, EMBODIMENTS, RoboTwinError
from .unit import UnitError

EXIT_OK = 0
#: `selftest` alone: it ran end to end and its own replay did not end `success`.
EXIT_GATE = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_PROMPT = 4

PROG = "rt-bench"
#: The variable `eval run --policy` reads the served policy's key from unless told another.
AUTHKEY_ENV = "VECTOR_AUTHKEY"
#: `--stub-policy`: a policy run here instead of a served one. `zero` holds the robot where it is,
#: `replay` plays the demonstration's actions back and, in the Same Scene setting, must succeed.
STUBS = ("zero", "replay")


class UsageError(ValueError):
    """A command line this harness will not take."""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = getattr(args, "expect_source_sha256", None)
    if expected is not None and expected != source_sha256():
        # Before anything is cleared or written: the command line was built for other code, and
        # its flags may not mean here what its builder meant.
        return _fail(
            f"this harness's source digests to {source_sha256()}, not the {expected} the command "
            "was built for: the checkout holds other robotensor_bench code",
            EXIT_USAGE,
        )
    if getattr(args, "denoiser", None):
        # Read by the RoboTwin seam when it sets a scene up, which is after this.
        os.environ[DENOISER_ENV] = args.denoiser
    try:
        return args.run(args)
    except KeyboardInterrupt:
        return EXIT_UNAVAILABLE
    except (UsageError, tasks_.TaskTableError, PolicyError, UnitError) as exc:
        return _fail(str(exc), EXIT_USAGE)
    except PromptError as exc:
        return _fail(str(exc), EXIT_PROMPT)
    except (RoboTwinError, RecordError, DemonstrationError, ImportError) as exc:
        return _fail(f"the harness is unavailable: {exc}", EXIT_UNAVAILABLE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="group", required=True)

    demo = sub.add_parser("demo", help="make and verify demonstrations").add_subparsers(
        dest="verb", required=True
    )
    make = demo.add_parser(
        "make",
        help="one unit: candidate seeds in, prompt.npz, demonstration.mp4 and result.json out",
    )
    _add_unit(make)
    make.add_argument(
        "--legal-joints",
        action="store_true",
        help="also reject an expert run whose planned arm joints leave their legal range, as "
        "RoboTwin's collector does",
    )
    make.add_argument("--out", required=True, help="directory the three files are written into")
    _add_denoiser(make)
    _add_expect_source(make)
    make.set_defaults(run=_demo_make)

    verify = demo.add_parser(
        "verify", help="check a made demonstration is the one its unit asked for; no simulator"
    )
    verify.add_argument("--demo", required=True, help="the directory `demo make` wrote into")
    _add_unit(verify)
    _add_expect_source(verify)
    verify.set_defaults(run=_demo_verify)

    evaluate = sub.add_parser("eval", help="evaluate a policy").add_subparsers(
        dest="verb", required=True
    )
    run = evaluate.add_parser(
        "run",
        help="one demonstration against one policy: rebuilds and verifies the scene, writes "
        "result.json and evaluation.mp4",
    )
    run.add_argument("--demo", required=True, help="the directory `demo make` wrote into")
    run.add_argument("--out", required=True, help="directory the result and clip are written into")
    served = run.add_mutually_exclusive_group(required=True)
    served.add_argument(
        "--policy",
        metavar="ADDR",
        help=(
            "a policy served by `python -m vector_protocol.serve`: a Unix socket path or host:port"
        ),
    )
    served.add_argument(
        "--stub-policy",
        choices=STUBS,
        help="a model-free stub run here instead: zero holds the robot, replay plays the "
        "demonstration's actions back",
    )
    run.add_argument(
        "--authkey-env",
        default=AUTHKEY_ENV,
        metavar="NAME",
        help="the environment variable holding the served policy's key as hex, at least 16 "
        f"bytes; the key itself is never an argument (default {AUTHKEY_ENV})",
    )
    run.add_argument(
        "--act-timeout-s",
        type=_positive_seconds,
        metavar="S",
        help=f"how long one act of the served policy may take (default {ACT_TIMEOUT_S:g})",
    )
    run.add_argument(
        "--policy-budget-s",
        type=_positive_seconds,
        metavar="S",
        help="how long all the calls to the served policy may take together in the unit; the "
        f"call it runs out in voids the unit on the policy (default {POLICY_BUDGET_S:g})",
    )
    run.add_argument(
        "--unit-timeout-s",
        type=_positive_seconds,
        metavar="S",
        help="how long eval run has before whoever started it kills it: no call to the served "
        f"policy runs within {RESULT_RESERVE_S:g}s of that, and one cut short there while the "
        "policy is within its budget voids the unit on the harness",
    )
    run.add_argument(
        "--policy-log",
        metavar="PATH",
        help="the served policy's log file, whose tail ends the error of a unit it voided",
    )
    run.add_argument(
        "--step-ratio",
        type=_positive_ratio,
        metavar="R",
        help="stop the rollout after R times the expert's steps in the scored scene, or at the "
        "task's own step limit if that comes first (default: the task's limit alone)",
    )
    _add_denoiser(run)
    _add_expect_source(run)
    run.set_defaults(run=_eval_run)

    tasks = sub.add_parser("tasks", help="the task table").add_subparsers(
        dest="verb", required=True
    )
    listing = tasks.add_parser("list", help="every task, its skill category and its arms")
    listing.add_argument("--json", action="store_true")
    listing.add_argument("--category")
    listing.add_argument(
        "--arms",
        choices=[ONE, TWO],
        default=TWO,
        help="1 selects only tasks whose expert uses one arm; 2 (the default) changes nothing",
    )
    listing.set_defaults(run=_tasks_list)

    config = sub.add_parser("config", help="task configs").add_subparsers(
        dest="verb", required=True
    )
    check = config.add_parser(
        "check",
        help="load a task config as a unit would, and print it with what this harness is",
    )
    check.add_argument("--task-config", required=True)
    _add_embodiment(check, required=False)
    check.set_defaults(run=_config_check)

    selftest = sub.add_parser(
        "selftest", help="one task: demo make, demo verify, then a replay that must succeed"
    )
    selftest.add_argument("--task", default="click_bell")
    selftest.add_argument("--task-config", default="demo_clean")
    selftest.add_argument("--embodiment", choices=sorted(EMBODIMENTS), default="franka-panda")
    selftest.add_argument("--seed", type=int, default=0, help="the first of four candidates")
    selftest.add_argument("--out", default=str(BENCH_ROOT / "runs" / "selftest"))
    _add_denoiser(selftest)
    selftest.set_defaults(run=_selftest)
    return parser


# -- commands ---------------------------------------------------------------------------------


def _demo_make(args: argparse.Namespace) -> int:
    from . import robotwin
    from .unit import MATERIALIZE_OUTPUTS, clear_outputs, materialize

    # First, so a command that fails from here on leaves nothing an earlier one wrote.
    out = clear_outputs(args.out, MATERIALIZE_OUTPUTS)
    task = tasks_.table()[args.task]
    config = robotwin.SceneConfig(
        task_config=args.task_config, save_freq=args.save_freq, embodiment=args.embodiment
    )
    done = materialize(
        task.name,
        args.seed_candidates,
        config,
        out,
        eval_seeds=args.eval_seed_candidates,
        legal_joints=args.legal_joints,
    )
    # Every candidate rejected is a result, written to result.json like a prompt.
    print(json.dumps(done.result, indent=2, sort_keys=True))
    return EXIT_OK


def _demo_verify(args: argparse.Namespace) -> int:
    from . import check

    verdict = check.verify_prompt(Path(args.demo) / PROMPT_FILE, _unit(args))
    print(json.dumps(verdict))
    return EXIT_OK if verdict["ok"] else EXIT_PROMPT


def _eval_run(args: argparse.Namespace) -> int:
    from .policy import DummyPolicy, ReplayPolicy
    from .unit import RUN_UNIT_OUTPUTS, clear_outputs, run_unit

    started = time.monotonic()
    served_only = [
        flag
        for flag, value in (
            ("--act-timeout-s", args.act_timeout_s),
            ("--policy-budget-s", args.policy_budget_s),
            ("--unit-timeout-s", args.unit_timeout_s),
            ("--policy-log", args.policy_log),
        )
        if value is not None
    ]
    if args.stub_policy and served_only:
        raise UsageError(f"{', '.join(served_only)} go with --policy, not --stub-policy")
    prompt = Path(args.demo) / PROMPT_FILE
    # First, so a command that fails from here on leaves nothing an earlier one wrote.
    out = clear_outputs(args.out, RUN_UNIT_OUTPUTS, prompt=prompt)
    if args.policy is not None:
        from . import remote

        deadline = None
        if args.unit_timeout_s is not None:
            # No call to the policy runs into the caller's kill: eval run keeps time to write why.
            deadline = started + args.unit_timeout_s - remote.RESULT_RESERVE_S
        # The key comes from the environment, never argv; nothing connects until the unit resets
        # the policy, so a policy that cannot be reached is the unit's result, not an exit code.
        policy = remote.RemotePolicy(
            args.policy,
            remote.authkey_from_env(args.authkey_env),
            act_timeout_s=(
                remote.ACT_TIMEOUT_S if args.act_timeout_s is None else args.act_timeout_s
            ),
            policy_budget_s=(
                remote.POLICY_BUDGET_S if args.policy_budget_s is None else args.policy_budget_s
            ),
            deadline=deadline,
            log_file=args.policy_log,
        )
    else:
        policy = {"zero": DummyPolicy, "replay": ReplayPolicy}[args.stub_policy]()
    try:
        # Resolved before entering the RoboTwin seam, which moves the working directory.
        result = run_unit(prompt.resolve(), policy, out, step_ratio=args.step_ratio)
    finally:
        # However the unit ended: a served policy's server exits once its client has gone.
        policy.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    # A failed or void unit is a result, written to result.json.
    return EXIT_OK


def _tasks_list(args: argparse.Namespace) -> int:
    table = tasks_.table()
    if args.category is not None and args.category not in table.categories:
        raise UsageError(
            f"unknown category {args.category!r}; the table has {', '.join(table.categories)}"
        )
    selected = [
        task
        for task in table.tasks.values()
        if (args.category is None or task.category == args.category)
        and (args.arms != ONE or task.arms == ONE)
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "categories": dict(table.categories),
                    "tasks": {t.name: {"category": t.category, "arms": t.arms} for t in selected},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return EXIT_OK
    for category, members in table.by_category().items():
        members = tuple(task for task in members if task in selected)
        if not members:
            continue
        print(f"{table.categories[category]} ({category})")
        for task in members:
            print(f"  {task.name}  ({LABELS[task.arms]})")
    return EXIT_OK


def _config_check(args: argparse.Namespace) -> int:
    """Load a task config as a unit is built with it, and print it with the harness's facts: what
    the competition checks before it deals a unit to this checkout."""
    from . import robotwin

    config = _task_config(args.task_config)
    resolved = robotwin.SceneConfig(
        task_config=args.task_config, embodiment=args.embodiment
    ).resolve()
    camera = config.get("camera") or {}
    cameras = [
        name
        for name, collected in (
            ("head_camera", camera.get("collect_head_camera")),
            ("left_camera", camera.get("collect_wrist_camera")),
            ("right_camera", camera.get("collect_wrist_camera")),
        )
        if collected
    ]
    report = {
        "task_config": args.task_config,
        "level": config.get("level"),
        "embodiment": resolved["embodiment_name"],
        "robotwin_embodiment": list(resolved["embodiment"]),
        "cameras": cameras,
        "harness": harness(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK


def _selftest(args: argparse.Namespace) -> int:
    """One task end to end, the chain a duel depends on: `demo make` records a demonstration and
    fingerprints its scene, `demo verify` holds the prompt to its unit, and `eval run` rebuilds the
    scene and plays the demonstration back with the replay stub, in the Same Scene setting, where
    the replay must succeed."""
    out = Path(args.out).resolve()
    demo = out / "unit"
    unit_flags = {
        "task": args.task,
        "task_config": args.task_config,
        "embodiment": args.embodiment,
        "save_freq": 15,
        "seed_candidates": [args.seed + i for i in range(4)],
        "eval_seed_candidates": None,
    }
    started = time.monotonic()
    status = _demo_make(
        argparse.Namespace(**unit_flags, legal_joints=False, out=str(demo), denoiser=None)
    )
    if status != EXIT_OK:
        return status
    made = json.loads((demo / "result.json").read_text(encoding="utf-8"))
    if not made.get("ok"):
        return _fail(f"the expert made no demonstration: {made.get('error')}", EXIT_UNAVAILABLE)
    status = _demo_verify(argparse.Namespace(**unit_flags, demo=str(demo)))
    if status != EXIT_OK:
        return status
    verified = time.monotonic()
    status = _eval_run(
        argparse.Namespace(
            demo=str(demo),
            out=str(out / "result"),
            policy=None,
            stub_policy="replay",
            authkey_env=AUTHKEY_ENV,
            act_timeout_s=None,
            policy_budget_s=None,
            unit_timeout_s=None,
            policy_log=None,
            step_ratio=None,
        )
    )
    if status != EXIT_OK:
        return status
    record = json.loads((out / "result" / "result.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "task": args.task,
                "embodiment": args.embodiment,
                "success": record.get("success"),
                "void": record.get("void"),
                "steps": record.get("steps"),
                "demo_seconds": round(verified - started, 1),
                "eval_seconds": round(time.monotonic() - verified, 1),
            }
        )
    )
    if record.get("success") is True:
        return EXIT_OK
    trouble = record.get("error") or "the replay did not solve the task"
    if record.get("void") and record.get("void_cause") == "harness":
        return _fail(f"the replay was void on the harness: {trouble}", EXIT_UNAVAILABLE)
    return _fail(f"the replay did not end success: {trouble}", EXIT_GATE)


# -- what the harness is ----------------------------------------------------------------------


def harness() -> dict[str, Any]:
    """What this checkout's harness is and does, for the competition to record and check: the
    command line's contract version, the source it runs, the commit, the robots and their action
    widths, the cameras, the prompt's channel map and the limits a served policy gets."""
    from . import check, prompt, remote, robotwin
    from .records import git_commit

    return {
        "cli": "robotensor_bench.cli",
        "api": CLI_API,
        "version": __version__,
        "source_sha256": source_sha256(),
        "commit": git_commit(robotwin.REPO_ROOT),
        "embodiments": {
            name: {
                "robotwin": list(robotwin.EMBODIMENTS[name]),
                "action_dims": {"qpos": check.QPOS_DIMS[name], "ee": robotwin.EE_ACTION_DIM},
            }
            for name in sorted(robotwin.EMBODIMENTS)
        },
        "action_types": list(remote.ACTION_TYPES),
        "cameras": list(check.CAMERAS),
        "channels": {name: list(arrays) for name, arrays in prompt.CHANNELS.items()},
        "prefix_channels": list(prompt.PREFIX_CHANNELS),
        "metadata": list(prompt.METADATA),
        "void_causes": ["harness", "policy"],
        # What eval run gives a served policy unless told otherwise; a caller's unit timeout must
        # leave the harness `result_reserve_s` beyond the policy's budget.
        "limits": {
            "connect_timeout_s": remote.CONNECT_TIMEOUT_S,
            "setup_timeout_s": remote.SETUP_TIMEOUT_S,
            "act_timeout_s": remote.ACT_TIMEOUT_S,
            "close_timeout_s": remote.CLOSE_TIMEOUT_S,
            "policy_budget_s": remote.POLICY_BUDGET_S,
            "result_reserve_s": remote.RESULT_RESERVE_S,
        },
    }


# -- arguments --------------------------------------------------------------------------------


def _add_unit(parser: argparse.ArgumentParser) -> None:
    """The flags that say what a unit is: `demo make` builds it from them, and `demo verify`
    holds a prompt to them."""
    parser.add_argument("--task", required=True, help="a task of the table (`tasks list`)")
    parser.add_argument(
        "--task-config", required=True, help="a RoboTwin env_cfg/task_config name, e.g. level1"
    )
    _add_embodiment(parser, required=True)
    parser.add_argument(
        "--seed-candidates",
        type=_seeds,
        required=True,
        metavar="S,S,...",
        help="candidate scenes, tried in order until the expert solves one: where the "
        "demonstration is recorded",
    )
    parser.add_argument(
        "--eval-seed-candidates",
        type=_seeds,
        metavar="S,S,...",
        help="candidate evaluation scenes, which make the unit cross-scene: it is scored on the "
        "first the expert solves (verified without rendering), not where it was demonstrated",
    )
    parser.add_argument(
        "--save-freq", type=int, default=15, help="control steps per demonstration frame"
    )


def _add_embodiment(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--embodiment",
        choices=sorted(EMBODIMENTS),
        required=required,
        help="the robot: aloha-agilex (one dual-arm URDF, 14-wide qpos) or franka-panda "
        "(two Franka arms 0.8 m apart, 16-wide qpos)"
        + ("" if required else "; without it, the task config's own"),
    )


def _add_expect_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expect-source-sha256",
        metavar="HEX",
        help="refuse to run, exit 2, unless this harness's source digests to HEX "
        "(robotensor_bench.source_sha256); the competition passes the one it checked",
    )


def _add_denoiser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--denoiser",
        choices=DENOISERS,
        help=f"the ray-tracing denoiser, as ${DENOISER_ENV} sets it, for a caller that passes "
        "the command no such variable (default: none on GPUs OIDN cannot run on, else RoboTwin's)",
    )


def _unit(args: argparse.Namespace) -> Any:
    from . import check

    tasks_.table()[args.task]
    level = None
    if args.eval_seed_candidates is not None:
        level = _task_config(args.task_config).get("level")
        if set(args.eval_seed_candidates) & set(args.seed_candidates):
            raise UsageError("--seed-candidates and --eval-seed-candidates share a seed")
    return check.Unit(
        task=args.task,
        embodiment=args.embodiment,
        task_config=args.task_config,
        save_freq=args.save_freq,
        seed_candidates=tuple(args.seed_candidates),
        eval_seed_candidates=(
            None if args.eval_seed_candidates is None else tuple(args.eval_seed_candidates)
        ),
        level=level,
    )


def _task_config(name: str) -> dict[str, Any]:
    from . import robotwin

    try:
        return robotwin.read_task_config(name)
    except FileNotFoundError:
        raise UsageError(f"no task config {name!r} in {robotwin.task_config_dir()}") from None


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in str(text).split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of seeds"
        ) from None
    if not seeds:
        raise argparse.ArgumentTypeError("no seed given")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"{text!r} holds a seed twice")
    if any(not 0 <= seed < 2**32 for seed in seeds):
        raise argparse.ArgumentTypeError(f"{text!r} holds a seed outside [0, 2**32)")
    return seeds


def _positive_seconds(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = float("nan")
    if not 0 < seconds < float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number of seconds")
    return seconds


def _positive_ratio(text: str) -> float:
    try:
        ratio = float(text)
    except ValueError:
        ratio = float("nan")
    if not 0 < ratio < float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive ratio")
    return ratio


def _fail(message: str, code: int) -> int:
    print(f"{PROG}: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
