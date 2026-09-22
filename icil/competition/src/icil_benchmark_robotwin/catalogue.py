"""What the plugin draws units from: tasks, their categories and arms, and the suites.

The tasks, their categories and the arms each task's expert uses (`Task.arms`, read off its
`play_once`) are `robotwin_icil.tasks`'s. The suites are the competition's own:

- `franka_1arm`, the suite the orchestrator's one-arm Franka track draws from, chosen from the
  Franka survey (robotensor/RoboTwin-ICIL#83, measurements at `SURVEY`, rule and decision in this
  package's README): Same Scene, RoboTwin's `demo_clean`.
- `vector_level1`, the Vector competition's: the 16 tasks of the difficulty-level benchmark
  (`docs/LEVELS.md` of the Vector level experiment) on two Franka arms at level 1, cross-scene: the
  demonstration is recorded in one scene and the unit is scored in another (`robotwin_icil`'s
  `cross_scene_level`).

How a suite's units are built and checked — its robot, task config, setting, seed range and
derivation label — is its `SuiteProfile`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from robotwin_icil import tasks
from robotwin_icil.arms import ONE, SWITCHING, TWO
from robotwin_icil.records import CROSS_SCENE_LEVEL, SAME_SCENE

#: The suite the orchestrator's one-arm Franka track names in `spec.json`.
FRANKA_1ARM = "franka_1arm"
#: The categories that track scores, as its skills name them.
FRANKA_1ARM_CATEGORIES = ("pick_and_place", "stacking", "press_push")
#: The Vector competition's suite: the level benchmark's 16 tabletop tasks.
VECTOR_LEVEL1 = "vector_level1"
#: The track's tasks, in task-table order: the tasks whose expert, on two Franka arms, solved at
#: least 2 of 3 surveyed seeds and moved one arm in every demonstration it produced. Stacking has
#: no one-arm task, so stack_bowls_two, an arm-switching task, stands in for it.
SUITES: dict[str, tuple[str, ...]] = {
    FRANKA_1ARM: ("place_empty_cup", "stack_bowls_two", "click_bell", "press_stapler"),
    # The tasks whose load_actors the level gate covers (envs/_levels.py), in the level
    # benchmark's own order; Vector's base checkpoints were trained on exactly these.
    VECTOR_LEVEL1: (
        "beat_block_hammer",
        "move_stapler_pad",
        "place_bread_skillet",
        "place_burger_fries",
        "place_can_basket",
        "place_cans_plasticbox",
        "place_container_plate",
        "place_dual_shoes",
        "place_empty_cup",
        "place_fan",
        "place_mouse_pad",
        "place_object_basket",
        "place_object_scale",
        "place_object_stand",
        "place_phone_stand",
        "place_shoe",
    ),
}
#: Where the survey that chose `franka_1arm` gives its measurements.
SURVEY = "docs/results/survey-franka-seed0.json"
#: What `info()` calls a `franka_1arm` task whose expert is not a one-arm one.
STAND_IN_KINDS = {SWITCHING: "an arm-switching", TWO: "a two-arm"}

#: The robot a suite's units run on: the Franka suites on two Franka arms, any other suite on the
#: benchmark's default robot.
FRANKA = "franka-panda"
DEFAULT_EMBODIMENT = "aloha-agilex"

#: `np.random.seed`'s bound, below which every seed is drawn (the benchmark's own seed streams use
#: it too).
SEED_BOUND = 2**31 - 1
#: The level benchmark's public seed ranges (`envs/_levels.py`: train `[0, 1M)`, eval `[1M, 2M)`)
#: end here; a competition unit never draws a scene either of them published.
PUBLIC_SEEDS_END = 2_000_000


@dataclass(frozen=True)
class SuiteProfile:
    """How one suite's units are built: the robot and RoboTwin task config every unit of it is
    materialized with, the evaluation setting, the derivation label its draws are hashed under,
    the range seeds are drawn from, and whether the expert must plan inside the joint limits."""

    embodiment: str
    task_config: str
    evaluation_setting: str
    derivation: str
    seed_low: int = 0
    seed_high: int = SEED_BOUND
    legal_joints: bool = False
    level: int | None = None

    @property
    def cross_scene(self) -> bool:
        return self.evaluation_setting == CROSS_SCENE_LEVEL

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "cross_scene": self.cross_scene}


PROFILES: dict[str, SuiteProfile] = {
    # Unchanged since units/2: its pinned unit hashes must not move.
    FRANKA_1ARM: SuiteProfile(
        embodiment=FRANKA,
        task_config="demo_clean",
        evaluation_setting=SAME_SCENE,
        derivation="robotwin-icil-competition/units/2",
    ),
    VECTOR_LEVEL1: SuiteProfile(
        embodiment=FRANKA,
        task_config="level1",
        evaluation_setting=CROSS_SCENE_LEVEL,
        derivation="robotwin-icil-competition/vector_level1/units/1",
        seed_low=PUBLIC_SEEDS_END,
        legal_joints=True,
        level=1,
    ),
}


def profile(suite: str) -> SuiteProfile:
    """The profile of a suite, or `ValueError` naming the suites there are."""
    try:
        return PROFILES[suite]
    except KeyError:
        raise ValueError(
            f"unknown suite {suite!r}; the catalogue has {', '.join(sorted(PROFILES))}"
        ) from None


def suites(table: tasks.TaskTable) -> dict[str, tuple[str, ...]]:
    """Every suite units can be drawn from, each in its own order. A member the task table does
    not hold is refused, naming it."""
    for members in SUITES.values():
        for name in members:
            table[name]
    return dict(SUITES)


def stand_ins(table: tasks.TaskTable) -> dict[str, str]:
    """The `franka_1arm` tasks whose expert is not a one-arm one, in suite order, each with what it
    stands in for and the limit that comes with it: stack_bowls_two, for stacking, which has no
    one-arm task at the pinned RoboTwin. Read off the table, so a note cannot outlive it."""
    notes = {}
    for name in suites(table)[FRANKA_1ARM]:
        task = table[name]
        if task.arms == ONE:
            continue
        without = not any(
            other.category == task.category and other.arms == ONE for other in table.tasks.values()
        )
        where = f"{task.category}, which has no one-arm task" if without else task.category
        notes[name] = (
            f"{name} is {STAND_IN_KINDS[task.arms]} stand-in for {where}: every demonstration the "
            "Franka survey kept moved one arm, but a scene the survey did not see can still make "
            "its expert move both arms, and neither materialize nor verify_prompt refuses such a "
            "demonstration"
        )
    return notes


def franka_1arm_basis(table: tasks.TaskTable) -> dict[str, Any]:
    """What `franka_1arm` is made of, for `info()`: the survey that chose it, each category of the
    track with its tasks and the arms their experts use, and `stand_ins`."""
    members = suites(table)[FRANKA_1ARM]
    categories = {}
    for category in FRANKA_1ARM_CATEGORIES:
        names = [name for name in members if table[name].category == category]
        categories[category] = {"tasks": names, "arms": sorted({table[n].arms for n in names})}
    return {"survey": SURVEY, "categories": categories, "stand_ins": stand_ins(table)}


def embodiment_of(suite: str) -> str:
    """The robot a suite's units run on, as `robotwin-icil --embodiment` names it."""
    return PROFILES[suite].embodiment if suite in PROFILES else DEFAULT_EMBODIMENT


def task_label(name: str) -> str:
    """A task's name for people: `click_bell` -> `Click bell`."""
    return name.replace("_", " ").capitalize()


def catalogue(table: tasks.TaskTable | None = None) -> dict[str, Any]:
    """`{"suites": {suite: [task]}, "categories": {id: label}, "tasks": {task: {"category",
    "arms", "label"}}, "embodiments": {suite: robot}}`."""
    table = table or tasks.table()
    return {
        "suites": {name: list(members) for name, members in suites(table).items()},
        "categories": dict(table.categories),
        "tasks": {
            name: {"category": task.category, "arms": task.arms, "label": task_label(name)}
            for name, task in table.tasks.items()
        },
        "embodiments": {name: embodiment_of(name) for name in suites(table)},
    }
