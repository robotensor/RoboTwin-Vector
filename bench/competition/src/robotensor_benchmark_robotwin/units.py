"""A duel's units, as a pure function of the seed material the orchestrator hands over.

A third party holding a published record must derive the same units, on any Python and any numpy,
so nothing here consults a library RNG or Python's hash: every number is drawn from `Sha256Counter`,
sha256 over the seed material and a counter, in a fixed order. The task order is drawn first and
then each unit's candidates in turn, so a smaller duel's units are the first units of a larger one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from robotwin_bench import tasks

from . import catalogue

#: Candidate scene seeds per unit, tried in order by `materialize` until the expert solves one. A
#: unit is void for both sides only when all of them are rejected: at 75% expert success (the
#: lowest the Aloha-AgileX survey measured outside place_object_basket), four candidates all fail
#: 0.4% of the time, at the 2 of 3 the Franka survey kept 1.2%, and at 50% about 6% (the spec
#: voids a duel past 20%). Every rejected candidate costs one scene and one expert run, and four of
#: the slowest experts still fit materialize's wall clock.
SCENE_SEED_CANDIDATES = 4

#: Scene seeds are drawn from [0, SEED_BOUND): RoboTwin feeds them to `np.random.seed`, and this is
#: the bound the benchmark's own seed streams use.
SEED_BOUND = 2**31 - 1

#: Hashed ahead of the seed material, so these draws are this derivation's and no other's. A change
#: to how units are drawn changes this label, and with it every unit. units/1 drew `franka_1arm`
#: from a provisional suite derived from each task's arms; units/2 draws it from the suite the
#: Franka survey named, so a unit list recorded under one label never reads as the other's. Each
#: suite has a label of its own (`catalogue.PROFILES`), so adding a suite moves no other suite's
#: units; this is `franka_1arm`'s.
DERIVATION = catalogue.PROFILES[catalogue.FRANKA_1ARM].derivation

#: What `catalogue_sha256` gave when the derivation was pinned. The orchestrator pins this plugin
#: by its wheel's sha256, but each task's category and arms are robotwin-bench's: another table
#: would derive other units under the same pin, so `derive_units` refuses one. A change to the
#: catalogue that is meant updates this, with `DERIVATION` and the pinned test if units change.
CATALOGUE_SHA256 = "3e9f05023783dddcd7372ff3862172006551608d2c580f387974bc3e82bf4e02"


class Sha256Counter:
    """Integers from sha256(label | material | counter), the counter counting up from 0."""

    def __init__(self, material: str, label: str = DERIVATION) -> None:
        self._prefix = f"{label}|{material}|".encode()
        self._counter = 0

    def next_u64(self) -> int:
        digest = hashlib.sha256(self._prefix + str(self._counter).encode("ascii")).digest()
        self._counter += 1
        return int.from_bytes(digest[:8], "big")

    def below(self, bound: int) -> int:
        """Uniform in [0, bound), by rejecting the draws that would bias a modulo."""
        if bound < 1:
            raise ValueError(f"bound must be at least 1, not {bound}")
        limit = (1 << 64) - (1 << 64) % bound
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def catalogue_sha256(table: tasks.TaskTable) -> str:
    """sha256 of everything a unit is drawn from or carries: every suite's tasks in order, each
    suite's robot, and each task's category, arms and label."""
    known = catalogue.suites(table)
    payload = {
        "suites": {name: list(members) for name, members in known.items()},
        "embodiments": {name: catalogue.embodiment_of(name) for name in known},
        "tasks": {
            name: {
                "category": task.category,
                "arms": task.arms,
                "label": catalogue.task_label(name),
            }
            for name, task in table.tasks.items()
        },
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def suite_tasks(table: tasks.TaskTable, suite: str, category: str | None) -> list[str]:
    """The tasks units of `suite` (within `category`) are drawn from, in table order."""
    known = catalogue.suites(table)
    if suite not in known:
        raise ValueError(f"unknown suite {suite!r}; the catalogue has {', '.join(sorted(known))}")
    members = list(known[suite])
    if category is not None:
        if category not in table.categories:
            raise ValueError(f"unknown category {category!r}")
        members = [name for name in members if table[name].category == category]
        if not members:
            raise ValueError(f"suite {suite!r} has no task in category {category!r}")
    return members


def derive_units(
    *,
    seed_material: str,
    count: int,
    suite: str,
    category: str | None = None,
    table: tasks.TaskTable | None = None,
) -> list[dict[str, Any]]:
    """`count` units of `suite` (within `category`), each a task and its candidate scene seeds.

    Tasks are spread evenly: the suite's tasks are shuffled once and dealt in turn, so no task gets
    two units more than another. Each unit carries `task`, `task_label`, `suite`, `category` and
    `instance_params` — `scene_seeds`, its `SCENE_SEED_CANDIDATES` candidates in the order they
    are tried; `embodiment`, the robot; and `scene_seed`, None, because which candidate becomes
    the unit's scene is known only once materialize has run (its `result.json` and the prompt's
    `meta` name it).

    Draws are hashed under the suite's own label and taken from its seed range
    (`catalogue.PROFILES`). A cross-scene suite's unit also carries `demo_seeds`, the candidates
    its demonstration is recorded on, drawn after its `scene_seeds` and disjoint from them, and
    its `task_config` and `level`: `scene_seeds` stay the scenes the unit is scored in.
    """
    if not isinstance(seed_material, str):
        raise TypeError(f"seed_material must be a string, not {type(seed_material).__name__}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be a non-negative integer, not {count!r}")
    table = table or tasks.table()
    found = catalogue_sha256(table)
    if found != CATALOGUE_SHA256:
        raise ValueError(
            f"the task catalogue digests to {found}, not the {CATALOGUE_SHA256} units were pinned "
            "on: this robotwin-bench's table would derive other units under the same plugin"
        )
    members = suite_tasks(table, suite, category)
    embodiment = catalogue.embodiment_of(suite)
    profile = catalogue.profile(suite)
    span = profile.seed_high - profile.seed_low

    def seed_list(taken: list[int]) -> list[int]:
        seeds: list[int] = []
        while len(seeds) < SCENE_SEED_CANDIDATES:
            seed = profile.seed_low + draws.below(span)
            if seed not in seeds and seed not in taken:
                seeds.append(seed)
        return seeds

    draws = Sha256Counter(seed_material, profile.derivation)
    order = list(members)
    for i in range(len(order) - 1, 0, -1):  # Fisher-Yates, on sha256 draws
        j = draws.below(i + 1)
        order[i], order[j] = order[j], order[i]

    units = []
    for index in range(count):
        name = order[index % len(order)]
        seeds = seed_list([])
        params: dict[str, Any] = {
            "scene_seed": None,
            "scene_seeds": seeds,
            "embodiment": embodiment,
        }
        if profile.cross_scene:
            params["demo_seeds"] = seed_list(seeds)
            params["task_config"] = profile.task_config
            params["level"] = profile.level
        units.append(
            {
                "task": name,
                "task_label": catalogue.task_label(name),
                "suite": suite,
                "category": table[name].category,
                "instance_params": params,
            }
        )
    return units
