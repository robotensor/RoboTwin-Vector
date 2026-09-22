"""Difficulty-level gating for the RoboTwin level benchmark.

A single integer ``level`` (1-4) decides how much of a scene varies between episodes.  Every source
of randomization is registered here with the lowest level at which it is active, and levels are
strict supersets: a source that is live at level N is live at every level above N.

The one rule that makes the levels comparable: **always draw, then override**.  A gated source still
consumes exactly the same values from ``np.random`` at every level; only the value that reaches the
scene changes.  One seed therefore produces the same object layout at every level, and two levels of
the same seed differ only by the factor under test.
"""

from collections import defaultdict

import numpy as np

# factor name -> lowest level at which it is active
FACTOR_LEVEL = {
    # L1: where the task's own objects are, and which arm does the job
    "object_pose": 1,
    "arm_choice": 1,
    # L2: global scene parameters
    "light": 2,
    "background": 2,
    "table_height": 2,
    # L3: which object it is and what it looks like -- instance (mesh shape + size together),
    # alternative categories, and size/colour for objects the task builds procedurally (boxes)
    "model_id": 3,
    "object_size": 3,  # for objects the task builds procedurally (boxes)
    "object_color": 3,  # for objects the task builds procedurally (boxes)
    "object_category": 3,
    # L4: distractors on the table
    "clutter": 4,
}

MIN_LEVEL, MAX_LEVEL = 1, 4

# what each level adds on top of the one below it, for docs and metadata
LEVEL_DESCRIPTION = {
    1: "object pose (position + yaw) and arm choice",
    2: "L1 + lighting, wall/table texture, table height",
    3: "L2 + object instance (model_id), alternative object categories, procedural object size and colour",
    4: "L3 + clutter (distractor objects on the table)",
}


def factors_for_level(level):
    return sorted(f for f, lv in FACTOR_LEVEL.items() if lv <= int(level))


class LevelController:
    """Decides whether a randomization source applies, and records what it resolved to.

    ``pick`` takes the value that was *already drawn* from ``np.random`` -- the draw happens in the
    caller's argument list, so it happens at every level -- and returns either that value or the
    fixed default for the levels where the factor is not active yet.
    """

    def __init__(self, level):
        level = int(level)
        if not MIN_LEVEL <= level <= MAX_LEVEL:
            raise ValueError(f"level must be in [{MIN_LEVEL}, {MAX_LEVEL}], got {level}")
        self.level = level
        self.resolved = {}  # "<factor>.<name>" -> value that reached the scene
        self._counter = defaultdict(int)

    def active(self, factor):
        try:
            return self.level >= FACTOR_LEVEL[factor]
        except KeyError:
            raise KeyError(f"unregistered randomization factor {factor!r}; "
                           f"add it to FACTOR_LEVEL in envs/_levels.py") from None

    def pick(self, factor, drawn, default, name=None):
        """Return ``drawn`` when ``factor`` is active at this level, else ``default``.

        ``drawn`` must be the result of the random draw itself: evaluating it in the call keeps the
        random stream identical across levels.
        """
        value = drawn if self.active(factor) else default
        if name is None:
            self._counter[factor] += 1
            name = str(self._counter[factor])
        self.resolved[f"{factor}.{name}"] = _plain(value)
        return value

    def record(self, factor, name, value):
        """Record a resolved value that ``pick`` did not produce (e.g. a whole clutter layout)."""
        self.resolved[f"{factor}.{name}"] = _plain(value)

    def metadata(self):
        return {
            "level": self.level,
            "active_factors": factors_for_level(self.level),
            "factor_levels": dict(FACTOR_LEVEL),
            "resolved": dict(self.resolved),
        }


def _plain(value):
    """numpy -> json-friendly, so resolved settings can go straight into episode metadata."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


# Palette for objects the task builds procedurally (boxes).  Level 3 draws from it; below level 3
# the box keeps the colour the upstream task hard-coded.
PROCEDURAL_COLORS = {
    "Red": (1, 0, 0),
    "Green": (0, 1, 0),
    "Blue": (0, 0, 1),
    "Yellow": (1, 1, 0),
    "Cyan": (0, 1, 1),
    "Magenta": (1, 0, 1),
    "Black": (0, 0, 0),
    "Gray": (0.5, 0.5, 0.5),
    "Orange": (1, 0.5, 0),
    "Purple": (0.5, 0, 0.5),
    "Brown": (0.65, 0.4, 0.16),
    "Pink": (1, 0.75, 0.8),
}

# How far a procedural box may be scaled at level 3, per axis.  Kept moderate on purpose: a box's
# contact and functional points scale with its half size, so the scripted expert follows the new
# geometry, but a large change still moves the grasp out of the gripper's range.
BOX_SIZE_RANGE = (0.8, 1.25)


# Disjoint seed ranges per split: a seed used for training can never turn up in evaluation.
# scripts/gen_levels.py draws training/evaluation seeds from these, and
# scripts/eval_policy_xpolicylab.py starts its rollouts inside the evaluation range.
SEED_RANGES = {
    "train": (0, 1_000_000),
    "eval": (1_000_000, 2_000_000),
}
