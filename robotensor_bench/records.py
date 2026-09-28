"""What an episode records, the evaluation settings, and the git and JSON helpers every result uses.

`EpisodeRecord` is what `episode.run_episode`, a unit's two halves run back to back in one process,
returns; a duel's commands write `result.json` instead (`unit`).
"""

from __future__ import annotations

import enum
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SAME_SCENE = "same_scene"
#: The demonstration comes from one scene and the rollout is scored in another, both drawn at the
#: task config's difficulty `level` (the fork's `envs/_levels.py`): the Vector competition's setting.
CROSS_SCENE_LEVEL = "cross_scene_level"
EVALUATION_SETTINGS = (SAME_SCENE, CROSS_SCENE_LEVEL)


class RecordError(ValueError):
    """A record is malformed."""


class Status(str, enum.Enum):
    # The expert produced a demonstration, the reset reproduced its scene, the policy rolled out.
    SCORED = "scored"
    # No seed in the attempt budget produced a successful expert demonstration.
    REJECTED = "rejected"
    # The evaluation scene did not match the demonstration's, or the harness itself failed.
    INVALID = "invalid"


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    evaluation_setting: str
    skill_category: str
    task: str
    scene_seed: int | None
    status: Status
    success: bool | None
    steps: int
    step_limit: int | None
    demonstration_frames: int
    expert_generation_attempts: int
    rejections: dict[str, int]
    scene_max_error: float
    model: str
    # The robot, as `robotwin.embodiment_name` names it (the `--embodiment` choice, or the arms and
    # their distance): a 14-wide aloha-agilex episode and a 16-wide franka-panda one are not the
    # same measurement, and neither are two Frankas at different distances.
    embodiment: str
    checkpoint: str | None = None
    detail: str = ""
    duration_s: float = 0.0
    # One example detail per rejection reason: why seeds were rejected, not only how often.
    rejection_details: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = Status(self.status)
        object.__setattr__(self, "status", status)
        if status is Status.SCORED and not isinstance(self.success, bool):
            raise RecordError(f"episode {self.episode}: a scored episode needs a boolean success")
        if status is not Status.SCORED and self.success is not None:
            raise RecordError(f"episode {self.episode}: only scored episodes carry a success")

    @property
    def scored(self) -> bool:
        return self.status is Status.SCORED

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EpisodeRecord:
        # Before a run could choose its robot, every episode ran on aloha-agilex, the embodiment of
        # every shipped task config; records from then still load, and say so.
        return cls(**{"embodiment": "aloha-agilex", **data})


def write_json(path: Path, data: Any, *, sort_keys: bool = False) -> None:
    """Replace `path` with `data` as JSON, whole or not at all.

    The text goes to a sibling `.tmp` and is renamed into place, so a process killed mid-write
    leaves the previous file rather than a truncated one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")
    tmp.replace(path)


def is_checkout_top(path: Path) -> bool:
    """Whether `path` is the top of a git checkout (a worktree's or a submodule's included).

    A benchmark installed from a wheel lives inside whatever directory the environment is in, and
    that may be some other repository, or a submodule directory left empty may sit inside the
    benchmark's: git would answer for the repository around it.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(top) and Path(top).resolve() == Path(path).resolve()


def git_commit(path: Path) -> str | None:
    """HEAD of the checkout at `path`, suffixed `-dirty` when it has uncommitted changes; None
    unless `path` is the top of a checkout (`is_checkout_top`)."""
    if not is_checkout_top(path):
        return None
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def environment() -> dict[str, str]:
    return {"python": sys.version.split()[0], "platform": platform.platform()}
