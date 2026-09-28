import json
import subprocess
from pathlib import Path

import pytest

from robotensor_bench.records import (
    SAME_SCENE,
    EpisodeRecord,
    RecordError,
    Status,
    git_commit,
    write_json,
)


def record(episode=0, status=Status.SCORED, success=True, **overrides) -> EpisodeRecord:
    base = dict(
        episode=episode,
        evaluation_setting=SAME_SCENE,
        skill_category="pick_and_place",
        task="place_object_basket",
        scene_seed=123,
        status=status,
        success=success,
        steps=40,
        step_limit=400,
        demonstration_frames=30,
        expert_generation_attempts=1,
        rejections={},
        scene_max_error=0.0,
        model="replay",
        embodiment="aloha-agilex",
    )
    base.update(overrides)
    return EpisodeRecord(**base)


def test_only_scored_episodes_carry_a_success():
    with pytest.raises(RecordError):
        record(status=Status.SCORED, success=None)
    with pytest.raises(RecordError):
        record(status=Status.REJECTED, success=False)
    assert not record(status=Status.REJECTED, success=None).scored


def test_records_round_trip_through_json():
    original = record(rejections={"unstable": 2}, expert_generation_attempts=3)
    restored = EpisodeRecord.from_json(json.loads(json.dumps(original.to_json())))
    assert restored == original
    assert restored.status is Status.SCORED


def test_git_commit_of_a_non_repository_is_none(tmp_path):
    assert git_commit(tmp_path) is None


def test_git_commit_is_a_checkouts_own_never_the_repository_around_it(tmp_path):
    # A wheel's package sits wherever its environment does, which may be inside another repository.
    repo = tmp_path / "unrelated"
    site = repo / ".venv" / "lib" / "python3.10"
    site.mkdir(parents=True)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README").write_text("another project\n")
    subprocess.run([*git, "add", "README"], check=True)
    subprocess.run([*git, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert git_commit(repo) == head
    assert git_commit(site) is None


def test_records_written_before_rejection_details_still_load():
    data = record().to_json()
    data.pop("rejection_details")
    assert EpisodeRecord.from_json(data).rejection_details == {}


def test_records_written_before_a_run_could_choose_its_robot_are_aloha():
    data = record(embodiment="franka-panda").to_json()
    assert EpisodeRecord.from_json(data).embodiment == "franka-panda"
    data.pop("embodiment")
    assert EpisodeRecord.from_json(data).embodiment == "aloha-agilex"


def test_write_json_replaces_the_file_whole_and_keeps_key_order(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("old", encoding="utf-8")
    write_json(path, [{"task": "click_bell", "seeds": 2, "arms": ["left"]}])
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == [{"task": "click_bell", "seeds": 2, "arms": ["left"]}]
    assert text.index('"task"') < text.index('"seeds"') < text.index('"arms"')
    assert text.endswith("\n") and not path.with_suffix(".json.tmp").exists()


def test_write_json_torn_mid_write_leaves_the_previous_file(tmp_path, monkeypatch):
    # A survey runs for hours and rewrites its json after every task; a kill that lands during
    # the write must leave the last complete payload, not a truncated one.
    path = tmp_path / "out.json"
    path.write_text('{"complete": true}\n', encoding="utf-8")
    original = Path.write_text

    def torn(self, text, *args, **kwargs):
        original(self, text[: len(text) // 2], *args, **kwargs)
        raise OSError("killed mid-write")

    monkeypatch.setattr(Path, "write_text", torn)
    with pytest.raises(OSError):
        write_json(path, {"complete": False, "more": list(range(50))})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
