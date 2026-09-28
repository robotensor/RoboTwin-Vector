"""The benchmark's source digest: which code runs, however it was installed."""

import shutil
from pathlib import Path

import robotensor_bench

PACKAGE = Path(robotensor_bench.__file__).resolve().parent


def test_the_digest_is_the_packages_own_source_wherever_it_lies(tmp_path):
    copy = tmp_path / "elsewhere" / "robotensor_bench"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert robotensor_bench.source_digest(copy) == robotensor_bench.source_sha256()
    assert len(robotensor_bench.source_sha256()) == 64


def test_compiled_caches_and_other_files_are_not_source(tmp_path):
    copy = tmp_path / "robotensor_bench"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = robotensor_bench.source_digest(copy)
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "unit.cpython-310.py").write_text("stale")
    (copy / "notes.txt").write_text("not source")
    assert robotensor_bench.source_digest(copy) == before


def test_a_changed_renamed_or_added_file_changes_the_digest(tmp_path):
    copy = tmp_path / "robotensor_bench"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = robotensor_bench.source_digest(copy)

    unit = copy / "unit.py"
    unit.write_text(unit.read_text() + "\n# edited\n")
    edited = robotensor_bench.source_digest(copy)
    assert edited != before

    unit.rename(copy / "unit2.py")
    assert robotensor_bench.source_digest(copy) not in (before, edited)

    (copy / "unit2.py").rename(unit)
    (copy / "tasks_extra.yml").write_text("extra: {}\n")
    assert robotensor_bench.source_digest(copy) not in (before, edited)
