"""The Robotensor Vector competition's harness inside the RoboTwin fork.

Everything the fork adds for the competition lives here: the task table, the one module that
touches RoboTwin (`robotwin`), making a demonstration and saving it (`unit.materialize`), checking a
saved one against its unit (`check`), evaluating a served policy from it (`unit.run_unit`), and the
`rt-bench` command line. It is run from the checkout, never installed:

    python -m robotensor_bench.cli demo make --task place_container_plate --task-config level1 ...
    python -m robotensor_bench.cli eval run --demo POOL/UNIT --policy /tmp/policy.sock ...
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

__all__ = ["CLI_API", "__version__", "source_digest", "source_sha256"]

__version__ = "0.3.0"

#: The version of the command-line contract the competition drives: the verbs, their flags, the
#: files they write and the exit codes. 1 was the plugin ABI of the `robotensor-benchmark-robotwin`
#: distribution this folder replaced; 3 added `eval run --step-ratio`.
CLI_API = 3

#: The files a source digest covers: every module and data file (`tasks.yml`) of the harness.
SOURCE_SUFFIXES = (".py", ".yml")
#: What a source digest skips: the tests, what runs write, and compiled caches.
NOT_SOURCE = ("tests", "runs", "__pycache__")


def source_digest(root: str | Path) -> str:
    """sha256 over the source files under `root`: each one's path relative to `root` and the
    sha256 of its bytes, in path order. Tests, run outputs and compiled caches are not source."""
    root = Path(root)
    files = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.suffix in SOURCE_SUFFIXES
        and path.is_file()
        and not set(path.relative_to(root).parts) & set(NOT_SOURCE)
    )
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@cache
def source_sha256() -> str:
    """The digest of this harness's own source: which code is running. A git commit hides an
    uncommitted edit; this does not. The competition reads it from `config check`, passes it back
    to every command (`--expect-source-sha256`), which refuses to run under any other, and holds
    every result to it: both commands record theirs."""
    return source_digest(Path(__file__).resolve().parent)
