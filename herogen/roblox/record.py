"""The stage-6 record's provenance: the running code's commit, and the inputs' hashes."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

GIT_TIMEOUT = 5.0  # seconds per git call; a slow or missing git never holds up the stage


def code_version(where: Path | None = None) -> dict | None:
    """`{"commit": <HEAD sha>, "dirty": <tracked files differ from it>}` for the checkout holding `where`
    (default: this package), or None outside a git checkout or when git is missing or too slow."""
    where = Path(__file__).resolve().parent if where is None else Path(where)

    def git(*args: str) -> str | None:
        try:
            p = subprocess.run(["git", "-C", str(where), *args], capture_output=True, text=True,
                               timeout=GIT_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            return None
        return p.stdout if p.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    if not head or len(head.strip()) < 7:
        return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": head.strip(), "dirty": None if status is None else bool(status.strip())}


def file_hash(path: Path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": os.path.getsize(path)}


def write_atomic(path: Path, text: str) -> None:
    """Write a text file in one step (a temporary file, then a rename), so a reader never sees half of it."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
