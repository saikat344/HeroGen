"""The stage-6 record's provenance (the running code's commit, the inputs' hashes) and its checks as build-log
events."""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
from pathlib import Path

from ..pipeline import events

CHECK_OPS = {"<", "<=", "==", ">=", ">", "!="}

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



def number(v):
    """A plain finite int or float for an event's `m` (numpy scalars and booleans converted), or None: names,
    lists and missing values are not sent."""
    if hasattr(v, "item") and not isinstance(v, (list, tuple, dict, str)):
        try:
            v = v.item()
        except (TypeError, ValueError):
            return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and math.isfinite(v):
        return float(v)
    return None


def measures(values: dict) -> dict | None:
    """An event's `m`: the entries that are finite numbers, or None when none is."""
    out = {k: n for k, v in values.items() if (n := number(v)) is not None}
    return out or None


def emit_checks(checks: list[dict], seconds: float | None = None) -> None:
    """Each stage-record check as a `check` event (name, measured value, limit, comparison, passed, hard limit or
    informational), then one `checks` summary. A check without a recorded `rule` but with a threshold uses `>=`,
    the default of the stage records' `check()`. A no-op unless the service names an events file."""
    for c in checks:
        m = measures({"value": c.get("value"), "limit": c.get("threshold")})
        rule = c.get("rule")
        op = rule if rule in CHECK_OPS else (">=" if m and "limit" in m else None)
        events.emit("check", who="herogen", name=events.slug(c["name"]), ok=bool(c["pass"]),
                    gate=bool(c.get("gate", True)), op=op, m=m)
    gated = [c for c in checks if c.get("gate", True)]
    events.emit("checks", who="herogen", n=len(checks), passed=sum(1 for c in checks if c["pass"]),
                gated=len(gated), failed=[events.slug(c["name"]) for c in gated if not c["pass"]][:20],
                advisory=[events.slug(c["name"]) for c in checks if not c.get("gate", True) and not c["pass"]][:20],
                s=None if seconds is None else round(seconds, 3))
