"""Stage records (`stage-<n>.json`) in the contract's format."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

GIT_TIMEOUT = 5.0     # seconds per git call; a slow or missing git never holds up a stage


def code_version(where: Path | None = None) -> dict | None:
    """The code that is running: `{"commit": <HEAD sha>, "dirty": <tracked files differ from it>}`, or None when
    `where` (default: this package) is not in a git checkout or git is missing or too slow. Untracked files do
    not count as dirty (run folders and local inputs sit next to the code)."""
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


def write_atomic(path: Path, text: str) -> None:
    """Write a text file in one step (a temporary file in the same folder, then a rename), so a reader never sees
    a partial file and a failure leaves the previous one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def check(name: str, value, threshold=None, passed: bool | None = None, gate: bool = True, **extra) -> dict:
    """A measured check. `passed` defaults to value >= threshold when a threshold is given."""
    if passed is None:
        passed = True if threshold is None else bool(value >= threshold)
    out = {"name": name, "value": value, "threshold": threshold, "pass": bool(passed), "gate": gate}
    out.update(extra)
    return out


class StageRecord:
    def __init__(self, run: Path, stage: int, name: str):
        self.run = run
        self.data = {
            "stage": stage, "name": name, "ok": False,
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seconds": None, "cost_usd": 0.0,
            "cost_basis": "local compute only (numpy + Blender); no paid 3D generation",
            "models": [], "tools": [], "outputs": [], "checks": [], "notes": "",
            "code": code_version(),
        }
        self._t0 = time.monotonic()

    def add(self, c: dict) -> dict:
        self.data["checks"].append(c)
        return c

    def note(self, text: str) -> None:
        self.data["notes"] = (self.data["notes"] + "; " if self.data["notes"] else "") + text

    def finish(self, extra: dict | None = None) -> dict:
        self.data["seconds"] = round(time.monotonic() - self._t0, 2)
        self.data["ok"] = all(c["pass"] for c in self.data["checks"] if c.get("gate", True))
        if extra:
            self.data.update(extra)
        path = self.run / f"stage-{self.data['stage']}.json"
        write_atomic(path, json.dumps(self.data, indent=2))      # never a partial record
        return self.data
