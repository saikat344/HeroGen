"""Stage records (`stage-<n>.json`) in the contract's format."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


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
        path.write_text(json.dumps(self.data, indent=2))
        return self.data
