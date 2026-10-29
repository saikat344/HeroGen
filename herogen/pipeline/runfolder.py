"""The run folder: `$HEROGEN_RUNS/<run-id>/`, the only interface between stages (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_RUNS = "./runs"


def runs_root(explicit: str | os.PathLike | None = None) -> Path:
    root = Path(explicit if explicit is not None else os.environ.get("HEROGEN_RUNS", DEFAULT_RUNS))
    root.mkdir(parents=True, exist_ok=True)
    # Run folders hold generated images and must never be committed; keep the folder self-ignoring.
    ignore = root / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")
    return root


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RunFolder:
    def __init__(self, root: Path, run_id: str) -> None:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError(f"bad run id {run_id!r}")
        self.path = Path(root) / run_id
        self.run_id = run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def file(self, rel: str) -> Path:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, rel: str) -> bool:
        return (self.path / rel).exists()

    def write_json(self, rel: str, data: Any) -> Path:
        p = self.file(rel)
        p.write_text(json.dumps(data, indent=2) + "\n")
        return p

    def read_json(self, rel: str) -> Any:
        return json.loads((self.path / rel).read_text())

    def write_stage(self, record: dict[str, Any]) -> Path:
        return self.write_json(f"stage-{record['stage']}.json", record)

    def read_stage(self, n: int) -> dict[str, Any] | None:
        p = self.path / f"stage-{n}.json"
        return json.loads(p.read_text()) if p.exists() else None
