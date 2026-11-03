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
        """Write atomically: a reader sees the old file or the new one, never half of one, and the new one is
        on disk (file and folder synced) before this returns. A failure leaves the old file as it was."""
        p = self.file(rel)
        text = json.dumps(data, indent=2) + "\n"  # serialised first: a bad value never touches the file
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink()
        fd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return p

    def read_json(self, rel: str) -> Any:
        return json.loads((self.path / rel).read_text())

    def write_stage(self, record: dict[str, Any]) -> Path:
        if self.exists("redo.json"):
            record = {**record, **self.read_json("redo.json")}
        return self.write_json(f"stage-{record['stage']}.json", record)

    def read_stage(self, n: int | str) -> dict[str, Any] | None:
        p = self.path / f"stage-{n}.json"
        return json.loads(p.read_text()) if p.exists() else None


# Images a running stage shows before it ends (`save_draft`): the website publishes them as drafts only while
# their stage runs. They are never outputs, and the stage clears them when it starts and ends.
DRAFTS = "drafts"


def save_draft(run: Path | str, rel: str, img) -> None:
    """Write ``img`` as ``drafts/<rel>`` atomically (a reader sees the old file or the new one), quickly
    compressed. A draft that cannot be written is skipped: it never stops the stage."""
    p = Path(run) / DRAFTS / rel
    part = p.with_name(p.name + ".part")   # `.part` files are never copied off a local copy
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(part, format="PNG", compress_level=1)
        os.replace(part, p)
    except Exception:  # noqa: BLE001  (a draft is only for the page)
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass


def clear_drafts(run: Path | str, *rels: str) -> None:
    """Remove the drafts named by ``rels``, or every draft. Like `save_draft`, it never stops the stage."""
    import shutil

    base = Path(run) / DRAFTS
    if not rels:
        shutil.rmtree(base, ignore_errors=True)
    for rel in rels:
        try:
            (base / rel).unlink(missing_ok=True)
        except OSError:
            pass
