"""Independent run versions, copied from passing stages of a read-only parent."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from herogen.pipeline.runfolder import RunFolder


def clean_note(note: object = "") -> str:
    """An optional, bounded character-change slot, using the character brief's text rules."""
    if note == "":
        return ""
    from herogen.pipeline.prompt_sketch import PromptRejected, clean_user_text

    try:
        return clean_user_text(note, max_chars=300, delimiters=("«", "»"))
    except PromptRejected as e:
        raise ValueError("Describe changes to the character's look in words, using at most 300 characters.") from e


def _source(parent: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or not p.parts or any(x in (".", "..") for x in p.parts) or "\\" in rel:
        raise ValueError("The earlier version has an invalid output path.")
    src = parent / p
    if any(x.is_symlink() for x in (parent, *[parent.joinpath(*p.parts[:i]) for i in range(1, len(p.parts) + 1)])):
        raise ValueError("The earlier version has a linked output.")
    if not src.is_file():
        raise ValueError("The earlier version is missing an output. Choose another version.")
    return src


def fork_run(root: Path, run_id: str, parent: Path, from_stage: int, note: object = "") -> RunFolder:
    """Publish a complete copy atomically into a new, absent folder; never open the parent for writing.

    Earlier records retain their measurements and historical calls, with a zero cost for this version and
    ``reused_from`` provenance. A cover redo edits a copied parent cover, still checked against the sketch.
    """
    if from_stage not in (1, 2):
        raise ValueError("Choose cover or turnaround to redo.")
    note = clean_note(note)
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise ValueError("Choose a new run id.")
    root, parent = Path(root), Path(parent)
    if root.resolve().is_relative_to(parent.resolve()):
        raise ValueError("The new run must be outside the earlier version.")
    dest = root / run_id
    if dest.exists() or dest.is_symlink():
        raise ValueError("That version already exists. Choose a new run id.")
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("The earlier version could not be found.")
    records = []
    for stage in ("p", *range(from_stage)):
        path = parent / f"stage-{stage}.json"
        if stage == "p" and not path.exists():
            continue
        rec = json.loads(_source(parent, path.name).read_text())
        if (not isinstance(rec, dict) or not isinstance(rec.get("outputs", []), list)
                or any(not isinstance(rel, str) for rel in rec.get("outputs", []))
                or not isinstance(rec.get("checks", []), list)
                or any(not isinstance(c, dict) for c in rec.get("checks", []))):
            raise ValueError("The earlier version has an invalid stage record.")
        if rec.get("stage") != stage or rec.get("ok") is not True or any(c.get("gate", True) and c.get("pass") is False
                                            for c in rec.get("checks", [])):
            raise ValueError("The earlier stages must pass their checks before this step can be redone.")
        records.append(rec)
    lineage = {"parent_run": parent.name, "from_stage": from_stage, "note": note}
    files = {rel for rec in records for rel in rec.get("outputs", [])}
    files.update(("00-sketch.png", "01-cover.png"))
    # For a cover redo, the old cover is an editing reference, not a passing stage record.
    sources = {rel: _source(parent, rel) for rel in files}
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".redo-", dir=root) as temp:
        temp_root = Path(temp)
        run = RunFolder(temp_root, run_id)
        for rel, src in sources.items():
            target = "redo/parent-cover.png" if from_stage == 1 and rel == "01-cover.png" else rel
            shutil.copyfile(src, run.file(target))
        run.write_json("redo.json", lineage)
        for rec in records:
            run.write_stage({**rec, "reused_from": parent.name, "original_cost_usd": rec.get("cost_usd", 0),
                             "cost_usd": 0, "seconds": 0, **lineage})
        # Reserve the destination exclusively before publishing the complete staged copy.
        dest.mkdir()
        try:
            os.replace(run.path, dest)
        except BaseException:
            dest.rmdir()
            raise
    return RunFolder(root, run_id)


def note_for(run: RunFolder, stage: int) -> str:
    if not run.exists("redo.json"):
        return ""
    meta = run.read_json("redo.json")
    return clean_note(meta.get("note", "")) if meta.get("from_stage") == stage else ""
