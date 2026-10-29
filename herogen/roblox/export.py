"""Stage 6: write the R15 rig, the Studio scripts and the user's README for a stage-4 body, check the rig
against the body, and record the result in `stage-6.json`.

The record goes where the other stages put theirs, at the run root, when the output folder is a run's
`06-roblox` (its parent holds `stage-4.json` or `04-body/`); for any other output folder it goes in that
folder."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from importlib import resources
from pathlib import Path

from . import (BIND_FILE, OUTPUT_FILES, README_FILE, RECORD_FILE, RIG_FILE, RIG_NAME, RIG_TEST_FILE,
               RobloxExportError)
from .check import check_rig
from .rbxmx import write_rbxmx
from .record import code_version, file_hash, write_atomic
from .rig import build_rig

STAGE = 6
STAGE_NAME = "roblox export"
OUT_DIR_NAME = "06-roblox"
TEMPLATE_FILES = (BIND_FILE, RIG_TEST_FILE, README_FILE)


def template(name: str) -> str:
    return (resources.files(__package__) / "templates" / name).read_text(encoding="utf-8")


def render(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        # a bug in this module, not in the input
        raise RuntimeError("a template placeholder was left unfilled")
    return text


def _rel(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:  # another drive on Windows
        return str(path)


def record_path(out: Path) -> Path:
    """Where the stage record goes for output folder `out` (see the module docstring)."""
    out = Path(out)
    run = out.parent
    if out.name == OUT_DIR_NAME and ((run / "stage-4.json").is_file() or (run / "04-body").is_dir()):
        return run / RECORD_FILE
    return out / RECORD_FILE


def run_export(body: Path, out: Path, glb_name: str = "high.glb") -> dict:
    """Export `body` (a stage-4 folder) into `out`. Returns the stage record, which is also written to
    `record_path(out)`. A body that can't be exported gives a record with ok false and an "error" line.
    Paths in the record are relative to the record's folder."""
    body, out = Path(body), Path(out)
    t0 = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    rec_path = record_path(out)
    base = rec_path.parent
    for name in OUTPUT_FILES:  # never leave an earlier run's rig next to this run's record
        (out / name).unlink(missing_ok=True)
    record = {
        "stage": STAGE, "name": STAGE_NAME, "ok": False,
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": None, "cost_usd": 0.0,
        "cost_basis": "local compute only (Python and numpy); nothing uploaded",
        "models": [], "tools": [], "code": code_version(),
        "body": _rel(body, base),
        "inputs": {n: file_hash(body / n) for n in ("sections.json", glb_name) if (body / n).is_file()},
        "outputs": [], "checks": [], "notes": "",
    }
    try:
        spec = build_rig(body, glb_name)
        write_atomic(out / RIG_FILE, write_rbxmx(spec))
        values = {
            "RIG_NAME": RIG_NAME, "RIG_FILE": RIG_FILE, "BIND_FILE": BIND_FILE,
            "RIG_TEST_FILE": RIG_TEST_FILE, "RECORD_PATH": _rel(rec_path, out),
            "GLB_FILE": glb_name, "GLB_PATH": _rel(body / glb_name, out),
            "IMPORTED_NAME": Path(glb_name).stem, "HIP_HEIGHT": f"{spec['hip_height']:.3f}",
        }
        for name in TEMPLATE_FILES:
            write_atomic(out / name, render(template(name), values))
        record["outputs"] = [_rel(out / n, base) for n in (RIG_FILE, *TEMPLATE_FILES)]
        record["checks"] = check_rig(body, out / RIG_FILE, glb_name)
        record["ok"] = all(c["pass"] for c in record["checks"] if c.get("gate", True))
        record["notes"] = "; ".join(spec["notes"])
        record["rig"] = {
            "name": spec["name"], "file": RIG_FILE, "frame": spec["frame"], "joints": spec["joints_kind"],
            "hip_height": spec["hip_height"], "hrp": spec["hrp"],
            "meshes": "MeshId left empty in every section; bind_meshes.luau binds the meshes Studio imports "
                      f"from {glb_name}",
            "parts": {p["name"]: {"center": p["center"], "size": p["size"], "pivot": p["pivot"],
                                  "triangles": p["triangles"]} for p in spec["parts"]},
        }
    except RobloxExportError as exc:
        for name in OUTPUT_FILES:
            (out / name).unlink(missing_ok=True)
        record["error"] = str(exc)
    record["seconds"] = round(time.monotonic() - t0, 2)
    write_atomic(rec_path, json.dumps(record, indent=2) + "\n")
    return record
