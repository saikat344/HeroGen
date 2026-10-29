"""Downloads: the body's GLB, and the Roblox bundle (a zip of the stage 6 rig, its Studio scripts and README, the
stage record and the GLB the scripts bind).

The bundle takes exactly the files `stage-6.json` lists in its `outputs` (paths relative to the record, which the
export writes at the run root), and only when that record passed: each must be a plain run-relative path, resolve
inside the run folder with no link along the way, and be one of the rig's file types. The zip keeps the run's own
layout (`06-roblox/...`, `04-body/high.glb`, `stage-6.json`), because the README the export writes refers to the
GLB and the record by those relative paths. Names inside the zip come from the record and fixed names, never from
a request.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from herogen.service.storage import SAFE_SEGMENT

BUNDLE_TYPES = (".rbxmx", ".luau", ".lua", ".txt", ".md")
MAX_BUNDLE_FILE = 64 * 1024 * 1024
ZIP_DATE = (2026, 1, 1, 0, 0, 0)
LODS = ("high", "low")
RECORD = "stage-6.json"
GLB = "04-body/high.glb"
FALLBACK_README = """HeroGen Roblox bundle

04-body/high.glb is the hero's body. The .rbxmx file in 06-roblox is the R15 rig and bind_meshes.luau binds the
meshes Studio imports from high.glb onto it. This bundle's stage did not include its own README; see the
comments at the top of bind_meshes.luau for the steps.
"""


class BundleUnavailable(Exception):
    """The download does not exist for this job; the message says why, for the user."""


def glb_path(run_dir: Path, lod: str = "high") -> Path | None:
    if lod not in LODS:
        return None
    p = run_dir / "04-body" / f"{lod}.glb"
    try:
        if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(run_dir.resolve()):
            return p
    except OSError:
        return None
    return None


def _inside(run_dir: Path, rel: object) -> Path | None:
    if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel or "\x00" in rel:
        return None
    parts = rel.split("/")
    if any(not SAFE_SEGMENT.match(p) or ".." in p for p in parts):
        return None
    try:
        base = run_dir.resolve(strict=True)
        target = base.joinpath(*parts)
        resolved = target.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if resolved != target or not resolved.is_file():
        return None
    return resolved


def roblox_files(run_dir: Path) -> list[tuple[Path, str]]:
    """The rig files stage 6 made, as (file, run-relative name), or BundleUnavailable."""
    try:
        rec = json.loads((run_dir / RECORD).read_text())
    except (OSError, ValueError):
        raise BundleUnavailable("The Roblox rig has not been built for this hero.") from None
    if not isinstance(rec, dict) or rec.get("ok") is not True:
        raise BundleUnavailable("The Roblox rig did not pass its checks for this hero.")
    files = []
    for rel in rec.get("outputs") or []:
        p = _inside(run_dir, rel)
        if p is not None and p.suffix.lower() in BUNDLE_TYPES and p.stat().st_size <= MAX_BUNDLE_FILE:
            files.append((p, rel))
    if not any(p.suffix.lower() == ".rbxmx" for p, _ in files) or not any(
            p.name == "bind_meshes.luau" for p, _ in files):
        raise BundleUnavailable("The Roblox rig's files are incomplete for this hero.")
    return files


def roblox_readme(run_dir: Path) -> str | None:
    try:
        files = roblox_files(run_dir)
    except BundleUnavailable:
        return None
    for p, _ in files:
        if p.name.lower() in ("readme.txt", "readme.md"):
            return p.read_text(encoding="utf-8", errors="replace")[:20000]
    return None


def roblox_bundle(run_dir: Path) -> bytes:
    files = roblox_files(run_dir)
    glb = glb_path(run_dir, "high")
    if glb is None:
        raise BundleUnavailable("The body's GLB is missing for this hero.")
    buf = io.BytesIO()
    names: set[str] = set()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        def add(name: str, data: bytes) -> None:
            if name in names:
                return
            info = zipfile.ZipInfo(name, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
            names.add(name)

        for p, rel in files:
            add(rel, p.read_bytes())
        add(GLB, glb.read_bytes())
        add(RECORD, (run_dir / RECORD).read_bytes())
        if not any(n.rsplit("/", 1)[-1].lower().startswith("readme") for n in names):
            add("README.txt", FALLBACK_README.encode())
    return buf.getvalue()
