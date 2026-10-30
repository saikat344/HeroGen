"""Frozen versions of a body, read from a versions folder (`--versions DIR`); read-only like everything in the lab.

A version is a folder holding `version.json` with the body's files, either at `DIR/<group>/<version>/` (one group
per sketch, as `freeze` writes them) or at `DIR/<version>/`. `version.json`:

```json
{"label": "v0-carve", "milestone": "...", "mesh_source": "carve", "joints": "silhouette",
 "profile": "roblox-ingame budgets 24000/6000", "built": "2026-10-28T21:58:35-07:00", "code": "...",
 "provenance": "...", "sketch_run": "pencil-2", "files_sha256": {"high.glb": "..."}, "frozen": "2026-10-29",
 "immutable": true}
```

`sketch_run` names the run folder the version was built from; the lab shows a version with the character whose
runs include a run of that name. Scores come from a frozen score file next to the versions (`05-scores*.json` in
the group folder or the version folder): the one body whose mesh source and triangle count both match the
version's own high GLB, when exactly one does. Each listed file's SHA-256 is checked against the file on disk.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from herogen.lab.runindex import (SAFE_NAME, _dict, _num, _read_json, _sections_summary, _text, file_entry,
                                  glb_summary, read_scores, stage_summary)

VERSION_FILE = "version.json"
VERSION_FILE_TYPES = (".glb", ".png", ".json")

_hash_lock = threading.Lock()
_hash_cache: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str | None:
    """SHA-256 of a file, cached by path, size and modification time; None when it cannot be read."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _hash_lock:
        if key in _hash_cache:
            return _hash_cache[key]
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    digest = h.hexdigest()
    with _hash_lock:
        _hash_cache[key] = digest
    return digest


def version_dirs(root: Path | None) -> list[tuple[str, Path]]:
    """(id, folder) of every version under `root`: `<group>/<version>` or `<version>`, plain names only."""
    if root is None:
        return []
    out = []
    try:
        top = sorted(p for p in root.iterdir() if SAFE_NAME.match(p.name) and not p.is_symlink() and p.is_dir())
    except OSError:
        return []
    for a in top:
        if (a / VERSION_FILE).is_file():
            out.append((a.name, a))
            continue
        try:
            inner = sorted(p for p in a.iterdir() if SAFE_NAME.match(p.name) and not p.is_symlink() and p.is_dir())
        except OSError:
            continue
        out += [(f"{a.name}/{b.name}", b) for b in inner if (b / VERSION_FILE).is_file()]
    return out


def version_dir(root: Path | None, version_id: str) -> Path | None:
    """The folder of `version_id` inside `root`, or None when the id is not one of its versions."""
    return dict(version_dirs(root)).get(version_id)


def _integrity(folder: Path, listed: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name, want in sorted(listed.items()):
        if not isinstance(name, str) or not SAFE_NAME.match(name) or not isinstance(want, str):
            continue
        got = file_sha256(folder / name) if (folder / name).is_file() else None
        out.append({"file": name, "present": got is not None, "matches": got == want.lower() if got else None})
    return out


def _match_scores(folder: Path, mesh_source: str | None, triangles: float | None) -> dict[str, Any] | None:
    """The frozen score row for this version, or None when no score file has exactly one matching body."""
    if mesh_source is None or triangles is None:
        return None
    files = []
    for d in (folder, folder.parent):
        try:
            files += sorted(p for p in d.glob("05-scores*.json") if p.is_file() and not p.is_symlink())
        except OSError:
            continue
    for f in files:
        sc = read_scores(f, f.parent)
        if not sc or sc.get("error"):
            continue
        hits = [b for b in sc["bodies"] if not b["variant_of"] and b["mesh_source"] == mesh_source
                and b["triangles"] == triangles]
        if len(hits) == 1:
            body = {**hits[0], "renders": {}, "heldout_render": None}  # renders live in the original run
            return {"file": f.name, "where": "group" if f.parent == folder.parent else "version", "body": body,
                    "views": sc["views"], "view_info": sc["view_info"], "heldout_views": sc["heldout_views"],
                    "matched_by": "mesh source and high-LOD triangle count"}
    return None


def version_summary(version_id: str, folder: Path) -> dict[str, Any]:
    """A version as the page lists and compares it. A `version.json` that cannot be read is said so, not hidden."""
    rec, err = _read_json(folder / VERSION_FILE)
    if err or not isinstance(rec, dict):
        return {"id": version_id, "label": version_id.split("/")[-1], "error": err or "version.json is not an object"}
    high = "high.glb" if (folder / "high.glb").is_file() else None
    low = "low.glb" if (folder / "low.glb").is_file() else None
    glb = {lod: glb_summary(folder / rel) for lod, rel in (("high", high), ("low", low)) if rel}
    triangles = _num(glb["high"].get("triangles")) if "high" in glb else None
    mesh_source = rec["mesh_source"] if isinstance(rec.get("mesh_source"), str) else None
    frozen = rec.get("frozen")
    return {
        "id": version_id,
        "label": _text(rec.get("label")) or version_id.split("/")[-1],
        "milestone": _text(rec.get("milestone")) or None,
        "mesh_source": mesh_source,
        "joints": _text(rec.get("joints")) or None,
        "profile": _text(rec.get("profile")) or None,
        "built": rec["built"] if isinstance(rec.get("built"), str) else None,
        "code": _text(rec.get("code")) or None,
        "provenance": _text(rec.get("provenance")) or None,
        "sketch_run": rec["sketch_run"] if isinstance(rec.get("sketch_run"), str) else None,
        "frozen": frozen if isinstance(frozen, str) else None,
        "immutable": rec.get("immutable") is True,
        "high": high,
        "low": low,
        "glb": glb,
        "triangles": triangles,
        "triangles_low": _num(glb["low"].get("triangles")) if "low" in glb else None,
        "files": [e for e in (file_entry(folder, n) for n in ("high.glb", "low.glb", "atlas-high.png",
                                                              "atlas-low.png", "sections.json")) if e],
        "integrity": _integrity(folder, _dict(rec.get("files_sha256"))),
        "sections": _sections_summary(folder),
        "stage4": stage_summary(folder, "4", "game body"),
        "scores": _match_scores(folder, mesh_source, triangles),
    }


def safe_version_file(root: Path | None, url_path: str) -> Path | None:
    """The file `/versions/<id>/<name>` names (the id one or two plain segments, the file directly in the version
    folder and of a type in VERSION_FILE_TYPES), resolved inside that folder; None when refused."""
    if root is None or not url_path or "\x00" in url_path or "\\" in url_path:
        return None
    parts = url_path.split("/")
    if len(parts) not in (2, 3) or any(not SAFE_NAME.match(p) for p in parts):
        return None
    folder = version_dir(root, "/".join(parts[:-1]))
    if folder is None or not parts[-1].lower().endswith(VERSION_FILE_TYPES):
        return None
    try:
        resolved = (folder / parts[-1]).resolve(strict=True)
        inside = resolved.parent == folder.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if inside and resolved.is_file() else None
