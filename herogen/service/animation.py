"""The website's animation preview: the body GLB and its R15 joint pivots, for the 3D viewer's Idle, Walk, Wave
and Jump buttons.

Stage 4 writes `04-body/high.glb` (one mesh node per R15 section, each translated to its pivot, no hierarchy)
and `04-body/sections.json` (each section's parent, joint and pivot). Stage 6 puts every Motor6D and RigAttachment
of the Roblox rig exactly on those pivots and checks it (`attachments_at_pivots`), so the pivots here are the
joints the Roblox rig turns about. They are converted from the contract frame (+Z up, faces -Y) to the GLB
frame (+Y up, faces +Z): G = (Bx, Bz, -By).

Only reads two small files; nothing is computed with a model. The result is cached per file version.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from herogen.blender import PARENT, PIVOT_JOINT, SECTIONS
from herogen.service.bundle import _inside, complete_glb

BODY_GLB = "04-body/high.glb"
SECTIONS_FILE = "04-body/sections.json"
FRAME = "glTF: +Y up, faces +Z, the character's left is +X; studs"
# Root first, then every part after its parent: the order the viewer builds the joint tree in.
ORDER = ("LowerTorso", "UpperTorso", "Head",
         "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm", "RightLowerArm", "RightHand",
         "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot")
MAX_SECTIONS_BYTES = 4 * 1024 * 1024
MAX_COORD = 1000.0
_CACHE: OrderedDict[tuple, dict[str, Any] | None] = OrderedDict()
_CACHE_SIZE = 256
_LOCK = threading.Lock()


def _key(path: Path) -> tuple:
    st = path.stat()
    return str(path), st.st_mtime_ns, st.st_size


def body_joints(run_dir: Path) -> dict[str, Any] | None:
    """{"glb": run-relative body GLB, "frame": ..., "parts": [{"part", "parent", "joint", "pivot"}]} in GLB
    coordinates, or None when the body or its sections are missing, incomplete or not the 15 R15 parts."""
    glb = _inside(run_dir, BODY_GLB)
    sections = _inside(run_dir, SECTIONS_FILE)
    if glb is None or sections is None:
        return None
    try:
        key = (_key(glb), _key(sections))
    except OSError:
        return None
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    value = _read(glb, sections)
    with _LOCK:
        _CACHE[key] = value
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return value


def _read(glb: Path, sections: Path) -> dict[str, Any] | None:
    try:
        if sections.stat().st_size > MAX_SECTIONS_BYTES or not complete_glb(glb):
            return None
        doc = json.loads(sections.read_text(encoding="utf-8"))
        names = _node_names(glb)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("units") != "studs":
        return None
    frame = str(doc.get("frame", ""))
    if "+Z up" not in frame or "faces -Y" not in frame:
        return None
    by_name = {s.get("name"): s for s in doc.get("sections") or [] if isinstance(s, dict)}
    if set(by_name) != set(SECTIONS) or not set(SECTIONS) <= names:
        return None
    parts = []
    for name in ORDER:
        s = by_name[name]
        if s.get("parent") != PARENT[name] or s.get("joint") != PIVOT_JOINT[name]:
            return None
        pivot = s.get("pivot")
        if not isinstance(pivot, list) or len(pivot) != 3:
            return None
        try:
            bx, by, bz = (float(v) for v in pivot)
        except (TypeError, ValueError):
            return None
        g = (bx, bz, -by)
        if not all(math.isfinite(v) and abs(v) <= MAX_COORD for v in g):
            return None
        parts.append({"part": name, "parent": PARENT[name], "joint": PIVOT_JOINT[name],
                      "pivot": [round(v + 0.0, 5) for v in g]})
    return {"glb": BODY_GLB, "frame": FRAME, "parts": parts}


def _node_names(glb: Path) -> set[str]:
    """The names of the GLB's mesh nodes, from its JSON chunk (the header was checked by complete_glb)."""
    with glb.open("rb") as f:
        head = f.read(20)
        doc = json.loads(f.read(int.from_bytes(head[12:16], "little")))
    return {n.get("name") for n in doc.get("nodes") or [] if isinstance(n, dict) and "mesh" in n}
