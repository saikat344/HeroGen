"""Synthetic stage-4 bodies for tests: one box per R15 section, written as `high.glb` and `sections.json` in
the layout stage 4 writes (contract frame, studs, one translated GLB node per section, its origin at the
pivot).

The figure is deliberately asymmetric (a bigger left hand, an off-centre body) so a left/right or front/back
mix-up changes the numbers instead of hiding behind symmetry.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import R15

# Contract frame B (+Z up, faces -Y, the character's left is +X): section -> (box centre, box size, pivot).
OFFSET = np.array([0.03, -0.02, 0.0])
BOXES = {
    "LowerTorso": ((0.0, 0.0, 2.55), (1.4, 0.8, 0.7), (0.0, 0.0, 2.7)),
    "UpperTorso": ((0.0, 0.02, 3.5), (1.6, 0.8, 1.2), (0.0, 0.0, 2.9)),
    "Head": ((0.0, -0.05, 4.6), (1.0, 1.0, 1.0), (0.0, 0.0, 4.1)),
    "LeftUpperArm": ((1.05, 0.0, 3.6), (0.5, 0.5, 1.0), (0.78, 0.0, 3.95)),
    "LeftLowerArm": ((1.05, 0.0, 2.7), (0.45, 0.45, 0.8), (1.05, 0.0, 3.1)),
    "LeftHand": ((1.05, 0.0, 2.08), (0.48, 0.42, 0.44), (1.05, 0.0, 2.3)),
    "RightUpperArm": ((-1.05, 0.0, 3.6), (0.5, 0.5, 1.0), (-0.78, 0.0, 3.95)),
    "RightLowerArm": ((-1.05, 0.0, 2.7), (0.45, 0.45, 0.8), (-1.05, 0.0, 3.1)),
    "RightHand": ((-1.05, 0.0, 2.1), (0.4, 0.4, 0.4), (-1.05, 0.0, 2.3)),
    "LeftUpperLeg": ((0.4, 0.0, 1.75), (0.6, 0.6, 0.9), (0.4, 0.0, 2.2)),
    "LeftLowerLeg": ((0.4, 0.0, 0.85), (0.5, 0.5, 0.9), (0.4, 0.0, 1.3)),
    "LeftFoot": ((0.4, -0.15, 0.2), (0.5, 0.9, 0.4), (0.4, 0.0, 0.4)),
    "RightUpperLeg": ((-0.4, 0.0, 1.75), (0.6, 0.6, 0.9), (-0.4, 0.0, 2.2)),
    "RightLowerLeg": ((-0.4, 0.0, 0.85), (0.5, 0.5, 0.9), (-0.4, 0.0, 1.3)),
    "RightFoot": ((-0.4, -0.15, 0.2), (0.5, 0.9, 0.4), (-0.4, 0.0, 0.4)),
}
TOES = {"LeftToe": (0.4, -0.6, 0.1), "RightToe": (-0.4, -0.6, 0.1)}
FRAME = ("Blender: +Z up, feet at z=0, faces -Y, x/y centred on the figure's bounding box; GLBs are exported "
         "Y-up (glTF), front along +Z")

_CORNERS = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
_BOX_TRIS = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.uint32)


def _b_to_g(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _turn(p, facing_back: bool) -> np.ndarray:
    """The figure turned 180 degrees about Z (facing +Y, against the contract) when facing_back."""
    p = np.asarray(p, dtype=np.float64)
    return p * np.array([-1.0, -1.0, 1.0]) if facing_back else p


def body(triangles: dict[str, int] | None = None, facing_back: bool = False) -> dict:
    """The synthetic body as data: per section the box corners (B), triangles and pivot (B).
    `triangles` repeats a section's 12 box triangles up to that count (same bounds, more triangles)."""
    triangles = triangles or {}
    out = {}
    for name, _, _ in R15:
        c, s, pv = BOXES[name]
        corners = _turn(np.asarray(c) + OFFSET + _CORNERS * np.asarray(s), facing_back)
        reps = max(1, -(-triangles.get(name, 12) // 12))
        tris = np.tile(_BOX_TRIS, (reps, 1))[: max(12, triangles.get(name, 12))]
        out[name] = {"corners": corners, "tris": tris, "pivot": _turn(np.asarray(pv) + OFFSET, facing_back)}
    return out


def write_body(folder: Path, triangles: dict[str, int] | None = None, facing_back: bool = False,
               glb_name: str = "high.glb") -> Path:
    """Write `high.glb` and `sections.json` for the synthetic body into `folder`; returns the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = body(triangles, facing_back)
    blob, views, accessors, meshes, nodes = bytearray(), [], [], [], []
    for i, (name, _, _) in enumerate(R15):
        d = data[name]
        origin = _b_to_g(d["pivot"])
        pos = (_b_to_g(d["corners"]) - origin).astype("<f4")
        idx = d["tris"].astype("<u4").ravel()
        for arr, target in ((pos, 34962), (idx, 34963)):
            while len(blob) % 4:
                blob.append(0)
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes, "target": target})
            blob += arr.tobytes()
        accessors.append({"bufferView": 2 * i, "componentType": 5126, "count": len(pos), "type": "VEC3",
                          "min": pos.min(0).astype(float).tolist(), "max": pos.max(0).astype(float).tolist()})
        accessors.append({"bufferView": 2 * i + 1, "componentType": 5125, "count": len(idx),
                          "type": "SCALAR"})
        meshes.append({"name": name,
                       "primitives": [{"attributes": {"POSITION": 2 * i}, "indices": 2 * i + 1}]})
        nodes.append({"name": name, "mesh": i, "translation": origin.tolist()})
    while len(blob) % 4:
        blob.append(0)
    doc = {"asset": {"version": "2.0", "generator": "herogen.roblox.synth"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob))
    glb += struct.pack("<II", len(js), 0x4E4F534A) + js
    glb += struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    (folder / glb_name).write_bytes(glb)

    sections = []
    for name, parent, joint in R15:
        d = data[name]
        sections.append({"name": name, "parent": None if parent == "HumanoidRootPart" else parent,
                         "joint": joint, "pivot": [round(float(v), 4) for v in d["pivot"]],
                         "triangles_high": int(len(d["tris"])),
                         "bounds_min": [round(float(v), 4) for v in d["corners"].min(0)],
                         "bounds_max": [round(float(v), 4) for v in d["corners"].max(0)]})
    sj = {"units": "studs", "height": 5.5, "frame": FRAME, "sections": sections,
          "ends": {k: [round(float(v), 4) for v in _turn(np.asarray(p) + OFFSET, facing_back)]
                   for k, p in TOES.items()}}
    (folder / "sections.json").write_text(json.dumps(sj, indent=1))
    return folder


def edit_glb_json(path: Path, change) -> None:
    """Rewrite a GLB's JSON chunk in place with `change(doc)`, keeping its BIN chunk (to corrupt test
    inputs)."""
    path = Path(path)
    data = path.read_bytes()
    jlen = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + jlen])
    change(doc)
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    rest = struct.pack("<II", len(js), 0x4E4F534A) + js + data[20 + jlen:]
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(rest)) + rest)
