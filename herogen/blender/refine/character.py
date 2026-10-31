"""Per-character setup: bounding box, head location (SAM 3.1 on a front render), camera set, texture prep.

The head is found by SAM 3.1 `face` and `head` on a flat front render of SAM's mesh. Its centre's x and z come
from the head box; its y is the middle of the mesh vertices inside the head box (so profile cameras turn about the
skull, not the face surface). The head sphere that gets extra texels while unwrapping has the head box's larger
half-side as radius.
Without a detected head, setup keeps the body bounds and omits head cameras and extra head texel allocation.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from herogen.blender.glb import load_glb

from .cameras import camera, decode_pos, read_exr
from .backend import blender

BG = 0.45            # flat grey behind every render sent for editing
BODY_RES = (1024, 1536)
HEAD_RES = (1024, 1024)
HEAD_FRAME = 1.30    # head close-up ortho = this x the head box's larger side
BODY_FRAME = 1.06
TEX = 4096
HEAD_GAIN = 2.0
HEAD_ANGLES = (0, 40, -40, 90, -90, 180, 135, -135)
UPPER_ANGLES = (180, 135, -135)   # upper-body close-ups of the back: head top to UPPER_FRAME of the height
UPPER_FRAME = 0.55
BODY_ANGLES = (0, 45, -45, 90, -90, 180)


def blender_points(glb):
    meshes = load_glb(glb).meshes
    if not meshes:
        raise ValueError("the source GLB contains no triangle meshes")
    return np.concatenate([mesh.verts for mesh in meshes])


def composite(rgba, bg=BG):
    a = np.asarray(rgba, dtype=np.float64) / 255
    rgb = a[..., :3] * a[..., 3:] + bg * (1 - a[..., 3:])
    return Image.fromarray((rgb * 255).round().astype(np.uint8)), a[..., 3]


def setup(glb, work, client, timeout=900.0):
    work = Path(work)
    info_path = work / "character.json"
    pts = blender_points(glb)
    lo, hi = pts.min(0), pts.max(0)
    H = hi[2] - lo[2]
    if not np.isfinite(pts).all() or H <= 0:
        raise ValueError("the source mesh has invalid bounds")
    centre = (lo + hi) / 2
    body0 = camera("probe0", centre, 0, H * BODY_FRAME, BODY_RES)
    blender("probe", {"glb": str(glb), "cams": [body0], "out": str(work / "probe")}, work / "blender.log", timeout=timeout)
    with Image.open(work / "probe" / "probe0.png") as rgba:
        img, alpha = composite(rgba.convert("RGBA"))
    if not (alpha > 0.5).any():
        raise ValueError("the front render is empty")
    img.save(work / "probe" / "probe0-grey.png")
    boxes = {}
    for text in ("face", "head"):
        ins = client.segment(img, text)
        if ins:
            best = max(ins, key=lambda i: i.confidence or 0)
            boxes[text] = list(best.box)
    info = {"lo": lo.tolist(), "hi": hi.tolist(), "height": float(H),
            "body_centre": centre.tolist(), "boxes_px": boxes, "head_centre": None, "head_side": None}
    if "head" in boxes:
        info.update(locate_head(pts, centre, body0, boxes["head"]))
    info_path.write_text(json.dumps(info, indent=1))
    return info


def locate_head(pts, centre, body0, box):
    k = max(BODY_RES) / body0["ortho"]
    l, t, r, b = box
    x0 = centre[0] + (l - BODY_RES[0] / 2) / k
    x1 = centre[0] + (r - BODY_RES[0] / 2) / k
    z1 = centre[2] + (BODY_RES[1] / 2 - t) / k
    z0 = centre[2] + (BODY_RES[1] / 2 - b) / k
    inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 2] > z0) & (pts[:, 2] < z1)
    ys = pts[inside, 1]
    if not len(ys):
        raise ValueError("the detected head contains no mesh vertices")
    hc = [(x0 + x1) / 2, float((np.quantile(ys, 0.02) + np.quantile(ys, 0.98)) / 2), (z0 + z1) / 2]
    side = max(x1 - x0, z1 - z0)
    return {"head_centre": hc, "head_side": float(side)}


def cameras_for(info):
    H, c = info["height"], info["body_centre"]
    cams = []
    if info["head_centre"] is not None:
        cams = [camera(f"head{a:+d}", info["head_centre"], a, info["head_side"] * HEAD_FRAME, HEAD_RES)
                for a in HEAD_ANGLES]
    cams += [camera(f"body{a:+d}", c, a, H * BODY_FRAME, BODY_RES) for a in BODY_ANGLES]
    top = info["hi"][2]
    uc = [c[0], c[1], top - 0.5 * UPPER_FRAME * H]
    cams += [camera(f"upper{a:+d}", uc, a, H * UPPER_FRAME, HEAD_RES) for a in UPPER_ANGLES]
    return {cam["name"]: cam for cam in cams}


def prep(glb, info, work, timeout=900.0):
    work = Path(work)
    head = None
    if info["head_centre"] is not None:
        head = {"centre": info["head_centre"], "radius": info["head_side"] / 2, "gain": HEAD_GAIN}
    blender("prep", {"glb": str(glb), "out": str(work / "prep"), "tex": TEX, "head": head},
            work / "blender.log", timeout=timeout)
    return work / "prep"


def texel_maps(prep_dir):
    P, valid = decode_pos(read_exr(prep_dir / "pos.exr"))
    N = read_exr(prep_dir / "nrm.exr")[..., :3] * 2 - 1
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-6)
    cov = np.asarray(Image.open(prep_dir / "cov.png").convert("L")) > 127
    return P, N, cov & valid
