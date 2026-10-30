"""Synthetic test input: a humanoid of capsules, ellipsoids and rounded boxes in A-pose.

Masks are rendered analytically at the contract's orthographic cameras (the projection of a capsule is a
stadium, of an axis-aligned ellipsoid an ellipse, of an axis-aligned rounded box a rounded rectangle), so the
silhouettes are exact and the true joints are known. Units are studs; the figure stands 5.5 studs tall,
faces -Y, feet at Z = 0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .views import PROJ, save_mask


@dataclass
class Capsule:
    a: tuple
    b: tuple
    r: float


@dataclass
class Ellipsoid:
    c: tuple
    radii: tuple


@dataclass
class RBox:
    c: tuple
    half: tuple
    round: float


@dataclass
class Figure:
    parts: list = field(default_factory=list)
    joints: dict = field(default_factory=dict)


def humanoid(arm_angle_deg: float = 50.0, head_scale: float = 1.0, leg_spread: float = 0.02,
             elbow_bend_deg: float = 0.0, clothed: bool = False) -> Figure:
    """A-pose humanoid. `arm_angle_deg` is the arm's angle from vertical; `elbow_bend_deg` bends the
    forearm forward (towards -Y) about the elbow; `clothed` adds hair, a neck scarf, a backpack and a hanging
    scarf strip (same joints)."""
    f = Figure()
    hs = head_scale
    head_r = (0.42 * hs, 0.45 * hs, 0.52 * hs)
    head_c = (0.0, 0.0, 5.5 - head_r[2])
    f.parts.append(Ellipsoid(head_c, head_r))
    neck_lo, neck_hi = 4.30, head_c[2] - head_r[2] + 0.06
    f.parts.append(Capsule((0, 0, neck_lo), (0, 0, neck_hi), 0.17))
    f.parts.append(RBox((0, 0, 3.825), (0.62, 0.32, 0.475), 0.12))     # chest, z 3.35..4.30
    f.parts.append(RBox((0, 0, 3.225), (0.50, 0.28, 0.225), 0.10))     # waist, z 3.00..3.45
    f.parts.append(RBox((0, 0, 2.845), (0.56, 0.30, 0.255), 0.12))     # pelvis, z 2.59..3.10
    # Truth joints are the features the estimator is meant to find: the neck and waist joints at the centre
    # of the narrow band between head and chest / chest and pelvis, the limb joints at the capsule ends.
    f.joints["HeadTop"] = (0.0, 0.0, 5.5)
    f.joints["Neck"] = (0.0, 0.0, (neck_lo + head_c[2] - head_r[2]) / 2)
    f.joints["Waist"] = (0.0, 0.0, (3.10 + 3.35) / 2)

    a = math.radians(arm_angle_deg)
    lens = (0.85, 0.75, 0.45)
    radii = (0.17, 0.14, 0.12)
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        sh = np.array([0.72 * sx, 0.0, 4.12])
        d = np.array([math.sin(a) * sx, 0.0, -math.cos(a)])
        el = sh + lens[0] * d
        b = math.radians(elbow_bend_deg)
        d2 = d * math.cos(b) + np.array([0.0, -1.0, 0.0]) * math.sin(b)
        wr = el + lens[1] * d2
        tip = wr + lens[2] * d2
        f.parts += [Capsule(tuple(sh), tuple(el), radii[0]), Capsule(tuple(el), tuple(wr), radii[1]),
                    Capsule(tuple(wr), tuple(tip), radii[2])]
        f.joints[f"{side}Shoulder"] = tuple(sh)
        f.joints[f"{side}Elbow"] = tuple(el)
        f.joints[f"{side}Wrist"] = tuple(wr)

        hip = np.array([0.28 * sx, 0.0, 2.70])
        knee = np.array([(0.30 + leg_spread) * sx, 0.0, 1.45])
        ankle = np.array([(0.31 + 2 * leg_spread) * sx, 0.0, 0.30])
        f.parts += [Capsule(tuple(hip), tuple(knee), 0.22), Capsule(tuple(knee), tuple(ankle), 0.17),
                    RBox((ankle[0], -0.18, 0.13), (0.14, 0.32, 0.13), 0.06)]
        f.joints[f"{side}Hip"] = tuple(hip)
        f.joints[f"{side}Knee"] = tuple(knee)
        f.joints[f"{side}Ankle"] = tuple(ankle)
    if clothed:
        # Clothing that hides the silhouette cues without moving any joint: big hair (wider than the head, top
        # still at 5.5), a thick scarf wrapped over the neck up to the chin and spreading onto the shoulders
        # (so the outline widens row by row from chin to shoulders, with no neck narrowing), a backpack, and a
        # scarf tail hanging behind the legs, clear of the body in the side view.
        f.parts.append(Ellipsoid((0.0, 0.07, 5.5 - 0.40 * hs), (0.52 * hs, 0.55 * hs, 0.40 * hs)))
        f.parts.append(Ellipsoid((0.0, -0.02, 4.15), (0.80, 0.38, 0.42)))
        f.parts.append(RBox((0.0, 0.50, 3.75), (0.40, 0.18, 0.42), 0.08))
        f.parts.append(RBox((0.12, 0.43, 2.65), (0.08, 0.03, 0.75), 0.02))   # scarf tail behind, z 1.9-3.4
    lh, rh = np.array(f.joints["LeftHip"]), np.array(f.joints["RightHip"])
    f.joints["Root"] = tuple(((lh + rh) / 2 + np.array(f.joints["Waist"])) / 2)   # LowerTorso centre (R15 root)
    return f


def _project(part, view: str):
    p = PROJ[view]
    if isinstance(part, Capsule):
        return ("capsule", p @ np.array(part.a, float), p @ np.array(part.b, float), part.r)
    if isinstance(part, Ellipsoid):
        return ("ellipse", p @ np.array(part.c, float), np.abs(p) @ np.array(part.radii, float))
    return ("rrect", p @ np.array(part.c, float), np.abs(p) @ np.array(part.half, float), part.round)


def _bounds(prim) -> tuple[np.ndarray, np.ndarray]:
    kind = prim[0]
    if kind == "capsule":
        lo = np.minimum(prim[1], prim[2]) - prim[3]
        hi = np.maximum(prim[1], prim[2]) + prim[3]
        return lo, hi
    return prim[1] - prim[2], prim[1] + prim[2]


def _inside(prim, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    kind = prim[0]
    if kind == "capsule":
        a, b, r = prim[1], prim[2], prim[3]
        ab = b - a
        t = ((u - a[0]) * ab[0] + (v - a[1]) * ab[1]) / max(float(ab @ ab), 1e-12)
        t = np.clip(t, 0.0, 1.0)
        du, dv = u - (a[0] + t * ab[0]), v - (a[1] + t * ab[1])
        return du * du + dv * dv <= r * r
    if kind == "ellipse":
        c, rr = prim[1], prim[2]
        return ((u - c[0]) / rr[0]) ** 2 + ((v - c[1]) / rr[1]) ** 2 <= 1.0
    c, h, ro = prim[1], prim[2], prim[3]
    qu = np.maximum(np.abs(u - c[0]) - (h[0] - ro), 0.0)
    qv = np.maximum(np.abs(v - c[1]) - (h[1] - ro), 0.0)
    return qu * qu + qv * qv <= ro * ro


def render_masks(fig: Figure, size: int = 512, ppu: float = 80.0) -> tuple[dict, dict]:
    """Masks for the five views, each centred on the figure's bounding box. Returns (masks, centres)."""
    masks, centres = {}, {}
    for view in PROJ:
        prims = [_project(p, view) for p in fig.parts]
        lo = np.min([_bounds(p)[0] for p in prims], axis=0)
        hi = np.max([_bounds(p)[1] for p in prims], axis=0)
        u0, v0 = (lo + hi) / 2
        idx = (np.arange(size) + 0.5 - size / 2) / ppu
        uu, vv = np.meshgrid(u0 + idx, v0 - idx)
        m = np.zeros((size, size), dtype=bool)
        for p in prims:
            m |= _inside(p, uu, vv)
        masks[view] = m
        centres[view] = (float(u0), float(v0))
    return masks, centres


PALETTE = np.array([[200, 60, 50], [235, 190, 60], [60, 150, 80], [50, 100, 190], [150, 70, 170],
                    [240, 240, 235]], dtype=np.float64)
BACKGROUND = 225.0


def paint(p: np.ndarray) -> np.ndarray:
    """The synthetic figure's known surface colour: bands in z, x and y over a six-colour palette."""
    idx = (np.floor(p[:, 2] / 0.4) + 2 * np.floor((p[:, 0] + 4) / 0.6) + 3 * np.floor((p[:, 1] + 4) / 0.35))
    return PALETTE[idx.astype(np.int64) % len(PALETTE)]


def sdf(fig: Figure, p: np.ndarray) -> np.ndarray:
    d = np.full(len(p), np.inf)
    for part in fig.parts:
        if isinstance(part, Capsule):
            a, b = np.array(part.a, float), np.array(part.b, float)
            ab = b - a
            t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
            dd = np.linalg.norm(p - (a + t[:, None] * ab), axis=1) - part.r
        elif isinstance(part, Ellipsoid):
            r = np.array(part.radii, float)
            q = p - np.array(part.c, float)
            k0 = np.linalg.norm(q / r, axis=1)
            k1 = np.linalg.norm(q / (r * r), axis=1)
            dd = k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)
        else:
            h = np.array(part.half, float) - part.round
            q = np.abs(p - np.array(part.c, float)) - h
            dd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - part.round
        d = np.minimum(d, dd)
    return d


def render_views(fig: Figure, masks: dict, centres: dict, size: int, ppu: float) -> dict:
    """Colour views for front/back/left/right by sphere tracing the figure's SDF inside each mask."""
    out = {}
    origin_dir = {  # world point of image (u, v) at depth 0, and the camera's looking direction
        "front": (lambda u, v: np.stack([u, np.zeros_like(u), v], 1), np.array([0, 1.0, 0])),
        "back": (lambda u, v: np.stack([-u, np.zeros_like(u), v], 1), np.array([0, -1.0, 0])),
        "left": (lambda u, v: np.stack([np.zeros_like(u), u, v], 1), np.array([-1.0, 0, 0])),
        "right": (lambda u, v: np.stack([np.zeros_like(u), -u, v], 1), np.array([1.0, 0, 0])),
    }
    idx = (np.arange(size) + 0.5 - size / 2) / ppu
    for view, (to_world, fwd) in origin_dir.items():
        img = np.full((size, size, 3), BACKGROUND)
        m = masks[view]
        rr, cc = np.nonzero(m)
        u0, v0 = centres[view]
        u, v = u0 + idx[cc], v0 - idx[rr]
        p = to_world(u, v) - fwd * 4.0
        for _ in range(160):
            d = sdf(fig, p)
            p = p + fwd * np.maximum(d, 2e-4)[:, None]
        img[rr, cc] = paint(p)
        out[view] = img.astype(np.uint8)
    return out


def write_run(run: Path, fig: Figure | None = None, size: int = 512, ppu: float = 80.0) -> dict:
    """Write a synthetic stage-2 output (masks, colour views painted with `paint`, meta), front reference sketch and
    `synthetic/truth.json`."""
    fig = fig or humanoid()
    masks, centres = render_masks(fig, size, ppu)
    for sub in ("masks", "views", "synthetic"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    from PIL import Image

    colours = render_views(fig, masks, centres, size, ppu)
    for view, m in masks.items():
        save_mask(run / "masks" / f"{view}.png", m)
        if view in colours:
            Image.fromarray(colours[view]).save(run / "views" / f"{view}.png")
        else:
            save_mask(run / "views" / f"{view}.png", m)
    Image.fromarray(colours["front"]).save(run / "00-sketch.png")
    (run / "views" / "meta.json").write_text(json.dumps(
        {"pixels_per_unit": ppu, "height_units": 5.5, "image_size": size}, indent=2))
    # The pipeline's frame is centred on the figure's X/Y bounding box (see views.py); shift the truth into it.
    shift = np.array([centres["front"][0], centres["left"][0], 0.0])
    truth = {
        "frame": "x/y bounding-box centre, feet at z=0, studs",
        "joints": {k: (np.array(v) - shift).round(5).tolist() for k, v in fig.joints.items()},
        "height": 5.5,
    }
    (run / "synthetic" / "truth.json").write_text(json.dumps(truth, indent=2))
    # Stand in for a passed stage 2 so the CLI's upstream check accepts the folder.
    (run / "stage-2.json").write_text(json.dumps(
        {"stage": 2, "name": "turnaround", "ok": True, "notes": "synthetic masks (herogen.blender.synth)",
         "checks": []}, indent=2))
    return truth
