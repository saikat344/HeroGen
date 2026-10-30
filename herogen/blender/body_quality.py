"""Body acceptance measurements: cross-section shape and a fixed multiview visual question.

This module supplies measurements, not fitted acceptance thresholds. Visual calls use the pipeline's
strict-schema client, retry policy, content cache and token cost accounting.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw

from herogen.pipeline.client import ClientError, TEXT_MODEL, schema_errors

YAWS = {"front": 0, "left": 90, "back": 180, "right": 270,
        "front-left-30": 30, "front-right-30": 330}
SCALE_YAWS = (0, 30, 60, 90, 180, 270, 300, 330)
PROMPT = (
    "Is this a usable game character body for this sketch? The first tile is the sketch. "
    "The next six tiles are the same untextured body at front, left, back, right, +30 and -30 degrees. "
    "The oblique views were not used to construct or register the body. Judge shape and major parts "
    "across ALL six views: a plausible solid body, a distinct head, torso, two arms and two legs, "
    "and proportions and major costume masses matching the sketch. Reject slabs, box/prism bodies, "
    "buried or missing limbs, collapsed parts, severe distortion and implausible depth. Ignore colour, "
    "small details, fine surface banding and simplified held weapons. Usable means an acceptable base "
    "for later painting and rigging from every common angle shown, not proof of topology or export. "
    "Return usable_probability as an integer from 0 (certainly unusable) to 100 (certainly usable), "
    "and a short concrete reason. Do not infer quality from a silhouette overlap score."
)
SCHEMA = {"name": "body_usability_v1", "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {"usable_probability": {"type": "integer", "enum": list(range(101))},
                   "reason": {"type": "string"}},
    "required": ["usable_probability", "reason"],
}}
PROMPT_SHA256 = "8e054e5d6ba4b8efb4316878ed01c4f9c33eb3d889b5d6edc91f0781ad1ad38c"


def _hull_area(points: np.ndarray) -> float:
    """Area of a 2D convex hull (monotone chain, no optional dependency)."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    halves = []
    for seq in (pts, pts[::-1]):
        half = []
        for p in seq:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves += half[:-1]
    p = np.asarray(halves)
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) -
                     np.dot(p[:, 1], np.roll(p[:, 0], 1))) / 2)


def cross_sections(verts: np.ndarray, faces: np.ndarray) -> dict:
    """Mean XY convex-hull area / bounding rectangle at 35%, 45%, 55% body height.

    A rectangular prism scores 1; a finely tessellated ellipse approaches pi/4. This is a shape
    descriptor, not a universal anatomical rule. All intersected parts contribute to each slice.
    Invalid or degenerate sections have no score and must be rejected by the caller.
    """
    if (verts.ndim != 2 or verts.shape[1] != 3 or not len(verts) or
            not np.isfinite(verts).all() or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)):
        raise ValueError("body quality needs finite vertices and triangular faces")
    if faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError("body quality face index outside vertices")
    lo, hi = verts[:, 2].min(), verts[:, 2].max()
    if hi <= lo:
        raise ValueError("body quality needs positive body height")
    tri = verts[faces]
    sections = []
    for fraction in (0.35, 0.45, 0.55):
        z = lo + fraction * (hi - lo)
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[:, i], tri[:, j]
            hit = ((a[:, 2] <= z) & (b[:, 2] > z)) | ((b[:, 2] <= z) & (a[:, 2] > z))
            a, b = a[hit], b[hit]
            t = (z - a[:, 2]) / (b[:, 2] - a[:, 2])
            points.extend((a[:, :2] + t[:, None] * (b[:, :2] - a[:, :2])).tolist())
        p = np.asarray(points)
        area = _hull_area(p) if len(p) else 0.0
        box = float(np.prod(np.ptp(p, axis=0))) if len(p) else 0.0
        sections.append({"height_fraction": fraction, "rectangularity": area / box if box > 0 else None})
    scores = [s["rectangularity"] for s in sections]
    return {"sections": sections, "rectangularity": float(np.mean(scores))
            if all(s is not None and s > 0 for s in scores) else None}


def render_views(verts: np.ndarray, faces: np.ndarray, size: int = 512) -> dict[str, Image.Image]:
    """Smooth two-sided grey Lambert shading, shared orthographic scale, white background.

    Scale includes the eight cardinal/30/60 degree directions used by the body measurement renderer.
    No simplification or per-view stretching is applied.
    """
    from . import raster
    from .score import project, view_depth

    height = float(np.ptp(verts[:, 2]))
    width = max(float(np.ptp(project(verts, yaw)[:, 0])) for yaw in SCALE_YAWS)
    if not np.isfinite(verts).all() or min(height, width) <= 0:
        raise ValueError("cannot render a degenerate body")
    pps = 0.88 * size / max(height, width)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts, dtype=np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    images = {}
    for name, yaw in YAWS.items():
        uv = project(verts, yaw)
        cols = (uv[:, 0] - (uv[:, 0].min() + uv[:, 0].max()) / 2) * pps + size / 2
        rows = size / 2 - (uv[:, 1] - (uv[:, 1].min() + uv[:, 1].max()) / 2) * pps
        fr = raster.rasterize(rows, cols, view_depth(verts, yaw), faces, (size, size))
        m = fr.mask
        a = math.radians(yaw)
        camera = np.array([math.sin(a), -math.cos(a), 0.0])
        side = np.array([math.cos(a), math.sin(a), 0.0])
        light = camera * 0.75 + np.array([0, 0, 0.55]) - side * 0.35
        light /= np.linalg.norm(light)
        n = np.einsum("pk,pkj->pj", fr.bary[m], vn[faces[fr.tri[m]]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[n @ camera < 0] *= -1
        arr = np.full((size, size), 255.0)
        arr[m] = 40 + 185 * np.clip(n @ light, 0, 1)
        images[name] = Image.fromarray(arr.astype(np.uint8))
    return images


def contact_sheet(sketch: Image.Image, images: dict[str, Image.Image]) -> Image.Image:
    """Seven tiles, fixed 4x2 grid; body tiles keep their common scale and all their whitespace."""
    tile, label = 320, 24
    sheet = Image.new("RGB", (4 * tile, 2 * (tile + label)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (name, im) in enumerate([("sketch", sketch), *[(n, images[n]) for n in YAWS]]):
        im = im.convert("RGB")
        if name == "sketch":
            box = im.convert("L").point(lambda x: 255 if x < 245 else 0).getbbox()
            if box:
                im = im.crop(box)
        im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x, y = (k % 4) * tile, (k // 4) * (tile + label)
        draw.text((x + 4, y + 4), name, fill="black")
        sheet.paste(im, (x + (tile - im.width) // 2, y + label + (tile - im.height) // 2))
    return sheet


def visual_score(client, sketch: Image.Image, images: dict[str, Image.Image]) -> dict:
    """One hash-pinned strict-schema question. Every missing/invalid answer raises; never a pass."""
    if hashlib.sha256(PROMPT.encode()).hexdigest() != PROMPT_SHA256:
        raise ClientError("body usability prompt hash changed; recalibration required")
    answer = client.compare_images(PROMPT, [contact_sheet(sketch, images)], schema=SCHEMA,
                                   model=TEXT_MODEL, reasoning_effort="low", max_output_tokens=1200,
                                   image_max_side=1280, jpeg_quality=90)
    errors = schema_errors(answer, SCHEMA["schema"])
    score = answer.get("usable_probability")
    if (errors or type(score) is not int or not 0 <= score <= 100 or
            not isinstance(answer.get("reason"), str) or not answer["reason"].strip()):
        raise ClientError("body usability response violates the frozen schema")
    return {**answer, "model": TEXT_MODEL, "prompt_sha256": PROMPT_SHA256}
