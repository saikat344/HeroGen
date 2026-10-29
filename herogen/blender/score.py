"""Score bodies against a run's views: silhouette IoU, colour error and SSIM per view, mesh sanity and, optionally,
a platform's limits.

`python3 -m herogen.blender.score --run <folder> [--body NAME=GLB ...] [--atlas NAME=PNG] [--seen NAME=views]
[--seen-shape NAME=views] [--frame NAME=contract|fit] [--profile PROFILE] [--budget NAME=N]
[--yaw-range VIEW=LO,HI]` (also
`python3 -m herogen.blender score ...`) writes `05-scores.json` (the results page shows it) and, per
body and view, `05-renders/<name>/<view>.png` (the body rendered flat in its own colours, or its silhouette when
it has none), `<view>-silhouette.png` and `<view>-diff.png`. Any GLB can be scored, ours or another tool's, so
bodies from different methods are compared on the same views. Numpy only: no Blender, no model calls.

Bodies. Any number of `--body NAME=GLB` bodies, inside the run folder or not (frozen builds of earlier
milestones, other tools' outputs); a `version.json` next to a GLB is recorded under the body's `version` (its
`label` and plain fields). Without `--body`, every stage 4 body in the run is scored: `04-body-carve/high.glb` as `carve`,
`04-body/high.glb` as `carve` or `hybrid` by its `sections.json` mesh source, any other `04-body-<x>/high.glb` as
`<x>`, and each extra atlas a body folder holds (`atlas-high-<variant>.png`) as `<name>-<variant>`: the same mesh
with that image in place of its textures (`variant_of` names the body). `--atlas NAME=PNG` does the same for a
`--body`.

Views. The four turnaround views (`masks/<view>.png`, colours from `views/<view>.png`) at their contract yaw, and
every held-out view found as `heldout/<name>.png` with `heldout/<name>-mask.png`. No building stage reads
`heldout/` (and files in its subfolders, such as `review/` and `rejected/`, are never scored). A held-out view's
true angle is unknown, so its yaw is searched (coarse, then fine) for the best IoU, around the target its name
gives: `front-left-30` in 5-55 degrees, `front-right-60` in -85 to -35 (25 degrees each side of the target),
`front-left-34` (stage 2h's three-quarter view, asked for at 45 degrees) in 15-75, any other name all the way
round; `--yaw-range VIEW=LO,HI` overrides one view's range. A named view is also scored at its target angle
(`at_fixed_yaw`). When `stage-2h.json` says the held-out views are generated images (`"reference":
"generated"`), scores against them are labelled `"measures": "consistency"`, with the reference kind and stage
2h's own status and found yaw for the view, since a generated view can invent detail and so measures
consistency with the generated character, not accuracy; without that field both are null.

Silhouettes. A body is projected orthographically at a yaw about +Z: u = cos(yaw) X + sin(yaw) Y, v = Z, so
yaw 0 is the front view, 90 left, 180 back, 270 right (docs/ARCHITECTURE.md). The turnaround views use stage 3's
mapping (`views.View`): one pixel scale for every view (the front mask's height over the body's height), u = 0
on each mask's bounding-box centre and the feet on its bottom edge. A body's `frame` says where u = 0 and the
feet are: `contract` (stage 4 bodies, built in the contract frame: the origin as it is, the height
`sections.json` declares) or `fit` (any other GLB, the default for `--body`: its own bounding box, X/Y centre
and lowest point, and its own height). Colours are registered to the views to a pixel or better only in the
contract frame, so a stage 4 body is scored with the same mapping and metric as stage 4's gates (the renderer
differs: numpy here, Blender there). The held-out views' camera is unknown, so there scale and offset are fitted
from the mask's bounding box: the figure's height sets the pixel scale, the feet sit on the box's bottom edge
and the box centres line up. Width, outline and pose are what IoU then
measures. A held-out view's `scale` is its fitted pixel scale over the turnaround views', so 1.0 means the
held-out figure is the height the front view implies.

Colour. The nearest surface at every pixel centre (same projection, a z-buffer, no lighting, no antialiasing)
takes its material's base colour: the base-colour texture sampled bilinearly at the interpolated texture
coordinate (filtered in linear light, as a GPU samples an sRGB texture), times the base-colour factor and
`COLOR_0` (multiplied in linear light). Mean absolute colour error (0-255) and grey SSIM are measured inside both
silhouettes, a few pixels in from their edges, with the same metric as stage 4's texture gates
(`texture.compare`). A body with no material, texture or vertex colour gets no colour scores.

Seen / unseen, per view. `seen_shape`: the body's shape was built from the view; `seen_colour`: its colours
were; `seen`: either. Discovered bodies take both from `sections.json` (`shape_views`, `texture.colour_views`;
a body without them saw every observed turnaround view). `--seen NAME=views` sets both for a `--body`,
`--seen-shape NAME=views` the shape alone. A held-out view is never seen. `--frame NAME=contract` scores a
`--body` built in the contract frame as stage 4 bodies are.

Mesh sanity, per mesh after welding coincident vertices: degenerate triangles, boundary edges (used once),
non-manifold edges (used more than twice), edges whose two faces disagree on winding, connected components and
inside-out closed meshes (negative volume). Self-intersections are not checked.

No triangle budget applies by default: the default target is the best-quality body. A target profile
(`herogen.blender.PROFILES`, the same names stage 4 builds to) adds its limits as extra checks: `best` none,
`roblox-ingame` the high body's 24,000 and the low LOD's 6,000 triangles, `roblox-avatar` the triangle,
watertight-part and texture limits of Roblox's character body specification (total and per part-group
triangles, watertight parts, texture size; the rig, attachments, facial animation and cages are listed as not
built, not checked). A profile passes (`"pass": true`) only when every limit was measured and met: `false` when
one fails, `null` ("incomplete") when one could not be measured (no low LOD next to the body under
`roblox-ingame`; meshes not named after the sections, or no texture, under `roblox-avatar`).
`--budget` sets a plain total for one body.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import PROFILES as TARGET_PROFILES
from . import raster
from .carve import components
from .glb import Glb, GlbError, load_glb
from .record import code_version, write_atomic
from .texture import compare
from .views import largest_component, load_mask, save_mask

CONTRACT_YAW = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}
TURNAROUND = ("front", "back", "left", "right")
HELDOUT_DIR = "heldout"
# The held-out view the pipeline writes today (stage 2h); more may sit next to it.
HELDOUT_VIEW = "front-left-34"
HELDOUT_MASK = f"{HELDOUT_DIR}/{HELDOUT_VIEW}-mask.png"
HELDOUT_YAW_RANGE = (15.0, 75.0)
# Held-out names: `front-left-<deg>` / `front-right-<deg>` name their target yaw (stage 2h: 30 and 60); the
# older single view `front-left-34` means a three-quarter view asked for at 45 degrees.
TURNED = re.compile(r"^front-(left|right)-(\d+)$")
LEGACY_THREE_QUARTER = {"34": ((15.0, 75.0), 45.0)}
TARGET_MARGIN = 25.0           # degrees searched each side of a named target yaw (kept inside 1-89)
OTHER_YAW_RANGE = (-180.0, 180.0)
COARSE_STEP, FINE_STEP = 2.0, 0.5
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")   # the results page's rule: names become paths
WELD_REL = 1e-6        # weld tolerance, relative to the body's bounding-box diagonal
MAX_PIXELS_PER_BATCH = 1 << 22
BODY_DIR = re.compile(r"^04-body(-[A-Za-z0-9][A-Za-z0-9_.-]*)?$")
VARIANT_ATLAS = re.compile(r"^atlas-high-([A-Za-z0-9][A-Za-z0-9_.-]*)\.png$")

# Target profiles (`herogen.blender.PROFILES`, shared with stage 4), checked only when asked for.
PROFILES = TARGET_PROFILES


# ---------------------------------------------------------------------------
# Views


@dataclass
class ScoreView:
    name: str
    mask: np.ndarray                 # bool, largest component
    image: np.ndarray | None         # float RGB, same shape as the mask, or None
    yaw: float | None                # contract yaw; None for a held-out view (searched)
    yaw_range: tuple[float, float] | None = None
    fixed_yaw: float | None = None   # the angle the image model was asked for, when known
    mask_path: str = ""
    image_path: str | None = None
    reference: str | None = None     # stage-2h.json's "reference" ("generated": scores measure consistency)
    stage2h: dict | None = None      # that view's status and found yaw in stage-2h.json, when recorded

    @property
    def heldout(self) -> bool:
        return self.yaw is None


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def heldout_yaw(name: str) -> tuple[tuple[float, float], float | None]:
    """(search range, the angle asked for) for a held-out view, from its name: `front-left-30` is searched in
    5-55 degrees and measured at 30, `front-right-60` in -85 to -35 and at -60, the older `front-left-34` in
    15-75 and at 45; any other name all the way round, with no target."""
    m = TURNED.match(name)
    if m is None:
        return OTHER_YAW_RANGE, None
    sign = 1.0 if m.group(1) == "left" else -1.0
    if m.group(2) in LEGACY_THREE_QUARTER:
        (lo, hi), fixed = LEGACY_THREE_QUARTER[m.group(2)]
    else:
        fixed = float(m.group(2))
        if not 0 < fixed < 90:
            return OTHER_YAW_RANGE, None
        lo, hi = max(1.0, fixed - TARGET_MARGIN), min(89.0, fixed + TARGET_MARGIN)
    return ((lo, hi) if sign > 0 else (-hi, -lo)), sign * fixed


def heldout_record(run: Path) -> dict:
    """What `stage-2h.json` says about the held-out views: the reference kind and each view's status and found
    yaw (empty when the run has no readable record)."""
    try:
        rec = json.loads((run / "stage-2h.json").read_text())
    except (OSError, ValueError):
        return {}
    details = rec.get("details") if isinstance(rec.get("details"), dict) else {}
    views = details.get("views") if isinstance(details.get("views"), dict) else {}
    ref = details.get("reference", rec.get("reference"))
    out = {"reference": ref if isinstance(ref, str) else None, "views": {}}
    for name, d in views.items():
        if isinstance(d, dict):
            out["views"][name] = {k: d[k] for k in ("status", "found_yaw_deg", "yaw_window_deg") if k in d}
    return out


def load_score_views(run: Path, yaw_ranges: dict[str, tuple[float, float]] | None = None) -> list[ScoreView]:
    """The turnaround views and every held-out view (`heldout/<name>.png` + `-mask.png`) present in the run."""
    out = []

    def colour(p: Path, mask: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        if not p.exists():
            return None, None
        img = _load_rgb(p)
        return (img, p.relative_to(run).as_posix()) if img.shape[:2] == mask.shape else (None, None)

    for v in TURNAROUND:
        p = run / "masks" / f"{v}.png"
        if not p.exists():
            continue
        m = load_mask(p)
        if not m.any():
            continue
        m = largest_component(m)
        img, ipath = colour(run / "views" / f"{v}.png", m)
        out.append(ScoreView(v, m, img, CONTRACT_YAW[v], mask_path=p.relative_to(run).as_posix(),
                             image_path=ipath))
    held = run / HELDOUT_DIR
    # only files directly in heldout/: its review/ and rejected/ subfolders hold views that must not be scored
    names = sorted(p.name[: -len("-mask.png")] for p in held.glob("*-mask.png") if p.is_file()) \
        if held.is_dir() else []
    record = heldout_record(run)
    for name in names:
        ip, mp = held / f"{name}.png", held / f"{name}-mask.png"
        if not ip.exists() or not SAFE_NAME.match(name) or name in CONTRACT_YAW:
            continue
        m = load_mask(mp)
        if not m.any():
            continue
        m = largest_component(m)
        rng, fixed = heldout_yaw(name)
        if yaw_ranges and name in yaw_ranges:
            rng = yaw_ranges[name]
        img, ipath = colour(ip, m)
        out.append(ScoreView(name, m, img, None, tuple(rng), fixed, mp.relative_to(run).as_posix(), ipath,
                             record.get("reference"), record.get("views", {}).get(name)))
    return out


# ---------------------------------------------------------------------------
# Silhouettes


def project(verts: np.ndarray, yaw_deg: float) -> np.ndarray:
    """(u, v) of each vertex in the orthographic view at ``yaw_deg``."""
    a = math.radians(yaw_deg)
    return np.stack([math.cos(a) * verts[:, 0] + math.sin(a) * verts[:, 1], verts[:, 2]], axis=1)


def view_depth(verts: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Distance along the viewing direction (smaller = nearer the camera): the camera looks along
    (-sin yaw, cos yaw, 0), so yaw 0 looks along +Y at the character's front."""
    a = math.radians(yaw_deg)
    return -math.sin(a) * verts[:, 0] + math.cos(a) * verts[:, 1]


def rasterize(xy: np.ndarray, faces: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pixels whose centre lies inside any triangle. ``xy`` is (column, row) in pixel-edge coordinates.

    Triangles are grouped by the size of their pixel bounding box, and each group tests every pixel centre in
    its boxes at once, so dense meshes of sub-pixel triangles and a few large ones both stay fast.
    """
    rows, cols = shape
    out = np.zeros(shape, dtype=bool)
    if len(faces) == 0:
        return out
    t = xy[faces]                                              # [M, 3, 2]
    lo = np.ceil(t.min(axis=1) - 0.5).astype(np.int64)
    hi = np.floor(t.max(axis=1) - 0.5).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [cols - 1, rows - 1])
    area = ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
            - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))
    keep = (hi >= lo).all(axis=1) & (area != 0)
    t, lo, hi = t[keep], lo[keep], hi[keep]
    extent = (hi - lo + 1).max(axis=1)
    size = 1
    while len(t):
        sel = extent <= size
        if sel.any():
            _fill(out, t[sel], lo[sel], hi[sel], size)
            t, lo, hi, extent = t[~sel], lo[~sel], hi[~sel], extent[~sel]
        size *= 2
    return out


def _fill(out: np.ndarray, t: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int) -> None:
    oy, ox = np.divmod(np.arange(size * size), size)
    step = max(1, MAX_PIXELS_PER_BATCH // (size * size))
    for s in range(0, len(t), step):
        tt, ll, hh = t[s : s + step], lo[s : s + step], hi[s : s + step]
        px = ll[:, :1] + ox[None, :]
        py = ll[:, 1:] + oy[None, :]
        valid = (px <= hh[:, :1]) & (py <= hh[:, 1:])
        cx, cy = px + 0.5, py + 0.5
        e = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            ax, ay = tt[:, i, :1], tt[:, i, 1:]
            bx, by = tt[:, j, :1], tt[:, j, 1:]
            e.append((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        inside = ((e[0] >= 0) & (e[1] >= 0) & (e[2] >= 0)) | ((e[0] <= 0) & (e[1] <= 0) & (e[2] <= 0))
        hit = valid & inside
        out[py[hit], px[hit]] = True


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(top, bottom, left, right) of the figure, inclusive."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def fitted_xy(verts: np.ndarray, yaw_deg: float, mask: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Pixel (column, row) of each vertex at ``yaw_deg``, fitted to ``mask``'s bounding box, and px/unit."""
    box = mask_box(mask)
    uv = project(verts, yaw_deg)
    span = float(uv[:, 1].max() - uv[:, 1].min()) if len(uv) else 0.0
    if box is None or span <= 0:
        return None
    top, bottom, left, right = box
    pps = (bottom - top + 1) / span
    col0 = (left + right + 1) / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2 * pps
    row0 = bottom + 1 + uv[:, 1].min() * pps
    return np.stack([uv[:, 0] * pps + col0, row0 - uv[:, 1] * pps], axis=1), float(pps)


def silhouette(verts: np.ndarray, faces: np.ndarray, yaw_deg: float, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """The body's silhouette at ``yaw_deg`` fitted to ``mask``'s bounding box; returns (silhouette, px/unit)."""
    fit = fitted_xy(verts, yaw_deg, mask)
    if fit is None:
        return np.zeros(mask.shape, dtype=bool), 0.0
    xy, pps = fit
    return rasterize(xy, faces, mask.shape), pps


FRAMES = ("contract", "fit")


def registration(verts: np.ndarray, views: list[ScoreView], frame: str, height: float | None) -> dict:
    """Stage 3's mapping for the turnaround views (see the module notes): the pixel scale shared by every view
    and the body-frame point that sits at u = 0 and on the feet."""
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    turn = {v.name: v for v in views if not v.heldout}
    ref = next((turn[v] for v in TURNAROUND if v in turn), None)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    if frame == "contract":
        origin = np.zeros(3)
        span = float(height) if height else float(hi[2])
    else:
        origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        span = float(hi[2] - lo[2])
    box = mask_box(ref.mask) if ref is not None else None
    pps = (box[1] - box[0] + 1) / span if box is not None and span > 0 else 0.0
    return {"frame": frame, "origin": origin, "pps": float(pps), "reference_view": ref.name if ref else None,
            "height": round(span, 6)}


def turnaround_xy(verts: np.ndarray, view: ScoreView, reg: dict) -> np.ndarray | None:
    """Pixel (column, row) of each vertex in a turnaround view under `registration`'s mapping."""
    box = mask_box(view.mask)
    if box is None or reg["pps"] <= 0:
        return None
    top, bottom, left, right = box
    uv = project(verts - reg["origin"], view.yaw)
    return np.stack([uv[:, 0] * reg["pps"] + (left + right + 1) / 2, bottom + 1 - uv[:, 1] * reg["pps"]], axis=1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def yaw_search(verts: np.ndarray, faces: np.ndarray, mask: np.ndarray,
               yaw_range: tuple[float, float] = HELDOUT_YAW_RANGE) -> tuple[float, float, np.ndarray, float, list]:
    """Search yaw for the best IoU against ``mask``: a coarse sweep, then a fine one around the best (ties keep
    the first). Returns (yaw, iou, silhouette, px/unit, [[yaw, iou], ...] in search order)."""
    curve = []

    def at(y: float):
        sil, pps = silhouette(verts, faces, y, mask)
        r = (iou(sil, mask), y, sil, pps)
        curve.append([round(y, 2), round(r[0], 4)])
        return r

    lo, hi = yaw_range
    coarse = np.arange(lo, hi + 1e-9, COARSE_STEP)
    best = max((at(float(y)) for y in coarse), key=lambda r: r[0])
    fine = np.arange(max(lo, best[1] - COARSE_STEP), min(hi, best[1] + COARSE_STEP) + 1e-9, FINE_STEP)
    done = {round(float(y), 6) for y in coarse}
    best = max([best] + [at(float(y)) for y in fine if round(float(y), 6) not in done], key=lambda r: r[0])
    return best[1], best[0], best[2], best[3], curve


def best_yaw(verts: np.ndarray, faces: np.ndarray, mask: np.ndarray,
             yaw_range: tuple[float, float] = HELDOUT_YAW_RANGE) -> tuple[float, float, np.ndarray, float]:
    """`yaw_search` without the curve: (yaw, iou, silhouette, px/unit)."""
    return yaw_search(verts, faces, mask, yaw_range)[:4]


def diff_image(sil: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB overlay: grey where both agree, red where only the mask has figure, blue where only the body does."""
    img = np.zeros(mask.shape + (3,), dtype=np.uint8)
    img[sil & mask] = (170, 170, 170)
    img[mask & ~sil] = (220, 40, 40)
    img[sil & ~mask] = (40, 90, 230)
    return img


# ---------------------------------------------------------------------------
# Colour


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


@dataclass
class BodyColour:
    """What the flat colour render needs, per face of the scored (compacted) mesh."""
    face_material: np.ndarray            # [M] material index, -1 = none
    corner_uv: np.ndarray | None         # [M, 3, 2]
    corner_rgb: np.ndarray | None        # [M, 3, 3] linear COLOR_0
    factors: np.ndarray                  # [materials + 1, 3] linear base-colour factor (last row: no material)
    textures: dict[int, np.ndarray]      # material -> float RGB image in linear light 0-1
    source: str                          # what the colours come from


def body_colour(glb: Glb, faces: np.ndarray, atlas: np.ndarray | None = None) -> BodyColour | None:
    """The body's colours, or None when it has no material, texture or vertex colour. ``atlas`` replaces every
    material's base-colour texture (a variant of the same mesh)."""
    fm = glb.merged_face_material()
    uv = glb.merged_uv()
    has_colors = any(m.colors is not None for m in glb.meshes)
    if not glb.materials and not has_colors:
        return None
    rgb = None
    if has_colors:
        cols = np.concatenate([m.colors if m.colors is not None else np.ones((len(m.verts), 4))
                               for m in glb.meshes])
        rgb = cols[faces, :3]
    factors = np.array([m.base_color[:3] for m in glb.materials] + [(1.0, 1.0, 1.0)], dtype=np.float64)
    textures: dict[int, np.ndarray] = {}
    for i, m in enumerate(glb.materials):
        if atlas is not None and (m.image is not None or uv is not None):
            textures[i] = atlas
        elif m.image is not None:
            img = glb.image(m.image)
            if img is not None:
                textures[i] = img[..., :3]
    # decoded once per image, so textures filter in linear light as a GPU samples an sRGB texture
    linear = {id(img): _to_linear(img / 255.0) for img in textures.values()}
    textures = {i: linear[id(img)] for i, img in textures.items()}
    source = ("variant atlas" if atlas is not None else "texture" if textures else
              "vertex colours" if has_colors else "material colour")
    return BodyColour(fm, uv[faces] if uv is not None else None, rgb, factors, textures, source)


def _sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample at glTF texture coordinates (v down; repeat wrap outside [0, 1])."""
    u, v = uv[:, 0], uv[:, 1]
    u = np.where((u < 0) | (u > 1), np.mod(u, 1.0), u)
    v = np.where((v < 0) | (v > 1), np.mod(v, 1.0), v)
    return raster.bilinear(img, v * img.shape[0] - 0.5, u * img.shape[1] - 0.5)


def colour_render(colour: BodyColour, xy: np.ndarray, depth: np.ndarray, faces: np.ndarray,
                  shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Flat render (float RGB 0-255, black background) and its coverage, nearest surface per pixel centre."""
    frame = raster.rasterize(xy[:, 1], xy[:, 0], depth, faces, shape)
    out = np.zeros(shape + (3,))
    m = frame.mask
    tri = frame.tri[m]
    bary = frame.bary[m]
    mat = colour.face_material[tri]
    lin = colour.factors[np.where(mat >= 0, mat, len(colour.factors) - 1)]
    if colour.corner_uv is not None:
        uv = np.einsum("nk,nkj->nj", bary, colour.corner_uv[tri])
        for mi, img in colour.textures.items():
            sel = mat == mi
            if sel.any():
                lin[sel] = lin[sel] * _sample(img, uv[sel])
    if colour.corner_rgb is not None:
        lin = lin * np.einsum("nk,nkj->nj", bary, colour.corner_rgb[tri])
    out[m] = _to_srgb(lin) * 255.0
    return out, m


# ---------------------------------------------------------------------------
# Mesh sanity


def mesh_sanity(verts: np.ndarray, faces: np.ndarray, tol: float) -> dict:
    """Counts for one mesh after welding vertices closer than ``tol``."""
    if len(faces) == 0:
        return {"triangles": 0, "degenerate": 0, "boundary_edges": 0, "nonmanifold_edges": 0,
                "inconsistent_edges": 0, "components": 0, "watertight": False, "inside_out": False}
    key = np.round(verts / tol).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.ravel()
    f = weld[faces]
    p = verts[faces]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    degenerate = repeated | (np.linalg.norm(cross, axis=1) <= tol * tol)
    good = f[~degenerate]
    directed = np.concatenate([good[:, [0, 1]], good[:, [1, 2]], good[:, [2, 0]]])
    undirected, uses = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    _, dir_uses = np.unique(directed, axis=0, return_counts=True)
    boundary = int((uses == 1).sum())
    nonmanifold = int((uses > 2).sum())
    inconsistent = int((dir_uses > 1).sum())
    n_comp = 0
    if len(good):
        used = np.unique(good)
        n_comp = int(len(np.unique(components(int(weld.max()) + 1, good)[used])))
    watertight = len(good) > 0 and boundary == 0 and nonmanifold == 0
    volume = float(np.einsum("ij,ij->i", p[~degenerate][:, 0], cross[~degenerate]).sum() / 6.0)
    return {"triangles": int(len(faces)), "degenerate": int(degenerate.sum()), "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold, "inconsistent_edges": inconsistent, "components": n_comp,
            "watertight": bool(watertight), "inside_out": bool(watertight and inconsistent == 0 and volume < 0)}


def body_sanity(glb: Glb) -> dict:
    """Per-mesh sanity summed over the body, with the meshes that are not closed named."""
    verts, _ = glb.merged()
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) if len(verts) else 0.0
    tol = max(diag * WELD_REL, 1e-12)
    total = {"meshes": len(glb.meshes), "degenerate": 0, "boundary_edges": 0, "nonmanifold_edges": 0,
             "inconsistent_edges": 0, "components": 0}
    open_meshes, inside_out = [], []
    for m in glb.meshes:
        s = mesh_sanity(m.verts, m.faces, tol)
        for k in ("degenerate", "boundary_edges", "nonmanifold_edges", "inconsistent_edges", "components"):
            total[k] += s[k]
        if not s["watertight"]:
            open_meshes.append(m.name)
        if s["inside_out"]:
            inside_out.append(m.name)
    total["watertight"] = bool(glb.meshes) and not open_meshes
    total["open_meshes"] = open_meshes[:20]
    total["inside_out_meshes"] = inside_out[:20]
    total["height"] = round(float(verts[:, 2].max() - verts[:, 2].min()), 4) if len(verts) else 0.0
    total["self_intersections"] = "not checked"
    return total


def _check(name: str, value, limit, passed: bool | None, note: str = "", required: bool = True) -> dict:
    """A profile check; `passed` None means it could not be measured. `required` False marks a listing (what the
    profile does not build) that never counts towards the profile's result."""
    out = {"name": name, "value": value, "limit": limit, "pass": passed}
    if note:
        out["note"] = note
    if not required:
        out["required"] = False
    return out


def profile_result(checks: list[dict]) -> tuple[bool | None, str]:
    """A profile's `pass` and `status`: False ("fail") when a required check fails, None ("incomplete") when one
    could not be measured, True ("pass") only when every required check was measured and met."""
    req = [c for c in checks if c.get("required", True)]
    if any(c["pass"] is False for c in req):
        return False, "fail"
    if any(c["pass"] is None for c in req):
        return None, "incomplete"
    return True, "pass"


def profile_checks(glb: Glb, sanity: dict, profile: str, atlas: int | None = None,
                   triangles_low: int | None = None) -> list[dict]:
    """A target profile's limits as checks. `best` has none. `roblox-ingame`: the high body (this GLB) and, when
    known, its low LOD within their budgets. `roblox-avatar`: the total, each part group found by mesh name (a
    Blender `.001` suffix is ignored; a body whose meshes are not named after the sections gets those checks with
    value None), watertight parts and the texture size; what the profile does not build is listed, not
    checked."""
    spec = PROFILES[profile]
    if profile == "best":
        return []
    if "groups" not in spec:
        hi, lo = spec["lods"]["high"], spec["lods"]["low"]
        return [_check("triangles_high", glb.triangles, hi, glb.triangles <= hi),
                _check("triangles_low", triangles_low, lo, None if triangles_low is None else triangles_low <= lo,
                       "" if triangles_low is not None else "no low LOD next to this body")]
    out = [_check("triangles_total", glb.triangles, spec["triangles_total"],
                  glb.triangles <= spec["triangles_total"])]
    counts: dict[str, int] = {}
    for m in glb.meshes:
        base = m.name.split(".")[0]
        counts[base] = counts.get(base, 0) + len(m.faces)
    for group, (sections, limit) in spec["groups"].items():
        if not any(s in counts for s in sections):
            out.append(_check(f"triangles_{group}", None, limit, None, "no meshes named after these sections"))
            continue
        n = sum(counts.get(s, 0) for s in sections)
        out.append(_check(f"triangles_{group}", n, limit, n <= limit))
    out.append(_check("texture_max_px", atlas, spec["texture_max_px"],
                      None if atlas is None else atlas <= spec["texture_max_px"]))
    if spec["watertight_parts"]:
        out.append(_check("watertight_parts", len(sanity["open_meshes"]), 0, sanity["watertight"],
                          "meshes that are not closed: " + ", ".join(sanity["open_meshes"])
                          if sanity["open_meshes"] else ""))
    out.append(_check("not_built", None, None, None, "not built, not checked: " + ", ".join(spec["not_built"]),
                      required=False))
    return out


def profile_budget(profile: str | None) -> int | None:
    """The plain triangle total a profile implies for one GLB (its high body), or None."""
    if profile is None or profile == "best":
        return None
    spec = PROFILES[profile]
    return spec["triangles_total"] if "groups" in spec else spec["lods"]["high"]


# ---------------------------------------------------------------------------
# Bodies


@dataclass
class BodySpec:
    name: str
    glb: Path
    atlas: Path | None = None
    variant_of: str | None = None
    seen_shape: set | None = None    # None: from sections.json, else every observed turnaround view
    seen_colour: set | None = None
    meta: dict | None = None         # sections.json next to the GLB, when there is one
    frame: str = "fit"


def _observed(run: Path) -> set:
    """Turnaround views that are observations (not derived from the others by stage 2)."""
    meta_path = run / "views" / "meta.json"
    try:
        method = json.loads(meta_path.read_text()).get("mask_method") or {} if meta_path.exists() else {}
    except ValueError:
        method = {}
    return {v for v in TURNAROUND if (run / "masks" / f"{v}.png").exists()
            and "visual_hull" not in str(method.get(v, ""))}


def _sections(d: Path) -> dict:
    p = d / "sections.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return {}


def discover_bodies(run: Path) -> list[BodySpec]:
    """Every stage 4 body in the run (see the module notes), with its atlas variants."""
    out: list[BodySpec] = []
    dirs = sorted((p for p in run.iterdir() if BODY_DIR.match(p.name) and p.is_dir() and not p.is_symlink()),
                  key=lambda p: (p.name != "04-body-carve", p.name != "04-body", p.name)) if run.is_dir() else []
    taken: set[str] = set()
    for d in dirs:
        high = d / "high.glb"
        if not high.is_file():
            continue
        meta = _sections(d)
        if d.name == "04-body":
            name = "hybrid" if meta.get("mesh_source") == "sam3d-objects" else "carve"
        else:
            name = d.name[len("04-body-"):]
        if name in taken:
            name = d.name
        taken.add(name)
        shape = set(meta["shape_views"]) if meta.get("shape_views") else None
        colour = set((meta.get("texture") or {}).get("colour_views") or []) or None
        out.append(BodySpec(name, high, None, None, shape, colour, meta, "contract"))
        for atlas in sorted(d.glob("atlas-high-*.png")):
            m = VARIANT_ATLAS.match(atlas.name)
            if m:
                out.append(BodySpec(f"{name}-{m.group(1)}", high, atlas, name, shape, colour, meta, "contract"))
    return out


def _display_path(run: Path, path: Path) -> str:
    """Run-relative when the file is inside the run folder, else its file name (no personal paths)."""
    try:
        return path.resolve().relative_to(run.resolve()).as_posix()
    except ValueError:
        return path.name


VERSION_KEYS = ("label", "frozen", "sketch", "profile", "note")


def body_version(glb_path: Path) -> dict | None:
    """The `version.json` next to a GLB (a frozen build of an earlier milestone, say), reduced to its plain
    fields: `label` first, plus `frozen` (when it was frozen), `sketch`, `profile`, `note` and `code`
    (`{"commit", "dirty"}`, the code that built it) when present. None when there is no such file."""
    p = Path(glb_path).with_name("version.json")
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"label": None, "error": "version.json could not be read"}
    if not isinstance(raw, dict):
        return {"label": None, "error": "version.json is not an object"}

    def plain(v):
        return v[:200] if isinstance(v, str) else v if isinstance(v, (bool, int, float)) or v is None else None

    out = {"label": plain(raw.get("label"))}
    out.update({k: plain(raw[k]) for k in VERSION_KEYS[1:] if k in raw})
    if isinstance(raw.get("code"), dict):
        out["code"] = {k: plain(raw["code"][k]) for k in ("commit", "dirty") if k in raw["code"]}
    return out


def seen_flags(view: ScoreView, shape: set, colour: set) -> dict:
    if view.heldout:
        return {"seen": False, "seen_shape": False, "seen_colour": False}
    s, c = view.name in shape, view.name in colour
    return {"seen": bool(s or c), "seen_shape": bool(s), "seen_colour": bool(c)}


def _save_png(path: Path, img: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8)).save(path)


def score_body(run: Path, name: str, glb_path: Path, *, seen: set[str] | None = None,
               seen_shape: set[str] | None = None, budget: int | None = None, profile: str | None = None,
               views: list[ScoreView] | None = None, atlas: Path | None = None, variant_of: str | None = None,
               meta: dict | None = None, yaw_range: tuple[float, float] | None = None,
               frame: str = "fit", height: float | None = None, geometry_cache: dict | None = None) -> dict:
    """Score one GLB on the run's views and write its renders; returns its `05-scores.json` entry.

    ``seen`` is the set of turnaround views the body was built from (shape and colour; default: every observed
    view), ``seen_shape`` overrides the shape's. ``budget`` is a plain triangle total; it defaults to the
    profile's total when a profile is given, and to no limit otherwise. ``yaw_range`` overrides every held-out
    view's search range. ``frame`` is `contract` or `fit` (see the module notes); ``height`` is the contract
    frame's body height (default: `sections.json`'s, else the body's top). ``geometry_cache`` lets bodies that
    share a GLB (atlas variants) share its silhouettes and yaw search."""
    if not SAFE_NAME.match(name):
        raise ValueError(f"body name {name!r} must be a plain file name (letters, digits, _ . -)")
    views = load_score_views(run) if views is None else views
    held_names = {v.name for v in views if v.heldout} | {HELDOUT_VIEW}
    for given in (seen or set()) | (seen_shape or set()):
        if given in held_names:
            raise ValueError(f"the held-out view {given} cannot be seen: no body may be built from it")
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r} (known: {', '.join(sorted(PROFILES))})")
    if budget is None and profile is not None:
        budget = profile_budget(profile)
    observed = _observed(run)
    colour_seen = set(seen) if seen is not None else observed
    shape_seen = set(seen_shape) if seen_shape is not None else (set(seen) if seen is not None else observed)

    meta = meta or {}
    if height is None and frame == "contract":
        height = meta.get("height")
    key = (str(Path(glb_path).resolve()), frame, height)
    cached = geometry_cache.get(key) if geometry_cache is not None else None
    if cached is None:
        glb = load_glb(glb_path)
        verts, faces = glb.merged()
        if len(faces) == 0:
            raise GlbError(f"{Path(glb_path).name} holds no triangles")
        used = np.unique(faces)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        cached = {"glb": glb, "faces": faces, "pv": verts[used], "pf": remap[faces], "views": {},
                  "sanity": body_sanity(glb), "reg": registration(verts[used], views, frame, height)}
        if geometry_cache is not None:
            geometry_cache[key] = cached
    glb, pv, pf, reg = cached["glb"], cached["pv"], cached["pf"], cached["reg"]
    atlas_img = _load_rgb(atlas) if atlas is not None else None
    colour = body_colour(glb, cached["faces"], atlas_img)
    out_dir = run / "05-renders" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    def placed(view: ScoreView, yaw: float) -> np.ndarray | None:
        if view.heldout:
            fit = fitted_xy(pv, yaw, view.mask)
            return fit[0] if fit is not None else None
        return turnaround_xy(pv, view, reg)

    def colour_scores(view: ScoreView, yaw: float, sil: np.ndarray, tag: str = "") -> dict:
        xy = placed(view, yaw) if colour is not None and view.image is not None else None
        if xy is None:
            _save_png(out_dir / f"{view.name}{tag}.png", np.repeat(sil[..., None] * 255.0, 3, axis=2))
            return {"colour_error": None, "ssim": None, "pixels": None}
        img, cover = colour_render(colour, xy, view_depth(pv, yaw), pf, view.mask.shape)
        _save_png(out_dir / f"{view.name}{tag}.png", img)
        c = compare(img, cover, view.image, view.mask)
        return {"colour_error": c["mean_abs_error"], "ssim": c["ssim"], "pixels": c["pixels"]}

    rows, front_pps = {}, None
    for view in views:
        geo = cached["views"].get(view.name)
        if geo is None:
            if view.heldout:
                rng = tuple(yaw_range) if yaw_range is not None else view.yaw_range
                yaw, value, sil, pps, curve = yaw_search(pv, pf, view.mask, rng)
                geo = {"yaw": yaw, "iou": value, "sil": sil, "pps": pps, "curve": curve, "range": rng}
            else:
                xy = turnaround_xy(pv, view, reg)
                sil = rasterize(xy, pf, view.mask.shape) if xy is not None else np.zeros(view.mask.shape, bool)
                geo = {"yaw": view.yaw, "iou": iou(sil, view.mask), "sil": sil, "pps": reg["pps"]}
            cached["views"][view.name] = geo
        if not view.heldout:
            front_pps = reg["pps"]
        row = {"iou": round(geo["iou"], 4), **colour_scores(view, geo["yaw"], geo["sil"]),
               "yaw_deg": round(geo["yaw"], 2), **seen_flags(view, shape_seen, colour_seen)}
        if view.heldout:
            row["reference"] = view.reference
            row["measures"] = "consistency" if view.reference == "generated" else None
            if view.stage2h:
                row["stage2h"] = view.stage2h
            lo, hi = geo["range"]
            row["scale"] = round(geo["pps"] / front_pps, 4) if front_pps else None
            row["yaw_range"] = [lo, hi]
            row["yaw_at_range_edge"] = bool(min(abs(geo["yaw"] - lo), abs(geo["yaw"] - hi)) < COARSE_STEP / 2)
            row["yaw_search"] = geo["curve"]
            if view.fixed_yaw is not None:
                fsil, _ = silhouette(pv, pf, view.fixed_yaw, view.mask)
                row["at_fixed_yaw"] = {f"{view.fixed_yaw:g}": {
                    "iou": round(iou(fsil, view.mask), 4),
                    **colour_scores(view, view.fixed_yaw, fsil, f"-yaw{view.fixed_yaw:g}")}}
        rows[view.name] = row
        save_mask(out_dir / f"{view.name}-silhouette.png", geo["sil"])
        _save_png(out_dir / f"{view.name}-diff.png", diff_image(geo["sil"], view.mask))
    sanity = cached["sanity"]
    lods = meta.get("lods") or {}
    atlas_px = (max(atlas_img.shape[:2]) if atlas_img is not None
                else max((max(s) for s in glb.image_sizes), default=None))
    low = Path(glb_path).with_name("low.glb") if Path(glb_path).name == "high.glb" else None
    triangles_low = (lods.get("low") or {}).get("triangles")
    if triangles_low is None and low is not None and low.is_file():
        try:
            triangles_low = load_glb(low).triangles
        except GlbError:
            triangles_low = None
    entry = {
        "name": name,
        "glb": _display_path(run, Path(glb_path)),
        "mesh_source": meta.get("mesh_source", "carve") if meta else None,
        "joints_source": meta.get("joints_source", "silhouette") if meta else None,
        "triangles": glb.triangles,
        "triangles_low": triangles_low,
        "budget": budget,
        "within_budget": None if budget is None else glb.triangles <= budget,
        "atlas": atlas_px,
        "colour_source": colour.source if colour is not None else None,
        "frame": frame,
        "pixels_per_unit": round(reg["pps"], 4),
        "height": reg["height"],
        "shape_views": sorted(shape_seen),
        "colour_views": sorted(colour_seen),
        "sanity": sanity,
        "views": rows,
        "notes": glb.notes,
        "version": body_version(Path(glb_path)),
    }
    if atlas is not None:
        entry["atlas_image"] = _display_path(run, atlas)
        entry["variant_of"] = variant_of
    if profile is not None:
        checks = profile_checks(glb, sanity, profile, atlas_px, triangles_low)
        passed, status = profile_result(checks)
        entry["profile"] = {"name": profile, "checks": checks, "pass": passed, "status": status}
    return entry


def comparison_sheets(run: Path, bodies: list[dict], views: list[ScoreView], tile: int = 640) -> list[Path]:
    """Per held-out view, one image: the held-out image, then each (non-variant) body's render at its found
    yaw. The front-left view's (or else the first one's) is also written as `heldout-comparison.png`."""
    from PIL import Image, ImageDraw

    out = []
    held = [v for v in views if v.heldout and v.image_path]
    for v in held:
        shown = [b for b in bodies if not b.get("variant_of") and v.name in b["views"]
                 and (run / "05-renders" / b["name"] / f"{v.name}.png").exists()]
        if not shown:
            continue
        paths = [run / v.image_path] + [run / "05-renders" / b["name"] / f"{v.name}.png" for b in shown]
        labels = [f"held-out view {v.name}"] + [f"{b['name']} (yaw {b['views'][v.name]['yaw_deg']:g})"
                                                for b in shown]
        sheet = Image.new("RGB", (tile * len(paths), tile + 28), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, (p, lab) in enumerate(zip(paths, labels)):
            with Image.open(p) as im:
                im = im.convert("RGB")
                n = max(im.size)
                sq = Image.new("RGB", (n, n), (255, 255, 255) if i == 0 else (0, 0, 0))
                sq.paste(im, ((n - im.size[0]) // 2, (n - im.size[1]) // 2))
                sheet.paste(sq.resize((tile, tile)), (tile * i, 28))
            draw.text((tile * i + 8, 8), lab, fill=(0, 0, 0))
        path = run / "05-renders" / f"heldout-comparison-{v.name}.png"
        sheet.save(path)
        out.append(path)
    if out:
        main = next((p for p in out if p.name == f"heldout-comparison-{HELDOUT_VIEW}.png"), out[0])
        legacy = run / "05-renders" / "heldout-comparison.png"
        legacy.write_bytes(main.read_bytes())
        out.append(legacy)
    return out


def write_scores(run: Path, bodies: list[dict], views: list[ScoreView] | None = None) -> Path:
    """Write `05-scores.json`, replacing bodies of the same name and keeping the others."""
    path = run / "05-scores.json"
    kept = []
    if path.exists():
        try:
            kept = [b for b in json.loads(path.read_text()).get("bodies", [])
                    if isinstance(b, dict) and b.get("name") not in {x["name"] for x in bodies}]
        except (ValueError, AttributeError):
            kept = []
    views = views or []
    data = {
        "scored": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scorer": "herogen.blender.score",
        "code": code_version(),
        "method": {
            "iou": "orthographic silhouette; turnaround views at their contract yaw with stage 3's mapping (one "
                   "pixel scale from the front mask's height over the body's, u = 0 on each mask's box centre, "
                   "feet on its bottom edge; the body's origin as built (frame contract) or its own bounding box "
                   "(frame fit)); held-out views fitted to their mask's bounding box",
            "colour": "flat render (base-colour texture x factor x COLOR_0, nearest surface, no lighting or "
                      "antialiasing) on the same projection; mean absolute error 0-255 and grey SSIM inside both "
                      "silhouettes a few pixels in from their edges (texture.compare, as stage 4)",
            "heldout_yaw": f"searched in {COARSE_STEP:g} then {FINE_STEP:g} degree steps; best IoU kept",
            "heldout_scale": "held-out fitted pixel scale / front fitted pixel scale",
            "heldout_reference": "a held-out view stage-2h.json marks \"reference\": \"generated\" is a generated "
                                 "image: its scores measure consistency with the generated character, not accuracy",
            "seen": "seen_shape: the shape was built from the view; seen_colour: the colours were; seen: either",
            "budget": "none unless --budget or --profile is given",
        },
        "views": [{"view": v.name, "mask": v.mask_path, "image": v.image_path, "heldout": v.heldout,
                   **({"yaw_range": list(v.yaw_range), "fixed_yaw": v.fixed_yaw, "reference": v.reference,
                       "measures": "consistency" if v.reference == "generated" else None} if v.heldout else
                      {"yaw_deg": v.yaw})} for v in views],
        "bodies": kept + bodies,
    }
    write_atomic(path, json.dumps(data, indent=2) + "\n")
    return path


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"{what} must be NAME=VALUE, got {item!r}")
        out[name] = value
    return out


def _view_set(text: str) -> set[str]:
    return {x.strip() for x in text.split(",") if x.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.blender.score", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="run folder holding masks/, views/ and heldout/")
    ap.add_argument("--body", action="append", default=[],
                    help="NAME=path/to/body.glb (default: every stage 4 body in the run, with its variants)")
    ap.add_argument("--atlas", action="append", default=[],
                    help="NAME=atlas.png: score that body with this image in place of its textures")
    ap.add_argument("--seen", action="append", default=[],
                    help="NAME=front,back,left,right: views the body was built from, shape and colour "
                         "(default: every observed view)")
    ap.add_argument("--seen-shape", action="append", default=[],
                    help="NAME=front,...: views the body's shape was built from, when they differ from --seen")
    ap.add_argument("--frame", action="append", default=[],
                    help="NAME=contract|fit: how a --body is placed in the turnaround views (default fit)")
    ap.add_argument("--profile", choices=tuple(PROFILES), default=None,
                    help="also check a target profile's limits: best, roblox-ingame or roblox-avatar "
                         "(default: none, no triangle budget)")
    ap.add_argument("--budget", action="append", default=[],
                    help="NAME=triangles: a plain total for one body (default: the profile's, else none)")
    ap.add_argument("--yaw-range", action="append", default=[], metavar="VIEW=LO,HI",
                    help="degrees searched for one held-out view (default from its name)")
    a = ap.parse_args(argv)
    try:
        given = _pairs(a.body, "--body")
        atlases = _pairs(a.atlas, "--atlas")
        seen = {k: _view_set(v) for k, v in _pairs(a.seen, "--seen").items()}
        seen_shape = {k: _view_set(v) for k, v in _pairs(a.seen_shape, "--seen-shape").items()}
        budgets = {k: int(v) for k, v in _pairs(a.budget, "--budget").items()}
        frames = _pairs(a.frame, "--frame")
        bad_frames = {k: v for k, v in frames.items() if v not in FRAMES}
        if bad_frames:
            raise ValueError(f"--frame must be one of {FRAMES}, got {bad_frames}")
        ranges = {}
        for k, v in _pairs(a.yaw_range, "--yaw-range").items():
            lo, sep, hi = v.partition(",")
            if not sep:
                raise ValueError(f"--yaw-range must be VIEW=LO,HI, got {k}={v}")
            ranges[k] = (float(lo), float(hi))
    except ValueError as e:
        ap.error(str(e))
    for name in set(seen) | set(seen_shape) | set(atlases) | set(frames):
        if name not in given:
            ap.error(f"--seen/--seen-shape/--atlas/--frame name {name!r} has no --body")
    for name, views in list(seen.items()) + list(seen_shape.items()):
        bad = views - set(TURNAROUND)
        if bad:
            ap.error(f"--seen {name}: unknown or held-out views {sorted(bad)}")
    if not a.run.is_dir():
        ap.error(f"run folder {a.run} not found")
    views = load_score_views(a.run, ranges)
    if not any(v.heldout for v in views):
        print(f"warning: no held-out view (heldout/<name>.png with <name>-mask.png); only the turnaround views "
              "will be scored", file=sys.stderr)
    unknown = set(ranges) - {v.name for v in views if v.heldout}
    if unknown:
        ap.error(f"--yaw-range names views that are not held-out views of this run: {sorted(unknown)}")
    if given:
        specs = [BodySpec(n, Path(p), Path(atlases[n]) if n in atlases else None, None,
                          seen_shape.get(n), seen.get(n), None, frames.get(n, "fit")) for n, p in given.items()]
    else:
        specs = discover_bodies(a.run)
        if not specs:
            print("no body to score: give --body NAME=GLB or run stage 4 (04-body*/high.glb)", file=sys.stderr)
            return 1
    for name in budgets:
        if name not in {s.name for s in specs}:
            ap.error(f"--budget name {name!r} names no body")
    results, cache = [], {}
    for s in specs:
        # --seen sets shape and colour; a discovered body's sections.json sets each.
        colour_set = s.seen_colour
        shape_set = s.seen_shape if s.seen_shape is not None else colour_set
        try:
            entry = score_body(a.run, s.name, s.glb, seen=colour_set, seen_shape=shape_set,
                               budget=budgets.get(s.name), profile=a.profile, views=views, atlas=s.atlas,
                               variant_of=s.variant_of, meta=s.meta, frame=s.frame, geometry_cache=cache)
        except (OSError, GlbError, ValueError) as e:
            print(f"{s.name}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        results.append(entry)
        line = {"body": s.name, "triangles": entry["triangles"], "within_budget": entry["within_budget"],
                "watertight": entry["sanity"]["watertight"],
                "views": {v: {k: r[k] for k in ("iou", "colour_error", "ssim", "yaw_deg") if r.get(k) is not None}
                          for v, r in entry["views"].items()}}
        if "profile" in entry:
            line["profile_pass"] = entry["profile"]["pass"]
        print(json.dumps(line))
    comparison_sheets(a.run, results, views)
    print(f"wrote {write_scores(a.run, results, views)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
