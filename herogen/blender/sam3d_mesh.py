"""Stage 3 mesh source `sam3d-objects`: put SAM 3D Objects' textured GLB into the contract frame.

SAM 3D Objects reconstructs the figure from the front view and its mask alone, as an object seen by a
perspective camera. The shape and every choice here but one use that same front view; the one exception is the
pitch, a rigid tilt the front view cannot see, which is registered on the observed left and right masks (step
3b). Back, top, a view stage 2 derived from the others and the held-out views are never used:

1. Axes: glTF +Y up becomes +Z up (`glb.load_glb`, Blender's import convention). The vertices are welded
   (glTF splits them at UV seams) so the body is one surface; each face keeps its corner UVs into the source
   texture.
2. Facing: of the four quarter turns about +Z, the one whose front render best matches the front view
   (highest silhouette IoU minus colour error / 100, the source texture rendered flat; a figure and its back
   have near-identical outlines, so the colour decides between them).
3. Fit: small yaw / roll and a perspective-to-orthographic correction (the model assumes a perspective camera;
   points farther from that camera are scaled up about its height so the orthographic front outline matches the
   drawing), coordinate descent on the front IoU at half resolution with the pitch held.
3b. Pitch: a tilt about X (the body's side-to-side axis once the facing is applied; `rotation`) barely changes
   the front outline (foreshortening and the renormalised height move the front IoU by a few hundredths over 20
   degrees, enough for a front-only fit to drift towards its limit) but tips the side profile. With the step 3
   fit held, the pitch is swept from -30 to 30 degrees in 2 degree steps, scored by the mean IoU of the left and
   right masks at half resolution, and the peaks refined to 1 degree; the front fit is then redone with that
   pitch. The pitch stays 0, with the reason recorded, unless both side views are observed (a missing, empty or
   derived mask counts as not), and when the side IoU varies by less than 0.02 over the sweep (flat), another
   peak more than 8 degrees away comes within 0.01 of the best (ambiguous), or the best lies beyond the +-20
   degree limit. No Y offset is fitted: the frame stays centred on the bounding box (step 4), as each view's u
   is centred on its mask's box, and the pitch is chosen in that frame, so a part the mesh lacks or places
   differently (a tail, a pack), which moves a mask's box, can pull the pitch. This registers the model's pose
   only: the side masks never change its shape.
4. Normalise: height `height` studs, feet at Z = 0, X/Y centred on the bounding box (the contract frame).
5. Conform: the front outline is pulled onto the front mask with a smooth 2D displacement field in X/Z
   (silhouette-rim pixels move to the nearest mask edge, outward moves only along the rim's outward normal,
   capped, Gaussian-blended). It moves front and back surfaces together, so depth is the model's own.
6. Remesh: the model's low-poly surface has a few edges shared by four triangles, handles, and the conform can
   fold it locally, all of which break planar cutting. It is voxelised by ray parity along Y (a voxel is inside
   when an odd number of surface crossings lie in front of it), blurred, made well-composed (no diagonal-only
   voxel contacts), surfaced with surface nets and Taubin smoothed like the carve, giving one closed manifold
   surface with the model's shape. The source mesh (with its UVs) is kept for the texture fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import glb, raster
from .views import View

FACINGS = (0.0, 90.0, 180.0, 270.0)
FIT_LIMIT_DEG = 20.0
# Pitch registration on the side views (step 3b).
POSE_VIEWS = ("left", "right")
POSE_STEP_DEG = 2.0         # sweep step; the best is refined to half a step
POSE_MARGIN_DEG = 10.0      # the sweep runs this far past the limit, so a best beyond it is seen, not clipped
POSE_MIN_RANGE = 0.02       # flat: the side IoU varies by less than this over the whole sweep
POSE_AMBIGUOUS = 0.01       # ambiguous: another peak this close to the best ...
POSE_PEAK_SEP_DEG = 8.0     # ... more than this far from it
POSE_OFFSET_NOTE = ("not fitted: the contract frame centres Y on the figure's bounding box, as the views "
                    "centre u on each mask's box, and the pitch is fitted in that frame")
CONFORM_ITERS = 3
CONFORM_SIGMA_PX = 6.0
CONFORM_CAP_PX = 30.0
# Remesh resolution (voxels along the height). Front IoU after remeshing the pencil-sketch figure's source (the
# result recorded by the pinned runner; the conformed source: 0.980): 0.968 at 192 (86,788 triangles), 0.965 at
# 256 (159,816), 0.963 at 320. A result recorded before the weights were pinned measured 0.957 at 192 and 0.976
# at 256, so the resolution matters less than the run-to-run variation of the source; 256 keeps more of its
# detail for the full-detail profile.
REMESH_NZ = 256


@dataclass
class SourceMesh:
    verts: np.ndarray          # [N, 3] welded, Z-up, source units
    faces: np.ndarray          # [M, 3]
    corner_uv: np.ndarray      # [M, 3, 2] glTF texcoords (v down)
    texture: np.ndarray | None  # [H, W, 3] float RGB


def load_source(path) -> SourceMesh:
    """The source GLB (read by `glb.load_glb`, already in the contract axes) welded, with corner UVs kept."""
    g = glb.load_glb(path)
    v, raw_faces = g.merged()
    if len(raw_faces) == 0:
        raise glb.GlbError(f"{path} holds no triangle mesh")
    _, first, inverse = np.unique(np.round(v, 6), axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[raw_faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    uv = g.merged_uv()
    uv = uv if uv is not None else np.zeros((len(v), 2))
    return SourceMesh(v[first], faces[keep], uv[raw_faces[keep]], g.texture_rgb())


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Source -> contract frame: the yaw about +Z (the facing and the small fitted yaw) first, then the pitch
    about +X (the body's side-to-side axis once it faces -Y) and the roll about +Y (the front view's axis), so
    pitch and roll mean the same tilt of the body whatever the facing."""
    a, b, c = (math.radians(x) for x in (yaw, pitch, roll))
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    ry = np.array([[math.cos(c), 0, math.sin(c)], [0, 1, 0], [-math.sin(c), 0, math.cos(c)]])
    return ry @ rx @ rz


def normalise(v: np.ndarray, height: float) -> np.ndarray:
    lo, hi = v.min(axis=0), v.max(axis=0)
    s = height / max(hi[2] - lo[2], 1e-12)
    return (v - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])) * s


def perspective_to_ortho(v: np.ndarray, distance: float, cam_height: float, height: float) -> np.ndarray:
    """Scale X/Z about a camera on -Y at `distance` (in body heights) and `cam_height` (fraction of height)."""
    d = distance * height
    k = d / np.maximum(d + v[:, 1], 1e-6)
    hz = cam_height * height
    return np.stack([v[:, 0] * k, v[:, 1], hz + (v[:, 2] - hz) * k], axis=1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _downsample(mask: np.ndarray, shape) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(mask).resize((shape[1], shape[0]), Image.NEAREST))


def _place(src: SourceMesh, p: dict, height: float) -> np.ndarray:
    v = normalise(src.verts @ rotation(p["yaw"], p["pitch"], p["roll"]).T, height)
    if p.get("distance"):
        v = normalise(perspective_to_ortho(v, p["distance"], p["cam_height"], height), height)
    return v


def outline_field(src: np.ndarray, dst: np.ndarray, sigma: float = CONFORM_SIGMA_PX, cap: float = CONFORM_CAP_PX,
                  ind_in=None, ind_out=None) -> tuple[np.ndarray, np.ndarray, int]:
    """A smooth 2D displacement field (rows, cols; pixels) taking the outline of mask `src` onto mask `dst`.

    Rim pixels of `src` move to the nearest `dst` pixel when outside `dst`, or to the nearest background pixel
    when inside it if that move points along the rim's outward normal; moves are capped at `cap` and blended
    with a Gaussian of `sigma` (normalised by the rim weight, faded to zero away from the rim). Also returns the
    number of pixels where the mapping folds (Jacobian determinant <= 0)."""
    from scipy import ndimage

    if ind_in is None:
        _, ind_in = ndimage.distance_transform_edt(~dst, return_indices=True)
    if ind_out is None:
        _, ind_out = ndimage.distance_transform_edt(dst, return_indices=True)
    rim = src & ~ndimage.binary_erosion(src)
    rr, cc = np.nonzero(rim)
    outside = ~dst[rr, cc]
    tr = np.where(outside, ind_in[0][rr, cc], ind_out[0][rr, cc]).astype(np.float64)
    tc = np.where(outside, ind_in[1][rr, cc], ind_out[1][rr, cc]).astype(np.float64)
    dr, dc = tr - rr, tc - cc
    sd = ndimage.distance_transform_edt(src) - ndimage.distance_transform_edt(~src)
    gr, gc = np.gradient(ndimage.gaussian_filter(sd, 1.5))
    nr, nc = -gr[rr, cc], -gc[rr, cc]                   # outward normal of the source outline
    length = np.hypot(dr, dc)
    outward = (dr * nr + dc * nc) / (np.hypot(nr, nc) + 1e-9) > 0.5 * length
    k = np.where(outside | outward, np.minimum(1.0, cap / np.maximum(length, 1e-9)), 0.0)
    w = np.zeros(src.shape)
    fr, fc = np.zeros(src.shape), np.zeros(src.shape)
    w[rr, cc], fr[rr, cc], fc[rr, cc] = 1.0, dr * k, dc * k
    ws = ndimage.gaussian_filter(w, sigma)
    fade = np.clip(ws / (0.02 * ndimage.gaussian_filter(np.ones(src.shape), sigma)), 0, 1)
    field_r = ndimage.gaussian_filter(fr, sigma) / np.maximum(ws, 1e-6) * fade
    field_c = ndimage.gaussian_filter(fc, sigma) / np.maximum(ws, 1e-6) * fade
    jr_r, jr_c = np.gradient(field_r)
    jc_r, jc_c = np.gradient(field_c)
    det = (1 + jr_r) * (1 + jc_c) - jr_c * jc_r
    return field_r, field_c, int(((det <= 0) & (fade > 0)).sum())


def conform(verts: np.ndarray, faces: np.ndarray, front: View, iters: int = CONFORM_ITERS,
            sigma: float = CONFORM_SIGMA_PX, cap: float = CONFORM_CAP_PX) -> tuple[np.ndarray, dict]:
    """Pull the front outline onto the front mask (see the module notes). Returns new vertices and stats."""
    from scipy import ndimage

    v = verts.copy()
    mask = front.mask
    _, ind_in = ndimage.distance_transform_edt(~mask, return_indices=True)    # nearest figure pixel
    _, ind_out = ndimage.distance_transform_edt(mask, return_indices=True)    # nearest background pixel
    history, folded = [], 0
    for _ in range(iters):
        s = raster.render_view(front, v, faces).mask
        history.append(round(iou(s, mask), 4))
        field_r, field_c, fold = outline_field(s, mask, sigma, cap, ind_in, ind_out)
        folded = max(folded, fold)
        rows, cols, _ = raster.project(front.name, front.pps, front.cmid, front.vref, v)
        d = raster.bilinear(np.stack([field_r, field_c], axis=-1), rows - 0.5, cols - 0.5)
        v[:, 0] += d[:, 1] / front.pps
        v[:, 2] -= d[:, 0] / front.pps
    final = raster.render_view(front, v, faces).mask
    history.append(round(iou(final, mask), 4))
    moved = np.linalg.norm(v - verts, axis=1)
    return v, {"iou_by_iteration": history, "folded_pixels": folded,
               "vertex_move_studs": {"mean": round(float(moved.mean()), 4), "max": round(float(moved.max()), 4)},
               "iterations": iters, "sigma_px": sigma, "cap_px": cap}


def pose_views(vs) -> tuple[dict, dict]:
    """The observed side views (`POSE_VIEWS`) the pitch is registered on, and why each other one is not used. A
    view stage 2 derived from the others (mask_method "visual_hull") is not an observation; `load_views` skips a
    missing or empty mask (and notes an empty one). Held-out views are never in a `ViewSet`."""
    method = vs.meta.get("mask_method") or {}
    use, skip = {}, {}
    for name in POSE_VIEWS:
        if name not in vs.views:
            empty = any(n.startswith(f"{name} mask is empty") for n in vs.notes)
            skip[name] = "empty mask" if empty else "missing mask"
        elif "visual_hull" in str(method.get(name, "")):
            skip[name] = f"derived view ({method[name]})"
        else:
            use[name] = vs.views[name]
    return use, skip


def fit_pitch(src: SourceMesh, p: dict, sides: dict, height: float, excluded: dict | None = None,
              scale: float = 0.5) -> tuple[float, dict]:
    """Register the pitch on the observed side masks (see the module notes, step 3b). Returns the pitch in
    degrees (0 when the fit is degenerate) and its record."""
    rec: dict = {"views": sorted(sides), "excluded": dict(excluded or {}), "limit_deg": FIT_LIMIT_DEG,
                 "y_offset_studs": 0.0, "y_offset_note": POSE_OFFSET_NOTE}

    def fallback(reason: str) -> tuple[float, dict]:
        rec.update(status="fallback", reason=reason, pitch_deg=0.0)
        return 0.0, rec

    absent = {k: rec["excluded"].get(k, "not given") for k in POSE_VIEWS if k not in sides}
    if absent:                          # one profile alone cannot be told from a figure seen slightly turned
        why = "; ".join(f"{k}: {v}" for k, v in sorted(absent.items()))
        return fallback(f"needs both observed side views ({why})")
    small = {k: _downsample(v.mask, (int(round(v.mask.shape[0] * scale)), int(round(v.mask.shape[1] * scale))))
             for k, v in sides.items()}
    cache: dict[float, float] = {}

    def score(pitch: float) -> float:
        if pitch not in cache:
            v = _place(src, {**p, "pitch": pitch}, height)
            cache[pitch] = float(np.mean([iou(raster.render_view(sides[k], v, src.faces, scale).mask, small[k])
                                          for k in sides]))
        return cache[pitch]

    span = FIT_LIMIT_DEG + POSE_MARGIN_DEG
    pts = np.arange(-span, span + 1e-9, POSE_STEP_DEG)
    curve = {float(pt): score(float(pt)) for pt in pts}
    rec["sweep_half_res"] = {f"{k:g}": round(s, 4) for k, s in curve.items()}
    vals = np.array(list(curve.values()))
    rec["side_iou_half_res"] = {"pitch_0": round(curve[0.0], 4),
                                "note": "sweep scores (source placed by the step 3 fit, before the front refit, "
                                        "conform and remesh); the final side IoU is in the stage checks"}
    if vals.max() - vals.min() < POSE_MIN_RANGE:
        return fallback(f"flat: the side IoU varies by {vals.max() - vals.min():.4f} over the sweep "
                        f"(below {POSE_MIN_RANGE})")
    # every local maximum of the sweep that could be the best, refined to half a step
    half = POSE_STEP_DEG / 2
    peaks: dict[float, float] = {}
    for i, pt in enumerate(pts):
        if ((i == 0 or vals[i] >= vals[i - 1]) and (i == len(pts) - 1 or vals[i] >= vals[i + 1])
                and vals[i] > vals.max() - 3 * POSE_AMBIGUOUS):
            s, q = max((score(float(q)), float(q)) for q in (pt - half, pt, pt + half))
            peaks[q] = s
    best = max(peaks, key=peaks.get)
    s_best = peaks[best]
    rivals = [q for q in peaks if abs(q - best) > POSE_PEAK_SEP_DEG and peaks[q] > s_best - POSE_AMBIGUOUS]
    if rivals:
        return fallback(f"ambiguous: the side IoU peaks at {best:g} and {rivals[0]:g} degrees "
                        f"({s_best:.4f}, {peaks[rivals[0]]:.4f})")
    if abs(best) > FIT_LIMIT_DEG:
        return fallback(f"the best pitch, {best:g} degrees, lies beyond the {FIT_LIMIT_DEG:g} degree limit")
    rec["side_iou_half_res"]["fitted"] = round(s_best, 4)    # the sweep's score, before the refit and conform
    rec.update(status="fitted", reason=None, pitch_deg=best)
    return best, rec


def _front_fit(src: SourceMesh, front: View, height: float, facing: float, pitch: float,
               scale: float = 0.5) -> tuple[dict, float, float]:
    """Yaw, roll and the perspective correction by coordinate descent on the front IoU at `scale` resolution,
    with the pitch held. Returns the parameters, the IoU at the facing alone and the fitted IoU."""
    shape = (int(round(front.mask.shape[0] * scale)), int(round(front.mask.shape[1] * scale)))
    small = _downsample(front.mask, shape)

    def score(p: dict) -> float:
        return iou(raster.render_view(front, _place(src, p, height), src.faces, scale).mask, small)

    p = {"yaw": facing, "pitch": pitch, "roll": 0.0, "distance": 0.0, "cam_height": 0.0}
    s0 = score({**p, "pitch": 0.0})
    # perspective correction: coarse grid first (0 = none)
    grid = [(0.0, 0.0)] + [(d, h) for d in (1.0, 1.5, 2.0, 3.0, 5.0) for h in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0)]
    cand = max(grid, key=lambda dh: score({**p, "distance": dh[0], "cam_height": dh[1]}))
    p["distance"], p["cam_height"] = cand
    best_s = score(p)
    steps = {"yaw": (4, 2, 1), "roll": (4, 2, 1), "distance": (0.5, 0.25, 0.1), "cam_height": (0.25, 0.1, 0.05)}
    for _ in range(3):
        changed = False
        for k, ladder in steps.items():
            if k in ("distance", "cam_height") and not p["distance"]:
                continue
            for st in ladder:
                for sgn in (-1, 1):
                    q = dict(p)
                    q[k] += sgn * st
                    if k in ("yaw", "roll") and abs(q[k] - (facing if k == "yaw" else 0.0)) > FIT_LIMIT_DEG:
                        continue
                    if k == "distance" and q[k] < 0.5:
                        continue
                    sc = score(q)
                    if sc > best_s + 1e-5:
                        best_s, p, changed = sc, q, True
        if not changed:
            break
    return p, s0, best_s


def orient(src: SourceMesh, front: View, front_img: np.ndarray | None, height: float,
           conform_outline: bool = True, sides: dict | None = None,
           sides_excluded: dict | None = None) -> tuple[np.ndarray, dict]:
    """Source mesh -> contract-frame vertices (faces unchanged). The front view decides everything but the
    pitch, which is registered on `sides` (observed side views, from `pose_views`; none: pitch 0)."""
    from .texture import compare

    info: dict = {"facing_search": {}}
    best = None
    for yaw in FACINGS:
        p = {"yaw": yaw, "pitch": 0.0, "roll": 0.0}
        v = _place(src, p, height)
        fr = raster.render_view(front, v, src.faces)
        row = {"iou": round(iou(fr.mask, front.mask), 4)}
        score = row["iou"]
        if front_img is not None and src.texture is not None:
            c = compare(raster.sample_texture(fr, src.corner_uv, src.texture), fr.mask, front_img, front.mask)
            row["colour_error"] = c["mean_abs_error"]
            score -= c["mean_abs_error"] / 100.0
        info["facing_search"][str(int(yaw))] = row
        if best is None or score > best[0]:
            best = (score, yaw)
    facing = best[1]
    info["facing_deg"] = facing

    # the front fit with the pitch held at 0, the pitch registered on the side views, then the front fit again
    # with that pitch (the front view barely reacts to pitch, so it is not left to choose one)
    p, s0, best_s = _front_fit(src, front, height, facing, 0.0)
    pitch, info["pose_fit"] = fit_pitch(src, p, sides or {}, height, sides_excluded)
    if pitch:
        p, _, best_s = _front_fit(src, front, height, facing, pitch)
    info["fit"] = {k: round(float(x), 3) for k, x in p.items()}
    info["fit_note"] = "distance 0 = no perspective correction; distance in body heights, cam_height in heights"
    info["front_iou_half_res"] = {"facing_only": round(s0, 4), "fitted": round(best_s, 4)}
    v = _place(src, p, height)
    info["front_iou_fitted"] = round(iou(raster.render_view(front, v, src.faces).mask, front.mask), 4)
    if conform_outline:
        v, info["conform"] = conform(v, src.faces, front)
        v = normalise(v, height)
    info["front_iou_final_numpy"] = round(iou(raster.render_view(front, v, src.faces).mask, front.mask), 4)
    return v, info


def voxelise(verts: np.ndarray, faces: np.ndarray, voxel: float, pad: int = 2):
    """Occupancy grid of a closed mesh by ray parity along +Y. Returns (occ [nx, ny, nz], origin)."""
    lo, hi = verts.min(axis=0) - pad * voxel, verts.max(axis=0) + pad * voxel
    xs = np.arange(lo[0], hi[0] + voxel, voxel) + 1.3e-5        # a tiny offset keeps rays off vertices
    ys = np.arange(lo[1], hi[1] + voxel, voxel)
    zs = np.arange(lo[2], hi[2] + voxel, voxel) + 0.7e-5
    # columns (x_i, z_k) as image pixels: row = k, col = i; pixel centres at the grid values
    rows = (verts[:, 2] - zs[0]) / voxel + 0.5
    cols = (verts[:, 0] - xs[0]) / voxel + 0.5
    tri_r, tri_c, tri_y = rows[faces], cols[faces], verts[faces][:, :, 1]
    hit_pix, hit_y = [], []
    nr, nc = len(zs), len(xs)
    for s0 in range(0, len(faces), 4096):
        r, c, y = tri_r[s0:s0 + 4096], tri_c[s0:s0 + 4096], tri_y[s0:s0 + 4096]
        r0 = np.clip(np.ceil(r.min(axis=1) - 0.5).astype(np.int64), 0, nr - 1)
        r1 = np.clip(np.floor(r.max(axis=1) - 0.5).astype(np.int64), 0, nr - 1)
        c0 = np.clip(np.ceil(c.min(axis=1) - 0.5).astype(np.int64), 0, nc - 1)
        c1 = np.clip(np.floor(c.max(axis=1) - 0.5).astype(np.int64), 0, nc - 1)
        nh, nw = np.maximum(r1 - r0 + 1, 0), np.maximum(c1 - c0 + 1, 0)
        n = nh * nw
        t = np.repeat(np.arange(len(r)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        pr = r0[t] + k // np.maximum(nw[t], 1)
        pc = c0[t] + k % np.maximum(nw[t], 1)
        qy, qx = pr + 0.5, pc + 0.5
        ay, ax = r[t, 0], c[t, 0]
        v0y, v0x, v1y, v1x = r[t, 1] - ay, c[t, 1] - ax, r[t, 2] - ay, c[t, 2] - ax
        v2y, v2x = qy - ay, qx - ax
        den = v0x * v1y - v1x * v0y
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        l1 = (v2x * v1y - v1x * v2y) / den
        l2 = (v0x * v2y - v2x * v0y) / den
        l0 = 1 - l1 - l2
        inside = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        b = np.stack([l0, l1, l2], axis=1)[inside]
        hit_pix.append(pr[inside] * nc + pc[inside])
        hit_y.append(np.einsum("nk,nk->n", b, y[t[inside]]))
    pix, hy = np.concatenate(hit_pix), np.concatenate(hit_y)
    order = np.lexsort((hy, pix))
    pix, hy = pix[order], hy[order]
    occ = np.zeros((nc, len(ys), nr), dtype=bool)
    bounds = np.flatnonzero(np.diff(np.concatenate([[-1], pix, [-1]])))
    for a, b2 in zip(bounds[:-1], bounds[1:]):
        p = pix[a]
        cnt = np.searchsorted(hy[a:b2], ys)                     # crossings in front of each voxel centre
        occ[p % nc, :, p // nc] = (cnt % 2) == 1
    return occ, np.array([xs[0], ys[0], zs[0]])


def well_composed(field: np.ndarray, level: float = 0.5, iters: int = 10) -> tuple[np.ndarray, int]:
    """Raise just above `level` every voxel needed so the inside set has no diagonal-only contacts (a 2x2
    face pattern 10/01, or a 2x2x2 cube whose only inside corners are opposite): with them, surface nets makes
    edges shared by four triangles. Returns the field and the number of voxels raised."""
    f = field.copy()
    total = 0
    for _ in range(iters):
        b = f > level
        fix = np.zeros_like(b)
        # 2D diagonal patterns in each axis pair: (1,0;0,1) or (0,1;1,0) -> fill the zeros
        for ax in ((0,1),(0,2),(1,2)):
            def sl(i, j, ax=ax):
                return tuple(slice(i, b.shape[k]-1+i) if k==ax[0] else slice(j, b.shape[k]-1+j) if k==ax[1] else slice(None) for k in range(3))

            a00,a01,a10,a11 = b[sl(0,0)],b[sl(0,1)],b[sl(1,0)],b[sl(1,1)]
            d1 = a00 & a11 & ~a01 & ~a10
            d2 = a01 & a10 & ~a00 & ~a11
            fx01 = np.zeros_like(b); fx01[sl(0,1)] |= d1; fix |= fx01
            fx10 = np.zeros_like(b); fx10[sl(1,0)] |= d1; fix |= fx10
            fx00 = np.zeros_like(b); fx00[sl(0,0)] |= d2; fix |= fx00
            fx11 = np.zeros_like(b); fx11[sl(1,1)] |= d2; fix |= fx11
        # 3D: a 2x2x2 cube whose only inside corners are a body diagonal pair
        n = b.shape
        def c(i,j,k): return b[i:n[0]-1+i, j:n[1]-1+j, k:n[2]-1+k]
        corners = {(i,j,k): c(i,j,k) for i in (0,1) for j in (0,1) for k in (0,1)}
        cnt = sum(v.astype(int) for v in corners.values())
        for (i,j,k), v in corners.items():
            opp = corners[(1-i,1-j,1-k)]
            m = v & opp & (cnt == 2)
            if m.any():
                # fill one neighbour along x of corner (i,j,k)
                t = np.zeros_like(b); t[1-i:n[0]-1+1-i, j:n[1]-1+j, k:n[2]-1+k] |= m; fix |= t
        fix &= ~b
        if not fix.any():
            break
        total += int(fix.sum())
        f[fix] = level + 0.05
    return f, total


def remesh(verts: np.ndarray, faces: np.ndarray, height: float, nz: int = REMESH_NZ, sigma: float = 0.8,
           smooth_iters: int = 10) -> tuple[np.ndarray, np.ndarray, dict]:
    """Closed manifold surface of the same shape (see the module notes, step 6), in the contract frame."""
    from . import carve

    voxel = height / nz
    occ, origin = voxelise(verts, faces, voxel)
    field, raised = well_composed(carve.blur(occ, sigma))
    v, f = carve.surface_nets(field, origin, voxel)
    v, f, dropped = carve.keep_large_components(v, f)
    v = carve.taubin(v, f, smooth_iters)
    v = normalise(v, height)
    return v, f, {"voxel_size_studs": round(voxel, 5), "voxel_resolution": [int(x) for x in occ.shape],
                  "voxels_kept": int(occ.sum()), "voxels_raised_well_composed": raised, "dropped_island_faces": dropped, "blur_sigma_voxels": sigma,
                  "taubin_iterations": smooth_iters, "triangles": int(len(f))}
