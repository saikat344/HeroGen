"""Colour projection from the turnaround views onto the UV-unwrapped body, and its measurement.

For every texel of a section's UV atlas the matching surface point is projected into the observed side views
(front, back, left, right; a top view that stage 2 derived from the others carries no colour of its own and
is skipped). Each view's weight is the cosine between the surface normal and the direction to that view's
camera, raised to `FACING_POWER` so the view that faces the surface dominates while seams still blend, and is
zero where the point is hidden in that view (a depth map of the body splatted from the texels themselves) or
falls outside the view's mask. Empty texels next to islands are filled outward so filtering never pulls in
background.

Surfaces no view faces: with a source texture (mesh source `sam3d-objects`, whose own baked texture covers every
surface), a texel whose best facing cosine among the views that see it is below `FALLBACK_FACING` takes the
source texture's colour at the nearest source-surface point instead, blended linearly in between. Without one,
such texels average the views that contain them (facing-weighted, ignoring occlusion): a smear.


Close-ups (the face source, `herogen.pipeline.facesource`, stage 2f): stage 2f may leave sharp close-ups of the
head for the front, left and right views, each with its affine A (close-up pixels to view pixels) and its footprint
(`masks/closeup-<view>.png`). `load_closeups` takes the ones `views/closeup.json` marks used whose view image is
still the one they were made from (its SHA-256; a record from a stage 2f that fell back gives the note "stage 2f
fell back: <reason>"), and feathers each footprint over `CLOSEUP_FEATHER_PX` (the
footprint's alpha stays 1 inside and falls to 0 over about that width outside, never past the crop box). In `bake`,
a close-up adds detail to its view, it never replaces the view's colour: inside the footprint (alpha-blended at
its feathered edge) the view's colour gets the close-up at the texture's sampling density minus the close-up at
the view's own density. Each is the close-up blurred with a Gaussian of sigma `CLOSEUP_PREFILTER` x close-up pixels
per texel (per view pixel), about a box filter over one texel (view pixel); the texel density is the baked
sections' texel count over their surface area. So the view keeps every colour it shows at its own resolution, and
the close-up only adds what lies above it; where the texture is no finer than the view, nothing changes. Weights,
visibility and masks are unchanged. Stage 4 passes close-ups only to the Head's own texture.

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .views import FORWARD, PROJ, ViewSet

COLOUR_VIEWS = ("front", "back", "left", "right")
FACING_POWER = 4.0
DEPTH_TOL = 0.04          # studs: a texel within this of the view's nearest surface counts as visible
FALLBACK_FACING = 0.3     # below this best facing cosine (about 72.5 degrees) the source texture takes over
CLOSEUP_VIEWS = ("front", "left", "right")
CLOSEUP_RECORD = "views/closeup.json"
CLOSEUP_FEATHER_PX = 3.0  # view pixels
# Gaussian sigma of the prefilter, in close-up pixels per texel: about a box filter over one texel.
CLOSEUP_PREFILTER = 0.42


@dataclass
class Closeup:
    """A close-up of one view (stage 2f): its image, the affine A from close-up pixel edge coordinates to view
    pixel edge coordinates and the feathered footprint alpha in the view's pixels."""
    view: str
    img: np.ndarray             # float RGB [S, S, 3]
    A: np.ndarray               # [2, 3]
    alpha: np.ndarray           # float [rows, cols] in [0, 1], the view's shape


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (edge-padded) of a [H, W] or [H, W, C] float image."""
    if sigma < 0.3:
        return img.astype(np.float64, copy=True)
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = img.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        out = sum(w * np.take(p, np.arange(i, i + n), axis=axis) for i, w in enumerate(k))
    return out


def load_closeups(run: Path, vs: ViewSet) -> tuple[dict[str, Closeup], dict]:
    """The usable close-ups of the run (module notes) and a record of what was used and why not."""
    import hashlib

    from PIL import Image

    from .views import load_mask

    p = run / CLOSEUP_RECORD
    if not p.exists():
        return {}, {"used": [], "notes": ["no close-ups in the run (stage 2f made none)"]}
    try:
        rec = json.loads(p.read_text())
        if not isinstance(rec, dict) or not isinstance(rec.get("views"), dict):
            raise ValueError("invalid close-up record")
    except (OSError, ValueError):
        return {}, {"used": [], "notes": [f"{CLOSEUP_RECORD} could not be read"]}
    out: dict[str, Closeup] = {}
    notes: list[str] = []
    if rec.get("fallback"):
        notes.append(f"stage 2f fell back: {str(rec['fallback'])[:600]}")
    for name in CLOSEUP_VIEWS:
        r = rec["views"].get(name) or {}
        if not isinstance(r, dict):
            notes.append(f"{name}: could not be read (invalid view record)")
            continue
        if not r.get("used"):
            if r.get("reason"):
                notes.append(f"{name}: {r['reason']}")
            continue
        view = vs.views.get(name)
        vpath = run / "views" / f"{name}.png"
        if view is None or not vpath.exists():
            notes.append(f"{name}: no such view")
            continue
        if hashlib.sha256(vpath.read_bytes()).hexdigest() != r.get("view_sha256"):
            notes.append(f"{name}: made from another {name} view than the run's current one; not used")
            continue
        try:
            with Image.open(run / r["file"]) as im:
                img = np.asarray(im.convert("RGB"), dtype=np.float64)
            foot = load_mask(run / r["footprint"])
            A = np.asarray(r["A"], dtype=np.float64)
            l, t, rgt, b = (float(x) for x in r["crop_box"])
        except (OSError, KeyError, ValueError, TypeError) as e:
            notes.append(f"{name}: could not be read ({type(e).__name__})")
            continue
        if foot.shape != view.mask.shape or img.shape[0] != img.shape[1] or A.shape != (2, 3):
            notes.append(f"{name}: sizes do not match the view")
            continue
        if (img.shape[0] < 2 or not np.isfinite(A).all() or abs(np.linalg.det(A[:, :2])) < 1e-12
                or not np.isfinite([l, t, rgt, b]).all() or l >= rgt or t >= b):
            notes.append(f"{name}: invalid transform or crop box; not used")
            continue
        rows = np.arange(foot.shape[0]) + 0.5
        cols = np.arange(foot.shape[1]) + 0.5
        box = ((rows >= t) & (rows < b))[:, None] & ((cols >= l) & (cols < rgt))[None, :]
        alpha = np.minimum(_blur(foot.astype(np.float64), CLOSEUP_FEATHER_PX) * 2.0, 1.0) * box
        out[name] = Closeup(name, img, A, alpha)
    return out, {"used": sorted(out), "notes": notes}


def _closeup_colour(cu: Closeup, view, row: np.ndarray, col: np.ndarray, colour: np.ndarray,
                    texel_ppu: float | None) -> tuple[np.ndarray, np.ndarray]:
    """The view's colour plus the close-up's detail inside its footprint (module notes), and the close-up's
    alpha. ``row``, ``col`` are `_project`'s (pixel-centre based)."""
    a = _bilinear(cu.alpha[..., None], row, col)[:, 0]
    Ainv = np.linalg.inv(np.vstack([cu.A, [0.0, 0.0, 1.0]]))[:2]
    x = Ainv[0, 0] * (col + 0.5) + Ainv[0, 1] * (row + 0.5) + Ainv[0, 2]
    y = Ainv[1, 0] * (col + 0.5) + Ainv[1, 1] * (row + 0.5) + Ainv[1, 2]
    S = cu.img.shape[0]
    a = a * ((x >= 0) & (x < S) & (y >= 0) & (y < S))
    per_view_px = 1.0 / float(np.sqrt(abs(np.linalg.det(cu.A[:, :2]))))       # close-up px per view px
    per_texel = per_view_px * view.pps / texel_ppu if texel_ppu else 1.0       # close-up px per texel
    if per_texel >= per_view_px:          # the texture is no finer than the view: nothing to add
        return colour, np.zeros_like(a)
    fine = _blur(cu.img, CLOSEUP_PREFILTER * per_texel) if per_texel > 1.05 else cu.img
    coarse = _blur(cu.img, CLOSEUP_PREFILTER * per_view_px)
    detail = _bilinear(fine, y - 0.5, x - 0.5) - _bilinear(coarse, y - 0.5, x - 0.5)
    return colour + a[:, None] * detail, a



class SourceColour:
    """Colour of the source mesh's own texture at the nearest point of its surface (dense surface samples)."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray, corner_uv: np.ndarray, texture: np.ndarray,
                 samples_per_edge: int = 4):
        from scipy.spatial import cKDTree

        n = samples_per_edge
        bary = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
        tri = verts[faces]                                        # [M, 3, 3]
        pts = np.einsum("bk,mkj->mbj", bary, tri).reshape(-1, 3)
        uv = np.einsum("bk,mkj->mbj", bary, corner_uv).reshape(-1, 2)
        self.tree = cKDTree(pts)
        self.uv = uv
        self.texture = texture

    def __call__(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist, idx = self.tree.query(p, workers=-1)
        uv = self.uv[idx]
        t = self.texture
        return _bilinear(t, uv[:, 1] * t.shape[0] - 0.5, uv[:, 0] * t.shape[1] - 0.5), dist


def observed_colour_views(vs: ViewSet, run: Path) -> dict:
    """View name -> (View, float RGB image) for the observed side views that have a colour image."""
    from PIL import Image

    out = {}
    method = vs.meta.get("mask_method") or {}
    for name in COLOUR_VIEWS:
        if name not in vs.views or "visual_hull" in str(method.get(name, "")):
            continue
        p = run / "views" / f"{name}.png"
        if not p.exists():
            continue
        with Image.open(p) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.float64)
        if img.shape[:2] != vs.views[name].mask.shape:
            continue
        out[name] = (vs.views[name], img)
    return out


def _texels(pos: np.ndarray, uv: np.ndarray, size: int, chunk: int = 4096):
    """Rasterise triangles in UV space. Returns (flat texel index, 3D position, triangle index)."""
    idx_all, p_all, t_all = [], [], []
    for s in range(0, len(pos), chunk):
        P, U = pos[s:s + chunk], uv[s:s + chunk] * size
        lo = np.floor(U.min(axis=1) - 0.5).astype(np.int64)
        hi = np.ceil(U.max(axis=1) - 0.5).astype(np.int64)
        lo, hi = np.clip(lo, 0, size - 1), np.clip(hi, 0, size - 1)
        w, h = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
        n = w * h
        tri = np.repeat(np.arange(len(P)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        x = lo[tri, 0] + k % w[tri]
        y = lo[tri, 1] + k // w[tri]
        q = np.stack([x + 0.5, y + 0.5], axis=1)
        a, b, c = U[tri, 0], U[tri, 1], U[tri, 2]
        v0, v1, v2 = b - a, c - a, q - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        l0 = 1 - l1 - l2
        inside = ok & (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        tri, x, y = tri[inside], x[inside], y[inside]
        bary = np.stack([l0[inside], l1[inside], l2[inside]], axis=1)
        p3 = np.einsum("nk,nkj->nj", bary, P[tri])
        row = size - 1 - y                  # image row 0 is the top (v = 1)
        idx_all.append(row * size + x)
        p_all.append(p3)
        t_all.append(tri + s)
    return np.concatenate(idx_all), np.concatenate(p_all), np.concatenate(t_all)


def _project(view, p: np.ndarray):
    """Float (row, col) of world points in a view image."""
    uv = p @ PROJ[view.name].T
    col = uv[:, 0] * view.pps + view.cmid - 0.5
    row = view.vref - uv[:, 1] * view.pps - 0.5
    return row, col


def depth_maps(views: dict, samples: np.ndarray) -> dict:
    """Per view, the nearest surface depth (along the view direction) at every pixel, from dense samples."""
    out = {}
    for name, (view, _) in views.items():
        n = view.mask.shape[0]
        row, col = _project(view, samples)
        r, c = np.round(row).astype(np.int64), np.round(col).astype(np.int64)
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        z = np.full(n * n, np.inf)
        np.minimum.at(z, r[ok] * n + c[ok], samples[ok] @ FORWARD[name])
        z = z.reshape(n, n)
        # fill splat holes: nearest surface over a 3x3 neighbourhood
        pad = np.pad(z, 1, constant_values=np.inf)
        z = np.min([pad[i:i + n, j:j + n] for i in range(3) for j in range(3)], axis=0)
        out[name] = z
    return out


def _bilinear(img: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    n = img.shape[0]
    r0 = np.clip(np.floor(row).astype(np.int64), 0, n - 2)
    c0 = np.clip(np.floor(col).astype(np.int64), 0, n - 2)
    fr = np.clip(row - r0, 0, 1)[:, None]
    fc = np.clip(col - c0, 0, 1)[:, None]
    return ((img[r0, c0] * (1 - fc) + img[r0, c0 + 1] * fc) * (1 - fr)
            + (img[r0 + 1, c0] * (1 - fc) + img[r0 + 1, c0 + 1] * fc) * fr)


def texel_samples(sections: dict, uvs: dict, size: int) -> np.ndarray:
    """The surface point of every texel the sections' UVs cover in a size x size texture."""
    keep = [n for n, s in sections.items() if n in uvs and len(s.faces)]
    if not keep:
        return np.zeros((0, 3))
    pos = np.concatenate([sections[n].verts[sections[n].faces] for n in keep])
    return _texels(pos, np.concatenate([uvs[n] for n in keep]), size)[1]


def bake(sections: dict, uvs: dict, views: dict, size: int, zbuf: dict | None = None,
         source: SourceColour | None = None, fallback_facing: float = FALLBACK_FACING, trust: dict | None = None,
         closeups: dict | None = None):
    """Bake one texture of a LOD for the given sections. `uvs[name]` is [faces, 3, 2] in [0, 1]. Returns
    (texture uint8, stats, zbuf); when only some of the body's sections are baked, pass the whole body's `zbuf`
    (`depth_maps` of its `texel_samples`) so the rest still hides what it covers. `trust` scales a view's weight
    (default 1 for every view). `closeups` ({view: Closeup}) the close-ups
    (module notes)."""
    pos, uv, nrm = [], [], []
    for name, s in sections.items():
        if name not in uvs or len(s.faces) == 0:
            continue
        tri = s.verts[s.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        pos.append(tri)
        uv.append(uvs[name])
        nrm.append(n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None])
    if not pos:
        # nothing to bake (an empty section, say the Head of a broken body): a blank texture, so stage 4 reaches
        # its gates (`sections_non_empty`) and fails there with a clear error instead of here
        return np.zeros((size, size, 3), dtype=np.uint8), {"texels_used": 0, "atlas_fill": 0.0,
                                                           "notes": ["no faces to bake"]}, zbuf
    pos, uv, nrm = np.concatenate(pos), np.concatenate(uv), np.concatenate(nrm)
    tidx, p3, tri = _texels(pos, uv, size)
    n3 = nrm[tri]
    texel_ppu = None
    if closeups:
        area = float(np.linalg.norm(np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0]), axis=1).sum() / 2)
        texel_ppu = float(np.sqrt(len(np.unique(tidx)) / area)) if area > 0 else None
    cu_alpha: dict = {}
    if zbuf is None:
        zbuf = depth_maps(views, p3)
    acc = np.zeros((len(p3), 3))
    wsum = np.zeros(len(p3))
    fallback = np.zeros((len(p3), 3))
    fb_w = np.zeros(len(p3))
    best_facing = np.zeros(len(p3))
    for name, (view, img) in views.items():
        fwd = FORWARD[name]
        facing = np.clip(-(n3 @ fwd), 0, 1)
        row, col = _project(view, p3)
        nimg = img.shape[0]
        r, c = np.round(row).astype(np.int64), np.round(col).astype(np.int64)
        ok = (r >= 0) & (r < nimg) & (c >= 0) & (c < nimg)
        rr, cc = np.clip(r, 0, nimg - 1), np.clip(c, 0, nimg - 1)
        visible = (p3 @ fwd) <= zbuf[name][rr, cc] + DEPTH_TOL
        inmask = ok & view.mask[rr, cc]
        colour = _bilinear(img, row, col)
        if closeups and name in closeups:
            colour, cu_alpha[name] = _closeup_colour(closeups[name], view, row, col, colour, texel_ppu)
        w = facing ** FACING_POWER * (inmask & visible) * (1.0 if trust is None else trust.get(name, 1.0))
        best_facing = np.maximum(best_facing, facing * (inmask & visible))
        acc += w[:, None] * colour
        wsum += w
        if name in cu_alpha:
            cu_alpha[name] = (cu_alpha[name], w)
        fw = (facing + 1e-3) * inmask
        fallback += fw[:, None] * colour
        fb_w += fw
    col = np.where((wsum > 1e-6)[:, None], acc / np.maximum(wsum, 1e-12)[:, None],
                   fallback / np.maximum(fb_w, 1e-12)[:, None])
    src_stats = {}
    if source is not None:
        alpha = np.clip(best_facing / fallback_facing, 0.0, 1.0)[:, None]
        need = alpha[:, 0] < 1.0
        src_col = np.zeros_like(col)
        if need.any():
            src_col[need], dist = source(p3[need])
            src_stats = {"source_nearest_distance_mean": round(float(dist.mean()), 4),
                         "source_nearest_distance_max": round(float(dist.max()), 4)}
        col = alpha * col + (1 - alpha) * src_col
        src_stats.update({"texels_source_blend": round(float(need.mean()), 4),
                          "texels_source_only": round(float((alpha[:, 0] == 0).mean()), 4),
                          "fallback_facing": fallback_facing})
    atlas = np.zeros((size * size, 3))
    filled = np.zeros(size * size, dtype=bool)
    cnt = np.zeros(size * size)
    np.add.at(atlas, tidx, col)
    np.add.at(cnt, tidx, 1)
    filled[tidx] = True
    atlas[filled] /= cnt[filled][:, None]
    stats = {"texels_used": int(filled.sum()), "atlas_fill": round(float(filled.mean()), 4),
             "texels_seen_by_a_view": round(float((wsum > 1e-6).mean()), 4),
             "texels_best_facing_below_fallback": round(float((best_facing < fallback_facing).mean()), 4), **src_stats}
    if closeups:
        share = sum((a * w for a, w in cu_alpha.values()), np.zeros(len(p3))) / np.maximum(wsum, 1e-12)
        stats["closeups"] = {
            "views": sorted(cu_alpha), "prefilter": CLOSEUP_PREFILTER,
            "texels_per_stud": None if texel_ppu is None else round(texel_ppu, 2),
            "texels_with_closeup_detail": int((share > 0.01).sum()),
            "closeup_weight_share_mean": round(float(share.mean()), 4),
            "texels_per_view": {v: int(((a > 0.01) & (w > 0)).sum()) for v, (a, w) in cu_alpha.items()}}
    atlas = atlas.reshape(size, size, 3)
    filled = filled.reshape(size, size)
    for _ in range(8):                      # pad islands outward
        pad_a = np.pad(atlas, ((1, 1), (1, 1), (0, 0)))
        pad_f = np.pad(filled, 1).astype(np.float64)
        s_a = sum(pad_a[1 + i:1 + i + size, 1 + j:1 + j + size] * pad_f[1 + i:1 + i + size, 1 + j:1 + j + size, None]
                  for i, j in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        s_f = sum(pad_f[1 + i:1 + i + size, 1 + j:1 + j + size] for i, j in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        grow = (~filled) & (s_f > 0)
        atlas[grow] = s_a[grow] / s_f[grow][:, None]
        filled |= grow
    return np.clip(atlas, 0, 255).astype(np.uint8), stats, zbuf


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray, sigma: float = 1.5) -> float:
    """Mean SSIM (grey levels 0-255, Gaussian window) over the masked pixels."""
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()

    def blur(x):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r, r)
            xp = np.pad(x, pad, mode="edge")
            x = sum(w * np.take(xp, np.arange(i, i + x.shape[axis]), axis=axis) for i, w in enumerate(k))
        return x

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a ** 2
    sbb = blur(b * b) - mu_b ** 2
    sab = blur(a * b) - mu_a * mu_b
    smap = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(smap[mask].mean()) if mask.any() else 0.0


def compare(render: np.ndarray, render_mask: np.ndarray, view_img: np.ndarray, view_mask: np.ndarray) -> dict:
    """Mean absolute colour error (0-255, averaged over RGB) and grey SSIM inside both silhouettes."""
    both = render_mask & view_mask
    # stay a few pixels inside the silhouettes so edge pixels (background in one, figure in the other) do not count
    pad = np.pad(both, 2)
    n_r, n_c = both.shape
    core = np.min([pad[2 + i:2 + i + n_r, 2 + j:2 + j + n_c] for i in (-2, 0, 2) for j in (-2, 0, 2)], axis=0)
    err = float(np.abs(render[core] - view_img[core]).mean()) if core.any() else 255.0
    grey = np.array([0.299, 0.587, 0.114])
    return {"mean_abs_error": round(err, 2), "ssim": round(ssim(render @ grey, view_img @ grey, core), 4),
            "pixels": int(core.sum())}
