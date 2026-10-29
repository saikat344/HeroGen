"""Colour projection from the turnaround views onto the UV-unwrapped body, and its measurement.

For every texel of a section's UV atlas the matching surface point is projected into the observed side views
(front, back, left, right; a top view that stage 2 derived from the others carries no colour of its own and
is skipped). Each view's weight is the cosine between the surface normal and the direction to that view's
camera, raised to `FACING_POWER` so the view that faces the surface dominates while seams still blend, and is
zero where the point is hidden in that view (a depth map of the body splatted from the texels themselves) or
falls outside the view's mask. Empty texels next to islands are filled outward so filtering never pulls in
background.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .views import FORWARD, PROJ, ViewSet

COLOUR_VIEWS = ("front", "back", "left", "right")
FACING_POWER = 4.0
DEPTH_TOL = 0.04          # studs: a texel within this of the view's nearest surface counts as visible


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


def bake(sections: dict, uvs: dict, views: dict, size: int, zbuf: dict | None = None):
    """Bake one atlas for a LOD. `uvs[name]` is [faces, 3, 2] in [0, 1]. Returns (atlas uint8, stats, zbuf)."""
    pos, uv, nrm = [], [], []
    for name, s in sections.items():
        if name not in uvs or len(s.faces) == 0:
            continue
        tri = s.verts[s.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        pos.append(tri)
        uv.append(uvs[name])
        nrm.append(n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None])
    pos, uv, nrm = np.concatenate(pos), np.concatenate(uv), np.concatenate(nrm)
    tidx, p3, tri = _texels(pos, uv, size)
    n3 = nrm[tri]
    if zbuf is None:
        zbuf = depth_maps(views, p3)
    acc = np.zeros((len(p3), 3))
    wsum = np.zeros(len(p3))
    fallback = np.zeros((len(p3), 3))
    fb_w = np.zeros(len(p3))
    for name, (view, img) in views.items():
        fwd = FORWARD[name]
        facing = np.clip(-(n3 @ fwd), 0, 1)
        row, col = _project(view, p3)
        nimg = img.shape[0]
        r, c = np.round(row).astype(np.int64), np.round(col).astype(np.int64)
        ok = (r >= 0) & (r < nimg) & (c >= 0) & (c < nimg)
        rr, cc = np.clip(r, 0, nimg - 1), np.clip(c, 0, nimg - 1)
        inmask = ok & view.mask[rr, cc]
        visible = (p3 @ fwd) <= zbuf[name][rr, cc] + DEPTH_TOL
        colour = _bilinear(img, row, col)
        w = facing ** FACING_POWER * (inmask & visible)
        acc += w[:, None] * colour
        wsum += w
        fw = (facing + 1e-3) * inmask
        fallback += fw[:, None] * colour
        fb_w += fw
    col = np.where((wsum > 1e-6)[:, None], acc / np.maximum(wsum, 1e-12)[:, None],
                   fallback / np.maximum(fb_w, 1e-12)[:, None])
    atlas = np.zeros((size * size, 3))
    filled = np.zeros(size * size, dtype=bool)
    cnt = np.zeros(size * size)
    np.add.at(atlas, tidx, col)
    np.add.at(cnt, tidx, 1)
    filled[tidx] = True
    atlas[filled] /= cnt[filled][:, None]
    stats = {"texels_used": int(filled.sum()), "atlas_fill": round(float(filled.mean()), 4),
             "texels_seen_by_a_view": round(float((wsum > 1e-6).mean()), 4)}
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
