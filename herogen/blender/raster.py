"""A small numpy rasteriser for orthographic views: per-pixel triangle id, barycentrics and depth (z-buffer).

Used where many quick renders are needed without a Blender round trip (orienting a source mesh against the
front view, sampling a source texture). Image coordinates follow `views.View`: column = u * pps + cmid and
row = vref - v * pps in pixel-edge units, so pixel (r, c) has its centre at (r + 0.5, c + 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .views import FORWARD, PROJ


@dataclass
class Frame:
    """Per-pixel result: triangle id (-1 = background), barycentrics [H, W, 3] and depth along the view."""
    tri: np.ndarray
    bary: np.ndarray
    depth: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return self.tri >= 0


def project(view_name: str, pps: float, cmid: float, vref: float, verts: np.ndarray):
    """(row, col) in pixel-edge units and depth along the view direction for world points."""
    uv = verts @ PROJ[view_name].T
    return vref - uv[:, 1] * pps, uv[:, 0] * pps + cmid, verts @ FORWARD[view_name]


def rasterize(rows: np.ndarray, cols: np.ndarray, depth: np.ndarray, faces: np.ndarray, shape: tuple[int, int],
              chunk: int = 4096) -> Frame:
    """Nearest-surface rasterisation of triangles given their vertices' (row, col, depth)."""
    h, w = shape
    best_d = np.full(h * w, np.inf)
    best_t = np.full(h * w, -1, dtype=np.int64)
    best_b = np.zeros((h * w, 3))
    R, C, D = rows[faces], cols[faces], depth[faces]          # [M, 3]
    for s in range(0, len(faces), chunk):
        r, c, d = R[s:s + chunk], C[s:s + chunk], D[s:s + chunk]
        r0 = np.clip(np.ceil(r.min(axis=1) - 0.5).astype(np.int64), 0, h - 1)
        r1 = np.clip(np.floor(r.max(axis=1) - 0.5).astype(np.int64), 0, h - 1)
        c0 = np.clip(np.ceil(c.min(axis=1) - 0.5).astype(np.int64), 0, w - 1)
        c1 = np.clip(np.floor(c.max(axis=1) - 0.5).astype(np.int64), 0, w - 1)
        nh, nw = np.maximum(r1 - r0 + 1, 0), np.maximum(c1 - c0 + 1, 0)
        n = nh * nw
        if n.sum() == 0:
            continue
        t = np.repeat(np.arange(len(r)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        pr = r0[t] + k // np.maximum(nw[t], 1)
        pc = c0[t] + k % np.maximum(nw[t], 1)
        qy, qx = pr + 0.5, pc + 0.5
        ay, ax = r[t, 0], c[t, 0]
        v0y, v0x = r[t, 1] - ay, c[t, 1] - ax
        v1y, v1x = r[t, 2] - ay, c[t, 2] - ax
        v2y, v2x = qy - ay, qx - ax
        den = v0x * v1y - v1x * v0y
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        l1 = (v2x * v1y - v1x * v2y) / den
        l2 = (v0x * v2y - v2x * v0y) / den
        l0 = 1 - l1 - l2
        inside = ok & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        t, pr, pc = t[inside], pr[inside], pc[inside]
        if t.size == 0:                       # no pixel centre of this chunk lies in the image
            continue
        bary = np.stack([l0[inside], l1[inside], l2[inside]], axis=1)
        z = np.einsum("nk,nk->n", bary, d[t])
        pix = pr * w + pc
        order = np.lexsort((z, pix))
        pix, z, t, bary = pix[order], z[order], t[order], bary[order]
        first = np.concatenate([[True], pix[1:] != pix[:-1]])
        pix, z, t, bary = pix[first], z[first], t[first], bary[first]
        better = z < best_d[pix]
        pix, z, t, bary = pix[better], z[better], t[better], bary[better]
        best_d[pix] = z
        best_t[pix] = t + s
        best_b[pix] = bary
    return Frame(best_t.reshape(h, w), best_b.reshape(h, w, 3), best_d.reshape(h, w))


def render_view(view, verts: np.ndarray, faces: np.ndarray, scale: float = 1.0) -> Frame:
    """Rasterise a mesh in a `views.View`'s camera; `scale` < 1 renders a proportionally smaller image."""
    n = int(round(view.mask.shape[0] * scale))
    m = int(round(view.mask.shape[1] * scale))
    rows, cols, depth = project(view.name, view.pps * scale, view.cmid * scale, view.vref * scale, verts)
    return rasterize(rows, cols, depth, faces, (n, m))


def sample_texture(frame: Frame, corner_uv: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """RGB image from per-face corner UVs (glTF convention: v down) and a float texture; black background."""
    out = np.zeros(frame.tri.shape + (3,))
    m = frame.mask
    uv = np.einsum("nk,nkj->nj", frame.bary[m], corner_uv[frame.tri[m]])
    out[m] = bilinear(texture, uv[:, 1] * texture.shape[0] - 0.5, uv[:, 0] * texture.shape[1] - 0.5)
    return out


def bilinear(img: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    r0 = np.clip(np.floor(row).astype(np.int64), 0, h - 2)
    c0 = np.clip(np.floor(col).astype(np.int64), 0, w - 2)
    fr = np.clip(row - r0, 0, 1)[:, None]
    fc = np.clip(col - c0, 0, 1)[:, None]
    return ((img[r0, c0] * (1 - fc) + img[r0, c0 + 1] * fc) * (1 - fr)
            + (img[r0 + 1, c0] * (1 - fc) + img[r0 + 1, c0 + 1] * fc) * fr)
