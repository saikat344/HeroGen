"""What each pixel centre sees, by rasterising the triangles (pure numpy, so it also runs inside Blender).

For an orthographic camera from `cameras.py`, one ray through the centre of pixel (r, c) hits a triangle when
the point (c + 0.5, r + 0.5) lies inside the triangle's projection (its edges included), at the depth found
by interpolating the corners' depths with the barycentric weights there. The nearest such triangle is what
the ray hits first. Rasterising gives that answer for every pixel at once, much faster than casting the rays
one at a time, and also says which triangle was hit and where on it (for UVs and region labels).

`keep(tri, bary)` may reject hits, as a ray goes on through an alpha-clipped texel; the nearest kept hit
counts. Triangles seen edge-on (no projected area) are never hit.
"""
from __future__ import annotations

import numpy as np

CHUNK_CANDIDATES = 2_000_000    # pixel-triangle pairs tested at once
INSIDE_EPS = 1e-9               # barycentric slack, so a centre on a shared edge belongs to both triangles
AREA_EPS = 1e-12                # projected area (pixels squared, doubled) below which a triangle is edge-on


def _spans(lo: np.ndarray, hi: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """First and last pixel index whose centre (i + 0.5) lies in [lo, hi], clipped to the image."""
    first = np.maximum(np.ceil(lo - 0.5), 0).astype(np.int64)
    last = np.minimum(np.floor(hi - 0.5), size - 1).astype(np.int64)
    return first, last


def fragments(cr: np.ndarray, depth: np.ndarray, tris: np.ndarray, size: int):
    """Every (pixel, triangle) pair where the pixel centre lies in the triangle's projection, in chunks:
    yields (pixel index row * size + col, triangle index, barycentric weights [K, 3], depth [K])."""
    cr = np.asarray(cr, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    p = cr[tris]                                    # [T, 3, 2]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    c0, c1 = _spans(p[:, :, 0].min(1), p[:, :, 0].max(1), size)
    r0, r1 = _spans(p[:, :, 1].min(1), p[:, :, 1].max(1), size)
    w, h = c1 - c0 + 1, r1 - r0 + 1
    live = np.nonzero((np.abs(area) > AREA_EPS) & (w > 0) & (h > 0))[0]
    # Triangles go in groups whose boxes are alike in width and in height (each within a factor of two), so
    # padding every box in a group to the group's largest wastes little.
    wb = np.ceil(np.log2(np.maximum(w[live], 1))).astype(np.int64)
    hb = np.ceil(np.log2(np.maximum(h[live], 1))).astype(np.int64)
    order = live[np.lexsort((hb, wb))]
    group = (wb * 64 + hb)[np.lexsort((hb, wb))]
    start = 0
    while start < len(order):
        g_end = start + int(np.searchsorted(group[start:], group[start], side="right"))
        cap = int((1 << int(group[start] // 64)) * (1 << int(group[start] % 64)))
        stop = min(g_end, start + max(1, CHUNK_CANDIDATES // cap))
        ids = order[start:stop]
        bw, bh = int(w[ids].max()), int(h[ids].max())
        dc, dr = np.meshgrid(np.arange(bw), np.arange(bh))
        cols = c0[ids, None] + dc.ravel()[None, :]                   # [n, bw*bh]
        rows = r0[ids, None] + dr.ravel()[None, :]
        ok = (cols <= c1[ids, None]) & (rows <= r1[ids, None])
        px, py = cols + 0.5, rows + 0.5
        ta, tb, tc = a[ids], b[ids], c[ids]
        inv = 1.0 / area[ids, None]
        ax, ay, bx, by = ta[:, 0, None], ta[:, 1, None], tb[:, 0, None], tb[:, 1, None]
        cx, cy = tc[:, 0, None], tc[:, 1, None]
        wa = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv
        wb = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv
        wc = 1.0 - wa - wb
        ok &= (wa >= -INSIDE_EPS) & (wb >= -INSIDE_EPS) & (wc >= -INSIDE_EPS)
        ti, ki = np.nonzero(ok)
        tri = ids[ti]
        bary = np.stack([wa[ti, ki], wb[ti, ki], wc[ti, ki]], axis=1)
        z = (depth[tris[tri]] * bary).sum(1)
        yield rows[ti, ki] * size + cols[ti, ki], tri, bary, z
        start = stop


def rasterise(cr: np.ndarray, depth: np.ndarray, tris: np.ndarray, size: int, keep=None):
    """(tri, bary, z) for a size x size image: the triangle each pixel centre hits first (-1 for none) [size,
    size] int32, the barycentric weights of the hit on it [size, size, 3] float32, and its depth [size, size]
    float64 (NaN for none). `cr` is the vertices' (col, row) pixel-edge coordinates, `depth` their depth
    along the looking direction (smaller is nearer)."""
    best = np.full(size * size, np.inf)
    tri_map = np.full(size * size, -1, dtype=np.int64)
    bary_map = np.zeros((size * size, 3), dtype=np.float64)
    for pix, tri, bary, z in fragments(cr, depth, tris, size):
        if keep is not None and len(tri):
            k = np.asarray(keep(tri, bary), dtype=bool)
            pix, tri, bary, z = pix[k], tri[k], bary[k], z[k]
        if not len(pix):
            continue
        # the nearest fragment per pixel in the chunk (ties: the lower triangle), then against the buffer
        o = np.lexsort((tri, z, pix))
        pix, tri, bary, z = pix[o], tri[o], bary[o], z[o]
        first = np.ones(len(pix), dtype=bool)
        first[1:] = pix[1:] != pix[:-1]
        pix, tri, bary, z = pix[first], tri[first], bary[first], z[first]
        win = z < best[pix]
        pix = pix[win]
        best[pix], tri_map[pix], bary_map[pix] = z[win], tri[win], bary[win]
    z_map = np.where(tri_map >= 0, best, np.nan)
    return (tri_map.reshape(size, size).astype(np.int32), bary_map.reshape(size, size, 3).astype(np.float32),
            z_map.reshape(size, size))


def interpolate(corner_values: np.ndarray, tri_map: np.ndarray, bary_map: np.ndarray, fill=0.0) -> np.ndarray:
    """A per-corner attribute [T, 3, K] at every hit pixel ([size, size, K]); `fill` where nothing is hit."""
    v = np.asarray(corner_values)
    out = np.full(tri_map.shape + v.shape[2:], fill, dtype=np.float64)
    hit = tri_map >= 0
    t = tri_map[hit]
    out[hit] = np.einsum("nk,nk...->n...", bary_map[hit].astype(np.float64), v[t])
    return out
