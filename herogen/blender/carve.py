"""Silhouette carving (visual hull) and surfacing with surface nets, in numpy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .views import PROJ, ViewSet


@dataclass
class Grid:
    occ: np.ndarray        # bool [nx, ny, nz], True = voxel survives carving
    origin: np.ndarray     # world position of voxel (0, 0, 0)'s centre
    voxel: float           # edge length in studs


def carve(vs: ViewSet, height: float, nz: int) -> Grid:
    """Keep voxels whose centre projects inside every available mask.

    The views are separable: front/back constrain (X, Z), left/right (Y, Z) and top (X, Y), so the carve is
    a broadcast AND of three 2D tables, exact at the voxel centres.
    """
    voxel = height / nz
    ext = {"x": [], "y": []}
    for name, v in vs.views.items():
        cols = np.flatnonzero(v.mask.any(axis=0))
        u_lo, u_hi = (cols[0] - v.cmid) / v.pps, (cols[-1] + 1 - v.cmid) / v.pps
        axis = {"front": "x", "back": "x", "top": "x", "left": "y", "right": "y"}[name]
        sign = -1 if name in ("back", "right") else 1
        lo, hi = sorted((sign * u_lo, sign * u_hi))
        ext[axis].append((lo, hi))
        if name == "top":
            rows = np.flatnonzero(v.mask.any(axis=1))
            ext["y"].append(((v.vref - rows[-1] - 1) / v.pps, (v.vref - rows[0]) / v.pps))
    pad = 2 * voxel
    x_lo = max(e[0] for e in ext["x"]) - pad if ext["x"] else -height
    x_hi = min(e[1] for e in ext["x"]) + pad if ext["x"] else height
    y_lo = max(e[0] for e in ext["y"]) - pad if ext["y"] else -height
    y_hi = min(e[1] for e in ext["y"]) + pad if ext["y"] else height
    xs = np.arange(x_lo, x_hi + voxel, voxel)
    ys = np.arange(y_lo, y_hi + voxel, voxel)
    zs = (np.arange(-2, nz + 2) + 0.5) * voxel
    occ = np.ones((len(xs), len(ys), len(zs)), dtype=bool)

    def table(view, a: np.ndarray, b: np.ndarray, ax_a: int, ax_b: int) -> np.ndarray:
        pts = np.zeros((len(a) * len(b), 3))
        aa, bb = np.meshgrid(a, b, indexing="ij")
        pts[:, ax_a] = aa.ravel()
        pts[:, ax_b] = bb.ravel()
        return view.inside(pts).reshape(len(a), len(b))

    for name, v in vs.views.items():
        if name in ("front", "back"):
            occ &= table(v, xs, zs, 0, 2)[:, None, :]
        elif name in ("left", "right"):
            occ &= table(v, ys, zs, 1, 2)[None, :, :]
        else:
            occ &= table(v, xs, ys, 0, 1)[:, :, None]
    origin = np.array([xs[0], ys[0], zs[0]])
    return Grid(occ, origin, voxel)


def blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (edges clamped)."""
    if sigma <= 0:
        return field.astype(np.float64)
    r = max(1, int(np.ceil(2.5 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = field.astype(np.float64)
    for axis in range(3):
        padw = [(0, 0)] * 3
        padw[axis] = (r, r)
        p = np.pad(out, padw, mode="edge")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for i, w in enumerate(k):
            sl = [slice(None)] * 3
            sl[axis] = slice(i, i + n)
            acc += w * p[tuple(sl)]
        out = acc
    return out


def surface_nets(field: np.ndarray, origin: np.ndarray, spacing: float, level: float = 0.5):
    """Surface nets on a scalar field sampled at grid points; inside where field > level.

    One vertex per cell that the surface crosses, at the mean of its edge crossings; one quad per sign-changing
    grid edge, oriented outward. Returns (verts [N, 3], faces [M, 3]).
    """
    f = np.pad(field.astype(np.float64) - level, 1, mode="constant", constant_values=-1.0)
    origin = origin - spacing
    inside = f > 0
    nx, ny, nz = f.shape
    corners = [(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    cin = np.stack([inside[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k] for i, j, k in corners], axis=-1)
    n_in = cin.sum(axis=-1)
    active = (n_in > 0) & (n_in < 8)
    cell_idx = np.full(active.shape, -1, dtype=np.int64)
    ai = np.argwhere(active)
    cell_idx[tuple(ai.T)] = np.arange(len(ai))
    # Vertex placement: mean of edge crossings.
    vals = np.stack([f[ai[:, 0] + i, ai[:, 1] + j, ai[:, 2] + k] for i, j, k in corners], axis=1)
    offs = np.array(corners, dtype=np.float64)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(offs[a] - offs[b]).sum() == 1]
    acc = np.zeros((len(ai), 3))
    cnt = np.zeros(len(ai))
    for a, b in edges:
        va, vb = vals[:, a], vals[:, b]
        cross = (va > 0) != (vb > 0)
        t = np.where(cross, va / np.where(cross, va - vb, 1.0), 0.0)
        p = offs[a][None, :] + t[:, None] * (offs[b] - offs[a])[None, :]
        acc += np.where(cross[:, None], p, 0.0)
        cnt += cross
    verts = origin + (ai + acc / cnt[:, None]) * spacing
    quads = []
    # x-edges: grid points (i,j,k)-(i+1,j,k); cells (i, j-dj, k-dk).
    for axis in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        ia, ib = inside[tuple(sl_a)], inside[tuple(sl_b)]
        change = ia != ib
        e = np.argwhere(change)
        o1, o2 = [(1, 2), (2, 0), (0, 1)][axis]
        keep = (e[:, o1] >= 1) & (e[:, o2] >= 1)
        e = e[keep]
        outward = ia[tuple(e.T)]  # inside at the low end -> surface faces +axis
        c = []
        for d1, d2 in ((1, 1), (0, 1), (0, 0), (1, 0)):
            q = e.copy()
            q[:, o1] -= d1
            q[:, o2] -= d2
            c.append(cell_idx[tuple(q.T)])
        c = np.stack(c, axis=1)
        c[~outward] = c[~outward][:, ::-1]
        quads.append(c)
    quads = np.concatenate(quads)
    quads = quads[(quads >= 0).all(axis=1)]
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return verts, faces


def edges_of(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def taubin(verts: np.ndarray, faces: np.ndarray, iters: int = 10, lam: float = 0.5, mu: float = -0.53):
    """Taubin smoothing (low-pass without the shrinkage of plain Laplacian smoothing)."""
    e = edges_of(faces)
    n = len(verts)
    deg = np.bincount(e.ravel(), minlength=n).astype(np.float64)
    deg[deg == 0] = 1
    v = verts.copy()
    src = np.concatenate([e[:, 1], e[:, 0]])
    dst = np.concatenate([e[:, 0], e[:, 1]])
    for _ in range(iters):
        for w in (lam, mu):
            s = np.stack([np.bincount(dst, weights=v[src, k], minlength=n) for k in range(3)], axis=1)
            v = v + w * (s / deg[:, None] - v)
    return v


def components(n_verts: int, faces: np.ndarray) -> np.ndarray:
    """Connected-component label per vertex (union-find over face edges)."""
    parent = np.arange(n_verts)

    def find(x: np.ndarray) -> np.ndarray:
        while True:
            p = parent[x]
            if np.array_equal(p, x):
                return x
            parent[x] = parent[p]
            x = parent[x]

    e = edges_of(faces)
    changed = True
    while changed:
        ra, rb = find(e[:, 0]), find(e[:, 1])
        lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
        diff = lo != hi
        changed = bool(diff.any())
        if changed:
            np.minimum.at(parent, hi[diff], lo[diff])
    return find(np.arange(n_verts))


def keep_large_components(verts, faces, min_share: float = 0.01):
    """Drop islands holding fewer than `min_share` of the faces; returns (verts, faces, dropped_faces)."""
    lab = components(len(verts), faces)
    flab = lab[faces[:, 0]]
    ids, counts = np.unique(flab, return_counts=True)
    keep_ids = ids[counts >= min_share * len(faces)]
    keep = np.isin(flab, keep_ids)
    return (*compact(verts, faces[keep]), int((~keep).sum()))


def compact(verts: np.ndarray, faces: np.ndarray):
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def arm_depth_prior(g: Grid, skeleton) -> int:
    """Round the arms where no view constrains their depth.

    In the side views an A-posed arm hangs over the torso, so unless a real top view exists the carved arm is
    a slab as deep as the torso. Where an arm's carved depth exceeds its width, keep an elliptical cross-section
    instead: across the width of the front-view arm run, as deep as the run is wide times the arm's vertical
    direction cosine (the run of a slanted arm is wider than the arm), centred on the arm's estimated depth.
    Returns the number of (x, z) voxel columns changed.
    """
    F = skeleton.front
    if F is None or not skeleton.arms:
        return 0
    nx, ny, nz = g.occ.shape
    xs = g.origin[0] + np.arange(nx) * g.voxel
    ys = g.origin[1] + np.arange(ny) * g.voxel
    zs = g.origin[2] + np.arange(nz) * g.voxel
    changed = 0
    for arm in skeleton.arms.values():
        m = arm["mask"]
        cos_z = max(abs(arm["dir"][1]), 0.3)
        for k, z in enumerate(zs):
            r = F.row(z)
            if not m[r].any() or abs(F.v[r] - z) > F.px:
                continue
            cols = np.flatnonzero(m[r])
            x_lo, x_hi = F.h[cols[0]] - F.px / 2, F.h[cols[-1]] + F.px / 2
            a = (x_hi - x_lo) / 2
            xc = (x_hi + x_lo) / 2
            b = a * cos_z
            ix = np.flatnonzero((xs >= x_lo) & (xs <= x_hi))
            for i in ix:
                col = g.occ[i, :, k]
                if not col.any():
                    continue
                yy = ys[col]
                if (yy[-1] - yy[0]) / 2 <= b * 1.05:
                    continue
                yc = float(np.clip(arm.get("y", (yy[-1] + yy[0]) / 2), yy[0] + b, yy[-1] - b))
                half = b * np.sqrt(max(0.0, 1 - ((xs[i] - xc) / a) ** 2))
                g.occ[i, :, k] = col & (np.abs(ys - yc) <= max(half, g.voxel))
                changed += 1
    return changed


def detached_side_prior(g: Grid, vs: ViewSet, skeleton) -> int:
    """Thin things hanging clear of the body in the side view (a scarf tail, a strap) carve as plates as wide
    as the body, because the front view hides them behind it, and their true width and position are unknown.
    A rigid R15 body cannot animate them anyway, so in body columns (the front view's figure minus the arms)
    voxels whose depth falls in a side-view run detached from the main body run are removed. The
    side-view IoU shows what this costs. Returns the number of voxels removed.
    """
    from .joints import _plane

    S = _plane(vs, "left", "right")
    F = skeleton.front if skeleton is not None else None
    if S is None or F is None:
        return 0
    x0 = float(skeleton.joints["Neck"][0])
    nx, ny, nz = g.occ.shape
    xs = g.origin[0] + np.arange(nx) * g.voxel
    ys = g.origin[1] + np.arange(ny) * g.voxel
    zs = g.origin[2] + np.arange(nz) * g.voxel
    y_body = float(np.median([skeleton.joints[k][1] for k in ("Neck", "Waist", "Root")]))
    removed = 0
    for k, z in enumerate(zs):
        if z < 0 or z > float(skeleton.joints["Neck"][2]):
            continue
        runs = S.runs(S.row(z))
        if len(runs) < 2:
            continue
        main = S.run_at(S.row(z), y_body)
        # body columns: the front view's figure at this height minus the arms (torso, or both legs)
        r = F.row(z)
        body_px = F.m[r].copy()
        for arm in skeleton.arms.values():
            body_px &= ~arm["mask"][r]
        if not body_px.any():
            continue
        in_body = body_px[np.clip(np.round((xs - F.h[0]) / F.px).astype(int), 0, len(F.h) - 1)]
        in_body &= (xs >= F.h[0] - F.px) & (xs <= F.h[-1] + F.px)
        for lo, hi in runs:
            if (lo, hi) == main:
                continue
            ysel = (ys >= lo) & (ys <= hi)
            cut = in_body[:, None] & ysel[None, :]
            removed += int((g.occ[:, :, k] & cut).sum())
            g.occ[:, :, k] &= ~cut
    return removed


def build_mesh(vs: ViewSet, height: float, nz: int, sigma: float = 0.8, smooth_iters: int = 10, skeleton=None):
    """Carve, surface and lightly smooth. Feet land at Z = 0 and the height is rescaled to `height`.

    With a `skeleton` (joints from the masks), arms whose depth no view constrains are rounded first.
    """
    g = carve(vs, height, nz)
    prior_cols = arm_depth_prior(g, skeleton) if skeleton is not None else 0
    detached = detached_side_prior(g, vs, skeleton) if skeleton is not None else 0
    field = blur(g.occ, sigma)
    verts, faces = surface_nets(field, g.origin, g.voxel)
    verts, faces, dropped = keep_large_components(verts, faces)
    verts = taubin(verts, faces, smooth_iters)
    z0, z1 = verts[:, 2].min(), verts[:, 2].max()
    verts[:, 2] = (verts[:, 2] - z0) * (height / (z1 - z0))
    info = {"voxel_resolution": [int(s) for s in g.occ.shape], "voxel_size_studs": round(g.voxel, 5),
            "voxels_kept": int(g.occ.sum()), "dropped_island_faces": dropped,
            "blur_sigma_voxels": sigma, "taubin_iterations": smooth_iters,
            "z_rescale": round(height / (z1 - z0), 5), "arm_depth_prior_columns": prior_cols,
            "detached_side_prior_voxels": detached}
    return verts, faces, info


def winding_inside(points: np.ndarray, verts: np.ndarray, faces: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Generalised winding number > 0.5: robust inside test for closed (or nearly closed) meshes."""
    tri = verts[faces]
    out = np.zeros(len(points), dtype=bool)
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        a = tri[None, :, 0, :] - p[:, None, :]
        b = tri[None, :, 1, :] - p[:, None, :]
        c = tri[None, :, 2, :] - p[:, None, :]
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la
               + np.einsum("ijk,ijk->ij", c, a) * lb)
        w = np.arctan2(det, den).sum(axis=1) / (2 * np.pi)
        out[s:s + chunk] = w > 0.5
    return out


def point_mesh_distance(points: np.ndarray, verts: np.ndarray, faces: np.ndarray, chunk: int = 128):
    """Exact unsigned distance from each point to the nearest triangle."""
    tri = verts[faces]
    a0, b0, c0 = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b0 - a0, c0 - a0)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-12
    a0, b0, c0, n, nn = a0[ok], b0[ok], c0[ok], n[ok], nn[ok]
    nhat = n / nn[:, None]
    out = np.empty(len(points))

    def seg(p, s0, s1):
        d = s1 - s0
        t = np.einsum("ijk,jk->ij", p[:, None, :] - s0[None], d) / np.maximum((d * d).sum(1), 1e-18)[None]
        t = np.clip(t, 0, 1)
        q = s0[None] + t[..., None] * d[None]
        return np.linalg.norm(p[:, None, :] - q, axis=2)

    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        rel = p[:, None, :] - a0[None]
        h = np.einsum("ijk,jk->ij", rel, nhat)
        proj = p[:, None, :] - h[..., None] * nhat[None]
        # barycentric inside test via edge cross products
        def side(u, v):
            return np.einsum("ijk,jk->ij", np.cross(v[None] - u[None], proj - u[None]), nhat)
        ins = (side(a0, b0) >= 0) & (side(b0, c0) >= 0) & (side(c0, a0) >= 0)
        d = np.where(ins, np.abs(h), np.inf)
        d = np.minimum(d, seg(p, a0, b0))
        d = np.minimum(d, seg(p, b0, c0))
        d = np.minimum(d, seg(p, c0, a0))
        out[s:s + chunk] = d.min(axis=1)
    return out


def ray_hits(origins: np.ndarray, dirs: np.ndarray, tri: np.ndarray, t_min: float = 1e-3,
             chunk: int = 256) -> np.ndarray:
    """True for each ray (origin, unit dir) that hits any triangle [m, 3, 3] at t > t_min (Moller-Trumbore)."""
    out = np.zeros(len(origins), dtype=bool)
    if len(tri) == 0:
        return out
    a = tri[:, 0]
    e1 = tri[:, 1] - a
    e2 = tri[:, 2] - a
    for s in range(0, len(origins), chunk):
        o = origins[s:s + chunk][:, None, :]
        d = dirs[s:s + chunk][:, None, :]
        pvec = np.cross(d, e2[None])
        det = np.einsum("ijk,jk->ij", pvec, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = o - a[None]
        u = np.einsum("ijk,ijk->ij", tvec, pvec) * inv
        qvec = np.cross(tvec, e1[None])
        v = np.einsum("ijk,ijk->ij", d, qvec) * inv
        t = np.einsum("ijk,jk->ij", qvec, e2) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > t_min)
        out[s:s + chunk] = hit.any(axis=1)
    return out
