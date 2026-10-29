"""Fast exact queries on a triangle mesh through a uniform grid: inside, distance to the surface and ray hits.

The pose test asks, per joint and pose, which rim points lie outside the neighbouring section, how far outside,
and whether a ray from each exposed point leaves the body. The brute-force functions in `carve`
(`winding_inside`, `point_mesh_distance`, `ray_hits`) test every point against every triangle, which at full
detail (a 170k-triangle body) takes minutes per pass. `MeshIndex` bins the triangles once into a uniform grid of
cells (each triangle in every cell its bounding box overlaps), so each query only looks at the triangles near it:

- `distance(points)`: the exact unsigned distance to the nearest triangle. Cells within a box of half-width `r`
  around each point are searched; a triangle in no searched cell lies farther than `r`, so the nearest found is
  exact once it is within `r` (`r` doubles until it is; a point far from the mesh falls back to the brute
  force). Each point-triangle pair is measured with the same formula as `carve.point_mesh_distance` (degenerate
  triangles skipped there and here), so the result is that function's.
- `inside(points)`: signed crossings of a ray along +X, the winding number of a closed mesh: each triangle the
  ray crosses beyond the point adds the sign of its normal's X. Only triangles in the point's YZ column are
  tested. A ray through an edge or vertex is resolved by moving the point by an infinitesimal (simulation of
  simplicity) and every shared edge is evaluated from its endpoints in one fixed order, so each crossing is
  counted exactly once. For a closed (or several overlapping closed) meshes this is `carve.winding_inside`'s
  answer (winding number > 0.5) for every point not on the surface; a point on the surface itself (within
  rounding) may go either way in both, and its distance to the surface is then about zero.
- `ray_hits(origins, dirs)`: whether each ray hits a triangle at t > `t_min`; the cells along each ray are walked
  (3D DDA) and only their triangles are tested, with `carve.ray_hits`'s Moller-Trumbore arithmetic.

Numpy only, so stage 4 needs nothing beyond numpy and Blender.
"""
from __future__ import annotations

import numpy as np

MAX_CELLS = 1 << 22            # grid size cap; the cell grows to fit a large mesh
BRUTE_BOX_CELLS = 4096         # a distance query box larger than this uses the brute force for that point


class MeshIndex:
    def __init__(self, verts: np.ndarray, faces: np.ndarray, cell: float | None = None):
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        self.verts, self.faces = verts, faces
        tri = verts[faces]
        self.tri = tri
        self.n = len(tri)
        if self.n == 0:
            self.lo, self.h, self.dims = np.zeros(3), 1.0, np.ones(3, dtype=np.int64)
            return
        tlo, thi = tri.min(axis=1), tri.max(axis=1)
        self.lo = tlo.min(axis=0)
        span = np.maximum(thi.max(axis=0) - self.lo, 1e-9)
        if cell is None:
            ext = np.median((thi - tlo).max(axis=1))
            cell = max(2.0 * float(ext), float(span.max()) / 2048.0, 1e-6)
        h = float(cell)
        while np.prod(np.floor(span / h).astype(np.int64) + 1) > MAX_CELLS:
            h *= 1.25
        self.h = h
        self.dims = np.floor(span / h).astype(np.int64) + 1
        # triangle -> every cell its (slightly grown) bounding box overlaps
        eps = 1e-7 * h
        self.cells3 = self._bin(tlo - eps, thi + eps, (0, 1, 2))
        self.cells_yz = self._bin(tlo - eps, thi + eps, (1, 2))
        # carve.point_mesh_distance's per-triangle terms (degenerate triangles are skipped there too)
        a0, b0, c0 = tri[:, 0], tri[:, 1], tri[:, 2]
        nrm = np.cross(b0 - a0, c0 - a0)
        nn = np.linalg.norm(nrm, axis=1)
        self.ok = nn > 1e-12
        self.nhat = nrm / np.where(self.ok, nn, 1.0)[:, None]
        self.normal = nrm
        self.xmax = thi[:, 0]
        self.centroid = tri.mean(axis=1)

    # ---- grid ---------------------------------------------------------------------------------------------

    def _index(self, p: np.ndarray, axes) -> np.ndarray:
        """Integer cell coordinates of points along `axes`, clipped to the grid."""
        ax = list(axes)
        i = np.floor((p[..., ax] - self.lo[ax]) / self.h).astype(np.int64)
        return np.clip(i, 0, self.dims[ax] - 1)

    def _flat(self, ijk: np.ndarray, axes) -> np.ndarray:
        dims = self.dims[list(axes)]
        out = np.zeros(ijk.shape[:-1], dtype=np.int64)
        for k in range(len(dims)):
            out = out * dims[k] + ijk[..., k]
        return out

    def _bin(self, lo: np.ndarray, hi: np.ndarray, axes) -> tuple[np.ndarray, np.ndarray]:
        """CSR (cell start [ncell + 1], triangle ids sorted by cell) of each triangle's overlapped cells."""
        i0, i1 = self._index(lo, axes), self._index(hi, axes)
        size = i1 - i0 + 1
        count = np.prod(size, axis=1)
        tri_id = np.repeat(np.arange(self.n), count)
        k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        sz = size[tri_id]
        coords = np.empty((len(tri_id), len(axes)), dtype=np.int64)
        for a in range(len(axes) - 1, -1, -1):
            coords[:, a] = i0[tri_id, a] + k % sz[:, a]
            k = k // sz[:, a]
        flat = self._flat(coords, axes)
        ncell = int(np.prod(self.dims[list(axes)]))
        order = np.argsort(flat, kind="stable")
        start = np.zeros(ncell + 1, dtype=np.int64)
        np.cumsum(np.bincount(flat, minlength=ncell), out=start[1:])
        return start, tri_id[order]

    @staticmethod
    def _gather(csr, query: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(query, triangle) pairs for (query, cell) pairs."""
        start, ids = csr
        cnt = start[cell + 1] - start[cell]
        rep = np.repeat(np.arange(len(cell)), cnt)
        offs = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        return query[rep], ids[start[cell][rep] + offs]

    # ---- distance -----------------------------------------------------------------------------------------

    def _pair_distance(self, p: np.ndarray, t: np.ndarray) -> np.ndarray:
        """carve.point_mesh_distance's distance for each (point, triangle) pair."""
        tri = self.tri[t]
        a0, b0, c0 = tri[:, 0], tri[:, 1], tri[:, 2]
        nhat = self.nhat[t]
        h = np.einsum("ik,ik->i", p - a0, nhat)
        proj = p - h[:, None] * nhat

        def side(u, v):
            return np.einsum("ik,ik->i", np.cross(v - u, proj - u), nhat)

        def seg(s0, s1):
            d = s1 - s0
            w = np.einsum("ik,ik->i", p - s0, d) / np.maximum((d * d).sum(1), 1e-18)
            w = np.clip(w, 0, 1)
            return np.linalg.norm(p - (s0 + w[:, None] * d), axis=1)

        ins = (side(a0, b0) >= 0) & (side(b0, c0) >= 0) & (side(c0, a0) >= 0)
        d = np.where(ins, np.abs(h), np.inf)
        return np.minimum(np.minimum(d, seg(a0, b0)), np.minimum(seg(b0, c0), seg(c0, a0)))

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Exact unsigned distance from each point to the nearest (non-degenerate) triangle."""
        from .carve import point_mesh_distance

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.full(len(points), np.inf)
        if self.n == 0 or not self.ok.any() or len(points) == 0:
            return out
        todo = np.arange(len(points))
        r = self.h
        full = np.prod(self.dims)
        while len(todo):
            p = points[todo]
            i0 = self._index(p - r, (0, 1, 2))
            i1 = self._index(p + r, (0, 1, 2))
            size = i1 - i0 + 1
            count = np.prod(size, axis=1)
            brute = count > BRUTE_BOX_CELLS
            if brute.any():
                out[todo[brute]] = point_mesh_distance(p[brute], self.verts, self.faces)
            keep = ~brute
            q = np.repeat(np.arange(len(p))[keep], count[keep])
            k = np.arange(len(q)) - np.repeat(np.cumsum(count[keep]) - count[keep], count[keep])
            sz = size[q]
            coords = np.empty((len(q), 3), dtype=np.int64)
            for a in (2, 1, 0):
                coords[:, a] = i0[q, a] + k % sz[:, a]
                k = k // sz[:, a]
            qq, tt = self._gather(self.cells3, q, self._flat(coords, (0, 1, 2)))
            good = self.ok[tt]
            qq, tt = qq[good], tt[good]
            dmin = np.full(len(p), np.inf)
            if len(qq):
                np.minimum.at(dmin, qq, self._pair_distance(p[qq], tt))
            whole = count >= full          # the box covers the whole grid: nothing is left unsearched
            done = keep & ((dmin <= r) | whole)
            out[todo[done]] = dmin[done]
            todo = todo[keep & ~done]
            r *= 2.0
        return out

    # ---- inside -------------------------------------------------------------------------------------------

    def winding(self, points: np.ndarray) -> np.ndarray:
        """Signed crossings of the +X ray from each point (the winding number of a closed mesh)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(len(points), dtype=np.int64)
        if self.n == 0 or len(points) == 0:
            return out
        col = self._flat(self._index(points, (1, 2)), (1, 2))
        q, t = self._gather(self.cells_yz, np.arange(len(points)), col)
        p = points[q]
        keep = self.xmax[t] > p[:, 0]
        q, t, p = q[keep], t[keep], p[keep]
        tri = self.tri[t][:, :, 1:]                       # [K, 3, 2] in (y, z)
        y, z = p[:, 1], p[:, 2]
        sgn = []
        for i in range(3):
            u, v = tri[:, i], tri[:, (i + 1) % 3]
            # one fixed order per edge (lexicographic in y, z), so both triangles of a shared edge compute it alike
            swap = (u[:, 0] > v[:, 0]) | ((u[:, 0] == v[:, 0]) & (u[:, 1] > v[:, 1]))
            a = np.where(swap[:, None], v, u)
            b = np.where(swap[:, None], u, v)
            dy, dz = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
            e = dy * (z - a[:, 1]) - dz * (y - a[:, 0])
            # on the edge: the point moved by (eps, eps^2) decides, the same for every triangle
            tie = np.where(dz != 0, -np.sign(dz), np.sign(dy))
            s = np.where(e != 0, np.sign(e), tie)
            sgn.append(np.where(swap, -s, s))
        area = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        o = np.sign(area)
        hit = (o != 0) & (sgn[0] == o) & (sgn[1] == o) & (sgn[2] == o)
        nx = self.normal[t, 0]
        hit &= nx != 0
        a0 = self.tri[t, 0]
        nrm = self.normal[t]
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a0[:, 0] - (nrm[:, 1] * (y - a0[:, 1]) + nrm[:, 2] * (z - a0[:, 2])) / nx
        hit &= xc > p[:, 0]
        np.add.at(out, q[hit], np.sign(nx[hit]).astype(np.int64))
        return out

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Inside a closed mesh (winding number above one half, as `carve.winding_inside`)."""
        return self.winding(points) >= 1

    # ---- rays ---------------------------------------------------------------------------------------------

    def ray_hits(self, origins: np.ndarray, dirs: np.ndarray, t_min: float = 1e-3, step_cells: int = 8) -> np.ndarray:
        """True for each ray (origin, unit direction) that hits a triangle at t > t_min (`carve.ray_hits`)."""
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
        hit = np.zeros(len(origins), dtype=bool)
        if self.n == 0 or len(origins) == 0:
            return hit
        a = self.tri[:, 0]
        e1 = self.tri[:, 1] - a
        e2 = self.tri[:, 2] - a
        lo, hi = self.lo, self.lo + self.dims * self.h
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(dirs != 0, 1.0 / dirs, np.inf)
            t1 = (lo - origins) * inv
            t2 = (hi - origins) * inv
        tnear = np.nanmax(np.where(np.isnan(np.minimum(t1, t2)), -np.inf, np.minimum(t1, t2)), axis=1)
        tfar = np.nanmin(np.where(np.isnan(np.maximum(t1, t2)), np.inf, np.maximum(t1, t2)), axis=1)
        t0 = np.maximum(tnear, 0.0)
        live = np.nonzero(t0 <= tfar)[0]
        if not len(live):
            return hit
        # walk the cells from the grid entry (or the origin)
        p0 = origins[live] + t0[live, None] * dirs[live]
        cell = self._index(p0, (0, 1, 2))
        d = dirs[live]
        step = np.where(d > 0, 1, np.where(d < 0, -1, 0))
        nxt = self.lo + (cell + (step > 0)) * self.h
        with np.errstate(divide="ignore", invalid="ignore"):
            tmax = np.where(step != 0, (nxt - origins[live]) / d, np.inf)
            tdelta = np.where(step != 0, self.h / np.abs(d), np.inf)
        tend = tfar[live]
        rays = live
        while len(rays):
            pairs_r, pairs_c = [], []
            alive = np.ones(len(rays), dtype=bool)
            for _ in range(step_cells):
                idx = np.nonzero(alive)[0]
                if not len(idx):
                    break
                pairs_r.append(idx)
                pairs_c.append(self._flat(cell[idx], (0, 1, 2)))
                ax = np.argmin(tmax[idx], axis=1)
                tnext = tmax[idx, ax]
                cell[idx, ax] += step[idx, ax]
                tmax[idx, ax] += tdelta[idx, ax]
                out = (tnext > tend[idx]) | (cell[idx, ax] < 0) | (cell[idx, ax] >= self.dims[ax])
                alive[idx[out]] = False
            r = np.concatenate(pairs_r)
            qq, tt = self._gather(self.cells3, r, np.concatenate(pairs_c))
            if len(qq):
                o, dd = origins[rays[qq]], dirs[rays[qq]]
                ea, eb = e1[tt], e2[tt]
                pvec = np.cross(dd, eb)
                det = np.einsum("ik,ik->i", pvec, ea)
                ok = np.abs(det) > 1e-12
                inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
                tvec = o - a[tt]
                u = np.einsum("ik,ik->i", tvec, pvec) * inv_det
                qvec = np.cross(tvec, ea)
                v = np.einsum("ik,ik->i", dd, qvec) * inv_det
                t = np.einsum("ik,ik->i", qvec, eb) * inv_det
                got = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > t_min)
                hit_local = np.zeros(len(rays), dtype=bool)
                hit_local[qq[got]] = True
                hit[rays[hit_local]] = True
                alive &= ~hit_local
            keep = np.nonzero(alive)[0]
            rays, cell, tmax, tdelta, step, tend = (rays[keep], cell[keep], tmax[keep], tdelta[keep], step[keep],
                                                     tend[keep])
        return hit
