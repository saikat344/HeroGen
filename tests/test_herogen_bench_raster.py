"""Ground-truth bench rasteriser: the same hits as one orthographic ray per pixel centre."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

try:
    import numpy as np

    from herogen.bench import cameras as CAM
    from herogen.bench import raster as RAS
    from herogen.bench import texture as TEX

    HAVE_DEPS = True
except ImportError:  # numpy missing
    HAVE_DEPS = False


def brute_force(cr, depth, tris, size, keep=None):
    """One ray per pixel centre against every triangle, nearest kept hit: the reference."""
    tri_map = np.full((size, size), -1)
    z_map = np.full((size, size), np.nan)
    for r in range(size):
        for c in range(size):
            p = np.array([c + 0.5, r + 0.5])
            best = None
            for t, (i, j, k) in enumerate(tris):
                a, b, cc = cr[i], cr[j], cr[k]
                area = (b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0])
                if abs(area) <= RAS.AREA_EPS:
                    continue
                wa = ((b[0] - p[0]) * (cc[1] - p[1]) - (b[1] - p[1]) * (cc[0] - p[0])) / area
                wb = ((cc[0] - p[0]) * (a[1] - p[1]) - (cc[1] - p[1]) * (a[0] - p[0])) / area
                wc = 1 - wa - wb
                if min(wa, wb, wc) < -RAS.INSIDE_EPS:
                    continue
                bary = np.array([wa, wb, wc])
                if keep is not None and not keep(np.array([t]), bary[None])[0]:
                    continue
                z = float(bary @ depth[[i, j, k]])
                if best is None or z < best[0] or (z == best[0] and t < best[1]):
                    best = (z, t)
            if best is not None:
                z_map[r, c], tri_map[r, c] = best
    return tri_map, z_map


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class RasteriseTest(unittest.TestCase):
    def random_mesh(self, seed, n_tris, size):
        rng = np.random.default_rng(seed)
        cr = rng.uniform(-2, size + 2, size=(n_tris * 3, 2))
        # small and large triangles, some overlapping
        centres = rng.uniform(0, size, size=(n_tris, 1, 2))
        scale = rng.choice([0.6, 3.0, size / 2.0], size=(n_tris, 1, 1))
        cr = (centres + (cr.reshape(n_tris, 3, 2) / size - 0.5) * scale).reshape(-1, 2)
        depth = rng.uniform(-1, 1, size=n_tris * 3)
        tris = np.arange(n_tris * 3).reshape(n_tris, 3)
        return cr, depth, tris

    def test_matches_one_ray_per_pixel_centre(self):
        for seed in range(4):
            cr, depth, tris = self.random_mesh(seed, 40, 24)
            tri, _, z = RAS.rasterise(cr, depth, tris, 24)
            ref_tri, ref_z = brute_force(cr, depth, tris, 24)
            np.testing.assert_array_equal(tri >= 0, ref_tri >= 0)
            np.testing.assert_allclose(z[tri >= 0], ref_z[ref_tri >= 0], atol=1e-9)
            # where two triangles meet at exactly the same depth either may win; elsewhere the same one
            same_depth_tie = np.isclose(z, ref_z) & (tri != ref_tri)
            self.assertTrue(np.all((tri == ref_tri) | same_depth_tie), seed)

    def test_chunks_give_the_same_answer(self):
        cr, depth, tris = self.random_mesh(7, 60, 32)
        whole = RAS.rasterise(cr, depth, tris, 32)
        old = RAS.CHUNK_CANDIDATES
        try:
            RAS.CHUNK_CANDIDATES = 50
            chunked = RAS.rasterise(cr, depth, tris, 32)
        finally:
            RAS.CHUNK_CANDIDATES = old
        np.testing.assert_array_equal(whole[0] >= 0, chunked[0] >= 0)
        np.testing.assert_allclose(whole[2], chunked[2], atol=1e-12)

    def test_rejected_hits_are_passed_through(self):
        # two squares, the near one rejected on its left half: the far one shows through there
        size = 8
        cr = np.array([[0, 0], [8, 0], [8, 8], [0, 8]] * 2, dtype=float)
        depth = np.array([0.0] * 4 + [1.0] * 4)
        tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])

        def keep(tri, bary):
            x = (bary[:, :, None] * cr[tris[tri]]).sum(1)[:, 0]
            return (tri >= 2) | (x > 4)
        _, _, z = RAS.rasterise(cr, depth, tris, size, keep=keep)
        self.assertTrue(np.all(z[:, :4] == 1.0))
        self.assertTrue(np.all(z[:, 4:] == 0.0))
        _, ref_z = brute_force(cr, depth, tris, size, keep=keep)
        np.testing.assert_allclose(z, ref_z)

    def test_edge_on_triangles_are_never_hit(self):
        cr = np.array([[0, 0], [8, 8], [4, 4]], dtype=float)
        tri, _, _ = RAS.rasterise(cr, np.zeros(3), np.array([[0, 1, 2]]), 8)
        self.assertTrue(np.all(tri == -1))

    def test_a_figure_through_the_bench_cameras(self):
        # a box figure in the contract frame, seen from every bench camera: the hit points lie on the box
        lo, hi = np.array([-0.8, -0.4, 0.0]), np.array([0.8, 0.4, 5.5])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        tris = np.array([t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))])
        size = 64
        ppu = CAM.choose_ppu(corners, size)
        for cam in CAM.all_cameras(ppu, size, distance=CAM.camera_distance(corners)):
            cr = CAM.project(corners, cam)
            tri, _, z = RAS.rasterise(cr, CAM.depth_of(corners, cam), tris, size)
            rows, cols = np.nonzero(tri >= 0)
            origins, fwd = CAM.pixel_rays(cam, rows, cols)
            pts = origins + fwd * (z[rows, cols] + cam["distance"])[:, None]
            # every hit point lies on the box surface
            self.assertTrue(np.all(pts >= lo - 1e-9) and np.all(pts <= hi + 1e-9), cam["view"])
            on_face = np.isclose(pts, lo, atol=1e-9) | np.isclose(pts, hi, atol=1e-9)
            self.assertTrue(np.all(on_face.any(1)), cam["view"])
            self.assertTrue(np.allclose(CAM.depth_of(pts, cam), z[rows, cols]))

    def test_interpolate_corner_values(self):
        cr = np.array([[0, 0], [4, 0], [0, 4]], dtype=float)
        uv = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        tri, bary, _ = RAS.rasterise(cr, np.zeros(3), np.array([[0, 1, 2]]), 4)
        out = RAS.interpolate(uv, tri, bary, fill=-1.0)
        self.assertEqual(out.shape, (4, 4, 2))
        self.assertTrue(np.allclose(out[0, 0], [0.125, 0.125]))
        self.assertTrue(np.all(out[3, 3] == -1.0))


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class SampleManyTest(unittest.TestCase):
    def test_matches_the_single_sample(self):
        rng = np.random.default_rng(0)
        t = rng.uniform(size=(5, 7))
        u, v = rng.uniform(-1.5, 2.5, size=200), rng.uniform(-1.5, 2.5, size=200)
        for interp in TEX.INTERPOLATIONS:
            for ext in TEX.EXTENSIONS:
                many = TEX.sample_many(t, u, v, interp, ext)
                one = [TEX.sample(t, float(a), float(b), interp, ext) for a, b in zip(u, v)]
                np.testing.assert_allclose(many, one, atol=1e-12, err_msg=f"{interp} {ext}")


if __name__ == "__main__":
    unittest.main()
