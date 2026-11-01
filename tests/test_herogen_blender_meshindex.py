"""The grid-accelerated mesh queries (herogen.blender.meshindex) give the brute-force answers, the pose test gives
the same gaps through either, the cut centre does not move with mesh density, and the cut search keeps every
joint's gap at or below the unmoved cut's.

Synthetic inputs only; nothing needs Blender.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from herogen.blender import carve, cuts, joints, pose, split, synth, views
from herogen.blender.carve import point_mesh_distance, ray_hits, winding_inside
from herogen.blender.meshindex import MeshIndex


def box(lo, hi):
    """A closed, outward-wound axis-aligned box: every face is axis-aligned, so rays meet edges and vertices."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


class TestFirstHit(unittest.TestCase):
    def test_triangle_in_early_cell_does_not_hide_a_closer_later_hit(self):
        # Triangle 0's box starts at x=0, but this ray meets it at x=8. Triangle 1 appears only at x=3.
        verts = np.array([[0, -.8, -1], [0, -.8, 1], [10, .2, 0],
                          [3, -.1, -.1], [3, .1, -.1], [3, 0, .1]])
        index = MeshIndex(verts, np.array([[0, 1, 2], [3, 4, 5]]), cell=.25)
        for steps in (1, 8, 64):
            ids, bary, t = index.ray_first_hit([[-1, 0, 0]], [[1, 0, 0]], step_cells=steps)
            np.testing.assert_array_equal(ids, [1])
            np.testing.assert_allclose(bary, [[.25, .25, .5]])
            np.testing.assert_allclose(t, [4])

    def test_first_hit_matches_independent_linear_solves(self):
        verts, faces = box((0, 0, 0), (1, 1, 1))
        # A tilted second box exercises oblique triangles and later hits in the same bins.
        rotation = np.array([[.8, -.6, 0], [.6, .8, 0], [0, 0, 1]])
        verts = np.concatenate((verts, verts @ rotation.T + [.3, .2, 1.4]))
        faces = np.concatenate((faces, faces + 8))
        rng = np.random.default_rng(891)
        origins = rng.uniform(-2, 3, (180, 3))
        dirs = rng.normal(size=origins.shape)
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        expected_ids = np.full(len(origins), -1)
        expected_bary = np.full((len(origins), 3), np.nan)
        expected_t = np.full(len(origins), np.inf)
        for q, (o, d) in enumerate(zip(origins, dirs)):
            for k, (a, b, c) in enumerate(verts[faces]):
                try:
                    t, u, v = np.linalg.solve(np.column_stack((d, a - b, a - c)), a - o)
                except np.linalg.LinAlgError:
                    continue
                if t > 0 and u >= 0 and v >= 0 and u + v <= 1 and t < expected_t[q]:
                    expected_ids[q], expected_bary[q], expected_t[q] = k, [1 - u - v, u, v], t
        index = MeshIndex(verts, faces, cell=.13)
        for steps in (1, 8, 64):
            for limit in (.1, 1.7, np.inf):
                with self.subTest(steps=steps, limit=limit):
                    ids, bary, t = index.ray_first_hit(origins, dirs, t_max=limit, step_cells=steps)
                    hit = expected_t <= limit
                    np.testing.assert_array_equal(ids, np.where(hit, expected_ids, -1))
                    np.testing.assert_allclose(t, np.where(hit, expected_t, np.inf), atol=2e-12)
                    np.testing.assert_allclose(bary, np.where(hit[:, None], expected_bary, np.nan), atol=2e-12)

    def test_boundaries_ties_parallel_and_empty_queries(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        f = np.array([[0, 1, 2], [0, 1, 2], [0, 0, 0]])
        index = MeshIndex(v, f, cell=.1)
        o = np.array([[.2, .3, 1], [0, 0, 1], [.2, .3, 0], [2, 2, 1], [2, 0, 0], [0, 0, 0]])
        d = np.array([[0, 0, -1]] * 4 + [[0, 1, 0], [0, 0, 0]], float)
        ids, bary, t = index.ray_first_hit(o, d, t_max=1)
        np.testing.assert_array_equal(ids, [0, 0, -1, -1, -1, -1])
        np.testing.assert_allclose(bary[:2], [[.5, .2, .3], [1, 0, 0]])
        np.testing.assert_allclose(t[:2], 1)
        self.assertTrue((index.ray_first_hit(o, d, t_max=.999)[0] == -1).all())
        self.assertTrue((index.ray_first_hit(o, d, t_min=1)[0] == -1).all())
        self.assertEqual(index.ray_first_hit(o[2:3], d[2:3], t_min=-1e-9)[0][0], 0)
        self.assertEqual(index.ray_first_hit(o[:1], d[:1], side="front")[0][0], 0)
        self.assertEqual(index.ray_first_hit(o[:1], d[:1], side="back")[0][0], -1)
        self.assertEqual(index.ray_first_hit(-o[:1], -d[:1], side="back")[0][0], -1)
        self.assertEqual(index.ray_first_hit(o[:1] * [1, 1, -1], -d[:1], side="back")[0][0], 0)
        for empty in (MeshIndex(v, np.zeros((0, 3), int)), MeshIndex(v, f[2:])):
            ids, bary, t = empty.ray_first_hit(o, d)
            self.assertTrue((ids == -1).all())
            self.assertTrue(np.isnan(bary).all())
            self.assertTrue(np.isinf(t).all())
        self.assertEqual(len(index.ray_first_hit(o[:0], d[:0])[0]), 0)
        for kw in ({"t_max": -1}, {"t_max": np.nan}, {"step_cells": 0}, {"side": "unknown"}):
            with self.assertRaisesRegex(ValueError, "invalid rays"):
                index.ray_first_hit(o, d, **kw)


class TestClosest(unittest.TestCase):
    def test_tight_and_incomplete_bounds_preserve_exact_answers(self):
        verts, faces = box((0, 0, 0), (1, 1, 1))
        index = MeshIndex(verts, faces, cell=.2)
        points = np.random.default_rng(4).uniform(-1, 2, (80, 3))
        expected = index.closest(points)
        for bound in (expected[2], np.zeros(len(points)), np.full(len(points), 100.)):
            actual = index.closest(points, bound)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=1e-12)
        for bound in ([-1.] * len(points), [np.inf] * len(points), [1.]):
            with self.assertRaisesRegex(ValueError, "upper bounds"):
                index.closest(points, bound)

    def test_projection_edges_vertices_and_ties_have_expected_barycentrics(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        idx = MeshIndex(v, np.array([[0, 1, 2], [0, 1, 2], [0, 0, 0]]))
        p = np.array([[.2, .3, 2], [.5, -.2, 0], [2, 2, 0], [-1, -1, 0]])
        ids, bary, distances = idx.closest(p)
        np.testing.assert_array_equal(ids, 0)
        np.testing.assert_allclose(bary, [[.5, .2, .3], [.5, .5, 0], [0, .5, .5], [1, 0, 0]], atol=1e-15)
        np.testing.assert_allclose(distances, [2, .2, np.sqrt(4.5), np.sqrt(2)])

    def test_grid_and_brute_search_match_independent_triangle_distances(self):
        from unittest import mock
        rng = np.random.default_rng(718)
        verts = rng.uniform(-1, 1, (90, 3))
        faces = np.arange(90).reshape(-1, 3)
        points = np.r_[rng.uniform(-2, 2, (200, 3)), [[100, -200, 300]]]
        for scale in (.1, 1, 10):
            for limit in (4096, 0):
                with self.subTest(scale=scale, limit=limit), \
                        mock.patch("herogen.blender.meshindex.BRUTE_BOX_CELLS", limit):
                    idx = MeshIndex(verts * scale, faces, cell=.1 * scale)
                    ids, bary, distances = idx.closest(points * scale)
                    nearest = np.einsum("ni,nij->nj", bary, idx.tri[ids])
                    np.testing.assert_allclose(distances, point_mesh_distance(points * scale, verts * scale, faces),
                                               rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(np.linalg.norm(points * scale - nearest, axis=1), distances,
                                               rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(bary.sum(axis=1), 1, atol=1e-15)
                    self.assertGreaterEqual(bary.min(), 0)

    def test_empty_degenerate_and_nonfinite_queries(self):
        for faces in (np.empty((0, 3), int), np.array([[0, 0, 0]])):
            idx = MeshIndex(np.zeros((1, 3)), faces)
            ids, bary, distances = idx.closest([[1, 2, 3]])
            np.testing.assert_array_equal(ids, [-1])
            self.assertTrue(np.isnan(bary).all())
            self.assertTrue(np.isinf(distances).all())
            self.assertEqual(idx.closest(np.empty((0, 3)))[1].shape, (0, 3))
            with self.assertRaisesRegex(ValueError, "finite"):
                idx.closest([[np.nan, 0, 0]])


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="herogen-meshindex-"))
        synth.write_run(cls.tmp, synth.humanoid(), size=384, ppu=60.0)
        cls.vs = views.load_views(cls.tmp, 5.5)
        cls.verts, cls.faces, _ = carve.build_mesh(cls.vs, 5.5, 96)
        cls.sk = joints.estimate(cls.vs, 5.5)
        rng = np.random.default_rng(7)
        tri = cls.verts[cls.faces]
        k = rng.integers(0, len(cls.faces), 1500)
        surf = np.einsum("nk,nkj->nj", rng.dirichlet([1, 1, 1], 1500), tri[k])
        n = np.cross(tri[k, 1] - tri[k, 0], tri[k, 2] - tri[k, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        cls.points = np.concatenate([surf + n * rng.normal(0, 0.05, (1500, 1)),
                                     rng.uniform(cls.verts.min(0) - 0.3, cls.verts.max(0) + 0.3, (500, 3))])
        cls.rng = rng

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestMeshIndex(Base):
    def test_inside_matches_the_winding_number_off_the_surface(self):
        idx = MeshIndex(self.verts, self.faces)
        fast, slow = idx.inside(self.points), winding_inside(self.points, self.verts, self.faces)
        dist = point_mesh_distance(self.points, self.verts, self.faces)
        off = dist > 1e-9
        self.assertTrue(np.array_equal(fast[off], slow[off]))
        self.assertGreater(fast.sum(), 100)
        self.assertGreater((~fast).sum(), 100)

    def test_distance_and_rays_match_the_brute_force(self):
        idx = MeshIndex(self.verts, self.faces)
        self.assertTrue(np.allclose(idx.distance(self.points), point_mesh_distance(self.points, self.verts, self.faces),
                                    rtol=0, atol=1e-12))
        d = self.rng.normal(size=(len(self.points), 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        fast, slow = idx.ray_hits(self.points, d), ray_hits(self.points, d, self.verts[self.faces])
        self.assertTrue(np.array_equal(fast, slow))
        self.assertTrue(0 < fast.sum() < len(fast))

    def test_rays_through_edges_and_vertices_count_once(self):
        v, f = box((0, 0, 0), (1, 1, 1))
        idx = MeshIndex(v, f)
        g = np.linspace(0, 1, 5)        # the +X ray from these meets the box's edges and vertices exactly
        pts = np.array([[-0.5, y, z] for y in g for z in g] + [[0.5, y, z] for y in g[1:-1] for z in g[1:-1]])
        w = idx.winding(pts)
        self.assertTrue(np.all(w[:25] == 0))                # every ray from outside enters and leaves
        self.assertTrue(np.all(w[25:] == 1))                # from inside, it leaves once
        # a second, overlapping closed shell: the winding number adds, as the generalised winding number does
        v2, f2 = box((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))
        both = MeshIndex(np.concatenate([v, v2]), np.concatenate([f, f2 + len(v)]))
        self.assertEqual(int(both.winding(np.array([[0.5, 0.5, 0.5]]))[0]), 2)
        self.assertTrue(both.inside(np.array([[0.5, 0.5, 0.5]]))[0])

    def test_far_points_and_empty_meshes(self):
        idx = MeshIndex(self.verts, self.faces)
        far = np.array([[40.0, 0.0, 2.0]])
        self.assertAlmostEqual(float(idx.distance(far)[0]), float(point_mesh_distance(far, self.verts, self.faces)[0]))
        empty = MeshIndex(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        self.assertFalse(empty.inside(far)[0])
        self.assertTrue(np.isinf(empty.distance(far)[0]))
        self.assertFalse(empty.ray_hits(far, np.array([[1.0, 0, 0]]))[0])


class TestPoseTestThroughTheIndex(Base):
    def test_fast_and_brute_pose_tests_agree(self):
        secs, _ = split.build_sections(self.verts, self.faces, self.sk, cap="dome", rings=2, ring_points=12)
        poses = pose.pose_library(pose.bone_directions(self.sk))
        poses = {k: poses[k] for k in ("rest", "crouch", "walk_b")}
        for visible in (True, False):
            fast = pose.joint_gaps(secs, poses, visible=visible)
            slow = pose.joint_gaps(secs, poses, visible=visible, fast=False)
            self.assertEqual(fast.keys(), slow.keys())
            for p in slow:
                for j in slow[p]:
                    self.assertAlmostEqual(fast[p][j], slow[p][j], places=6, msg=f"{p} {j} visible={visible}")


class TestCuts(Base):
    def test_the_cut_centre_does_not_follow_the_vertex_density(self):
        t = np.concatenate([np.linspace(0, np.pi, 200, endpoint=False), np.linspace(np.pi, 2 * np.pi, 20,
                                                                                    endpoint=False)])
        ring = np.stack([np.cos(t), 0.5 * np.sin(t), np.zeros_like(t)], axis=1) + [0.3, 0.1, 2.0]
        nrm = np.array([0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(split.cut_centre(ring, nrm), [0.3, 0.1, 2.0], atol=2e-3))
        self.assertGreater(np.linalg.norm(split.cut_centre(ring, nrm, "vertex") - [0.3, 0.1, 2.0]), 0.2)

    def test_the_pivot_opening_is_the_largest_by_area_not_by_vertex_count(self):
        nrm = np.array([0.0, 0.0, 1.0])
        t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        small_dense = np.stack([0.1 * np.cos(t), 0.1 * np.sin(t), np.zeros_like(t)], axis=1)
        t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        big_sparse = np.stack([0.5 * np.cos(t), 0.5 * np.sin(t), np.zeros_like(t)], axis=1)
        self.assertGreater(len(small_dense), len(big_sparse))
        self.assertGreater(split.loop_area(big_sparse, nrm), 10 * split.loop_area(small_dense, nrm))
        self.assertAlmostEqual(split.loop_area(big_sparse, nrm), 0.5 * 12 * 0.25 * np.sin(2 * np.pi / 12), places=9)

    def test_a_cut_offset_moves_the_plane_along_its_normal(self):
        rig0, rig1 = split.build_rig(self.sk), split.build_rig(self.sk, {"Waist": -0.1})
        p0, n0 = rig0.planes["Waist"]
        p1, n1 = rig1.planes["Waist"]
        self.assertTrue(np.allclose(n0, n1))
        self.assertTrue(np.allclose(p1, p0 - 0.1 * n0))
        self.assertTrue(np.allclose(rig1.planes["Neck"][0], rig0.planes["Neck"][0]))

    def test_the_search_moves_only_near_the_gate_and_only_for_a_gain_beyond_noise(self):
        poses = pose.pose_library(pose.bone_directions(self.sk))
        gate = {k: poses[k] for k in ("rest", "crouch", "walk_b")}
        cover = {"owner": "UpperTorso", "radius": 2.5, "outset": 0.0}
        args = (self.verts, self.faces, self.sk, gate, (self.verts, self.faces), cover, (1, 8))
        # a gate far above every gap: no joint is near it, nothing moves, nothing is tried
        offsets, rec = cuts.choose(*args, gate=10.0)
        self.assertEqual(rec["eligible"], [])
        self.assertEqual(rec["trials"], {})
        self.assertTrue(all(o == 0.0 for o in offsets.values()))
        # a gate at zero: every joint is near it and may move (three fractions keep the test short)
        full = cuts.FRACTIONS
        cuts.FRACTIONS = (-0.25, 0.0, 0.25)
        try:
            offsets, rec = cuts.choose(*args, gate=0.0)
        finally:
            cuts.FRACTIONS = full
        self.assertEqual(set(rec["eligible"]), {k for ks in cuts.GROUPS.values() for k in ks})
        zero = rec["unmoved_worst_gap"]
        for group, kinds in cuts.GROUPS.items():
            trial = rec["trials"][group]
            chosen = {k: rec["chosen_fraction"][k] for k in kinds}
            moved = {k: f for k, f in chosen.items() if f}
            pick = next(t for t in trial["tried"] if {k: f for k, f in t["fractions"].items() if f} == moved)
            unmoved = trial["tried"][0]
            self.assertEqual(unmoved["fractions"], {})
            if moved:      # a move must beat the unmoved cuts by more than the noise and open no neighbour
                self.assertLess(pick["worst_gap"], unmoved["worst_gap"] - cuts.NOISE_STUDS)
                self.assertEqual(pick["neighbours_opened"], [])
                for t in trial["tried"][1:]:      # and it has the lowest gap plus distance cost of those that do
                    if t["worst_gap"] < unmoved["worst_gap"] - cuts.NOISE_STUDS and not t["neighbours_opened"]:
                        self.assertLessEqual(pick["score"], t["score"])
            else:          # no move beat the unmoved cuts by more than the noise without opening a neighbour
                for t in trial["tried"][1:]:
                    self.assertTrue(t["worst_gap"] >= unmoved["worst_gap"] - cuts.NOISE_STUDS or t["neighbours_opened"])
        for kind in ("Elbow", "Wrist", "Knee", "Ankle"):      # left and right alike
            f = rec["chosen_fraction"][kind]
            for side in ("Left", "Right"):
                j = side + kind
                self.assertAlmostEqual(offsets[j], f * rec["radius_studs"][j], places=3)
        self.assertTrue(set(cuts.affected(("Neck", "Waist"))) <= set(zero))


if __name__ == "__main__":
    unittest.main()
