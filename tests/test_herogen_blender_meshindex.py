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
