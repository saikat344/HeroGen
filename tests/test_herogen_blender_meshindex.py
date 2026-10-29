"""The grid-accelerated mesh queries (herogen.blender.meshindex) give the brute-force answers, and the pose test
gives the same gaps through either.

Synthetic inputs only; nothing needs Blender.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from herogen.blender import carve, joints, pose, split, synth, views
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


if __name__ == "__main__":
    unittest.main()
