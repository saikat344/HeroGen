"""Exact distance and visibility equivalence on generated geometry."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from herogen.blender import PARENT, SECTIONS, carve, pose, split, stage4
from herogen.blender.meshindex import MeshIndex


class TestExactQueries(unittest.TestCase):
    def test_visible_max_bound_matches_exhaustive_values_and_skips_losers(self):
        world = np.column_stack((np.arange(70), np.zeros((70, 2))))
        # Descending distances, including a tie and adjacent representable floats. Do not round the bound.
        dist = np.linspace(.02, 2., len(world))[::-1]
        dist[1] = dist[0]
        dist[2] = np.nextafter(dist[1], 0.)
        for visible in ({0}, {2}, {31}, {32}, {69}, set()):
            for frontier in (1, 32, 128):
                tested = []

                def visibility(near, points):
                    ids = points[:, 0].astype(int)
                    tested.extend(ids.tolist())
                    return np.array([i in visible for i in ids])

                for prior in (0., .5, 3.):
                    tested.clear()
                    expected = max([prior] + [float(dist[i]) for i in visible])
                    with patch.object(pose, "_visible", side_effect=visibility):
                        result = pose._visible_max(None, world, dist, prior, frontier)
                    self.assertEqual(result.hex(), expected.hex())
                    self.assertTrue(all(dist[i] > prior for i in tested))
                    if prior == 3.:
                        self.assertEqual(tested, [])
                    if visible in ({0}, {31}) and prior == 0. and frontier == 32:
                        self.assertEqual(len(tested), 32)
                    if not visible and prior == 0.:
                        self.assertEqual(len(tested), len(world))

        with patch.object(pose, "_visible", return_value=np.array([False])):
            self.assertEqual(pose._visible_max(None, world[:1], dist[:1], 0.), 0.)
        self.assertEqual(pose._visible_max(None, world[:0], dist[:0], .25), .25)
        rng = np.random.default_rng(434)
        for _ in range(25):
            shuffled = rng.permutation(dist)
            flags = rng.random(len(world)) < .2
            prior = float(rng.uniform(0., 3.))
            expected = max([prior] + list(shuffled[flags]))
            for frontier in (1, 7, 32, 100):
                with patch.object(pose, "_visible", side_effect=lambda near, p: flags[p[:, 0].astype(int)]):
                    actual = pose._visible_max(None, world, shuffled, prior, frontier)
                self.assertEqual(actual.hex(), float(expected).hex())

    def test_rest_cache_matches_uncached_queries_and_stays_within_one_call(self):
        verts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        sections = {n: split.Section(n, verts.copy(), faces.copy(), np.zeros(3), [], 0) for n in SECTIONS}
        child = "Head"
        sections[child].rims = [{"neighbour": PARENT[child], "points": np.array([[.2, .2, .1]])}]
        poses = {"rest": {}, "repeat": {}, "turn": {child: pose.rot([1, 0, 0], 20)}}
        for fast in (False, True):
            with patch.object(pose, "REST_CACHE_BYTES", 0):
                expected = pose.joint_gaps(sections, poses, fast=fast)
            with self.subTest(fast=fast):
                self.assertEqual(pose.joint_gaps(sections, poses, fast=fast), expected)
                # Mutated geometry on a later call must be queried afresh.
                sections[child].verts[:, 2] += .25
                with patch.object(pose, "REST_CACHE_BYTES", 0):
                    changed = pose.joint_gaps(sections, poses, fast=fast)
                self.assertEqual(pose.joint_gaps(sections, poses, fast=fast), changed)
                sections[child].verts[:, 2] -= .25

        class Counting(MeshIndex):
            calls = 0

            def inside(self, points):
                Counting.calls += 1
                return super().inside(points)

        with patch.object(pose, "MeshIndex", Counting):
            pose.joint_gaps(sections, {"rest": {}, "repeat": {}})
        self.assertEqual(Counting.calls, 2)  # each rest section once, despite two identical poses
        Counting.calls = 0
        almost_identity = np.eye(3)
        almost_identity[0, 0] = np.nextafter(1., 2.)
        with patch.object(pose, "MeshIndex", Counting):
            pose.joint_gaps(sections, {"rest": {}, "one_bit": {child: almost_identity}})
        self.assertEqual(Counting.calls, 4)  # a last-bit coordinate change must miss the cache
        Counting.calls = 0
        with patch.object(pose, "MeshIndex", Counting), patch.object(pose, "REST_CACHE_BYTES", 1):
            pose.joint_gaps(sections, {"rest": {}, "repeat": {}})
        self.assertEqual(Counting.calls, 4)

    def test_invalid_workers_preserve_the_previous_record(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            record = run / "stage-4.json"
            record.write_text("previous")
            with patch.object(stage4.bl, "run_job") as blender:
                with self.assertRaisesRegex(ValueError, "cut workers"):
                    stage4.run_stage4(run, cut_workers=0)
                blender.assert_not_called()
            self.assertEqual(record.read_text(), "previous")

    def test_distance_matches_original_brute_arithmetic(self):
        rng = np.random.default_rng(805)
        verts = rng.normal(size=(150, 3))
        faces = np.arange(150).reshape(-1, 3)
        faces = np.concatenate((faces, [[0, 0, 0]], faces[:3]))
        points = np.concatenate((rng.normal(size=(91, 3)), verts[:5], [[20, 20, 20]]))
        expected = carve.point_mesh_distance(points, verts, faces)
        for cell in (.35, 1.0):
            with self.subTest(cell=cell):
                index = MeshIndex(verts, faces, cell=cell)
                np.testing.assert_array_equal(index.distance(points), expected)
                self.assertEqual(index.distance(points[:0]).shape, (0,))

    def test_distance_evaluates_each_pair_once_per_expansion(self):
        verts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        index = MeshIndex(verts, np.array([[0, 1, 2]]), cell=.1)
        original = index._pair_distance
        calls = []

        def measured(points, triangles):
            pairs = [(p.tobytes(), int(t)) for p, t in zip(points, triangles)]
            self.assertEqual(len(pairs), len(set(pairs)))
            calls.append(len(pairs))
            return original(points, triangles)

        index._pair_distance = measured
        np.testing.assert_array_equal(index.distance([[.3, .3, .01]]), [.01])
        self.assertEqual(calls, [1])

    def test_visibility_matches_all_original_rays(self):
        verts = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
        faces = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                          [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
        points = np.concatenate((np.random.default_rng(42).uniform(-1, 2, (47, 3)),
                                 verts, [[.5, .5, .5], [.5, .5, 0]]))
        for query in (MeshIndex(verts, faces, cell=.2), pose._Brute(verts, faces),
                      MeshIndex(verts, faces[:0])):
            directions = np.tile(pose.VIEW_DIRS, (len(points), 1))
            origins = np.repeat(points, len(pose.VIEW_DIRS), axis=0) + directions * 2e-3
            blocked = (query.ray_hits(origins, directions, step_cells=8) if isinstance(query, MeshIndex)
                       else query.ray_hits(origins, directions))
            expected = ~blocked.reshape(len(points), -1).all(axis=1)
            for batch in (1, 4, 7, 26):
                np.testing.assert_array_equal(pose._visible(query, points, batch), expected)
            if isinstance(query, MeshIndex):
                for steps in (1, 2, 4, 8, 32):
                    actual = query.ray_hits(origins, directions, step_cells=steps)
                    np.testing.assert_array_equal(actual, blocked)
            self.assertEqual(pose._visible(query, points[:0]).shape, (0,))

    def test_visibility_short_circuit_has_an_exact_bound(self):
        class Rays:
            def __init__(self):
                self.count = {0: 0, 10: 0, 20: 0}

            def ray_hits(self, origins, directions):
                # Point 0 escapes immediately, point 10 at direction 17, point 20 is enclosed.
                ids = np.rint(origins[:, 0]).astype(int)
                out = np.ones(len(ids), dtype=bool)
                for k, point in enumerate(ids):
                    number = self.count[point]
                    self.count[point] += 1
                    out[k] = not (point == 0 or (point == 10 and number == 17))
                return out

        query = Rays()
        np.testing.assert_array_equal(pose._visible(query, np.array([[0., 0, 0], [10., 0, 0], [20., 0, 0]])),
                                      [True, True, False])
        self.assertEqual(query.count, {0: 4, 10: 20, 20: 26})


if __name__ == "__main__":
    unittest.main()
