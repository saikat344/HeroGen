"""Ground-truth bench dataset: the held-out draw, manifests and head close-up cameras (no Blender needed)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

try:
    import numpy as np

    from herogen.bench import cameras as CAM
    from herogen.bench import dataset as DS

    HAVE_DEPS = True
except ImportError:  # numpy or Pillow missing
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class HeldoutDrawTest(unittest.TestCase):
    def test_share_and_stability(self):
        groups = [f"g{i}" for i in range(20000)]
        drawn = [g for g in groups if DS.heldout_draw(g, "seed-1", 0.1)]
        self.assertAlmostEqual(len(drawn) / len(groups), 0.1, delta=0.01)
        # each group's side depends on the group alone: adding groups never moves an existing one
        first = set(groups[:5000])
        self.assertEqual([g for g in drawn if g in first],
                         [g for g in groups[:5000] if DS.heldout_draw(g, "seed-1", 0.1)])
        self.assertNotEqual(drawn, [g for g in groups if DS.heldout_draw(g, "seed-2", 0.1)])

    def test_edges(self):
        self.assertFalse(any(DS.heldout_draw(f"g{i}", "s", 0.0) for i in range(100)))
        self.assertTrue(all(DS.heldout_draw(f"g{i}", "s", 1.0) for i in range(100)))


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class ManifestTest(unittest.TestCase):
    def _load(self, chars):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            p.write_text(json.dumps({"characters": chars}))
            return DS.load_manifest(p)

    def test_ids_must_be_unique_folder_names(self):
        self.assertEqual(len(self._load([{"id": "a", "asset": "a.glb"}, {"id": "b", "asset": "b.glb"}])), 2)
        for bad in ([{"id": "a", "asset": "x"}, {"id": "a", "asset": "y"}], [{"id": "../x", "asset": "x"}],
                    [{"id": "", "asset": "x"}], [{"id": "..", "asset": "x"}]):
            with self.assertRaises(ValueError):
                self._load(bad)

    def test_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "c"
            self.assertEqual(DS.status(d), "todo")
            d.mkdir()
            (d / DS.FAILED).write_text("{}")
            self.assertEqual(DS.status(d), "failed")
            (d / DS.SAMPLE).write_text("{}")
            self.assertEqual(DS.status(d), "done")


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class ViewCheckTest(unittest.TestCase):
    def test_a_few_pixels_at_an_alpha_cutoff_pass_a_misplaced_figure_does_not(self):
        rec = {"off_edge_px": 4, "border_px": 0, "mask_px": 460000}
        self.assertTrue(DS.view_ok(rec, "building"))
        self.assertFalse(DS.view_ok(dict(rec, off_edge_px=2000), "building"))
        self.assertFalse(DS.view_ok(dict(rec, off_edge_px=1, mask_px=1000), "building"))

    def test_nothing_seen_fails(self):
        self.assertFalse(DS.view_ok({"off_edge_px": 0, "border_px": 0, "mask_px": 0}, "building"))
        self.assertFalse(DS.view_ok({"off_edge_px": 0, "border_px": 0, "mask_px": 0}, "head"))

    def test_invisible_holes(self):
        res = {"alpha": {"A": {"mode": "clip", "image": None}, "B": {"mode": "opaque", "image": None},
                         "C": {"mode": "clip", "image": "tex.png"}}}
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(Path(tmp) / "front.npz", mask=np.zeros((4, 4), dtype=bool))
            self.assertEqual(DS.invisible_holes(res, Path(tmp)), ["A"])
            np.savez_compressed(Path(tmp) / "front.npz", mask=np.ones((4, 4), dtype=bool))
            self.assertEqual(DS.invisible_holes(res, Path(tmp)), [])
        self.assertEqual(DS.invisible_holes({"alpha": {"B": {"mode": "opaque"}}}, Path(".")), [])

    def test_border(self):
        rec = {"off_edge_px": 0, "border_px": 12, "mask_px": 1000}
        self.assertFalse(DS.view_ok(rec, "evaluation"))
        self.assertTrue(DS.view_ok(rec, "head"))            # a close-up crops the body below the head


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class UsabilityTest(unittest.TestCase):
    def setUp(self):
        self.sample = {"height_units": 5.5, "pose": "a-pose", "limbs": [{"converged": True}] * 4,
                       "head": {"source": "rig", "centre": [0.0, 0.0, 4.9]}}
        rng = np.random.default_rng(0)
        self.flat = np.zeros((64, 64, 4), dtype=np.uint8)
        self.flat[..., :3] = rng.integers(0, 256, size=(64, 64, 3))
        self.flat[..., 3] = 255
        self.head = np.zeros((64, 64), dtype=bool)
        self.head[10:50, 12:52] = True

    def test_a_good_sample(self):
        self.assertEqual(DS.usability(self.sample, self.flat, self.head), [])

    def test_each_reason(self):
        s = dict(self.sample, head={"source": "none"})
        self.assertIn("no head region", DS.usability(s, self.flat, self.head)[0])
        s = dict(self.sample, head={"source": "rig", "centre": [0, 0, 2.0]})
        self.assertIn("not upright", DS.usability(s, self.flat, self.head)[0])
        small = np.zeros_like(self.head)
        small[30:33, 30:33] = True
        self.assertIn("head fills", DS.usability(self.sample, self.flat, small)[0])
        plain = self.flat.copy()
        plain[..., :3] = 200
        self.assertIn("untextured", DS.usability(self.sample, plain, self.head)[0])
        self.assertIn("not posed", DS.usability(dict(self.sample, pose="rest"), self.flat, self.head)[0])
        s = dict(self.sample, limbs=[{"converged": True}, {"converged": False}])
        self.assertIn("A-pose", DS.usability(s, self.flat, self.head)[0])


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class HeadCameraTest(unittest.TestCase):
    def test_centred_on_the_head_at_one_scale(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform([-0.3, -0.2, 4.6], [0.3, 0.4, 5.5], size=(500, 3))
        cams = CAM.head_cameras(pts, 256, 12.0)
        self.assertEqual([c["view"] for c in cams], ["head-front", "head-front-left-45", "head-left",
                                                      "head-back", "head-right", "head-front-right-45"])
        self.assertEqual(len({c["pixels_per_unit"] for c in cams}), 1)
        centre = (pts.min(0) + pts.max(0)) / 2
        widest = 0.0
        for cam in cams:
            cr = CAM.project(pts, cam)
            self.assertTrue(np.all((cr >= 0) & (cr <= 256)), cam["view"])
            # the bounding-box centre projects to the image centre
            np.testing.assert_allclose(CAM.project(centre[None], cam)[0], [128, 128], atol=1e-9)
            d = pts - centre
            widest = max(widest, *(float(np.abs(d @ np.asarray(cam[k])).max()) for k in ("right", "up")))
        self.assertAlmostEqual(widest * 2 * cams[0]["pixels_per_unit"], CAM.HEAD_FILL * 256, delta=0.01)

    def test_rays_start_clear_of_the_figure(self):
        pts = np.array([[0.0, 0.0, 5.0], [0.1, 0.1, 5.4]])
        for cam in CAM.head_cameras(pts, 64, 12.0):
            origins, _ = CAM.pixel_rays(cam, np.array([0, 63]), np.array([0, 63]))
            self.assertTrue(np.all(CAM.depth_of(origins, cam) <= -12.0 + 1e-9))

    def test_a_point_has_no_size(self):
        with self.assertRaises(ValueError):
            CAM.head_cameras(np.zeros((3, 3)), 64, 12.0)


if __name__ == "__main__":
    unittest.main()
