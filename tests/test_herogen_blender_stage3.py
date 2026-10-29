"""Tests for herogen stage 3 (herogen/blender): masks, cameras and the carved base mesh, on a synthetic
A-pose humanoid.

The numpy parts run everywhere numpy and Pillow are installed; the end-to-end test needs Blender and skips
cleanly without it. Nothing generated here is written inside the repository.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import PIL  # noqa: F401
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

if HAVE_DEPS:
    from herogen.blender import bl, carve, synth, views


def _open_edges(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class SyntheticBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="herogen-blender-test-"))
        cls.rundir = cls.tmp / "run"
        cls.truth = synth.write_run(cls.rundir, synth.humanoid(), size=384, ppu=60.0)
        cls.vs = views.load_views(cls.rundir, 5.5)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestViewsAndCarve(SyntheticBase):
    def test_masks_and_scale(self):
        self.assertEqual(sorted(self.vs.views), sorted(views.VIEWS))
        self.assertEqual(self.vs.missing, [])
        self.assertAlmostEqual(self.vs.pps, 60.0, delta=0.3)       # 330 px figure / 5.5 studs

    def test_view_mapping_puts_joints_inside_every_mask(self):
        pts = np.array(list(self.truth["joints"].values()))
        for name, v in self.vs.views.items():
            self.assertTrue(v.inside(pts).all(), name)

    def test_carved_hull_is_closed_and_contract_sized(self):
        verts, faces, info = carve.build_mesh(self.vs, 5.5, 96)
        self.assertEqual(_open_edges(faces), 0)
        self.assertAlmostEqual(verts[:, 2].min(), 0.0, places=6)
        self.assertAlmostEqual(verts[:, 2].max(), 5.5, places=6)
        # arm span from the front mask: within two voxels
        r0, r1, c0, c1 = views.bbox(self.vs.views["front"].mask)
        span = (c1 - c0 + 1) / self.vs.pps
        self.assertAlmostEqual(verts[:, 0].max() - verts[:, 0].min(), span, delta=2 * info["voxel_size_studs"])
        occ_pts = carve.winding_inside(np.array(list(self.truth["joints"].values())), verts, faces)
        self.assertGreaterEqual(int(occ_pts.sum()), len(occ_pts) - 1)   # HeadTop sits on the surface

    def test_missing_views_are_skipped_and_recorded(self):
        run = self.tmp / "missing"
        shutil.copytree(self.rundir, run)
        (run / "masks" / "top.png").unlink()
        (run / "masks" / "back.png").unlink()
        vs = views.load_views(run, 5.5)
        self.assertEqual(sorted(vs.missing), ["back", "top"])
        verts, faces, _ = carve.build_mesh(vs, 5.5, 64)
        self.assertEqual(_open_edges(faces), 0)


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class TestStage3Blender(SyntheticBase):
    def test_stage3_silhouette_iou(self):
        from herogen.blender.stage3 import run_stage3

        r3 = run_stage3(self.rundir, nz=128, timeout=300)
        self.assertTrue(r3["ok"], r3["checks"])
        ious = [c["value"] for c in r3["checks"] if c["name"].startswith("silhouette_iou_")]
        self.assertEqual(len(ious), 5)
        self.assertGreater(min(ious), 0.9)
        self.assertEqual((self.rundir / "03-mesh.glb").read_bytes()[:4], b"glTF")
        self.assertEqual(sorted(p.name for p in (self.rundir / "03-renders").glob("*.png")),
                         sorted(f"{v}.png" for v in views.VIEWS))
        rec = json.loads((self.rundir / "stage-3.json").read_text())
        self.assertEqual((rec["stage"], rec["cost_usd"]), (3, 0.0))


if __name__ == "__main__":
    unittest.main()
