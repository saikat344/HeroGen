"""Silhouette measurement failures must identify the missing anatomical band."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from herogen.blender import body_joints, joints, stage3, stage4, views
from herogen.blender.record import StageRecord


def short_torso_views():
    # A large head over a robe with no leg gap; short arms separate below the 0.47 H crotch prior.
    view = views.View("front", np.zeros((600, 600), bool), 100., 300., 575.)
    x, z = view.pixel_uv()
    view.mask = ((abs(x) < .4) & (z >= 0) & (z < 5.5)
                 | (abs(x) < 1) & (z > 3.5) & (z < 5.5)
                 | (abs(x) > .6) & (abs(x) < 1.1) & (z > 2.2) & (z < 2.5)
                 | (abs(x) < 1.1) & (z >= 2.5) & (z < 2.6))
    return views.views_from_masks({"front": view.mask}, 5.5)


class JointMeasurementTests(unittest.TestCase):
    def test_short_torso_rejects_inverted_waist_band(self):
        with self.assertRaises(joints.JointEstimationError) as caught:
            joints.estimate(short_torso_views(), 5.5)
        message = str(caught.exception)
        for text in ("silhouette waist cannot be measured", "crotch at 2.585", "armpit at 2.495",
                     "no waist search rows", "no gap between the legs"):
            self.assertIn(text, message)

    def test_empty_width_band_identifies_its_rows(self):
        for lo, hi in ((2, 2), (3, 1), (4, 5)):
            with self.subTest(lo=lo, hi=hi), self.assertRaisesRegex(
                    joints.JointEstimationError, rf"no width samples in rows \[{lo}, {hi}\)"):
                joints._band_centre(np.ones(4), np.arange(4), lo, hi, .03)

    def test_measurable_band_preserves_centre_and_width(self):
        self.assertEqual(joints._band_centre(np.array([8., 2., 2., 4.]), np.arange(4), 0, 4, .03),
                         (1.5, 2.))

    def test_stage3_reports_missing_optional_prior_before_mesh_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rec = StageRecord(run, 3, "base mesh")
            with mock.patch.object(stage3, "load_views", return_value=short_torso_views()), \
                    mock.patch.object(stage3.carve, "build_mesh", side_effect=RuntimeError("stop before mesh")) as build:
                with self.assertRaisesRegex(RuntimeError, "stop before mesh"):
                    stage3._run_stage3(run, rec=rec)
            self.assertIsNone(build.call_args.kwargs["skeleton"])
            self.assertIn("silhouette waist cannot be measured", rec.data["notes"])
            self.assertIn("arm depth prior skipped", rec.data["notes"])

    def test_stage4_records_rejection_even_with_complete_body_joints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "03-mesh.glb").touch()
            (run / "stage-3.json").write_text(json.dumps({"mesh_source": "sam3d-objects", "shape_views": ["front"]}))
            saved = body_joints.write_joints(run / "03-joints.json", "sam3d-body",
                                             {k: [0., 0., 1.] for k in body_joints.JOINT_NAMES}, {})
            (run / "stage-4.json").write_text('{"ok": true}')
            with mock.patch.object(stage4, "load_views", return_value=short_torso_views()), \
                    mock.patch.object(stage4.bl, "run_job") as blender:
                with self.assertRaisesRegex(joints.JointEstimationError, "SAM 3D Body joint positions alone"):
                    stage4.run_stage4(run)
            blender.assert_not_called()
            self.assertEqual(json.loads((run / "03-joints.json").read_text()), saved)
            rec = json.loads((run / "stage-4.json").read_text())
            self.assertFalse(rec["ok"])
            self.assertEqual(rec["status"], "error")
            self.assertIn("silhouette waist cannot be measured", rec["error"])
            self.assertEqual(rec["outputs"], [])


if __name__ == "__main__":
    unittest.main()
