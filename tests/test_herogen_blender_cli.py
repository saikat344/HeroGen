"""The Blender CLI refuses to build on a stage that did not pass."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.blender.__main__ import main  # noqa: E402


class UpstreamGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "r"
        self.run.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _main(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["run", "--run", str(self.run), *args])
        return code, err.getvalue()

    def test_missing_stage2_record_stops_stage3(self):
        code, err = self._main("--stages", "3")
        self.assertEqual(code, 2)
        self.assertIn("stage-2.json not found", err)

    def test_failed_stage2_stops_stage3(self):
        (self.run / "stage-2.json").write_text(json.dumps({"stage": 2, "ok": False}))
        code, err = self._main("--stages", "3,4")
        self.assertEqual(code, 2)
        self.assertIn("ok=False", err)

    def test_stage2_passed_on_threshold_masks_stops_stage3(self):
        # A record an earlier version passed on background-threshold masks is not a measurement.
        (self.run / "stage-2.json").write_text(json.dumps({"stage": 2, "ok": True, "details": {
            "mask_method": {"front": "sam-3.1", "right": "threshold", "top": "visual_hull"}}}))
        code, err = self._main("--stages", "3")
        self.assertEqual(code, 2)
        self.assertIn("threshold masks (right)", err)

    def test_empty_stages_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            self._main("--stages", ",")
        self.assertEqual(cm.exception.code, 2)

    def test_failed_stage3_stops_stage4(self):
        (self.run / "stage-3.json").write_text(json.dumps({"stage": 3, "ok": False}))
        code, err = self._main("--stages", "4")
        self.assertEqual(code, 2)
        self.assertIn("stage-3.json", err)

    def test_missing_remote_result_without_fallback_is_a_clear_error(self):
        from herogen.blender import synth

        synth.write_run(self.run, synth.humanoid(), size=256, ppu=40.0)
        code, err = self._main("--stages", "3", "--mesh-source", "sam3d-objects", "--no-fallback",
                               "--remote-cache", str(Path(self.tmp.name) / "empty-cache"))
        self.assertEqual(code, 1)
        self.assertIn("stage 3 stopped:", err)
        self.assertNotIn("Traceback", err)

    def test_default_requests_the_hybrid_and_reports_fallback(self):
        from unittest import mock
        from herogen.blender import stage3, synth

        synth.write_run(self.run, synth.humanoid(), size=256, ppu=40.0)
        result = {"ok": True, "seconds": 0.0, "checks": [], "fallback": {"reason": "cache miss"}}
        with mock.patch.object(stage3, "run_stage3", return_value=result) as build:
            code, err = self._main("--stages", "3")
        self.assertEqual(code, 0)
        self.assertEqual(build.call_args.kwargs["mesh_source"], "sam3d-objects")
        self.assertEqual(build.call_args.kwargs["joints_source"], "sam3d-body")
        self.assertIn("fallback: cache miss; using carve", err)

    def test_failed_hybrid_build_records_fallback_and_preserves_spend(self):
        from unittest import mock
        from herogen.blender import stage3
        from herogen.blender.bl import BlenderError

        def build(run, **kwargs):
            rec = kwargs["rec"]
            if kwargs["mesh_source"] == "sam3d-objects":
                rec.data["cost_usd"] = 0.12
                rec.add({"name": "partial", "pass": False})
                raise BlenderError("hybrid export failed")
            self.assertEqual(kwargs["joints_source"], "silhouette")
            self.assertFalse(kwargs["fallback"])
            self.assertEqual(rec.data["checks"], [])
            rec.data["mesh_source"] = "carve"
            return rec.finish()

        with mock.patch.object(stage3, "_run_stage3", side_effect=build) as build_mock:
            rec = stage3.run_stage3(self.run)
        self.assertEqual(build_mock.call_count, 2)
        self.assertEqual(rec["mesh_source_requested"], "sam3d-objects")
        self.assertEqual(rec["cost_usd"], 0.12)
        self.assertIn("hybrid export failed", rec["fallback"]["reason"])
        self.assertTrue(rec["ok"])

    def test_quality_rejection_never_retries_carve(self):
        from unittest import mock
        from herogen.blender import stage3

        with mock.patch.object(stage3, "_run_stage3", return_value={"ok": False}) as build:
            self.assertFalse(stage3.run_stage3(self.run)["ok"])
        self.assertEqual(build.call_count, 1)

    def test_body_detector_flag_reaches_stage3(self):
        from unittest import mock

        from herogen.blender import stage3, synth

        synth.write_run(self.run, synth.humanoid(), size=256, ppu=40.0)
        seen = []
        result = {"ok": True, "seconds": 0.0, "checks": []}
        with mock.patch.object(stage3, "run_stage3", side_effect=lambda *a, **k: seen.append(k) or result):
            self._main("--stages", "3", "--joints", "sam3d-body")
            self._main("--stages", "3", "--joints", "sam3d-body", "--body-detector", "vitdet")
        self.assertEqual([k["body_detector"] for k in seen], [None, "vitdet"])
        with self.assertRaises(SystemExit) as cm:
            self._main("--stages", "3", "--body-detector", "yolo")
        self.assertEqual(cm.exception.code, 2)

    def test_modal_that_cannot_start_is_a_clear_error(self):
        from herogen.blender import synth

        synth.write_run(self.run, synth.humanoid(), size=256, ppu=40.0)
        broken = Path(self.tmp.name) / "modal"
        broken.write_text("not a program")          # exists, but cannot be executed
        broken.chmod(0o644)
        old = os.environ.get("HEROGEN_MODAL")
        os.environ["HEROGEN_MODAL"] = str(broken)
        try:
            code, err = self._main("--stages", "3", "--mesh-source", "sam3d-objects", "--no-fallback",
                                   "--remote", "live", "--remote-cache", str(Path(self.tmp.name) / "cache"))
        finally:
            if old is None:
                os.environ.pop("HEROGEN_MODAL", None)
            else:
                os.environ["HEROGEN_MODAL"] = old
        self.assertEqual(code, 1)
        self.assertIn("stage 3 stopped: objects: could not start modal", err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn(self.tmp.name, err)

    def test_stage4_without_a_mesh_is_a_clear_error(self):
        (self.run / "stage-3.json").write_text(json.dumps({"stage": 3, "ok": True}))
        (self.run / "stage-4.json").write_text(json.dumps({"stage": 4, "ok": True}))   # an earlier, passed run
        code, err = self._main("--stages", "4")
        self.assertEqual(code, 1)
        self.assertIn("stage 4 stopped: FileNotFoundError", err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertFalse((self.run / "stage-4.json").exists())      # no stale pass left for later stages

    def test_a_bad_input_is_a_clear_error(self):
        (self.run / "stage-2.json").write_text(json.dumps({"stage": 2, "ok": True}))
        code, err = self._main("--stages", "3")            # no views or masks in the run folder
        self.assertEqual(code, 1)
        self.assertIn("stage 3 stopped:", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
