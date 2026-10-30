"""Input checks of stages 0 and 1 on synthetic sketches drawn by code: each check's verdict (pass, warn or reject),
its record and message, and that a rejected sketch costs no generated image."""

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

from herogen.pipeline import inputs  # noqa: E402
from herogen.pipeline.__main__ import run_pipeline  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place  # noqa: E402


class InputCheckTest(unittest.TestCase):
    def test_verdicts(self):
        ok = inputs.input_check("x", 5, 3, ">=", "reject", "too low")
        self.assertEqual((ok["verdict"], ok["pass"], ok["gate"]), ("pass", True, True))
        self.assertNotIn("message", ok)
        bad = inputs.input_check("x", 2, 3, ">=", "reject", "too low")
        self.assertEqual((bad["verdict"], bad["pass"], bad["gate"], bad["message"]), ("reject", False, True, "too low"))
        warn = inputs.input_check("x", 2, 3, ">=", "warn", "low")
        self.assertEqual((warn["verdict"], warn["pass"], warn["gate"]), ("warn", False, False))
        self.assertEqual(inputs.rejections([ok, bad, warn]), [bad])
        self.assertEqual(inputs.warnings([ok, bad, warn]), [warn])
        self.assertEqual(inputs.rejection_error([bad, warn, {**bad, "message": "also"}]), "too low; also")
        with self.assertRaises(ValueError):
            inputs.input_check("x", 2, 3, ">=", "gate", "")

    def test_size(self):
        self.assertEqual(inputs.size_check(256, 900)["verdict"], "pass")
        c = inputs.size_check(1000, 255)
        self.assertEqual((c["value"], c["verdict"]), (255, "reject"))
        self.assertIn("255 px", c["message"])

    def test_line_contrast(self):
        fig = place(figure_mask(400, "front"), (600, 560), (100, 80))
        self.assertEqual(inputs.line_contrast(paint(fig, colour=(60, 60, 60), bg=(250, 250, 250))), 190)
        faint = paint(fig, colour=(238, 238, 238), bg=(250, 250, 250))
        self.assertEqual(inputs.line_contrast(faint), 12)
        c = inputs.contrast_check(faint)
        self.assertEqual((c["verdict"], c["threshold"]), ("reject", inputs.MIN_LINE_CONTRAST))
        self.assertIn("only 12 grey levels darker than the paper", c["message"])
        self.assertEqual(inputs.contrast_check(paint(fig, colour=(200, 200, 200)))["verdict"], "pass")


class _SketchRun(unittest.TestCase):
    """A temporary runs root; runs replay from an empty cache, so any model call would stop the run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, img, stage_list=(0,), run_id="r") -> int:
        path = Path(self.tmp.name) / f"{run_id}.png"
        img.save(path)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = run_pipeline(path, run_id, list(stage_list), mode="replay", runs=str(self.root))
        self.err = err.getvalue()
        return code

    def _stage(self, n, run_id="r") -> dict:
        return json.loads((self.root / run_id / f"stage-{n}.json").read_text())


class Stage0Test(_SketchRun):
    def test_a_small_sketch_is_rejected(self):
        img = paint(place(figure_mask(150, "front"), (200, 240), (25, 40)), colour=(60, 60, 60))
        self.assertEqual(self._run(img, (0, 1)), 3)
        s0 = self._stage(0)
        self.assertEqual((s0["ok"], s0["status"]), (False, "input_rejected"))
        self.assertIn("too small: its shorter side is 200 px", s0["error"])
        self.assertEqual({c["name"]: c["verdict"] for c in s0["checks"]},
                         {"min_side_px": "reject", "sketch_line_contrast": "pass"})
        self.assertIn("input rejected: the sketch is too small", self.err)
        self.assertFalse((self.root / "r" / "stage-1.json").exists())

    def test_faint_lines_are_rejected_before_any_model_call(self):
        img = paint(place(figure_mask(400, "front"), (600, 560), (100, 80)), colour=(240, 240, 240))
        self.assertEqual(self._run(img, (0, 1, 2)), 3)  # replays from an empty cache: a model call would stop it
        s0 = self._stage(0)
        self.assertEqual(s0["status"], "input_rejected")
        self.assertEqual({c["name"]: c["verdict"] for c in s0["checks"]},
                         {"min_side_px": "pass", "sketch_line_contrast": "reject"})
        self.assertIn("the lines are too faint", s0["error"])
        self.assertEqual(s0["calls"], [])
        self.assertFalse((self.root / "r" / "stage-1.json").exists())

    def test_every_rejection_is_reported_at_once(self):
        img = paint(place(figure_mask(150, "front"), (200, 240), (25, 40)), colour=(240, 240, 240))
        self.assertEqual(self._run(img), 3)
        err = self._stage(0)["error"]
        self.assertIn("too small", err)
        self.assertIn("too faint", err)

    def test_a_good_sketch_passes(self):
        img = paint(place(figure_mask(400, "front"), (600, 560), (100, 80)), colour=(60, 60, 60))
        self.assertEqual(self._run(img), 0)
        s0 = self._stage(0)
        self.assertTrue(s0["ok"])
        self.assertEqual({c["verdict"] for c in s0["checks"]}, {"pass"})


if __name__ == "__main__":
    unittest.main()
