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
from unittest import mock

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import client as client_mod  # noqa: E402
from herogen.pipeline import inputs, stages  # noqa: E402
from herogen.pipeline.__main__ import run_pipeline  # noqa: E402
from herogen.pipeline.client import ModelClient, SamInstance  # noqa: E402
from test_herogen_pipeline_flow import DUMMY_KEY, FakeAPI  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place  # noqa: E402


def inst(mask: np.ndarray) -> SamInstance:
    ys, xs = np.nonzero(mask)
    return SamInstance(mask=mask, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                       confidence=0.9)


def box_mask(shape, x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def two_figures(second_h: int, size=(900, 600)) -> np.ndarray:
    """A 400 px figure and a second one ``second_h`` px tall beside it, on one canvas."""
    m = place(figure_mask(400, "front"), size, (60, 100))
    small = figure_mask(second_h, "front")
    m[100 : 100 + small.shape[0], 520 : 520 + small.shape[1]] |= small
    return m


class PromptAPI(FakeAPI):
    """The flow tests' fake API, with SAM answers per prompt: ``parts`` maps a prompt to a function of the
    segmented image's figure mask that returns that prompt's masks. Other prompts get the fake's figures (one
    per column of the background-threshold mask)."""

    def __init__(self, parts=None, **kw) -> None:
        super().__init__(**kw)
        self.parts = parts or {}
        self.sam_texts: list[str] = []

    def post(self, c, endpoint: str, body: dict) -> dict:
        if endpoint == "responses":
            self.sam_texts.append(body["input"][0]["content"][0]["text"])
        return super().post(c, endpoint, body)

    def decode(self, text: str, size) -> list[SamInstance]:
        fn = self.parts.get(self.sam_texts[-1])
        if fn is None:
            return super().decode(text, size)
        from herogen.pipeline import imaging  # noqa: PLC0415
        return [inst(m) for m in fn(imaging.threshold_mask(self.last_sam_image)) if m.any()]


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

    def _live(self, img, api: PromptAPI, stage_list=(0, 1), run_id="r") -> int:
        path = Path(self.tmp.name) / f"{run_id}.png"
        img.save(path)
        err, out = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", lambda c, e, b, **_: api.post(c, e, b)), \
                mock.patch.object(client_mod, "decode_sam_output", api.decode), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = run_pipeline(path, run_id, list(stage_list), mode="live", runs=str(self.root))
        self.err, self.out, self.api = err.getvalue(), out.getvalue(), api
        return code

    def _verdicts(self, rec) -> dict:
        return {c["name"]: c["verdict"] for c in rec["checks"] if "verdict" in c}


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


class CharacterCountTest(unittest.TestCase):
    def test_counts(self):
        one = [inst(box_mask((100, 100), 10, 10, 50, 90))]
        checks, det = inputs.character_checks(one, "none")
        self.assertEqual([(c["value"], c["verdict"]) for c in checks], [(1, "pass"), (0, "pass")])
        self.assertEqual(det["figure_area_fracs"], [1.0])
        inset = one + [inst(box_mask((100, 100), 60, 10, 70, 41))]  # 310 px against 3200: 0.097
        checks, det = inputs.character_checks(inset, "none")
        self.assertEqual([c["verdict"] for c in checks], ["pass", "warn"])
        self.assertIn("1 smaller figure (each under 25%", checks[1]["message"])
        second = one + [inst(box_mask((100, 100), 60, 10, 80, 50))]  # 800 px: 0.25, another character
        checks, _ = inputs.character_checks(second, "none")
        self.assertEqual([(c["value"], c["verdict"]) for c in checks], [(2, "reject"), (0, "pass")])
        self.assertIn("the sketch shows 2 characters", checks[0]["message"])
        self.assertIn("the next is 25% of it", checks[0]["message"])
        empty = SamInstance(mask=np.zeros((10, 10), bool), box=(0, 0, 0, 0))
        checks, det = inputs.character_checks([empty], "no character found")
        self.assertEqual((checks[0]["value"], checks[0]["message"], det["figure_area_fracs"]),
                         (0, "no character found", []))


class Stage1CharactersTest(_SketchRun):
    def test_two_characters_are_rejected_before_the_cover(self):
        img = paint(two_figures(400), colour=(60, 60, 60))
        self.assertEqual(self._live(img, PromptAPI()), 3)
        s1 = self._stage(1)
        self.assertEqual((s1["ok"], s1["status"]), (False, "input_rejected"))
        self.assertEqual(self._verdicts(s1)["sketch_characters_found"], "reject")
        self.assertIn("the sketch shows 2 characters", s1["error"])
        self.assertIn("input rejected: the sketch shows 2 characters", self.err)
        self.assertEqual(s1["details"]["input"]["figure_area_fracs"], [1.0, 1.0])
        self.assertNotIn("images/edits", self.api.calls)
        self.assertFalse((self.root / "r" / "01-cover.png").exists())

    def test_a_small_extra_figure_warns_and_only_the_main_figure_is_built(self):
        img = paint(two_figures(120), colour=(60, 60, 60))
        self.assertEqual(self._live(img, PromptAPI()), 0)
        s1 = self._stage(1)
        self.assertTrue(s1["ok"])
        v = self._verdicts(s1)
        self.assertEqual((v["sketch_characters_found"], v["sketch_small_extra_figures"]), ("pass", "warn"))
        self.assertIn("stage 1: warning: besides the main character the sketch has 1 smaller figure", self.err)
        self.assertIn("Warning: besides the main character", s1["notes"])
        with Image.open(self.root / "r" / "01-masks" / "sketch.png") as m:
            mask = np.array(m.convert("L")) > 127
        self.assertFalse(mask[:, 500:].any())  # the small figure is not part of the sketch's mask
        self.assertTrue(mask[:, :500].any())

    def test_no_character_is_a_recorded_rejection(self):
        img = paint(place(figure_mask(400, "front"), (600, 560), (100, 80)), colour=(60, 60, 60))
        api = PromptAPI({t: (lambda fig: []) for t in stages.SAM_TEXTS})
        self.assertEqual(self._live(img, api), 3)
        s1 = self._stage(1)
        self.assertIn("no character found in the sketch", s1["error"])
        c = next(c for c in s1["checks"] if c["name"] == "sketch_characters_found")
        self.assertEqual((c["value"], c["verdict"]), (0, "reject"))
        self.assertEqual(api.sam_texts, list(stages.SAM_TEXTS))


def bottom(fig: np.ndarray, frac: float) -> np.ndarray:
    """The lowest ``frac`` of a figure mask's height (its feet, or its legs for a larger share)."""
    ys = np.flatnonzero(fig.any(axis=1))
    out = fig.copy()
    out[: ys[-1] + 1 - int(frac * (ys[-1] - ys[0] + 1))] = False
    return out


class FeetTest(unittest.TestCase):
    def test_parts_on_the_figure(self):
        fig = box_mask((100, 100), 10, 10, 50, 90)
        parts = [inst(box_mask((100, 100), 20, 80, 30, 95)), inst(box_mask((100, 100), 60, 80, 70, 90)),
                 inst(box_mask((100, 100), 45, 80, 55, 90))]  # 2/3 on, off, half on
        self.assertEqual(inputs.parts_on(parts, fig), 2)

    def test_verdicts(self):
        self.assertEqual(inputs.feet_check(2, None)["verdict"], "pass")
        cropped = inputs.feet_check(0, 2)
        self.assertEqual((cropped["verdict"], cropped["gate"]), ("reject", True))
        self.assertIn("looks cropped", cropped["message"])
        hidden = inputs.feet_check(0, 0)
        self.assertEqual((hidden["verdict"], hidden["gate"]), ("warn", False))
        self.assertIn("long robe", hidden["message"])


class Stage1FeetTest(_SketchRun):
    def setUp(self):
        super().setUp()
        self.img = paint(place(figure_mask(400, "front"), (600, 560), (100, 80)), colour=(60, 60, 60))

    def test_feet_found(self):
        api = PromptAPI({"foot": lambda fig: [bottom(fig, 0.04)]})
        self.assertEqual(self._live(self.img, api), 0)
        s1 = self._stage(1)
        self.assertEqual(self._verdicts(s1)["sketch_feet_on_figure"], "pass")
        self.assertEqual((s1["details"]["input"]["feet_on_figure"], s1["details"]["input"]["legs_on_figure"]),
                         (1, None))
        self.assertNotIn("leg", api.sam_texts)  # legs are asked only when no feet are found

    def test_legs_without_feet_are_a_cropped_figure(self):
        api = PromptAPI({"foot": lambda fig: [], "leg": lambda fig: [bottom(fig, 0.4)]})
        self.assertEqual(self._live(self.img, api), 3)
        s1 = self._stage(1)
        self.assertEqual(s1["status"], "input_rejected")
        self.assertEqual(self._verdicts(s1)["sketch_feet_on_figure"], "reject")
        self.assertIn("the figure looks cropped", s1["error"])
        self.assertEqual(s1["details"]["input"]["sam_prompts"], ["foot", "leg"])
        self.assertEqual(api.sam_texts[-2:], ["foot", "leg"])
        self.assertNotIn("images/edits", api.calls)

    def test_no_feet_or_legs_warn(self):
        api = PromptAPI({"foot": lambda fig: [], "leg": lambda fig: []})
        self.assertEqual(self._live(self.img, api), 0)
        s1 = self._stage(1)
        self.assertTrue(s1["ok"])
        self.assertEqual(self._verdicts(s1)["sketch_feet_on_figure"], "warn")
        self.assertIn("warning: no feet or legs found on the figure", self.err)

    def test_parts_are_not_asked_for_a_rejected_sketch(self):
        api = PromptAPI()
        self.assertEqual(self._live(paint(two_figures(400), colour=(60, 60, 60)), api), 3)
        self.assertNotIn("foot", api.sam_texts)


if __name__ == "__main__":
    unittest.main()
