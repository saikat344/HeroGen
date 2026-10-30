"""Stage 2's carve check: stage 3's carve and IoU gate, run on a sheet's views before the sheet is kept."""

from __future__ import annotations

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

from herogen.blender import stage3, synth, views  # noqa: E402
from herogen.pipeline import imaging, stages
from herogen.pipeline.client import ModelClient
from herogen.pipeline.runfolder import RunFolder  # noqa: E402

SIDES = ("front", "right", "back", "left")


def side_masks(**kw) -> dict[str, np.ndarray]:
    """The four side-view masks of the synthetic humanoid (one 3D figure, so the views agree)."""
    masks, _ = synth.render_masks(synth.humanoid(**kw), size=384, ppu=60.0)
    return {v: masks[v] for v in SIDES}


class ViewsFromMasksTest(unittest.TestCase):
    def test_same_view_set_as_a_run_folder(self):
        masks, _ = synth.render_masks(synth.humanoid(), size=256, ppu=40.0)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "masks").mkdir()
            for v, m in masks.items():
                views.save_mask(Path(d) / "masks" / f"{v}.png", m)
            a = views.load_views(Path(d), 5.5)
        b = views.views_from_masks(masks, 5.5)
        self.assertEqual(set(a.views), set(b.views))
        self.assertEqual((a.pps, a.height_px, a.missing), (b.pps, b.height_px, b.missing))
        for v in a.views:
            self.assertTrue(np.array_equal(a.views[v].mask, b.views[v].mask))
            self.assertEqual((a.views[v].cmid, a.views[v].vref), (b.views[v].cmid, b.views[v].vref))

    def test_views_not_given_are_missing(self):
        vs = views.views_from_masks(side_masks(), 5.5)
        self.assertEqual(vs.missing, ["top"])


class CarveCheckTest(unittest.TestCase):
    def test_views_of_one_figure_pass_stage_3s_gate(self):
        check, ious = stages.carve_check(side_masks(), 5.5)
        self.assertEqual(check["name"], stages.CARVE_GATE)
        self.assertEqual(check["threshold"], stage3.IOU_GATE)
        self.assertTrue(check["pass"], check)
        self.assertEqual(set(ious), set(SIDES))
        self.assertEqual(check["value"], min(ious.values()))

    def test_a_back_view_with_the_arms_elsewhere_fails(self):
        # The back view drawn with the arms hanging while the front holds them out: the carve keeps only what
        # both show, so the arms vanish from the body and the front no longer matches it.
        masks = side_masks()
        masks["back"] = side_masks(arm_angle_deg=8.0)["back"]
        check, ious = stages.carve_check(masks, 5.5)
        self.assertFalse(check["pass"], check)
        self.assertLess(ious["front"], stage3.IOU_GATE)
        self.assertIn("worst: front", check["note"])

    def test_a_drawn_top_that_disagrees_fails(self):
        masks = side_masks()
        top, _ = synth.render_masks(synth.humanoid(), size=384, ppu=60.0)
        squashed = np.zeros_like(top["top"])
        ys, xs = np.nonzero(top["top"])
        squashed[ys.min():ys.min() + (ys.max() - ys.min()) // 3, xs.min():xs.max()] = True  # a third of the depth
        self.assertTrue(stages.carve_check({**masks, "top": top["top"]}, 5.5)[0]["pass"])
        self.assertFalse(stages.carve_check({**masks, "top": squashed}, 5.5)[0]["pass"])

    def test_is_one_of_the_sheet_gates(self):
        self.assertIn(stages.CARVE_GATE, stages.SHEET_GATES)


class SavedTopFlowTest(unittest.TestCase):
    """Synthetic replay of a passing side carve invalidated by the saved fallback top."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = RunFolder(Path(self.tmp.name), "sheet")
        self.image = Image.new("RGB", (96, 96), "white")
        self.image.save(self.run.file("01-cover.png"))
        self.client = ModelClient(Path(self.tmp.name) / "cache", mode="replay")
        self.masks = {}
        for name in SIDES:
            m = np.zeros((96, 96), dtype=bool)
            width = 40 if name in ("front", "back") else 18
            m[10:80, 48 - width // 2:48 + width // 2] = True
            self.masks[name] = m
        self.top = imaging.top_from_hull(*(self.masks[n] for n in ("front", "back", "right", "left")))
        self.checked = []

    def _flow(self, score, drawn=None, *, extra_checks=(), expected_failure=False):
        def square(*_):
            return 96, {n: {"mask": m.copy(), "image": self.image.copy(), "mask_method": "sam-3.1"}
                        for n, m in self.masks.items()}

        def measure(vs, height):
            self.checked.append({n: v.mask.copy() for n, v in vs.views.items()})
            return score(vs, len(self.checked))

        split = {"views": {"synthetic": True}, "runs": [(0, 1)] * 4,
                 "sam_coverage": {n: 1.0 for n in SIDES}, "split_source": "sam-3.1"}
        if drawn is None:
            drawn = {"raw": self.image, "mask": None}
        side_view_checks = stages.C.side_view_checks
        with mock.patch.object(self.client, "edit_image", return_value=self.image) as edits, \
                mock.patch.object(stages, "sam_figure_mask", return_value=(self.masks["front"], "sam-3.1", 4)), \
                mock.patch.object(stages, "split_sheet", return_value=split), \
                mock.patch.object(stages, "assign_views", return_value=({}, {})), \
                mock.patch.object(stages, "square_views", side_effect=square), \
                mock.patch.object(stages, "orient_sides", return_value=([], [], {})), \
                mock.patch.object(stages.C, "side_view_checks", side_effect=lambda *a, **kw:
                                  side_view_checks(*a, **kw) + list(extra_checks)), \
                mock.patch.object(stages, "model_top_view", return_value=drawn) as tops, \
                mock.patch("herogen.pipeline.parts.write_part_masks", return_value={}), \
                mock.patch.object(stage3, "carve_view_ious", side_effect=measure):
            if expected_failure:
                with self.assertRaises(stages.GateFailed) as caught:
                    stages.stage2_turnaround(self.run, self.client)
                result = caught.exception.record
            else:
                result = stages.stage2_turnaround(self.run, self.client)
        return result, edits.call_count, tops.call_count

    def _assert_saved_check(self, result):
        saved = views.load_views(self.run.path, 5.5)
        final = next(m for m in self.checked if "top" in m and all(
            np.array_equal(m[n], v.mask) for n, v in saved.views.items()))
        self.assertEqual(set(final), set(stages.ALL_VIEWS))
        c = next(c for c in result["checks"] if c["name"] == stages.CARVE_GATE)
        self.assertEqual(c["value"], min(result["details"]["carve_ious_with_saved_top"][n] for n in SIDES))
        attempt_checks = result["details"]["attempts"][result["details"]["kept_attempt"]]["checks"]
        self.assertEqual([(c["name"], c["value"], c["pass"]) for c in attempt_checks],
                         [(c["name"], c["value"], c["pass"]) for c in result["checks"]])
        self.assertTrue(next(c for c in attempt_checks if c["name"] == stages.CARVE_GATE)["gate"])
        return c

    def test_final_carve_only_failure_retries_then_passes_stage_with_warning(self):
        def score(vs, _):
            return {n: (0.8961 if "top" in vs.views else 0.9076) for n in vs.views}

        result, edits, tops = self._flow(score)
        self.assertTrue(result["ok"])
        self.assertEqual((edits, tops), (4, 4))
        self.assertEqual(result["details"]["kept_attempt"], 0)
        self.assertEqual(result["details"]["top_method"], "visual_hull")
        check = self._assert_saved_check(result)
        self.assertEqual(check["value"], 0.8961)
        self.assertEqual((check["pass"], check["gate"]), (False, False))
        self.assertIn("does not fail stage 2", check["note"])
        self.assertIn("carve fallback would be refused", result["notes"])
        self.assertTrue(all(a["rank"]["failed_gates"] == 1 for a in result["details"]["attempts"]))
        self.assertTrue(self.run.exists("views/meta.json"))

    def test_preliminary_carve_failure_still_checks_saved_top_and_retries(self):
        result, edits, tops = self._flow(lambda vs, _: {n: 0.8 for n in vs.views})
        self.assertTrue(result["ok"])
        self.assertEqual((edits, tops), (4, 4))
        self.assertFalse(self._assert_saved_check(result)["gate"])

    def test_carve_and_another_gate_failure_still_raise(self):
        extra = stages.C.check("height_agreement_4_views", 0.8, stages.C.HEIGHT_AGREE_MIN)
        result, edits, tops = self._flow(lambda vs, _: {n: 0.8 for n in vs.views},
                                        extra_checks=[extra], expected_failure=True)
        self.assertFalse(result["ok"])
        self.assertEqual((edits, tops), (4, 0))
        failed = {c["name"] for c in result["checks"] if c["gate"] and not c["pass"]}
        self.assertEqual(failed, {stages.CARVE_GATE, "height_agreement_4_views"})
        self.assertNotIn("does not fail stage 2", result["notes"])

    def test_carve_exception_is_propagated(self):
        def score(*_):
            raise RuntimeError("carve measurement failed")

        with self.assertRaisesRegex(RuntimeError, "carve measurement failed"):
            self._flow(score)

    def test_retry_accepts_later_sheet_with_passing_final_carve(self):
        def score(vs, call):
            return {n: (0.8961 if call == 2 else 0.9076) for n in vs.views}

        result, edits, tops = self._flow(score)
        self.assertTrue(result["ok"])
        self.assertEqual((edits, tops), (2, 2))
        self.assertEqual(result["details"]["kept_attempt"], 1)
        self.assertTrue(self._assert_saved_check(result)["pass"])

    def test_derived_tops_own_iou_is_not_gated(self):
        result, edits, _ = self._flow(lambda vs, _: {n: 0.1 if n == "top" else 0.9076 for n in vs.views})
        self.assertTrue(result["ok"])
        self.assertEqual(edits, 1)
        self.assertEqual(self._assert_saved_check(result)["value"], 0.9076)

    def _drawn(self):
        # Same extents as the hull, with a corner missing: dimensionally valid, geometrically distinct.
        mask = self.top.copy()
        y, x = np.nonzero(mask)
        mask[y.min(), x.min()] = False
        return {"raw": self.image, "image": self.image, "mask": mask, "mask_method": "sam-3.1"}

    def test_passing_drawn_top_is_saved_and_checked(self):
        drawn = self._drawn()
        result, edits, _ = self._flow(lambda vs, _: {n: 0.95 for n in vs.views}, drawn)
        self.assertTrue(result["ok"])
        self.assertEqual(edits, 1)
        self.assertEqual(result["details"]["top_method"], stages.IMAGE_MODEL)
        self.assertTrue(np.array_equal(views.load_mask(self.run.file("masks/top.png")), drawn["mask"]))
        self._assert_saved_check(result)

    def test_dimensionally_valid_drawn_top_fails_carve_and_fallback_is_rechecked(self):
        def score(vs, call):
            return {n: 0.8 if n == "top" and call == 2 else 0.95 for n in vs.views}

        result, edits, _ = self._flow(score, self._drawn())
        self.assertTrue(result["ok"])
        self.assertEqual(edits, 1)
        self.assertEqual(len(self.checked), 3)
        self.assertEqual(result["details"]["top_method"], "visual_hull")
        self.assertTrue(np.array_equal(views.load_mask(self.run.file("masks/top.png")), self.top))
        self._assert_saved_check(result)


if __name__ == "__main__":
    unittest.main()
