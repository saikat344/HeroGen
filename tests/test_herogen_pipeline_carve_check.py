"""Stage 2's carve check: stage 3's carve and IoU gate, run on a sheet's views before the sheet is kept."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.blender import stage3, synth, views  # noqa: E402
from herogen.pipeline import stages  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
