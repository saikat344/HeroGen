"""herogen.pipeline.imaging on synthetic images: threshold masks, splitting, padding, facing, top hull."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import imaging  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place, sheet  # noqa: E402


class ThresholdMaskTest(unittest.TestCase):
    def test_solid_figure_on_paper(self):
        m = figure_mask(200, "front")
        full = place(m, (300, 260), (40, 30))
        rng = np.random.default_rng(0)
        img = np.array(paint(full)).astype(np.int16) + rng.integers(-6, 7, size=(260, 300, 3))
        got = imaging.threshold_mask(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)))
        self.assertGreater(imaging.iou(got, full), 0.95)

    def test_pencil_outline_is_filled(self):
        img = Image.new("RGB", (200, 200), (245, 243, 238))
        d = ImageDraw.Draw(img)
        d.ellipse([50, 40, 150, 160], outline=(90, 90, 90), width=3)
        got = imaging.threshold_mask(img)
        truth = np.zeros((200, 200), dtype=bool)
        tm = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(tm).ellipse([50, 40, 150, 160], fill=255)
        truth = np.array(tm) > 127
        self.assertGreater(imaging.iou(got, truth), 0.95)

    def test_fill_holes(self):
        m = np.zeros((20, 20), dtype=bool)
        m[5:15, 5:15] = True
        m[8:12, 8:12] = False
        self.assertTrue(imaging.fill_holes(m)[8:12, 8:12].all())
        self.assertFalse(imaging.fill_holes(m)[0:4].any())


class SplitTest(unittest.TestCase):
    def test_four_figures(self):
        img, mask = sheet()
        runs = imaging.split_columns(mask, expected=4)
        self.assertEqual(len(runs), 4)
        for (_, end), (start, _) in zip(runs, runs[1:], strict=False):
            self.assertLess(end, start)

    def test_broken_figure_is_merged_and_speck_dropped(self):
        _, mask = sheet()
        runs = imaging.split_columns(mask, expected=4)
        # Cut a 3 px gap through the first figure and add a 1 px speck in a gap.
        x0, x1 = runs[0]
        cut = (x0 + x1) // 2
        mask[:, cut : cut + 3] = False
        speck_x = runs[1][1] + 5
        mask[50, speck_x] = True
        self.assertEqual(len(imaging.column_runs(mask)), 6)
        got = imaging.split_columns(mask, expected=4)
        self.assertEqual(len(got), 4)
        self.assertEqual(got[0], (x0, x1))

    def test_too_few_figures_reported_as_is(self):
        _, mask = sheet(order=("front", "right", "front"))
        self.assertEqual(len(imaging.split_columns(mask, expected=4)), 3)


class PadTest(unittest.TestCase):
    def test_paste_centred_keeps_scale_and_centres(self):
        m = figure_mask(100, "front")
        out = imaging.from_mask_image(
            imaging.paste_centred(imaging.to_mask_image(m), 240, 0, m.shape[1] / 2, m.shape[0] / 2)
        )
        b = imaging.bbox(out)
        self.assertEqual(imaging.box_size(b), (m.shape[1], m.shape[0]))
        self.assertLessEqual(abs((b[0] + b[2]) / 2 - 120), 1)
        self.assertLessEqual(abs((b[1] + b[3]) / 2 - 120), 1)


class FacingTest(unittest.TestCase):
    def test_toes_give_direction(self):
        self.assertGreater(imaging.facing_offset(figure_mask(200, "right")), 0.05)
        self.assertLess(imaging.facing_offset(figure_mask(200, "left")), -0.05)
        self.assertAlmostEqual(imaging.facing_offset(figure_mask(200, "front")), 0.0, delta=0.02)


class HullTest(unittest.TestCase):
    def test_box_figure(self):
        n = 100
        front = np.zeros((n, n), dtype=bool)
        front[20:80, 30:70] = True  # 40 wide in X
        side = np.zeros((n, n), dtype=bool)
        side[20:80, 40:60] = True  # 20 deep in Y
        top = imaging.top_from_hull(front, front.copy(), side, side.copy())
        self.assertEqual(imaging.bbox(top), (30, 40, 70, 60))
        self.assertEqual(int(top.sum()), 40 * 20)

    def test_front_is_at_the_bottom(self):
        n = 60
        front = np.zeros((n, n), dtype=bool)
        front[10:50, 25:35] = True
        right = np.zeros((n, n), dtype=bool)
        right[10:50, 25:35] = True
        right[46:50, 35:45] = True  # toes toward image right = the character's front
        left = np.fliplr(right)
        top = imaging.top_from_hull(front, front.copy(), right, left)
        b = imaging.bbox(top)
        self.assertEqual(b[3], 45)  # toes reach rows 35..44, below (in front of) the body rows 25..34


if __name__ == "__main__":
    unittest.main()
