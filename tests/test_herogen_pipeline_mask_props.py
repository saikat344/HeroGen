"""Attached foreground completion on synthetic white-background turnaround views."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import imaging, stages  # noqa: E402


class AttachedForegroundTest(unittest.TestCase):
    def setUp(self):
        self.sam = np.zeros((80, 80), dtype=bool)
        self.sam[15:65, 30:50] = True
        self.rgb = np.full((80, 80, 3), 255, dtype=np.uint8)
        self.rgb[self.sam] = (70, 90, 130)

    def complete(self):
        return imaging.complete_attached_foreground(Image.fromarray(self.rgb), self.sam)

    def test_holding_one_pixel_pole_and_missing_arm(self):
        truth = self.sam.copy()
        truth[30:34, 18:30] = True  # arm omitted by segmentation
        truth[6:72, 17] = True  # pole meets the hand diagonally; no opening may erase it
        self.rgb[truth] = (90, 65, 40)
        np.testing.assert_array_equal(self.complete(), truth)
        self.assertFalse(self.sam[6, 17])  # the input is unchanged

    def test_attached_backpack_outside_original_crop(self):
        truth = self.sam.copy()
        truth[20:45, 49:63] = True
        self.rgb[truth] = (85, 65, 40)
        np.testing.assert_array_equal(self.complete(), truth)

    def test_detached_blob_and_speck_stay_out(self):
        self.rgb[20:35, 5:15] = (40, 40, 40)
        self.rgb[14, 30] = (40, 40, 40)  # touching the body is retained
        self.rgb[40, 51] = (40, 40, 40)  # detached 1px speck near the seed is removed
        expected = self.sam.copy()
        expected[14, 30] = True
        np.testing.assert_array_equal(self.complete(), expected)

    def test_touching_light_and_detached_dark_drop_shadows_stay_out(self):
        self.rgb[65:70, 15:65] = (225, 225, 225)  # touches feet, but is border-reachable near-white
        self.rgb[73:76, 15:65] = (160, 160, 160)  # dark, but detached from the figure
        np.testing.assert_array_equal(self.complete(), self.sam)

    def test_enclosed_white_is_filled_but_open_white_gap_is_not(self):
        self.rgb[20:32, 50:65] = (40, 40, 40)
        self.rgb[22:30, 52:63] = 255
        got = self.complete()
        self.assertTrue(got[22:30, 52:63].all())
        self.assertFalse(got[35:45, 52:63].any())

    def test_empty_sam_never_becomes_a_threshold_fallback(self):
        self.sam[:] = False
        self.assertFalse(self.complete().any())

    def test_invalid_masks_raise(self):
        for mask in (self.sam.astype(np.uint8), self.sam[:20]):
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaisesRegex(ValueError, 'boolean SAM mask'):
                imaging.complete_attached_foreground(Image.fromarray(self.rgb), mask)

    def test_dark_background_raises_instead_of_filling_canvas(self):
        self.rgb[~self.sam] = (180, 180, 180)
        with self.assertRaisesRegex(ValueError, 'near-white image background'):
            self.complete()

    def test_diagonal_connectivity_and_component_area(self):
        m = np.eye(20, dtype=bool)
        labels, areas = imaging._components8(m)
        self.assertEqual(len(np.unique(labels[m])), 1)
        self.assertEqual(areas[labels[0, 0]], 20)
        self.assertEqual(areas[0], 380)

    def test_completion_is_idempotent(self):
        self.rgb[20:45, 49:63] = (85, 65, 40)
        got = self.complete()
        np.testing.assert_array_equal(imaging.complete_attached_foreground(Image.fromarray(self.rgb), got), got)

    def test_multiple_components_merge_through_a_later_bridge(self):
        m = np.zeros((10, 12), dtype=bool)
        m[1:5, 1:4] = True
        m[1:5, 7:10] = True
        m[5, 4:7] = True
        m[7:9, 1:3] = True
        labels, areas = imaging._components8(m)
        self.assertEqual(labels[1, 1], labels[1, 7])
        self.assertEqual(areas[labels[1, 1]], 27)
        self.assertNotEqual(labels[1, 1], labels[7, 1])


class Stage2CompletionTest(unittest.TestCase):
    def sheet(self):
        sam = np.zeros((80, 320), dtype=bool)
        full = sam.copy()
        for x in (30, 110, 190, 270):
            sam[15:65, x:x+20] = True
            full[15:65, x:x+20] = True
            full[30:34, x-12:x] = True
            full[6:72, x-13] = True
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        rgb[full] = (80, 65, 40)
        return Image.fromarray(rgb), sam, full

    def test_every_panel_completed_before_cropping_and_coverage_stays_raw(self):
        image, sam, full = self.sheet()
        split = stages.split_sheet(image, sam, full, 'sam-3.1:character')
        self.assertEqual(len(split['views']), 4)
        for name, view in split['views'].items():
            x0, y0, x1, y1 = view['sheet_box']
            np.testing.assert_array_equal(view['mask'], full[y0:y1, x0:x1])
            self.assertTrue(view['mask'][:, 0].any())
            self.assertEqual(view['image'].size, (33, 66))
            self.assertGreater(view['mask_completion']['added_pixels'], 0)
            self.assertEqual(split['sam_coverage'][name], round(1000 / int(full[:, x0:x1].sum()), 4))
            self.assertLess(split['sam_coverage'][name], 1)

    def test_missing_figure_still_fails_split(self):
        image, sam, full = self.sheet()
        sam[:, 260:] = False
        self.assertEqual(stages.split_sheet(image, sam, full, 'sam-3.1:character')['views'], {})

    def test_completion_refusal_keeps_sam_and_records_reason(self):
        image, sam, full = self.sheet()
        with mock.patch.object(imaging, 'complete_attached_foreground', side_effect=ValueError('invalid foreground')):
            out = stages.split_sheet(image, sam, full, 'sam-3.1:character')
        self.assertEqual(len(out['views']), 4)
        for view in out['views'].values():
            self.assertEqual(view['mask_completion'], {'method': 'none', 'reason': 'invalid foreground', 'added_pixels': 0})

    def test_completion_that_merges_figures_keeps_sam(self):
        image, sam, full = self.sheet()
        with mock.patch.object(imaging, 'complete_attached_foreground', return_value=np.ones_like(sam)):
            out = stages.split_sheet(image, sam, full, 'sam-3.1:character')
        self.assertEqual(len(out['views']), 4)
        self.assertEqual({v['mask_completion']['reason'] for v in out['views'].values()}, {'completion merged figures'})

    def test_top_completion_precedes_crop_and_scale(self):
        image, sam, full = self.sheet()
        image = image.crop((0, 0, 80, 80))
        sam = sam[:, :80]
        full = full[:, :80]
        front = {'front': {'mask': full}}
        client = mock.Mock()
        client.edit_image.return_value = image
        with mock.patch.object(stages, 'sam_figure_mask', return_value=(sam, 'sam-3.1:character', 1)):
            top = stages.model_top_view(client, image, front, 100, (255, 255, 255))
        self.assertEqual(int(top['mask'].sum()), int(full.sum()))
        self.assertEqual(imaging.box_size(imaging.bbox(top['mask'])), (33, 66))
        self.assertEqual(top['scale'], 1)
        self.assertGreater(top['mask_completion']['added_pixels'], 0)

    def test_segmentation_failure_propagates(self):
        client = mock.Mock()
        client.edit_image.return_value = Image.new('RGB', (80, 80), 'white')
        with mock.patch.object(stages, 'sam_figure_mask', side_effect=RuntimeError('segmentation failed')):
            with self.assertRaisesRegex(RuntimeError, 'segmentation failed'):
                stages.model_top_view(client, client.edit_image.return_value, {}, 100, (255, 255, 255))


if __name__ == '__main__':
    unittest.main()
