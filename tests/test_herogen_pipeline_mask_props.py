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

    def test_enclosed_arm_opening_and_open_white_gap_stay_empty(self):
        self.rgb[20:36, 50:65] = (40, 40, 40)
        self.rgb[22:34, 52:61] = 255  # 12 by 9 pixels of paper inside the bent arm
        for accurate_sam in (False, True):
            with self.subTest(accurate_sam=accurate_sam):
                if accurate_sam:
                    self.sam |= self.rgb.min(axis=2) < 215
                got = self.complete()
                self.assertFalse(got[22:34, 52:61].any())
                self.assertFalse(got[40:50, 52:61].any())
                self.assertTrue(got[self.sam].all())
                np.testing.assert_array_equal(got & ~self.sam, (self.rgb.min(axis=2) < 215) & ~self.sam)

    def test_white_interior_already_in_sam_is_preserved(self):
        self.rgb[25:35, 35:45] = 255
        np.testing.assert_array_equal(self.complete(), self.sam)

    def test_gray_background_inside_white_margins_is_refused(self):
        self.rgb[5:75, 10:70] = 200
        self.rgb[self.sam] = (70, 90, 130)
        with self.assertRaisesRegex(ValueError, 'size cap'):
            self.complete()

    def test_added_size_cap_boundary_and_total_across_components(self):
        self.rgb[5:15, 30:50] = 60   # 200-pixel head extension
        self.rgb[20:50, 50:60] = 60  # 300-pixel backpack: exactly half the 1000-pixel seed
        self.assertEqual(int((self.complete() & ~self.sam).sum()), 500)
        self.rgb[50, 50] = 60
        with self.assertRaisesRegex(ValueError, '501/1000.*50%'):
            self.complete()

    def test_near_white_cutoff_checks_every_channel(self):
        self.rgb[5:15, 30:50] = (215, 255, 255)
        np.testing.assert_array_equal(self.complete(), self.sam)
        self.rgb[5:15, 30:50, 0] = 214
        self.assertEqual(int((self.complete() & ~self.sam).sum()), 200)

    def test_sam_at_image_border_without_additions_is_preserved(self):
        self.sam[:, 30:50] = True
        np.testing.assert_array_equal(self.complete(), self.sam)

    def test_added_pixels_touching_each_image_border_are_refused(self):
        for edge in ('top', 'bottom', 'left', 'right'):
            with self.subTest(edge=edge):
                rgb = self.rgb.copy()
                if edge == 'top':
                    rgb[:15, 30] = 60
                elif edge == 'bottom':
                    rgb[65:, 30] = 60
                elif edge == 'left':
                    rgb[30, :30] = 60
                else:
                    rgb[30, 50:] = 60
                with self.assertRaisesRegex(ValueError, 'image border'):
                    imaging.complete_attached_foreground(Image.fromarray(rgb), self.sam)

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
    def assert_refused(self, out, sam, reason):
        self.assertEqual(out['runs'], imaging.split_columns(sam, expected=4))
        self.assertEqual(len(out['views']), 4)
        for view in out['views'].values():
            x0, y0, x1, y1 = view['sheet_box']
            np.testing.assert_array_equal(view['mask'], sam[y0:y1, x0:x1])
            self.assertEqual(view['mask_completion']['method'], 'none')
            self.assertIn(reason, view['mask_completion']['reason'])
            self.assertEqual(view['mask_completion']['added_pixels'], 0)

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
        self.assert_refused(out, sam, 'original panels')

    def test_bridge_cannot_merge_panels_while_splitting_another(self):
        sam = np.zeros((80, 400), dtype=bool)
        for x in (20, 50, 120, 220, 320):
            sam[15:65, x:x+20] = True
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        rgb[sam] = 60
        self.assertEqual(imaging.split_columns(sam, expected=4), [(20, 70), (120, 140), (220, 240), (320, 340)])
        for y in (35, 65):  # overlap and diagonal contact both connect the original panels
            with self.subTest(y=y):
                connected = rgb.copy()
                connected[y, 140:220] = 60
                out = stages.split_sheet(Image.fromarray(connected), sam, sam, 'sam-3.1:character')
                self.assert_refused(out, sam, 'original panels')
                self.assertEqual(set(out['sam_coverage'].values()), {1.0})

    def test_disconnected_fragments_keep_original_panel_membership(self):
        sam = np.zeros((80, 400), dtype=bool)
        for x in (20, 50, 120, 220, 320):
            sam[15:65, x:x+20] = True
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        rgb[sam] = 60
        rgb[35, 119] = 60  # widen panel1 without splitting panel0 into its two components
        out = stages.split_sheet(Image.fromarray(rgb), sam, sam, 'sam-3.1:character')
        self.assertEqual(out['runs'], [(20, 70), (119, 140), (220, 240), (320, 340)])
        self.assertTrue(out['views']['panel0']['mask'][:, :20].any())
        self.assertTrue(out['views']['panel0']['mask'][:, 30:].any())
        self.assertEqual({v['mask_completion']['method'] for v in out['views'].values()}, {'attached_foreground'})
        self.assertEqual(set(out['sam_coverage'].values()), {1.0})

    def test_disjoint_components_with_overlapping_expanded_crops_are_refused(self):
        image, sam, _ = self.sheet()
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        rgb[sam] = 60
        rgb[20, 50:90] = 60  # panel0 reaches into the gutter
        rgb[60, 80:110] = 60  # panel1 reaches back at a different height
        out = stages.split_sheet(Image.fromarray(rgb), sam, sam, 'sam-3.1:character')
        self.assert_refused(out, sam, 'overlapping crops')

    def test_component_crossing_original_columns_without_touching_other_sam_is_refused(self):
        image, sam, _ = self.sheet()
        rgb = np.array(image)
        rgb[10, 49:121] = 60
        rgb[10:16, 49] = 60  # meets panel0 but passes above panel1
        out = stages.split_sheet(Image.fromarray(rgb), sam, sam, 'sam-3.1:character')
        self.assert_refused(out, sam, 'original panels')

    def test_gray_panel_with_white_margins_refuses_the_entire_sheet(self):
        image, sam, _ = self.sheet()
        for gray_panels in ((0, 1, 2, 3), (2,)):
            with self.subTest(gray_panels=gray_panels):
                rgb = np.full((80, 320, 3), 255, dtype=np.uint8)
                for i in gray_panels:
                    rgb[5:75, 10+80*i:70+80*i] = 200
                rgb[sam] = 60
                image = Image.fromarray(rgb)
                reference = imaging.threshold_mask(image)
                out = stages.split_sheet(image, sam, reference, 'sam-3.1:character')
                self.assert_refused(out, sam, 'size cap')
                for i in gray_panels:
                    self.assertEqual(out['sam_coverage'][f'panel{i}'], 0.7143)

    def test_large_white_margins_cannot_hide_gray_panel_additions(self):
        sam = np.zeros((160, 320), dtype=bool)
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        for x in (30, 110, 190, 270):
            rgb[45:115, x-20:x+40] = 200
            sam[55:105, x:x+20] = True
        rgb[sam] = 60
        image = Image.fromarray(rgb)
        out = stages.split_sheet(image, sam, imaging.threshold_mask(image), 'sam-3.1:character')
        self.assert_refused(out, sam, '3200/1000')
        self.assertEqual(set(out['sam_coverage'].values()), {0.7143})

    def test_missing_head_arm_and_pole_are_all_recovered(self):
        image, sam, full = self.sheet()
        rgb = np.array(image)
        for x in (30, 110, 190, 270):
            rgb[7:65, x:x+20] = 60
            full[7:65, x:x+20] = True
        image = Image.fromarray(rgb)
        out = stages.split_sheet(image, sam, imaging.threshold_mask(image), 'sam-3.1:character')
        for view in out['views'].values():
            x0, y0, x1, y1 = view['sheet_box']
            np.testing.assert_array_equal(view['mask'], full[y0:y1, x0:x1])
            self.assertEqual(view['mask_completion']['added_pixels'], 274)
        self.assertEqual(set(out['sam_coverage'].values()), {0.8251})

    def test_gray_slab_at_midpoint_column_edges_refuses_entire_sheet(self):
        image, sam, _ = self.sheet()
        # Panel1 owns [80, 160); both outermost columns are forbidden to additions.
        for start, end in ((80, 110), (130, 160)):
            with self.subTest(start=start, end=end):
                rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
                rgb[sam] = 60
                rgb[30:34, start:end] = 200  # only 120 added pixels: size cap alone accepts this
                out = stages.split_sheet(Image.fromarray(rgb), sam, sam, 'sam-3.1:character')
                self.assert_refused(out, sam, 'panel column edge')
                self.assertEqual(set(out['sam_coverage'].values()), {1.0})
                # Stopping one column short of the same edge must permit completion.
                rgb[30:34, start if start == 80 else end-1] = 255
                out = stages.split_sheet(Image.fromarray(rgb), sam, sam, 'sam-3.1:character')
                self.assertEqual(out['views']['panel1']['mask_completion']['added_pixels'], 116)
                self.assertEqual(out['views']['panel1']['mask_completion']['method'], 'attached_foreground')

    def test_sheet_preserves_enclosed_arm_openings(self):
        _, sam, _ = self.sheet()
        for x in (30, 110, 190, 270):
            sam[20:36, x+20:x+35] = True
            sam[22:34, x+22:x+31] = False
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        rgb[sam] = 60
        out = stages.split_sheet(Image.fromarray(rgb), sam, imaging.threshold_mask(Image.fromarray(rgb)), 'sam-3.1:character')
        for view in out['views'].values():
            x0, y0, x1, y1 = view['sheet_box']
            np.testing.assert_array_equal(view['mask'], sam[y0:y1, x0:x1])
            self.assertEqual(view['mask_completion']['added_pixels'], 0)

    def test_top_preserves_enclosed_arm_opening(self):
        sam = np.zeros((80, 80), dtype=bool)
        sam[15:65, 30:50] = True
        sam[20:36, 50:65] = True
        sam[22:34, 52:61] = False
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        rgb[sam] = 60
        image = Image.fromarray(rgb)
        client = mock.Mock()
        client.edit_image.return_value = image
        with mock.patch.object(stages, 'sam_figure_mask', return_value=(sam, 'sam-3.1:character', 1)):
            top = stages.model_top_view(client, image, {'front': {'mask': sam}}, 100, (255, 255, 255))
        self.assertEqual(top['scale'], 1)
        np.testing.assert_array_equal(top['mask'][25:75, 32:67], sam[15:65, 30:65])
        self.assertEqual(top['mask_completion']['added_pixels'], 0)

    def test_top_gray_background_inside_white_margins_is_refused(self):
        sam = np.zeros((80, 80), dtype=bool)
        sam[15:65, 30:50] = True
        rgb = np.full((*sam.shape, 3), 255, dtype=np.uint8)
        rgb[5:75, 10:70] = 200
        rgb[sam] = 60
        image = Image.fromarray(rgb)
        client = mock.Mock()
        client.edit_image.return_value = image
        with mock.patch.object(stages, 'sam_figure_mask', return_value=(sam, 'sam-3.1:character', 1)):
            top = stages.model_top_view(client, image, {'front': {'mask': sam}}, 100, (255, 255, 255))
        self.assertEqual(top['mask_completion']['method'], 'none')
        self.assertIn('size cap', top['mask_completion']['reason'])
        self.assertEqual(top['mask_completion']['added_pixels'], 0)
        self.assertEqual(int(top['mask'].sum()), int(sam.sum()))

    def test_top_border_addition_refuses_only_the_top_image(self):
        image, sam, _ = self.sheet()
        sam = sam[:, :80]
        rgb = np.array(image)[:, :80]
        rgb[:15, 30] = 60  # below the size cap, but attached to the top image border
        client = mock.Mock()
        client.edit_image.return_value = Image.fromarray(rgb)
        front = sam.copy()
        with mock.patch.object(stages, 'sam_figure_mask', return_value=(sam, 'sam-3.1:character', 1)):
            top = stages.model_top_view(client, image, {'front': {'mask': front}}, 100, (255, 255, 255))
        self.assertEqual(top['mask_completion']['method'], 'none')
        self.assertIn('image border', top['mask_completion']['reason'])
        self.assertEqual(top['mask_completion']['added_pixels'], 0)
        self.assertEqual(int(top['mask'].sum()), int(sam.sum()))
        np.testing.assert_array_equal(front, sam)

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
