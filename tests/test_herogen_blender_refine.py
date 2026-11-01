"""Synthetic texture-refinement checks, with no model service calls."""
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender import bl
from herogen.blender import refine as stage
from herogen.blender.final_model import final_model
from herogen.pipeline.client import ApiError, CallLog, IMAGE_MODEL, SAM_MODEL, ModelClient, ReplayMiss, SamInstance
try:
    from herogen.blender.refine import cameras, character, imaging as refine, prompts
    HAVE_REFINE = True
except ImportError:
    HAVE_REFINE = False

def disc(shape, cx, cy, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class CameraTests(unittest.TestCase):
    def test_front_camera_maps_centre_to_image_centre_and_up_to_up(self):
        cam = cameras.camera("c", [0.1, 0.2, 0.3], 0, 2.0, (100, 200))
        x, y, _ = cameras.project(cam, np.array([[0.1, 5.0, 0.3], [0.1, 0.2, 0.8]]))
        np.testing.assert_allclose([x[0], y[0]], [50, 100])
        np.testing.assert_allclose([x[1], y[1]], [50, 100 - 0.5 * 100])   # 200 px over 2 units

    def test_turned_camera_depth_grows_away_from_camera(self):
        cam = cameras.camera("c", [0, 0, 0], 90, 1.0, (64, 64))
        _, _, fwd = cameras.axes(cam)
        _, _, d = cameras.project(cam, np.array([fwd * 0.5, -fwd * 0.5]))
        self.assertGreater(d[0], d[1])

    def test_position_codec_round_trips(self):
        p = np.array([[[0.25, -1.5, 1.9, 1.0]]], np.float32)
        enc = p.copy()
        enc[..., :3] = (p[..., :3] - cameras.POS_LO) / cameras.POS_EXT
        dec, valid = cameras.decode_pos(enc)
        np.testing.assert_allclose(dec, p[..., :3], atol=1e-6)
        self.assertTrue(valid.all())


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class GateTests(unittest.TestCase):
    def test_chamfer_is_zero_for_identical_and_grows_with_shift(self):
        a = disc((128, 128), 64, 64, 30)
        self.assertEqual(refine.chamfer(a, a), (0.0, 0.0))
        med, p90 = refine.chamfer(a, disc((128, 128), 70, 64, 30))
        self.assertGreater(med, 1.0)
        self.assertLessEqual(p90, 7.0)

    def test_chamfer_ignores_the_image_border(self):
        a = np.zeros((64, 64), bool)
        a[20:, 10:50] = True            # touches the bottom border
        b = a.copy()
        b[-2:] = False                  # differs only next to the border
        self.assertEqual(refine.chamfer(a, b)[0], 0.0)

    def test_figure_mask_finds_the_figure_on_the_grey(self):
        img = np.full((64, 64, 3), refine.C.BG, np.float32)
        m = disc((64, 64), 32, 32, 20)
        img[m] = (0.8, 0.6, 0.5)
        got = refine.figure_mask(img)
        self.assertGreater((got & m).sum() / (got | m).sum(), 0.97)

    def test_shifted_neutral_background_preserves_aligned_outline(self):
        fig = disc((192, 192), 96, 96, 54)
        for background in (0.58, 0.82):
            with self.subTest(background=background):
                rgb = np.full((192, 192, 3), background, np.float32)
                rgb[fig] = (0.2, 0.3, 0.25)
                measured = refine.figure_mask(rgb, render_figure=fig)
                passed, checks = refine.outline_gate(fig, measured)
                self.assertTrue(passed, checks)
                self.assertGreater(checks["iou"], 0.99)

    def test_shifted_figure_still_fails_with_calibrated_background(self):
        fig = disc((192, 192), 96, 96, 54)
        moved = disc((192, 192), 118, 96, 54)
        rgb = np.full((192, 192, 3), 0.58, np.float32)
        rgb[moved] = (0.2, 0.3, 0.25)
        measured = refine.figure_mask(rgb, render_figure=fig)
        passed, checks = refine.outline_gate(fig, measured)
        self.assertFalse(passed)
        self.assertGreater((measured & moved).sum() / (measured | moved).sum(), 0.99)
        self.assertGreater(checks["median_px"], checks["median_limit"])

    def test_small_background_drift_keeps_near_grey_surface(self):
        fig = disc((192, 192), 96, 96, 54)
        rgb = np.full((192, 192, 3), 0.46, np.float32)
        rgb[fig] = 0.525
        measured = refine.figure_mask(rgb, render_figure=fig)
        passed, checks = refine.outline_gate(fig, measured)
        self.assertTrue(passed, checks)
        self.assertGreater(checks["iou"], 0.99)

    def test_background_calibration_requires_uniform_neutral_exterior(self):
        fig = disc((128, 128), 64, 64, 30)
        coloured = np.full((128, 128, 3), (0.8, 0.2, 0.2), np.float32)
        self.assertEqual(refine._background_colour(coloured, fig), refine.C.BG)
        nonuniform = np.full((128, 128, 3), 0.8, np.float32)
        nonuniform[:64] = 0.2
        self.assertEqual(refine._background_colour(nonuniform, fig), refine.C.BG)
        self.assertEqual(refine._background_colour(coloured, np.ones_like(fig)), refine.C.BG)

    def test_part_gate_rejects_a_face_painted_where_the_render_has_none(self):
        fig = disc((100, 100), 50, 50, 45)
        none = {"face": np.zeros_like(fig)}
        face = {"face": disc((100, 100), 50, 40, 15)}
        ok, rec, _ = refine.part_gate(none, face, fig)
        self.assertFalse(ok)
        self.assertEqual(rec["face"]["outside"], 1.0)
        ok, rec, _ = refine.part_gate(face, face, fig)
        self.assertTrue(ok)
        ok, rec, _ = refine.part_gate(none, none, fig)
        self.assertTrue(ok)
        self.assertIsNone(rec["face"])

    def test_part_gate_accepts_a_sharper_face_inside_a_soft_one(self):
        fig = disc((100, 100), 50, 50, 45)
        soft = {"face": disc((100, 100), 50, 50, 30)}      # face and beard on the soft render
        sharp = {"face": disc((100, 100), 50, 40, 15)}     # face only on the repaint
        ok, rec, _ = refine.part_gate(soft, sharp, fig)
        self.assertTrue(ok)
        self.assertLess(rec["face"]["iou"], 0.3)

    def test_figure_mask_preserves_grey_armour_cut_by_covered_frame_border(self):
        h = w = 128
        render_fig = np.zeros((h, w), bool)
        render_fig[24:64, 44:84] = True
        render_fig[60:, 28:100] = True                      # torso armour cut by the bottom border
        rgb = np.full((h, w, 3), refine.C.BG, np.float32)
        rgb[24:64, 44:84] = (0.72, 0.56, 0.46)              # head
        rgb[60:66, 28:100] = (0.20, 0.22, 0.25)             # dark collar
        rgb[60:, 28:34] = (0.20, 0.22, 0.25)                # dark left shoulder outline
        rgb[60:, 94:100] = (0.20, 0.22, 0.25)               # dark right shoulder outline
        rgb[66:, 34:94] = refine.C.BG + 0.02                # grey steel within BG_TOL, touching bottom border

        flooded = refine.figure_mask(rgb)
        self.assertFalse(flooded[96, 64])
        self.assertLess((flooded & render_fig).sum() / (flooded | render_fig).sum(), 0.55)

        preserved = refine.figure_mask(rgb, render_figure=render_fig)
        self.assertTrue(preserved[96, 64])
        self.assertGreater((preserved & render_fig).sum() / (preserved | render_fig).sum(), 0.98)

        blank = np.full((h, w, 3), refine.C.BG, np.float32)
        self.assertFalse(refine.figure_mask(blank, render_figure=render_fig).any())

    def test_chamfer_unclipped_keeps_frame_touching_five_px_offset_near_five_px(self):
        # Left silhouette sits at x=2 (inside default 4 px clip) while bottom is cut by the frame.
        a = np.zeros((128, 128), bool)
        a[20:, 2:62] = True
        b = np.zeros((128, 128), bool)
        b[20:, 7:67] = True
        b[-6:] = False                                      # frame-cut difference inside the 16 px cut zone
        _, old_p90 = refine.chamfer(a, b, unclipped=False)
        self.assertGreater(old_p90, 20.0)
        med, p90 = refine.chamfer(a, b, unclipped=True)
        np.testing.assert_allclose((med, p90), (5.0, 5.0), atol=0.2)

        # Disc directly touching x=0 on `a`, shifted right by 5 px on `b` (normal offset 5*|cos(theta)| <= 5 px).
        da = disc((128, 128), 50, 64, 50)
        db = disc((128, 128), 55, 64, 50)
        self.assertTrue(refine.border_touch(da).any())
        d_med, d_p90 = refine.chamfer(da, db, unclipped=True)
        self.assertGreater(d_med, 2.5)
        self.assertGreaterEqual(d_p90, 4.5)
        self.assertLessEqual(d_p90, 5.2)

    def test_closeup_gate_caps_allowance_and_rejects_shifted_silhouette(self):
        body = cameras.camera("body+0", [0, 0, 1], 0, 2.24, (1024, 1536))
        head = cameras.camera("head+0", [0, 0, 1.8], 0, 0.55, (1024, 1024))
        self.assertTrue(refine.is_closeup(head, body))
        self.assertFalse(refine.is_closeup(body, body))

        base = disc((1024, 1024), 512, 512, 340)
        mild = disc((1024, 1024), 524, 512, 340)            # ~8.5 px median / ~11.4 px p90: > 6/16, <= 10/24 cap
        passed_default, _ = refine.outline_gate(base, mild, unclipped=True, closeup=False)
        passed_closeup, checks = refine.outline_gate(base, mild, unclipped=True, closeup=True)
        self.assertFalse(passed_default)
        self.assertTrue(passed_closeup)
        self.assertEqual((checks["median_limit"], checks["p90_limit"]), (10.0, 24.0))

        # 26 px shift keeps IoU > 0.90 and would pass raw PPU scaling (~22.9 / 61.1 px), but exceeds the 10/24 cap.
        shifted = disc((1024, 1024), 538, 512, 340)
        passed_shifted, shifted_checks = refine.outline_gate(base, shifted, unclipped=True, closeup=True)
        self.assertGreater(shifted_checks["iou"], 0.90)
        self.assertFalse(passed_shifted)


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class ColourTests(unittest.TestCase):
    def test_colour_match_takes_render_palette_and_keeps_repaint_detail(self):
        rng = np.random.default_rng(0)
        h = w = 256
        both = np.ones((h, w), bool)
        render = np.zeros((h, w, 3), np.float32)
        render[:, : w // 2] = (0.2, 0.3, 0.7)        # SAM's blue shirt
        render[:, w // 2:] = (0.85, 0.65, 0.5)       # SAM's skin
        repaint = np.zeros_like(render)
        repaint[:, : w // 2] = (0.35, 0.45, 0.7)     # repaint's lighter blue
        repaint[:, w // 2:] = (0.95, 0.8, 0.75)      # repaint's paler skin
        detail = (rng.random((h, w)) > 0.97)          # sharp dark specks only the repaint has
        repaint[detail] *= 0.3
        out = refine.colour_match(render, repaint, both)
        inner = np.zeros((h, w), bool)
        inner[20:-20, 20:100] = True
        plain = inner & ~detail
        np.testing.assert_allclose(out[plain].mean(0), (0.2, 0.3, 0.7), atol=0.04)
        contrast = out[plain].mean() - out[inner & detail].mean()
        self.assertGreater(contrast, 0.15)


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class HairModeTests(unittest.TestCase):
    def test_interior_agreement_keeps_detail_and_drops_a_swapped_region(self):
        rng = np.random.default_rng(1)
        h = w = 256
        fig = disc((h, w), 128, 128, 110)
        render = np.full((h, w, 3), refine.C.BG, np.float32)
        render[fig] = (0.25, 0.15, 0.1)                  # soft dark hair
        repaint = render.copy()
        strands = fig & (rng.random((h, w)) > 0.8)
        repaint[strands] = (0.35, 0.22, 0.15)            # fine strands: same colour at low frequency
        ear = disc((h, w), 170, 128, 25)
        repaint[ear] = (0.9, 0.7, 0.6)                   # skin painted where the render has hair
        frac, agree, interior = refine.interior_agreement(render, repaint, fig, fig)
        self.assertLess(agree[disc((h, w), 170, 128, 10)].max(), 0.1)
        self.assertGreater(agree[disc((h, w), 80, 128, 20)].min(), 0.9)
        self.assertGreater(frac, 0.8)
        self.assertFalse(interior[0, 0])


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class AffineDegeneracyTests(unittest.TestCase):
    def setUp(self):
        self.figure = np.zeros((128, 96), bool)
        self.figure[16:48, 32:64] = True
        self.centroid = np.array([48.0, 32.0])

    def matrix(self, linear, shift=(0, 0)):
        linear = np.asarray(linear, float)
        return np.column_stack([linear, self.centroid - linear @ self.centroid + shift])

    def reasons(self, matrix):
        return refine._affine_degeneracy(matrix, self.figure)["reasons"]

    def test_ordinary_nonuniform_affine_is_retained(self):
        self.assertEqual(self.reasons(self.matrix([[0.944, -0.009], [-0.004, 0.971]])), [])
        self.assertEqual(self.reasons(self.matrix([[0.911, 0.003], [0.003, 1.004]])), [])

    def test_failed_nonfinite_reflected_and_collapsed_fits(self):
        self.assertEqual(self.reasons(None), ["fit_failed"])
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertEqual(self.reasons(np.full((2, 3), value)), ["nonfinite"])
        for linear in ([[0.002, 0], [0, 1]], [[0, 0], [0, 1]], [[-1, 0], [0, 1]]):
            with self.subTest(linear=linear):
                self.assertIn("determinant", self.reasons(self.matrix(linear)))

    def test_directional_scale_bounds(self):
        for scale in (0.85, 1.25):
            self.assertNotIn("scale", self.reasons(self.matrix([[scale, 0], [0, 1]])))
        for scale in (0.85 - 1e-6, 1.25 + 1e-6):
            self.assertIn("scale", self.reasons(self.matrix([[scale, 0], [0, 1]])))
        # Both column lengths are ordinary, but their near-parallel directions collapse area.
        self.assertIn("scale", self.reasons(self.matrix([[0.71, 0.70], [0.70, 0.71]])))

    def test_area_shrinkage_without_single_axis_collapse(self):
        for area, expected in ((0.8 - 1e-6, ["determinant"]), (0.8 + 1e-6, [])):
            with self.subTest(area=area):
                self.assertEqual(self.reasons(self.matrix(np.eye(2) * np.sqrt(area))), expected)

    def test_directional_ratio_catches_moderate_axis_collapse(self):
        for ratio in (1.1161, 1.125 - 1e-6):
            self.assertEqual(self.reasons(self.matrix([[1.06, 0], [0, 1.06 / ratio]])), [])
        for ratio in (1.125 + 1e-6, 1.1394):
            self.assertEqual(self.reasons(self.matrix([[1.06, 0], [0, 1.06 / ratio]])), ["axis_ratio"])

    def test_shear_bound_uses_normalized_column_angle(self):
        for shear, expected in ((0.15 - 1e-6, False), (0.15 + 1e-6, True)):
            with self.subTest(shear=shear):
                linear = [[1, shear], [0, np.sqrt(1 - shear ** 2)]]
                self.assertEqual("shear" in self.reasons(self.matrix(linear)), expected)

    def test_rotation_bound_about_silhouette_centroid(self):
        for angle, expected in ((4.999, []), (-4.999, []), (5.001, ["rotation"]), (-5.001, ["rotation"])):
            with self.subTest(angle=angle):
                a = np.radians(angle)
                linear = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
                self.assertEqual(self.reasons(self.matrix(linear)), expected)

    def test_centroid_displacement_includes_linear_map_and_translation(self):
        # Rotation and scaling about this off-centre silhouette do not move its centroid.
        linear = [[0.98, -0.02], [0.02, 0.98]]
        self.assertEqual(self.reasons(self.matrix(linear, (16, 0))), [])
        self.assertEqual(self.reasons(self.matrix(linear, (16.001, 0))), ["centroid"])
        self.assertEqual(self.reasons(self.matrix(linear, (0, -16.001))), ["centroid"])
        scaled = self.matrix(linear, (16, 0))
        scaled[:, 2] *= 2
        figure = np.repeat(np.repeat(self.figure, 2, axis=0), 2, axis=1)
        self.assertEqual(refine._affine_degeneracy(scaled, figure)["reasons"], [])

    def test_empty_silhouette_is_explicit(self):
        record = refine._affine_degeneracy(self.matrix(np.eye(2)), np.zeros_like(self.figure))
        self.assertEqual(record["reasons"], ["empty_figure"])
        self.assertIsNone(record["centroid_displacement"])

    def test_rejected_outline_does_not_invoke_fallback_for_valid_affine(self):
        figure = disc((128, 128), 64, 64, 35)
        render = np.full((128, 128, 3), character.BG, np.float32)
        render[figure] = 0.2
        edit = np.full_like(render, character.BG)
        primary = np.array([[0.95, 0.01, 10], [0, 0.98, -4]])
        with mock.patch.object(refine, "refine_affine", return_value=primary), \
                mock.patch.object(refine, "_register_similarity") as fallback:
            _, _, warped, rec = refine.register(render, figure.astype(float), edit)
        fallback.assert_not_called()
        expected = primary.copy()
        expected[:, 2] /= 4  # 512 work pixels to 128 render pixels
        np.testing.assert_allclose(rec["affine_render_to_edit"], expected)
        self.assertEqual(rec["method"], "affine")
        self.assertEqual(rec["affine_diagnostics"]["reasons"], [])
        passed, _ = refine.outline_gate(figure, refine.figure_mask(warped, render_figure=figure))
        self.assertFalse(passed)

    def test_degenerate_primary_invokes_fallback_once_before_gates(self):
        figure = disc((128, 128), 64, 64, 35)
        render = np.full((128, 128, 3), character.BG, np.float32)
        render[figure] = 0.2
        identity = np.array([[1., 0, 0], [0, 1., 0]])
        for primary in (None, np.array([[0.002, 0, 255], [0, 1., 0]])):
            with self.subTest(primary=primary), \
                    mock.patch.object(refine, "refine_affine", return_value=primary), \
                    mock.patch.object(refine, "_register_similarity", return_value=identity) as fallback:
                _, _, warped, rec = refine.register(render, figure.astype(float), render)
            fallback.assert_called_once()
            self.assertEqual(rec["method"], "similarity_fallback")
            self.assertTrue(rec["affine_diagnostics"]["reasons"])
            np.testing.assert_array_equal(warped, render)


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class RegistrationTests(unittest.TestCase):
    def test_affine_recovers_a_translation_and_brightness_change(self):
        import cv2
        yy, xx = np.mgrid[:128, :128]
        mask = disc((128, 128), 64, 64, 45)
        render = np.full((128, 128, 3), character.BG, np.float32)
        render[mask] = np.stack([0.2 + xx[mask] / 300, 0.2 + yy[mask] / 400,
                                0.4 + np.sin(xx[mask] / 8) / 9], -1)
        edit = cv2.warpAffine(render, np.float32([[1, 0, 4], [0, 1, -3]]), (128, 128),
                              borderMode=cv2.BORDER_REPLICATE) * 0.9 + 0.03
        _, _, warped, rec = refine.register(render, mask.astype(float), edit)
        affine = np.array(rec["affine_render_to_edit"])
        np.testing.assert_allclose(affine[:, 2], (4, -3), atol=0.7)
        self.assertLess(np.abs(warped[mask] - (render[mask] * 0.9 + 0.03)).mean(), 0.01)

    def test_narrow_repaint_does_not_collapse_when_luminance_contrast_reverses(self):
        h, w = 384, 256
        yy, _ = np.indices((h, w))
        fig = np.zeros((h, w), bool)
        fig[10:75, 112:150] = True
        fig[70:202, 112:145] = True
        fig[190:365, 114:138] = True
        fig[358:374, 114:155] = True
        render = np.full((h, w, 3), character.BG, np.float32)
        render[fig & (yy < 75)] = 0.7
        shirt = fig & (yy >= 75) & (yy < 202)
        render[shirt] = (0.25, 0.10, 0.3)
        render[fig & (yy >= 202)] = (0.3, 0.32, 0.33)
        edit = render.copy()
        edit[shirt] = (0.65, 0.52, 0.7)

        _, _, warped, rec = refine.register(render, fig.astype(float), edit)
        affine = np.array(rec["affine_render_to_edit"])
        np.testing.assert_allclose(affine[:, :2], np.eye(2), atol=0.02)
        np.testing.assert_allclose(affine[:, 2], 0, atol=2)
        passed, checks = refine.outline_gate(fig, refine.figure_mask(warped, render_figure=fig))
        self.assertTrue(passed, checks)

    def test_registration_does_not_hide_a_displaced_limb(self):
        shape = (256, 192)
        fig = disc(shape, 96, 48, 30)
        fig[74:178, 66:126] = True
        fig[88:160, 40:66] = True
        fig[88:160, 126:152] = True
        fig[174:244, 68:90] = True
        fig[174:244, 102:124] = True
        moved = fig.copy()
        moved[88:160, 40:66] = False
        moved[88:160, 18:44] = True
        render = np.full((*shape, 3), character.BG, np.float32)
        edit = render.copy()
        render[fig] = (0.2, 0.25, 0.3)
        edit[moved] = (0.2, 0.25, 0.3)
        _, _, warped, _ = refine.register(render, fig.astype(float), edit)
        passed, checks = refine.outline_gate(fig, refine.figure_mask(warped, render_figure=fig))
        self.assertFalse(passed, checks)
        self.assertLess(checks["iou"], refine.GATE["iou_min"])

    def test_registration_measures_uniform_scale_for_the_hair_gate(self):
        import cv2
        yy, xx = np.indices((128, 128))
        fig = disc((128, 128), 64, 64, 38)
        render = np.full((128, 128, 3), character.BG, np.float32)
        render[fig] = np.stack([0.15 + xx[fig] / 700, 0.2 + yy[fig] / 600,
                                0.25 + np.sin(xx[fig] / 8) / 12], axis=-1)
        transform = cv2.getRotationMatrix2D((64, 64), 3, 1.14)
        edit = cv2.warpAffine(render, transform, (128, 128), borderMode=cv2.BORDER_REPLICATE)
        _, _, _, rec = refine.register(render, fig.astype(float), edit)
        self.assertAlmostEqual(rec["scale"], 1.14, delta=0.01)
        self.assertGreater(abs(rec["scale"] - 1), refine.HAIR_GATE["scale_max_change"])
        np.testing.assert_allclose(np.array(rec["affine_render_to_edit"]), transform, atol=0.7)

    def test_registration_does_not_hide_nonuniform_figure_resize(self):
        import cv2
        fig = disc((192, 192), 96, 96, 54)
        render = np.full((192, 192, 3), character.BG, np.float32)
        render[fig] = (0.2, 0.25, 0.3)
        edit = cv2.warpAffine(render, np.float32([[1.3, 0, -28.8], [0, 0.85, 14.4]]),
                              (192, 192), borderMode=cv2.BORDER_REPLICATE)
        _, _, warped, _ = refine.register(render, fig.astype(float), edit)
        passed, checks = refine.outline_gate(fig, refine.figure_mask(warped, render_figure=fig))
        self.assertFalse(passed, checks)

    def test_repaint_detail_does_not_rotate_a_matching_outline(self):
        fig = disc((192, 128), 64, 96, 50)
        fig[80:170, 44:84] = True
        render = np.full((192, 128, 3), character.BG, np.float32)
        render[fig] = (0.2, 0.25, 0.3)
        edit = render.copy()
        yy, xx = np.indices(fig.shape)
        edit[fig & (np.abs(xx - yy / 3 - 28) < 5)] = 0.9
        _, _, warped, rec = refine.register(render, fig.astype(float), edit)
        np.testing.assert_allclose(np.array(rec["affine_render_to_edit"]), [[1, 0, 0], [0, 1, 0]], atol=0.1)
        passed, checks = refine.outline_gate(fig, refine.figure_mask(warped, render_figure=fig))
        self.assertTrue(passed, checks)

    def test_empty_colour_overlap_is_finite(self):
        rgb = np.ones((16, 16, 3), np.float32) * 0.2
        np.testing.assert_array_equal(refine.colour_match(rgb, rgb, np.zeros((16, 16), bool)), rgb)

    def test_camera_set_and_frozen_prompts(self):
        import hashlib

        self.assertEqual({k: hashlib.sha256(v.encode()).hexdigest() for k, v in prompts.PROMPTS.items()}, {
            "RF2W": "aebe4402f963921b91f79ba9fd54b8ff9d0ec0242110b21fa210578b7db4e090",
            "RF2R": "794b08622f4ff650261513c39aee5f8ac9b92e6c9bfaf65a1334c96b86d4b224",
            "RF4U": "8bdcf59595af8968867cbd4c096d29d9dced84a96382a21fa3d4f025e088265e",
            "RF5U": "950c00c50c62bc9d35035a303de4c14622323b2fac5204ff029d87b0a2034791",
            "RF3H": "f4f45c11280cac98d8234ee729650cffe1bcc73c8f6fc844df9b407248936f2f",
        })
        info = {"height": 2, "body_centre": [0, 0, 1], "hi": [1, 1, 2],
                "head_centre": [0, 0, 1.8], "head_side": 0.4}
        cams = character.cameras_for(info)
        self.assertEqual(len(stage.REFINE_ORDER), 12)
        self.assertEqual(len(stage.HAIR_ORDER), 8)
        for name in (*stage.REFINE_ORDER, *stage.HAIR_ORDER):
            self.assertIn(name, cams)
        self.assertIn("in profile", prompts.prompt("RF2R", 270))
        self.assertIn("back and side of the head", prompts.prompt("RF3H", 225))
        self.assertIn("second image", prompts.prompt("RF3H", 180))

    def test_hair_weight_cannot_write_face_scalp_or_outline(self):
        shape = (128, 128)
        fig = disc(shape, 64, 64, 50)
        head = disc(shape, 64, 50, 30)
        hair = head & (np.indices(shape)[1] < 55)
        face = head & (np.indices(shape)[1] > 70)
        rgb = np.full((*shape, 3), character.BG, np.float32)
        rgb[fig] = (0.2, 0.15, 0.1)
        inp, edit = Image.fromarray((rgb * 255).astype(np.uint8)), Image.fromarray((rgb * 255).astype(np.uint8))
        def segment(image, text):
            m = head if text == "head" else hair
            return [SamInstance(m, (0, 0, 128, 128))]
        client = SimpleNamespace(segment=segment)
        yy, xx = np.mgrid[:128, :128]
        weight = refine.hair_weight(client, inp, edit, rgb, rgb, fig, xx + .5, yy + .5, face)
        self.assertEqual(weight[face].max(), 0)
        self.assertEqual(weight[head & (xx > 60)].max(), 0)
        self.assertEqual(weight[~fig].max(), 0)
        self.assertGreater(weight[hair & (xx < 50)].max(), .5)
        client.segment = lambda image, text: [SamInstance(head if text == "head" else np.zeros(shape, bool),
                                                        (0, 0, 128, 128))]
        allowed = refine.hair_allowed(client, inp, edit, xx + .5, yy + .5, shape)
        self.assertFalse(allowed[head].any())

    def test_projection_preserves_occluded_backfacing_and_unseen_texels(self):
        tex = object.__new__(refine.Texture)
        tex.P = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [5, 0, 0]], float)
        tex.N = np.array([[0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]], float)
        tex.idx = np.arange(4)
        tex.Q = np.ones(4) * 0.01
        tex.height = 1
        tex.T = np.full((2, 2, 3), 0.2)
        cam = cameras.camera("c", [0, 0, 0], 0, 1, (32, 32))
        count = tex.apply(cam, np.ones((32, 32, 3)), np.ones((32, 32)), np.zeros((32, 32)),
                          np.ones((32, 32), bool), np.ones((32, 32)))
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(tex.T.reshape(-1, 3)[1:], 0.2)
        self.assertGreater(tex.T[0, 0].min(), .9)

    def test_projection_cannot_interpolate_into_forbidden_hair_or_face_pixels(self):
        tex = object.__new__(refine.Texture)
        tex.P = np.array([[-.225, 0, -.125]], float)
        tex.N = np.array([[0, -1, 0]], float)
        tex.idx = np.array([0])
        tex.Q = np.array([.01])
        tex.height = 1
        tex.T = np.full((1, 1, 3), .2)
        cam = cameras.camera("c", [0, 0, 0], 0, 1, (4, 4))
        weights = np.zeros((4, 4))
        weights[:, 0] = 1
        # The texel projects to (1.1, 2.5): inside forbidden column 1, beside allowed column 0.
        count = tex.apply(cam, np.ones((4, 4, 3)), np.ones((4, 4)), np.zeros((4, 4)),
                          np.ones((4, 4), bool), weights)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(tex.T, .2)

    def test_closeup_feather_scales_with_resolution_and_spares_uncut_silhouettes(self):
        self.assertEqual(refine.closeup_border_px((512, 512), 96), 48)
        self.assertEqual(refine.closeup_border_px((1024, 1024), 96), 96)
        self.assertEqual(refine.closeup_border_px((1536, 1024), 96), 144)
        self.assertEqual(refine.closeup_border_px((2048, 2048), 96), 192)
        for bad in (0, -8, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                refine.closeup_border_px((128, 128), bad)
        fig = np.zeros((128, 128), bool)
        # Uncut crown/nose 4 px from top and left borders; cut neck touching bottom border.
        fig[4:, 4:100] = True
        f = refine.closeup_feather(fig, 96)  # w = 12 px at 128
        self.assertEqual(float(f[-1, 20]), 0.0)
        self.assertAlmostEqual(float(f[-1 - 6, 20]), 0.5, delta=0.02)
        self.assertEqual(float(f[-1 - 12, 20]), 1.0)
        # Uncut top/left silhouette pixels near empty frame borders retain full weight 1.0.
        self.assertEqual(float(f[4, 4]), 1.0)
        self.assertEqual(float(f[10, 4]), 1.0)

    def test_closeup_projection_blends_into_body_without_base_bleed(self):
        # Vertical front-facing plate spanning x in [-0.2, 0.2], z in [0.3, 0.85]; head camera cuts at z = 0.55.
        xs, zs = np.meshgrid(np.linspace(-0.2, 0.2, 32), np.linspace(0.3, 0.85, 48))
        pos = np.stack([xs, np.zeros_like(xs), zs], -1)
        nrm = np.broadcast_to(np.array([0.0, -1.0, 0.0]), pos.shape).copy()
        cov = np.ones((48, 32), bool)
        base = np.full((48, 32, 3), 0.05, np.float32)  # dark SAM base must never bleed into the feather strip
        body_cam = cameras.camera("body+0", [0.0, 0.0, 0.5], 0, 1.0, (128, 128))
        head_cam = cameras.camera("head+0", [0.0, 0.0, 0.75], 0, 0.4, (128, 128))
        body_rgb = np.full((128, 128, 3), 0.40, np.float32)
        head_rgb = np.full((128, 128, 3), 0.80, np.float32)
        valid = np.ones((128, 128), bool)
        depth_b = np.full((128, 128), 2.0, np.float32)
        depth_h = np.full((128, 128), 0.8, np.float32)
        fig_h = np.zeros((128, 128), bool)
        fig_h[16:, 16:112] = True  # cut along bottom edge only (plate top z=0.85 projects to row 32, inside fig_h)
        f_head = refine.closeup_feather(fig_h, 96)  # 12 px ramp at bottom border
        ones = np.ones((128, 128), np.float32)

        with mock.patch.object(character, "BODY_RES", (128, 128)):
            tex_seq = refine.Texture.from_arrays(pos, nrm, cov, base, 1.0)
            tex_seq.apply(head_cam, head_rgb, f_head, depth_h, valid, fig_h.astype(np.float32))
            tex_seq.apply(body_cam, body_rgb, ones, depth_b, valid, ones)

            tex_c2f = refine.Texture.from_arrays(pos, nrm, cov, base, 1.0)
            tex_c2f.apply(body_cam, body_rgb, ones, depth_b, valid, ones)
            tex_c2f.apply(head_cam, head_rgb, f_head, depth_h, valid, fig_h.astype(np.float32))

        np.testing.assert_allclose(tex_seq.T, tex_c2f.T, atol=1e-5)
        col = tex_seq.T[:, 16, 0]
        # Below z = 0.55 (outside head frame): pure body 0.40; deep inside head frame: pure head 0.80.
        self.assertAlmostEqual(float(col[4]), 0.40, places=3)
        self.assertAlmostEqual(float(col[-4]), 0.80, places=3)
        # Every texel across the transition strip stays strictly within [0.40, 0.80] (no 0.05 base bleed)
        # and rises monotonically from body to head.
        self.assertGreaterEqual(float(col.min()), 0.40 - 1e-4)
        self.assertLessEqual(float(col.max()), 0.80 + 1e-4)
        self.assertTrue(np.all(np.diff(col) >= -1e-4))

    def test_solve_exposure_recovers_drift_preserves_artwork_anchor_and_ignores_bg(self):
        shape = (128, 128)
        cam_b0 = cameras.camera("body+0", [0.0, 0.0, 0.5], 0, 1.0, shape)
        cam_b45 = cameras.camera("body+45", [0.0, 0.0, 0.5], 45, 1.0, shape)
        cam_h0 = cameras.camera("head+0", [0.0, 0.0, 0.65], 0, 0.45, shape)

        def make_cyl_view(cam, drift, *, hair=False, closeup=False, outlier=False):
            # Project each pixel ray of `cam` onto cylindrical front shell x^2 + y^2 = 0.22^2.
            right, up, fwd = cameras.axes(cam)
            centre = np.asarray(cam["centre"], float)
            ortho = float(cam["ortho"])
            xs = ((np.arange(shape[1]) + 0.5) - shape[1] / 2.0) * (ortho / max(shape))
            ys = (shape[0] / 2.0 - (np.arange(shape[0]) + 0.5)) * (ortho / max(shape))
            gx, gy = np.meshgrid(xs, ys)
            p0 = centre + gx[..., None] * right + gy[..., None] * up
            # Ray p0 + t * fwd intersects x^2 + y^2 = R^2
            r2 = 0.22 ** 2
            a = fwd[0] ** 2 + fwd[1] ** 2
            b = 2.0 * (p0[..., 0] * fwd[0] + p0[..., 1] * fwd[1])
            c = p0[..., 0] ** 2 + p0[..., 1] ** 2 - r2
            disc_v = b * b - 4.0 * a * c
            hit = (disc_v > 0) & (p0[..., 2] >= 0.1) & (p0[..., 2] <= 0.9)
            t = (-b - np.sqrt(np.maximum(disc_v, 0.0))) / (2.0 * a)
            pos = (p0 + t[..., None] * fwd).astype(np.float32)
            # Smooth true albedo varying across z and azimuth (not linked to any SAM base render).
            theta = np.arctan2(pos[..., 0], -pos[..., 1])
            albedo = (0.42 + 0.12 * np.sin(5.0 * pos[..., 2]) + 0.06 * np.cos(2.0 * theta)).astype(np.float32)
            rgb = np.full((*shape, 3), character.BG, np.float32)
            base_col = np.stack([albedo * 1.05, albedo, albedo * 0.90], -1)
            rgb[hit] = np.clip(base_col[hit] * drift, 0.0, 1.0)
            if outlier:
                patch = hit & (np.abs(gx) < 0.04) & (np.abs(gy) < 0.04)
                rgb[patch] = 0.92  # localized repaint ornament difference must not skew exposure
            return {
                "key": ("hair-" if hair else "") + cam["name"],
                "cam": cam,
                "hair": hair,
                "closeup": closeup,
                "alpha": hit.astype(np.float32),
                "pos": pos,
                "valid": hit,
                "depth": t.astype(np.float32),
                "refined": rgb,
                "weight": hit.astype(np.float32),
            }

        drifts = {"body+0": 0.90, "body+45": 1.10, "head+0": 0.86}
        views = [
            make_cyl_view(cam_b0, drifts["body+0"]),
            make_cyl_view(cam_b45, drifts["body+45"], outlier=True),
            make_cyl_view(cam_h0, drifts["head+0"], closeup=True),
            make_cyl_view(cam_h0, 0.72, hair=True, closeup=True),
        ]
        solved = refine.solve_exposure(views, 1.0, mode="scalar")
        # Relative product `gain * drift` is equalised across all three non-hair views within 1.5%.
        corr = [solved[k][0] * d for k, d in drifts.items()]
        self.assertLess(max(corr) / min(corr) - 1.0, 0.015)

        # Anchored mean luminance matches the accepted non-hair views (hair pass cannot drag anchor).
        solved_no_hair = refine.solve_exposure(views[:3], 1.0, mode="scalar")
        for k in drifts:
            self.assertAlmostEqual(solved[k][0], solved_no_hair[k][0], delta=0.008)

        # Exterior background pixels remain untouched by apply_exposure.
        adjusted = refine.apply_exposure(views[0]["refined"], views[0]["alpha"] > 0.5, 1.15, 0.02)
        np.testing.assert_allclose(adjusted[~views[0]["valid"]], character.BG, atol=1e-6)

        # Affine mode, single-view / disconnected fallbacks, and invalid mode check.
        aff = refine.solve_exposure(views[:3], 1.0, mode="affine")
        self.assertTrue(all(len(v) == 2 for v in aff.values()))
        self.assertEqual(refine.solve_exposure(views[:1], 1.0), {"body+0": (1.0, 0.0)})
        cam_back = cameras.camera("body+180", [0.0, 0.0, 0.5], 180, 1.0, shape)
        disc_views = [views[0], make_cyl_view(cam_back, 1.12)]
        self.assertEqual(refine.solve_exposure(disc_views, 1.0), {"body+0": (1.0, 0.0), "body+180": (1.0, 0.0)})
        with self.assertRaises(ValueError):
            refine.solve_exposure(views[:2], 1.0, mode="quadratic")

    def test_rebake_views_coarse_to_fine_and_hair_atlas_agreement(self):
        xs, zs = np.meshgrid(np.linspace(-0.2, 0.2, 32), np.linspace(0.3, 0.85, 48))
        pos = np.stack([xs, np.zeros_like(xs), zs], -1)
        nrm = np.broadcast_to(np.array([0.0, -1.0, 0.0]), pos.shape).copy()
        cov = np.ones((48, 32), bool)
        base = np.full((48, 32, 3), 0.10, np.float32)
        body_cam = cameras.camera("body+180", [0.0, 0.0, 0.5], 0, 1.0, (128, 128))
        upper_cam = cameras.camera("upper+180", [0.0, 0.0, 0.6], 0, 0.6, (128, 128))
        valid = np.ones((128, 128), bool)
        depth = np.full((128, 128), 1.0, np.float32)
        body_rgb = np.full((128, 128, 3), 0.55, np.float32)
        # Hair pass agrees in top half (0.56) where hair texture lives, but carries a stale dark band (0.18) lower down.
        hair_rgb = np.full((128, 128, 3), 0.56, np.float32)
        hair_rgb[72:96, :] = 0.18
        views = [
            {"key": "body+180", "cam": body_cam, "hair": False, "closeup": False,
             "alpha": np.ones((128, 128), np.float32), "pos": np.zeros((128, 128, 3), np.float32),
             "valid": valid, "depth": depth, "refined": body_rgb, "weight": np.ones((128, 128), np.float32)},
            {"key": "hair-upper+180", "cam": upper_cam, "hair": True, "closeup": True,
             "alpha": np.ones((128, 128), np.float32), "pos": np.zeros((128, 128, 3), np.float32),
             "valid": valid, "depth": depth, "refined": hair_rgb, "weight": np.ones((128, 128), np.float32)},
        ]
        with mock.patch.object(character, "BODY_RES", (128, 128)):
            tex = refine.Texture.from_arrays(pos, nrm, cov, base, 1.0)
            gains = refine.rebake_views(tex, views=views, solve=False)
        self.assertEqual(gains, {"body+180": 1.0, "hair-upper+180": 1.0})
        # Stale dark band (0.18 vs 0.55 > AGREE_TOL[1]) is suppressed, keeping body colour ~0.55 everywhere.
        self.assertGreater(float(tex.T[..., 0].min()), 0.50)


class BudgetTests(unittest.TestCase):
    def test_cap_boundary_and_prior_spend(self):
        client = ModelClient(Path("unused"))
        budget = stage.Budget(client, .02, prior=.01)
        budget.before_attempt(IMAGE_MODEL)
        client.log.add(IMAGE_MODEL, "edit", "k", False)
        with self.assertRaises(stage.BudgetStop):
            budget.before_attempt(SAM_MODEL)
        for cap in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                stage.Budget(client, cap)
        self.assertAlmostEqual(stage.cost_estimate(), .425)
        self.assertAlmostEqual(stage.cost_estimate(True), .825)

    def test_unknown_outcome_retry_is_capped_before_http(self):
        from herogen.pipeline import client as module, retry
        with tempfile.TemporaryDirectory() as tmp:
            limiter = mock.Mock()
            limiter.acquire.return_value = ("slot", 0)
            client = ModelClient(tmp, mode="live", policy=retry.RetryPolicy(max_attempts=4, sleep=lambda s: None,
                                                                           base_s=0), limiter=limiter)
            client.before_api_attempt = stage.Budget(client, .02).before_attempt
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": "test-placeholder"}), \
                    mock.patch.object(module, "_open", side_effect=urllib.error.URLError(TimeoutError())) as request:
                with self.assertRaises(stage.BudgetStop):
                    client.edit_image("p", [Image.new("RGB", (8, 8))])
            self.assertEqual(request.call_count, 2)
            self.assertEqual(client.log.cost_usd(), .02)
            self.assertEqual(limiter.release.call_count, 2)

    def test_replay_at_zero_cap_misses_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = ModelClient(tmp)
            client.before_api_attempt = stage.Budget(client, 0).before_attempt
            with self.assertRaises(ReplayMiss):
                client.segment(Image.new("RGB", (8, 8)), "head")

    def test_cached_edit_and_segment_are_free_at_zero_cap(self):
        import hashlib
        from herogen.pipeline.client import png_bytes, request_key
        image = Image.new("RGB", (8, 8), "red")
        digest = hashlib.sha256(png_bytes(image)).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            client = ModelClient(tmp)
            client.before_api_attempt = stage.Budget(client, 0).before_attempt
            edit = {"endpoint": "images/edits", "model": IMAGE_MODEL, "prompt": "frozen",
                    "images_sha256": [digest, digest], "n": 1, "size": "1024x1024", "output_format": "png"}
            client._save_png("cached.png", image)
            client._save_entry(request_key(edit), {"image": "cached.png"})
            segment = {"endpoint": "responses", "model": SAM_MODEL, "text": "head", "image_sha256": digest,
                       "metadata": {"include_confidence": "true"}}
            client._save_entry(request_key(segment), {"instances": []})
            self.assertEqual(client.edit_image("frozen", [image, image]).getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(client.segment(image, "head"), [])
            self.assertEqual(client.log.cost_usd(), 0)
            self.assertTrue(all(c["cached"] for c in client.log.calls))


class FakeClient(ModelClient):
    def __init__(self, cache, reject=False, error=None):
        super().__init__(cache)
        self.reject, self.error = reject, error
        self.inputs = []

    def _call(self, model, kind):
        if self.before_api_attempt:
            self.before_api_attempt(model)
        self.log.add(model, kind, "mock-request", False)

    def segment(self, image, text):
        self._call(SAM_MODEL, "segment")
        return []

    def edit_image(self, prompt, images, **kwargs):
        self.inputs.append((prompt, images, kwargs))
        self._call(IMAGE_MODEL, "edit")
        if self.error:
            raise self.error
        return Image.new("RGB", images[0].size, (115, 115, 115)) if self.reject else images[0].copy()


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        (self.run / "03-source.glb").write_bytes(b"source")
        (self.run / "stage-3.json").write_text('{"ok":true}')
        Image.new("RGB", (64, 64), "blue").save(self.run / "01-cover.png")
        self.client = FakeClient(self.run / "cache")
        self.tex = mock.Mock(height=2)
        self.tex.apply.return_value = 17
        self.tex.save.side_effect = lambda path: Path(path)
        self.info = {"height": 2, "body_centre": [0, 0, 1], "hi": [1, 1, 2],
                     "head_centre": [0, 0, 1.8], "head_side": 0.4}
        self.renders = []
        def blender(mode, args, log, **kwargs):
            if mode in ("probe", "render"):
                if mode == "render":
                    self.renders.append(args["texture"])
                folder = Path(args["out"])
                folder.mkdir(parents=True, exist_ok=True)
                mask = disc((64, 64), 32, 32, 22)
                rgba = np.full((64, 64, 4), 0, np.uint8)
                rgba[mask] = (60, 80, 100, 255)
                Image.fromarray(rgba).save(folder / (args["cams"][0]["name"] + ".png"))
            if mode == "export":
                Path(args["out"]).write_bytes(b"refined")
        self.blender = blender
        self.setup_patch = mock.patch.object(character, "setup", return_value=self.info)
        self.prep_patch = mock.patch.object(character, "prep", return_value=self.run / "prep")
        patches = [self.setup_patch, self.prep_patch,
                   mock.patch.object(character, "HEAD_RES", (64, 64)),
                   mock.patch.object(character, "BODY_RES", (64, 64)),
                   mock.patch.object(refine, "Texture", return_value=self.tex),
                   mock.patch("herogen.blender.refine.backend.blender", side_effect=blender),
                   mock.patch.object(cameras, "read_exr", return_value=np.ones((64, 64, 4), np.float32)),
                   mock.patch.object(refine, "register", side_effect=self.identity)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def identity(rgb, alpha, edit):
        yy, xx = np.indices(alpha.shape)
        return xx + .5, yy + .5, edit, {"scale": 1.0}

    def use_character_setup(self, *, head=False, hair=False):
        self.setup_patch.stop()
        self.prep_patch.stop()
        points = np.array([[-.5, -.2, 0], [.5, .2, 2], [0, -.2, 1.5], [0, .2, 1.6]])
        for patch in (mock.patch.object(character, "blender", side_effect=self.blender),
                      mock.patch.object(character, "blender_points", return_value=points)):
            patch.start()
            self.addCleanup(patch.stop)

        def segment(image, text):
            self.client._call(SAM_MODEL, "segment")
            mask = np.zeros(image.size[::-1], bool)
            if head and text == "head":
                mask[8:24, 24:40] = True
            if hair and text == "hair" and self.client.inputs and image is self.client.inputs[-1][1][0]:
                mask[25:40, 25:40] = True  # render-only hair must remain protected on body passes too
            return [SamInstance(mask, (24, 8, 40, 24))] if mask.any() else []

        self.client.segment = mock.Mock(side_effect=segment)

    def test_no_head_completes_body_refinement_with_hair_protection(self):
        self.use_character_setup(hair=True)
        with mock.patch.object(refine, "hair_weight", wraps=refine.hair_weight) as hair_weight:
            rec = stage.run_stage3f(self.run, self.client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["status"], "complete")
        self.assertEqual(rec["outputs"], ["03-refined.glb"])
        self.assertEqual(rec["final_model"], "03-refined.glb")
        self.assertEqual((self.run / "03-refined.glb").read_bytes(), b"refined")
        self.assertEqual((self.run / "03-source.glb").read_bytes(), b"source")
        head_check = next(c for c in rec["checks"] if c["name"] == "head_found")
        self.assertIs(head_check["value"], False)
        self.assertFalse(head_check["gate"])
        self.assertEqual(head_check["note"], "no head on the front render; head views skipped")
        self.assertIn("face exclusion skipped", rec["notes"])
        for name, view in rec["views"].items():
            if name.startswith(("head", "hair-head")):
                self.assertFalse(view["accepted"])
                self.assertEqual(view["tries"], 0)
                self.assertEqual(view["attempts"], [])
                self.assertEqual(view["reason"], "head_not_found")
            else:
                self.assertTrue(view["accepted"])
        self.assertEqual(len(self.renders), len(character.BODY_ANGLES) + len(character.UPPER_ANGLES))
        self.assertEqual(self.tex.apply.call_count, len(self.renders))
        for call in self.tex.apply.call_args_list[:len(character.BODY_ANGLES)]:
            weight = call.args[5]
            self.assertEqual(weight[25:40, 25:40].max(), 0)
            self.assertGreater(weight[40:45, 30:35].max(), 0)
        self.assertEqual(hair_weight.call_count, len(character.UPPER_ANGLES))
        self.assertTrue(all(call.args[-1] is None for call in hair_weight.call_args_list))
        prep_call = next(call for call in character.blender.call_args_list if call.args[0] == "prep")
        self.assertIsNone(prep_call.args[1]["head"])
        info = json.loads((self.run / "blender-work/texture-refine/character.json").read_text())
        self.assertIsNone(info["head_centre"])
        self.assertFalse(any(name.startswith("head") for name in character.cameras_for(info)))

    def test_head_without_face_keeps_all_views_and_skips_empty_face_exclusion(self):
        self.use_character_setup(head=True)
        with mock.patch.object(refine, "hair_weight", wraps=refine.hair_weight) as hair_weight:
            rec = stage.run_stage3f(self.run, self.client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["status"], "complete")
        self.assertEqual(rec["outputs"], ["03-refined.glb"])
        self.assertTrue(all(v["accepted"] for v in rec["views"].values()))
        self.assertTrue(next(c["value"] for c in rec["checks"] if c["name"] == "head_found"))
        self.assertTrue(hair_weight.called)
        self.assertTrue(all(not call.args[-1].any() for call in hair_weight.call_args_list))
        info = json.loads((self.run / "blender-work/texture-refine/character.json").read_text())
        self.assertIn("head", info["boxes_px"])
        self.assertNotIn("face", info["boxes_px"])
        prep_call = next(call for call in character.blender.call_args_list if call.args[0] == "prep")
        self.assertIsNotNone(prep_call.args[1]["head"])

    def test_no_head_and_all_views_rejected_keeps_source(self):
        self.use_character_setup()
        self.client.reject = True
        (self.run / "03-refined.glb").write_bytes(b"stale")
        rec = stage.run_stage3f(self.run, self.client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["status"], "complete")
        self.assertEqual(rec["outputs"], [])
        self.assertEqual(rec["final_model"], "03-source.glb")
        self.assertEqual(final_model(self.run).name, "03-source.glb")
        self.assertFalse((self.run / "03-refined.glb").exists())
        self.assertFalse(any(v["accepted"] for v in rec["views"].values()))
        self.assertIn("all views rejected; keeping 03-source.glb", rec["notes"])
        self.tex.save.assert_not_called()

    def test_empty_front_render_fails_before_segmentation(self):
        self.use_character_setup()
        def empty_probe(mode, args, *unused, **kwargs):
            folder = Path(args["out"])
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (64, 64)).save(folder / "probe0.png")
        character.blender.side_effect = empty_probe
        with self.assertRaisesRegex(ValueError, "front render is empty"):
            stage.run_stage3f(self.run, self.client)
        self.assert_error_record()
        self.client.segment.assert_not_called()

    def test_missing_and_unreadable_front_render_fail(self):
        self.use_character_setup()
        for unreadable in (False, True):
            with self.subTest(unreadable=unreadable):
                def broken_probe(mode, args, *unused, **kwargs):
                    folder = Path(args["out"])
                    folder.mkdir(parents=True, exist_ok=True)
                    if unreadable:
                        (folder / "probe0.png").write_bytes(b"invalid image")
                character.blender.side_effect = broken_probe
                with self.assertRaises(OSError):
                    stage.run_stage3f(self.run, self.client)
                self.assert_error_record()
        self.client.segment.assert_not_called()

    def test_unreadable_source_fails(self):
        self.setup_patch.stop()
        with self.assertRaises(ValueError):
            stage.run_stage3f(self.run, self.client)
        self.assert_error_record()

    def assert_error_record(self):
        rec = json.loads((self.run / "stage-3f.json").read_text())
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["status"], "error")
        self.assertFalse((self.run / "03-refined.glb").exists())

    def test_probe_model_errors_propagate(self):
        self.use_character_setup()
        for error in (ApiError("model failed"), ReplayMiss("no entry")):
            with self.subTest(error=type(error).__name__):
                self.client.segment.side_effect = error
                with self.assertRaises(type(error)):
                    stage.run_stage3f(self.run, self.client)
                self.assert_error_record()

    def test_missing_and_unreadable_cover_fail(self):
        cover = self.run / "01-cover.png"
        cover.unlink()
        with self.assertRaises(FileNotFoundError):
            stage.run_stage3f(self.run, self.client)
        self.assert_error_record()
        cover.write_bytes(b"invalid image")
        with self.assertRaises(OSError):
            stage.run_stage3f(self.run, self.client)
        self.assert_error_record()

    def test_success_records_models_cost_views_and_sequential_textures(self):
        rec = stage.run_stage3f(self.run, self.client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["outputs"], ["03-refined.glb"])
        self.assertEqual(len(rec["views"]), 20)
        self.assertTrue(all(v["accepted"] and v["tries"] == 1 for v in rec["views"].values()))
        self.assertEqual(set(rec["models"]), {IMAGE_MODEL, SAM_MODEL})
        self.assertGreater(rec["cost_usd"], .3)
        self.assertIn("replayed (cached) calls cost 0", rec["cost_basis"])
        self.assertEqual(final_model(self.run), self.run / "03-refined.glb")
        self.assertNotEqual(self.renders[0], self.renders[1])
        self.assertIn("head+0/texture.png", self.renders[1])
        self.assertTrue(all(len(images) == 2 for _, images, _ in self.client.inputs))
        self.assertTrue(all(images[1].getpixel((0, 0)) == (0, 0, 255) for _, images, _ in self.client.inputs))
        self.assertIsNone(self.client.before_api_attempt)

    def test_all_rejected_is_recorded_fallback_with_attempts(self):
        self.client.reject = True
        (self.run / "03-refined.glb").write_bytes(b"stale")
        rec = stage.run_stage3f(self.run, self.client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["outputs"], [])
        self.assertFalse((self.run / "03-refined.glb").exists())
        self.assertEqual(final_model(self.run).name, "03-source.glb")
        self.assertTrue(all(not v["accepted"] and 1 <= v["tries"] <= 3 for v in rec["views"].values()))
        self.assertIn("outline", rec["views"]["head+0"]["reason"])
        self.assertEqual([x[2]["attempt"] for x in self.client.inputs[:3]], [0, 1, 2])

    def test_cap_stops_calls_and_keeps_completed_views(self):
        rec = stage.run_stage3f(self.run, self.client, cap_usd=.025)
        self.assertEqual(rec["status"], "cap_reached")
        self.assertLessEqual(rec["cost_usd"], .025)
        self.assertTrue(rec["views"]["head+0"]["accepted"])
        self.assertFalse(rec["views"]["head+40"]["accepted"])
        self.assertEqual(rec["outputs"], ["03-refined.glb"])
        before = len(self.client.inputs)
        rec2 = stage.run_stage3f(self.run, self.client, cap_usd=.025)
        self.assertEqual(rec2["budget"]["prior_usd"], rec["cost_usd"])
        self.assertEqual(len(self.client.inputs), before + 1)  # blocked before the simulated API call
        self.assertEqual(rec2["cost_usd"], 0)

    def test_model_and_replay_errors_propagate_and_record_spend(self):
        for error in (ApiError("model failed"), ReplayMiss("no entry")):
            with self.subTest(error=type(error).__name__):
                self.client.error = error
                with self.assertRaises(type(error)):
                    stage.run_stage3f(self.run, self.client)
                rec = json.loads((self.run / "stage-3f.json").read_text())
                self.assertFalse(rec["ok"])
                self.assertEqual(rec["status"], "error")
                self.assertEqual(rec["views"]["head+0"]["reason"], type(error).__name__)
                self.assertGreater(rec["cost_usd"], 0)
                self.assertFalse((self.run / "03-refined.glb").exists())

    def test_interrupted_attempt_reservation_survives_a_retry(self):
        self.client.error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            stage.run_stage3f(self.run, self.client, cap_usd=.01)
        interrupted = json.loads((self.run / "stage-3f.json").read_text())
        self.assertEqual(interrupted["budget"]["spent_usd_total"], .01)
        self.assertEqual(interrupted["budget"]["pending_attempt_usd"], .01)
        self.client.error = None
        rec = stage.run_stage3f(self.run, self.client, cap_usd=.01)
        self.assertEqual(rec["status"], "cap_reached")
        self.assertEqual(rec["cost_usd"], 0)
        self.assertEqual(rec["budget"]["prior_usd"], .01)

    def test_answered_call_survives_a_failed_ledger_write(self):
        from herogen.pipeline import client as module
        client = ModelClient(self.run / "live-cache", mode="live", limiter=mock.Mock())
        client.limiter.acquire.return_value = ("slot", 0)
        response = mock.MagicMock(status=200)
        response.__enter__.return_value = response
        response.read.return_value = b'{"data": []}'
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "test-placeholder"}), \
                mock.patch.object(module, "_open", return_value=response), \
                mock.patch.object(client, "_ledger", side_effect=OSError("ledger unavailable")):
            with self.assertRaises(OSError):
                stage.run_stage3f(self.run, client, cap_usd=.01)
        self.assertEqual(client.log.cost_usd(), .01)
        rec = json.loads((self.run / "stage-3f.json").read_text())
        self.assertEqual(rec["budget"]["spent_usd_total"], .01)
        retry = stage.run_stage3f(self.run, self.client, cap_usd=.01)
        self.assertEqual(retry["status"], "cap_reached")
        self.assertEqual(retry["cost_usd"], 0)

    def test_unresolved_reservation_survives_error_without_a_logged_call(self):
        def interrupted(client, *args, **kwargs):
            client.before_api_attempt(IMAGE_MODEL)
            raise OSError("lost answer")
        with mock.patch.object(stage, "_view", side_effect=interrupted):
            with self.assertRaises(OSError):
                stage.run_stage3f(self.run, self.client, cap_usd=.01)
        rec = json.loads((self.run / "stage-3f.json").read_text())
        self.assertEqual(rec["cost_usd"], 0)
        self.assertEqual(rec["budget"]["spent_usd_total"], .01)
        self.assertEqual(rec["budget"]["pending_attempt_usd"], .01)
        retry = stage.run_stage3f(self.run, self.client, cap_usd=.01)
        self.assertEqual(retry["status"], "cap_reached")
        self.assertEqual(retry["budget"]["pending_attempt_usd"], .01)

    def test_service_budget_is_read_even_when_the_local_record_is_lost(self):
        from herogen.refine_budget import update
        from herogen.service.refine_budget import budget_bridge
        durable = {"state": None}
        def operation(action, **kwargs):
            durable["state"] = update(durable["state"], action, **kwargs)
            return durable["state"]
        with budget_bridge(operation) as env, mock.patch.dict(os.environ, env):
            self.client.error = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                stage.run_stage3f(self.run, self.client, cap_usd=.01)
            (self.run / "stage-3f.json").unlink()
            replacement = FakeClient(self.run / "replacement-cache")
            rec = stage.run_stage3f(self.run, replacement, cap_usd=.01)
        self.assertEqual(rec["status"], "cap_reached")
        self.assertEqual(rec["budget"]["prior_usd"], .01)
        self.assertEqual(replacement.log.cost_usd(), 0)

    def test_normal_repaint_cannot_add_hair_to_a_bald_scalp(self):
        scalp = disc((64, 64), 32, 22, 7)
        face = disc((64, 64), 32, 36, 5)
        seen = []
        def segment(image, text):
            seen.append(text)
            mask = face if text == "face" else scalp if text == "head" else np.zeros_like(scalp)
            if text == "hair" and image.getpixel((32, 22)) == (20, 20, 20):
                mask = scalp
            return [SamInstance(mask, (0, 0, 64, 64))]
        def edit(prompt, images, **kwargs):
            image = np.array(images[0])
            image[scalp] = (20, 20, 20)
            return Image.fromarray(image)
        self.client.segment = segment
        self.client.edit_image = edit
        rec = {"attempts": []}
        cam = character.cameras_for(self.info)["head+0"]
        stage._view(self.client, self.run / "prep", self.run / "base.png", self.tex, cam, False,
                    self.run / "view", rec, 900, Image.new("RGB", (64, 64)))
        self.assertTrue(rec["accepted"])
        self.assertIn("hair", seen)
        weight = self.tex.apply.call_args.args[5]
        self.assertEqual(weight[scalp].max(), 0)

    def test_other_profiles_make_no_calls(self):
        (self.run / "03-refined.glb").write_bytes(b"stale")
        rec = stage.run_stage3f(self.run, self.client, profile="preview")
        self.assertEqual(rec["status"], "skipped")
        self.assertIn("roblox-avatar", rec["reason"])
        self.assertFalse((self.run / "03-refined.glb").exists())
        self.assertEqual(self.client.log.calls, [])
        self.tex.apply.assert_not_called()

    def test_entry_invalidates_old_refinement_before_budget_initialisation(self):
        for name in ("03-refined.glb", "03-refined.part.glb"):
            (self.run / name).write_bytes(b"stale")
        with mock.patch.object(stage, "Budget", side_effect=ValueError("budget unavailable")):
            with self.assertRaisesRegex(ValueError, "budget unavailable"):
                stage.run_stage3f(self.run, self.client, profile="roblox-ingame")
        self.assertFalse((self.run / "03-refined.glb").exists())
        self.assertFalse((self.run / "03-refined.part.glb").exists())

    def test_roblox_ingame_refines_with_the_same_default_cap(self):
        rec = stage.run_stage3f(self.run, self.client, profile="roblox-ingame")
        self.assertEqual(rec["status"], "complete")
        self.assertEqual(rec["budget"]["cap_usd"], .8)
        self.tex.apply.assert_called()

    def test_roblox_avatar_refines_with_the_same_default_cap(self):
        rec = stage.run_stage3f(self.run, self.client, profile="roblox-avatar")
        self.assertEqual(rec["status"], "complete")
        self.assertEqual(rec["budget"]["cap_usd"], .8)
        self.tex.apply.assert_called()

    def test_closeup_gate_keeps_face_gate_and_back_of_head_protection(self):
        render_face = disc((64, 64), 30, 34, 6)
        shifted_face = disc((64, 64), 40, 34, 6)
        cams = character.cameras_for(self.info)

        def edit_ok(prompt, images, **kwargs):
            return images[0].copy()

        def segment_shifted_face(image, text):
            if text != "face":
                return []
            # First call on render input gets `render_face`; second call on edit gets `shifted_face`.
            mask = render_face if segment_shifted_face.calls == 0 else shifted_face
            segment_shifted_face.calls += 1
            return [SamInstance(mask, (0, 0, 64, 64))]

        segment_shifted_face.calls = 0
        self.client.edit_image = edit_ok
        self.client.segment = segment_shifted_face
        rec = {"attempts": []}
        stage._view(self.client, self.run / "prep", self.run / "base.png", self.tex, cams["head+0"], False,
                    self.run / "head0", rec, 900, Image.new("RGB", (64, 64)),
                    body_cam=cams["body+0"])
        self.assertFalse(rec.get("accepted", False))
        self.assertEqual(rec["reason"], "new face outside the render face")

        # Back-of-head (head+180) with no render face rejects a painted face as well.
        def segment_back(image, text):
            if text != "face":
                return []
            mask = np.zeros((64, 64), bool) if segment_back.calls == 0 else shifted_face
            segment_back.calls += 1
            return [SamInstance(mask, (0, 0, 64, 64))]

        segment_back.calls = 0
        self.client.segment = segment_back
        back_rec = {"attempts": []}
        stage._view(self.client, self.run / "prep", self.run / "base.png", self.tex, cams["head+180"], False,
                    self.run / "head180", back_rec, 900, Image.new("RGB", (64, 64)),
                    body_cam=cams["body+0"])
        self.assertFalse(back_rec.get("accepted", False))
        self.assertEqual(back_rec["reason"], "new face outside the render face")

    def closeup_artwork(self):
        self.info["lo"] = [-1, -1, 0]
        (self.run / "views").mkdir()
        (self.run / "masks").mkdir()
        (self.run / "views/meta.json").write_text(json.dumps({"height_units": 5.5, "pixels_per_unit": 10}))
        for v in ("front", "back"):
            Image.new("RGB", (64, 64), "red").save(self.run / f"views/{v}.png")
            m = np.zeros((64, 64), np.uint8)
            m[5:60, 10:54] = 255
            Image.fromarray(m).save(self.run / f"masks/{v}.png")

    def test_closeups_follow_twenty_views_with_crop_and_cover(self):
        from herogen.blender.refine import closeups
        self.closeup_artwork()
        with mock.patch.object(closeups, "RES", (64, 64)), \
                mock.patch.object(closeups, "BORDER_PX", 4):
            rec = stage.run_stage3f(self.run, self.client, upscale_texture=True, cap_usd=5)
        self.assertEqual(len(rec["views"]), 36)
        self.assertEqual(len(self.client.inputs), 36)
        self.assertTrue(all(len(images) == 2 for _, images, _ in self.client.inputs[:20]))
        for text, images, _ in self.client.inputs[20:]:
            self.assertEqual(text, prompts.RF5U)
            self.assertEqual(len(images), 3)
            self.assertEqual(images[1].getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(images[2].getpixel((0, 0)), (0, 0, 255))
        for name, view in list(rec["views"].items())[20:]:
            self.assertTrue(view["accepted"])
            self.assertEqual(view["prompt"], "RF5U")
            self.assertIn("crop_box_px", view["references"])
            self.assertEqual(view["camera"]["zoom"], 3)
        self.assertTrue(rec["upscale_texture"])
        # The frame feather must reach the projection, not just the diagnostic image.
        for call in self.tex.apply.call_args_list[20:]:
            feather = call.args[2]
            self.assertEqual(feather[0].max(), 0)
            self.assertEqual(feather[:, -1].max(), 0)
        self.assertIn("texture.png", self.renders[20])

    def test_cap_during_closeups_keeps_completed_views_and_records_pending(self):
        from herogen.blender.refine import closeups
        self.closeup_artwork()
        with mock.patch.object(closeups, "RES", (64, 64)), \
                mock.patch.object(closeups, "BORDER_PX", 4):
            rec = stage.run_stage3f(self.run, self.client, upscale_texture=True, cap_usd=.445)
        self.assertEqual(rec["status"], "cap_reached")
        self.assertTrue(rec["views"]["close-chest+0"]["accepted"])
        self.assertFalse(rec["views"]["close-chest+45"]["accepted"])
        self.assertEqual(rec["views"]["close-waist+0"]["tries"], 0)
        self.assertEqual(rec["outputs"], ["03-refined.glb"])
        self.assertLessEqual(rec["cost_usd"], .445)

    def test_unmeasurable_source_outline_skips_paid_closeup_requests(self):
        cam = cameras.camera("close-test", [0, 0, 1], 0, 1, (64, 64))
        cam.update(zoom=3, frame_border_px=4)
        record = {"accepted": False, "tries": 0, "attempts": []}
        current = self.run / "current.png"

        def render(mode, args, *unused, **kwargs):
            folder = Path(args["out"])
            folder.mkdir(parents=True, exist_ok=True)
            # No contrast with the background: outline measurements cannot be verified.
            Image.new("RGBA", (64, 64), (115, 115, 115, 255)).save(folder / "close-test.png")

        with mock.patch("herogen.blender.refine.backend.blender", side_effect=render):
            result = stage._view(self.client, self.run / "prep", current, self.tex, cam, False,
                                 self.run / "close-test", record, 30, Image.new("RGB", (64, 64)),
                                 closeup_reference=Image.new("RGB", (64, 64)))
        self.assertEqual(result, current)
        self.assertEqual(self.client.inputs, [])
        self.assertEqual(record["tries"], 0)
        self.assertEqual(record["reason"], "render outline not measurable")
        self.assertIn("source_outline_checks", record)
        self.tex.apply.assert_not_called()

    def test_missing_closeup_calibration_fails_before_repaint_calls(self):
        with self.assertRaises(FileNotFoundError):
            stage.run_stage3f(self.run, self.client, upscale_texture=True)
        self.assertEqual(self.client.inputs, [])
        self.assert_error_record()

    def test_closeup_model_failure_is_recorded_and_propagates(self):
        from herogen.blender.refine import closeups
        self.closeup_artwork()
        original = self.client.edit_image

        def edit(text, images, **kwargs):
            if text == prompts.RF5U:
                raise ApiError("model failed")
            return original(text, images, **kwargs)

        self.client.edit_image = edit
        with mock.patch.object(closeups, "RES", (64, 64)), \
                mock.patch.object(closeups, "BORDER_PX", 4):
            with self.assertRaises(ApiError):
                stage.run_stage3f(self.run, self.client, upscale_texture=True, cap_usd=5)
        rec = json.loads((self.run / "stage-3f.json").read_text())
        self.assertEqual(rec["status"], "error")
        self.assertEqual(rec["views"]["close-chest+0"]["reason"], "ApiError")
        self.assertEqual(rec["views"]["close-chest+0"]["tries"], 1)
        self.assert_error_record()

    def test_default_recipe_is_rf2r(self):
        rec = stage.run_stage3f(self.run, self.client)
        self.assertEqual(rec["recipes"], ["RF2R", "RF3H"])
        self.assertEqual(rec["repaint_recipe"], "RF2R")
        sent = [prompt for prompt, _, _ in self.client.inputs]
        self.assertTrue(any("Only make it sharp" in text for text in sent))
        self.assertFalse(any("scratches, pitting, rust" in text for text in sent))

    def test_rf2w_recipe_selects_wear_prompt_and_hair_keeps_rf3h(self):
        rec = stage.run_stage3f(self.run, self.client, recipe="RF2W")
        self.assertEqual(rec["recipes"], ["RF2W", "RF3H"])
        self.assertEqual(rec["repaint_recipe"], "RF2W")
        sent = [prompt for prompt, _, _ in self.client.inputs]
        self.assertTrue(any("scratches, pitting, rust" in text for text in sent))
        self.assertFalse(any("Only make it sharp" in text for text in sent))
        self.assertTrue(all(view["prompt"] in ("RF2W", "RF3H") for view in rec["views"].values()))
        self.assertTrue(any(view["prompt"] == "RF3H" for view in rec["views"].values()))

    def test_original_recipe_remains_selectable(self):
        rec = stage.run_stage3f(self.run, self.client, recipe="RF2R")
        self.assertEqual(rec["recipes"], ["RF2R", "RF3H"])
        self.assertTrue(any("Only make it sharp" in text for text, _, _ in self.client.inputs))

    def test_unknown_recipe_is_rejected_before_any_call(self):
        with self.assertRaises(ValueError):
            stage.run_stage3f(self.run, self.client, recipe="RF9X")
        self.assertEqual(self.client.inputs, [])
        self.assertEqual(self.client.log.calls, [])

    def test_no_source_is_skipped(self):
        (self.run / "03-source.glb").unlink()
        rec = stage.run_stage3f(self.run, self.client)
        self.assertEqual(rec["reason"], "no SAM source mesh")
        self.assertEqual(rec["cost_usd"], 0)

    def test_head_frame_cut_feather_and_exposure_switches(self):
        cams = character.cameras_for(self.info)
        cam_h180 = {**cams["head+180"], "res": [256, 256]}

        def cut_head_render(mode, args, *unused, **kwargs):
            folder = Path(args["out"])
            folder.mkdir(parents=True, exist_ok=True)
            rgba = np.zeros((256, 256, 4), np.uint8)
            rgba[32:, 56:200] = (70, 90, 110, 255)  # neck touches bottom border, crown stays uncut
            Image.fromarray(rgba).save(folder / (args["cams"][0]["name"] + ".png"))

        with mock.patch("herogen.blender.refine.backend.blender", side_effect=cut_head_render), \
                mock.patch.object(cameras, "read_exr", return_value=np.ones((256, 256, 4), np.float32)):
            rec_on = {"attempts": []}
            stage._view(self.client, self.run / "prep", self.run / "base.png", self.tex, cam_h180, False,
                        self.run / "h180_on", rec_on, 900, Image.new("RGB", (256, 256)), body_cam=cams["body+0"])
            f_on = self.tex.apply.call_args.args[2]
            self.assertEqual(float(f_on[-1, 128]), 0.0)
            # At 256px, CLOSEUP_FEATHER_PX=96 gives a 24px ramp, attenuating row -18 where FEATHER_PX=14 is already 1.0.
            self.assertLess(float(f_on[-18, 128]), 0.88)
            self.assertGreater(float(f_on[64, 128]), 0.99)

            with mock.patch.object(refine, "CLOSEUP_FEATHER", False):
                rec_off = {"attempts": []}
                stage._view(self.client, self.run / "prep", self.run / "base.png", self.tex, cam_h180, False,
                            self.run / "h180_off", rec_off, 900, Image.new("RGB", (256, 256)), body_cam=cams["body+0"])
                f_off = self.tex.apply.call_args.args[2]
                self.assertEqual(float(f_off[-18, 128]), 1.0)

        with mock.patch.object(refine, "rebake_views", return_value={"body+0": 0.97, "body+45": 1.03}) as rebake:
            rec = stage.run_stage3f(self.run, self.client)
        self.assertEqual(rebake.call_count, 1)
        self.assertNotIn("exposure_gains", rec)
        self.assertEqual(rec["thresholds"]["closeup_feather_px"], refine.CLOSEUP_FEATHER_PX)
        self.assertEqual(rec["thresholds"]["exposure_mode"], "off")
        self.assertFalse(refine.EXPOSURE_SOLVE)

        with mock.patch.object(refine, "CLOSEUP_FEATHER", False), \
                mock.patch.object(refine, "EXPOSURE_SOLVE", False), \
                mock.patch.object(refine, "rebake_views") as rebake_off:
            rec_disabled = stage.run_stage3f(self.run, self.client)
        rebake_off.assert_not_called()
        self.assertNotIn("exposure_gains", rec_disabled)
        self.assertEqual(rec_disabled["thresholds"]["closeup_feather_px"], 0)
        self.assertEqual(rec_disabled["thresholds"]["exposure_mode"], "off")


class ConsumerTests(unittest.TestCase):
    def test_default_selection_and_roblox_isolation(self):
        from herogen.lab.runindex import bodies, deliverables
        from herogen.service.bundle import glb_path
        from herogen.service.runner import plan_for
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "04-body").mkdir()
            for name in ("04-body/high.glb", "03-source.glb", "03-refined.glb"):
                (run / name).write_bytes(b"synthetic")
            self.assertEqual(final_model(run).name, "03-refined.glb")
            self.assertEqual(glb_path(run).name, "03-refined.glb")
            body = bodies(run)[0]
            self.assertEqual(body["high"], "03-refined.glb")
            self.assertEqual(deliverables(run, body, None)[0]["path"], "03-refined.glb")
            (run / "stage-4.json").write_text('{"profile":{"name":"roblox-ingame"}}')
            self.assertEqual(final_model(run).relative_to(run).as_posix(), "04-body/high.glb")
            self.assertEqual(bodies(run)[0]["high"], "04-body/high.glb")
            self.assertIn("3f", [s.key for s in plan_for("sketch", "best")])
            for profile in ("best", "roblox-ingame", "roblox-avatar"):
                for kind in ("sketch", "prompt"):
                    keys = [s.key for s in plan_for(kind, profile)]
                    self.assertIn("3f", keys)
                    self.assertEqual(keys[keys.index("3"):keys.index("4") + 1], ["3", "3f", "4"])
            self.assertNotIn("3f", [s.key for s in plan_for("sketch", "preview")])


@unittest.skipUnless(HAVE_REFINE and bl.find_blender(), "Blender and texture-refine dependencies are required")
class BlenderRefineTests(unittest.TestCase):
    def test_prep_maps_are_unchanged_by_added_source_normals(self):
        import io
        import math
        from herogen.blender.glb import add_missing_normals, write_glb
        from herogen.blender.refine.backend import blender

        # Split every panel at its UV seams, including the two interior height rings.
        zs = (0.0, 0.25, 0.75, 1.0)
        verts, faces, uvs = [], [], []
        for lo, hi in zip(zs, zs[1:]):
            for k in range(8):
                base = len(verts)
                for corner, z in ((k, lo), (k + 1, lo), (k + 1, hi), (k, hi)):
                    angle = 2 * math.pi * corner / 8
                    verts.append((0.35 * math.cos(angle), 0.35 * math.sin(angle), z))
                    uvs.append((corner / 8, z))
                faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
        image = io.BytesIO()
        Image.new("RGB", (16, 16), (180, 180, 180)).save(image, format="PNG")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = write_glb(root / "raw.glb", [("Body", np.array(verts), np.array(faces), np.array(uvs))],
                            texture_png=image.getvalue())
            source = root / "source.glb"
            self.assertEqual(add_missing_normals(raw, source), 1)
            maps = []
            for name, glb in (("raw", raw), ("normals", source)):
                out = root / name
                blender("prep", {"glb": str(glb), "out": str(out), "tex": 256, "head": None},
                        root / f"{name}.log", timeout=120)
                maps.append(character.texel_maps(out))
            (raw_pos, raw_nrm, raw_cov), (pos, nrm, cov) = maps
            np.testing.assert_array_equal(cov, raw_cov)
            self.assertGreater(int(cov.sum()), 1000)
            np.testing.assert_allclose(pos[cov], raw_pos[cov], atol=1e-6, rtol=0)
            angles = np.degrees(np.arccos(np.clip((nrm[cov] * raw_nrm[cov]).sum(-1), -1, 1)))
            self.assertLess(float(angles.max()), 0.5)
            wall = cov & (pos[..., 2] > 0.30) & (pos[..., 2] < 0.70)
            radial = np.column_stack((pos[wall, :2] / np.linalg.norm(pos[wall, :2], axis=-1, keepdims=True),
                                      np.zeros(int(wall.sum()))))
            angles = np.degrees(np.arccos(np.clip((nrm[wall] * radial).sum(-1), -1, 1)))
            self.assertLess(float(angles.max()), 1.0)

    def test_prep_render_export_preserves_frame_and_base_colour_image(self):
        from herogen.blender.glb import load_glb
        from herogen.blender.refine.backend import blender
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.glb"
            # Give the synthetic cube a real UV layer and embedded base-colour image inside Blender.
            from herogen.blender import bl as runner
            script = root / "source.py"
            script.write_text('''import bpy
from pathlib import Path
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(location=(0.2, -0.3, 0.7), scale=(0.2, 0.3, 0.7))
o = bpy.context.object
m = bpy.data.materials.new("m")
m.use_nodes = True
i = bpy.data.images.new("colour", 32, 32)
i.generated_color = (0.2, 0.4, 0.7, 1)
i.pack()
t = m.node_tree.nodes.new("ShaderNodeTexImage")
t.image = i
m.node_tree.links.new(t.outputs["Color"], m.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
o.data.materials.append(m)
# A real triangle with two corners 5e-7 apart: smoothing the export's normals must not weld it away.
tiny = bpy.data.meshes.new("tiny")
tiny.from_pydata([(0.2, -0.7, 0.7), (0.2000005, -0.7, 0.7), (0.2, -0.7, 0.9)], [], [(0, 1, 2)])
tiny.uv_layers.new()
tiny.materials.append(m)
bpy.context.scene.collection.objects.link(bpy.data.objects.new("tiny", tiny))
bpy.ops.export_scene.gltf(filepath=str(Path(__file__).with_name("source.glb")), export_format="GLB")
''')
            import subprocess
            proc = subprocess.run([runner.find_blender(), "--background", "--factory-startup", "--python-exit-code", "1",
                                   "--python", str(script)], capture_output=True, timeout=120)
            if proc.returncode < 0:
                self.skipTest(f"Blender cannot start in this environment (signal {-proc.returncode})")
            self.assertEqual(proc.returncode, 0, proc.stdout.decode(errors="replace")[-1500:])
            log = root / "blender.log"
            blender("prep", {"glb": str(source), "out": str(root / "prep"), "tex": 4096, "head": None}, log)
            blender("export", {"blend": str(root / "prep/work.blend"), "texture": str(root / "prep/base.png"),
                               "out": str(root / "refined.glb")}, log)
            before, after = load_glb(source), load_glb(root / "refined.glb")
            self.assertEqual(before.triangles, after.triangles)
            verts_before = np.concatenate([m.verts for m in before.meshes])
            verts_after = np.concatenate([m.verts for m in after.meshes])
            for axis in (0, 1, 2):
                np.testing.assert_allclose(np.unique(verts_before[:, axis]), np.unique(verts_after[:, axis]),
                                           atol=1e-6)
            self.assertIn((4096, 4096), after.image_sizes)
            # Smooth normals across the source's split vertices: every copy of a point carries one normal, and a
            # cube corner blends its three faces (no zero component, unlike any one face's axis), so a viewer
            # never lights the faces flat.
            from herogen.blender import glb as G
            doc, binary = G._chunks((root / "refined.glb").read_bytes())
            attrs = doc["meshes"][0]["primitives"][0]["attributes"]
            self.assertIn("NORMAL", attrs)
            pos = G._accessor(doc, binary, attrs["POSITION"]).round(5)
            nrm = G._accessor(doc, binary, attrs["NORMAL"])
            for p in np.unique(pos, axis=0):
                at = nrm[(pos == p).all(axis=1)]
                np.testing.assert_allclose(at, np.repeat(at[:1], len(at), axis=0), atol=1e-3)
                if len(at) >= 3:
                    self.assertGreater(np.abs(at[0]).min(), 0.05)
            cam = cameras.camera("front", [0.2, -0.3, .7], 0, 2, (128, 128))
            blender("render", {"blend": str(root / "prep/work.blend"), "texture": str(root / "prep/base.png"),
                               "cams": [cam], "out": str(root / "render"), "pos": True}, log)
            self.assertTrue((root / "render/front.png").is_file())
            self.assertTrue((root / "render/front.pos.exr").is_file())


class RunnerTests(unittest.TestCase):
    def test_cli_selects_normal_recipe_and_keeps_optional_closeups_off(self):
        import contextlib
        import io
        from herogen.blender.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "stage-3.json").write_text('{"ok":true}')
            for args, expected in (([], "RF2R"), (["--refine-recipe", "RF2R"], "RF2R"),
                                   (["--refine-recipe", "RF2W"], "RF2W")):
                with mock.patch.object(stage, "run_stage3f", return_value={"ok": True}) as called, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["run", "--run", str(run), "--stages", "3f", *args]), 0)
                self.assertEqual(called.call_args.kwargs["recipe"], expected)
                self.assertFalse(called.call_args.kwargs["upscale_texture"])

    def test_cli_default_orders_refine_after_mesh_and_before_body(self):
        from herogen.blender.__main__ import main
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "stage-2.json").write_text('{"ok":true}')
            order = []
            result = {"ok": True, "checks": [], "seconds": 0}
            def called(name):
                def fn(*args, **kwargs):
                    order.append((name, kwargs))
                    return result
                return fn
            with mock.patch("herogen.blender.stage3.run_stage3", side_effect=called("3")), \
                    mock.patch.object(stage, "run_stage3f", side_effect=called("3f")), \
                    mock.patch("herogen.blender.stage4.run_stage4", side_effect=called("4")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", "--run", str(run), "--refine-cap-usd", "0.42"]), 0)
            self.assertEqual([name for name, _ in order], ["3", "3f", "4"])
            self.assertEqual(order[1][1]["cap_usd"], .42)
            order.clear()
            with mock.patch("herogen.blender.stage3.run_stage3", side_effect=called("3")), \
                    mock.patch.object(stage, "run_stage3f", side_effect=called("3f")), \
                    mock.patch("herogen.blender.stage4.run_stage4", side_effect=called("4")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", "--run", str(run), "--profile", "roblox-ingame"]), 0)
            self.assertEqual([name for name, _ in order], ["3", "3f", "4"])

    def test_cli_explicit_stage_selection_for_every_profile(self):
        import contextlib
        import io
        from herogen.blender.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for key in (2, 3):
                (run / f"stage-{key}.json").write_text('{"ok":true}')
            result = {"ok": True, "checks": [], "seconds": 0}
            for profile in ("best", "roblox-ingame", "roblox-avatar"):
                for selection, expected in (("3,4", ["3", "3f", "4"]), ("3", ["3"]),
                                            ("4", ["4"]), ("3f", ["3f"])):
                    with self.subTest(profile=profile, stages=selection):
                        order = []
                        with mock.patch("herogen.blender.stage3.run_stage3",
                                        side_effect=lambda *a, **k: order.append("3") or result), \
                                mock.patch.object(stage, "run_stage3f",
                                                  side_effect=lambda *a, **k: order.append("3f") or result), \
                                mock.patch("herogen.blender.stage4.run_stage4",
                                           side_effect=lambda *a, **k: order.append("4") or result), \
                                contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(main(["run", "--run", str(run), "--profile", profile,
                                                   "--stages", selection]), 0)
                        self.assertEqual(order, expected)

    def test_cli_model_error_is_nonzero_without_traceback(self):
        import contextlib
        import io
        from herogen.blender.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "stage-3.json").write_text('{"ok":true}')
            for error in (ApiError("model failure"), ReplayMiss("cache miss")):
                stderr = io.StringIO()
                with mock.patch.object(stage, "run_stage3f", side_effect=error), contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(["run", "--run", str(run), "--stages", "3f"]), 1)
                self.assertIn(type(error).__name__, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_service_cap_configuration_is_finite_and_defaults_to_best(self):
        from herogen.service.config import ConfigError, from_env
        settings = from_env({}, require_key=False)
        self.assertEqual(settings.default_profile, "best")
        self.assertEqual(settings.refine_cap_usd, .80)
        configured = from_env({"HEROGEN_SERVICE_REFINE_CAP_USD": "0.42"}, require_key=False)
        self.assertEqual(configured.refine_cap_usd, .42)
        for raw in ("nan", "inf", "-1", "bad"):
            with self.assertRaises(ConfigError):
                from_env({"HEROGEN_SERVICE_REFINE_CAP_USD": raw}, require_key=False)

    def test_service_runs_refinement_with_model_key_and_cache(self):
        from herogen.service.runner import PLAN, CliRunner, StageContext
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            ctx = StageContext("j", "sketch", "best", "", "", run, run / "input.png", run / "cache",
                               run / "remote-cache")
            spec = next(s for s in PLAN if s.key == "3f")
            runner = CliRunner("test-placeholder")
            runner._flags["herogen.blender"] = "--profile --refine --refine-cache"
            argv, stdin = runner.command(spec, ctx, {}, run)
            self.assertEqual(argv[argv.index("--stages") + 1], "3f")
            self.assertEqual(argv[argv.index("--refine") + 1], "live")
            self.assertEqual(argv[argv.index("--refine-cache") + 1], str(ctx.cache_dir))
            self.assertIsNone(stdin)
            captured = {}
            def process(*args, **kwargs):
                captured.update(kwargs)
                (run / "stage-3f.json").write_text('{"ok":true}')
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            with mock.patch("herogen.service.runner.subprocess.run", side_effect=process):
                self.assertEqual(runner.run(spec, ctx).state, "passed")
            self.assertEqual(captured["env"]["MODEL_API_KEY"], "test-placeholder")
