"""Synthetic camera, artwork-projection and frame-cut outline checks; no model calls."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from herogen.blender.refine import closeups as U, cameras as C, imaging as I, prompts


def info():
    return {"lo": [-.35, -.15, -.5], "hi": [.35, .15, .5], "height": 1.,
            "body_centre": [0, 0, 0], "head_centre": None, "head_side": None}


class CloseupCameraTests(unittest.TestCase):
    def test_four_bands_four_yaws_and_threefold_density(self):
        cams = U.cameras_for(info())
        self.assertEqual(len(cams), 16)
        body = C.camera("body", [0, 0, 0], 0, 1.06, (1024, 1536))
        for cam in cams:
            self.assertAlmostEqual(C.px_per_unit(cam) / C.px_per_unit(body), 3)
            self.assertEqual(cam["frame_border_px"], 96)
            self.assertAlmostEqual(cam["zoom"], body["ortho"] / cam["ortho"])
            self.assertEqual(cam["pixels_per_unit_gain"], 3)
        self.assertEqual({c["angle"] for c in cams}, {0, 45, -45, 180})
        front = [c for c in cams if c["angle"] == 0]
        span = front[0]["res"][1] / C.px_per_unit(front[0])
        for a, b in zip(front, front[1:]):
            overlap = span - (a["centre"][2] - b["centre"][2])
            self.assertGreaterEqual(overlap + 1e-9, 2 * 96 / C.px_per_unit(a))
        self.assertLessEqual(front[-1]["centre"][2] - span / 2, -.5)

    def test_joints_map_contract_units_to_imported_mesh(self):
        joints = {"Waist": [0, 0, 2.9], "LeftShoulder": [.5, 0, 4.3],
                  "RightShoulder": [-.5, 0, 4.3], "LeftKnee": [.3, 0, 1.4],
                  "RightKnee": [-.3, 0, 1.4], "LeftAnkle": [.3, 0, .2], "RightAnkle": [-.3, 0, .2]}
        cams = U.cameras_for(info(), {"joints": joints}, {"height_units": 5.5})
        waist = next(c for c in cams if c["name"] == "close-waist+0")
        self.assertAlmostEqual(waist["centre"][2], -.5 + 2.9 / 5.5)
        self.assertEqual(waist["location_source"], "sam3d-body joints")
        joints["Waist"][2] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            U.cameras_for(info(), {"joints": joints}, {"height_units": 5.5})

    def test_wide_band_contains_both_ankles_inside_its_feather(self):
        # A square 3x-density frame crops away most of the boots on an ordinary stance.
        cam = next(c for c in U.cameras_for(info()) if c["name"] == "close-boots+0")
        points = np.array([[-.16, 0, -.43], [.16, 0, -.43]])
        x, y, _ = C.project(cam, points)
        self.assertTrue(np.all(x > U.BORDER_PX))
        self.assertTrue(np.all(x < cam["res"][0] - U.BORDER_PX))
        self.assertTrue(np.all(y > U.BORDER_PX))
        self.assertTrue(np.all(y < cam["res"][1] - U.BORDER_PX))
        # After image-size normalization, distance limits retain the body's world-space tolerance.
        body = C.camera("body", [0, 0, 0], 0, 1.06, (1024, 1536))
        body_limit = I.GATE["median_px"] * max(body["res"]) / 1024 / C.px_per_unit(body)
        close_limit = I.GATE["median_px"] * cam["zoom"] * max(cam["res"]) / 1024 / C.px_per_unit(cam)
        self.assertAlmostEqual(body_limit, close_limit)

    def test_missing_or_malformed_calibration_has_clear_errors(self):
        for meta in ({}, None, {"height_units": "invalid"}, {"height_units": -1}):
            with self.subTest(meta=meta), self.assertRaisesRegex(ValueError, "height_units"):
                U.contract_scale(info(), meta)
        with self.assertRaisesRegex(ValueError, "joints need"):
            U.cameras_for(info(), {"joints": {}}, {"height_units": 5.5})

    def test_nearest_yaw_wraps_and_diagonal_ties_use_front(self):
        for a, v in [(0, "front"), (45, "front"), (-45, "front"), (80, "right"),
                     (-90, "left"), (180, "back"), (270, "left"), (360, "front")]:
            self.assertEqual(U.nearest_view(a), v)

    def test_box_projects_to_its_exact_camera_frame(self):
        for cam in U.cameras_for(info()):
            x, y, _ = C.project(cam, U.band_box(cam, info()))
            np.testing.assert_allclose([x.min(), x.max(), y.min(), y.max()], [0, cam["res"][0], 0, cam["res"][1]], atol=1e-9)


class CropTests(unittest.TestCase):
    def test_known_projection_with_offset_padding_and_non_square_art(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "views").mkdir()
            (run / "masks").mkdir()
            (run / "views/meta.json").write_text(json.dumps({"pixels_per_unit": 100, "height_units": 5.5}))
            # 550px figure, feet at row 580, off-centre at x=170 in a 400x620 frame.
            mask = np.zeros((620, 400), np.uint8)
            mask[30:580, 100:240] = 255
            for v in U.ART_ANGLES:
                Image.fromarray(mask).save(run / f"masks/{v}.png")
                rgb = np.zeros((620, 400, 3), np.uint8)
                rgb[..., 0] = np.arange(400) % 256
                Image.fromarray(rgb).save(run / f"views/{v}.png")
            front = U.artwork_camera(run, info(), "front")
            x, y, _ = C.project(front, np.array([[0, 0, -.5], [0, 0, .5], [.1, 0, 0]]))
            np.testing.assert_allclose(x, [170, 170, 225])
            np.testing.assert_allclose(y, [580, 30, 305])
            cam = C.camera("close", [0, 0, 0], 0, .2, (1280, 1280))
            self.assertEqual(U.crop_box(cam, front, info()), [115, 250, 225, 360])
            crop, record = U.reference(run, cam, info(), run)
            self.assertEqual(crop.size, (110, 110))
            self.assertEqual(record["artwork"], "views/front.png")
            self.assertTrue((run / "reference-crop.png").is_file())
            back = U.artwork_camera(run, info(), "back")
            x, _, _ = C.project(back, np.array([[.1, 0, 0]]))
            self.assertAlmostEqual(x[0], 115)

    def test_crop_clips_to_image_and_rejects_nonintersection(self):
        art = C.camera("front", [0, 0, 0], 0, 1, (100, 100))
        cam = C.camera("close", [0, 0, .45], 0, .3, (100, 100))
        self.assertEqual(U.crop_box(cam, art, info()), [35, 0, 65, 20])
        cam["centre"][2] = 4
        with self.assertRaisesRegex(ValueError, "intersect"):
            U.crop_box(cam, art, info())


class BorderGateTests(unittest.TestCase):
    def test_frame_cut_grey_surface_does_not_become_background(self):
        fig = np.zeros((512, 768), bool)
        fig[:, 80:688] = True
        rgb = np.full((512, 768, 3), .45, np.float32)
        rgb[fig] = (.6, .4, .2)
        rgb[:360, 280:490] = .45  # grey metal connected to a foreground frame edge
        self.assertFalse(I.closeup_outline(fig, I.figure_mask(rgb), 2.4)[0])
        fixed = I.closeup_figure_mask(rgb, fig)
        self.assertTrue(fixed[20:350, 300:470].all())
        self.assertTrue(I.closeup_outline(fig, fixed, 2.4)[0])

    def test_real_enclosed_background_gaps_are_preserved(self):
        fig = np.zeros((512, 768), bool)
        fig[:, 80:688] = True
        fig[100:400, 130:150] = False  # real narrow arm/torso gap, not a hole in painted surface
        rgb = np.full((512, 768, 3), .45, np.float32)
        rgb[fig] = (.6, .4, .2)
        mask = I.closeup_figure_mask(rgb, fig)
        self.assertFalse(mask[120:380, 132:148].any())
        self.assertTrue(I.closeup_outline(fig, mask, 2.4)[0])

    def test_seeded_background_does_not_lock_the_edited_outline(self):
        fig = np.zeros((512, 768), bool)
        fig[:, 80:688] = True
        rgb = np.full((512, 768, 3), .45, np.float32)
        rgb[:, 180:688] = (.6, .4, .2)  # a large missing strip remains connected to real background
        self.assertFalse(I.closeup_outline(fig, I.closeup_figure_mask(rgb, fig), 2.4)[0])
        rgb[:, :180] = (.6, .4, .2)  # new geometry outside the prior also stays visible to the gate
        self.assertFalse(I.closeup_outline(fig, I.closeup_figure_mask(rgb, fig), 2.4)[0])
        self.assertFalse(I.closeup_figure_mask(np.full_like(rgb, .45), fig).any())

    def test_frame_cut_disagreement_is_excluded(self):
        a = np.zeros((512, 512), bool)
        a[:, 130:380] = True
        b = a.copy()
        b[:8] = False
        b[-8:] = False
        passed, checks = I.closeup_outline(a, b, 3)
        self.assertTrue(passed)
        self.assertEqual(checks["iou"], 1)
        self.assertEqual(checks["p90_px"], 0)

    def test_interior_outline_motion_still_fails(self):
        a = np.zeros((512, 512), bool)
        a[:, 130:380] = True
        b = np.zeros_like(a)
        b[:, 190:440] = True
        self.assertFalse(I.closeup_outline(a, b, 3)[0])

    def test_zoom_scales_distances_without_relaxing_iou(self):
        a = np.zeros((1024, 1024), bool)
        a[:, 200:824] = True
        b = np.zeros_like(a)
        b[:, 216:840] = True
        self.assertFalse(I.closeup_outline(a, b, 1)[0])
        passed, checks = I.closeup_outline(a, b, 3)
        self.assertTrue(passed)
        self.assertEqual(checks["median_limit_px"], 18)
        self.assertEqual(I.GATE, {"iou_min": .90, "median_px": 6., "p90_px": 16.})

    def test_full_surface_and_missing_contours(self):
        a = np.ones((256, 256), bool)
        self.assertTrue(I.closeup_outline(a, a, 3)[0])
        b = a.copy()
        b[80:120, 80:120] = False
        self.assertFalse(I.closeup_outline(a, b, 3)[0])
        self.assertFalse(I.closeup_outline(~a, ~a, 3)[0])

    def test_feather_is_symmetric_zero_at_border_and_flat_inside(self):
        f = I.frame_feather((400, 300), 96)
        self.assertTrue((f[0] == 0).all())
        self.assertTrue((f[:, -1] == 0).all())
        np.testing.assert_allclose(f, f[::-1, ::-1])
        self.assertEqual(f[96, 96], 1)
        self.assertAlmostEqual(f[48, 150], .5)
        self.assertTrue(np.all(np.diff(f[:97, 150]) >= 0))

    def test_recipe_is_registered(self):
        self.assertEqual(prompts.prompt("RF4U", -45), prompts.RF4U)
        self.assertIn("second image", prompts.RF4U)
        self.assertIn("third image", prompts.RF4U)
        self.assertEqual(prompts.prompt("RF5U", 180), prompts.RF5U)
        self.assertIn("cloth stays unbroken cloth", prompts.RF5U)
