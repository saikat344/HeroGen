"""Ground-truth bench cameras: azimuths, the contract's view axes, the pixel mapping and the frame."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

try:
    import numpy as np

    from herogen.bench import BUILDING_AZIMUTHS, EVALUATION_AZIMUTHS, HEIGHT_STUDS, view_name
    from herogen.bench import cameras as CAM

    HAVE_DEPS = True
except ImportError:  # numpy missing
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class AzimuthTest(unittest.TestCase):
    def test_every_evaluation_view_is_ten_degrees_from_every_building_view(self):
        for e in EVALUATION_AZIMUTHS:
            gap = min(CAM.angular_gap(e, b) for b in BUILDING_AZIMUTHS)
            self.assertGreaterEqual(gap, CAM.MIN_EVAL_SEPARATION_DEG, f"azimuth {e}")

    def test_view_names_and_roles(self):
        self.assertEqual([view_name(a) for a in (0, 90, 180, 270)], ["front", "left", "back", "right"])
        self.assertEqual(view_name(25), "az025")
        self.assertEqual(view_name(335), "az335")
        cams = CAM.all_cameras(80.0, 512)
        self.assertEqual([c["role"] for c in cams].count("building"), 8)
        self.assertEqual([c["role"] for c in cams].count("evaluation"), 4)
        self.assertEqual(len({c["view"] for c in cams}), 12)

    def test_angular_gap_wraps(self):
        self.assertAlmostEqual(CAM.angular_gap(335, 0), 25.0)
        self.assertAlmostEqual(CAM.angular_gap(10, 350), 20.0)


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class AxesTest(unittest.TestCase):
    def test_cardinal_views_match_the_stage3_projection(self):
        from herogen.blender.views import FORWARD, PROJ

        for az, name in ((0, "front"), (90, "left"), (180, "back"), (270, "right")):
            right, up, fwd = CAM.axes(az)
            np.testing.assert_allclose(right, PROJ[name][0], atol=1e-12, err_msg=name)
            np.testing.assert_allclose(up, PROJ[name][1], atol=1e-12, err_msg=name)
            np.testing.assert_allclose(fwd, FORWARD[name], atol=1e-12, err_msg=name)

    def test_axes_are_orthonormal_and_right_handed_for_a_blender_camera(self):
        for az in BUILDING_AZIMUTHS + EVALUATION_AZIMUTHS:
            r, u, f = CAM.axes(az)
            m = np.stack([r, u, f])
            np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
            # a Blender camera looks down its -Z, so (right, up, -forward) must be right-handed
            np.testing.assert_allclose(np.cross(r, u), -f, atol=1e-12)

    def test_camera_sits_on_the_azimuth_direction(self):
        cam = CAM.camera(90, "building", 80.0, 512)
        loc = np.asarray(cam["location"]) - np.asarray(cam["image_centre_world"])
        np.testing.assert_allclose(loc / np.linalg.norm(loc), [1.0, 0.0, 0.0], atol=1e-12)


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class MappingTest(unittest.TestCase):
    def setUp(self):
        self.cam = CAM.camera(25, "evaluation", 80.0, 512)

    def test_feet_and_head_rows(self):
        cr = CAM.project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, HEIGHT_STUDS]]), self.cam)
        self.assertAlmostEqual(cr[0, 1], 256 + HEIGHT_STUDS * 80.0 / 2)
        self.assertAlmostEqual(cr[1, 1], 256 - HEIGHT_STUDS * 80.0 / 2)
        self.assertAlmostEqual(cr[0, 0], 256.0)

    def test_pixel_rays_pass_through_pixel_centres(self):
        rows, cols = np.array([0, 100, 511]), np.array([5, 256, 400])
        origins, fwd = CAM.pixel_rays(self.cam, rows, cols)
        for t in (0.0, 37.5, 60.0):
            cr = CAM.project(origins + t * fwd, self.cam)
            np.testing.assert_allclose(cr[:, 0], cols + 0.5, atol=1e-9)
            np.testing.assert_allclose(cr[:, 1], rows + 0.5, atol=1e-9)

    def test_depth_is_negative_toward_the_camera(self):
        centre = np.asarray(self.cam["image_centre_world"])
        fwd = np.asarray(self.cam["forward"])
        d = CAM.depth_of(np.stack([centre - 0.5 * fwd, centre + 0.25 * fwd]), self.cam)
        np.testing.assert_allclose(d, [-0.5, 0.25], atol=1e-12)


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class FrameTest(unittest.TestCase):
    def test_contract_matrix_puts_feet_at_zero_and_centres_xy(self):
        lo, hi = np.array([1.0, -3.0, 2.0]), np.array([3.0, 1.0, 12.0])
        m = CAM.contract_matrix(lo, hi, 5.5)
        corners = np.array([[x, y, z, 1.0]
                            for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        p = corners @ m.T
        self.assertAlmostEqual(p[:, 2].min(), 0.0)
        self.assertAlmostEqual(p[:, 2].max(), 5.5)
        self.assertAlmostEqual(p[:, 0].min() + p[:, 0].max(), 0.0)
        self.assertAlmostEqual(p[:, 1].min() + p[:, 1].max(), 0.0)

    def test_camera_distance_clears_any_figure(self):
        tall = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 120.0]])
        self.assertGreater(CAM.camera_distance(tall, 120.0), 240.0)
        wide = np.array([[-90.0, 0.0, 0.0], [90.0, 0.0, 5.5]])
        self.assertGreater(CAM.camera_distance(wide, 5.5), 180.0)
        cam = CAM.camera(25, "evaluation", 80.0, 512, 5.5, distance=CAM.camera_distance(wide, 5.5))
        origins, fwd = CAM.pixel_rays(cam, np.array([256]), np.array([256]))
        self.assertLess(float(origins[0] @ fwd), -90.0)   # the ray starts outside the figure

    def test_flat_figure_is_rejected(self):
        with self.assertRaises(ValueError):
            CAM.contract_matrix([0, 0, 1], [1, 1, 1])

    def test_yaw_turns_counter_clockwise(self):
        p = CAM.yaw_matrix(90.0) @ np.array([0.0, -1.0, 0.0, 1.0])
        np.testing.assert_allclose(p[:3], [1.0, 0.0, 0.0], atol=1e-12)

    def test_scale_fills_the_height_unless_a_view_is_too_wide(self):
        tall = np.array([[-0.5, -0.3, 0.0], [0.5, 0.3, HEIGHT_STUDS]])
        self.assertAlmostEqual(CAM.choose_ppu(tall, 512), round(CAM.HEIGHT_FILL * 512 / HEIGHT_STUDS, 4))
        wide = np.array([[-4.0, 0.0, 0.0], [4.0, 0.0, HEIGHT_STUDS]])
        ppu = CAM.choose_ppu(wide, 512)
        self.assertAlmostEqual(ppu, round(CAM.WIDTH_FILL * 512 / 8.0, 4))
        for cam in CAM.all_cameras(ppu, 512):
            cols = CAM.project(wide, cam)[:, 0]
            self.assertTrue(((cols >= 0) & (cols <= 512)).all(), cam["view"])


if __name__ == "__main__":
    unittest.main()
