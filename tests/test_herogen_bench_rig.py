"""Ground-truth bench rigs: limb bones from the naming schemes rigs use, and the A-pose geometry."""

from __future__ import annotations

import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

try:
    import numpy as np

    from herogen.bench import rig as RIG

    HAVE_DEPS = True
except ImportError:  # numpy missing
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class NameTest(unittest.TestCase):
    def test_split_name_across_naming_schemes(self):
        cases = {
            "upperarm.l": ("L", "upperarm"),
            "arm-left": ("L", "arm"),
            "DEF-upper_arm.L.001": ("L", "upper_arm"),
            "mixamorig:LeftArm": ("L", "arm"),
            "mixamorig:RightUpLeg": ("R", "upleg"),
            "Arm_R": ("R", "arm"),
            "arm_upper_L": ("L", "arm_upper"),
            "thigh.R": ("R", "thigh"),
            "Bip01 L UpperArm": ("L", "upperarm"),
            "spine": (None, "spine"),
        }
        for name, want in cases.items():
            self.assertEqual(RIG.split_name(name), want, name)

    def test_limb_roots(self):
        kaykit = ["hips", "upperarm.l", "lowerarm.l", "wrist.l", "upperleg.l", "lowerleg.l", "foot.l"]
        self.assertEqual(RIG.find_limb_root(kaykit, "arm", "L"), "upperarm.l")
        self.assertEqual(RIG.find_limb_root(kaykit, "leg", "L"), "upperleg.l")
        self.assertIsNone(RIG.find_limb_root(kaykit, "arm", "R"))
        self.assertEqual(RIG.find_limb_end(kaykit, "arm", "L"), "wrist.l")

    def test_mixamo_shin_is_not_taken_for_the_thigh(self):
        mixamo = ["mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftForeArm"]
        self.assertEqual(RIG.find_limb_root(mixamo, "leg", "L"), "mixamorig:LeftUpLeg")
        self.assertEqual(RIG.find_limb_end(mixamo, "leg", "L"), "mixamorig:LeftFoot")

    def test_unsegmented_name_wins_over_segments(self):
        rigify = ["DEF-upper_arm.L.001", "DEF-upper_arm.L", "DEF-forearm.L", "DEF-hand.L"]
        self.assertEqual(RIG.find_limb_root(rigify, "arm", "L"), "DEF-upper_arm.L")
        self.assertEqual(RIG.find_limb_end(rigify, "arm", "L"), "DEF-hand.L")

    def test_old_rigify_segment_names(self):
        self.assertEqual(RIG.split_name("DEF-upper_arm.01.L"), ("L", "upper_arm"))
        self.assertEqual(RIG.split_name("MCH-forearm_smth.01.R"), ("R", "forearm_smth"))
        deform = ["DEF-upper_arm.02.L", "DEF-upper_arm.01.L", "DEF-forearm.01.L", "DEF-hand.L",
                  "DEF-thigh.01.L"]
        self.assertEqual(RIG.find_limb_root(deform, "arm", "L"), "DEF-upper_arm.01.L")
        self.assertEqual(RIG.find_limb_root(deform, "leg", "L"), "DEF-thigh.01.L")

    def test_limb_bones_take_every_layer_but_leave_the_shoulder(self):
        names = ["ORG-shoulder.L", "MCH-shoulder_rh_ns_ch.L", "DEF-upper_arm.01.L", "ORG-upper_arm.L",
                 "MCH-forearm_smth.01.L", "DEF-hand.L", "f_index.01.L", "thumb.02.L", "upper_arm.R",
                 "armor_plate.L", "spine", "DEF-thigh.01.L"]
        self.assertEqual(RIG.limb_bones(names, "arm", "L"),
                         ["DEF-upper_arm.01.L", "ORG-upper_arm.L", "MCH-forearm_smth.01.L", "DEF-hand.L",
                          "f_index.01.L", "thumb.02.L"])
        self.assertEqual(RIG.limb_bones(names, "leg", "L"), ["DEF-thigh.01.L"])
        kenney = ["root", "leg-left", "leg-right", "torso", "arm-left", "arm-right", "head"]
        self.assertEqual(RIG.limb_bones(kenney, "arm", "R"), ["arm-right"])


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class GeometryTest(unittest.TestCase):
    def test_targets_point_outward_and_down(self):
        left = RIG.target_direction("arm", "L")
        self.assertGreater(left[0], 0)
        self.assertAlmostEqual(RIG.angle_below_horizontal(left), RIG.A_POSE_ARM_DEG)
        right_leg = RIG.target_direction("leg", "R")
        self.assertLess(right_leg[0], 0)
        self.assertAlmostEqual(RIG.angle_from_vertical(right_leg), RIG.A_POSE_LEG_DEG)
        for limb in ("arm", "leg"):
            for side in "LR":
                self.assertAlmostEqual(float(np.linalg.norm(RIG.target_direction(limb, side))), 1.0)
                self.assertEqual(RIG.target_direction(limb, side)[1], 0.0)

    def test_rotation_between_maps_a_onto_b(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a, b = rng.normal(size=3), rng.normal(size=3)
            r = RIG.rotation_between(a, b)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(r), 1.0)
            np.testing.assert_allclose(r @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)

    def test_rotation_between_parallel_and_opposite(self):
        np.testing.assert_allclose(RIG.rotation_between([1, 0, 0], [2, 0, 0]), np.eye(3), atol=1e-12)
        r = RIG.rotation_between([1, 0, 0], [-1, 0, 0])
        np.testing.assert_allclose(r @ [1, 0, 0], [-1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.0)

    def test_legs_are_aimed_in_the_frontal_plane_only(self):
        leaning = np.array([0.1, -0.4, -1.0])         # out a little, well forward (a big foot), down
        np.testing.assert_allclose(RIG.pose_aim("leg")(leaning), [0.1, 0.0, -1.0])
        np.testing.assert_allclose(RIG.pose_aim("arm")(leaning), leaning)
        self.assertGreater(RIG.sagittal_lean(leaning), 0)  # forward is -Y
        # turning the frontal part onto the target keeps the forward lean
        r = RIG.rotation_between(RIG.pose_aim("leg")(leaning), RIG.target_direction("leg", "L"))
        turned = r @ leaning
        self.assertAlmostEqual(RIG.angle_from_vertical(RIG.frontal(turned)), RIG.A_POSE_LEG_DEG)
        self.assertAlmostEqual(turned[1], leaning[1])

    def test_angle_between(self):
        self.assertAlmostEqual(RIG.angle_between([1, 0, 0], [0, 0, -2]), 90.0)
        self.assertAlmostEqual(RIG.angle_between([1, 0, 0], [3, 0, 0]), 0.0)
        horizontal_out = [1, 0, 0]
        self.assertAlmostEqual(RIG.angle_between(RIG.target_direction("arm", "L"), horizontal_out),
                               RIG.A_POSE_ARM_DEG)

    def test_t_pose_arm_is_horizontal(self):
        self.assertAlmostEqual(RIG.angle_below_horizontal([1.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(RIG.angle_from_vertical([0.0, 0.0, -1.0]), 0.0)
        self.assertAlmostEqual(RIG.angle_below_horizontal([1.0, 0.0, -1.0]), 45.0)
        self.assertAlmostEqual(RIG.angle_from_vertical([math.sin(0.1), 0, -math.cos(0.1)]), math.degrees(0.1))


if __name__ == "__main__":
    unittest.main()
