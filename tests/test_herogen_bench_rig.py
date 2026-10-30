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
            "J_Bip_L_UpperArm": ("L", "upperarm"),
            "J_Bip_R_UpperLeg": ("R", "upperleg"),
            "J_Bip_C_Head": (None, "head"),
            "Bip001 L UpperArm_010": ("L", "upperarm"),
            "CC_Base_L_Thigh_05": ("L", "thigh"),
            "mixamorig_LeftUpLeg_Armature": ("L", "upleg"),
            "mixamorig:Head_Armature": (None, "head"),
            "Character1_LeftUpLeg_02": ("L", "upleg"),
            "QuickRigCharacter_RightArm": ("R", "arm"),
            "Upper Arm R_7": ("R", "upper_arm"),
            "Head_08": (None, "head"),
        }
        for name, want in cases.items():
            self.assertEqual(RIG.split_name(name), want, name)

    def test_limb_roots(self):
        kaykit = ["hips", "upperarm.l", "lowerarm.l", "wrist.l", "upperleg.l", "lowerleg.l", "foot.l"]
        self.assertEqual(RIG.find_limb_root(kaykit, "arm", "L"), "upperarm.l")
        self.assertEqual(RIG.find_limb_root(kaykit, "leg", "L"), "upperleg.l")
        self.assertIsNone(RIG.find_limb_root(kaykit, "arm", "R"))
        self.assertEqual(RIG.find_limb_end(kaykit, "arm", "L"), "wrist.l")

    def test_vrm_limbs(self):
        vrm = ["J_Bip_C_Hips", "J_Bip_L_Shoulder", "J_Bip_L_UpperArm", "J_Bip_L_LowerArm", "J_Bip_L_Hand",
               "J_Bip_L_Index1", "J_Bip_L_UpperLeg", "J_Bip_L_LowerLeg", "J_Bip_L_Foot", "J_Sec_L_Bust1"]
        self.assertEqual(RIG.find_limb_root(vrm, "arm", "L"), "J_Bip_L_UpperArm")
        self.assertEqual(RIG.find_limb_end(vrm, "arm", "L"), "J_Bip_L_Hand")
        self.assertEqual(RIG.find_limb_root(vrm, "leg", "L"), "J_Bip_L_UpperLeg")
        self.assertEqual(RIG.find_limb_end(vrm, "leg", "L"), "J_Bip_L_Foot")
        self.assertEqual(sorted(RIG.limb_bones(vrm, "arm", "L")),
                         ["J_Bip_L_Hand", "J_Bip_L_Index1", "J_Bip_L_LowerArm", "J_Bip_L_UpperArm"])

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
        self.assertEqual(RIG.limb_bones(kenney, "leg", "L"), ["leg-left"])
        self.assertEqual(RIG.limb_bones(kenney, "arm", "R"), ["arm-right"])

    def test_limb_words_are_whole_words(self):
        names = ["earring.L", "sword_handle.L", "charm.L", "Armature.L", "mixamorig:LeftHandIndex1",
                 "mixamorig:LeftForeArm", "LowerArm_L", "handslot.l", "toes.l", "hips.L"]
        self.assertEqual(RIG.limb_bones(names, "arm", "L"),
                         ["mixamorig:LeftHandIndex1", "mixamorig:LeftForeArm", "LowerArm_L", "handslot.l"])
        self.assertEqual(RIG.limb_bones(names, "leg", "L"), ["toes.l"])
        self.assertEqual(RIG.name_tokens("mixamorig:LeftHandIndex1"),
                         {"mixamorig", "left", "hand", "index", "1"})

    def test_a_root_below_another_root_is_not_turned_twice(self):
        # an arm group whose roots are the upper arm and a forearm helper that hangs (through a bone outside
        # the group) below it: only the upper arm is turned
        parents = {"upper_arm.L": "shoulder.L", "MCH-elbow.L": "upper_arm.L",
                   "forearm_helper.L": "MCH-elbow.L", "hand_ik.L": "root", "shoulder.L": "root", "root": None}
        roots = ["upper_arm.L", "forearm_helper.L", "hand_ik.L"]
        self.assertEqual(RIG.independent_roots(parents, roots), ["upper_arm.L", "hand_ik.L"])


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


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class HeadTest(unittest.TestCase):
    def test_head_bone_names(self):
        cases = [
            ({"Hips": None, "Spine": "Hips", "Neck": "Spine", "Head": "Neck", "HeadTop_End": "Head"}, "Head"),
            ({"mixamorig:Hips": None, "mixamorig:Neck": "mixamorig:Hips", "mixamorig:Head": "mixamorig:Neck",
              "mixamorig:HeadTop_End": "mixamorig:Head"}, "mixamorig:Head"),
            ({"J_Bip_C_Hips": None, "J_Bip_C_Neck": "J_Bip_C_Hips", "J_Bip_C_Head": "J_Bip_C_Neck"},
             "J_Bip_C_Head"),
            ({"root": None, "chest": "root", "head": "chest", "forehead": "head"}, "head"),
            ({"root": None, "DEF-spine": "root", "DEF-spine.005": "DEF-spine",
              "DEF-spine.006": "DEF-spine.005"},
             "DEF-spine.006"),
            ({"root": None, "Forehead": "root", "Eye_L": "root"}, None),
        ]
        for parents, want in cases:
            self.assertEqual(RIG.find_head(list(parents), parents), want, parents)

    def test_a_deforming_head_wins_over_a_control(self):
        parents = {"root": None, "head": "root", "DEF-head": "root", "DEF-jaw": "DEF-head"}
        self.assertEqual(RIG.find_head(list(parents), parents, deform={"DEF-head", "DEF-jaw"}), "DEF-head")

    def test_rigify_head_control_is_not_the_skin(self):
        parents = {"root": None, "head": "root", "DEF-spine.005": "root", "DEF-spine.006": "DEF-spine.005",
                   "ORG-spine.006": "root"}
        self.assertEqual(RIG.find_head(list(parents), parents, deform={"DEF-spine.005", "DEF-spine.006"}),
                         "DEF-spine.006")

    def test_every_head_candidate_in_order(self):
        # a helper beside the real head: both are offered, the caller keeps the one the skin follows
        parents = {"root": None, "neck": "root", "head_ref.x": "neck", "head_scale_fix.x": "neck",
                   "jaw": "head_scale_fix.x"}
        self.assertEqual(RIG.head_candidates(list(parents), parents), ["head_ref.x", "head_scale_fix.x"])
        self.assertEqual(RIG.head_candidates(list(parents), parents, deform={"head_scale_fix.x", "jaw"}),
                         ["head_scale_fix.x", "head_ref.x"])
        self.assertEqual(RIG.head_candidates(["root"], {"root": None}), [])

    def test_head_region_is_the_bone_and_everything_below(self):
        parents = {"Hips": None, "Neck": "Hips", "Head": "Neck", "Jaw": "Head", "Eye.L": "Head",
                   "Hair1": "Jaw", "Hand.L": "Hips"}
        self.assertEqual(sorted(RIG.below(list(parents), parents, "Head")), ["Eye.L", "Hair1", "Head", "Jaw"])

    def test_face_materials(self):
        for name in ("Face", "MTL_Eyes", "EyeBrow", "mouth_inside", "Teeth.001", "M_FaceSkin"):
            self.assertTrue(RIG.is_face_material(name), name)
        for name in ("Surface", "Interface", "Body", "Hair", "Skin", "Eyewear"):
            self.assertFalse(RIG.is_face_material(name), name)


if __name__ == "__main__":
    unittest.main()
