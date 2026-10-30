"""Keypoint geometry for stage 2h, on synthetic keypoints of a 3D stick figure seen by exact cameras."""
from __future__ import annotations

import math
import unittest

from herogen.pipeline import keypoints as P

H = 800.0  # figure height, px
BOTTOM = 900.0  # image row of the soles
CX = 512.0


def figure(arm_drop_deg: float = 45.0, stride_deg: float = 0.0, arm_forward_deg: float = 0.0) -> dict:
    """3D joints (x across to the character's left, y depth toward the back, z up), px."""
    j = {"nose": (0, -40, 740), "neck": (0, 0, 680), "left_eye": (14, -32, 755), "right_eye": (-14, -32, 755),
         "left_ear": (34, 0, 745), "right_ear": (-34, 0, 745)}
    for s, sx in (("left", 1), ("right", -1)):
        j[f"{s}_shoulder"] = (sx * 70, 0, 650)
        j[f"{s}_hip"] = (sx * 45, 0, 420)
        leg = math.radians(stride_deg) * (1 if s == "left" else -1)
        for name, drop in (("knee", 210), ("ankle", 390)):
            j[f"{s}_{name}"] = (sx * 45, -math.sin(leg) * drop, 420 - math.cos(leg) * drop)
        ax, ay, az = j[f"{s}_ankle"]
        j[f"{s}_heel"] = (ax, ay + 15, az - 25)
        j[f"{s}_big_toe"] = (ax - sx * 6, ay - 45, az - 28)
        j[f"{s}_small_toe"] = (ax + sx * 10, ay - 38, az - 28)
        a, f = math.radians(arm_drop_deg), math.radians(arm_forward_deg)
        for name, length in (("elbow", 140), ("wrist", 270)):
            out = length * math.cos(a)
            j[f"{s}_{name}"] = (sx * (70 + out * math.cos(f)), -out * math.sin(f), 650 - length * math.sin(a))
    return {k: tuple(float(v) for v in p) for k, p in j.items()}


def view(joints: dict, yaw_deg: float, score: float = 0.9, scale: float = 1.0, shift: float = 0.0) -> P.Keypoints:
    """Image keypoints at ``yaw_deg`` (u = x cos + y sin, rows down from the soles at BOTTOM), at ``scale`` image
    px per figure px."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return {n: (CX + shift + (x * c + y * s) * scale, BOTTOM - z * scale, score)
            for n, (x, y, z) in joints.items()}


def model(joints: dict | None = None, swap_profiles: bool = False) -> P.BodyModel:
    joints = joints or figure()
    left, right = view(joints, 90.0), view(joints, -90.0)
    if swap_profiles:  # the network often swaps a profile's left and right
        left = {(P.pair_name(n) or n): v for n, v in left.items()}
    return P.body_model(view(joints, 0.0), left, right, front_bottom=BOTTOM, front_height=H)


class TestBodyModel(unittest.TestCase):
    def test_depth_from_profiles(self) -> None:
        m = model()
        self.assertAlmostEqual(m.y["nose"] - m.y["neck"], -40.0, places=6)
        self.assertAlmostEqual(m.x["left_shoulder"] - m.x["right_shoulder"], 140.0, places=6)
        self.assertAlmostEqual(m.arm_drop_deg["left"], 45.0, places=6)

    def test_profile_left_right_swap_does_not_change_depth(self) -> None:
        a, b = model(), model(swap_profiles=True)
        for n in P.TURN_JOINTS:
            self.assertAlmostEqual(a.y[n], b.y[n], places=6)

    def test_too_few_joints_is_none(self) -> None:
        f = {"nose": (0.0, 0.0, 0.9)}
        self.assertIsNone(P.body_model(f, f, f, front_bottom=BOTTOM, front_height=H))

    def test_low_scores_are_dropped(self) -> None:
        j = figure()
        front = view(j, 0.0)
        front["nose"] = (front["nose"][0], front["nose"][1], 0.1)
        m = P.body_model(front, view(j, 90.0), view(j, -90.0), front_bottom=BOTTOM, front_height=H)
        self.assertNotIn("nose", m.x)


class TestFitYaw(unittest.TestCase):
    def test_finds_the_yaw_and_its_sign(self) -> None:
        m = model()
        for yaw in (-65.0, -30.0, -25.0, 20.0, 30.0, 45.0, 60.0):
            seen = P.to_uz(view(figure(), yaw, shift=37.0), BOTTOM)
            fit = P.fit_yaw(m, seen)
            self.assertAlmostEqual(fit["yaw_deg"], yaw, delta=0.51, msg=yaw)
            self.assertLess(fit["rms"], 1e-3)
            other = fit["facing_right" if yaw > 0 else "facing_left"]
            self.assertGreater(other["rms"], fit["rms"])

    def test_arms_do_not_move_the_turn(self) -> None:
        m = model()
        for drop in (20.0, 70.0):  # arms raised or lowered from the turnaround's 45 degrees
            seen = P.to_uz(view(figure(arm_drop_deg=drop), 30.0), BOTTOM)
            self.assertAlmostEqual(P.fit_yaw(m, seen)["yaw_deg"], 30.0, delta=0.51)

    def test_scale_maps_view_pixels_to_turnaround_pixels(self) -> None:
        m = model()
        seen = P.to_uz(view(figure(), -60.0, scale=2.0), BOTTOM, scale=0.5)
        self.assertAlmostEqual(P.fit_yaw(m, seen)["yaw_deg"], -60.0, delta=0.51)

    def test_too_few_joints_is_none(self) -> None:
        seen = P.to_uz({"nose": (1.0, 1.0, 0.9)}, BOTTOM)
        self.assertIsNone(P.fit_yaw(model(), seen))


class TestPose(unittest.TestCase):
    def test_kept_pose_has_no_error(self) -> None:
        m = model()
        seen = P.to_uz(view(figure(), 30.0), BOTTOM)
        self.assertLess(P.pose_rms(P.pose_residuals(m, seen, 30.0)), 1e-6)

    def test_raised_arms_and_stride_show_as_error(self) -> None:
        m = model()
        for j in (figure(arm_drop_deg=10.0), figure(stride_deg=20.0), figure(arm_forward_deg=60.0)):
            seen = P.to_uz(view(j, 30.0), BOTTOM)
            yaw = P.fit_yaw(m, seen)["yaw_deg"]
            self.assertGreater(P.pose_rms(P.pose_residuals(m, seen, yaw)), 0.01)

    def test_arm_drop_is_independent_of_the_yaw(self) -> None:
        m = model()
        for drop in (10.0, 45.0, 80.0):
            for yaw in (25.0, 60.0, -60.0):
                seen = P.to_uz(view(figure(arm_drop_deg=drop), yaw), BOTTOM)
                for side in P.SIDES:
                    self.assertAlmostEqual(P.arm_drop_deg(m, seen, side), drop, delta=1e-6)

    def test_arm_drop_needs_both_joints(self) -> None:
        seen = P.to_uz(view(figure(), 30.0), BOTTOM)
        del seen["left_wrist"]
        self.assertIsNone(P.arm_drop_deg(model(), seen, "left"))
        self.assertIsNotNone(P.arm_drop_deg(model(), seen, "right"))


if __name__ == "__main__":
    unittest.main()
