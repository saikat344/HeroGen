"""Rig landmark regressions on generated geometry, with no inference or saved character assets."""
from __future__ import annotations

import contextlib
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from herogen.blender import body_joints, carve, joints, rig_geometry, split, synth, views


@contextlib.contextmanager
def captured_events():
    out = []
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "events.jsonl"
        try:
            with mock.patch.dict(os.environ, {"HEROGEN_EVENTS_FILE": str(path), "HEROGEN_EVENTS_STEP": "4"}):
                yield out
        finally:
            if path.exists():
                out.extend(json.loads(line) for line in path.read_text().splitlines())


class LandmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.figure = synth.humanoid()
        synth.write_run(Path(cls.tmp.name), cls.figure, size=256, ppu=40)
        cls.vs = views.load_views(Path(cls.tmp.name), 5.5)
        cls.verts, cls.faces, _ = carve.build_mesh(cls.vs, 5.5, 80)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.record = {"source": "sam3d-body", "joints": self.figure.joints,
                       "sources": {"sam3d-body": self.figure.joints}}
        self.sk = body_joints.apply_to_skeleton(joints.estimate(self.vs, 5.5), self.record)

    def reconcile(self):
        return rig_geometry.reconcile_skeleton(self.sk, self.record, self.verts, self.faces, self.vs.views["front"])

    def test_consistent_landmarks_are_unchanged(self):
        before = copy.deepcopy(self.sk)
        self.assertEqual(self.reconcile(), {})
        self.assertEqual(before.as_json(), self.sk.as_json())
        for name in before.ends:
            np.testing.assert_array_equal(before.ends[name], self.sk.ends[name])
        self.assertEqual(before.radii, self.sk.radii)

    def test_contradictory_toes_are_preserved(self):
        before = self.verts.copy()
        for side in ("Left", "Right"):
            self.sk.ends[side + "Toe"][1] = .5
        ends = copy.deepcopy(self.sk.ends)
        self.assertEqual(self.reconcile(), {})
        for name in ends:
            np.testing.assert_array_equal(self.sk.ends[name], ends[name])
        np.testing.assert_array_equal(self.verts, before)

    def test_a_staff_endpoint_is_replaced_by_the_actual_hand(self):
        self.sk.ends["RightHandTip"] = np.array([0., 0., .8])
        self.sk.radii["RightArm"] = .9
        changes = self.reconcile()
        self.assertEqual(set(changes), {"RightHandTip"})
        self.assertLess(self.sk.ends["RightHandTip"][0], self.sk.joints["RightWrist"][0])
        self.assertLess(self.sk.radii["RightArm"], .25)
        self.assertGreater(self.sk.radii["RightArm"], .05)

    def test_crossed_hips_and_low_shoulders_use_recorded_anatomy(self):
        self.sk.joints["RightHip"] = np.array([.8, .4, 2.8])
        self.sk.joints["LeftHip"] = np.array([1., 0., 2.8])
        self.sk.joints["RightShoulder"] = np.array([-.1, 0., 2.9])
        changes = self.reconcile()
        self.assertEqual(set(changes), {"LeftHip", "RightHip", "RightShoulder"})
        self.assertLess(self.sk.joints["RightHip"][0], self.sk.joints["Root"][0])
        self.assertGreater(self.sk.joints["RightShoulder"][2], self.sk.joints["RightElbow"][2])
        np.testing.assert_allclose(self.sk.ends["HipCentre"], (self.sk.joints["LeftHip"] + self.sk.joints["RightHip"]) / 2)

    def test_shoulder_on_a_scarf_outside_the_elbow_uses_recorded_anatomy(self):
        self.sk.joints["LeftShoulder"][0] = self.sk.joints["LeftElbow"][0] + .3
        changes = self.reconcile()
        self.assertEqual(set(changes), {"LeftShoulder"})
        np.testing.assert_array_equal(self.sk.joints["LeftShoulder"], self.figure.joints["LeftShoulder"])

    def test_upward_scarf_tip_follows_the_actual_downward_hand_surface(self):
        wrist = self.sk.joints["LeftWrist"]
        self.sk.ends["LeftHandTip"] = wrist + [.4, 0, .3]
        changes = self.reconcile()
        self.assertEqual(set(changes), {"LeftHandTip"})
        self.assertLess(self.sk.ends["LeftHandTip"][2], wrist[2])

    def test_ribbon_radius_is_remeasured_at_the_recorded_forearm(self):
        self.sk.radii["RightArm"] = .001
        before = self.sk.as_json()
        changes = self.reconcile()
        self.assertEqual(set(changes), {"RightArmRadius"})
        self.assertGreater(self.sk.radii["RightArm"], .1)
        self.assertEqual(self.sk.as_json(), before)

    def test_each_repair_is_a_recovery_event_with_its_reason_and_distance(self):
        self.sk.joints["RightHip"] = np.array([.8, .4, 2.8])
        self.sk.joints["LeftHip"] = np.array([1., 0., 2.8])
        self.sk.joints["RightShoulder"] = np.array([-.1, 0., 2.9])
        self.sk.ends["LeftHandTip"] = self.sk.joints["LeftWrist"] + [.4, 0, .3]
        with captured_events() as got:
            changes = self.reconcile()
        self.assertEqual([(e["kind"], e["what"], e["why"]) for e in got], [
            ("recover", "landmark_left_hip", "hips_cross_pelvis"),
            ("recover", "landmark_right_hip", "hips_cross_pelvis"),
            ("recover", "landmark_right_shoulder", "shoulder_elbow_order"),
            ("recover", "landmark_left_hand_tip", "hand_direction")])
        for e in got:
            name = e["what"].removeprefix("landmark_").title().replace("_", "")
            self.assertAlmostEqual(e["m"]["distance_studs"], changes[name]["distance_studs"], places=4)

    def test_a_remeasured_arm_radius_reports_both_radii(self):
        self.sk.radii["RightArm"] = .001
        with captured_events() as got:
            changes = self.reconcile()
        self.assertEqual([(e["what"], e["why"]) for e in got], [("landmark_right_arm_radius", "radius_below_forearm")])
        self.assertEqual(got[0]["m"], {"before_studs": .001,
                                       "after_studs": round(changes["RightArmRadius"]["after_studs"], 4)})

    def test_consistent_landmarks_log_nothing(self):
        with captured_events() as got:
            self.assertEqual(self.reconcile(), {})
        self.assertEqual(got, [])

    def test_missing_anatomical_alternative_is_loud(self):
        self.sk.joints["RightShoulder"] = np.array([0., 0., 1.])
        self.record.pop("sources")
        with self.assertRaisesRegex(ValueError, "RightShoulder.*without a recorded anatomical"):
            self.reconcile()

    def test_mirrored_anatomical_hips_are_not_relabelled(self):
        mirrored = copy.deepcopy(self.figure.joints)
        for name, point in mirrored.items():
            mirrored[name] = [-point[0], point[1], point[2]]
        self.record["sources"]["sam3d-body"] = mirrored
        self.sk.joints["LeftHip"] *= [-1, 1, 1]
        self.sk.joints["RightHip"] *= [-1, 1, 1]
        with self.assertRaisesRegex(ValueError, "anatomical hips do not straddle"):
            self.reconcile()

    def test_a_silhouette_only_skeleton_is_not_treated_as_a_hybrid(self):
        self.sk.joints["RightShoulder"] = np.array([-.7, 0., 2.9])
        before = self.sk.as_json()
        self.assertEqual(rig_geometry.reconcile_skeleton(self.sk, None, self.verts, self.faces,
                                                        self.vs.views["front"]), {})
        self.assertEqual(self.sk.as_json(), before)



def box(name, lo, hi, pivot):
    verts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return split.Section(name, verts, faces, np.array(pivot, float), [], 0)


class SleeveSupportTests(unittest.TestCase):
    def support(self, sections, a=(0, 0, 0), b=(1, 0, 0)):
        verts = np.concatenate([s.verts for s in sections])
        offsets = np.cumsum([0] + [len(s.verts) for s in sections[:-1]])
        faces = np.concatenate([s.faces + offset for s, offset in zip(sections, offsets)])
        before = verts.copy(), faces.copy()
        result = rig_geometry.sleeve_support(verts, faces, np.array(a, float), np.array(b, float))
        np.testing.assert_array_equal(verts, before[0])
        np.testing.assert_array_equal(faces, before[1])
        return result

    def test_hollow_sleeve_supports_a_bone_in_its_empty_centre(self):
        outer = box("outer", [0, -.2, -.2], [1, .2, .2], [0, 0, 0])
        inner = box("inner", [0, -.15, -.15], [1, .15, .15], [0, 0, 0])
        inner.faces = inner.faces[:, ::-1]
        result = self.support([outer, inner])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["centre"], [.5, 0, 0], atol=1e-12)

    def test_displaced_sleeve_has_local_support(self):
        sleeve = box("sleeve", [0, .1, -.1], [1, .3, .1], [0, 0, 0])
        result = self.support([sleeve])
        np.testing.assert_allclose(result["centre"], [.5, .2, 0], atol=1e-12)

    def test_a_remote_prop_is_not_a_missing_arm(self):
        prop = box("prop", [0, 2, -.1], [1, 2.2, .1], [0, 0, 0])
        self.assertIsNone(self.support([prop]))

    def test_a_large_torso_is_not_local_sleeve_support(self):
        torso = box("torso", [0, -1, -1], [1, 1, 1], [0, 0, 0])
        self.assertIsNone(self.support([torso]))

    def test_an_open_surface_is_not_a_closed_sleeve(self):
        sleeve = box("sleeve", [0, -.2, -.2], [1, .2, .2], [0, 0, 0])
        sleeve.faces = sleeve.faces[:-2]
        self.assertIsNone(self.support([sleeve]))

    def test_zero_length_segment_cannot_supply_support(self):
        sleeve = box("sleeve", [0, -.2, -.2], [1, .2, .2], [0, 0, 0])
        self.assertIsNone(self.support([sleeve], b=(0, 0, 0)))


class PivotTests(unittest.TestCase):
    def test_existing_valid_pivots_are_exactly_preserved(self):
        section = box("LowerTorso", [0, 0, 0], [1, 1, 1], [.5, .5, .5])
        before = section.pivot.copy()
        self.assertEqual(rig_geometry.fit_pivots({"LowerTorso": section}), {})
        np.testing.assert_array_equal(section.pivot, before)

    def test_outside_child_pivot_moves_to_nearest_shared_edge(self):
        parent = box("UpperTorso", [-1, 0, 0], [0, 1, 1], [-.5, .5, .5])
        child = box("LeftUpperArm", [0, 0, 0], [1, 1, 1], [-.2, .5, 0])
        changes = rig_geometry.fit_pivots({"UpperTorso": parent, "LeftUpperArm": child})
        np.testing.assert_allclose(child.pivot, [0, .5, 0])
        self.assertAlmostEqual(changes["LeftUpperArm"]["distance_studs"], .2)

    def test_outside_parent_also_uses_the_common_cut(self):
        parent = box("UpperTorso", [-1, 0, 0], [0, 1, 1], [-.5, .5, .5])
        child = box("LeftUpperArm", [0, 0, 0], [1, 1, 1], [.5, .5, 0])
        rig_geometry.fit_pivots({"UpperTorso": parent, "LeftUpperArm": child})
        np.testing.assert_allclose(child.pivot, [0, .5, 0])

    def test_shared_edge_fit_preserves_a_reflected_foot_cut(self):
        parent = box("LeftLowerLeg", [-1, -.2, 1], [1, .3, 2], [0, 0, 1.5])
        child = box("LeftFoot", [-1, -.2, 0], [1, .3, 1], [0, .6, 1])
        original = {"LeftLowerLeg": parent, "LeftFoot": child}
        reflected = copy.deepcopy(original)
        for section in reflected.values():
            section.verts[:, 1] *= -1
            section.pivot[1] *= -1
            section.faces = section.faces[:, ::-1]
        rig_geometry.fit_pivots(original)
        rig_geometry.fit_pivots(reflected)
        np.testing.assert_allclose(reflected["LeftFoot"].pivot, child.pivot * [1, -1, 1])

    def test_disconnected_parts_preserve_and_record_the_section_pivot(self):
        parent = box("UpperTorso", [-2, 0, 0], [-1, 1, 1], [-1.5, .5, .5])
        child = box("LeftUpperArm", [0, 0, 0], [1, 1, 1], [-.5, .5, 0])
        before = child.pivot.copy()
        changes = rig_geometry.fit_pivots({"UpperTorso": parent, "LeftUpperArm": child})
        np.testing.assert_array_equal(child.pivot, before)
        self.assertEqual(changes["LeftUpperArm"], {
            "source": "section pivot", "distance_studs": 0.0,
            "pivot_fallback": "no shared cut edge with UpperTorso; retained section pivot"})

    def test_disconnected_part_inside_its_parent_still_records_the_fallback(self):
        parent = box("UpperTorso", [-2, -2, -2], [2, 2, 2], [0, -2, 0])
        child = box("Head", [-.5, -.5, -.5], [.5, .5, .5], [0, 0, 0])
        before = child.pivot.copy()
        changes = rig_geometry.fit_pivots({"UpperTorso": parent, "Head": child})
        np.testing.assert_array_equal(child.pivot, before)
        self.assertEqual(changes["Head"], {
            "source": "section pivot", "distance_studs": 0.0,
            "pivot_fallback": "no shared cut edge with UpperTorso; retained section pivot"})

    def test_root_falls_on_its_own_section_surface(self):
        section = box("LowerTorso", [0, 0, 0], [1, 1, 1], [-.5, .5, .5])
        changes = rig_geometry.fit_pivots({"LowerTorso": section})
        np.testing.assert_allclose(section.pivot, [0, .5, .5])
        self.assertEqual(changes["LowerTorso"]["distance_studs"], .5)


class SharedRimTests(unittest.TestCase):
    def test_a_robe_opening_uses_each_hips_own_rim(self):
        figure = synth.humanoid()
        # A wide robe joins the legs. Its opening also borders the other leg and the torso, with points
        # below the hip plane. Those points cannot contribute to either hip's planar area centroid.
        figure.parts.append(synth.RBox((0, 0, 2.), (1., .4, .9), .08))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            synth.write_run(run, figure, size=256, ppu=40)
            vs = views.load_views(run, 5.5)
            sk = body_joints.apply_to_skeleton(joints.estimate(vs, 5.5), {"joints": figure.joints})
            verts, faces, _ = carve.build_mesh(vs, 5.5, 80)
            sections, _ = split.build_sections(verts, faces, sk, ring_points=12)
        # Each half of the roughly rectangular robe has its hip cut centred at half the half-width,
        # rather than at the robe's outside edge. The tolerance covers rounded corners and voxel spacing.
        np.testing.assert_allclose(sections["LeftUpperLeg"].pivot[:2], [.5, 0.], atol=.08)
        np.testing.assert_allclose(sections["RightUpperLeg"].pivot[:2], [-.5, 0.], atol=.08)


if __name__ == "__main__":
    unittest.main()
