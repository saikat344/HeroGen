"""Rig landmark regressions on generated geometry, with no inference or saved character assets."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from herogen.blender import body_joints, carve, joints, rig_geometry, split, synth, views


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
