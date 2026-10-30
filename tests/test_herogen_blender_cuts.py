"""Cut search rejects unknown gaps instead of treating an absent section or rim as a measurement.

Synthetic inputs only; no Blender is needed.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from herogen.blender import PARENT, PIVOT_JOINT, SECTIONS, carve, cuts, joints, split, synth, views


class TestCutMeasurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="herogen-cuts-"))
        synth.write_run(cls.tmp, synth.humanoid(), size=384, ppu=60.0)
        cls.sk = joints.estimate(views.load_views(cls.tmp, 5.5), 5.5)
        cls.verts, cls.faces, _ = carve.build_mesh(views.load_views(cls.tmp, 5.5), 5.5, 96)
        cls.secs, _ = split.build_sections(cls.verts, cls.faces, cls.sk, rings=1, ring_points=8)
        cls.expected = {PIVOT_JOINT[s] for s, p in PARENT.items() if p is not None}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def choose(self, verts=None, faces=None):
        return cuts.choose(self.verts if verts is None else verts, self.faces if faces is None else faces,
                           self.sk, {"rest": {}}, (self.verts, self.faces),
                           {"owner": "UpperTorso", "radius": 2.5, "outset": 0.0}, (1, 8), gate=0.15)

    def measure(self, requested=None, poses=None):
        return cuts.worst_by_joint(self.verts, self.faces, self.sk, {},
                                   {"gate": {"rest": {}} if poses is None else poses},
                                   (self.verts, self.faces),
                                   {"owner": "UpperTorso", "radius": 2.5, "outset": 0.0}, (1, 8), requested)

    def test_named_joint_with_no_section_has_no_measurable_gap(self):
        # Remove the upper arm's geometry, keeping the complete skeleton and the rest of the body.
        lb = split.Labeller(split.build_rig(self.sk))
        labels = lb.labels(self.verts)
        faces = self.faces[~(labels[self.faces] == SECTIONS.index("RightUpperArm")).any(axis=1)]
        secs, _ = split.build_sections(self.verts, faces, self.sk, rings=1, ring_points=8)
        self.assertIn("RightElbow", self.sk.joints)
        self.assertEqual(len(secs["RightUpperArm"].faces), 0)
        self.assertGreater(len(secs["RightLowerArm"].faces), 0)
        with self.assertRaisesRegex(ValueError, "RightElbow: empty section RightUpperArm"):
            self.choose(faces=faces)

    def test_missing_parent_child_and_shared_rim_are_distinct_failures(self):
        for section in ("RightUpperArm", "RightLowerArm", None, "empty-rim"):
            with self.subTest(section=section):
                secs = dict(self.secs)
                if section in ("RightUpperArm", "RightLowerArm"):
                    secs[section] = replace(secs[section], faces=np.zeros((0, 3), np.int64))
                    reason = "empty section " + section
                else:
                    child = secs["RightLowerArm"]
                    rims = [r for r in child.rims if r["neighbour"] != "RightUpperArm"]
                    if section == "empty-rim":
                        rims.append({"neighbour": "RightUpperArm", "joint": "RightElbow",
                                     "points": np.zeros((0, 3))})
                    secs["RightLowerArm"] = replace(child, rims=rims)
                    reason = "no shared rim between RightLowerArm and RightUpperArm"
                with patch.object(cuts.split, "build_sections", return_value=(secs, None)):
                    with self.assertRaisesRegex(ValueError, "RightElbow: " + reason):
                        self.measure({"RightElbow"})

    def test_requested_subset_does_not_require_unrelated_joints(self):
        secs = dict(self.secs)
        secs["RightUpperArm"] = replace(secs["RightUpperArm"], faces=np.zeros((0, 3), np.int64))
        with patch.object(cuts.split, "build_sections", return_value=(secs, None)):
            result = self.measure({"Neck"})
        self.assertEqual(set(result["gate"]), {"Neck"})
        self.assertLess(result["gate"]["Neck"], 0.01)

    def test_measurement_does_not_depend_on_a_pose_named_rest(self):
        result = self.measure(poses={"neutral": {}})
        self.assertEqual(set(result["gate"]), self.expected)
        self.assertLess(max(result["gate"].values()), 0.01)

    def test_candidates_losing_own_or_neighbour_gaps_are_rejected(self):
        for missing in ("Neck", "RightShoulder"):
            for valid_move in (False, True):
                with self.subTest(missing=missing, valid_move=valid_move):
                    zero = {j: 0.01 for j in self.expected}
                    zero["Neck"] = 0.2

                    def measure(verts, faces, sk, offsets, *args):
                        if offsets.get("Neck", 0.0) == 0.0:
                            return {"gate": zero}
                        if not valid_move or offsets["Neck"] < 0.0:
                            raise cuts.UnmeasurableJoints({missing: "no shared rim"})
                        return {"gate": dict(zero, Neck=0.1)}

                    with patch.object(cuts, "FRACTIONS", (-0.5, 0.0, 0.5)), \
                            patch.object(cuts, "worst_by_joint", side_effect=measure):
                        offsets, rec = self.choose()
                    self.assertEqual(rec["eligible"], ["Neck"])
                    self.assertEqual(rec["chosen_fraction"]["Neck"], 0.5 if valid_move else 0.0)
                    self.assertEqual(offsets["Waist"], 0.0)
                    rejected = [t for t in rec["trials"]["torso"]["tried"] if "unmeasurable_joints" in t]
                    self.assertEqual(len(rejected), 1 if valid_move else 2)
                    for trial in rejected:
                        self.assertEqual(trial["unmeasurable_joints"], {missing: "no shared rim"})
                        self.assertNotIn("worst_gap", trial)
                        self.assertNotIn("score", trial)


if __name__ == "__main__":
    unittest.main()
