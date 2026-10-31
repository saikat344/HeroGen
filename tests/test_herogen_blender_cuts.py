"""Cut search rejects unknown gaps instead of treating an absent section or rim as a measurement.

Synthetic inputs only; no Blender is needed.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
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

    def test_touching_vertex_is_not_a_shared_boundary(self):
        # The parent touches one vertex of an opening. A vertex label cannot invent shared mesh edges.
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0]], float)
        f = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
        child = "LeftLowerArm"
        sections = {n: np.zeros((0, 3), np.int64) for n in SECTIONS}
        sections[child] = f
        sections["LeftUpperArm"] = np.array([[0, 4, 5]])
        pair = {0: (split.SID[child], split.SID["LeftUpperArm"]),
                1: (split.SID[child], split.SID["UpperTorso"]),
                2: (split.SID[child], split.SID["LowerTorso"])}
        rig = split.build_rig(self.sk)
        with patch.object(split, "split_mesh", return_value=(rig, None, v, sections, pair, 0)):
            secs, _ = split.build_sections(v, f, self.sk, cap="flat")
        self.assertEqual(secs[child].rims, [])
        self.assertEqual(cuts.unmeasurable_sections(secs, {"LeftElbow"}),
                         {"LeftElbow": "no shared rim between LeftLowerArm and LeftUpperArm"})
        self.assertEqual(len(split.boundary_loops(secs[child].faces)), 0)

    def test_two_neighbour_opening_splits_and_closes_both_caps(self):
        from herogen.blender.score import mesh_sanity

        angles = np.arange(6) * np.pi / 3
        rim = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
        v = np.concatenate([rim, [[0, 0, -1], [0, 0, 1]]])
        f = np.array([[k, (k + 1) % 6, 6] for k in range(6)])
        child, parent, other = "LeftLowerArm", "LeftUpperArm", "UpperTorso"
        sections = {n: np.zeros((0, 3), np.int64) for n in SECTIONS}
        sections[child] = f
        sections[parent] = np.array([[1, 0, 7], [2, 1, 7]])
        pair = {k: (split.SID[child], split.SID[parent if k < 3 else other]) for k in range(6)}
        rig = split.build_rig(self.sk)
        with patch.object(split, "split_mesh", return_value=(rig, None, v, sections, pair, 0)):
            secs, _ = split.build_sections(v, f, replace(self.sk, source="sam3d-body"), cap="flat")
        section = secs[child]
        self.assertTrue(mesh_sanity(section.verts, section.faces, 1e-9)["watertight"])
        self.assertEqual(len(split.boundary_loops(section.faces)), 0)
        points = np.concatenate([r["points"] for r in section.rims if r["neighbour"] == parent])
        np.testing.assert_array_equal(np.unique(points, axis=0), np.unique(rim[:3], axis=0))
        self.assertEqual({r["neighbour"] for r in section.rims}, {parent})
        self.assertFalse(split.is_closed_rim(section.rims[0]))
        self.assertIn("3 vertices/2 edges", cuts.unmeasurable_sections(secs, {"LeftElbow"})["LeftElbow"])

    def test_rim_requires_a_connected_nondegenerate_cycle(self):
        triangle = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        ring = np.array([[0, 1], [1, 2], [2, 0]])
        self.assertTrue(split.is_closed_rim({"points": triangle, "edges": ring}))
        # No tuned size threshold: a small genuine opening is still measurable.
        self.assertTrue(split.is_closed_rim({"points": triangle * 1e-9, "edges": ring}))
        for points, edges in ((triangle[:1], []), (triangle[:2], [[0, 1]]),
                              (triangle, ring[:2]), (triangle, [[0, 1], [1, 0], [2, 2]]),
                              (triangle * [1, 0, 0], ring), (triangle[:1], ring),
                              (np.concatenate([triangle, triangle + 2]), np.concatenate([ring, ring + 3]))):
            with self.subTest(points=points, edges=edges):
                self.assertFalse(split.is_closed_rim({"points": points, "edges": edges}))

    def test_touching_rings_and_dangling_contacts_do_not_erase_real_rims(self):
        child, parent = "LeftLowerArm", "LeftUpperArm"
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        ring_edges = split.boundary_edges(faces)
        loops = {child: split.boundary_loops(faces)}
        for dangling in (False, True):
            edges = ring_edges | ({(0, 5)} if dangling else set())
            boundaries = {n: set() for n in SECTIONS}
            boundaries.update({child: edges, parent: edges})
            rims = split._shared_rims(child, boundaries, points, loops)
            self.assertEqual(sum(split.is_closed_rim(r) for r in rims), 2)
            self.assertEqual(len(rims), 3 if dangling else 2)

    def test_truncated_rim_is_unknown_and_cannot_hide_the_elbow_gap(self):
        from herogen.blender import pose

        child = self.secs["LeftLowerArm"]
        parent = "LeftUpperArm"
        original = next(r for r in child.rims if r["neighbour"] == parent)
        poses = pose.pose_library(pose.bone_directions(self.sk))
        full_gap = pose.joint_gaps(self.secs, poses, joints={"LeftElbow"})
        self.assertEqual(full_gap["arms_up"]["LeftElbow"], 0.0286)
        nearest = np.argmin(np.linalg.norm(original["points"] - child.pivot, axis=1))
        for points, edges in ((original["points"][[nearest]], []),
                              (original["points"][:3], [[0, 1], [1, 2]])):
            truncated = dict(original, points=points, edges=np.asarray(edges, dtype=np.int64))
            secs = dict(self.secs)
            secs[child.name] = replace(child, rims=[truncated])
            self.assertIn("LeftElbow", cuts.unmeasurable_sections(secs))
            with self.assertRaises(cuts.UnmeasurableJoints):
                pose.joint_gaps(secs, poses, joints={"LeftElbow"})
            with patch.object(cuts.split, "build_sections", return_value=(secs, None)):
                with self.assertRaises(cuts.UnmeasurableJoints):
                    self.measure({"LeftElbow"})

    def test_selected_lod_missing_rim_records_a_failure_with_nonempty_sections(self):
        from herogen.blender import stage4

        low = dict(self.secs)
        child = low["LeftLowerArm"]
        low["LeftLowerArm"] = replace(child, rims=[r for r in child.rims if r["neighbour"] != "LeftUpperArm"])
        self.assertGreater(len(low["LeftLowerArm"].faces), 0)
        self.assertGreater(len(low["LeftUpperArm"].faces), 0)
        with tempfile.TemporaryDirectory() as tmp:
            def fail(*args):
                stage4._require_measurable_lods({"high": self.secs, "low": low})

            with patch.object(stage4, "_run_stage4", side_effect=fail):
                rec = stage4.run_stage4(Path(tmp), profile="roblox-ingame")
            self.assertFalse(rec["ok"])
            self.assertEqual(rec["unmeasurable_joints"],
                             {"LeftElbow": "low LOD: no shared rim between LeftLowerArm and LeftUpperArm"})
            self.assertEqual(json.loads((Path(tmp) / "stage-4.json").read_text()), rec)

    def test_selected_lod_point_or_chain_contact_records_a_failure(self):
        from herogen.blender import stage4

        child = self.secs["LeftLowerArm"]
        original = next(r for r in child.rims if r["neighbour"] == "LeftUpperArm")
        for count, edges in ((1, []), (3, [[0, 1], [1, 2]])):
            low = dict(self.secs)
            low[child.name] = replace(child, rims=[dict(original, points=original["points"][:count], edges=edges)])
            with tempfile.TemporaryDirectory() as tmp:
                def fail(*args):
                    stage4._require_measurable_lods({"high": self.secs, "low": low})

                with patch.object(stage4, "_run_stage4", side_effect=fail):
                    rec = stage4.run_stage4(Path(tmp), profile="roblox-ingame")
                self.assertFalse(rec["ok"])
                self.assertIn(f"{count} vertices/{len(edges)} edges", rec["unmeasurable_joints"]["LeftElbow"])
                self.assertIn("low LOD", rec["unmeasurable_joints"]["LeftElbow"])
                self.assertEqual(json.loads((Path(tmp) / "stage-4.json").read_text()), rec)

    def test_unmeasurable_stage_writes_failure_and_replaces_stale_pass(self):
        from herogen.blender import stage4

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "stage-4.json").write_text('{"ok": true}')
            reasons = {"LeftElbow": "empty section LeftUpperArm"}

            def fail(run, rec, *args):
                rec.data["tools"] = ["blender test"]
                raise cuts.UnmeasurableJoints(reasons)

            with patch.object(stage4, "_run_stage4", side_effect=fail):
                rec = stage4.run_stage4(run, profile="roblox-ingame")
            self.assertFalse(rec["ok"])
            self.assertEqual(json.loads((run / "stage-4.json").read_text()), rec)
            self.assertEqual(rec["unmeasurable_joints"], reasons)
            self.assertIn("UnmeasurableJoints", rec["error"])
            self.assertIn("LeftElbow", rec["error"])
            self.assertEqual(rec["outputs"], [])
            self.assertEqual(rec["tools"], ["blender test"])
            self.assertEqual(rec["cost_usd"], 0)
            self.assertEqual(rec["profile"]["name"], "roblox-ingame")
            self.assertFalse(rec["checks"][-1]["pass"])
            from herogen.blender.__main__ import main

            stderr = io.StringIO()
            with patch.object(stage4, "run_stage4", return_value=rec), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                status = main(["run", "--run", str(run), "--stages", "4", "--force"])
            self.assertEqual(status, 1)
            self.assertIn("stage 4 stopped: UnmeasurableJoints", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

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


class TestCarveGeometry(unittest.TestCase):
    def test_sections_match_frozen_silhouette_geometry(self):
        # SHA-256 of every section's vertices, faces, pivot and cap counts at e3d2b4c. Inputs are synthetic;
        # the arm-down and costume variants exercise openings beside both joint and non-joint neighbours.
        fixtures = [
            ({}, ("3629adc9c32ad45a806c2d4795aa127f0d14e9f584ad6db1f2bdfa1a62dccbf0",
                  "a94775b61765f617e8aeccb41cef09daf9084ae9c3b33f94f7aeb68b9b173033")),
            ({"arm_angle_deg": 10}, ("59dc0f811c54b33ea35820c32f8e7ddd8c6aa6f7561d777f43b4a75f3dceab03",
                                     "8e5b20005a86deb3e9da57559b65b57d213a138b9bde63df2d1e2ec0087d8442")),
            ({"clothed": True}, ("dda1ae2ffa6efcd2fce747eaf0a829d43b3c3a80c2e4aee9dffd457415107caf",
                                 "6b14edae715ce7ed01a8640ef279d270d8186ee93344c8d6b010577cc59289d6")),
        ]
        covered_fixtures = [
            ("6ed7608eea11bd88d362433359131923a2d21b64b5994a11c76eb79d09ba5410",
             "3e552a3d08919ea91ebf9dc34082d6020ef6e8f7fcc8b4535f78add5cab860a2"),
            ("316d03c25ac61b4abeab2902554333610ea9434d85e8afa3c28d026b0ecd906c",
             "b01f3a332f6d77ffdb768600ddbaedced88f15e69ec548e8533885210b0afe39"),
            ("6ab8179d873861edc0b5a38aa33abec6f7745a4cb88c956f7d22fc1237c46232",
             "bc9520f20e73a7aa6af754744fd01e35169fbb899a83b4f3dc22b5a0aa3c7865"),
        ]

        def geometry_digest(secs):
            digest = hashlib.sha256()
            for name, section in secs.items():
                digest.update(name.encode())
                for array in (section.verts, section.faces, section.pivot):
                    digest.update(str(array.shape).encode())
                    digest.update(array.astype(array.dtype.newbyteorder("<")).tobytes())
                digest.update(str((section.cap_triangles, section.cap_points_outside)).encode())
            return digest.hexdigest()

        from herogen.blender import covers

        with tempfile.TemporaryDirectory() as tmp:
            for i, (variant, expected) in enumerate(fixtures):
                run = Path(tmp) / str(i)
                synth.write_run(run, synth.humanoid(**variant), size=384, ppu=60.0)
                vs = views.load_views(run, 5.5)
                sk = joints.estimate(vs, 5.5)
                verts, faces, _ = carve.build_mesh(vs, 5.5, 96)
                for cap, frozen, covered_frozen in zip(("flat", "dome"), expected, covered_fixtures[i]):
                    with self.subTest(variant=variant, cap=cap):
                        secs, _ = split.build_sections(verts, faces, sk, cap=cap, rings=1, ring_points=8)
                        self.assertEqual(geometry_digest(secs), frozen)
                        covered, _ = covers.add_covers(secs, verts, faces, owner="UpperTorso", radius=2.5, outset=0.0)
                        self.assertEqual(geometry_digest(covered), covered_frozen)


if __name__ == "__main__":
    unittest.main()
