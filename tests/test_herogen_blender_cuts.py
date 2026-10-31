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
        self.assertEqual(cuts.unmeasurable_sections(secs, {"LeftElbow"}), {})
        np.testing.assert_array_equal(split.rim_measurement_vertices(section.rims[0], section.cut_radius), [0, 1, 2])

    def test_rim_requires_a_nondegenerate_cycle(self):
        triangle = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        ring = np.array([[0, 1], [1, 2], [2, 0]])
        self.assertTrue(split.is_closed_rim({"points": triangle, "edges": ring}))
        # Closure has no size threshold; gap measurability uses the whole joint boundary length.
        self.assertTrue(split.is_closed_rim({"points": triangle * 1e-9, "edges": ring}))
        for points, edges in ((triangle[:1], []), (triangle[:2], [[0, 1]]),
                              (triangle, ring[:2]), (triangle, [[0, 1], [1, 0], [2, 2]]),
                              (triangle * [1, 0, 0], ring), (triangle[:1], ring)):
            with self.subTest(points=points, edges=edges):
                self.assertFalse(split.is_closed_rim({"points": points, "edges": edges}))

    def test_multiple_loops_and_junctions_keep_all_cyclic_vertices(self):
        child, parent = "LeftLowerArm", "LeftUpperArm"
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0],
                           [0, -1, 0], [0, -2, 0]])
        cases = [
            # Disconnected openings, a repeated/shared vertex, and a theta graph with shared edges.
            ([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], [0, 1, 2, 3, 4, 5]),
            ([(0, 1), (1, 2), (0, 2), (0, 3), (3, 5), (0, 5)], [0, 1, 2, 3, 5]),
            ([(0, 1), (1, 2), (0, 2), (1, 5), (2, 5)], [0, 1, 2, 5]),
            # A T-junction's dangling chain and a collinear cycle contribute no opening samples.
            ([(0, 1), (1, 2), (0, 2), (0, 6), (6, 7), (0, 7)], [0, 1, 2]),
            ([(0, 1), (1, 2), (0, 2), (0, 6), (6, 7)], [0, 1, 2]),
        ]
        for edges, expected in cases:
            with self.subTest(edges=edges):
                rim = {"points": points, "edges": np.array(edges)}
                self.assertTrue(split.is_closed_rim(rim))
                np.testing.assert_array_equal(split.rim_loop_vertices(rim), expected)
                boundaries = {n: set() for n in SECTIONS}
                boundaries[child] = boundaries[parent] = {tuple(sorted(e)) for e in edges}
                rims = split._shared_rims(child, boundaries, points)
                samples = np.concatenate([r["points"][split.rim_loop_vertices(r)] for r in rims])
                np.testing.assert_array_equal(np.unique(samples, axis=0), np.unique(points[expected], axis=0))

    def test_joint_gap_uses_every_loop_but_no_isolated_vertex(self):
        from herogen.blender import pose

        child = self.secs["LeftLowerArm"]
        original = next(r for r in child.rims if r["neighbour"] == "LeftUpperArm")
        points = original["points"][[0, 7, 14, 21, 28, 35]]
        ring = np.array([[0, 1], [1, 2], [2, 0]])
        first = dict(original, points=points[:3], edges=ring)
        second = dict(original, points=points[3:], edges=ring)
        multiple = dict(original, points=points, edges=np.concatenate([ring, ring + 3]))
        # A very distant isolated contact would dominate the gap if it leaked into the samples.
        branch = dict(multiple, points=np.concatenate([points, [[100., 100, 100]]]),
                      edges=multiple["edges"])
        poses = pose.pose_library(pose.bone_directions(self.sk))

        def measure(rim):
            secs = dict(self.secs)
            secs[child.name] = replace(child, rims=[rim], cut_radius=0.01)
            return pose.joint_gaps(secs, poses, joints={"LeftElbow"})

        a, b = measure(first), measure(second)
        expected = {p: {"LeftElbow": max(a[p]["LeftElbow"], b[p]["LeftElbow"])} for p in poses}
        self.assertNotEqual(a, b)
        self.assertEqual(measure(multiple), expected)
        self.assertEqual(measure(branch), expected)

    def test_open_chains_use_arc_length_and_one_cut_radius_at_every_scale(self):
        points = np.array([[0., 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
        for scale in (1e-9, 1.0, 1e9):
            rim = {"points": points * scale, "edges": [[0, 1], [1, 2]]}
            chains = split.rim_open_chains(rim)
            self.assertEqual(len(chains), 1)
            self.assertEqual(chains[0][1], scale)
            for radius, expected in ((scale, [0, 1, 2]), (scale * 1.01, []), (scale * 0.99, [0, 1, 2])):
                np.testing.assert_array_equal(split.rim_measurement_vertices(rim, radius), expected)
        # One real edge is a chain; coincident points and isolated vertices have zero length.
        for points, edges, expected in (([[0., 0, 0], [1, 0, 0]], [[0, 1]], [0, 1]),
                                        ([[0., 0, 0], [0, 0, 0]], [[0, 1]], []),
                                        ([[0., 0, 0]], [], [])):
            np.testing.assert_array_equal(split.rim_measurement_vertices({"points": points, "edges": edges}, 1),
                                          expected)

    def test_boundary_length_pools_loops_chains_components_and_branches(self):
        points = np.array([[0., 0, 0], [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75]])
        for edges in ([[0, 1], [0, 2], [0, 3]], [[0, 1], [2, 3]]):
            rim = {"points": points, "edges": edges}
            # Each chain is shorter than 1.5; both totals lie between 1.5 and 3.
            np.testing.assert_array_equal(split.rim_measurement_vertices(rim, 1.5), np.arange(4))
            np.testing.assert_array_equal(split.rim_measurement_vertices(rim, 3.), [])
        # Short bridges and a collinear cycle are still real shared edges of the same joint.
        rim = {"points": np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],
                                    [-0.75, 0, 0], [0, -0.75, 0]]),
               "edges": [[0, 1], [1, 2], [2, 0], [0, 3], [0, 4]]}
        np.testing.assert_array_equal(split.rim_measurement_vertices(rim, 1.0), np.arange(5))
        rim = {"points": points * [1, 0, 0], "edges": [[0, 1], [1, 2], [2, 0]]}
        np.testing.assert_array_equal(split.rim_measurement_vertices(rim, 0.1), [0, 1, 2])

    def test_joint_total_includes_short_components_but_never_other_joints_or_lods(self):
        from herogen.blender import pose, stage4

        child = self.secs["LeftLowerArm"]
        original = next(r for r in child.rims if r["neighbour"] == "LeftUpperArm")
        # Neither disconnected chain qualifies alone; together they reach exactly one radius.
        rims = [dict(original, points=np.array([[0., 0, z], [0.5, 0, z]]), edges=[[0, 1]])
                for z in (0., 1.)]
        secs = dict(self.secs)
        secs[child.name] = replace(child, rims=rims, cut_radius=1.)
        self.assertTrue(all(not len(split.rim_measurement_vertices(r, 1.)) for r in rims))
        self.assertEqual(len(split.rim_measurements(rims, 1.)), 2)
        self.assertEqual(cuts.unmeasurable_sections(secs, {"LeftElbow"}), {})
        pose.joint_gaps(secs, {"rest": {}}, joints={"LeftElbow"})
        record = cuts.rim_topology(secs)["LeftElbow"]
        self.assertEqual((record["length_studs"], record["length_over_radius"], record["piece_count"]),
                         (1., 1., 2))
        self.assertEqual(record["boundary_type"], "open_chain")
        self.assertTrue(all(c["measurable"] for c in record["components"]))
        for radius, contacts in ((1.01, rims), (1., [rims[0], dict(rims[1], neighbour="UpperTorso")])):
            low = dict(secs)
            low[child.name] = replace(child, rims=contacts, cut_radius=radius)
            self.assertIn("LeftElbow", cuts.unmeasurable_sections(low))
            with self.assertRaises(cuts.UnmeasurableJoints):
                stage4._require_measurable_lods({"high": secs, "low": low})
        # A loop also needs one radius in total; an isolated contact adds no length or samples.
        triangle = dict(original, points=np.array([[0., 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
                        edges=[[0, 1], [1, 2], [2, 0]])
        self.assertEqual(split.rim_measurements([triangle], 1.), [])
        contact = dict(original, points=np.array([[100., 100, 100]]), edges=[])
        self.assertEqual(len(split.rim_measurements(rims + [contact], 1.)), 2)

    def test_gap_uses_every_qualifying_chain_and_loop(self):
        from herogen.blender import pose

        child = self.secs["LeftLowerArm"]
        original = next(r for r in child.rims if r["neighbour"] == "LeftUpperArm")
        points = original["points"][[0, 7, 14, 21, 28, 35]]
        chain = [[0, 1], [1, 2]]
        loop = chain + [[2, 0]]
        poses = pose.pose_library(pose.bone_directions(self.sk))

        def measure(rims):
            secs = dict(self.secs)
            secs[child.name] = replace(child, rims=rims, cut_radius=0.01)
            return pose.joint_gaps(secs, poses, joints={"LeftElbow"})

        first = dict(original, points=points[:3], edges=loop)
        second = dict(original, points=points[3:], edges=chain)
        third = dict(original, points=points[:3], edges=chain)
        a, b, c = measure([first]), measure([second]), measure([third])
        expected = {p: {"LeftElbow": max(a[p]["LeftElbow"], b[p]["LeftElbow"], c[p]["LeftElbow"])} for p in poses}
        self.assertEqual(measure([first, second, third]), expected)
        mixed = dict(original, points=points, edges=loop + [[0, 3], [3, 4], [4, 5]])
        ids = split.rim_measurement_vertices(mixed, 0.01)
        np.testing.assert_array_equal(ids, np.arange(6))

    def test_chain_metadata_records_qualified_short_and_absent_boundaries(self):
        child = self.secs["LeftLowerArm"]
        original = next(r for r in child.rims if r["neighbour"] == "LeftUpperArm")
        for length, kind in ((0.5, "none"), (1.0, "open_chain"), (2.0, "open_chain")):
            secs = dict(self.secs)
            rim = dict(original, points=np.array([[0., 0, 0], [length, 0, 0]]), edges=[[0, 1]])
            secs[child.name] = replace(child, rims=[rim], cut_radius=1.0)
            record = cuts.rim_topology(secs)["LeftElbow"]
            self.assertEqual(record["boundary_type"], kind)
            self.assertEqual(record["components"][0]["open_chains"][0]["length_over_radius"], length)
            self.assertEqual(bool(cuts.unmeasurable_sections(secs, {"LeftElbow"})), length < 1)
        secs[child.name] = replace(child, rims=[])
        self.assertEqual(cuts.rim_topology(secs)["LeftElbow"]["boundary_type"], "none")
        self.assertEqual(cuts.rim_topology(secs)["LeftElbow"]["components"], [])

    def test_rim_topology_records_are_json_serializable(self):
        record = cuts.rim_topology(self.secs)
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertEqual(record["LeftElbow"]["boundary_type"], "loops")
        self.assertEqual(record["LeftElbow"]["cut_radius_studs"], self.secs["LeftLowerArm"].cut_radius)
        self.assertIs(record["LeftElbow"]["components"][0]["closed_ring"], True)
        self.assertEqual(record["LeftElbow"]["components"][0]["vertices"], 43)
        self.assertEqual(record["LeftElbow"]["components"][0]["edges"], 43)
        self.assertEqual(record["LeftElbow"]["components"][0]["loop_vertices"], 43)

    def test_touching_rings_and_dangling_contacts_do_not_erase_real_rims(self):
        child, parent = "LeftLowerArm", "LeftUpperArm"
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        ring_edges = split.boundary_edges(faces)
        for dangling in (False, True):
            edges = ring_edges | ({(0, 5)} if dangling else set())
            boundaries = {n: set() for n in SECTIONS}
            boundaries.update({child: edges, parent: edges})
            rims = split._shared_rims(child, boundaries, points)
            self.assertEqual(len(rims), 1)
            self.assertTrue(split.is_closed_rim(rims[0]))
            np.testing.assert_array_equal(split.rim_loop_vertices(rims[0]), np.arange(5))

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
                boundary = rec["joint_rims"]["low"]["LeftElbow"]
                self.assertEqual(boundary["boundary_type"], "none")
                if count == 3:
                    self.assertLess(boundary["components"][0]["open_chains"][0]["length_over_radius"], 1)
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
    def test_carve_gaps_match_frozen_qualifying_boundaries(self):
        from herogen.blender import pose

        # Frozen full pose maps on synthetic bodies, with unchanged section geometry. Thirteen cases
        # match e3d2b4c exactly. arm10 flat/dome include shared shoulder endpoints missing from
        # cap-owner samples; clothed dome includes hip endpoints. Waist includes four shared-edge
        # endpoints omitted at the arm junction.
        names = ['Neck', 'Waist', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightShoulder', 'RightElbow']
        names += ['RightWrist', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'RightHip', 'RightKnee', 'RightAnkle']
        fixtures = [
            ({}, (
                ((0.0825, 0.184, 0.185, 0.1557, 0.1215, 0.1934, 0.1577,
                  0.1222, 0.2364, 0.2254, 0.1874, 0.2368, 0.2255, 0.182),
                 "77e1fba39722c3b1d8d201e1be710c2a797259e2debe17ca33cb58d74c25de73"),
                ((0.04, 0.1752, 0.038, 0.0287, 0.033, 0.0479, 0.0347,
                  0.0362, 0.07, 0.0605, 0.1575, 0.0597, 0.0609, 0.1648),
                 "2a2b87a43409412a27cbd19b1a3e9d7d0eb75a8f377e0be7b23f85c19d476f44"),
            )),
            ({'arm_angle_deg': 10}, (
                ((0.0824, 0.3559, 0.7183, 0.1868, 0.1251, 0.7231, 0.2145,
                  0.1233, 0.2365, 0.2254, 0.1832, 0.2373, 0.2255, 0.1865),
                 "72979dcb2c5020ea1a8034946866cd55d347fc3d3308a0fb657e8c7feac59d69"),
                ((0.0415, 0.3559, 0.7084, 0.1868, 0.0244, 0.5606, 0.2145,
                  0.033, 0.0724, 0.0605, 0.1507, 0.0691, 0.0608, 0.1689),
                 "f6802765fbbef76b8e70c3c314d53e6ea21f3c98a84b3fc17c7ed289664d68e0"),
            )),
            ({'arm_angle_deg': 30}, (
                ((0.0826, 0.1936, 0.3066, 0.1647, 0.123, 0.3159, 0.1735,
                  0.123, 0.236, 0.2253, 0.1857, 0.2367, 0.2256, 0.1898),
                 "2af907b68ffc84d0e98d1eb99eeec7cf6ecd1209f09471b2587af02936af2d42"),
                ((0.0373, 0.1898, 0.1954, 0.0338, 0.0332, 0.2832, 0.0389,
                  0.0325, 0.0608, 0.0623, 0.1483, 0.0599, 0.0628, 0.1706),
                 "2e8270dc2ca3ff48e090d2894f7521ec1abc09e14a7749a685737322f3a37184"),
            )),
            ({'arm_angle_deg': 70}, (
                ((0.0825, 0.1936, 0.2119, 0.1675, 0.1231, 0.2167, 0.1604,
                  0.1235, 0.2368, 0.2255, 0.1897, 0.2368, 0.2256, 0.1842),
                 "21b644885d995ae354534cdd33e9f545f8112d9dffc47bc2e4491329352ca185"),
                ((0.04, 0.1898, 0.0935, 0.0368, 0.03, 0.1259, 0.0375,
                  0.0283, 0.0613, 0.0623, 0.1616, 0.0634, 0.0609, 0.1655),
                 "5b42cff8376f30bc8fd4771cfd3f2647a0812281c9b7c963d24e529c05a69384"),
            )),
            ({'head_scale': 1.2}, (
                ((0.1289, 0.1949, 0.1853, 0.1737, 0.1169, 0.1935, 0.1656,
                  0.1224, 0.236, 0.2286, 0.1877, 0.2359, 0.2287, 0.1824),
                 "38e78fb012258908b516e7d147174aefc79334e85c9dd102957b30bbf70096e7"),
                ((0.0904, 0.1717, 0.0413, 0.0359, 0.0278, 0.0478, 0.0338,
                  0.0248, 0.0698, 0.0615, 0.1569, 0.0601, 0.0621, 0.1664),
                 "9b3d9ae94c026e58ed594cfa85846464445666db79c3dcae0d6ed8f869e8be49"),
            )),
            ({'leg_spread': 0.1}, (
                ((0.0825, 0.184, 0.1846, 0.1557, 0.1215, 0.1931, 0.1577,
                  0.1222, 0.2373, 0.2266, 0.187, 0.2381, 0.2266, 0.1853),
                 "f2f55bb32ad9a90f8fb2070f0eb691f8b59871eb17e5f3f1aca79c12b15aaa56"),
                ((0.04, 0.1752, 0.0379, 0.0287, 0.033, 0.0478, 0.0347,
                  0.0362, 0.0641, 0.0629, 0.1575, 0.0623, 0.0603, 0.1675),
                 "c3521dfd29cae64f4bf307cded8be90ab13cd84592faa8807c0a032c609d8d4c"),
            )),
            ({'leg_spread': 0.2}, (
                ((0.0825, 0.184, 0.2597, 0.1546, 0.1215, 0.2479, 0.1587,
                  0.1222, 0.2378, 0.2218, 0.191, 0.2388, 0.2224, 0.1889),
                 "1e0a0c89d1ee2e6b0c276b333193c2acda0bf8b24687106f43b68eaca7d9b6a9"),
                ((0.04, 0.1752, 0.2017, 0.0309, 0.033, 0.1482, 0.0343,
                  0.0362, 0.0613, 0.0701, 0.1623, 0.0612, 0.0769, 0.1702),
                 "b559d8d5beba6c8604e7263b4b88c7524568369d74653b2e05f348c70c01b549"),
            )),
            ({'clothed': True}, (
                ((0.1241, 0.258, 0.2063, 0.1637, 0.1206, 0.2031, 0.1669,
                  0.1143, 0.2744, 0.2294, 0.1866, 0.2789, 0.2288, 0.182),
                 "dec86aa5bc78e06c048dccf89dd0d6b1308f0a6ec8f5abdf366f50b7f8daaa7c"),
                ((0.1001, 0.1711, 0.0873, 0.0369, 0.0306, 0.047, 0.0312,
                  0.029, 0.0721, 0.0663, 0.1485, 0.0679, 0.062, 0.1649),
                 "0f4ec5cf29b2cdadb81635d94fad29ed7c368bfdc7bc122b354bd9edb8dcf2a1"),
            )),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            for i, (variant, expected) in enumerate(fixtures):
                run = Path(tmp) / str(i)
                synth.write_run(run, synth.humanoid(**variant), size=384, ppu=60.0)
                vs = views.load_views(run, 5.5)
                sk = joints.estimate(vs, 5.5)
                verts, faces, _ = carve.build_mesh(vs, 5.5, 96)
                poses = pose.pose_library(pose.bone_directions(sk))
                for cap, (values, digest) in zip(("flat", "dome"), expected):
                    with self.subTest(variant=variant, cap=cap):
                        secs, _ = split.build_sections(verts, faces, sk, cap=cap, rings=1, ring_points=8)
                        frozen = {j: v for j, v in zip(names, values) if v is not None}
                        missing = {j for j, v in zip(names, values) if v is None}
                        self.assertEqual(set(cuts.unmeasurable_sections(secs)), missing)
                        if missing:
                            with self.assertRaises(cuts.UnmeasurableJoints):
                                pose.joint_gaps(secs, poses)
                        gaps = pose.joint_gaps(secs, poses, joints=set(frozen))
                        if variant == {"arm_angle_deg": 10} and cap == "dome":
                            # The arm junction splits the cut-plane Waist boundary into a loop and
                            # two sub-radius chains. The old 114 samples omitted four real shared-edge
                            # endpoints at the arm junction; all 118 vertices must contribute.
                            waist = cuts.rim_topology(secs)["Waist"]
                            self.assertEqual(waist["boundary_type"], "loops+open_chain")
                            self.assertEqual(waist["piece_count"], 3)
                            self.assertGreaterEqual(waist["length_over_radius"], 1)
                            self.assertTrue(all(c["length_over_radius"] < 1 for c in waist["components"]
                                                if c["boundary_type"] == "open_chain"))
                            measured = split.rim_measurements(
                                [r for r in secs["UpperTorso"].rims if r["neighbour"] == "LowerTorso"],
                                secs["UpperTorso"].cut_radius)
                            self.assertEqual(sum(len(ids) for _, ids in measured), 118)
                            self.assertEqual({p: gaps[p]["Waist"] for p in
                                              ("twist_turn", "torso_side_bend", "reach_up_L")},
                                             {"twist_turn": 0.3559, "torso_side_bend": 0.3034,
                                              "reach_up_L": 0.1078})
                        self.assertEqual({j: max(r[j] for r in gaps.values()) for j in frozen}, frozen)
                        self.assertEqual(hashlib.sha256(json.dumps(gaps, sort_keys=True,
                                                                   separators=(",", ":")).encode()).hexdigest(),
                                         digest)

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
