"""Stage 6 rig spec and writer on synthetic bodies: frames, joints, attachments, HipHeight, refusals."""
from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from herogen.blender import PARENT, PIVOT_JOINT, SECTIONS
from herogen.roblox import (HRP_SIZE, MESH_TRIANGLE_LIMIT, R15, RobloxExportError, b_to_r, g_to_r,
                            g_to_r_points)
from herogen.roblox.glb import mesh_nodes
from herogen.roblox.rbxmx import angles, write_rbxmx
from herogen.roblox.rig import build_rig
from herogen.roblox.synth import BOXES, OFFSET, edit_glb_json, write_body


class FrameTest(unittest.TestCase):
    def test_contract_axes_map_to_roblox_axes(self):
        # Contract: +Z up, faces -Y, left +X. Roblox: +Y up, faces -Z, left -X.
        self.assertEqual(b_to_r((0, 0, 1)), [0, 1, 0])     # up stays up
        self.assertEqual(b_to_r((0, -1, 0)), [0, 0, -1])   # the front becomes Roblox's front (-Z)
        self.assertEqual(b_to_r((1, 0, 0)), [-1, 0, 0])    # the character's left becomes -X
        self.assertEqual(b_to_r((1, 2, 3)), [-1, 3, 2])

    def test_glb_route_agrees_and_is_a_rotation_not_a_mirror(self):
        p = (0.3, -1.7, 2.9)
        g = (p[0], p[2], -p[1])  # how stage 4 exports the GLB
        self.assertEqual(g_to_r(g), b_to_r(p))
        m = np.array([b_to_r(e) for e in np.eye(3)]).T
        self.assertAlmostEqual(float(np.linalg.det(m)), 1.0)
        gm = np.array([g_to_r(e) for e in np.eye(3)]).T
        np.testing.assert_allclose(gm, angles((0, 180, 0)), atol=1e-12)  # a 180 degree turn about Y

    def test_point_arrays_convert_like_points(self):
        pts = np.array([[0.1, 0.2, 0.3], [-1.0, 2.5, -4.0]])
        np.testing.assert_array_equal(g_to_r_points(pts), [g_to_r(p) for p in pts])

    def test_r15_tables_match_the_body_contract(self):
        self.assertEqual(sorted(p for p, _, _ in R15), sorted(SECTIONS))
        self.assertEqual(R15[0][0], "LowerTorso")  # root first, every parent before its children
        self.assertTrue(all(any(q == parent for q, _, _ in R15[:i]) or parent == "HumanoidRootPart"
                            for i, (_, parent, _) in enumerate(R15)))
        for part, parent, joint in R15:
            self.assertEqual(PARENT[part] or "HumanoidRootPart", parent)
            self.assertEqual(PIVOT_JOINT[part], joint)

    def test_grip_rotation_points_forward(self):
        m = angles((-90, 0, 0))
        np.testing.assert_array_equal(m, [[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        np.testing.assert_allclose(m @ [0, 1, 0], [0, 0, -1])  # the grip's up axis points to Roblox's front


class RigSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.body = write_body(Path(cls.tmp.name) / "04-body")
        cls.spec = build_rig(cls.body)
        cls.parts = {p["name"]: p for p in cls.spec["parts"]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parts_are_the_vertex_bounds_in_roblox_frame(self):
        for name, (c, s, _) in BOXES.items():
            p = self.parts[name]
            np.testing.assert_allclose(p["center"], b_to_r(np.add(c, OFFSET)), atol=1e-6)
            np.testing.assert_allclose(p["size"], np.abs(b_to_r(s)), atol=1e-6)

    def test_joints_follow_the_r15_tree_at_the_pivots(self):
        self.assertEqual([(p["name"], p["parent"], p["joint"]) for p in self.spec["parts"]], list(R15))
        for name, (_, _, pv) in BOXES.items():
            np.testing.assert_allclose(self.parts[name]["pivot"], b_to_r(np.add(pv, OFFSET)), atol=1e-6)

    def test_root_part_and_hip_height(self):
        root = self.parts["LowerTorso"]["pivot"]
        self.assertEqual(self.spec["hrp"], {"center": root, "size": list(HRP_SIZE)})
        # feet at z = 0 in the contract, so HipHeight = root height - half the root part
        self.assertAlmostEqual(self.spec["hip_height"], BOXES["LowerTorso"][2][2] - HRP_SIZE[1] / 2, places=6)
        self.assertAlmostEqual(self.spec["feet_min_y"], 0.0, places=6)

    def test_hip_height_follows_the_lowest_foot_vertex(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = write_body(Path(tmp) / "b")
            sj = json.loads((body / "sections.json").read_text())
            self.assertEqual(sj["units"], "studs")
            # lift the whole GLB by 0.25 studs: the feet leave the floor, HipHeight drops by the same amount
            def lift(doc):
                for n in doc["nodes"]:
                    n["translation"][1] += 0.25
            edit_glb_json(body / "high.glb", lift)
            for s in sj["sections"]:
                s["pivot"][2] += 0.25
            (body / "sections.json").write_text(json.dumps(sj))
            spec = build_rig(body)
            self.assertAlmostEqual(spec["feet_min_y"], 0.25, places=6)
            self.assertAlmostEqual(spec["hip_height"], self.spec["hip_height"], places=6)

    def test_attachment_cframes_in_the_file(self):
        root = ET.fromstring(write_rbxmx(self.spec))
        items = {}
        for it in root.iter("Item"):
            name = it.find("Properties/string[@name='Name']").text
            items.setdefault(name, []).append(it)
        # both rig attachments of a joint sit at its pivot; the Motor6D's C0/C1 are their CFrames
        for part, parent, joint in R15:
            motor = items[joint][0]
            c0 = motor.find("Properties/CoordinateFrame[@name='C0']")
            c1 = motor.find("Properties/CoordinateFrame[@name='C1']")
            atts = items[joint + "RigAttachment"]
            self.assertEqual(len(atts), 2)
            owners = {}
            for a in atts:
                owner = _parent_of(root, a).find("Properties/string[@name='Name']").text
                owners[owner] = [x.text for x in a.find("Properties/CoordinateFrame[@name='CFrame']")]
            self.assertEqual(set(owners), {part, parent})
            self.assertEqual(owners[parent], [x.text for x in c0])
            self.assertEqual(owners[part], [x.text for x in c1])
            local = np.array([float(x) for x in owners[part][:3]])
            want = np.subtract(self.parts[part]["pivot"], self.parts[part]["center"])
            np.testing.assert_allclose(local, want, atol=1e-6)


class RefusalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertRefused(self, body, *fragments):
        with self.assertRaises(RobloxExportError) as ctx:
            build_rig(body)
        msg = str(ctx.exception)
        self.assertNotIn("\n", msg)
        for f in fragments:
            self.assertIn(f, msg)

    def test_over_the_triangle_limit_is_refused_with_the_profile_hint(self):
        body = write_body(self.root / "b", triangles={"LowerTorso": MESH_TRIANGLE_LIMIT + 4})
        self.assertRefused(body, "LowerTorso has 20004 triangles", "20000 per mesh",
                           "use the Roblox in-game or avatar profile")

    def test_exactly_at_the_limit_is_accepted(self):
        body = write_body(self.root / "b", triangles={"UpperTorso": MESH_TRIANGLE_LIMIT})
        self.assertEqual(mesh_nodes(body / "high.glb")["UpperTorso"]["triangles"], MESH_TRIANGLE_LIMIT)
        build_rig(body)

    def test_missing_inputs(self):
        self.assertRefused(self.root / "nowhere", "missing input", "sections.json")
        body = write_body(self.root / "b")
        (body / "high.glb").unlink()
        self.assertRefused(body, "missing input", "high.glb")

    def test_wrong_units_frame_or_tree(self):
        body = write_body(self.root / "b")
        good = (body / "sections.json").read_text()
        for change, fragment in ((lambda sj: sj.update(units="meters"), "units"),
                                 (lambda sj: sj.update(frame="Y up"), "frame"),
                                 (lambda sj: sj["sections"].pop(), "missing ['RightFoot']"),
                                 (lambda sj: sj["sections"][3].update(parent="Head"), "LeftUpperArm")):
            sj = json.loads(good)
            change(sj)
            (body / "sections.json").write_text(json.dumps(sj))
            self.assertRefused(body, fragment)
        (body / "sections.json").write_text("{not json")
        self.assertRefused(body, "unreadable")

    def test_glb_layouts_it_would_misread(self):
        body = write_body(self.root / "b")
        good = (body / "high.glb").read_bytes()
        cases = (
            (lambda d: d["nodes"][0].update(rotation=[0, 0.7071068, 0, 0.7071068]), "rotated"),
            (lambda d: d["nodes"][0].update(scale=[2, 2, 2]), "scaled"),
            (lambda d: d["nodes"][0].update(children=[1]), "children"),
            (lambda d: d["meshes"][0]["primitives"][0].update(mode=1), "not triangles"),
            (lambda d: d["nodes"][1].update(name=d["nodes"][0]["name"]), "two mesh nodes"),
            (lambda d: d["accessors"][0].update(componentType=5123), "float VEC3"),
            (lambda d: d["meshes"][0]["primitives"][0].pop("attributes"), "malformed glTF"),
        )
        for change, fragment in cases:
            (body / "high.glb").write_bytes(good)
            edit_glb_json(body / "high.glb", change)
            self.assertRefused(body, fragment)
        (body / "high.glb").write_bytes(b"not a glb file at all")
        self.assertRefused(body, "not a glTF 2.0 binary")

    def test_identity_transforms_are_accepted(self):
        body = write_body(self.root / "b")
        edit_glb_json(body / "high.glb",
                      lambda d: d["nodes"][0].update(rotation=[0, 0, 0, 1], scale=[1, 1, 1]))
        build_rig(body)


def _parent_of(root, el):
    for it in root.iter("Item"):
        if el in list(it):
            return it
    raise AssertionError("no parent")


if __name__ == "__main__":
    unittest.main()
