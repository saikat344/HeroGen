"""Stage 6 checker: a good synthetic rig passes, and every corrupted rig fails the check meant for it."""
from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np

from herogen.roblox import R15, STARTER_MARKER, b_to_r, g_to_r
from herogen.roblox import check as checker
from herogen.roblox import place as place_module
from herogen.roblox import rig as rig_module
from herogen.roblox.check import check_place, check_rig
from herogen.roblox.place import write_rbxlx
from herogen.roblox.rbxmx import write_rbxmx
from herogen.roblox.rig import build_rig
from herogen.roblox.synth import edit_glb_json, write_body

CF_KEYS = ("X", "Y", "Z", "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22")


def _find(root: ET.Element, *path: str) -> ET.Element:
    """The Item reached by following names from the model, e.g. ("LeftLowerArm", "LeftElbow")."""
    item = root.find("Item")
    for name in path:
        item = next(c for c in item.findall("Item") if c.find("Properties/string[@name='Name']").text == name)
    return item


def _get_cf(item: ET.Element, prop: str = "CFrame") -> list[float]:
    el = item.find(f"Properties/CoordinateFrame[@name='{prop}']")
    return [float(el.find(k).text) for k in CF_KEYS]


def _set_cf(item: ET.Element, values, prop: str = "CFrame") -> None:
    el = item.find(f"Properties/CoordinateFrame[@name='{prop}']")
    for k, v in zip(CF_KEYS, values):
        el.find(k).text = repr(float(v))


def _reflect(cf):
    """S * cf * S with S = diag(-1, 1, 1): a mirror image in X that keeps every rotation proper."""
    x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22 = cf
    return [-x, y, z, r00, -r01, -r02, -r10, r11, r12, -r20, r21, r22]


def _failed(checks) -> set[str]:
    return {c["name"] for c in checks if c.get("gate", True) and not c["pass"]}


class IndependenceTest(unittest.TestCase):
    """The checker restates the tables and the frame conversion; they must agree with the writer's."""

    def test_same_r15_tree(self):
        self.assertEqual(checker.R15, {part: (parent, joint) for part, parent, joint in R15})

    def test_same_starter_marker(self):
        self.assertEqual(checker.STARTER_MARKER, STARTER_MARKER)

    def test_same_frame_conversion(self):
        for p in ((1, 2, 3), (-0.5, 0.25, 4.0)):
            np.testing.assert_allclose(checker._b_to_r(p), b_to_r(p))
            np.testing.assert_allclose(checker._g_to_r(p), g_to_r(p))


class CheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.body = write_body(cls.root / "04-body")
        cls.good = write_rbxmx(build_rig(cls.body))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_check(self, text: str, body: Path | None = None) -> set[str]:
        path = self.root / "rig.rbxmx"
        path.write_text(text)
        return _failed(check_rig(body or self.body, path))

    def mutate(self, change) -> set[str]:
        tree = ET.fromstring(self.good)
        change(tree)
        return self.run_check(ET.tostring(tree, encoding="unicode"))

    def test_good_rig_passes_every_check(self):
        path = self.root / "good.rbxmx"
        path.write_text(self.good)
        checks = check_rig(self.body, path)
        self.assertEqual(_failed(checks), set())
        names = {c["name"] for c in checks}
        for want in ("humanoid_r15", "root_part", "sections_r15_names", "joints_named_and_linked",
                     "joint_chain_reachable", "rig_attachment_pairs", "attachments_at_pivots",
                     "motor_c0_c1_equal_attachments", "rest_pose_position", "rest_pose_rotation",
                     "section_extents_match_glb", "section_centres_match_glb", "hip_height", "faces_minus_z",
                     "not_mirrored", "rotations_proper", "triangles_per_mesh", "accessory_attachments"):
            self.assertIn(want, names)
        info = next(c for c in checks if c["name"] == "meshes_bound")
        self.assertEqual((info["value"], info["gate"]), (0, False))

    def test_every_check_records_the_comparison_it_passes_on(self):
        path = self.root / "good.rbxmx"
        path.write_text(self.good)
        for c in check_rig(self.body, path):
            if c["threshold"] is None:
                self.assertIsNone(c["rule"], c["name"])
                continue
            passes = {"<": c["value"] < c["threshold"], "<=": c["value"] <= c["threshold"],
                      "==": c["value"] == c["threshold"], ">": c["value"] > c["threshold"]}[c["rule"]]
            self.assertEqual(passes, c["pass"], c["name"])

    # Corrupted rigs: each must fail the checks named.

    def test_c0_offset(self):
        def change(t):
            m = _find(t, "LeftLowerArm", "LeftElbow")
            cf = _get_cf(m, "C0")
            cf[0] += 0.05
            _set_cf(m, cf, "C0")
        failed = self.mutate(change)
        self.assertTrue({"motor_c0_c1_equal_attachments", "rest_pose_position"} <= failed, failed)

    def test_part_renamed(self):
        def change(t):
            _find(t, "RightHand").find("Properties/string[@name='Name']").text = "RightPalm"
        self.assertIn("sections_r15_names", self.mutate(change))

    def test_missing_joint(self):
        def change(t):
            leg = _find(t, "LeftUpperLeg")
            leg.remove(_find(t, "LeftUpperLeg", "LeftHip"))
        failed = self.mutate(change)
        self.assertTrue({"joints_named_and_linked", "joint_chain_reachable"} <= failed, failed)

    def test_hip_height_off(self):
        def change(t):
            el = _find(t, "Humanoid").find("Properties/float[@name='HipHeight']")
            el.text = repr(float(el.text) + 0.3)
        self.assertEqual(self.mutate(change), {"hip_height"})

    def test_neck_pivot_moved(self):
        def change(t):
            a = _find(t, "UpperTorso", "NeckRigAttachment")
            cf = _get_cf(a)
            cf[1] += 0.1
            _set_cf(a, cf)
            _set_cf(_find(t, "Head", "Neck"), cf, "C0")  # a consistent Motor6D, but off the pivot
        failed = self.mutate(change)
        self.assertTrue({"attachments_at_pivots", "attachment_pairs_coincide"} <= failed, failed)

    def test_mirrored_rig(self):
        def change(t):
            for it in t.iter("Item"):
                for prop in ("CFrame", "C0", "C1"):
                    if it.find(f"Properties/CoordinateFrame[@name='{prop}']") is not None:
                        _set_cf(it, _reflect(_get_cf(it, prop)), prop)
        failed = self.mutate(change)
        want = {"not_mirrored", "section_centres_match_glb", "attachments_at_pivots"}
        self.assertTrue(want <= failed, failed)

    def test_foot_size(self):
        def change(t):
            el = _find(t, "LeftFoot").find("Properties/Vector3[@name='size']")
            for c in el:
                c.text = repr(float(c.text) * 1.1)
        self.assertIn("section_extents_match_glb", self.mutate(change))

    # Further corruptions.

    def test_improper_rotation(self):
        def change(t):
            p = _find(t, "Head")
            cf = _get_cf(p)
            cf[3] = -1.0  # R00: a reflection, determinant -1
            _set_cf(p, cf)
        self.assertIn("rotations_proper", self.mutate(change))

    def test_wrong_rig_type_and_missing_scale_value(self):
        def change(t):
            hum = _find(t, "Humanoid")
            hum.find("Properties/token[@name='RigType']").text = "0"
            hum.remove(_find(t, "Humanoid", "BodyTypeScale"))
        self.assertTrue({"humanoid_r15", "humanoid_scale_values"} <= self.mutate(change))

    def test_motor_linked_to_the_wrong_part(self):
        def change(t):
            hand = _find(t, "LeftLowerArm").get("referent")
            _find(t, "RightHand", "RightWrist").find("Properties/Ref[@name='Part0']").text = hand
        self.assertIn("joints_named_and_linked", self.mutate(change))

    def test_extra_part_and_missing_accessory(self):
        def change(t):
            model = t.find("Item")
            extra = ET.fromstring('<Item class="Part" referent="999"><Properties>'
                                  '<string name="Name">Cape</string></Properties></Item>')
            model.append(extra)
            _find(t, "Head").remove(_find(t, "Head", "HatAttachment"))
        self.assertTrue({"sections_r15_names", "accessory_attachments"} <= self.mutate(change))

    def test_not_xml(self):
        self.assertEqual(self.run_check(self.good[:-40]), {"xml_well_formed"})

    def test_unreadable_value_blames_the_rig(self):
        def change(t):
            _find(t, "Humanoid").find("Properties/float[@name='HipHeight']").text = "abc"
            _find(t, "Head").find("Properties/CoordinateFrame[@name='CFrame']").remove(
                _find(t, "Head").find("Properties/CoordinateFrame[@name='CFrame']/R00"))
        self.assertEqual(self.mutate(change), {"xml_values_valid"})

    # Faults in the body or the writer that the file alone wouldn't show.

    def test_body_facing_backwards_is_caught(self):
        body = write_body(self.root / "backwards", facing_back=True)
        failed = self.run_check(write_rbxmx(build_rig(body)), body)
        self.assertEqual(failed, {"faces_minus_z"})

    def test_writer_with_a_mirrored_frame_conversion_is_caught(self):
        # R = (Bx, Bz, By) = (Gx, Gy, -Gz): a left/right handedness bug, applied consistently everywhere
        with mock.patch.object(rig_module, "b_to_r", lambda p: [float(p[0]), float(p[2]), float(p[1])]), \
                mock.patch.object(rig_module, "g_to_r", lambda p: [float(p[0]), float(p[1]), -float(p[2])]), \
                mock.patch.object(rig_module, "g_to_r_points", lambda pts: pts * [1.0, 1.0, -1.0]):
            text = write_rbxmx(rig_module.build_rig(self.body))
        failed = self.run_check(text)
        want = {"not_mirrored", "section_centres_match_glb", "attachments_at_pivots"}
        self.assertTrue(want <= failed, failed)

    def test_section_below_the_feet_is_caught(self):
        body = write_body(self.root / "lowhand")
        def drop(doc):  # move the left hand's node 2.2 studs down, below the floor
            n = next(n for n in doc["nodes"] if n["name"] == "LeftHand")
            n["translation"][1] -= 2.2
        edit_glb_json(body / "high.glb", drop)
        failed = self.run_check(self.good, body)
        self.assertIn("feet_lowest", failed)


def _child(item: ET.Element, name: str, cls: str | None = None) -> ET.Element:
    return next(c for c in item.findall("Item") if c.find("Properties/string[@name='Name']").text == name
                and (cls is None or c.get("class") == cls))


def _service(root: ET.Element, cls: str) -> ET.Element:
    return next(c for c in root.findall("Item") if c.get("class") == cls)


PLACE_CHECKS = ("place_xml_well_formed", "place_referents_resolve", "place_services", "place_baseplate",
                "place_spawn", "place_spawn_on_baseplate", "place_marker", "place_starter_player_empty",
                "place_rig_present", "place_rig_matches_file", "place_rig_rigid_move_position",
                "place_rig_rigid_move_rotation", "place_rig_upright", "place_feet_on_baseplate",
                "place_rig_clear_of_spawn", "place_rig_faces_spawn")


class PlaceCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.body = write_body(cls.root / "04-body")
        cls.spec = build_rig(cls.body)
        cls.rig = cls.root / "rig.rbxmx"
        cls.rig.write_text(write_rbxmx(cls.spec))
        cls.good = write_rbxlx(cls.spec)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_check(self, text: str) -> list[dict]:
        path = self.root / "place.rbxlx"
        path.write_text(text)
        return check_place(path, self.rig)

    def mutate(self, change) -> set[str]:
        tree = ET.fromstring(self.good)
        change(tree)
        return _failed(self.run_check(ET.tostring(tree, encoding="unicode")))

    def test_good_place_passes_every_check(self):
        checks = self.run_check(self.good)
        self.assertEqual(tuple(c["name"] for c in checks), PLACE_CHECKS)
        self.assertEqual(_failed(checks), set())
        for c in checks:
            passes = {"<": c["value"] < c["threshold"], "<=": c["value"] <= c["threshold"],
                      "==": c["value"] == c["threshold"], ">": c["value"] > c["threshold"]}[c["rule"]]
            self.assertEqual(passes, c["pass"], c["name"])

    def test_place_layout(self):
        root = ET.fromstring(self.good)
        self.assertEqual((root.tag, root.get("version")), ("roblox", "4"))
        self.assertEqual([c.get("class") for c in root.findall("Item")],
                         ["Workspace", "Lighting", "StarterPlayer", "ServerScriptService", "ServerStorage"])
        ws = _service(root, "Workspace")
        self.assertEqual([(c.get("class"), c.find("Properties/string").text) for c in ws.findall("Item")],
                         [("Part", "Baseplate"), ("SpawnLocation", "SpawnLocation"), ("Model", "HeroGen-R15")])
        self.assertEqual(_service(root, "StarterPlayer").findall("Item"), [])
        marker = _child(_service(root, "ServerStorage"), STARTER_MARKER, "BoolValue")
        self.assertEqual(marker.find("Properties/bool[@name='Value']").text, "true")
        # BasePart.Color is written as rbx-dom writes it: Color3uint8, one 0xRRGGBB integer
        base = _child(ws, "Baseplate")
        self.assertEqual(int(base.find("Properties/Color3uint8[@name='Color3uint8']").text),
                         (91 << 16) | (93 << 8) | 105)
        self.assertEqual(_child(ws, "SpawnLocation").find("Properties/int[@name='Duration']").text, "0")

    def test_rig_stands_on_the_floor_facing_the_spawn(self):
        model = checker.Inst(_child(_service(ET.fromstring(self.good), "Workspace"), "HeroGen-R15"), None)
        hrp = model.child("HumanoidRootPart").cframe()
        np.testing.assert_allclose(hrp[:3, :3], np.diag([-1.0, 1.0, -1.0]), atol=1e-7)  # 180 degrees about Y
        self.assertEqual((hrp[0, 3], hrp[2, 3]), place_module.RIG_SPOT[::2])
        feet = [model.child(f) for f in ("LeftFoot", "RightFoot")]
        self.assertAlmostEqual(min(f.cframe()[1, 3] - f.size()[1] / 2 for f in feet), 0.0, places=5)
        # the part-relative values are the rig file's own, so HipHeight still puts the feet on that floor
        self.assertAlmostEqual(float(model.child("Humanoid").props["HipHeight"]), 1.7, places=5)

    def test_deterministic(self):
        self.assertEqual(write_rbxlx(build_rig(self.body)), self.good)

    # Corrupted places: each must fail the checks named.

    def test_not_xml(self):
        self.assertEqual(_failed(self.run_check(self.good[:-30])), {"place_xml_well_formed"})

    def test_missing_marker(self):
        self.assertEqual(self.mutate(lambda t: _service(t, "ServerStorage").remove(
            _child(_service(t, "ServerStorage"), STARTER_MARKER))), {"place_marker"})

    def test_marker_set_false(self):
        def change(t):
            _child(_service(t, "ServerStorage"), STARTER_MARKER).find("Properties/bool[@name='Value']").text = "false"
        self.assertEqual(self.mutate(change), {"place_marker"})

    def test_starter_character_in_the_file(self):
        def change(t):
            sp = _service(t, "StarterPlayer")
            sp.append(ET.fromstring('<Item class="Model" referent="sc"><Properties>'
                                    '<string name="Name">StarterCharacter</string></Properties></Item>'))
        self.assertEqual(self.mutate(change), {"place_starter_player_empty"})

    def test_missing_lighting_service(self):
        def change(t):
            t.remove(_service(t, "Lighting"))
        self.assertEqual(self.mutate(change), {"place_services"})

    def test_missing_spawn_and_unanchored_baseplate(self):
        def change(t):
            ws = _service(t, "Workspace")
            ws.remove(_child(ws, "SpawnLocation"))
            _child(ws, "Baseplate").find("Properties/bool[@name='Anchored']").text = "false"
        failed = self.mutate(change)
        self.assertTrue({"place_spawn", "place_baseplate", "place_feet_on_baseplate"} <= failed, failed)

    def test_rig_missing(self):
        def change(t):
            ws = _service(t, "Workspace")
            ws.remove(_child(ws, "HeroGen-R15"))
        self.assertEqual(self.mutate(change), {"place_rig_present"})

    def test_one_part_moved_on_its_own(self):
        def change(t):
            hand = _child(_child(_service(t, "Workspace"), "HeroGen-R15"), "LeftHand")
            cf = _get_cf(hand)
            cf[1] += 0.01
            _set_cf(hand, cf)
        self.assertEqual(self.mutate(change), {"place_rig_rigid_move_position"})

    def test_attachment_changed(self):
        def change(t):
            head = _child(_child(_service(t, "Workspace"), "HeroGen-R15"), "Head")
            att = _child(head, "HatAttachment")
            cf = _get_cf(att)
            cf[1] += 0.1
            _set_cf(att, cf)
        self.assertEqual(self.mutate(change), {"place_rig_matches_file"})

    def test_rig_floating_and_tilted(self):
        with mock.patch.object(place_module, "RIG_SPOT", (0.0, 0.5, -16.0)):
            self.assertEqual(_failed(self.run_check(write_rbxlx(self.spec))), {"place_feet_on_baseplate"})
        with mock.patch.object(place_module, "angles", lambda deg: checker._cf(
                (0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 1, 0))[:3, :3]):  # 90 degrees about X: lying down
            failed = _failed(self.run_check(write_rbxlx(self.spec)))
        self.assertIn("place_rig_upright", failed)

    def test_rig_on_the_spawn_or_facing_away(self):
        with mock.patch.object(place_module, "RIG_SPOT", (0.0, 0.0, -3.0)):
            self.assertEqual(_failed(self.run_check(write_rbxlx(self.spec))), {"place_rig_clear_of_spawn"})
        with mock.patch.object(place_module, "RIG_YAW_DEG", 0.0):
            self.assertEqual(_failed(self.run_check(write_rbxlx(self.spec))), {"place_rig_faces_spawn"})

    def test_wrong_rig_file(self):
        other = write_body(self.root / "other", facing_back=True)
        path = self.root / "place.rbxlx"
        path.write_text(write_rbxlx(build_rig(other)))
        failed = _failed(check_place(path, self.rig))
        self.assertIn("place_rig_matches_file", failed)


if __name__ == "__main__":
    unittest.main()
