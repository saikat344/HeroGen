"""The viewer's animation data: the body GLB's R15 joint pivots, read from stage 4's sections.json in the GLB
frame, and refused for anything that is not a complete 15-part R15 body."""

from __future__ import annotations

import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import TempDirCase

from herogen.blender import PARENT, PIVOT_JOINT, SECTIONS
from herogen.service import animation
from herogen.service.fakes import box_glb


def parts_glb(names=SECTIONS) -> bytes:
    """A glTF binary whose scene is one translated mesh node per name (like stage 4's body), sharing a box."""
    box = box_glb()
    js_len = struct.unpack_from("<I", box, 12)[0]
    doc = json.loads(box[20:20 + js_len])
    rest = box[20 + js_len:]
    doc["nodes"] = [{"name": n, "mesh": 0, "translation": [0.0, float(i), 0.0]} for i, n in enumerate(names)]
    doc["scenes"] = [{"nodes": list(range(len(names)))}]
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + len(rest)
    return struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js + rest


def sections_doc(**overrides) -> dict:
    """sections.json in the contract frame: section i's pivot is (i, 10 + i, 20 + i)."""
    sections = [{"name": n, "parent": PARENT[n], "joint": PIVOT_JOINT[n], "pivot": [i, 10.0 + i, 20.0 + i]}
                for i, n in enumerate(SECTIONS)]
    return {"units": "studs", "frame": "Blender: +Z up, feet at z=0, faces -Y, x/y centred", "sections": sections,
            **overrides}


def write_body(run, doc=None, glb=None) -> None:
    body = run / "04-body"
    body.mkdir(parents=True, exist_ok=True)
    (body / "high.glb").write_bytes(parts_glb() if glb is None else glb)
    (body / "sections.json").write_text(json.dumps(sections_doc() if doc is None else doc))


class TestBodyJoints(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.run = self.tmp / "run"
        self.run.mkdir()

    def test_pivots_in_the_glb_frame_root_first(self):
        write_body(self.run)
        got = animation.body_joints(self.run)
        self.assertEqual(got["glb"], "04-body/high.glb")
        self.assertIn("+Y up, faces +Z", got["frame"])
        names = [p["part"] for p in got["parts"]]
        self.assertEqual(sorted(names), sorted(SECTIONS))
        self.assertEqual(names[0], "LowerTorso")
        for at, p in enumerate(got["parts"]):
            if p["parent"] is not None:
                self.assertLess(names.index(p["parent"]), at, f"{p['part']} comes after its parent")
            i = SECTIONS.index(p["part"])
            # contract (x, y, z) -> glTF (x, z, -y)
            self.assertEqual(p["pivot"], [float(i), 20.0 + i, -(10.0 + i)])
            self.assertEqual((p["parent"], p["joint"]), (PARENT[p["part"]], PIVOT_JOINT[p["part"]]))

    def test_refuses_what_is_not_a_complete_r15_body(self):
        cases = {
            "no sections": None,
            "wrong units": sections_doc(units="metres"),
            "wrong frame": sections_doc(frame="+Y up"),
            "a part missing": {**sections_doc(), "sections": sections_doc()["sections"][:-1]},
            "wrong parent": {**sections_doc(), "sections": [{**s, "parent": "Head"} if s["name"] == "LeftHand" else s
                                                            for s in sections_doc()["sections"]]},
            "bad pivot": {**sections_doc(), "sections": [{**s, "pivot": [0, "x", 1]} if s["name"] == "Head" else s
                                                        for s in sections_doc()["sections"]]},
            "infinite pivot": {**sections_doc(), "sections": [{**s, "pivot": [0, 1e9, 1]} if s["name"] == "Head"
                                                             else s for s in sections_doc()["sections"]]},
        }
        for label, doc in cases.items():
            with self.subTest(label):
                run = self.tmp / label.replace(" ", "-")
                run.mkdir()
                if doc is None:
                    (run / "04-body").mkdir()
                    (run / "04-body" / "high.glb").write_bytes(parts_glb())
                else:
                    write_body(run, doc)
                self.assertIsNone(animation.body_joints(run))

    def test_refuses_a_glb_without_the_part_nodes_or_truncated(self):
        write_body(self.run, glb=box_glb())
        self.assertIsNone(animation.body_joints(self.run), "a single-mesh GLB has no parts to move")
        other = self.tmp / "cut"
        other.mkdir()
        write_body(other, glb=parts_glb()[:-10])
        self.assertIsNone(animation.body_joints(other), "a truncated GLB")

    def test_odd_shapes_in_the_files_mean_no_buttons_not_an_error(self):
        cases = {"sections not a list": {**sections_doc(), "sections": 7},
                 "unhashable name": {**sections_doc(), "sections": [{**s, "name": ["Head"]} if s["name"] == "Head" else s
                                                                    for s in sections_doc()["sections"]]}}
        for label, doc in cases.items():
            with self.subTest(label):
                run = self.tmp / label.replace(" ", "-")
                run.mkdir()
                write_body(run, doc)
                self.assertIsNone(animation.body_joints(run))

    def test_refuses_a_linked_body(self):
        real = self.tmp / "elsewhere"
        real.mkdir()
        write_body(real)
        (self.run / "04-body").symlink_to(real / "04-body")
        self.assertIsNone(animation.body_joints(self.run))

    def test_cache_follows_the_file(self):
        write_body(self.run)
        first = animation.body_joints(self.run)
        self.assertIs(animation.body_joints(self.run), first, "unchanged files are read once")
        doc = sections_doc()
        doc["sections"][0]["pivot"] = [5.0, 6.0, 7.0]
        (self.run / "04-body" / "sections.json").write_text(json.dumps(doc) + " " * 8)
        head = next(p for p in animation.body_joints(self.run)["parts"] if p["part"] == SECTIONS[0])
        self.assertEqual(head["pivot"], [5.0, 7.0, -6.0])


class TestAnimationEvent(TempDirCase):
    def test_the_play_event_takes_only_known_clips(self):
        from herogen.service.analytics import validate_batch

        batch = lambda clip: {"visitor_id": "a" * 32, "device": "desktop",
                              "events": [{"name": "animation_play", "props": {"clip": clip}}]}
        for clip in ("idle", "walk", "wave", "jump", "stop"):
            self.assertEqual(validate_batch(batch(clip))[2], [("animation_play", "clip", clip)])
        for bad in ("dance", "", 3):
            with self.assertRaises(ValueError):
                validate_batch(batch(bad))
