"""Rest shape preservation against a received mesh, using synthetic geometry and local Blender only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from herogen.blender import bl, split, views
from herogen.blender.stage3 import IOU_GATE, iou
from herogen.blender.stage4 import _rest_iou_checks, _save_lod, _shape_views, run_stage4


class TestShapeViews(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="herogen-shape-views-test-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.vs = SimpleNamespace(views=dict.fromkeys(("front", "left", "top")),
                                  meta={"mask_method": {"top": "visual_hull"}})
        (self.run / "03-mesh.glb").touch()

    def write_stage3(self, data):
        (self.run / "stage-3.json").write_text(json.dumps(data))

    def assert_stops(self, message):
        # A failed retry must replace an earlier pass before any Blender job can run.
        (self.run / "stage-4.json").write_text('{"ok": true}')
        with patch("herogen.blender.stage4.load_views", return_value=self.vs), \
                patch("herogen.blender.stage4.bl.run_job") as job:
            with self.assertRaisesRegex(ValueError, message):
                run_stage4(self.run)
        job.assert_not_called()
        rec = json.loads((self.run / "stage-4.json").read_text())
        self.assertFalse(rec["ok"])
        c = rec["checks"][0]
        self.assertEqual(c["name"], "shape_views")
        self.assertFalse(c["pass"])
        self.assertTrue(c["gate"])
        self.assertRegex(c["note"], message)

    def test_invalid_shape_views_stop_with_failed_record(self):
        for source in ("carve", "sam3d-objects"):
            for value in ("front", ["bogus"], [], None, 42, ["front", 42]):
                with self.subTest(source=source, shape_views=value):
                    self.write_stage3({"mesh_source": source, "shape_views": value})
                    self.assert_stops("shape_views must be a non-empty list of known view names")

    def test_hybrid_requires_front(self):
        self.write_stage3({"mesh_source": "sam3d-objects", "shape_views": ["left"]})
        self.assert_stops('sam3d-objects must include "front"')

    def test_missing_shape_views_uses_source_default_and_notes_check(self):
        for source, expected in (("carve", ["front", "left"]), ("sam3d-objects", ["front"])):
            with self.subTest(source=source):
                self.write_stage3({"mesh_source": source})
                actual_source, names, c = _shape_views(self.run, self.vs)
                self.assertEqual(actual_source, source)
                self.assertEqual(names, set(expected))
                self.assertEqual(c["value"], expected)
                self.assertTrue(c["pass"])
                self.assertIn(f"using the {source} default: " + ", ".join(expected), c["note"])

    def test_explicit_known_views_are_preserved(self):
        self.write_stage3({"mesh_source": "sam3d-objects", "shape_views": ["front", "back"]})
        _, names, c = _shape_views(self.run, self.vs)
        self.assertEqual(names, {"front", "back"})
        self.assertNotIn("note", c)

    def test_missing_stage3_record_stops(self):
        self.assert_stops("needs stage-3.json")

    def test_unreadable_record_and_unknown_source_stop(self):
        for text, message in (("{", "cannot read stage-3.json as JSON"),
                              ('[]', "mesh_source must be"),
                              ('{}', "mesh_source must be"),
                              ('{"mesh_source": "bogus"}', "mesh_source must be")):
            with self.subTest(text=text):
                (self.run / "stage-3.json").write_text(text)
                self.assert_stops(message)


def box(z0, z1, depth=0.5):
    # The received mesh has already been registered along Y; keep that offset in every reference.
    verts = np.array([[x, y + 0.2, z] for z in (z0, z1) for y in (-depth / 2, depth / 2)
                      for x in (-0.5, 0.5)])
    faces = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                      [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return verts, faces


@unittest.skipUnless(bl.find_blender(), "Blender is required")
class TestRestReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="herogen-rest-test-")
        cls.root = Path(cls.temp.name)
        # Turnaround profiles are twice as deep as the received body. The front agrees exactly.
        masks = {}
        for name in ("front", "back", "left", "right", "top"):
            m = np.zeros((128, 128), bool)
            m[24:104, 44:84] = True
            masks[name] = m
        cls.vs = views.views_from_masks(masks, 2.0, {"mask_method": {"top": "visual_hull"}})
        cls.cameras = [v.camera() for v in cls.vs.views.values()]
        verts, faces = box(0, 2)
        src = cls.root / "received.npz"
        np.savez(src, verts=verts, faces=faces)
        tasks = [{"task": "silhouettes", "mesh": str(src), "cameras": cls.cameras,
                  "glb": str(cls.root / "received.glb"), "out_dir": str(cls.root / "reference")}]
        for variant, depth in (("equal", 0.5), ("distorted", 1.0)):
            sections = {}
            for name, z0, z1 in (("LowerTorso", 0, 1), ("UpperTorso", 1, 2)):
                v, f = box(z0, z1, depth)
                sections[name] = split.Section(name, v, f, np.array([0, 0.2, z0]), [], 0)
            path = cls.root / f"{variant}.npz"
            _save_lod(path, sections)
            tasks.append({"task": "body", "lods": {"high": str(path)}, "colors": {},
                          "rest_cameras": cls.cameras, "rest_dir": str(cls.root / variant)})
        tasks.append({"task": "silhouettes", "mesh": str(cls.root / "received.glb"),
                      "cameras": cls.cameras, "out_dir": str(cls.root / "glb-reference")})
        results = bl.run_job({"tasks": tasks}, cls.root, "rest", timeout=120)["results"]
        cls.reference = results[0]["renders"]
        cls.equal = results[1]["rest_silhouettes"]
        cls.distorted = results[2]["rest_silhouettes"]
        cls.glb_reference = results[3]["renders"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def checks(self, rendered, shape_views=None, reference=None):
        shape_views = {"front"} if shape_views is None else shape_views
        values, rows = _rest_iou_checks(self.vs, shape_views, "sam3d-objects", rendered,
                                        self.reference if reference is None else reference)
        return values, {c["name"]: c for c in rows}

    def test_hybrid_preserves_received_shape_despite_turnaround_sides(self):
        values, checks = self.checks(self.equal)
        self.assertTrue(all(c["pass"] for c in checks.values() if c["gate"]))
        for view in ("left", "right"):
            c = checks[f"rest_iou_with_covers_{view}"]
            self.assertEqual(c["threshold"], IOU_GATE)
            self.assertEqual(c["reference"], "stage3_mesh")
            self.assertEqual(values[view], 1.0)
            old = checks[f"rest_iou_with_covers_turnaround_{view}"]
            self.assertLess(old["value"], IOU_GATE)  # The previous gate rejected this unchanged body.
            self.assertFalse(old["gate"])
            self.assertIn("shape not built from this view", old["note"])
        self.assertEqual(checks["rest_iou_with_covers_back"]["reference"], "stage3_mesh")
        self.assertEqual(checks["rest_iou_with_covers_front"]["reference"], "turnaround")
        self.assertFalse(checks["rest_iou_with_covers_top"]["gate"])
        self.assertEqual(checks["rest_iou_with_covers_top"]["reference"], "turnaround")
        self.assertNotIn("rest_iou_with_covers_turnaround_top", checks)

    def test_sectioning_distortion_fails_even_when_it_matches_turnaround(self):
        _, checks = self.checks(self.distorted)
        for view in ("left", "right"):
            c = checks[f"rest_iou_with_covers_{view}"]
            self.assertTrue(c["gate"])
            self.assertFalse(c["pass"])
            self.assertLess(c["value"], IOU_GATE)
            self.assertGreater(checks[f"rest_iou_with_covers_turnaround_{view}"]["value"], c["value"])

    def test_carve_keeps_every_turnaround_comparison(self):
        values, checks = self.checks(self.equal, set(self.vs.views), {})
        self.assertEqual(len(checks), len(self.vs.views))
        for view, path in self.equal.items():
            expected = round(iou(views.load_mask(Path(path)), self.vs.views[view].mask), 4)
            c = checks[f"rest_iou_with_covers_{view}"]
            self.assertEqual(values[view], expected)
            self.assertEqual(c["value"], expected)
            self.assertEqual(c["pass"], expected >= IOU_GATE)
            self.assertEqual(c["gate"], view != "top")
            self.assertEqual(c["reference"], "turnaround")
        # A carve's bad profile still fails; it must never fall back to the mesh reference.
        self.assertFalse(checks["rest_iou_with_covers_left"]["pass"])

    def test_glb_reference_preserves_registration_and_matches_npz(self):
        npz_values, _ = self.checks(self.equal)
        glb_values, checks = self.checks(self.equal, reference=self.glb_reference)
        self.assertEqual(glb_values, npz_values)
        self.assertTrue(all(c["pass"] for c in checks.values() if c["gate"]))


if __name__ == "__main__":
    unittest.main()
