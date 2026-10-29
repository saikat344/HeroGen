"""Ground-truth bench render: CLI errors, and a procedurally built rigged figure rendered end to end."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

try:
    import numpy as np
    from PIL import Image

    from herogen.bench.__main__ import main
    from herogen.blender import bl

    HAVE_DEPS = True
except ImportError:  # numpy or Pillow missing
    HAVE_DEPS = False

# Builds a box figure in Blender, facing -Y, arms straight out along X (a T-pose), 2 units tall, with a rig
# (KayKit-style bone names) whose arm and leg bones carry the boxes, and one colour per part: the character's
# left arm red, its right arm blue. With "cuff", a yellow box around the left forearm that the rig does not
# deform. Run with --python-expr, then `-- <out.glb> <1 for rigged, 0 for not> <cuff|none>`.
BUILD = textwrap.dedent('''
    import sys
    import bpy
    import bmesh

    out, rigged, cuff = sys.argv[-3], sys.argv[-2] == "1", sys.argv[-1] == "cuff"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # (name, centre, size, colour, bone)
    PARTS = [
        ("Torso", (0, 0, 1.25), (0.6, 0.3, 0.7), (0.2, 0.7, 0.2), "spine"),
        ("Head", (0, 0, 1.8), (0.4, 0.4, 0.4), (0.9, 0.8, 0.6), "spine"),
        ("ArmUpperL", (0.6, 0, 1.5), (0.3, 0.16, 0.16), (0.9, 0.1, 0.1), "upperarm.l"),
        ("ArmLowerL", (0.9, 0, 1.5), (0.3, 0.14, 0.14), (0.9, 0.1, 0.1), "lowerarm.l"),
        ("ArmUpperR", (-0.6, 0, 1.5), (0.3, 0.16, 0.16), (0.1, 0.1, 0.9), "upperarm.r"),
        ("ArmLowerR", (-0.9, 0, 1.5), (0.3, 0.14, 0.14), (0.1, 0.1, 0.9), "lowerarm.r"),
        ("LegL", (0.15, 0, 0.45), (0.2, 0.22, 0.9), (0.3, 0.3, 0.3), "upperleg.l"),
        ("LegR", (-0.15, 0, 0.45), (0.2, 0.22, 0.9), (0.3, 0.3, 0.3), "upperleg.r"),
    ]
    if cuff:
        PARTS.append(("Cuff", (0.95, 0, 1.5), (0.1, 0.2, 0.2), (0.9, 0.8, 0.05), None))
    BONES = [  # name, head, tail, parent
        ("spine", (0, 0, 0.9), (0, 0, 1.6), None),
        ("upperarm.l", (0.45, 0, 1.5), (0.75, 0, 1.5), "spine"),
        ("lowerarm.l", (0.75, 0, 1.5), (1.05, 0, 1.5), "upperarm.l"),
        ("wrist.l", (1.05, 0, 1.5), (1.1, 0, 1.5), "lowerarm.l"),
        ("upperarm.r", (-0.45, 0, 1.5), (-0.75, 0, 1.5), "spine"),
        ("lowerarm.r", (-0.75, 0, 1.5), (-1.05, 0, 1.5), "upperarm.r"),
        ("wrist.r", (-1.05, 0, 1.5), (-1.1, 0, 1.5), "lowerarm.r"),
        ("upperleg.l", (0.15, 0, 0.9), (0.15, 0, 0.0), "spine"),
        ("foot.l", (0.15, 0, 0.0), (0.15, -0.1, 0.0), "upperleg.l"),
        ("upperleg.r", (-0.15, 0, 0.9), (-0.15, 0, 0.0), "spine"),
        ("foot.r", (-0.15, 0, 0.0), (-0.15, -0.1, 0.0), "upperleg.r"),
    ]
    arm = None
    if rigged:
        data = bpy.data.armatures.new("Rig")
        arm = bpy.data.objects.new("Rig", data)
        bpy.context.scene.collection.objects.link(arm)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        for name, head, tail, parent in BONES:
            b = data.edit_bones.new(name)
            b.head, b.tail = head, tail
            if parent:
                b.parent = data.edit_bones[parent]
        bpy.ops.object.mode_set(mode="OBJECT")
    for name, c, s, col, bone in PARTS:
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co = (c[0] + v.co.x * s[0], c[1] + v.co.y * s[1], c[2] + v.co.z * s[2])
        bm.to_mesh(me)
        bm.free()
        mat = bpy.data.materials.new(name + "Mat")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col, 1.0)
        me.materials.append(mat)
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        if arm is not None and bone is not None:
            g = o.vertex_groups.new(name=bone)
            g.add(list(range(len(me.vertices))), 1.0, "REPLACE")
            o.modifiers.new("Armature", "ARMATURE").object = arm
            o.parent = arm
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
''')

# An unrigged box saved as a .blend whose material is a bare legacy colour, with a solid orange image packed
# into the file but connected to nothing: the case of older asset files, rendered with --image.
BUILD_BLEND = textwrap.dedent('''
    import sys
    import bpy

    out = sys.argv[-1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.0))
    box = bpy.context.active_object
    box.scale = (0.5, 0.3, 1.0)
    mat = bpy.data.materials.new("Legacy")
    mat.use_nodes = False
    mat.diffuse_color = (0.1, 0.1, 0.1, 1.0)
    box.data.materials.append(mat)
    img = bpy.data.images.new("Skin", 8, 8)
    img.pixels = [1.0, 0.5, 0.0, 1.0] * 64
    img.pack()
    img.use_fake_user = True
    bpy.ops.wm.save_as_mainfile(filepath=out)
''')

TORSO_DEPTH = 0.3
TORSO_COLOUR = (0.2, 0.7, 0.2)  # linear, as built


def build_figure(path: Path, rigged: bool, cuff: bool = False) -> None:
    proc = subprocess.run(
        [bl.find_blender(), "--background", "--factory-startup", "--python-expr", BUILD, "--", str(path),
         "1" if rigged else "0", "cuff" if cuff else "none"], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not path.exists():
        raise RuntimeError(f"building the test figure failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class BlendFileTest(unittest.TestCase):
    def test_packed_image_named_by_override_is_the_base_colour(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset, run = Path(tmp) / "box.blend", Path(tmp) / "run"
            cmd = [bl.find_blender(), "--background", "--factory-startup", "--python-expr", BUILD_BLEND, "--",
                   str(asset)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            self.assertTrue(asset.exists(), proc.stdout[-1500:] + proc.stderr[-1500:])
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["render", "--asset", str(asset), "--out", str(run), "--size", "96",
                             "--samples-flat", "4", "--samples-lit", "4", "--image", "Legacy=Skin"])
            rec = json.loads((run / "stage-2.json").read_text())
            self.assertEqual(code, 0, json.dumps(rec["checks"]))
            self.assertEqual(rec["details"]["pose"], "as-shipped")
            img = np.asarray(Image.open(run / "views" / "front.png").convert("RGB")).astype(int)
            # the image's stored sRGB bytes (1.0, 0.5, 0.0 -> 255, 128, 0), shown unlit and unchanged
            np.testing.assert_allclose(img[48, 48], [255, 128, 0], atol=2)


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class CliTest(unittest.TestCase):
    def _main(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(list(args))
        return code, err.getvalue()

    def test_missing_asset_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = self._main("render", "--asset", os.path.join(tmp, "nope.glb"), "--out", tmp)
        self.assertEqual(code, 2)
        self.assertIn("asset not found", err)

    def test_unsupported_type_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "figure.stl"
            p.write_bytes(b"solid x")
            code, err = self._main("render", "--asset", str(p), "--out", tmp)
        self.assertEqual(code, 2)
        self.assertIn("unsupported asset type", err)

    def test_material_override_needs_a_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "figure.glb"
            p.write_bytes(b"glTF")
            with self.assertRaises(SystemExit) as cm:
                self._main("render", "--asset", str(p), "--out", tmp, "--image", "no-equals-sign")
        self.assertIn("MATERIAL=VALUE", str(cm.exception.code))


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class RenderTest(unittest.TestCase):
    """One rigged and one unrigged box figure, rendered small, checked against the geometry they were built
    from."""

    SIZE = 160

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.runs = {}
        for key, rigged, cuff in ((True, True, False), (False, False, False), ("cuff", True, True)):
            asset = root / f"figure-{key}.glb"
            build_figure(asset, rigged, cuff)
            run = root / f"run-{key}"
            extra = ["--bind-unrigged"] if cuff else []
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["render", "--asset", str(asset), "--out", str(run), "--size", str(cls.SIZE),
                             "--samples-flat", "4", "--samples-lit", "4", "--timeout", "900", *extra])
            cls.runs[key] = (code, run)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _record(self, rigged):
        code, run = self.runs[rigged]
        record = run / "stage-2.json"
        self.assertEqual(code, 0, record.read_text() if record.exists() else "")
        return json.loads((run / "stage-2.json").read_text()), run

    def test_layout_and_records(self):
        rec, run = self._record(True)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["reference"], "ground-truth")
        meta = json.loads((run / "views" / "meta.json").read_text())
        self.assertEqual(meta["image_size"], self.SIZE)
        self.assertEqual(meta["height_units"], 5.5)
        self.assertEqual(set(meta["mask_method"].values()), {"render"})
        cams = json.loads((run / "cameras.json").read_text())
        self.assertEqual(len(cams["building"]), 8)
        self.assertEqual(cams["evaluation"], ["az025", "az335", "az065", "az295"])
        for cam in cams["cameras"]:
            for rel in cam["files"].values():
                self.assertTrue((run / rel).exists(), rel)
        for v in ("front", "left", "back", "right"):
            self.assertTrue((run / "views" / f"{v}.png").exists())
            self.assertTrue((run / "masks" / f"{v}.png").exists())
        self.assertTrue((run / "gt" / "body.glb").exists())
        self.assertEqual(rec["details"]["triangles"], 8 * 12)

    def test_rigged_figure_is_put_in_an_a_pose(self):
        rec, _ = self._record(True)
        self.assertEqual(rec["details"]["pose"], "a-pose")
        arms = {x["side"]: x for x in rec["details"]["limbs"] if x["limb"] == "arm"}
        self.assertEqual(set(arms), {"L", "R"})
        for x in arms.values():
            self.assertAlmostEqual(x["before_deg_below_horizontal"], 0.0, places=3)
            self.assertAlmostEqual(x["after_deg_below_horizontal"], 45.0, places=3)

    def test_unrigged_figure_keeps_its_shipped_pose(self):
        rec, run = self._record(False)
        self.assertEqual(rec["details"]["pose"], "as-shipped")
        self.assertEqual(rec["details"]["limbs"], [])
        # T-pose kept: the front view spans the outstretched arms, 2.1 units before scaling
        front = np.asarray(Image.open(run / "masks" / "front.png")) > 127
        ppu = json.loads((run / "views" / "meta.json").read_text())["pixels_per_unit"]
        width = np.flatnonzero(front.any(axis=0))
        scale = rec["details"]["contract_scale"]
        self.assertAlmostEqual((width[-1] - width[0] + 1) / ppu, 2.1 * scale, delta=1.5 / ppu)

    def test_the_characters_left_arm_is_on_the_image_right_in_the_front_view(self):
        _, run = self._record(True)
        img = np.asarray(Image.open(run / "views" / "front.png").convert("RGB")).astype(int)
        mask = np.asarray(Image.open(run / "masks" / "front.png")) > 127
        red = mask & (img[..., 0] > 200) & (img[..., 1] < 120) & (img[..., 2] < 120)
        blue = mask & (img[..., 2] > 200) & (img[..., 0] < 120) & (img[..., 1] < 120)
        self.assertGreater(red.sum(), 20)
        self.assertGreater(blue.sum(), 20)
        self.assertGreater(np.nonzero(red)[1].mean(), self.SIZE / 2)
        self.assertLess(np.nonzero(blue)[1].mean(), self.SIZE / 2)

    def _torso_rows(self, run, scale):
        """Rows of the centre column where the front view sees the torso's front face."""
        depth = np.load(run / "depth" / "front.npy")[:, self.SIZE // 2]
        return np.flatnonzero(np.abs(depth + TORSO_DEPTH / 2 * scale) < 1e-4)

    def test_flat_render_is_the_unlit_base_colour(self):
        rec, run = self._record(True)
        img = np.asarray(Image.open(run / "views" / "front.png").convert("RGB")).astype(int)
        rows = self._torso_rows(run, rec["details"]["contract_scale"])
        r = int(rows[len(rows) // 2])
        srgb = [round(255 * (1.055 * c ** (1 / 2.4) - 0.055)) for c in TORSO_COLOUR]
        np.testing.assert_allclose(img[r, self.SIZE // 2], srgb, atol=2)

    def test_mask_height_and_depth_match_the_geometry(self):
        rec, run = self._record(True)
        ppu = rec["details"]["pixels_per_unit"]
        scale = rec["details"]["contract_scale"]
        front = np.asarray(Image.open(run / "masks" / "front.png")) > 127
        rows = np.flatnonzero(front.any(axis=1))
        self.assertLessEqual(abs((rows[-1] - rows[0] + 1) - 5.5 * ppu), 1.0)
        depth = np.load(run / "depth" / "front.npy")
        self.assertTrue(np.isnan(depth[~front]).all())
        self.assertFalse(np.isnan(depth[front]).any())
        # torso (0.7 tall) and head (0.4 deep) front faces, seen down the centre column
        torso = self._torso_rows(run, scale)
        self.assertAlmostEqual(len(torso) / ppu, 0.7 * scale, delta=1.5 / ppu)
        head = np.flatnonzero(np.abs(depth[:, self.SIZE // 2] + 0.2 * scale) < 1e-4)
        self.assertAlmostEqual(len(head) / ppu, 0.4 * scale, delta=1.5 / ppu)
        self.assertLess(head.max(), torso.min())  # the head is above the torso

    def test_an_unrigged_part_bound_to_the_rig_follows_the_arm(self):
        rec, run = self._record("cuff")
        self.assertEqual(rec["details"]["pose"], "a-pose")
        self.assertTrue(any(r.startswith("Cuff: bound to") for r in rec["details"]["blender_report"]))
        img = np.asarray(Image.open(run / "views" / "front.png").convert("RGB")).astype(int)
        mask = np.asarray(Image.open(run / "masks" / "front.png")) > 127
        yellow = mask & (img[..., 0] > 200) & (img[..., 1] > 180) & (img[..., 2] < 120)
        red = mask & (img[..., 0] > 200) & (img[..., 1] < 120) & (img[..., 2] < 120)
        self.assertGreater(yellow.sum(), 10)
        # the cuff sits 83% of the way down the lowered arm; left behind at the T-pose it would sit on the
        # shoulder line, at the top of the arm's rows
        ry, cy = np.nonzero(yellow)
        rr, _ = np.nonzero(red)
        self.assertGreater(ry.mean() - rr.min(), 0.5 * (rr.max() - rr.min()))
        self.assertLess(ry.mean(), rr.max())
        self.assertGreater(cy.mean(), self.SIZE / 2)

    def test_heldout_views_are_not_building_views(self):
        _, run = self._record(True)
        meta = json.loads((run / "views" / "meta.json").read_text())
        self.assertFalse(set(meta["views"]) & set(meta["heldout_views"]))
        for v in meta["heldout_views"]:
            self.assertTrue((run / "heldout" / f"{v}-mask.png").exists())
            self.assertFalse((run / "views" / f"{v}.png").exists())


if __name__ == "__main__":
    unittest.main()
