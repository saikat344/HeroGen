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
# deform. With "turned", the whole figure faces +Y instead (a model exported the other way round); with
# "tilted", it is tilted 2 degrees about each axis, so no box edge runs along a pixel row or column. With
# "nolegs", the leg bones have names that are not limbs, so the rig has arms only.
# Run with --python-expr, then `-- <out.glb> <1 for rigged, 0 for not> <cuff|none|turned>`.
BUILD = textwrap.dedent('''
    import sys
    import bpy
    import bmesh

    out, rigged, extra = sys.argv[-3], sys.argv[-2] == "1", sys.argv[-1]
    cuff, turned = extra == "cuff", extra == "turned"
    rename = {"upperleg.l": "post.l", "upperleg.r": "post.r", "foot.l": "base.l", "foot.r": "base.r"}
    name = (lambda n: rename.get(n, n)) if extra == "nolegs" else (lambda n: n)
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
        for bone_name, head, tail, parent in BONES:
            b = data.edit_bones.new(name(bone_name))
            b.head, b.tail = head, tail
            if parent:
                b.parent = data.edit_bones[name(parent)]
        bpy.ops.object.mode_set(mode="OBJECT")
    for part, c, s, col, bone in PARTS:
        me = bpy.data.meshes.new(part)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co = (c[0] + v.co.x * s[0], c[1] + v.co.y * s[1], c[2] + v.co.z * s[2])
        bm.to_mesh(me)
        bm.free()
        mat = bpy.data.materials.new(part + "Mat")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col, 1.0)
        me.materials.append(mat)
        o = bpy.data.objects.new(part, me)
        bpy.context.scene.collection.objects.link(o)
        if arm is not None and bone is not None:
            g = o.vertex_groups.new(name=name(bone))
            g.add(list(range(len(me.vertices))), 1.0, "REPLACE")
            o.modifiers.new("Armature", "ARMATURE").object = arm
            o.parent = arm
    if turned:
        arm.rotation_euler[2] = 3.141592653589793
    if extra == "tilted":
        arm.rotation_euler = (0.03490658503988659,) * 3   # 2 degrees about each axis: no edge stays square
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
''')

# An unrigged box saved as a .blend whose material is a bare legacy colour, with a solid orange image packed
# into the file but connected to nothing: the case of older asset files, rendered with --image. With
# "hostile", the file also carries render settings that must not reach the bench's renders: a bright red
# world, film exposure, a 2:1 pixel aspect and a render border. Both carry a visible text object (geometry
# the bench skips and must report) and a keyframed box, whose animation the bench must clear. With "holes",
# the image is transparent (alpha 0) on its left half, as alpha cut-outs are.
BUILD_BLEND = textwrap.dedent('''
    import sys
    import bpy

    out, hostile = sys.argv[-1], sys.argv[-2] == "hostile"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.0))
    box = bpy.context.active_object
    box.scale = (0.5, 0.3, 1.0)
    mat = bpy.data.materials.new("Legacy")
    mat.use_nodes = False
    mat.diffuse_color = (0.1, 0.1, 0.1, 1.0)
    box.data.materials.append(mat)
    img = bpy.data.images.new("Skin", 8, 8, alpha=True)
    holes = sys.argv[-2] == "holes"
    img.pixels = [px for i in range(64) for px in (1.0, 0.5, 0.0, 0.0 if holes and i % 8 < 4 else 1.0)]
    img.pack()
    img.use_fake_user = True
    bpy.ops.object.text_add(location=(3.0, 0.0, 0.0))
    box.keyframe_insert("rotation_euler", frame=1)
    if hostile:
        scene = bpy.context.scene
        scene.world = bpy.data.worlds.new("Sunset")
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes["Background"]
        bg.inputs["Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = 50.0
        scene.cycles.film_exposure = 8.0
        scene.render.pixel_aspect_x = 2.0
        scene.render.use_border = True
        scene.render.border_min_x, scene.render.border_max_x = 0.4, 0.6
    bpy.ops.wm.save_as_mainfile(filepath=out)
''')

# A card (a 1 x 2 plane facing -Y) textured red, whose left half has alpha 0.35 (clipped at 0.5, and above
# it once sRGB-encoded, so it catches thresholding encoded bytes) and right half alpha 1, with a grey box
# behind it filling the card's lower half. "mask" wires the alpha as the glTF importer writes alphaMode MASK;
# "opaque" leaves it unwired (glTF OPAQUE) although the texture still carries it; "noise" takes the alpha from
# a procedural texture and "otherimage" from an image that is not the base colour, both of which only a
# .blend can carry; "samefile" takes it from the base-colour PNG loaded a second time, as FBX import does.
# Saved as GLB, or as .blend when the output name ends so.
BUILD_CARD = textwrap.dedent('''
    import sys
    import bpy

    out, variant = sys.argv[-2], sys.argv[-1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 1.0), rotation=(1.5707963267948966, 0, 0))
    card = bpy.context.active_object
    card.name = "Card"
    card.scale = (1.0, 2.0, 1.0)
    img = bpy.data.images.new("CardTex", 8, 8, alpha=True)
    img.pixels = [px for i in range(64) for px in (0.8, 0.1, 0.1, 0.35 if i % 8 < 4 else 1.0)]
    img.pack()
    mat = bpy.data.materials.new("CardMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if variant == "mask":
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.5
        sub = nt.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; sub.inputs[0].default_value = 1.0
        nt.links.new(tex.outputs["Alpha"], lt.inputs[0])
        nt.links.new(lt.outputs[0], sub.inputs[1])
        nt.links.new(sub.outputs[0], bsdf.inputs["Alpha"])
    elif variant == "noise":
        noise = nt.nodes.new("ShaderNodeTexNoise")
        nt.links.new(noise.outputs["Fac"], bsdf.inputs["Alpha"])
    elif variant == "samefile":
        # the same PNG loaded twice, as FBX import does: colour from one datablock, alpha from the other
        import os
        path = os.path.join(os.path.dirname(out), "card-tex.png")
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        img.unpack(method="REMOVE") if img.packed_file else None
        again = bpy.data.images.load(path, check_existing=False)
        tex2 = nt.nodes.new("ShaderNodeTexImage")
        tex2.image = again
        tex2.interpolation = "Closest"
        nt.links.new(tex2.outputs["Alpha"], bsdf.inputs["Alpha"])
    elif variant == "otherimage":
        other = bpy.data.images.new("OtherTex", 8, 8, alpha=True)
        other.pixels = [0.5, 0.5, 0.5, 0.2] * 64
        other.pack()
        tex2 = nt.nodes.new("ShaderNodeTexImage")
        tex2.image = other
        nt.links.new(tex2.outputs["Alpha"], bsdf.inputs["Alpha"])
    card.data.materials.append(mat)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.5, 0.5))
    box = bpy.context.active_object
    box.name = "Box"
    box.scale = (0.9, 0.2, 1.0)
    grey = bpy.data.materials.new("BoxMat")
    grey.use_nodes = True
    grey.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    box.data.materials.append(grey)
    if out.endswith(".blend"):
        bpy.ops.wm.save_as_mainfile(filepath=out)
    else:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
''')

TORSO_DEPTH = 0.3
TORSO_COLOUR = (0.2, 0.7, 0.2)  # linear, as built


def build_figure(path: Path, rigged: bool, extra: str = "none") -> None:
    proc = subprocess.run(
        [bl.find_blender(), "--background", "--factory-startup", "--python-expr", BUILD, "--", str(path),
         "1" if rigged else "0", extra], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not path.exists():
        raise RuntimeError(f"building the test figure failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class BlendFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = {}
        for mode, build, extra in (("plain", "plain", []), ("hostile", "hostile", []), ("holes", "holes", []),
                                   ("holes-clip", "holes", ["--alpha", "Legacy=clip"])):
            asset, run = Path(cls.tmp.name) / f"box-{build}.blend", Path(cls.tmp.name) / f"run-{mode}"
            if not asset.exists():
                cmd = [bl.find_blender(), "--background", "--factory-startup", "--python-expr", BUILD_BLEND,
                       "--", build, str(asset)]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                assert asset.exists(), proc.stdout[-1500:] + proc.stderr[-1500:]
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["render", "--asset", str(asset), "--out", str(run), "--size", "96",
                             "--samples-flat", "16", "--samples-lit", "16", "--image", "Legacy=Skin",
                             "--yaw", "0", *extra])
            cls.runs[mode] = (code, run)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, mode, expect=0):
        code, run = self.runs[mode]
        rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, expect, json.dumps(rec["checks"]))
        return rec, run

    def test_an_override_stays_opaque_unless_alpha_clip_is_asked_for(self):
        opaque, run = self._run("holes")
        self.assertEqual(opaque["details"]["alpha"]["Legacy"]["mode"], "opaque")
        full = np.asarray(Image.open(run / "masks" / "front.png")) > 127
        clipped, run = self._run("holes-clip")
        rec = clipped["details"]["alpha"]["Legacy"]
        self.assertEqual((rec["mode"], rec["cutoff"], rec["image"], rec["clipped_texel_share"]),
                         ("clip", 0.5, "Skin", 0.5))
        holed = np.asarray(Image.open(run / "masks" / "front.png")) > 127
        self.assertLess(holed.sum(), 0.8 * full.sum())          # the transparent half of the texture is gone
        self.assertGreater(holed.sum(), 0.2 * full.sum())
        glb = next(c for c in clipped["checks"] if c["name"] == "gt_glb_alpha_modes")
        self.assertTrue(glb["pass"])
        self.assertEqual(glb["written"]["Legacy"], "MASK")

    def test_packed_image_named_by_override_is_the_base_colour(self):
        rec, run = self._run("plain")
        self.assertEqual(rec["details"]["pose"], "as-shipped")
        img = np.asarray(Image.open(run / "views" / "front.png").convert("RGB")).astype(int)
        # the image's stored sRGB bytes (1.0, 0.5, 0.0 -> 255, 128, 0), shown unlit and unchanged
        np.testing.assert_allclose(img[48, 48], [255, 128, 0], atol=2)

    def test_the_files_world_exposure_aspect_and_border_do_not_reach_the_renders(self):
        _, plain = self._run("plain")
        _, hostile = self._run("hostile")
        for rel in ("views/front.png", "views-lit/front.png", "heldout/front-left-25-lit.png",
                    "masks/front.png"):
            a = np.asarray(Image.open(plain / rel).convert("RGB")).astype(int)
            b = np.asarray(Image.open(hostile / rel).convert("RGB")).astype(int)
            self.assertEqual(a.shape, b.shape, rel)
            self.assertLess(np.abs(a - b).mean(), 1.0, rel)

    def test_skipped_geometry_and_cleared_animation_are_recorded(self):
        rec, _ = self._run("plain")
        self.assertEqual(rec["details"]["skipped_geometry"], ["Text (font)"])
        self.assertTrue(any("cleared object action" in r for r in rec["details"]["blender_report"]))
        check = next(c for c in rec["checks"] if c["name"] == "skipped_visible_geometry")
        self.assertEqual(check["value"], 1)
        self.assertFalse(check["gate"])

    def test_logs_name_no_home_folder_or_work_folder(self):
        from herogen.bench.render import HOME_PATH

        _, run = self._run("plain")
        log = (run / "logs" / "bench-render.log").read_text(encoding="utf-8")
        self.assertIn("Blender", log)
        self.assertIsNone(HOME_PATH.search(log), HOME_PATH.search(log))
        self.assertNotIn(str(run.resolve()), log)
        self.assertNotIn(self.tmp.name, log)


def limb(limb, side, error, converged=True):
    return {"limb": limb, "side": side, "error_deg": error, "converged": converged, "steps": 1,
            "error_basis": "3D direction"}


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class PoseCheckTest(unittest.TestCase):
    def setUp(self):
        from herogen.bench.render import pose_checks

        self.pose_checks = pose_checks

    def _failed(self, limbs):
        return [c["name"] for c in self.pose_checks("a-pose", limbs) if not c["pass"]]

    def test_four_converged_limbs_pass(self):
        limbs = [limb("arm", "L", 0.0), limb("arm", "R", 0.004), limb("leg", "L", 0.0), limb("leg", "R", 0.0)]
        self.assertEqual(self._failed(limbs), [])

    def test_missing_legs_fail(self):
        self.assertEqual(self._failed([limb("arm", "L", 0.0), limb("arm", "R", 0.0)]), ["pose_limbs_posed"])

    def test_an_arm_at_the_right_height_but_pointing_forward_fails(self):
        # 45 degrees down but swung 30 degrees forward: the elevation alone would read 45
        limbs = [limb("arm", "L", 30.0, converged=False), limb("arm", "R", 0.0), limb("leg", "L", 0.0),
                 limb("leg", "R", 0.0)]
        self.assertEqual(self._failed(limbs), ["pose_arm_L_error_deg"])

    def test_a_limb_that_did_not_converge_fails(self):
        limbs = [limb("arm", "L", 0.0), limb("arm", "R", 0.0), limb("leg", "L", 0.005, converged=False),
                 limb("leg", "R", 0.0)]
        self.assertEqual(self._failed(limbs), ["pose_leg_L_error_deg"])

    def test_no_pose_checks_for_other_poses(self):
        self.assertEqual(self.pose_checks("rest", []), [])


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class RenderMaskAgreementTest(unittest.TestCase):
    def setUp(self):
        from herogen.bench.render import near_edge, off_edge_disagreement

        self.near_edge, self.off_edge = near_edge, off_edge_disagreement
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[10:30, 12:20] = True       # a thin bar, 8 px wide

    def test_edge_band_is_one_pixel_each_side(self):
        band = self.near_edge(self.mask)
        self.assertTrue(band[10, 15] and band[9, 15])      # first inside and first outside row
        self.assertFalse(band[12, 15] or band[8, 15])

    def test_coverage_differing_only_on_the_edge_passes(self):
        cover = self.mask.astype(float)
        cover[10, 12:20] = 0.4                              # a less-than-half-covered top row
        cover[9, 14] = 0.8                                  # a pixel just outside, mostly covered
        self.assertEqual(self.off_edge(cover, self.mask), 0)

    def test_a_sliver_between_pixel_centres_is_within_the_margin(self):
        from herogen.bench.render import coverage_margin

        cover = self.mask.astype(float)
        cover[34, 5] = 0.502                                # 8 of 16 samples hit a sliver no centre ray hit
        self.assertEqual(self.off_edge(cover, self.mask, coverage_margin(16)), 0)
        cover[34, 5] = 0.9                                  # well covered: that is not a sliver
        self.assertEqual(self.off_edge(cover, self.mask, coverage_margin(16)), 1)
        self.assertEqual(coverage_margin(16), 0.125)

    def test_a_shifted_camera_is_caught_at_any_size(self):
        from herogen.bench.render import coverage_margin

        shifted = np.roll(self.mask, 3, axis=1).astype(float)
        self.assertGreater(self.off_edge(shifted, self.mask, coverage_margin(16)), 0)
        big = np.kron(self.mask, np.ones((8, 8), dtype=bool))
        self.assertGreater(self.off_edge(np.roll(big, 3, axis=1).astype(float), big, coverage_margin(16)), 0)


def build_card(path: Path, variant: str) -> None:
    cmd = [bl.find_blender(), "--background", "--factory-startup", "--python-expr", BUILD_CARD, "--",
           str(path), variant]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not path.exists():
        raise RuntimeError(f"building the card failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class AlphaClipTest(unittest.TestCase):
    """Texture alpha below 0.5 is a hole in every output alike: flat and lit renders, masks, depth, GLB."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.runs = {}
        for variant in ("mask", "opaque"):
            asset, run = root / f"card-{variant}.glb", root / f"run-{variant}"
            build_card(asset, variant)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["render", "--asset", str(asset), "--out", str(run), "--size", "128",
                             "--yaw", "0", "--samples-flat", "16", "--samples-lit", "16"])
            cls.runs[variant] = (code, run)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, variant):
        code, run = self.runs[variant]
        rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, 0, json.dumps(rec["checks"]))
        return rec, run

    def _probe(self, run):
        """Front view: mask, depth, flat and lit colour at the card's upper-left (alpha 0.35, nothing behind),
        lower-left (alpha 0.35, the box behind) and upper-right (alpha 1)."""
        m = np.asarray(Image.open(run / "masks" / "front.png")) > 127
        d = np.load(run / "depth" / "front.npy")
        flat = np.asarray(Image.open(run / "views" / "front.png").convert("RGB")).astype(int)
        lit = np.asarray(Image.open(run / "views-lit" / "front.png").convert("RGB")).astype(int)
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        top, bot, left, right = rows[0], rows[-1], cols[0], cols[-1]
        mid = (left + right) // 2
        upper, lower = top + (bot - top) // 4, bot - (bot - top) // 4
        lc, rc = left + (mid - left) // 2, mid + (right - mid) // 2
        return {name: (bool(m[r, c]), float(d[r, c]), flat[r, c].tolist(), lit[r, c].tolist())
                for name, (r, c) in {"upper_left": (upper, lc), "lower_left": (lower, lc),
                                     "upper_right": (upper, rc), "lower_right": (lower, rc)}.items()}

    def test_clipped_texels_are_holes_in_the_mask_and_both_renders(self):
        rec, run = self._run("mask")
        p = self._probe(run)
        self.assertEqual(p["upper_left"], (False, p["upper_left"][1], [255, 255, 255], [255, 255, 255]))
        self.assertTrue(np.isnan(p["upper_left"][1]))
        self.assertTrue(p["upper_right"][0])
        # the texture's stored bytes, unlit
        np.testing.assert_allclose(p["upper_right"][2], [204, 26, 26], atol=3)
        alpha = rec["details"]["alpha"]["CardMat"]
        self.assertEqual((alpha["mode"], alpha["cutoff"], alpha["clipped_texel_share"]), ("clip", 0.5, 0.5))

    def test_a_ray_through_a_clipped_texel_finds_the_surface_behind(self):
        _, run = self._run("mask")
        p = self._probe(run)
        self.assertTrue(p["lower_left"][0])                    # the box, seen through the hole
        self.assertGreater(p["lower_left"][1], p["lower_right"][1] + 0.3)   # deeper than the card
        np.testing.assert_allclose(p["lower_left"][2], [149, 149, 149], atol=3)   # the box's grey

    def test_the_glb_reads_back_as_mask(self):
        from herogen.bench.render import glb_materials

        rec, run = self._run("mask")
        written = glb_materials(run / "gt" / "body.glb")
        self.assertEqual(written["CardMat"], {"alphaMode": "MASK", "alphaCutoff": 0.5})
        self.assertEqual(written["BoxMat"]["alphaMode"], "OPAQUE")
        self.assertTrue(next(c for c in rec["checks"] if c["name"] == "gt_glb_alpha_modes")["pass"])

    def test_a_gltf_opaque_material_is_not_clipped_although_its_texture_has_alpha(self):
        from herogen.bench.render import glb_materials

        rec, run = self._run("opaque")
        p = self._probe(run)
        self.assertTrue(p["upper_left"][0])
        np.testing.assert_allclose(p["upper_left"][2], [204, 26, 26], atol=3)
        self.assertEqual(rec["details"]["alpha"]["CardMat"]["mode"], "opaque")
        self.assertEqual(glb_materials(run / "gt" / "body.glb")["CardMat"]["alphaMode"], "OPAQUE")

    def test_the_same_texture_file_loaded_twice_counts_as_the_base_colour(self):
        asset, run = Path(self.tmp.name) / "card-samefile.blend", Path(self.tmp.name) / "r-samefile"
        build_card(asset, "samefile")
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["render", "--asset", str(asset), "--out", str(run), "--size", "128", "--yaw", "0",
                         "--samples-flat", "16", "--samples-lit", "4"])
        rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, 0, json.dumps([c for c in rec["checks"] if not c["pass"]]))
        alpha = rec["details"]["alpha"]["CardMat"]
        self.assertEqual((alpha["mode"], alpha["clipped_texel_share"]), ("clip", 0.5))
        self.assertFalse(self._probe(run)["upper_left"][0])     # the alpha 0.35 half is a hole

    def test_alpha_the_masks_cannot_sample_is_refused(self):
        for variant, words in (("noise", "cannot sample"), ("otherimage", "not its base-colour texture")):
            asset = Path(self.tmp.name) / f"card-{variant}.blend"
            build_card(asset, variant)
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                out = Path(self.tmp.name) / f"r-{variant}"
                code = main(["render", "--asset", str(asset), "--out", str(out), "--size", "64",
                             "--yaw", "0"])
            self.assertEqual(code, 2, variant)
            self.assertIn(words, err.getvalue(), variant)


def ellipse_coverage(size, cx, cy, rx, ry, sub=8):
    """Coverage (0..1) of an ellipse per pixel from sub x sub samples, and the mask from pixel centres."""
    offsets = (np.arange(sub) + 0.5) / sub
    rows, cols = np.indices((size, size), dtype=np.float64)
    cover = np.zeros((size, size))
    for dy in offsets:
        for dx in offsets:
            cover += (((cols + dx - cx) / rx) ** 2 + ((rows + dy - cy) / ry) ** 2) <= 1.0
    centre = (((cols + 0.5 - cx) / rx) ** 2 + ((rows + 0.5 - cy) / ry) ** 2) <= 1.0
    return cover / sub ** 2, centre


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class RegistrationTest(unittest.TestCase):
    """The registration gates see what the edge gate cannot: a half-pixel shift and a 0.2% scale error."""

    def setUp(self):
        from herogen.bench import render

        self.r = render
        self.cover, self.mask = ellipse_coverage(400, 200.0, 210.0, 90.0, 170.0)

    def test_an_aligned_render_is_within_both_gates(self):
        c, a = self.r.registration(self.cover, self.mask)
        self.assertLessEqual(c, self.r.CENTROID_MAX_PX)
        self.assertLessEqual(a, self.r.AREA_RATIO_MAX_DEV)

    def test_a_half_pixel_shift_fails_the_centroid_gate(self):
        shifted, _ = ellipse_coverage(400, 200.5, 210.0, 90.0, 170.0)
        c, _ = self.r.registration(shifted, self.mask)
        self.assertAlmostEqual(c, 0.5, delta=0.05)
        self.assertGreater(c, self.r.CENTROID_MAX_PX)
        self.assertEqual(self.r.off_edge_disagreement(shifted, self.mask, self.r.coverage_margin(16)), 0)

    def test_a_scale_error_of_a_fifth_of_a_percent_fails_the_area_gate(self):
        bigger, _ = ellipse_coverage(400, 200.0, 210.0, 90.0 * 1.002, 170.0 * 1.002)
        _, a = self.r.registration(bigger, self.mask)
        self.assertAlmostEqual(a, 0.004, delta=0.0006)
        self.assertGreater(a, self.r.AREA_RATIO_MAX_DEV)

    def test_the_gates_come_from_the_measured_runs(self):
        m = self.r.REGISTRATION_MEASURED
        margin = self.r.REGISTRATION_MARGIN
        self.assertAlmostEqual(self.r.CENTROID_MAX_PX, round(m["centroid_px"] * margin, 4))
        self.assertAlmostEqual(self.r.AREA_RATIO_MAX_DEV, round(m["area_ratio_dev"] * margin, 5))

    def test_the_gates_apply_at_the_measured_size_and_are_recorded_below_it(self):
        bad = {"front": 0.6}
        for size, gate in ((1024, True), (2048, True), (512, False)):
            checks = self.r.registration_checks(bad, {"front": 0.01}, size)
            self.assertEqual([c["gate"] for c in checks], [gate, gate], size)
            self.assertEqual([c["pass"] for c in checks], [not gate, not gate], size)
        ok = self.r.registration_checks({"front": 0.05}, {"front": 0.0001}, 1024)
        self.assertTrue(all(c["pass"] and c["gate"] for c in ok))

    def test_coverage_noise_allowance_follows_the_sample_count(self):
        self.assertEqual(self.r.coverage_margin(16), 0.125)
        self.assertEqual(self.r.coverage_margin(64), 0.03125)


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class RegistrationGateTest(unittest.TestCase):
    """At the size the registration thresholds were measured at, a correct render passes them as gates.

    The box figure is tilted 2 degrees about each axis. Square to the cameras, a long straight edge crosses
    every pixel along it at the same sub-pixel offset, so the render's sampled coverage and the centre ray
    differ the same way all along it instead of averaging out: the square figure reads 0.84 px of centroid
    and 0.0015 to 0.002 of area at 1024 px with no camera error. The 17 delivered runs, organic and blocky
    alike, reach 0.11 px and 0.00036."""

    def test_a_correct_render_at_1024_passes_both_gates(self):
        from herogen.bench import render

        with tempfile.TemporaryDirectory() as tmp:
            asset, run = Path(tmp) / "tilted.glb", Path(tmp) / "run"
            build_figure(asset, True, "tilted")
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["render", "--asset", str(asset), "--out", str(run), "--size", "1024",
                             "--samples-flat", "16", "--samples-lit", "2"])
            rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, 0, json.dumps([c for c in rec["checks"] if not c["pass"]]))
        checks = {c["name"]: c for c in rec["checks"]}
        for name, limit in (("render_mask_centroid_px", render.CENTROID_MAX_PX),
                            ("render_mask_area_ratio_dev", render.AREA_RATIO_MAX_DEV)):
            self.assertTrue(checks[name]["gate"], name)
            self.assertTrue(checks[name]["pass"], name)
            self.assertLessEqual(checks[name]["value"], limit, name)
        self.assertEqual(rec["details"]["yaw_deg"], 0.0)       # 2 degrees snaps to facing -Y


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class FacingCheckTest(unittest.TestCase):
    def _check(self, deg):
        import math

        from herogen.bench.render import facing_check

        a = math.radians(deg)
        return facing_check({"checked": True, "rig_axis": "Rig: R -> L",
                             "left_axis_after_yaw": [math.cos(a), math.sin(a)]})

    def test_hip_joints_a_little_off_square_pass(self):
        for deg in (0.0, 2.58, 7.0):
            c = self._check(deg)
            self.assertTrue(c["pass"], deg)
            self.assertAlmostEqual(c["residual_deg"], deg, places=2)

    def test_a_figure_turned_the_wrong_way_fails(self):
        for deg in (10.0, 45.0, 90.0, 180.0):
            self.assertFalse(self._check(deg)["pass"], deg)

    def test_the_threshold_comes_from_the_measured_runs(self):
        import math

        from herogen.bench import render

        allowed = render.FACING_MEASURED_DEG * render.FACING_MARGIN
        self.assertAlmostEqual(render.FACING_MIN_X, math.cos(math.radians(allowed)), places=4)


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

    def _dummy(self, tmp):
        p = Path(tmp) / "figure.glb"
        p.write_bytes(b"glTF")
        return p

    def test_bad_numbers_and_patterns_are_clear_errors(self):
        cases = ((["--size", "0"], "--size"), (["--size", "100000"], "--size"),
                 (["--height", "-1"], "--height"), (["--height", "nan"], "--height"),
                 (["--include", "("], "--include"), (["--exclude", "[a-"], "--exclude"),
                 (["--samples-lit", "0"], "--samples-lit"), (["--samples-flat", "8"], "--samples-flat"))
        with tempfile.TemporaryDirectory() as tmp:
            asset = self._dummy(tmp)
            for args, flag in cases:
                out = os.path.join(tmp, "r")
                code, err = self._main("render", "--asset", str(asset), "--out", out, *args)
                self.assertEqual(code, 2, args)
                self.assertIn(flag, err, args)
                self.assertNotIn("Traceback", err, args)

    def test_a_non_empty_folder_is_not_overwritten_by_default_or_cleared_unless_it_is_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset, out = self._dummy(tmp), Path(tmp) / "notes"
            out.mkdir()
            (out / "keep.txt").write_text("someone else's file")
            code, err = self._main("render", "--asset", str(asset), "--out", str(out))
            self.assertEqual(code, 2)
            self.assertIn("--overwrite", err)
            code, err = self._main("render", "--asset", str(asset), "--out", str(out), "--overwrite")
            self.assertEqual(code, 2)
            self.assertIn("refusing to clear it", err)
            self.assertTrue((out / "keep.txt").exists())

    def test_scrub_paths_hides_folders_and_home_paths(self):
        from herogen.bench.render import scrub_paths

        work = Path(tempfile.gettempdir()) / "herogen-bench-abc"
        windows_home = "\\".join(["C:", "Users", "pat", "x.blend"])   # built, so no home path sits in git
        posix_home = "/".join(["", "home", "pat", "y"])
        text = f"Saved: '{work}{os.sep}views{os.sep}front.png'; read {windows_home} and {posix_home}"
        out = scrub_paths(text, {"<work>": work})
        self.assertIn("<work>", out)
        self.assertNotIn("pat", out)

    def _folder(self, tmp, files):
        out = Path(tmp) / "run"
        for rel, text in files.items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(text)
        return out

    def test_overwrite_leaves_pipeline_and_mixed_folders_alone(self):
        from herogen.bench.render import not_a_bench_run

        pipeline = {"stage-2.json": json.dumps({"stage": 2, "ok": True}), "views/front.png": "x",
                    "masks/front.png": "x", "00-sketch.png": "x", "01-cover.png": "x"}
        bare_pipeline = {"stage-2.json": json.dumps({"stage": 2, "ok": True}), "views/meta.json": "{}",
                         "masks/front.png": "x", "cameras.json": "{}"}
        bench = {"bench-run.json": json.dumps({"tool": "herogen.bench"}), "stage-2.json": "{}",
                 "cameras.json": "{}", "views/front.png": "x"}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("00-sketch.png", not_a_bench_run(self._folder(tmp, pipeline)))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("ground-truth", not_a_bench_run(self._folder(tmp, bare_pipeline)))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(not_a_bench_run(self._folder(tmp, bench)))
        with tempfile.TemporaryDirectory() as tmp:
            mixed = dict(bench, **{"notes.txt": "mine"})
            self.assertIn("notes.txt", not_a_bench_run(self._folder(tmp, mixed)))

    def test_a_run_that_stopped_part_way_can_be_overwritten(self):
        from herogen.bench.render import not_a_bench_run

        partial = {"bench-run.json": json.dumps({"tool": "herogen.bench", "status": "started"}),
                   "logs/bench-render.log": "Blender failed"}
        older = {"stage-2.json": json.dumps({"reference": "ground-truth"}), "cameras.json": "{}"}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(not_a_bench_run(self._folder(tmp, partial)))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(not_a_bench_run(self._folder(tmp, older)))

    def test_alpha_mode_must_be_clip_or_opaque(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = self._dummy(tmp)
            with self.assertRaises(SystemExit) as cm:
                self._main("render", "--asset", str(asset), "--out", tmp, "--alpha", "Skin=blend")
        self.assertIn("clip or MATERIAL=opaque", str(cm.exception.code))

    def test_glb_materials_fills_in_the_gltf_defaults(self):
        import struct

        from herogen.bench.render import glb_materials

        doc = json.dumps({"materials": [{"name": "A", "alphaMode": "MASK"}, {"name": "B"},
                                        {"name": "C", "alphaMode": "MASK", "alphaCutoff": 0.3}]}).encode()
        doc += b" " * (-len(doc) % 4)
        blob = b"glTF" + struct.pack("<II", 2, 20 + len(doc)) + struct.pack("<I4s", len(doc), b"JSON") + doc
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.glb"
            path.write_bytes(blob)
            self.assertEqual(glb_materials(path), {"A": {"alphaMode": "MASK", "alphaCutoff": 0.5},
                                                   "B": {"alphaMode": "OPAQUE", "alphaCutoff": None},
                                                   "C": {"alphaMode": "MASK", "alphaCutoff": 0.3}})

    def test_material_override_needs_a_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "figure.glb"
            p.write_bytes(b"glTF")
            with self.assertRaises(SystemExit) as cm:
                self._main("render", "--asset", str(p), "--out", tmp, "--image", "no-equals-sign")
        self.assertIn("MATERIAL=VALUE", str(cm.exception.code))


def cams_name(cams: dict, azimuth: int) -> str:
    return next(name for name, c in cams.items() if c["azimuth_deg"] == azimuth)


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class FacingErrorTest(unittest.TestCase):
    """Facing that cannot be read or that contradicts the rig stops with a clear message, before posing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rigged, cls.unrigged = Path(cls.tmp.name) / "rigged.glb", Path(cls.tmp.name) / "unrigged.glb"
        build_figure(cls.rigged, True)
        build_figure(cls.unrigged, False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _main(self, asset, *args):
        err = io.StringIO()
        out = tempfile.mkdtemp(dir=self.tmp.name)
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["render", "--asset", str(asset), "--out", out, "--size", "64",
                         "--samples-flat", "16", "--samples-lit", "2", *args])
        return code, err.getvalue()

    def test_include_that_matches_nothing(self):
        code, err = self._main(self.rigged, "--include", "^NoSuchMesh$")
        self.assertEqual(code, 2)
        self.assertIn("no mesh objects selected", err)

    def test_a_pose_on_an_unrigged_figure(self):
        code, err = self._main(self.unrigged, "--pose", "a-pose", "--yaw", "0")
        self.assertEqual(code, 2)
        self.assertIn("needs both arms and both legs", err)

    def test_override_for_a_material_the_figure_does_not_use(self):
        code, err = self._main(self.rigged, "--image", "NoSuchMaterial=skin.png")
        self.assertEqual(code, 2)
        self.assertIn("NoSuchMaterial", err)
        self.assertIn("TorsoMat", err)

    def test_a_figure_far_taller_than_the_old_camera_distance(self):
        run = Path(tempfile.mkdtemp(dir=self.tmp.name))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["render", "--asset", str(self.rigged), "--out", str(run), "--size", "64",
                         "--samples-flat", "16", "--samples-lit", "2", "--height", "120"])
        rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, 0, json.dumps(rec["checks"]))
        cams = json.loads((run / "cameras.json").read_text())["cameras"]
        self.assertGreater(cams[0]["distance"], 2 * 120)
        self.assertTrue(np.isfinite(np.load(run / "depth" / "front.npy")).any())

    def test_overwrite_replaces_a_run_folder_and_its_stale_files(self):
        run = Path(tempfile.mkdtemp(dir=self.tmp.name))
        args = ["render", "--asset", str(self.rigged), "--out", str(run), "--size", "64",
                "--samples-flat", "16", "--samples-lit", "2"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 0)
            stale = run / "heldout" / "front-left-34.png"   # a view an older version wrote
            stale.write_text("left from an older run")
            self.assertEqual(main(args), 2)          # not empty: refused
            self.assertEqual(main(args + ["--overwrite"]), 0)
            self.assertFalse(stale.exists())
            self.assertTrue((run / "stage-2.json").exists())
            (run / "notes.txt").write_text("someone's notes")
            self.assertEqual(main(args + ["--overwrite"]), 2)   # a file the bench never writes: refused
        self.assertTrue((run / "notes.txt").exists())

    def test_a_yaw_that_contradicts_the_rig_is_refused(self):
        code, err = self._main(self.rigged, "--yaw", "180")
        self.assertEqual(code, 2)
        self.assertIn("disagrees with the rig", err)
        self.assertNotIn("Traceback", err)

    def test_a_pose_needs_legs_as_well_as_arms(self):
        armless_legs = Path(self.tmp.name) / "nolegs.glb"
        build_figure(armless_legs, True, "nolegs")
        code, err = self._main(armless_legs, "--pose", "a-pose")
        self.assertEqual(code, 2)
        self.assertIn("not found: leg L, leg R", err)
        run = Path(self.tmp.name) / "auto"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["render", "--asset", str(armless_legs), "--out", str(run), "--size", "64",
                         "--samples-flat", "16", "--samples-lit", "2"])
        rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(rec["details"]["pose"], "rest")
        self.assertEqual(rec["details"]["limbs"], [])

    def test_an_unrigged_figure_needs_a_yaw(self):
        code, err = self._main(self.unrigged)
        self.assertEqual(code, 2)
        self.assertIn("pass --yaw", err)
        self.assertNotIn("Traceback", err)


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
        cases = ((True, True, "none", []), (False, False, "none", ["--yaw", "0"]),
                 ("cuff", True, "cuff", ["--bind-unrigged"]), ("turned", True, "turned", []))
        for key, rigged, build, extra in cases:
            asset = root / f"figure-{key}.glb"
            build_figure(asset, rigged, build)
            run = root / f"run-{key}"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["render", "--asset", str(asset), "--out", str(run), "--size", str(cls.SIZE),
                             "--samples-flat", "16", "--samples-lit", "4", "--timeout", "900", *extra])
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
        self.assertEqual(cams["evaluation"],
                         ["front-left-25", "front-right-25", "front-left-65", "front-right-65"])
        for cam in cams["cameras"]:
            for rel in cam["files"].values():
                self.assertTrue((run / rel).exists(), rel)
        for v in ("front", "left", "back", "right"):
            self.assertTrue((run / "views" / f"{v}.png").exists())
            self.assertTrue((run / "masks" / f"{v}.png").exists())
        self.assertTrue((run / "gt" / "body.glb").exists())
        self.assertEqual(json.loads((run / "bench-run.json").read_text())["status"], "finished")
        self.assertEqual(rec["details"]["triangles"], 8 * 12)

    def test_rigged_figure_is_put_in_an_a_pose(self):
        rec, _ = self._record(True)
        self.assertEqual(rec["details"]["pose"], "a-pose")
        arms = {x["side"]: x for x in rec["details"]["limbs"] if x["limb"] == "arm"}
        self.assertEqual(set(arms), {"L", "R"})
        for x in arms.values():
            self.assertAlmostEqual(x["before_deg_below_horizontal"], 0.0, places=3)
            self.assertAlmostEqual(x["after_deg_below_horizontal"], 45.0, places=3)
            self.assertTrue(x["converged"])
            self.assertLessEqual(x["error_deg"], 0.01)
        self.assertEqual(len(rec["details"]["limbs"]), 4)

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

    def _side_of_red_and_blue(self, run, rel_colour, rel_mask):
        img = np.asarray(Image.open(run / rel_colour).convert("RGB")).astype(int)
        mask = np.asarray(Image.open(run / rel_mask)) > 127
        red = mask & (img[..., 0] > 200) & (img[..., 1] < 120) & (img[..., 2] < 120)
        blue = mask & (img[..., 2] > 200) & (img[..., 0] < 120) & (img[..., 1] < 120)
        return red, blue

    def test_a_figure_facing_plus_y_is_turned_round_from_its_rig(self):
        rec, run = self._record("turned")
        self.assertEqual(rec["details"]["yaw_deg"], 180.0)
        self.assertEqual(rec["details"]["facing"]["source"], "rig")
        cams = {c["view"]: c for c in json.loads((run / "cameras.json").read_text())["cameras"]}
        for rigged in (True, "turned"):  # built facing -Y, and built facing +Y then turned by the bench
            _, run = self._record(rigged)
            # the character's left (red) arm: image right from the front and the front three-quarter views,
            # image left from the back; nearest the camera from the left side, hidden behind from the right
            for view in ("front", cams_name(cams, 25), cams_name(cams, 335)):
                paths = cams[view]["files"]
                red, blue = self._side_of_red_and_blue(run, paths["colour"], paths["mask"])
                self.assertGreater(np.nonzero(red)[1].mean(), np.nonzero(blue)[1].mean(), (rigged, view))
            red, blue = self._side_of_red_and_blue(run, "views/back.png", "masks/back.png")
            self.assertLess(np.nonzero(red)[1].mean(), np.nonzero(blue)[1].mean(), (rigged, "back"))
            red, blue = self._side_of_red_and_blue(run, "views/left.png", "masks/left.png")
            self.assertGreater(red.sum(), blue.sum(), (rigged, "left"))
            red, blue = self._side_of_red_and_blue(run, "views/right.png", "masks/right.png")
            self.assertGreater(blue.sum(), red.sum(), (rigged, "right"))

    def test_stage_2h_marks_the_evaluation_views_as_ground_truth(self):
        _, run = self._record(True)
        rec = json.loads((run / "stage-2h.json").read_text())
        self.assertEqual(rec["stage"], "2h")
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["reference"], "ground-truth")
        self.assertIsInstance(rec["details"]["views"], dict)   # keyed by view name, as the scorer reads it
        views = {name: v["azimuth_deg"] for name, v in rec["details"]["views"].items()}
        for v in rec["details"]["views"].values():
            self.assertEqual(set(v["files"]), {"colour", "lit", "mask", "depth"})
        self.assertEqual(views, {"front-left-25": 25, "front-right-25": 335, "front-left-65": 65,
                                 "front-right-65": 295})
        for rel in rec["outputs"]:
            self.assertTrue((run / rel).exists(), rel)
            self.assertTrue(rel.startswith("heldout/"), rel)

    def test_heldout_views_are_not_building_views(self):
        _, run = self._record(True)
        meta = json.loads((run / "views" / "meta.json").read_text())
        self.assertFalse(set(meta["views"]) & set(meta["heldout_views"]))
        for v in meta["heldout_views"]:
            self.assertTrue((run / "heldout" / f"{v}-mask.png").exists())
            self.assertFalse((run / "views" / f"{v}.png").exists())


if __name__ == "__main__":
    unittest.main()
