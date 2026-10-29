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
# deform. With "turned", the whole figure faces +Y instead (a model exported the other way round). With
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
        for mode in ("plain", "hostile", "holes"):
            asset, run = Path(cls.tmp.name) / f"box-{mode}.blend", Path(cls.tmp.name) / f"run-{mode}"
            cmd = [bl.find_blender(), "--background", "--factory-startup", "--python-expr", BUILD_BLEND, "--",
                   mode, str(asset)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            assert asset.exists(), proc.stdout[-1500:] + proc.stderr[-1500:]
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["render", "--asset", str(asset), "--out", str(run), "--size", "96",
                             "--samples-flat", "4", "--samples-lit", "16", "--image", "Legacy=Skin",
                             "--yaw", "0"])
            cls.runs[mode] = (code, run)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, mode, expect=0):
        code, run = self.runs[mode]
        rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, expect, json.dumps(rec["checks"]))
        return rec, run

    def test_visible_transparent_texels_are_measured_and_fail_the_run(self):
        rec, _ = self._run("holes", expect=1)
        c = next(c for c in rec["checks"] if c["name"] == "visible_transparent_texels_px")
        self.assertFalse(c["pass"])
        self.assertGreater(c["per_view"]["front"], 100)
        self.assertEqual(c["image_alpha_transparent_share"], {"Skin": 0.5})
        failed = [x["name"] for x in rec["checks"] if not x["pass"]]
        self.assertEqual(failed, ["visible_transparent_texels_px"])
        plain, _ = self._run("plain")
        c = next(c for c in plain["checks"] if c["name"] == "visible_transparent_texels_px")
        self.assertEqual((c["value"], c["image_alpha_transparent_share"]), (0, {"Skin": 0.0}))

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
        cover = self.mask.copy()
        cover[10, 12:20] = False                            # a half-covered top row
        cover[9, 14] = True                                 # a pixel just outside, more than half covered
        self.assertEqual(self.off_edge(cover, self.mask), 0)

    def test_a_shifted_camera_is_caught_at_any_size(self):
        shifted = np.roll(self.mask, 3, axis=1)
        self.assertGreater(self.off_edge(shifted, self.mask), 0)
        big = np.kron(self.mask, np.ones((8, 8), dtype=bool))
        self.assertGreater(self.off_edge(np.roll(big, 3, axis=1), big), 0)


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
                 (["--samples-lit", "0"], "--samples-lit"))
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
            code = main(["render", "--asset", str(asset), "--out", out, "--size", "64", "--samples-flat", "2",
                         "--samples-lit", "2", *args])
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
                         "--samples-flat", "2", "--samples-lit", "2", "--height", "120"])
        rec = json.loads((run / "stage-2.json").read_text())
        self.assertEqual(code, 0, json.dumps(rec["checks"]))
        cams = json.loads((run / "cameras.json").read_text())["cameras"]
        self.assertGreater(cams[0]["distance"], 2 * 120)
        self.assertTrue(np.isfinite(np.load(run / "depth" / "front.npy")).any())

    def test_overwrite_replaces_a_run_folder_and_its_stale_files(self):
        run = Path(tempfile.mkdtemp(dir=self.tmp.name))
        args = ["render", "--asset", str(self.rigged), "--out", str(run), "--size", "64",
                "--samples-flat", "2", "--samples-lit", "2"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 0)
            (run / "stale.txt").write_text("left from an older run")
            self.assertEqual(main(args), 2)          # not empty: refused
            self.assertEqual(main(args + ["--overwrite"]), 0)
        self.assertFalse((run / "stale.txt").exists())
        self.assertTrue((run / "stage-2.json").exists())

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
                         "--samples-flat", "2", "--samples-lit", "2"])
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
        self.assertEqual(cams["evaluation"],
                         ["front-left-25", "front-right-25", "front-left-65", "front-right-65"])
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
        views = {v["view"]: v["azimuth_deg"] for v in rec["details"]["views"]}
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
