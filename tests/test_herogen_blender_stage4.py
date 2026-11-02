"""Tests for herogen stage 4 (herogen/blender): joints, R15 sections, the pose test, and stages 3-4 end to
end, on a synthetic A-pose humanoid.

The numpy parts run everywhere numpy and Pillow are installed; the end-to-end test needs Blender and skips
cleanly without it. Nothing generated here is written inside the repository.
"""
from __future__ import annotations

import json
import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_herogen_body_quality import quality_client

try:
    import numpy as np
    import PIL  # noqa: F401
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

if HAVE_DEPS:
    from herogen.blender import SECTIONS, bl, carve, cuts, joints, pose, split, synth, views


def _open_edges(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class FastJointRecordTests(unittest.TestCase):
    def finish_fast(self, rec, empty_part=False, triangles=1):
        from types import SimpleNamespace
        from herogen.blender.stage4 import _record_fast_outputs

        meshes = [SimpleNamespace(name=n, verts=[(0, 0, 0)], faces=[] if empty_part and n == "RightHand"
                                  else [(0, 0, 0)] * triangles) for n in SECTIONS]
        with mock.patch("herogen.blender.glb.load_glb", return_value=SimpleNamespace(meshes=meshes)):
            _record_fast_outputs(rec, ("high", "low"))
        return rec.finish()

    def make_record(self, tmp):
        from herogen.blender.record import StageRecord, check

        rec = StageRecord(Path(tmp), 4, "game body")
        (rec.run / "04-body").mkdir()
        rec.data["outputs"] = [f"04-body/{lod}.glb" for lod in ("high", "low")]
        for p in rec.data["outputs"]:
            (rec.run / p).write_bytes(b"glTF")
        rec.add(check("sections_non_empty", 15, 15))
        return rec

    def test_fast_measured_gap_failure_is_informational_with_visible_note(self):
        from herogen.blender.record import check
        from herogen.blender.stage4 import _gap_check

        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_record(tmp)
            rec.add(_gap_check("max_joint_gap_high", .417))
            rec.add(_gap_check("max_joint_gap_low", .387))
            for name in ("pivots_in_joint_region", "rest_iou_with_covers_front", "texture_colour_error_front",
                         "texture_ssim_front", "texture_transfer_views", "joint_error_max_synthetic"):
                rec.add(check(name, 0, passed=False))
            result = self.finish_fast(rec)
            self.assertTrue(result["ok"])
            checks = {c["name"]: c for c in result["checks"]}
            for name in ("max_joint_gap_high", "max_joint_gap_low"):
                self.assertFalse(checks[name]["gate"])
                self.assertFalse(checks[name]["pass"])
                self.assertEqual(checks[name]["threshold"], .15)
            self.assertEqual(checks["max_joint_gap_high"]["value"], .417)
            self.assertEqual(checks["max_joint_gap_low"]["value"], .387)
            self.assertTrue(all(not c["gate"] for c in result["checks"][1:9]))
            self.assertIn("Roblox files ready; some joints have visible seams (0.42 studs at the widest)",
                          result["notes"])
            self.assertIn("Some shape or texture details may need refinement.", result["notes"])
            self.assertEqual(json.loads((rec.run / "stage-4.json").read_text()), result)

    def test_unverified_seams_do_not_change_full_lane_measured_gap_gates(self):
        from herogen.blender.record import StageRecord
        from herogen.blender.stage4 import GAP_GATE, _gap_check, _record_unverified

        for gap, expected in ((GAP_GATE, True), (GAP_GATE + .000001, False)):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as tmp:
                rec = StageRecord(Path(tmp), 4, "game body")
                _record_unverified(rec, {"RightElbow": "no shared rim"})
                rec.add(_gap_check("max_joint_gap_high", gap))
                result = rec.finish()
                self.assertEqual(result["ok"], expected)
                self.assertEqual(result["unverified_joints"], [{
                    "joint": "RightElbow", "reason": "no shared rim",
                    "message": "The right elbow seam could not be checked on this character's joints."}])
                self.assertFalse(result["checks"][0]["gate"])
                self.assertTrue(result["checks"][1]["gate"])
                self.assertEqual(result["checks"][1]["threshold"], .15)

    def test_fast_seam_note_uses_the_measurement_result_before_rounding(self):
        from herogen.blender.stage4 import GAP_GATE, _gap_check

        for gap, visible in ((GAP_GATE, False), (GAP_GATE + .000001, True)):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as tmp:
                rec = self.make_record(tmp)
                rec.add(_gap_check("max_joint_gap_high", gap))
                result = self.finish_fast(rec)
                self.assertTrue(result["ok"])
                self.assertEqual("visible seams" in result["notes"], visible)

    def test_no_measured_gaps_is_unknown_rather_than_zero(self):
        from herogen.blender.stage4 import _gap_check, _max_gap

        self.assertIsNone(_max_gap([], fast=True))
        with self.assertRaises(ValueError) as original:
            max([])
        with self.assertRaises(ValueError) as full:
            _max_gap([], fast=False)
        self.assertEqual(str(full.exception).encode(), str(original.exception).encode())
        result = _gap_check("max_joint_gap_high", _max_gap([], fast=True))
        self.assertIsNone(result["value"])
        self.assertFalse(result["pass"])
        self.assertFalse(result["gate"])

    def test_unverified_seam_does_not_excuse_an_empty_final_part(self):
        from herogen.blender.record import check
        from herogen.blender.stage4 import _record_unverified

        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_record(tmp)
            _record_unverified(rec, {"RightWrist": "empty section RightHand"})
            rec.add(check("sections_non_empty", 14, 15))
            result = self.finish_fast(rec)
            self.assertFalse(result["ok"])
            self.assertNotIn("Roblox files ready", result["notes"])

    def test_missing_or_empty_output_stays_a_hard_failure(self):
        for empty in (True, False):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as tmp:
                rec = self.make_record(tmp)
                p = rec.run / "04-body/low.glb"
                if empty:
                    p.write_bytes(b"")
                else:
                    p.unlink()
                result = self.finish_fast(rec)
                self.assertFalse(result["ok"])
                c = next(c for c in result["checks"] if c["name"] == "output_files")
                self.assertTrue(c["gate"])
                self.assertEqual(c["missing_or_empty"], ["04-body/low.glb"])

    def test_empty_exported_r15_part_stays_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_record(tmp)
            result = self.finish_fast(rec, empty_part=True)
            self.assertFalse(result["ok"])
            for c in result["checks"]:
                if c["name"].startswith("exported_parts_"):
                    self.assertTrue(c["gate"])
                    self.assertFalse(c["pass"])
                    self.assertEqual(c["value"], 14)
                    self.assertEqual(c["missing_or_empty"], ["RightHand"])

    def test_fast_profile_budgets_and_watertight_parts_are_informational(self):
        from herogen.blender.record import check

        for name, value, threshold in (("triangles_high", 24001, 24000), ("triangles_low", 7128, 6000),
                                       ("watertight_parts", 1, 0)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                rec = self.make_record(tmp)
                rec.add(check(name, value, threshold, passed=False, note="original measurement"))
                self.assertTrue(self.finish_fast(rec)["ok"])
                c = next(c for c in rec.data["checks"] if c["name"] == name)
                self.assertFalse(c["gate"])
                self.assertFalse(c["pass"])
                self.assertEqual((c["value"], c["threshold"]), (value, threshold))
                self.assertIn("original measurement; fast lane:", c["note"])
                self.assertIn("informational", c["note"])
                self.assertIn("Some shape or texture details may need refinement.", rec.data["notes"])
                if name.startswith("triangles_"):
                    lod = name.removeprefix("triangles_")
                    self.assertIn(f"{lod}-detail body uses {value:,} triangles, above the {threshold:,} target",
                                  rec.data["notes"])

    def test_full_profile_budgets_and_watertight_parts_stay_hard_failures(self):
        from herogen.blender.record import check

        for name in ("triangles_high", "triangles_low", "watertight_parts"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                rec = self.make_record(tmp)
                measured = check(name, 1, 0, passed=False)
                rec.add(measured.copy())
                self.assertFalse(rec.finish()["ok"])
                self.assertEqual(rec.data["checks"][-1], measured)

    def test_other_topology_and_profile_limits_stay_hard_failures(self):
        from herogen.blender.record import check

        for name in ("open_edges", "triangles_total", "triangles_Head", "texture_max_px"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                rec = self.make_record(tmp)
                rec.add(check(name, 1, passed=False))
                self.assertFalse(self.finish_fast(rec)["ok"])
                c = next(c for c in rec.data["checks"] if c["name"] == name)
                self.assertTrue(c["gate"])

    def test_exported_mesh_over_roblox_limit_stays_a_hard_failure(self):
        from herogen.roblox import MESH_TRIANGLE_LIMIT

        for triangles, expected in ((MESH_TRIANGLE_LIMIT, True), (MESH_TRIANGLE_LIMIT + 1, False)):
            with self.subTest(triangles=triangles), tempfile.TemporaryDirectory() as tmp:
                rec = self.make_record(tmp)
                result = self.finish_fast(rec, triangles=triangles)
                self.assertEqual(result["ok"], expected)
                for c in result["checks"]:
                    if c["name"].startswith("exported_triangles_per_mesh_"):
                        self.assertTrue(c["gate"])
                        self.assertEqual(c["pass"], expected)

    def test_export_reader_failure_stays_loud(self):
        from herogen.blender.stage4 import _record_fast_outputs

        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_record(tmp)
            with mock.patch("herogen.blender.glb.load_glb", side_effect=ValueError("invalid GLB")):
                with self.assertRaisesRegex(ValueError, "invalid GLB"):
                    _record_fast_outputs(rec, ("high", "low"))
            self.assertFalse((rec.run / "stage-4.json").exists())


@unittest.skipUnless(HAVE_DEPS and bl.find_blender(), "Blender is required")
class TexturedReferenceTests(unittest.TestCase):
    def test_reference_image_uvs_and_contract_camera_frame(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            verts = np.array([[-.5, 0, 0], [.5, 0, 0], [-.5, 0, 2], [.5, 0, 2]])
            faces = np.array([[0, 1, 2], [2, 1, 3]])
            uv = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], float)
            np.savez(work / "ref.npz", verts=verts, faces=faces, corner_uv=uv[faces])
            image = np.empty((64, 64, 3), np.uint8)
            image[:32], image[32:] = [240, 20, 30], [30, 20, 240]
            Image.fromarray(image).save(work / "ref.png")
            cam = views.View("front", np.ones((64, 64), bool), 24., 40., 60.).camera()
            result = bl.run_job({"tasks": [{"task": "textured_mesh", "mesh": str(work / "ref.npz"),
                "texture": str(work / "ref.png"), "uv_origin": "top", "cameras": [cam],
                "out_dir": str(work)}]}, work, "reference")["results"][0]
            mask = views.load_mask(Path(result["silhouettes"]["front"]))
            np.testing.assert_array_equal(np.flatnonzero(mask.any(0)), np.arange(28, 52))
            np.testing.assert_array_equal(np.flatnonzero(mask.any(1)), np.arange(12, 60))
            with Image.open(result["textured"]["front"]) as im:
                rendered = np.asarray(im.convert("RGB"))
            np.testing.assert_allclose(rendered[20, 40], image[8, 32], atol=2)
            np.testing.assert_allclose(rendered[52, 40], image[55, 32], atol=2)


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class TextureMetricTests(unittest.TestCase):
    def test_view_limits_and_boundary_comparisons_are_unchanged(self):
        from herogen.blender import stage4
        self.assertEqual((stage4.TEX_ERROR_GATE, stage4.TEX_SSIM_GATE), (12.0, .70))
        metrics = {"front": {"mean_abs_error": 12., "ssim": .70},
                   "back": {"mean_abs_error": 12.01, "ssim": .6999}}
        checks = stage4._texture_source_checks(metrics, None, {"front", "back"})
        self.assertEqual([c["pass"] for c in checks], [True, True, False, False])
        self.assertTrue(all(c["gate"] and c["reference"] == "turnaround" for c in checks))

    def test_diagnostics_keep_numbers_but_never_gate(self):
        from herogen.blender import stage4
        checks = stage4._texture_checks({"front": {"mean_abs_error": 40., "ssim": .2}},
                                        12., .70, "turnaround", diagnostic=True)
        self.assertTrue(all(not c["gate"] for c in checks))
        self.assertEqual([c["value"] for c in checks], [40., .2])
        self.assertTrue(all(c["name"].startswith("texture_turnaround_") for c in checks))

    def test_compare_uses_inset_intersection_not_union(self):
        from herogen.blender import texture
        mask_a = np.zeros((32, 32), bool)
        mask_b = mask_a.copy()
        mask_a[2:25, 2:25] = True
        mask_b[8:30, 8:30] = True
        a, b = np.zeros((32, 32, 3)), np.zeros((32, 32, 3))
        b[mask_b & ~mask_a] = 255
        metrics = texture.compare(a, mask_a, b, mask_b)
        self.assertEqual(metrics["mean_abs_error"], 0)
        self.assertLess(metrics["pixels"], int((mask_a & mask_b).sum()))

    def test_refined_selects_transfer_gates_and_records_ungated_turnaround(self):
        from herogen.blender import stage4
        turnaround = {"front": {"mean_abs_error": 40., "ssim": .2}}
        transfer = {"front": {"mean_abs_error": 2., "ssim": .9}}
        checks = stage4._texture_source_checks(turnaround, transfer, {"front"})
        self.assertTrue(all(c["pass"] for c in checks if c["gate"]))
        active = next(c for c in checks if c["name"] == "texture_colour_error_front")
        self.assertEqual(active["value"], 2.)
        self.assertEqual(active["reference"], "refined")
        self.assertEqual(active["threshold"], stage4.TRANSFER_ERROR_GATE)
        diagnostic = next(c for c in checks if c["name"] == "texture_turnaround_colour_error_front")
        self.assertEqual(diagnostic["value"], 40.)
        self.assertFalse(diagnostic["gate"])

    def test_incomplete_refined_renders_fail_without_switching_reference(self):
        from herogen.blender import stage4
        good = {"front": {"mean_abs_error": 0., "ssim": 1.}}
        for transfer in ({}, good):
            checks = stage4._texture_source_checks(good, transfer, {"front", "back"})
            completeness = next(c for c in checks if c["name"] == "texture_transfer_views")
            self.assertFalse(completeness["pass"])
            self.assertTrue(completeness["gate"])
            self.assertTrue(all(c["reference"] == "refined" for c in checks if c["gate"]))

    def test_synthetic_view_bake_and_shifted_paint_are_rejected(self):
        from herogen.blender import stage4, texture
        row, col = np.indices((64, 64))
        stripes = ((row // 8 + col // 8) % 2) * 220. + 15.
        reference = np.repeat(stripes[..., None], 3, axis=2)
        mask = np.ones((64, 64), bool)
        def accepts(image):
            metrics = {"front": texture.compare(image, mask, reference, mask)}
            checks = stage4._texture_source_checks({}, metrics, {"front"})
            return all(c["pass"] for c in checks if c["gate"])
        self.assertTrue(accepts(reference))
        self.assertFalse(accepts(np.full_like(reference, 125.)))
        self.assertFalse(accepts(np.roll(reference, 8, axis=1)))


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow are required")
class SyntheticBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="herogen-blender-test-"))
        cls.rundir = cls.tmp / "run"
        cls.truth = synth.write_run(cls.rundir, synth.humanoid(), size=384, ppu=60.0)
        cls.vs = views.load_views(cls.rundir, 5.5)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestJoints(SyntheticBase):
    def test_joint_error_on_variants(self):
        for kw in ({}, {"arm_angle_deg": 40}, {"arm_angle_deg": 60, "head_scale": 1.2}, {"elbow_bend_deg": 25}):
            run = self.tmp / ("j" + "".join(f"{k}{v}" for k, v in kw.items()))
            truth = synth.write_run(run, synth.humanoid(**kw), size=384, ppu=60.0)
            sk = joints.estimate(views.load_views(run, 5.5), 5.5)
            err = joints.joint_errors(sk.as_json(), truth["joints"])
            self.assertEqual(len(err), 16)
            self.assertLess(max(err.values()), 0.10, (kw, err))
            self.assertLess(float(np.mean(list(err.values()))), 0.05, kw)


    def test_clothed_figure_regression(self):
        # Hair, a scarf spreading from chin to shoulders, a backpack and a hanging scarf tail: the outline has no
        # neck narrowing and widens row by row. The committed estimator before this case put the neck at 0.69 H
        # and the shoulders on the elbows (max error 0.85 studs); the neck is now bounded and evidenced.
        run = self.tmp / "clothed"
        truth = synth.write_run(run, synth.humanoid(clothed=True), size=384, ppu=60.0)
        sk = joints.estimate(views.load_views(run, 5.5), 5.5)
        err = joints.joint_errors(sk.as_json(), truth["joints"])
        # the anatomical neck is hidden under the scarf; the estimate sits at the scarf's top
        self.assertLess(err["Neck"], 0.20)
        self.assertLess(max(v for k, v in err.items() if k != "Neck"), 0.10, err)
        self.assertGreater(sk.joints["Neck"][2], 0.78 * 5.5)
        for k in ("Neck", "LeftShoulder", "Waist", "ShoulderLine", "LeftHip", "LeftAnkle"):
            self.assertIn(k, sk.evidence)

    def test_joints_without_back_and_top_views(self):
        run = self.tmp / "missing"
        shutil.copytree(self.rundir, run)
        (run / "masks" / "top.png").unlink()
        (run / "masks" / "back.png").unlink()
        sk = joints.estimate(views.load_views(run, 5.5), 5.5)
        err = joints.joint_errors(sk.as_json(), self.truth["joints"])
        self.assertLess(max(err.values()), 0.12)


class TestSections(SyntheticBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.verts, cls.faces, _ = carve.build_mesh(cls.vs, 5.5, 96)
        cls.sk = joints.estimate(cls.vs, 5.5)

    def test_a_dome_over_a_three_vertex_opening_stays_manifold(self):
        """An opening of three unevenly spaced rim vertices (a small separate cut loop): sampling the dome's ring
        evenly along it would keep only two points; the cap must still close the part cleanly."""
        from herogen.blender.score import mesh_sanity

        v = np.array([[0, 0, 0], [0.05, 0, 0], [0.1, 0.02, 0], [0.05, 0.01, -0.1]], float)
        f = np.array([[0, 2, 3], [2, 1, 3], [1, 0, 3]])          # a tetrahedron missing its top face
        loop = [0, 1, 2]
        cp = {"centre": v[:3].mean(axis=0), "dir": np.array([0.0, 0.0, 1.0]), "depth": 0.3, "rings": 2,
              "ring_points": 12}
        cv, cf = split.close_section(v, f, [loop], [cp])
        self.assertTrue(mesh_sanity(cv, cf, 1e-9)["watertight"])

    def test_a_cut_through_a_vertex_leaves_every_part_watertight(self):
        """A cut plane a hair from a vertex: the cut vertex must not land within the weld tolerance of it, or the
        sliver between them collapses on welding and opens the part."""
        from herogen.blender.score import mesh_sanity

        p, n = split.build_rig(self.sk).planes["Waist"]
        d = (self.verts - p) @ n
        for i in np.argsort(np.abs(d))[:8]:
            for eps in (2e-6, -2e-6):
                secs, _ = split.build_sections(self.verts, self.faces, self.sk, cap="dome", rings=1, ring_points=8,
                                               cut_offsets={"Waist": float(d[i]) - eps})
                for name in ("UpperTorso", "LowerTorso"):
                    sanity = mesh_sanity(secs[name].verts, secs[name].faces, 1e-6 * 5.5)
                    self.assertTrue(sanity["watertight"], (int(i), eps, name, sanity))

    def test_a_cover_patch_that_pinches_still_closes_into_a_manifold_shell(self):
        from herogen.blender import covers
        from herogen.blender.score import mesh_sanity

        # two squares of a surface that meet at one corner only, plus a third square on the bigger side
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0],
                      [3, 1, 0], [3, 2, 0]], float)
        f = np.array([[0, 1, 2], [0, 2, 3], [2, 4, 5], [2, 5, 6], [4, 7, 8], [4, 8, 5]])
        patch = covers.manifold_patch(f)
        self.assertEqual(sorted(map(tuple, patch.tolist())), sorted(map(tuple, f[2:].tolist())))
        sv, sf = covers.shell(v, patch, 0.05, 0.02)
        self.assertTrue(mesh_sanity(sv, sf, 1e-9)["watertight"])
        sv, sf = covers.shell(v, f, 0.05, 0.02)              # the pinched patch itself does not close cleanly
        self.assertGreater(mesh_sanity(sv, sf, 1e-9)["nonmanifold_edges"], 0)
        passes, largest = covers.MAX_PINCH_PASSES, covers._largest
        covers.MAX_PINCH_PASSES, covers._largest = 0, lambda p: p    # keep the pinch, allow no pass: an error
        try:
            with self.assertRaises(ValueError):
                covers.manifold_patch(f)
        finally:
            covers.MAX_PINCH_PASSES, covers._largest = passes, largest

    def test_a_cover_never_welds_into_its_owner(self):
        """Covers shaped from the very mesh that was cut, held at zero outset by the masks: every outer vertex
        would sit on a vertex of the owner. The owner must still be closed and manifold after welding."""
        from herogen.blender import covers
        from herogen.blender.score import mesh_sanity

        secs, _ = split.build_sections(self.verts, self.faces, self.sk, cap="dome", rings=1, ring_points=8)
        covered, info = covers.add_covers(secs, self.verts, self.faces, owner="UpperTorso", radius=2.5,
                                          outset=0.06, inside=lambda pts: np.zeros(len(pts), dtype=bool))
        self.assertEqual(len(info), 2)
        owner = covered["UpperTorso"]
        sanity = mesh_sanity(owner.verts, owner.faces, 1e-6 * 5.5)
        self.assertTrue(sanity["watertight"], sanity)
        self.assertEqual(sanity["nonmanifold_edges"], 0)

    def test_fifteen_closed_sections_with_pivots_inside(self):
        secs, _ = split.build_sections(self.verts, self.faces, self.sk)
        self.assertEqual(list(secs), list(SECTIONS))
        for name, s in secs.items():
            self.assertGreater(len(s.faces), 50, name)
            self.assertEqual(_open_edges(s.faces), 0, name)
            self.assertTrue(carve.winding_inside(s.pivot[None], s.verts, s.faces)[0], name)

    def test_domes_close_joints_better_than_flat_caps(self):
        poses = pose.pose_library(pose.bone_directions(self.sk))
        pick = {k: poses[k] for k in ("rest", "crouch", "arms_up")}
        dome, _ = split.build_sections(self.verts, self.faces, self.sk, cap="dome", rings=1, ring_points=8)
        flat, _ = split.build_sections(self.verts, self.faces, self.sk, cap="flat")
        g_dome = pose.joint_gaps(dome, pick)
        g_flat = pose.joint_gaps(flat, pick)
        self.assertLess(max(g_dome["rest"].values()), 1e-3)
        for p in ("crouch", "arms_up"):
            self.assertLess(max(g_dome[p].values()), 0.15, p)
            self.assertLess(max(g_dome[p].values()), max(g_flat[p].values()), p)

    def test_shoulder_covers_are_closed_and_counted(self):
        from herogen.blender import covers

        secs, _ = split.build_sections(self.verts, self.faces, self.sk)
        covered, info = covers.add_covers(secs, self.verts, self.faces, owner="UpperTorso")
        self.assertEqual(sorted(info), ["LeftShoulderCover", "RightShoulderCover"])
        added = len(covered["UpperTorso"].faces) - len(secs["UpperTorso"].faces)
        self.assertEqual(added, sum(i["triangles"] for i in info.values()))
        self.assertEqual(_open_edges(covered["UpperTorso"].faces), 0)
        for name, section in secs.items():
            self.assertEqual(int((~section.colour_exterior).sum()), section.cap_triangles, name)
        original = secs["UpperTorso"]
        section = covered["UpperTorso"]
        np.testing.assert_array_equal(section.colour_exterior[:len(original.faces)], original.colour_exterior)
        # Each shell's reversed inner patch and connecting walls are excluded.
        self.assertGreater(int(section.colour_exterior[len(original.faces):].sum()), 0)
        self.assertLess(int(section.colour_exterior[len(original.faces):].sum()), added / 2)

    def test_visible_gap_never_exceeds_geometric_gap(self):
        secs, _ = split.build_sections(self.verts, self.faces, self.sk, rings=1, ring_points=8)
        poses = pose.pose_library(pose.bone_directions(self.sk))
        pick = {k: poses[k] for k in ("arms_up", "crouch")}
        geo, vis = pose.joint_gaps(secs, pick), pose.joint_gaps(secs, pick, visible=True)
        for p in pick:
            for j in geo[p]:
                self.assertLessEqual(vis[p][j], geo[p][j] + 1e-9)

    def test_ray_hits(self):
        tri = np.array([[[0, 0, 1.0], [1, 0, 1.0], [0, 1, 1.0]]])
        o = np.array([[0.2, 0.2, 0.0], [0.9, 0.9, 0.0], [0.2, 0.2, 2.0]])
        d = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
        self.assertEqual(carve.ray_hits(o, d, tri).tolist(), [True, False, False])

    def test_world_matrices_follow_the_hierarchy(self):
        pivots = {n: np.array([0.0, 0.0, float(i)]) for i, n in enumerate(SECTIONS)}
        r = pose.rot([1, 0, 0], 90)
        m = pose.world_matrices({"LeftUpperLeg": r}, pivots)
        child = m["LeftLowerLeg"]
        # the child moves with its parent: its pivot is carried by the parent's rotation
        p = pivots["LeftLowerLeg"]
        expected = r @ (p - pivots["LeftUpperLeg"]) + pivots["LeftUpperLeg"]
        self.assertTrue(np.allclose(child[:3, :3] @ p + child[:3, 3], expected))
        self.assertTrue(np.allclose(m["Head"], np.eye(4)))


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class TestEndToEndBlender(SyntheticBase):
    def test_fast_rimless_mesh_exports_fifteen_parts_with_informational_gap_failure(self):
        from herogen.blender import glb, stage4

        run = self.tmp / "rimless"
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.)
        vs = views.load_views(run, 5.5)
        sk = joints.estimate(vs, 5.5)
        verts, faces, _ = carve.build_mesh(vs, 5.5, 96)
        sections, _ = split.build_sections(verts, faces, sk, rings=1, ring_points=8)
        verts, faces, count = [], [], 0
        for section in sections.values():
            centre = section.verts.mean(axis=0)
            verts.append(centre + .6 * (section.verts - centre))
            faces.append(section.faces + count)
            count += len(section.verts)
        work = run / "blender-work"
        work.mkdir()
        np.savez(work / "03-mesh.npz", verts=np.concatenate(verts), faces=np.concatenate(faces))
        (run / "stage-3.json").write_text(json.dumps({"ok": True, "mesh_source": "carve",
                                                     "shape_views": list(vs.views)}))
        with mock.patch.dict(os.environ, {"HEROGEN_FAST_LANE": "0"}):
            with self.assertRaises(cuts.UnmeasurableJoints) as exc:
                stage4.run_stage4(run, cut_search=False)
        full_message = str(exc.exception).encode()
        self.assertIn(b"RightElbow: no shared rim", full_message)
        measure = pose.joint_gaps

        def violated(*args, **kwargs):
            result = measure(*args, **kwargs)
            for row in result.values():
                # Waist retains a shared rim in this mesh. Its gap must still be measured.
                self.assertIn("Waist", row)
                row["Waist"] = stage4.GAP_GATE + .01
            return result

        with mock.patch.dict(os.environ, {"HEROGEN_FAST_LANE": "1"}), \
                mock.patch.object(pose, "joint_gaps", side_effect=violated):
            result = stage4.run_stage4(run, cut_search=False)
        self.assertTrue(result["ok"])
        recorded = json.loads((run / "stage-4.json").read_text())
        unknown = {row["joint"]: row for row in recorded["unverified_joints"]}
        self.assertIn("RightElbow", unknown)
        self.assertIn("no shared rim", unknown["RightElbow"]["reason"])
        self.assertIn("right elbow seam", unknown["RightElbow"]["message"])
        checks = {c["name"]: c for c in recorded["checks"]}
        self.assertFalse(checks["measurable_joints"]["gate"])
        self.assertFalse(checks["max_joint_gap_high"]["gate"])
        self.assertFalse(checks["max_joint_gap_high"]["pass"])
        self.assertIn("Roblox files ready; some joints have visible seams (0.16 studs at the widest)",
                      recorded["notes"])
        self.assertEqual(checks["sections_non_empty"]["value"], 15)
        body = json.loads((run / "04-body" / "sections.json").read_text())
        self.assertEqual(body["checks"], recorded["checks"])
        self.assertTrue(all(s["triangles_high"] > 0 for s in body["sections"]))
        meshes = glb.load_glb(run / "04-body" / "high.glb").meshes
        self.assertEqual({m.name for m in meshes}, set(SECTIONS))
        self.assertTrue(all(len(m.faces) for m in meshes))
        self.assertTrue(all("RightElbow" not in row for row in body["pose_test"]["gaps_high"].values()))

    @unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "scipy is required for refined colour")
    def test_refined_colours_reach_exported_head_and_torso_in_both_lods(self):
        import hashlib
        import io
        from herogen.blender import glb, sam3d_mesh, texture
        from herogen.blender.stage4 import run_stage4
        from PIL import Image

        run = self.tmp / "refined"
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.0)
        verts, faces, _ = carve.build_mesh(views.load_views(run, 5.5), 5.5, 96)
        work = run / "blender-work"
        work.mkdir()
        np.savez(work / "03-mesh.npz", verts=verts, faces=faces)
        glb.write_glb(run / "03-mesh.glb", [("body", verts, faces)])
        (run / "stage-3.json").write_text(json.dumps({"ok": True, "mesh_source": "sam3d-objects",
                                                       "shape_views": ["front"]}))
        R = sam3d_mesh.rotation(71, 13, -8)
        centre = np.array([.1, -.2, 2.7])
        # Invert a perspective correction plus a smooth warp, keeping the NPZ in the carved body's frame.
        unwarped = verts.copy()
        unwarped[:, 0] -= .06 * np.sin(verts[:, 2])
        k = 11 / (11 + unwarped[:, 1])
        unwarped[:, 0] /= k
        unwarped[:, 2] = 2.75 + (unwarped[:, 2] - 2.75) / k
        raw_verts = (unwarped - centre) @ R.T / 5.5
        stream = io.BytesIO()
        Image.new("RGB", (16, 16), "green").save(stream, format="PNG")
        glb.write_glb(run / "03-source.glb", [("raw", raw_verts, faces, np.full((len(verts), 2), .5))],
                      texture_png=stream.getvalue())
        raw = sam3d_mesh.load_source(run / "03-source.glb")
        target = sam3d_mesh.perspective_to_ortho(raw.verts @ R * 5.5 + centre, 2, .5, 5.5)
        target[:, 0] += .06 * np.sin(target[:, 2])
        design = np.c_[raw.verts, np.ones(len(raw.verts))]
        affine, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        self.assertGreater(np.linalg.norm(design @ affine - target, axis=1).max(), 100 * 5.5e-6)
        np.savez(work / "03-source-uv.npz", verts=target,
                 faces=raw.faces, corner_uv=raw.corner_uv)
        Image.new("RGB", (16, 16), "green").save(work / "03-source-texture.png")
        # A real 4096 image; every triangle has its own corners and its own region's constant UV.
        image = Image.new("RGB", (4096, 4096), (220, 30, 30))
        image.paste((30, 30, 220), (2048, 0, 4096, 4096))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        uv = np.where((verts[faces].mean(axis=1)[:, 2] > 4.4)[:, None, None], [.25, .5], [.75, .5])
        uv = np.broadcast_to(uv, (len(faces), 3, 2)).reshape(-1, 2).copy()
        glb.write_glb(run / "03-refined.glb", [("refined", raw_verts[faces[::-1]].reshape(-1, 3),
                                                np.arange(len(faces) * 3).reshape(-1, 3),
                                                uv.reshape(-1, 3, 2)[::-1].reshape(-1, 2))],
                      texture_png=stream.getvalue())
        source_sha = hashlib.sha256((run / "03-source.glb").read_bytes()).hexdigest()
        (run / "stage-3f.json").write_text(json.dumps({"status": "complete", "source_sha256": source_sha}))
        detail = np.zeros((64, 64, 3))
        detail[..., 1] = (np.indices((64, 64)).sum(axis=0) % 2) * 255
        closeup = texture.Closeup("front", detail, np.array([[.25, 0, 184], [0, .25, 40]]),
                                  np.ones((384, 384)))
        with mock.patch.object(texture, "load_closeups", return_value=(
                {"front": closeup}, {"used": ["front"], "notes": []})):
            r4 = run_stage4(run, timeout=600, profile="roblox-ingame")
        self.assertTrue(r4["ok"], r4)
        recorded = json.loads((run / "stage-4.json").read_text())
        info = recorded["body"]["texture"]
        self.assertEqual(info["colour_source"], "refined")
        self.assertFalse(info["source_fallback"])
        self.assertEqual(info["colour_views"], [])
        self.assertEqual(info["refined"]["stage3f_source_sha256"], source_sha)
        self.assertEqual(info["refined"]["refined_glb_sha256"],
                         hashlib.sha256((run / "03-refined.glb").read_bytes()).hexdigest())
        check = info["refined"]["source_correspondence"]
        self.assertNotIn("frame_fit", info["refined"])
        self.assertEqual(check["method"], "piecewise-linear source correspondence")
        self.assertLess(check["geometry_distance_raw"]["max"], check["threshold_raw"])
        self.assertLess(check["placement_distance_studs"]["max"], 1e-9)
        self.assertIn("Head texture takes the refined texture", recorded["notes"])
        self.assertIn("close-ups were not applied", recorded["notes"])
        self.assertEqual(info["closeups"]["used"], [])
        self.assertEqual(info["closeups"]["comparison_views_used"], ["front"])
        self.assertEqual(info["gates"]["reference"], "refined")
        self.assertEqual(info["head_front_island_scale"], 2.)
        self.assertIn("intersection", info["comparison_mask"])
        checks = {c["name"]: c for c in r4["checks"]}
        self.assertTrue(checks["texture_transfer_views"]["pass"])
        for view in info["transfer_metrics"]:
            for metric in ("colour_error", "ssim"):
                active = checks[f"texture_{metric}_{view}"]
                self.assertTrue(active["gate"])
                self.assertTrue(active["pass"], active)
                self.assertEqual(active["reference"], "refined")
                self.assertFalse(checks[f"texture_turnaround_{metric}_{view}"]["gate"])
            self.assertEqual(checks[f"texture_turnaround_colour_error_{view}"]["value"],
                             info["rest_metrics"][view]["mean_abs_error"])
        self.assertNotIn("closeups", info["bake"]["high"]["head"])
        self.assertEqual(info["bake"]["high_views"]["head"]["closeups"]["views"], ["front"])
        check = next(c for c in r4["checks"] if c["name"] == "refined_texture")
        self.assertTrue(check["pass"])
        for lod in ("high", "low"):
            body = glb.load_glb(run / "04-body" / f"{lod}.glb")
            self.assertEqual(len(body.images), 2)
            for section, expected in (("Head", [220, 30, 30]), ("UpperTorso", [30, 30, 220])):
                mesh = next(m for m in body.meshes if m.name == section)
                tri = mesh.verts[mesh.faces].mean(axis=1)
                # Surface points away from the neck boundary and from synthetic caps.
                keep = (tri[:, 2] > 4.7) if section == "Head" else ((tri[:, 2] > 3.1) & (tri[:, 2] < 3.8))
                self.assertTrue(keep.any(), (lod, section))
                face = np.flatnonzero(keep)
                uv = mesh.uv[mesh.faces[face]].mean(axis=1)
                material = body.materials[int(mesh.face_material[face[0]])]
                im = body.image(material.image)[..., :3]
                colours = texture._bilinear(im, uv[:, 1] * len(im) - .5, uv[:, 0] * im.shape[1] - .5)
                np.testing.assert_allclose(np.median(colours, axis=0), expected, atol=2,
                                           err_msg=f"{lod} {section}")
                stats = info["bake"][lod]["head"] if section == "Head" else info["bake"][lod]
                self.assertEqual(stats["texels_primary"], 1)
                self.assertFalse(stats["closeups_applied"])
                self.assertEqual(stats["sampling"], "reliable closest triangles with smooth-normal fold rays")
                projection = stats["primary_projection"]
                self.assertGreater(projection["max_distance"], 0)
                self.assertEqual(projection["distance_factor"], 4)
                self.assertEqual(projection["fallback_fraction"], projection["fallback_count"] / projection["samples"])
            stats = info["bake"][lod]
            overall = stats["primary_projection_overall"]
            self.assertEqual(overall["samples"], stats["primary_projection"]["samples"]
                             + stats["head"]["primary_projection"]["samples"])
            self.assertEqual(overall["fallback_count"], stats["primary_projection"]["fallback_count"]
                             + stats["head"]["primary_projection"]["fallback_count"])
            self.assertEqual(overall["fallback_fraction"], overall["fallback_count"] / overall["samples"])
            self.assertEqual(stats["head"]["primary_projection"]["normal_fields"], {"Head": "area-weighted vertex normals"})
        for name in ("atlas-high-views.png", "atlas-head-high-views.png"):
            self.assertTrue((run / "04-body" / name).is_file())
            self.assertFalse(any(name in path for path in r4["outputs"]))
        self.assertFalse((run / "04-body" / "atlas-high-no-source-fallback.png").exists())
        self.assertFalse((run / "04-body" / "atlas-high-side-trust.png").exists())
        self.assertEqual(json.loads((run / "04-body" / "sections.json").read_text())["texture"], info)

        from herogen.roblox.export import run_export
        r6 = run_export(run / "04-body", run / "06-roblox")
        self.assertTrue(r6["ok"], r6)
        self.assertFalse(any("views.png" in path for path in r6["outputs"]))
        self.assertNotIn("views.png", (run / "06-roblox" / "HeroGen-R15.rbxmx").read_text())

        # A validated refinement remains usable without either view colour or the raw source fallback.
        (work / "03-source-texture.png").unlink()
        with mock.patch.object(texture, "observed_colour_views", return_value={}):
            no_views = run_stage4(run, timeout=600, profile="roblox-ingame")
        self.assertTrue(no_views["ok"], no_views)
        self.assertEqual(no_views["body"]["texture"]["colour_views"], [])
        self.assertNotIn("high_views", no_views["body"]["texture"]["bake"])
        self.assertFalse((run / "04-body" / "atlas-high-views.png").exists())
        self.assertFalse((run / "04-body" / "atlas-head-high-views.png").exists())

    def test_stages_3_and_4(self):
        from herogen.blender.stage3 import run_stage3
        from herogen.blender.stage4 import run_stage4

        r3 = run_stage3(self.rundir, nz=128, timeout=300, mesh_source="carve", joints_source="silhouette", quality_client=quality_client())
        self.assertTrue(r3["ok"], r3["checks"])
        r4 = run_stage4(self.rundir, timeout=600, profile="roblox-ingame")
        self.assertTrue(r4["ok"], r4["checks"])
        self.assertEqual(r4["profile"]["name"], "roblox-ingame")
        body = json.loads((self.rundir / "04-body" / "sections.json").read_text())
        self.assertEqual([s["name"] for s in body["sections"]], list(SECTIONS))
        self.assertEqual(body["form"], "rigid")
        self.assertLessEqual(sum(s["triangles_high"] for s in body["sections"]), 24000)
        self.assertLessEqual(sum(s["triangles_low"] for s in body["sections"]), 6000)
        self.assertEqual(body["profile"], "roblox-ingame")
        self.assertEqual(body["budgets"], {"triangles_high_max": 24000, "triangles_low_max": 6000})
        for lod in ("high", "low"):
            self.assertEqual((self.rundir / "04-body" / f"{lod}.glb").read_bytes()[:4], b"glTF")
        self.assertGreaterEqual(len(list((self.rundir / "04-body" / "pose-renders").glob("*.png"))), 7)
        stage4 = json.loads((self.rundir / "stage-4.json").read_text())
        self.assertEqual(stage4["stage"], 4)
        self.assertEqual(stage4["cost_usd"], 0.0)
        checks = {c["name"]: c for c in stage4["checks"]}
        for v in ("front", "back", "left", "right"):
            self.assertTrue(checks[f"texture_colour_error_{v}"]["pass"], checks[f"texture_colour_error_{v}"])
            self.assertTrue(checks[f"texture_colour_error_{v}"]["gate"])
            self.assertEqual(checks[f"texture_colour_error_{v}"]["reference"], "turnaround")
            self.assertEqual(checks[f"texture_colour_error_{v}"]["threshold"], 12.)
            self.assertEqual(checks[f"texture_ssim_{v}"]["threshold"], .70)
            self.assertNotIn(f"texture_turnaround_colour_error_{v}", checks)
            self.assertTrue(checks[f"texture_ssim_{v}"]["pass"], checks[f"texture_ssim_{v}"])
            self.assertTrue(checks[f"rest_iou_with_covers_{v}"]["pass"])
            self.assertEqual(checks[f"rest_iou_with_covers_{v}"]["reference"], "turnaround")
            self.assertNotIn(f"rest_iou_with_covers_turnaround_{v}", checks)
        self.assertEqual(body["texture"]["atlas_sizes"], {"high": 1024, "low": 512})
        self.assertEqual(body["texture"]["head_texture_sizes"], {"high": 1024, "low": 512})
        self.assertEqual(body["texture"]["gates"]["reference"], "turnaround")
        self.assertEqual(body["texture"]["head_front_island_scale"], 1.)
        self.assertEqual(body["texture"]["transfer_metrics"], {})
        for lod in ("high", "low"):
            self.assertTrue((self.rundir / "04-body" / f"atlas-head-{lod}.png").exists(), lod)
        self.assertEqual(len(body["covers"]["pieces_high"]), 2)
        # the cuts were placed once, on the reference body, and the record says where and why
        cuts_rec = body["cuts"]
        self.assertEqual(set(cuts_rec["chosen_fraction"]), {k for ks in cuts.GROUPS.values() for k in ks})
        self.assertEqual(set(cuts_rec["offsets_studs"]), set(cuts.JOINTS))
        self.assertEqual(stage4["body"]["cuts"]["offsets_studs"], cuts_rec["offsets_studs"])
        self.assertIn("cuts moved along their normals", stage4["notes"])
        ref = cuts_rec["reference_worst_gap"]
        self.assertEqual(set(ref), {"unmoved", "moved"})
        self.assertEqual(set(ref["moved"]), {"gate", "heldout"})
        for kind, f in cuts_rec["chosen_fraction"].items():
            if f:       # only an eligible joint moves, and only for a gain beyond the noise
                self.assertIn(kind, cuts_rec["eligible"])
        self.assertTrue((self.rundir / "04-body" / "textured-vs-views.png").exists())
        for lod in ("high", "low"):
            raw = (self.rundir / "04-body" / f"{lod}.glb").read_bytes()
            n = int.from_bytes(raw[12:16], "little")
            gltf = json.loads(raw[20:20 + n])
            self.assertEqual(len(gltf.get("images", [])), 2, lod)        # the body atlas and the Head's own
            self.assertEqual(sorted(m["name"] for m in gltf["materials"]), ["BodyAtlas", "HeadTexture"], lod)
            self.assertEqual(len(gltf["meshes"]), 15, lod)


    def test_cli_runs_stage_4_alone_from_the_glb(self):
        import subprocess
        import sys

        from herogen.blender.stage3 import run_stage3

        run = self.tmp / "cli"
        synth.write_run(run, synth.humanoid(arm_angle_deg=45), size=384, ppu=60.0)
        self.assertTrue(run_stage3(run, nz=128, timeout=300, mesh_source="carve", joints_source="silhouette", quality_client=quality_client())["ok"])
        (run / "blender-work" / "03-mesh.npz").unlink()
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run([sys.executable, "-m", "herogen.blender", "run", "--run", str(run), "--stages", "4",
                               "--profile", "roblox-ingame"], cwd=root, capture_output=True, text=True, timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(json.loads((run / "stage-4.json").read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
