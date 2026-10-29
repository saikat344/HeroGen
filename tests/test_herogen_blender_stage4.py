"""Tests for herogen stage 4 (herogen/blender): joints, R15 sections, the pose test, and stages 3-4 end to
end, on a synthetic A-pose humanoid.

The numpy parts run everywhere numpy and Pillow are installed; the end-to-end test needs Blender and skips
cleanly without it. Nothing generated here is written inside the repository.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

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
    def test_stages_3_and_4(self):
        from herogen.blender.stage3 import run_stage3
        from herogen.blender.stage4 import run_stage4

        r3 = run_stage3(self.rundir, nz=128, timeout=300)
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
            self.assertTrue(checks[f"texture_ssim_{v}"]["pass"], checks[f"texture_ssim_{v}"])
            self.assertTrue(checks[f"rest_iou_with_covers_{v}"]["pass"])
        self.assertEqual(body["texture"]["atlas_sizes"], {"high": 1024, "low": 512})
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
            self.assertEqual(len(gltf.get("images", [])), 1, lod)
            self.assertEqual(len(gltf["meshes"]), 15, lod)


    def test_cli_runs_stage_4_alone_from_the_glb(self):
        import subprocess
        import sys

        from herogen.blender.stage3 import run_stage3

        run = self.tmp / "cli"
        synth.write_run(run, synth.humanoid(arm_angle_deg=45), size=384, ppu=60.0)
        self.assertTrue(run_stage3(run, nz=128, timeout=300)["ok"])
        (run / "blender-work" / "03-mesh.npz").unlink()
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run([sys.executable, "-m", "herogen.blender", "run", "--run", str(run), "--stages", "4",
                               "--profile", "roblox-ingame"], cwd=root, capture_output=True, text=True, timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(json.loads((run / "stage-4.json").read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
