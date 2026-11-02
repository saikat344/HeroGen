"""Tests for herogen stage 3 (herogen/blender): masks, cameras and the carved base mesh, on a synthetic
A-pose humanoid.

The numpy parts run everywhere numpy and Pillow are installed; the end-to-end test needs Blender and skips
cleanly without it. Nothing generated here is written inside the repository.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from test_herogen_body_quality import quality_client

try:
    import numpy as np
    import PIL  # noqa: F401
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

if HAVE_DEPS:
    from herogen.blender import bl, carve, synth, views


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


class TestViewsAndCarve(SyntheticBase):
    def test_missing_front_invalidates_previous_refinement_at_entry(self):
        from herogen.blender.final_model import final_model
        from herogen.blender.stage3 import run_stage3
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "03-refined.glb").write_bytes(b"stale")
            (run / "03-source.glb").write_bytes(b"source")
            with self.assertRaises(ValueError):
                run_stage3(run, mesh_source="carve", joints_source="silhouette", fallback=False)
            self.assertFalse((run / "03-refined.glb").exists())
            self.assertNotEqual(final_model(run).name, "03-refined.glb")

    def test_masks_and_scale(self):
        self.assertEqual(sorted(self.vs.views), sorted(views.VIEWS))
        self.assertEqual(self.vs.missing, [])
        self.assertAlmostEqual(self.vs.pps, 60.0, delta=0.3)       # 330 px figure / 5.5 studs

    def test_view_mapping_puts_joints_inside_every_mask(self):
        pts = np.array(list(self.truth["joints"].values()))
        for name, v in self.vs.views.items():
            self.assertTrue(v.inside(pts).all(), name)

    def test_carved_hull_is_closed_and_contract_sized(self):
        verts, faces, info = carve.build_mesh(self.vs, 5.5, 96)
        self.assertEqual(_open_edges(faces), 0)
        self.assertAlmostEqual(verts[:, 2].min(), 0.0, places=6)
        self.assertAlmostEqual(verts[:, 2].max(), 5.5, places=6)
        # arm span from the front mask: within two voxels
        r0, r1, c0, c1 = views.bbox(self.vs.views["front"].mask)
        span = (c1 - c0 + 1) / self.vs.pps
        self.assertAlmostEqual(verts[:, 0].max() - verts[:, 0].min(), span, delta=2 * info["voxel_size_studs"])
        occ_pts = carve.winding_inside(np.array(list(self.truth["joints"].values())), verts, faces)
        self.assertGreaterEqual(int(occ_pts.sum()), len(occ_pts) - 1)   # HeadTop sits on the surface

    def test_missing_views_are_skipped_and_recorded(self):
        run = self.tmp / "missing"
        shutil.copytree(self.rundir, run)
        (run / "masks" / "top.png").unlink()
        (run / "masks" / "back.png").unlink()
        vs = views.load_views(run, 5.5)
        self.assertEqual(sorted(vs.missing), ["back", "top"])
        verts, faces, _ = carve.build_mesh(vs, 5.5, 64)
        self.assertEqual(_open_edges(faces), 0)


class TestStage3SourceNormals(SyntheticBase):
    @staticmethod
    def geometry():
        return (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
                np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))

    def run_source(self, source, run, rec):
        from herogen.blender import body_acceptance, stage3

        # Mock the model source and render job: this integration check launches neither models nor Blender.
        verts, faces = self.geometry()
        result = SimpleNamespace(path=lambda name: source)

        def render(job, work, name, **kwargs):
            for camera in job["tasks"][0]["cameras"]:
                view = camera["view"]
                shutil.copyfile(run / "masks" / f"{view}.png", run / "03-renders" / f"{view}.png")
            return {"blender": "test"}

        with patch.object(stage3, "_objects_mesh", return_value=(verts, faces, {}, result)), \
                patch.object(stage3.bl, "run_job", side_effect=render), \
                patch.object(body_acceptance, "assess"):
            return stage3._run_stage3(run, mesh_source="sam3d-objects", joints_source="silhouette",
                                     fallback=False, remote=object(), rec=rec)

    def test_source_normals_are_written_and_recorded_without_changing_gates(self):
        from herogen.blender import glb
        from herogen.blender.record import StageRecord

        run = self.tmp / "smooth-source"
        shutil.copytree(self.rundir, run)
        verts, faces = self.geometry()
        source = glb.write_glb(self.tmp / "cached-source.glb", [("Body", verts, faces)])
        rec = StageRecord(run, 3, "base mesh")
        result = self.run_source(source, run, rec)
        self.assertIn("03-source.glb: added seam-aware smooth normals (1e-6 grid; primitives: 1)", result["notes"])
        self.assertIn("03-source.glb", result["outputs"])
        self.assertEqual(result, json.loads((run / "stage-3.json").read_text()))
        before, after = glb.load_glb(source), glb.load_glb(run / "03-source.glb")
        np.testing.assert_array_equal(before.merged()[0], after.merged()[0])
        np.testing.assert_array_equal(before.merged()[1], after.merged()[1])
        doc, binary = glb._chunks((run / "03-source.glb").read_bytes())
        normal = doc["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        np.testing.assert_allclose(np.linalg.norm(glb._accessor(doc, binary, normal), axis=1), 1, atol=1e-7)
        checks = result["checks"]
        self.assertEqual({c["name"] for c in checks}, {f"silhouette_iou_{v}" for v in views.VIEWS}
                         | {"front_view_present", "triangles", "open_edges", "non_manifold_edges"})
        self.assertTrue(all(not c["gate"] for c in checks if c["name"].startswith("silhouette_iou_")))
        glb.add_missing_normals(source, source)
        authored = source.read_bytes()
        result = self.run_source(source, run, StageRecord(run, 3, "base mesh"))
        self.assertNotIn("added seam-aware", result["notes"])
        self.assertEqual((run / "03-source.glb").read_bytes(), authored)

    def test_invalid_source_normals_rewrite_fails_without_partial_output(self):
        from herogen.blender.glb import GlbError
        from herogen.blender.record import StageRecord

        run = self.tmp / "invalid-source"
        shutil.copytree(self.rundir, run)
        source = self.tmp / "bad-source.glb"
        source.write_bytes(b"not a GLB")
        with self.assertRaises(GlbError):
            self.run_source(source, run, StageRecord(run, 3, "base mesh"))
        self.assertFalse((run / "03-source.glb").exists())
        self.assertFalse((run / "03-source.glb.part").exists())
        self.assertFalse((run / "stage-3.json").exists())


class TestStage3BuildLog(SyntheticBase):
    """Stage 3's local work in the build log: mesh placement, facing search and where each joint came from."""
    geometry = staticmethod(TestStage3SourceNormals.geometry)
    run_source = TestStage3SourceNormals.run_source

    def logged(self, fn):
        from herogen.pipeline import events

        log = self.tmp / f"events-{self.id().rsplit('.', 1)[-1]}.jsonl"
        log.unlink(missing_ok=True)
        with patch.dict("os.environ", {"HEROGEN_EVENTS_FILE": str(log)}):
            result = fn()
        lines = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        self.assertEqual([events.validate(e) for e in lines], lines)
        self.assertTrue(all(e["step"] == "3" for e in lines))
        return result, lines

    def test_placement_and_outline_joints_are_logged(self):
        from herogen.blender import glb
        from herogen.blender.record import StageRecord

        run = self.tmp / "logged-source"
        shutil.copytree(self.rundir, run, dirs_exist_ok=True)
        verts, faces = self.geometry()
        source = glb.write_glb(self.tmp / "logged-source.glb", [("Body", verts, faces)])
        rec = StageRecord(run, 3, "base mesh")
        rec.data["fast_lane"] = True
        result, log = self.logged(lambda: self.run_source(source, run, rec))
        placed = [e for e in log if e.get("what") == "mesh_placed"]
        self.assertEqual(len(placed), 1)
        self.assertEqual(placed[0]["m"], {"triangles": 4, "vertices": 4, "open_edges": 0,
                                          "front_overlap": result["mesh"]["iou"]["front"],
                                          "depth_offset_studs": 0.0})
        joints = json.loads((run / "03-joints.json").read_text())["joints"]
        rows = [e for e in log if e.get("what", "").startswith("joint_from_")]
        self.assertEqual({e["what"] for e in rows}, {"joint_from_outline"})
        self.assertEqual(len(rows), len(joints))
        self.assertIn("left_shoulder", {e["name"] for e in rows})
        summary = [e["m"] for e in log if e.get("what") == "joints_mapped"]
        self.assertEqual(summary, [{"joints": len(joints), "from_model": 0, "from_outline": len(joints)}])
        self.assertEqual([(e["what"], e["why"]) for e in log if e["kind"] == "skip"],
                         [("acceptance_judges", "fast_lane")])

    def test_model_and_outline_joints_are_told_apart_with_their_distance(self):
        from herogen.blender import stage3

        rec = {"source": "sam3d-body", "joints": {"Neck": [0, 0, 4], "LeftShoulder": [1, 0, 3.6],
                                                  "LeftElbow": [1, 0, 3.4]},
               "joint_sources": {"Neck": "sam3d-body", "LeftShoulder": "silhouette (X/Z), mesh (Y)",
                                 "LeftElbow": "sam3d-body"},
               "distance_to_silhouette": {"Neck": 0.12, "LeftShoulder": 0.0, "LeftElbow": 0.73},
               "flag_threshold_studs": 0.5, "flagged": ["LeftElbow"]}
        _, log = self.logged(lambda: stage3._log_joints(rec))
        self.assertEqual([(e["what"], e["name"], e["m"]) for e in log[:3]], [
            ("joint_from_model", "neck", {"off_outline_studs": 0.12, "flagged": 0}),
            ("joint_from_outline", "left_shoulder", {"off_outline_studs": 0.0, "flagged": 0}),
            ("joint_from_model", "left_elbow", {"off_outline_studs": 0.73, "flagged": 1})])
        self.assertEqual(log[3]["m"], {"joints": 3, "from_model": 2, "from_outline": 1, "flagged": 1,
                                       "flag_studs": 0.5})

    def test_facing_search_reports_the_rotations_tried(self):
        from herogen.blender import sam3d_mesh, stage3
        from herogen.blender.record import StageRecord

        run = self.tmp / "facing"
        shutil.copytree(self.rundir, run, dirs_exist_ok=True)
        verts, faces = self.geometry()
        src = SimpleNamespace(faces=faces, corner_uv=np.zeros((len(faces), 3, 2)), texture=None)
        info = {"facing_search": {"0": {"iou": 0.61}, "90": {"iou": 0.4}, "180": {"iou": 0.93}, "270": {"iou": 0.41}},
                "facing_deg": 180.0, "front_iou_final_numpy": 0.951}
        remote = SimpleNamespace(objects=lambda *a, **k: SimpleNamespace(
            cached=False, entry={"cost_usd_estimate": 0.1}, dir=run, path=lambda name: run / name))
        with patch.object(sam3d_mesh, "load_source", return_value=src), \
                patch.object(sam3d_mesh, "pose_views", return_value=({}, {})), \
                patch.object(sam3d_mesh, "orient", return_value=(verts, info)), \
                patch.object(sam3d_mesh, "remesh", return_value=(verts, faces, {})), \
                patch.object(sam3d_mesh, "register_depth", return_value=(0.0, {"offset_studs": 0.0})):
            _, log = self.logged(lambda: stage3._objects_mesh(run, self.vs, 5.5, remote,
                                                              StageRecord(run, 3, "base mesh")))
        self.assertEqual(len(log), 1)
        self.assertEqual((log[0]["kind"], log[0]["what"]), ("local", "facing_search"))
        self.assertEqual(log[0]["m"], {"tested": 4, "facing_deg": 180.0, "overlap": 0.93, "front_overlap": 0.951})
        self.assertGreaterEqual(log[0]["s"], 0)

    def test_without_a_build_log_file_nothing_is_written(self):
        from herogen.blender import stage3

        from herogen.pipeline import events

        with patch.dict("os.environ", {}) as env, patch.object(events, "_append") as append:
            env.pop("HEROGEN_EVENTS_FILE", None)
            stage3._log_joints({"source": "silhouette", "joints": {"Neck": [0, 0, 4]}})
        append.assert_not_called()


@unittest.skipUnless(HAVE_DEPS and bl.find_blender() is not None, "Blender not installed")
class TestStage3Blender(SyntheticBase):
    def test_stage3_silhouette_iou(self):
        from herogen.blender.stage3 import run_stage3

        r3 = run_stage3(self.rundir, nz=128, timeout=300, mesh_source="carve", joints_source="silhouette", quality_client=quality_client())
        self.assertTrue(r3["ok"], r3["checks"])
        ious = [c["value"] for c in r3["checks"] if c["name"].startswith("silhouette_iou_")]
        self.assertEqual(len(ious), 5)
        self.assertGreater(min(ious), 0.9)
        self.assertEqual((self.rundir / "03-mesh.glb").read_bytes()[:4], b"glTF")
        self.assertEqual(sorted(p.name for p in (self.rundir / "03-renders").glob("*.png")),
                         sorted(f"{v}.png" for v in views.VIEWS))
        rec = json.loads((self.rundir / "stage-3.json").read_text())
        self.assertEqual((rec["stage"], rec["cost_usd"]), (3, 0.0))


if __name__ == "__main__":
    unittest.main()
