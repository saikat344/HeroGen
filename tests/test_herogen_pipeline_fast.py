"""Fast-lane call counts, diagnostics and completion with synthetic inputs only."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from PIL import Image

from herogen.pipeline import client as client_module, stages
from herogen.pipeline.client import IMAGE_MODEL, ModelClient
from herogen.pipeline.__main__ import main, run_pipeline
from herogen.pipeline.report import report
from herogen.pipeline.runfolder import RunFolder
from herogen.blender import refine, stage3, body_acceptance
from herogen.blender.remote.client import RemoteClient
import test_herogen_pipeline_flow as flow
import test_herogen_blender_refine as refine_tests


class ConcurrentAPI(flow.FakeAPI):
    """Direct fake transport whose decoder keeps the calling thread's SAM input."""

    def __init__(self):
        self._sam = threading.local()
        super().__init__()

    @property
    def last_sam_image(self):
        return getattr(self._sam, "image", None)

    @last_sam_image.setter
    def last_sam_image(self, image):
        self._sam.image = image

    @property
    def last_sam_text(self):
        return getattr(self._sam, "text", "")

    @last_sam_text.setter
    def last_sam_text(self, text):
        self._sam.text = text


class FastPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_images_once_no_judges_no_side_stages_and_report_matches_records(self):
        api = ConcurrentAPI()
        original = api.post
        def post(client, endpoint, body, **kw):
            if endpoint == "images/edits" and not body["prompt"].startswith(
                    (stages.COVER_PROMPT, stages.TURNAROUND_PROMPT, stages.TOP_PROMPT)):
                api.calls.append(endpoint)
                img = Image.new("RGB", (300, 280), "white")  # intentionally fails all figure checks
                return {"data": [{"b64_json": flow._b64png(img)}]}
            return original(client, endpoint, body)
        failed = {"name": stages.CARVE_GATE, "pass": False, "gate": True, "value": 0, "threshold": .9}
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": flow.DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", post), \
                mock.patch.object(client_module, "decode_sam_output", api.decode), \
                mock.patch.object(stages, "carve_check", return_value=(failed, {})), \
                contextlib.redirect_stdout(io.StringIO()):
            code = run_pipeline(None, "hero", [0, 1, 2, "2h", "2f"], runs=str(self.root),
                                prompt="a blue knight", mode="live", fast=True)
        self.assertEqual(code, 0)
        run = RunFolder(self.root, "hero")
        self.assertFalse(run.exists("stage-2h.json"))
        self.assertFalse(run.exists("stage-2f.json"))
        self.assertEqual(api.calls.count("images/generations"), 1)
        self.assertEqual(api.calls.count("images/edits"), 4)
        self.assertNotIn("question", api.calls)
        for n in ("p", 0, 1, 2):
            rec = run.read_stage(n)
            self.assertTrue(rec["ok"])
            self.assertTrue(all(not c["gate"] for c in rec["checks"]))
        rec = run.read_stage(2)
        self.assertEqual(len(rec["details"]["attempts"]), 1)
        self.assertEqual(rec["details"]["top_method"], IMAGE_MODEL)
        self.assertTrue(any(not c["pass"] for c in rec["checks"]))
        result = report(run.path, stream=io.StringIO())
        records = [json.loads(p.read_text()) for p in run.path.glob("stage-*.json")]
        self.assertAlmostEqual(result["total"]["cost_usd"], sum(r["cost_usd"] for r in records))
        self.assertAlmostEqual(result["total"]["seconds"], sum(r["seconds"] for r in records))
        self.assertEqual(result["total"]["calls"], sum(len(r.get("calls", [])) for r in records))
        self.assertAlmostEqual(result["total"]["call_seconds"], sum(r["wall_seconds"] for r in result["calls"]))
        for row in result["calls"]:
            self.assertTrue({"stage", "purpose", "model", "kind", "started", "wall_seconds", "cost_usd",
                             "cached", "outcome"} <= row.keys())
        self.assertEqual(result, run.read_json("fast-report.json"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["report", str(run.path)]), 0)

    def test_cli_flag_and_environment_select_fast_execution(self):
        from herogen.pipeline import __main__ as pipeline_cli
        from herogen.blender.__main__ import main as blender_main
        with mock.patch.object(pipeline_cli, "run_pipeline", return_value=0) as execute:
            self.assertEqual(main(["run", "--run-id", "hero", "--prompt", "a knight", "--fast"]), 0)
        self.assertTrue(execute.call_args.kwargs["fast"])
        with mock.patch.dict(os.environ, {"HEROGEN_FAST_LANE": "1"}), \
                mock.patch("herogen.blender.fast.run", return_value=0) as execute:
            self.assertEqual(blender_main(["run", "--run", str(self.root)]), 0)
        self.assertEqual(execute.call_args.args[1], {"3", "3f", "4", "6", "5"})

    def test_unusable_run_directory_is_a_clear_cli_error(self):
        from herogen.blender.__main__ import main as blender_main
        invalid = self.root / "file"
        invalid.write_text("not a directory")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = blender_main(["run", "--run", str(invalid), "--fast"])
        self.assertEqual(code, 1)
        self.assertIn("fast run stopped:", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_remote_failure_recorded_without_fallback(self):
        remote = RemoteClient(self.root / "remote")
        with mock.patch.object(stage3, "load_views", side_effect=ValueError("bad input")):
            with self.assertRaises(ValueError):
                stage3.run_stage3(self.root, remote=remote, fast=True)
        rec = json.loads((self.root / "stage-3.json").read_text())
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["status"], "error")
        self.assertIsNone(rec["fallback"])
        self.assertFalse(remote.fast)

    def test_replay_miss_is_loud_in_pipeline_record_and_log(self):
        sketch = self.root / "sketch.png"
        Image.new("RGB", (64, 64)).save(sketch)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = run_pipeline(sketch, "miss", [0, 1], runs=str(self.root), fast=True)
        self.assertEqual(code, 1)
        rec = RunFolder(self.root, "miss").read_stage(1)
        self.assertEqual(rec["status"], "error")
        self.assertEqual(len(rec["calls"]), 1)
        self.assertEqual(rec["calls"][0]["outcome"], "ReplayMiss")
        self.assertIn("no cached", err.getvalue())

    def test_remote_shape_and_joints_once_each_in_call_order(self):
        from types import SimpleNamespace
        from herogen.blender import glb, synth
        import numpy as np
        synth.write_run(self.root, synth.humanoid(), size=64, ppu=10)
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        source = glb.write_glb(self.root / "cached.glb", [("Body", verts, faces)])
        remote = RemoteClient(self.root / "remote", fast=True)
        def gpu(runner, summary, *args, **kwargs):
            remote.log.add(summary["runner"], runner, True, 0.0)
            return SimpleNamespace(path=lambda name: source)
        def shape(run, vs, height, client, record):
            res = client.objects(run / "views/front.png", run / "masks/front.png")
            return verts, faces, {}, res
        def joints(run, vs, height, verts, faces, client, record, skeleton, **kwargs):
            client.body(run / "views/front.png", [0, 0, 64, 64], "synthetic")
            return {"flagged": []}
        def render(job, work, name, **kwargs):
            for camera in job["tasks"][0]["cameras"]:
                Image.new("L", (64, 64)).save(self.root / "03-renders" / f"{camera['view']}.png")
            return {"blender": "synthetic"}
        with mock.patch.object(remote, "_call", side_effect=gpu) as calls, \
                mock.patch.object(stage3, "_objects_mesh", side_effect=shape), \
                mock.patch.object(stage3, "_body_joints", side_effect=joints), \
                mock.patch.object(stage3.bl, "run_job", side_effect=render), \
                mock.patch.object(body_acceptance, "assess") as judge:
            rec = stage3.run_stage3(self.root, remote=remote, fast=True)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([c["model"] for c in rec["calls"]], ["sam-3d-objects", "sam-3d-body"])
        self.assertEqual([c["purpose"] for c in rec["calls"]], ["3D shape", "body joints"])
        self.assertTrue(all(c["kind"] == "gpu" and c["cached"] for c in rec["calls"]))
        self.assertTrue(all(c["stage"] == 3 and c["outcome"] == "ok" for c in rec["calls"]))
        self.assertEqual(rec["cost_usd"], 0)
        judge.assert_not_called()

    def test_stage3_failed_checks_are_informational_and_judge_is_skipped(self):
        from herogen.blender import synth
        synth.write_run(self.root, synth.humanoid(), size=64, ppu=10)
        def render(job, work, name, **kwargs):
            for camera in job["tasks"][0]["cameras"]:
                Image.new("L", (64, 64)).save(self.root / "03-renders" / f"{camera['view']}.png")
            return {"blender": "synthetic"}
        # An open triangle deliberately fails the structural check; no fallback or judge may decide its fate.
        import numpy as np
        mesh = (np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]], float), np.array([[0, 1, 2]]))
        remote = RemoteClient(self.root / "remote", fast=True)
        info = {"detached_side_prior_voxels": 0, "arm_depth_prior_columns": 0}
        with mock.patch.object(stage3.carve, "build_mesh", return_value=(*mesh, info)), \
                mock.patch.object(stage3.bl, "run_job", side_effect=render), \
                mock.patch.object(body_acceptance, "assess") as judge:
            rec = stage3.run_stage3(self.root, mesh_source="carve", joints_source="silhouette",
                                    remote=remote, fast=True)
        judge.assert_not_called()
        self.assertTrue(rec["ok"])
        self.assertTrue(all(not c["gate"] for c in rec["checks"]))
        self.assertFalse(next(c["pass"] for c in rec["checks"] if c["name"] == "open_edges"))

    def test_local_stage_failure_continues_export_results_and_keeps_source(self):
        from herogen.blender.__main__ import main as blender_main
        (self.root / "03-source.glb").write_bytes(b"source")
        (self.root / "stage-3.json").write_text(json.dumps({"stage": 3, "name": "base mesh", "ok": True,
                                                          "fast_lane": True}))
        def export(body, out):
            RunFolder(self.root.parent, self.root.name).write_stage(
                {"stage": 6, "name": "export", "ok": False, "error": "no body", "checks": []})
        with mock.patch("herogen.blender.stage4.run_stage4", side_effect=ValueError("bad sections")) as s4, \
                mock.patch("herogen.roblox.export.run_export", side_effect=export) as s6, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = blender_main(["run", "--run", str(self.root), "--stages", "4,6,5", "--fast"])
        self.assertEqual(code, 0)
        s4.assert_called_once()
        s6.assert_called_once()
        for n in (4, 6):
            rec = json.loads((self.root / f"stage-{n}.json").read_text())
            self.assertFalse(rec["ok"])
            self.assertEqual(rec["status"], "error")
            self.assertTrue(rec["error"])
            self.assertEqual(rec["final_model"], "03-source.glb")
        self.assertTrue((self.root / "results.html").exists())
        self.assertIn("03-source.glb", (self.root / "results.html").read_text())

    def test_failed_rebuild_cannot_export_a_previous_body(self):
        from herogen.blender.__main__ import main as blender_main
        (self.root / "03-source.glb").write_bytes(b"source")
        (self.root / "04-body").mkdir()
        (self.root / "04-body/high.glb").write_bytes(b"stale")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = blender_main(["run", "--run", str(self.root), "--stages", "4,6,5", "--fast"])
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "04-body/high.glb").exists())
        for stage in (4, 6):
            rec = json.loads((self.root / f"stage-{stage}.json").read_text())
            self.assertFalse(rec["ok"])
            self.assertTrue(rec["error"])
            self.assertEqual(rec["final_model"], "03-source.glb")
        self.assertTrue((self.root / "results.html").exists())

    def test_stage5_failure_is_recorded_and_refined_model_survives(self):
        from herogen.blender.__main__ import main as blender_main
        (self.root / "03-refined.glb").write_bytes(b"refined")
        with mock.patch("herogen.pipeline.results.stage5_results", side_effect=OSError("no space")) as s5, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = blender_main(["run", "--run", str(self.root), "--stages", "5", "--fast"])
        self.assertEqual(code, 0)
        s5.assert_called_once()
        rec = json.loads((self.root / "stage-5.json").read_text())
        self.assertEqual(rec["stage"], 5)
        self.assertFalse(rec["ok"])
        self.assertIn("no space", rec["error"])
        self.assertEqual(rec["final_model"], "03-refined.glb")

    def test_local_quality_failure_is_informational_without_extra_stages(self):
        from herogen.blender.__main__ import main as blender_main
        (self.root / "03-source.glb").write_bytes(b"source")
        def body(*args, **kwargs):
            RunFolder(self.root.parent, self.root.name).write_stage(
                {"stage": 4, "name": "body", "ok": False, "checks": [
                    {"name": "texture", "pass": False, "gate": True}]})
        with mock.patch("herogen.blender.stage4.run_stage4", side_effect=body) as s4, \
                mock.patch("herogen.roblox.export.run_export") as s6, \
                contextlib.redirect_stdout(io.StringIO()):
            code = blender_main(["run", "--run", str(self.root), "--stages", "4", "--fast"])
        self.assertEqual(code, 0)
        s4.assert_called_once()
        s6.assert_not_called()
        rec = json.loads((self.root / "stage-4.json").read_text())
        self.assertTrue(rec["ok"])
        self.assertFalse(rec["checks"][0]["gate"])
        self.assertFalse(rec["checks"][0]["pass"])


@unittest.skipUnless(refine_tests.HAVE_REFINE, "texture-refine dependencies are required")
class FastRefineTests(unittest.TestCase):
    setUp = refine_tests.StageTests.setUp
    identity = staticmethod(refine_tests.StageTests.identity)

    def test_every_repaint_once_and_applied_despite_all_gates_at_zero_cap(self):
        self.client.reject = True
        with mock.patch.object(refine_tests.refine, "part_gate", return_value=(False, {"face": "fails"}, None)):
            rec = refine.run_stage3f(self.run, self.client, fast=True, cap_usd=0)
        count = len(refine.REFINE_ORDER) + len(refine.HAIR_ORDER)
        self.assertEqual(len(self.client.inputs), count)
        self.assertEqual(self.tex.apply.call_count, count)
        self.assertTrue(all(c[2]["attempt"] == 0 for c in self.client.inputs))
        self.assertTrue(all(v["accepted"] and v["tries"] == 1 for v in rec["views"].values()))
        self.assertTrue(all(not v["attempts"][0]["gate_pass"] for v in rec["views"].values()))
        for name, view in rec["views"].items():
            expected = "hair outline, scale or colour agreement" if name.startswith("hair-") else "outline moved"
            self.assertEqual(view["attempts"][0]["reason"], expected)
        self.assertEqual(rec["final_model"], "03-refined.glb")
        self.assertEqual(rec["status"], "complete")
        self.assertIsNone(rec["budget"]["cap_usd"])
        self.assertEqual(rec["estimate_usd"], rec["estimate_max_samples_usd"])
        self.assertGreater(rec["cost_usd"], 0)
        normal = refine.run_stage3f(self.run, self.client, cap_usd=0)
        self.assertEqual(normal["status"], "cap_reached")
