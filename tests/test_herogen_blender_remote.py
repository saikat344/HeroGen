"""herogen.blender.remote: the record/replay client for SAM 3D Objects and SAM 3D Body on Modal.

Modal is never called. Replay entries are written by the tests; the live path runs against a stand-in `modal`
executable (a small script in a temporary folder) that writes the outputs a runner would, so the command line,
the environment handed to the runner, the stored entry and the log scrubbing are all checked offline.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from herogen.blender.remote import client as rc

REMOTE = Path(rc.__file__).resolve().parent

# A stand-in for the modal CLI: `modal run <runner>.py::main --job J --out O` -> writes O/job/* and O/run.json,
# echoes its environment so the test can check what the runner would have seen, and prints a fake token.
FAKE_MODAL = r'''#!{python}
import json, os, sys, time
args = sys.argv[1:]
assert args[0] == "run", args
runner = os.path.basename(args[1].split("::")[0])
job = json.load(open(args[args.index("--job") + 1]))
out = args[args.index("--out") + 1]
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
if os.environ.get("FAKE_FAIL"):
    print("boom hf_" + "x" * 30, file=sys.stderr)
    sys.exit(3)
os.makedirs(os.path.join(out, "job"))
name = "mesh.glb" if runner == "objects.py" else "joints.json"
open(os.path.join(out, "job", name), "w").write(json.dumps({{"runner": runner, "job": job}}))
if os.environ.get("FAKE_FAIL_AFTER_OUTPUT"):
    print("line one\nline two", file=sys.stderr)
    sys.exit(4)
env = {{k: v for k, v in os.environ.items() if k.startswith("HEROGEN_")}}
json.dump({{"gpu": "fake", "env": env}}, open(os.path.join(out, "run.json"), "w"))
print("using token hf_" + "a" * 30 + " Authorization: Bearer secret123")
'''


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="herogen-remote-"))
        self.img = self.tmp / "front.png"
        self.msk = self.tmp / "mask.png"
        self.img.write_bytes(b"image bytes")
        self.msk.write_bytes(b"mask bytes")
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Replay(Base):
    def test_miss_then_hit_costs_nothing(self):
        c = rc.RemoteClient(self.cache, mode="replay")
        with self.assertRaises(rc.ReplayMiss) as cm:
            c.objects(self.img, self.msk)
        self.assertNotIn(str(self.tmp), str(cm.exception))      # records hold no local paths
        self.assertIn("(cache)", str(cm.exception))
        summary = rc.objects_summary(self.img, self.msk)
        d = self.cache / rc.request_key(summary)
        d.mkdir(parents=True)
        (d / "mesh.glb").write_bytes(b"x")
        (d / "entry.json").write_text(json.dumps({"request": summary, "cost_usd_estimate": 0.1}))
        r = c.objects(self.img, self.msk)
        self.assertTrue(r.cached)
        self.assertEqual(r.path("mesh.glb").read_bytes(), b"x")
        self.assertEqual(c.log.cost_usd(), 0.0)
        self.assertEqual(c.log.calls[0]["runner"], "sam3d-objects")
        with self.assertRaises(ValueError):
            rc.RemoteClient(self.cache, mode="other")

    def test_every_key_component_is_in_the_key(self):
        """The key covers the runner, its source hash, the code commit, the pinned weights (and MoGe for Body),
        the renderer and the inputs; dropping any of them, or changing the runner's source, changes the key."""
        objects = rc.objects_summary(self.img, self.msk)
        body = rc.body_summary(self.img, [0, 0, 10, 10], "front mask")
        self.assertEqual(set(objects), {"runner", "rev", "weights", "runner_sha", "renderer", "image", "mask", "seed"})
        self.assertEqual(set(body), {"runner", "rev", "weights", "runner_sha", "moge", "dinov3", "image", "bbox",
                                     "box_source", "detector"})
        consts = {n: RunnerFiles.constants(None, n) for n in ("objects.py", "body.py")}
        self.assertIn(objects["rev"].split("@")[1], consts["objects.py"]["REPO_SHA"])
        self.assertIn(body["rev"].split("@")[1], consts["body.py"]["REPO_COMMIT"])
        self.assertIn(body["moge"], consts["body.py"]["MOGE_COMMIT"])
        self.assertIn(body["dinov3"], consts["body.py"]["DINOV3_COMMIT"])
        for summary, runner in ((objects, "objects.py"), (body, "body.py")):
            self.assertEqual(summary["runner_sha"],
                             hashlib.sha256((REMOTE / runner).read_bytes()).hexdigest()[:16])
            self.assertTrue(summary["weights"])
            for repo, rev in summary["weights"].items():
                self.assertRegex(rev, r"^[0-9a-f]{40}$", repo)
            key = rc.request_key(summary)
            for k in summary:
                self.assertNotEqual(rc.request_key({x: v for x, v in summary.items() if x != k}), key, k)

    def test_a_changed_runner_changes_the_key_and_old_entries_say_why_they_miss(self):
        old = rc.objects_summary(self.img, self.msk)
        d = self.cache / rc.request_key(old)
        d.mkdir(parents=True)
        (d / "mesh.glb").write_bytes(b"x")
        (d / "entry.json").write_text(json.dumps({"request": old}))
        copy = self.tmp / "objects.py"
        copy.write_bytes((REMOTE / "objects.py").read_bytes() + b"\n# changed\n")
        orig = dict(rc.RUNNERS)
        rc.RUNNERS["sam3d-objects"] = copy
        try:
            new = rc.objects_summary(self.img, self.msk)
            self.assertNotEqual(new["runner_sha"], old["runner_sha"])
            self.assertNotEqual(rc.request_key(new), rc.request_key(old))
            with self.assertRaises(rc.ReplayMiss) as cm:
                rc.RemoteClient(self.cache).objects(self.img, self.msk)
            self.assertIn("made by a different runner (runner_sha differ)", str(cm.exception))
        finally:
            rc.RUNNERS.clear()
            rc.RUNNERS.update(orig)

    def test_body_detector_is_in_the_key_and_the_job(self):
        b = rc.body_summary(self.img, [0, 0, 10, 10], "front mask")
        self.assertIsNone(b["detector"])
        v = rc.body_summary(self.img, [0, 0, 10, 10], "front mask", "vitdet")
        self.assertNotEqual(rc.request_key(b), rc.request_key(v))
        with self.assertRaisesRegex(ValueError, "detector"):
            rc.body_summary(self.img, [0, 0, 10, 10], "front mask", "yolo")

    def test_key_depends_on_inputs_and_revision_not_on_the_device(self):
        a = rc.request_key(rc.objects_summary(self.img, self.msk))
        self.assertEqual(a, rc.request_key(rc.objects_summary(self.img, self.msk)))
        self.assertNotEqual(a, rc.request_key(rc.objects_summary(self.img, self.msk, seed=7)))
        self.assertNotEqual(a, rc.request_key(rc.objects_summary(self.msk, self.img)))
        b = rc.body_summary(self.img, [0, 0, 10, 10], "front mask")
        self.assertNotIn("gpu", json.dumps(b))
        self.assertNotEqual(rc.request_key(b), rc.request_key(rc.body_summary(self.img, [0, 0, 10, 12], "front mask")))
        self.assertNotEqual(rc.request_key(b), rc.request_key(rc.body_summary(self.img, [0, 0, 10, 10], "other")))
        # The key format is fixed: entries recorded before a rename of apps or volumes still replay.
        fixed = {"runner": "sam3d-body", "rev": "r", "image": "0" * 64}
        self.assertEqual(rc.request_key(fixed), "477c18a97b0edc63cb9befbd")

    def test_scrub_hides_tokens(self):
        s = rc._scrub("hf_" + "a" * 30 + " Authorization: Bearer abc token=xyz")
        self.assertNotIn("a" * 30, s)
        self.assertNotIn("abc", s)
        self.assertNotIn("xyz", s)

    def test_scrub_hides_the_workspace_and_home(self):
        home = str(Path.home())
        s = rc._scrub("View run at\nhttps://modal.com/apps/some-team/main/ap-abc123\n"
                      "https://modal.com/id/fc-xyz and https://some-team--app-fn.modal.run/path\n"
                      f"File {home}/code/herogen/blender/remote/body.py")
        self.assertNotIn("some-team", s)
        self.assertIn("https://modal.com/apps/workspace/main/ap-abc123", s)
        self.assertNotIn(home, s)
        self.assertIn("~/code/herogen/blender/remote/body.py", s)


class Live(Base):
    def setUp(self):
        super().setUp()
        fake = self.tmp / "modal"
        fake.write_text(FAKE_MODAL.format(python=sys.executable))
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        self._env = dict(os.environ)
        os.environ["HEROGEN_MODAL"] = str(fake)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        super().tearDown()

    def test_records_outputs_entry_and_runner_environment(self):
        c = rc.RemoteClient(self.cache, mode="live", timeout=60, body_gpu="T4")
        r = c.body(self.img, [1, 2, 3, 4], "front mask")
        self.assertFalse(r.cached)
        out = json.loads(r.path("joints.json").read_text())
        self.assertEqual(out["runner"], "body.py")
        self.assertEqual(out["job"]["jobs"][0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(out["job"]["jobs"][0]["box_source"], "front mask")
        self.assertIsNone(out["job"]["detector"])
        entry = json.loads((r.dir / "entry.json").read_text())
        self.assertEqual(entry["gpu"], "T4")
        self.assertEqual(entry["run"]["env"]["HEROGEN_BODY_GPU"], "T4")
        self.assertEqual(entry["run"]["env"]["HEROGEN_BODY_TIMEOUT"], "60")
        self.assertIn("joints.json", entry["files"])
        log = (r.dir / "modal.log").read_text()
        self.assertNotIn("a" * 30, log)
        self.assertNotIn("secret123", log)
        self.assertEqual(entry["cost_basis"], rc.COST_BASIS)
        self.assertGreaterEqual(entry["cost_usd_estimate"], 0.0)
        self.assertFalse(c.log.calls[0]["cached"])
        # the same request now replays, even in replay mode
        again = rc.RemoteClient(self.cache, mode="replay").body(self.img, [1, 2, 3, 4], "front mask")
        self.assertTrue(again.cached)

    def test_body_needs_a_person_box(self):
        c = rc.RemoteClient(self.cache, mode="live", timeout=60)
        for bbox, source in ((None, "front mask"), ([1, 2, 3], "front mask"), ([1, 2, 3, 4], "")):
            with self.assertRaisesRegex(ValueError, "person box"):
                c.body(self.img, bbox, source)
        self.assertFalse(self.cache.exists() and any(self.cache.iterdir()))

    def test_objects_gpu_is_configurable(self):
        r = rc.RemoteClient(self.cache, mode="live", timeout=60, objects_gpu="H100").objects(self.img, self.msk)
        entry = json.loads((r.dir / "entry.json").read_text())
        self.assertEqual(entry["run"]["env"]["HEROGEN_OBJECTS_GPU"], "H100")
        self.assertEqual(entry["run"]["env"]["HEROGEN_OBJECTS_TIMEOUT"], "60")

    def test_failure_and_timeout_raise_and_record_nothing(self):
        os.environ["FAKE_FAIL"] = "1"
        with self.assertRaises(rc.RemoteError) as cm:
            rc.RemoteClient(self.cache, mode="live", timeout=60).objects(self.img, self.msk)
        self.assertNotIn("x" * 30, str(cm.exception))
        del os.environ["FAKE_FAIL"]
        os.environ["FAKE_SLEEP"] = "5"
        with self.assertRaisesRegex(rc.RemoteError, "timed out"):
            rc.RemoteClient(self.cache, mode="live", timeout=0.5).objects(self.img, self.msk)
        self.assertFalse(self.cache.exists() and any(self.cache.iterdir()))

    def test_a_failed_exit_with_outputs_records_nothing_and_says_so_in_one_line(self):
        os.environ["FAKE_FAIL_AFTER_OUTPUT"] = "1"
        with self.assertRaises(rc.RemoteError) as cm:
            rc.RemoteClient(self.cache, mode="live", timeout=60).objects(self.img, self.msk)
        self.assertNotIn("\n", str(cm.exception))
        self.assertIn("exit 4", str(cm.exception))
        self.assertIn("line two", cm.exception.log)
        self.assertFalse(self.cache.exists() and any(self.cache.iterdir()))

    def test_a_cpu_run_keys_separately_from_gpu_runs(self):
        gpu = rc.RemoteClient(self.cache, mode="live", timeout=60).body(self.img, [1, 2, 3, 4], "front mask", gpu="T4")
        cpu = rc.RemoteClient(self.cache, mode="live", timeout=60).body(self.img, [1, 2, 3, 4], "front mask",
                                                                         gpu="none")
        self.assertNotEqual(gpu.dir.name, cpu.dir.name)
        self.assertEqual(json.loads((cpu.dir / "entry.json").read_text())["request"]["device"], "cpu")
        self.assertNotIn("device", json.loads((gpu.dir / "entry.json").read_text())["request"])
        # another GPU type replays the GPU entry
        again = rc.RemoteClient(self.cache, mode="replay").body(self.img, [1, 2, 3, 4], "front mask", gpu="L4")
        self.assertEqual((again.cached, again.dir.name), (True, gpu.dir.name))

    def test_an_unknown_gpu_has_an_unknown_cost(self):
        c = rc.RemoteClient(self.cache, mode="live", timeout=60, body_gpu="B999")
        r = c.body(self.img, [1, 2, 3, 4], "front mask")
        self.assertIsNone(r.entry["cost_usd_estimate"])
        self.assertEqual(r.entry["cost_basis"], rc.COST_UNKNOWN)
        self.assertIsNone(c.log.cost_usd())

    def test_modal_that_cannot_start_is_a_remote_error(self):
        broken = self.tmp / "modal-broken"
        broken.write_text("not a program")
        broken.chmod(0o644)
        os.environ["HEROGEN_MODAL"] = str(broken)
        with self.assertRaisesRegex(rc.RemoteError, "could not start modal"):
            rc.RemoteClient(self.cache, mode="live", timeout=60).body(self.img, [1, 2, 3, 4], "front mask")
        self.assertFalse(self.cache.exists() and any(self.cache.iterdir()))

    def test_missing_modal_is_a_clear_error(self):
        os.environ["HEROGEN_MODAL"] = str(self.tmp / "nope")
        os.environ["PATH"] = str(self.tmp)
        orig = rc.find_modal
        rc.find_modal = lambda: None
        try:
            with self.assertRaisesRegex(rc.RemoteError, "modal CLI not found"):
                rc.RemoteClient(self.cache, mode="live").objects(self.img, self.msk)
        finally:
            rc.find_modal = orig


class RunnerFiles(unittest.TestCase):
    """The Modal apps are not imported (they need modal); their configuration is read from the source."""

    def constants(self, name: str) -> dict:
        tree = ast.parse((REMOTE / name).read_text())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                out[node.targets[0].id] = ast.unparse(node.value)
        return out

    def test_generic_app_and_volume_names_overridable_by_environment(self):
        objects, body = (REMOTE / "objects.py").read_text(), (REMOTE / "body.py").read_text()
        self.assertEqual(self.constants("objects.py")["APP"], "'herogen-sam3d-objects'")
        self.assertIn('modal.App("herogen-sam3d-body")', body)
        self.assertIn('os.environ.get("HEROGEN_OBJECTS_VOLUME", "herogen-sam3d-objects-ckpt")', objects)
        self.assertIn('os.environ.get("HEROGEN_BODY_VOLUME", "herogen-sam3d-body-ckpt")', body)
        for text, prefix in ((objects, "HEROGEN_OBJECTS"), (body, "HEROGEN_BODY")):
            self.assertIn(f'os.environ.get("{prefix}_GPU"', text)
            self.assertIn(f'os.environ.get("{prefix}_TIMEOUT"', text)
            self.assertIn('modal.Secret.from_name("huggingface")', text)
            self.assertIn("SAM License", text)
            self.assertIsNone(re.search(r"^\s*(from|import) herogen", text, re.M))

    def test_objects_image_stays_licence_clean(self):
        objects = (REMOTE / "objects.py").read_text()
        consts = self.constants("objects.py")
        self.assertIn('config.rendering_engine = "pytorch3d"', objects)
        installs = [line for line in objects.splitlines() if "pip install" in line or "PIPX" in line]
        self.assertFalse([line for line in installs if "nvdiffrast" in line])
        # smplx (non-commercial research licence) is filtered out of upstream's requirements, and the build
        # fails if it, nvdiffrast or the Inria rasteriser can be imported
        for m in ("'smplx'", "'point-cloud-utils'", "'pdoc3'"):
            self.assertIn(m, consts["EXCLUDE"])
        for m in ("smplx", "nvdiffrast", "diff_gaussian_rasterization", "point_cloud_utils", "pdoc"):
            self.assertIn(f"'{m}'", consts["ABSENT"])
        # the weights at pinned revisions, and a volume holding another revision refused
        self.assertIn("revision=WEIGHTS[repo]", objects)
        self.assertIn("depth.revision = WEIGHTS[", objects)
        self.assertIn("checkpoint_revision(", objects)
        self.assertIn("absent_check)", objects)
        self.assertIn('log["absent_modules_imported"]', objects)
        # nothing imports them, even to probe for them; the splat renders always use gsplat
        tree = ast.parse(objects)
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse(imported & {"smplx", "nvdiffrast", "diff_gaussian_rasterization"})
        # upstream's functions are wrapped, never redefined here
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertFalse(defined & {"render_multiview", "render_frames", "render_multiview_gsplat"})
        self.assertIn("render_utils.render_frames = gsplat_backend(render_utils.render_frames)", objects)

    def test_gsplat_backend_forces_the_backend_and_passes_everything_else(self):
        tree = ast.parse((REMOTE / "objects.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gsplat_backend")
        ns = {}
        exec(compile(ast.Module([fn], []), "objects.py", "exec"), ns)
        calls = []

        def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True,  # noqa: B006
                          **kwargs):
            calls.append((sample, extrinsics, intrinsics, dict(options), colors_overwrite, verbose, kwargs))
            return "frames"

        wrapped = ns["gsplat_backend"](render_frames)
        opts = {"resolution": 512, "bg_color": (0, 0, 0)}
        self.assertEqual(wrapped("s", "e", "i", opts), "frames")
        self.assertEqual(calls[-1][3], {"resolution": 512, "bg_color": (0, 0, 0), "backend": "gsplat"})
        self.assertNotIn("backend", opts)                     # the caller's dict is not changed
        wrapped("s", "e", "i", {"backend": "inria"}, None, False, kernel_size=0.2)
        self.assertEqual(calls[-1][3]["backend"], "gsplat")
        self.assertEqual(calls[-1][4:], (None, False, {"kernel_size": 0.2}))
        wrapped("s", "e", "i")
        self.assertEqual(calls[-1][3], {"backend": "gsplat"})

    def test_body_image_pins_moge_and_skips_unused_packages(self):
        body = (REMOTE / "body.py").read_text()
        self.assertIn("absent_check,", body)                  # the build fails if a left-out package is there
        for m in ("smplx", "nvdiffrast", "wandb"):
            self.assertIn(f"'{m}'", self.constants("body.py")["ABSENT"])
        self.assertIn('snapshot_download(repo_id=repo_id, revision=WEIGHTS[repo_id])', body)
        self.assertIn('revision=WEIGHTS["Ruicheng/moge-2-vitl-normal"]', body)
        self.assertNotIn('"ffmpeg"', body.split("apt_install(")[1].split(")")[0])
        self.assertRegex(self.constants("body.py")["MOGE_COMMIT"], r"^'[0-9a-f]{40}'$")
        self.assertIn("MoGe.git@{MOGE_COMMIT}", body)
        self.assertNotRegex(body, r"MoGe\.git['\"]")
        start = body.index(".pip_install(\n        \"pytorch-lightning\"")
        listed = re.findall(r'"([^"]+)"', body[start:body.index(")", start)])
        for unused in ("chump", "appnope", "ffmpeg", "black", "pytest", "wandb"):
            self.assertNotIn(unused, listed)

    def test_body_backbone_code_is_pinned_not_fetched_from_github(self):
        """Upstream loads the DINOv3 backbone code from GitHub's main branch at run time; the image clones it at
        DINOV3_COMMIT, every call points the hub load at that copy, and the build fails if the call changed."""
        body = (REMOTE / "body.py").read_text()
        consts = self.constants("body.py")
        self.assertRegex(consts["DINOV3_COMMIT"], r"^'[0-9a-f]{40}'$")
        self.assertIn('git checkout {DINOV3_COMMIT}"', body)
        self.assertIn("dinov3_check,", body)
        tree = ast.parse(body)
        infer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "infer")
        first = [ast.unparse(n) for n in infer.body if not isinstance(n, (ast.Import, ast.ImportFrom))][0]
        self.assertEqual(first, "_pin_dinov3()")              # before anything imports the model
        keep = [n for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in
                                         ("DINOV3_DIR", "DINOV3_HUB_CALL"))
                or (isinstance(n, ast.FunctionDef) and n.name == "pin_dinov3_source")]
        ns = {}
        exec(compile(ast.Module(keep, []), "body.py", "exec"), ns)
        upstream = ('        self.encoder = torch.hub.load(\n            "facebookresearch/dinov3",\n'
                    '            self.name,\n            source="github",\n            pretrained=False,\n')
        pinned = ns["pin_dinov3_source"](upstream)
        self.assertIn('"/dinov3",', pinned)
        self.assertIn('source="local",', pinned)
        self.assertNotIn("github", pinned)
        with self.assertRaisesRegex(RuntimeError, "DINOv3 hub call"):
            ns["pin_dinov3_source"](upstream.replace('source="github",', 'source="github", force_reload=True,'
                                                     ).replace("self.name", '"facebookresearch/dinov3", self.name'))
        with self.assertRaisesRegex(RuntimeError, "DINOv3 hub call"):
            ns["pin_dinov3_source"]("torch.hub.load('facebookresearch/dinov3:main', name)")

    def test_body_person_detector_is_optional_and_off_by_default(self):
        """By default SAM 3D Body gets the caller's box and no detector runs or is downloaded; ViTDet (detectron2,
        weights under CC BY-SA 3.0) runs only when asked for, falling back to the caller's box if it finds
        nobody, and the run log records the box's source."""
        body = (REMOTE / "body.py").read_text()
        consts = self.constants("body.py")
        self.assertEqual(consts["DETECTORS"], "('vitdet',)")
        self.assertNotIn("'detectron2'", consts["ABSENT"])
        self.assertIn("detectron2.git@", body)                 # code only; the weights come on first use
        tree = ast.parse(body)
        infer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "infer")
        self.assertEqual([a.arg for a in infer.args.args], ["jobs", "detector"])
        self.assertEqual(ast.literal_eval(infer.args.defaults[0]), None)
        src = ast.get_source_segment(body, infer)
        self.assertIn("det = None\n    if detector is not None:", src)
        self.assertIn("human_detector=det", src)
        self.assertIn("outs = est.process_one_image(tmp, bboxes=caller_box)", src)
        self.assertIn('box_source = f"detector ({detector})"', src)
        self.assertIn('infer.remote(payload, spec.get("detector"))', body)


if __name__ == "__main__":
    unittest.main()
