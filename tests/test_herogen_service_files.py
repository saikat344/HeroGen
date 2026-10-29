"""Files: path safety for run files, per-user layout, downloads (GLB and the Roblox bundle), the Lab API shape."""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import TempDirCase, png

from herogen.service import bundle, labapi
from herogen.service.fakes import FakeRunner, box_glb
from herogen.service.runner import PLAN, StageContext
from herogen.service.storage import Storage
from herogen.service.store import new_id


def full_run(storage: Storage, uid: str, jid: str, name: str = "") -> StageContext:
    """A run folder with every stage made by the fake runner."""
    storage.save_input(uid, jid, png(400, 600))
    ctx = StageContext(job_id=jid, kind="sketch", profile="roblox-ingame", name=name, prompt="",
                       run_dir=storage.run_dir(uid, jid), input_path=storage.input_path(uid, jid),
                       cache_dir=storage.cache_dir(uid), remote_cache_dir=storage.remote_cache_dir(uid))
    r = FakeRunner()
    for spec in PLAN:
        if not spec.prompt_only:
            assert r.run(spec, ctx).state == "passed", spec.key
    return ctx


class TestRunFiles(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.st = Storage(self.tmp / "data")
        self.uid, self.jid = new_id("u"), new_id("j")
        self.ctx = full_run(self.st, self.uid, self.jid)

    def test_allowed_files(self):
        for rel in ("00-sketch.png", "views/front.png", "04-body/high.glb", "stage-4.json"):
            got = self.st.run_file(self.uid, self.jid, rel)
            self.assertIsNotNone(got, rel)
            self.assertTrue(got[0].is_relative_to(self.ctx.run_dir.resolve()))
        self.assertEqual(self.st.run_file(self.uid, self.jid, "04-body/high.glb")[1], "model/gltf-binary")

    def test_refused_paths(self):
        outside = self.tmp / "secret.json"
        outside.write_text("{}")
        (self.ctx.run_dir / ".hidden.json").write_text("{}")
        (self.ctx.run_dir / "notes.txt").write_text("x")
        (self.ctx.run_dir / "page.html").write_text("<script>")
        (self.ctx.run_dir / "x.svg").write_text("<svg/>")
        os.symlink(outside, self.ctx.run_dir / "link.json")
        os.symlink(self.ctx.run_dir / "views", self.ctx.run_dir / "linkdir")
        other_jid = new_id("j")
        full_run(self.st, self.uid, other_jid)
        bad = ["", "/etc/passwd", "../" + other_jid + "/00-sketch.png", "views/../../" + other_jid + "/stage-0.json",
               "..", ".", "views/./front.png", "views//front.png", ".hidden.json", "notes.txt", "page.html", "x.svg",
               "link.json", "linkdir/front.png", "views\\front.png", "00-sketch.png\x00.json", "missing.png",
               "views", "a/b/c/d/e/f/g.png", "%2e%2e/stage-0.json", "views/front.png/"]
        for rel in bad:
            self.assertIsNone(self.st.run_file(self.uid, self.jid, rel), repr(rel))
        for uid, jid in (("u_bad", self.jid), (self.uid, "../x"), ("../../etc", self.jid), (self.uid, new_id("u"))):
            self.assertIsNone(self.st.run_file(uid, jid, "00-sketch.png"))
        with self.assertRaises(ValueError):
            self.st.run_dir("../x", self.jid)

    def test_symlinked_run_folder_is_refused(self):
        jid = new_id("j")
        os.symlink(self.ctx.run_dir, self.st.run_dir(self.uid, jid))
        self.assertIsNone(self.st.run_file(self.uid, jid, "00-sketch.png"))

    def test_bytes_and_delete(self):
        n = self.st.bytes_used(self.uid, self.jid)
        self.assertGreater(n, 1000)
        self.st.delete_job(self.uid, self.jid)
        self.assertFalse(self.st.run_dir(self.uid, self.jid).exists())
        self.assertFalse(self.st.input_path(self.uid, self.jid).exists())
        self.st.delete_user(self.uid)
        self.assertFalse(self.st.user_dir(self.uid).exists())


class TestBundle(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.st = Storage(self.tmp / "data")
        self.uid, self.jid = new_id("u"), new_id("j")
        self.run = full_run(self.st, self.uid, self.jid).run_dir

    def test_roblox_bundle_contents(self):
        data = bundle.roblox_bundle(self.run)
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertEqual(sorted(z.namelist()), sorted(
            ["06-roblox/HeroGen-R15.rbxmx", "06-roblox/bind_meshes.luau", "06-roblox/rig_test.server.luau",
             "06-roblox/README.txt", "04-body/high.glb", "stage-6.json"]), "the run's own layout")
        self.assertEqual(z.read("04-body/high.glb"), (self.run / "04-body" / "high.glb").read_bytes())
        self.assertTrue(all(i.date_time == bundle.ZIP_DATE for i in z.infolist()))
        self.assertIn("Placeholder", bundle.roblox_readme(self.run))

    def test_bundle_takes_only_safe_listed_outputs(self):
        rec = json.loads((self.run / "stage-6.json").read_text())
        secret = self.tmp / "secret.txt"
        secret.write_text("private")
        os.symlink(secret, self.run / "06-roblox" / "linked.txt")
        rec["outputs"] += ["../../../../secret.txt", "/etc/hosts", "06-roblox/linked.txt", "04-body/high.glb",
                           "stage-6.json", 5, None]
        (self.run / "stage-6.json").write_text(json.dumps(rec))
        names = zipfile.ZipFile(io.BytesIO(bundle.roblox_bundle(self.run))).namelist()
        self.assertFalse([n for n in names if "secret" in n or "linked" in n or "hosts" in n or ".." in n], names)
        self.assertEqual(names.count("stage-6.json"), 1)
        self.assertEqual(names.count("04-body/high.glb"), 1)
        self.assertFalse(any(n.startswith("/") for n in names))

    def test_bundle_unavailable_reasons(self):
        rec = json.loads((self.run / "stage-6.json").read_text())
        rec["ok"] = False
        (self.run / "stage-6.json").write_text(json.dumps(rec))
        with self.assertRaises(bundle.BundleUnavailable) as cm:
            bundle.roblox_bundle(self.run)
        self.assertIn("did not pass", str(cm.exception))
        rec["ok"] = True
        rec["outputs"] = ["06-roblox/README.txt"]
        (self.run / "stage-6.json").write_text(json.dumps(rec))
        with self.assertRaises(bundle.BundleUnavailable):
            bundle.roblox_bundle(self.run)
        (self.run / "stage-6.json").unlink()
        with self.assertRaises(bundle.BundleUnavailable) as cm:
            bundle.roblox_bundle(self.run)
        self.assertIn("not been built", str(cm.exception))

    def test_glb(self):
        self.assertEqual(bundle.glb_path(self.run, "high"), self.run / "04-body" / "high.glb")
        self.assertIsNone(bundle.glb_path(self.run, "low"))
        self.assertIsNone(bundle.glb_path(self.run, "../../x"))
        glb = box_glb()
        self.assertEqual(glb[:4], b"glTF")
        self.assertEqual(int.from_bytes(glb[8:12], "little"), len(glb))


class TestLabApi(TempDirCase):
    def test_workspace_and_detail_have_the_lab_shape(self):
        st = Storage(self.tmp / "data")
        uid = new_id("u")
        a, b, c = new_id("j"), new_id("j"), new_id("j")
        full_run(st, uid, a)
        full_run(st, uid, b)
        full_run(st, uid, c)
        ws = labapi.workspace(st.runs_root(uid), [a, b])
        self.assertEqual(ws["roots"][0]["key"], "r1")
        self.assertEqual(ws["roots"][0]["label"], "Your heroes", "no server path is shown")
        self.assertEqual(set(ws["runs"]), {f"r1/{a}", f"r1/{b}"}, "only the jobs the store lists")
        self.assertTrue(all(set(ch["runs"]) <= set(ws["runs"]) for ch in ws["characters"]))
        d = labapi.run_detail(st.runs_root(uid), a)
        self.assertEqual((d["id"], d["root"], d["name"]), (f"r1/{a}", "r1", a))
        self.assertNotIn(str(self.tmp), json.dumps(d))
        self.assertEqual([s["stage"] for s in d["stages"]][:3], ["0", "1", "2"])
        self.assertTrue(all("status" in s for s in d["stages"]))
        self.assertIn("sketch", d["images"])
        self.assertIsNone(labapi.run_detail(st.runs_root(uid), new_id("j")))


# The record herogen.roblox's export writes at the run root for `--out <run>/06-roblox` (copied from a real run
# of `python3 -m herogen.roblox export` on that package's synthetic body; checks and parts trimmed).
REAL_STAGE6 = {
    "stage": 6, "name": "roblox export", "ok": True, "started": "2026-10-29T16:17:48-07:00", "seconds": 0.07,
    "cost_usd": 0.0, "cost_basis": "local compute only (Python and numpy); nothing uploaded", "models": [],
    "tools": [], "code": {"commit": "80a38a1f8244a6e92777733c6f04cc76867c7f6f", "dirty": False}, "body": "04-body",
    "inputs": {"sections.json": {"sha256": "15a88174e15a438f7bc148f164a0df4f2096094348ff8fc9802aa85f2c692986",
                                 "bytes": 4335},
               "high.glb": {"sha256": "b254d25b8d23e64a65f34b7bcdf22d59d2e59457871226b65a185b345a2331d7",
                            "bytes": 12288}},
    "outputs": ["06-roblox/HeroGen-R15.rbxmx", "06-roblox/bind_meshes.luau", "06-roblox/rig_test.server.luau",
                "06-roblox/README.txt"],
    "checks": [{"name": "xml_well_formed", "value": 1, "threshold": 1, "pass": True, "gate": True,
                "detail": "root <roblox version='4'>"},
               {"name": "attachments_at_pivots", "value": 0.0, "threshold": 0.001, "pass": True, "gate": True,
                "detail": "studs from the sections.json pivots"}],
    "notes": "",
    "rig": {"name": "HeroGen-R15", "file": "HeroGen-R15.rbxmx", "joints": "Motor6D", "hip_height": 1.7,
            "parts": {"Head": {"center": [0.0, 4.9, 0.0], "size": [1.0, 1.2, 1.0], "pivot": [0.0, 4.3, 0.0],
                               "triangles": 12}}},
}
REAL_STAGE6_ERROR = {**REAL_STAGE6, "ok": False, "outputs": [], "checks": [], "rig": None,
                     "error": "missing input: /workers/tmp/herogen-work/runs/j_x/04-body/sections.json"}


def real_export_run(root: Path, record: dict) -> Path:
    """A run folder laid out as the real export leaves it: 04-body, 06-roblox and stage-6.json at the root."""
    run = root / "run"
    (run / "04-body").mkdir(parents=True)
    (run / "04-body" / "high.glb").write_bytes(box_glb())
    (run / "06-roblox").mkdir()
    for rel in record["outputs"]:
        (run / rel).write_text(f"-- {rel}\n" if rel.endswith(".luau") else f"{rel}\n")
    (run / "stage-6.json").write_text(json.dumps(record))
    return run


class TestRealExportRecord(TempDirCase):
    def test_bundle_from_the_real_record(self):
        run = real_export_run(self.tmp, REAL_STAGE6)
        names = sorted(zipfile.ZipFile(io.BytesIO(bundle.roblox_bundle(run))).namelist())
        self.assertEqual(names, sorted(REAL_STAGE6["outputs"] + ["04-body/high.glb", "stage-6.json"]))
        # The README the export writes points at ../04-body/high.glb and ../stage-6.json from 06-roblox/.
        self.assertIn("06-roblox/README.txt", bundle.roblox_readme(run))

    def test_outcomes_from_the_real_record(self):
        from herogen.service.runner import PLAN, CliRunner

        spec = next(s for s in PLAN if s.key == "6")
        ctx = StageContext(job_id="j_" + "a" * 26, kind="sketch", profile="roblox-ingame", name="", prompt="",
                           run_dir=real_export_run(self.tmp, REAL_STAGE6), input_path=self.tmp / "in.png",
                           cache_dir=self.tmp / "c", remote_cache_dir=self.tmp / "rc")
        self.assertEqual(CliRunner.outcome(spec, ctx, 0.1, 0, "", "", False).state, "passed")
        failed = {**REAL_STAGE6, "ok": False, "checks": [{**REAL_STAGE6["checks"][1], "value": 0.2, "pass": False}]}
        (ctx.run_dir / "stage-6.json").write_text(json.dumps(failed))
        out = CliRunner.outcome(spec, ctx, 0.1, 1, "", "", False)
        self.assertEqual(out.state, "rejected")
        self.assertIn("attachments_at_pivots measured 0.2", out.message)
        (ctx.run_dir / "stage-6.json").write_text(json.dumps(REAL_STAGE6_ERROR))
        out = CliRunner.outcome(spec, ctx, 0.1, 2, "", "error: missing input", False)
        self.assertEqual((out.state, out.code, out.transient), ("failed", "stage_error", False))
        self.assertNotIn("/workers", out.message, "a worker path never reaches the user")
        with self.assertRaises(bundle.BundleUnavailable):
            bundle.roblox_bundle(ctx.run_dir)

    def test_export_command(self):
        from herogen.service.runner import PLAN, CliRunner

        spec = next(s for s in PLAN if s.key == "6")
        run = self.tmp / "runs" / ("j_" + "a" * 26)
        ctx = StageContext(job_id=run.name, kind="sketch", profile="roblox-ingame", name="", prompt="", run_dir=run,
                           input_path=self.tmp / "in.png", cache_dir=self.tmp / "c", remote_cache_dir=self.tmp / "rc")
        argv, stdin = CliRunner(python="py").command(spec, ctx, {}, self.tmp)
        self.assertEqual(argv, ["py", "-m", "herogen.roblox", "export", "--body", str(run / "04-body"), "--out",
                                str(run / "06-roblox")])
        self.assertIsNone(stdin)
