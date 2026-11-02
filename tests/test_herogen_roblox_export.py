"""Stage 6 end to end: `python3 -m herogen.roblox export` on synthetic bodies: files, record, errors."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from herogen.roblox import (BIND_FILE, MESH_TRIANGLE_LIMIT, OUTPUT_FILES, README_FILE, RECORD_FILE, RIG_FILE,
                            RIG_NAME, RIG_TEST_FILE)
from herogen.roblox.__main__ import main
from herogen.roblox.check import Inst
from herogen.roblox.export import template
from herogen.roblox.record import file_hash
from herogen.roblox.synth import write_body


def run_cli(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(args))
    return code, out.getvalue(), err.getvalue()


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.tmp.name) / "run"
        cls.body = write_body(cls.run_dir / "04-body")
        cls.out = cls.run_dir / "06-roblox"
        cls.result = run_cli("export", "--body", str(cls.body), "--out", str(cls.out))
        cls.record = json.loads((cls.run_dir / RECORD_FILE).read_text())  # a run's record sits at its root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exit_code_and_summary(self):
        code, stdout, stderr = self.result
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("stage 6 ok", stdout)
        self.assertIn("HipHeight 1.7000", stdout)

    def test_writes_every_output(self):
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         sorted(n for n in OUTPUT_FILES if n != RECORD_FILE))
        self.assertEqual(self.record["outputs"],
                         [f"06-roblox/{n}" for n in (RIG_FILE, BIND_FILE, RIG_TEST_FILE, README_FILE)])

    def test_record(self):
        r = self.record
        self.assertEqual((r["stage"], r["name"], r["ok"], r["cost_usd"]), (6, "roblox export", True, 0.0))
        self.assertTrue(r["checks"] and all(c["pass"] for c in r["checks"]))
        self.assertEqual(r["inputs"], {"sections.json": file_hash(self.body / "sections.json"),
                                       "high.glb": file_hash(self.body / "high.glb")})
        self.assertEqual(r["body"], "04-body")
        self.assertIn("code", r)  # {"commit", "dirty"} in a checkout, None outside one
        if r["code"] is not None:
            self.assertEqual(len(r["code"]["commit"]), 40)
        self.assertEqual(r["rig"]["joints"], "Motor6D")
        self.assertEqual(len(r["rig"]["parts"]), 15)
        self.assertAlmostEqual(r["rig"]["hip_height"], 1.7, places=6)

    def test_xml_is_well_formed_with_resolving_referents(self):
        root = ET.parse(self.out / RIG_FILE).getroot()
        self.assertEqual((root.tag, root.get("version")), ("roblox", "4"))
        items = list(root.iter("Item"))
        refs = [i.get("referent") for i in items]
        self.assertEqual(len(refs), len(set(refs)))
        for el in root.iter("Ref"):
            self.assertIn(el.text, refs)
        model = Inst(root.find("Item"), None)
        self.assertEqual((model.cls, model.name), ("Model", RIG_NAME))
        classes = sorted(i.cls for i in model.walk())
        self.assertEqual(classes.count("MeshPart"), 15)
        self.assertEqual(classes.count("Motor6D"), 15)
        self.assertEqual(classes.count("Attachment"), 30 + 19)  # 15 joint pairs, 19 accessory attachments
        self.assertNotIn("MeshId", (self.out / RIG_FILE).read_text())  # placeholders: no mesh bound yet

    def test_deterministic(self):
        again = self.run_dir.parent / "again" / "06-roblox"
        shutil.copytree(self.body, again.parent / "04-body")
        code, _, _ = run_cli("export", "--body", str(again.parent / "04-body"), "--out", str(again))
        self.assertEqual(code, 0)
        for name in (RIG_FILE, BIND_FILE, RIG_TEST_FILE, README_FILE):
            self.assertEqual((again / name).read_bytes(), (self.out / name).read_bytes(), name)

    def test_scripts_and_readme_are_filled_in(self):
        bind = (self.out / BIND_FILE).read_text()
        self.assertIn(f'local RIG_NAME = "{RIG_NAME}"', bind)
        self.assertIn('local IMPORTED_NAME = "high"', bind)
        self.assertIn("bestErr < 0.01", bind)  # the fit residual gate
        self.assertEqual((self.out / RIG_TEST_FILE).read_text(), template(RIG_TEST_FILE))
        readme = (self.out / README_FILE).read_text()
        for want in ("../04-body/high.glb", "../stage-6.json", RIG_FILE, BIND_FILE, RIG_TEST_FILE, "Scale Unit: Stud",
                     "Rig Type: No Rig", "HipHeight is about 1.700", "StarterCharacter"):
            self.assertIn(want, readme)
        for name in (BIND_FILE, RIG_TEST_FILE, README_FILE):
            self.assertNotIn("{{", (self.out / name).read_text(), name)

    def test_check_command(self):
        code, stdout, _ = run_cli("check", "--body", str(self.body), "--rig", str(self.out / RIG_FILE))
        self.assertEqual(code, 0)
        self.assertIn("PASS", stdout.splitlines()[-1])
        bad = self.out.parent / "bad.rbxmx"
        bad.write_text((self.out / RIG_FILE).read_text().replace(">LeftHip<", ">LeftHipJoint<"))
        code, stdout, _ = run_cli("check", "--body", str(self.body), "--rig", str(bad))
        self.assertEqual(code, 1)
        self.assertIn("FAIL  joints_named_and_linked", stdout)


class ExportErrorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertOneLineError(self, stderr: str, *fragments: str):
        lines = stderr.splitlines()
        self.assertEqual(len(lines), 1, stderr)
        self.assertTrue(lines[0].startswith("error: "), stderr)
        self.assertNotIn("Traceback", stderr)
        for f in fragments:
            self.assertIn(f, lines[0])

    def test_triangle_refusal(self):
        body = write_body(self.root / "04-body", triangles={"LeftUpperLeg": MESH_TRIANGLE_LIMIT + 12})
        out = self.root / "06-roblox"
        out.mkdir()
        (out / RIG_FILE).write_text("an earlier run's rig")
        code, stdout, stderr = run_cli("export", "--body", str(body), "--out", str(out))
        self.assertEqual((code, stdout), (2, ""))
        self.assertOneLineError(stderr, "LeftUpperLeg has 20012 triangles",
                                "use the Roblox in-game or avatar profile")
        self.assertEqual(list(out.iterdir()), [])  # the stale rig is gone
        rec = json.loads((self.root / RECORD_FILE).read_text())
        self.assertFalse(rec["ok"])
        self.assertIn("use the Roblox in-game or avatar profile", rec["error"])

    def test_missing_body(self):
        code, _, stderr = run_cli("export", "--body", str(self.root / "nope"), "--out", str(self.root / "o"))
        self.assertEqual(code, 2)
        self.assertOneLineError(stderr, "missing input", "sections.json")

    def test_record_stays_in_out_outside_a_run(self):
        body = write_body(self.root / "bodies" / "hero")
        for out in (self.root / "elsewhere", self.root / "bodies" / "rig"):  # not a run's 06-roblox
            code, stdout, _ = run_cli("export", "--body", str(body), "--out", str(out))
            self.assertEqual(code, 0)
            self.assertIn(f"record {out / RECORD_FILE}", stdout)
            rec = json.loads((out / RECORD_FILE).read_text())
            self.assertEqual(rec["outputs"][0], RIG_FILE)
            self.assertIn(f"  - {RECORD_FILE}: the checks", (out / README_FILE).read_text())
        self.assertFalse((self.root / RECORD_FILE).exists())

    def test_record_at_run_root_found_by_stage_4_record(self):
        run = self.root / "run"
        body = write_body(self.root / "shared-body")
        (run / "06-roblox").mkdir(parents=True)
        (run / "stage-4.json").write_text("{}")
        self.assertEqual(run_cli("export", "--body", str(body), "--out", str(run / "06-roblox"))[0], 0)
        rec = json.loads((run / RECORD_FILE).read_text())
        self.assertEqual(rec["body"], "../shared-body")
        self.assertFalse((run / "06-roblox" / RECORD_FILE).exists())

    def test_check_with_missing_files(self):
        body = write_body(self.root / "04-body")
        code, _, stderr = run_cli("check", "--body", str(body), "--rig", str(self.root / "none.rbxmx"))
        self.assertEqual(code, 2)
        self.assertOneLineError(stderr, "none.rbxmx")

    def test_gate_failure_exits_1_and_names_the_check(self):
        body = write_body(self.root / "04-body", facing_back=True)
        code, stdout, stderr = run_cli("export", "--body", str(body), "--out", str(self.root / "06-roblox"))
        self.assertEqual(code, 1)
        self.assertIn("stage 6 FAILED", stdout)
        self.assertIn("FAIL  faces_minus_z", stderr)
        rec = json.loads((self.root / RECORD_FILE).read_text())
        self.assertFalse(rec["ok"])


# The 27 checks of a rig built from a complete body, and the one that is informational (not a hard limit).
RIG_CHECKS = ("xml_well_formed", "one_root_model", "humanoid_r15", "humanoid_scale_values", "root_part",
              "sections_r15_names", "joints_named_and_linked", "rig_attachment_pairs", "attachments_at_pivots",
              "attachment_pairs_coincide", "motor_c0_c1_equal_attachments", "rest_pose_position", "rest_pose_rotation",
              "joints_inside_parts", "joint_chain_reachable", "section_extents_match_glb", "section_centres_match_glb",
              "rotations_proper", "triangles_per_mesh", "hip_height", "feet_lowest", "faces_minus_z", "not_mirrored",
              "face_front_on_front", "accessory_attachments", "avatar_part_scale_type", "meshes_bound")
INFORMATIONAL = {"meshes_bound"}


class BuildLogEventsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"
        self.events = Path(self.tmp.name) / "events.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, **body_kw) -> tuple[dict, list[dict]]:
        from herogen.roblox.export import run_export

        body = write_body(self.root / "04-body", **body_kw)
        with mock.patch.dict(os.environ, {"HEROGEN_EVENTS_FILE": str(self.events), "HEROGEN_EVENTS_STEP": "6"}):
            rec = run_export(body, self.root / "06-roblox")
        lines = self.events.read_text().splitlines() if self.events.exists() else []
        return rec, [json.loads(line) for line in lines]

    def test_every_check_is_an_event_with_its_limit_comparison_and_gate(self):
        from herogen.pipeline import events

        rec, got = self.export()
        self.assertTrue(rec["ok"])
        for e in got:
            self.assertEqual(events.validate(e), e)
            self.assertEqual((e["step"], e["who"]), ("6", "herogen"))
        self.assertEqual([e["kind"] for e in got], ["local", "local"] + ["check"] * 27 + ["checks"])
        write, scripts = got[0], got[1]
        self.assertEqual(write["what"], "rig_write")
        self.assertEqual(write["m"], {"parts": 15, "motors": 15, "attachments": 19, "hip_height": 1.7})
        self.assertEqual((scripts["what"], scripts["m"]), ("studio_scripts", {"files": 3}))
        checks = got[2:-1]
        self.assertEqual(tuple(e["name"] for e in checks), RIG_CHECKS)
        self.assertEqual({e["name"] for e in checks if not e["gate"]}, INFORMATIONAL)
        self.assertEqual(sum(e["gate"] for e in checks), 26)
        self.assertTrue(all(e["ok"] for e in checks))
        by = {e["name"]: e for e in checks}
        for c in rec["checks"]:
            e = by[c["name"]]
            self.assertEqual(e["gate"], c["gate"])
            self.assertEqual(e.get("op"), c["rule"])
            self.assertEqual(e.get("m", {}).get("limit"), c["threshold"])
            # the build log keeps six decimals: a 5e-08 stud residual reads 0.0 against its 0.001 limit
            self.assertAlmostEqual(e["m"]["value"], c["value"], places=6)
        self.assertEqual({n: by[n]["op"] for n in ("attachments_at_pivots", "triangles_per_mesh", "hip_height",
                                                    "joints_inside_parts", "faces_minus_z", "not_mirrored",
                                                    "face_front_on_front")},
                         {"attachments_at_pivots": "<=", "triangles_per_mesh": "<=", "hip_height": "<=",
                          "joints_inside_parts": "==", "faces_minus_z": "<", "not_mirrored": ">",
                          "face_front_on_front": "<"})
        self.assertNotIn("op", by["meshes_bound"])
        summary = got[-1]
        self.assertEqual((summary["n"], summary["passed"], summary["gated"], summary["failed"], summary["advisory"]),
                         (27, 27, 26, [], []))
        self.assertGreaterEqual(summary["s"], 0)

    def test_a_failed_hard_check_is_named_and_the_rig_is_withheld(self):
        rec, got = self.export(facing_back=True)
        self.assertFalse(rec["ok"])
        failed = {e["name"] for e in got if e["kind"] == "check" and not e["ok"]}
        self.assertIn("faces_minus_z", failed)
        facing = next(e for e in got if e.get("name") == "faces_minus_z")
        self.assertEqual((facing["gate"], facing["op"], facing["m"]["limit"]), (True, "<", -0.9))
        self.assertGreater(facing["m"]["value"], 0.9)
        self.assertEqual(set(got[-1]["failed"]), failed)

    def test_an_unexportable_body_logs_an_error_without_its_message(self):
        from herogen.roblox.export import run_export

        body = write_body(self.root / "04-body")
        (body / "high.glb").unlink()
        with mock.patch.dict(os.environ, {"HEROGEN_EVENTS_FILE": str(self.events), "HEROGEN_EVENTS_STEP": "6"}):
            rec = run_export(body, self.root / "06-roblox")
        self.assertIn("error", rec)
        got = [json.loads(line) for line in self.events.read_text().splitlines()]
        self.assertEqual(got, [{**got[0], "kind": "error", "what": "rig_export", "code": "stage_error"}])
        self.assertNotIn(self.tmp.name, self.events.read_text())

    def test_without_an_events_file_the_export_writes_none(self):
        from herogen.pipeline import events
        from herogen.roblox.export import run_export

        body = write_body(self.root / "04-body")
        with mock.patch.dict(os.environ), \
                mock.patch.object(events, "_append", side_effect=AssertionError("written")):
            os.environ.pop("HEROGEN_EVENTS_FILE", None)
            self.assertTrue(run_export(body, self.root / "06-roblox")["ok"])


REPO = Path(__file__).resolve().parents[1]
INSTALLED_CHECK = """
import sys
from pathlib import Path
import herogen.roblox.export as export
from herogen.roblox.synth import write_body
site = Path(sys.argv[1]).resolve()
assert Path(export.__file__).resolve().is_relative_to(site), export.__file__
for name in export.TEMPLATE_FILES:
    assert export.template(name).strip(), name
run = Path(sys.argv[2])
rec = export.run_export(write_body(run / "04-body"), run / "06-roblox")
assert rec["ok"], rec.get("error") or [c for c in rec["checks"] if not c["pass"]]
print("templates found:", ", ".join(export.TEMPLATE_FILES))
"""


class InstalledPackageTest(unittest.TestCase):
    """`pip install` of the project ships the Studio script and README templates (package data)."""

    def test_installed_package_finds_its_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, site = Path(tmp) / "src", Path(tmp) / "site"
            # build from a copy, so the build never writes into the checkout
            shutil.copytree(REPO / "herogen", src / "herogen",
                            ignore=shutil.ignore_patterns("__pycache__", "runs"))
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(REPO / name, src / name)
            p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                                "--disable-pip-version-check", "--target", str(site), str(src)],
                               capture_output=True, text=True, timeout=300, cwd=tmp)
            if p.returncode != 0 and re.search(r"No matching distribution|Could not find a version|"
                                               r"Failed to establish a new connection", p.stderr):
                self.skipTest("pip could not fetch the build backend (offline)")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            q = subprocess.run([sys.executable, "-c", INSTALLED_CHECK, str(site), str(Path(tmp) / "run")],
                               capture_output=True, text=True, timeout=120, cwd=tmp,
                               env={**os.environ, "PYTHONPATH": str(site)})
            self.assertEqual(q.returncode, 0, q.stderr[-2000:])
            self.assertIn("templates found", q.stdout)


@unittest.skipUnless(shutil.which("luau-compile"), "luau-compile is not on PATH (optional syntax check)")
class LuauSyntaxTest(unittest.TestCase):
    def test_studio_scripts_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in (BIND_FILE, RIG_TEST_FILE):
                path = Path(tmp) / name
                path.write_text(template(name))
                p = subprocess.run(["luau-compile", "--binary", str(path)], capture_output=True, timeout=60)
                self.assertEqual(p.returncode, 0, f"{name}: {p.stderr.decode(errors='replace')}")


LUNE_READBACK = """
local fs = require("@lune/fs")
local process = require("@lune/process")
local serde = require("@lune/serde")
local roblox = require("@lune/roblox")
local model = roblox.deserializeModel(fs.readFile(process.args[1]))[1]
local function cf(c) return { c:GetComponents() } end
local out = { name = model.Name, primary = model.PrimaryPart and model.PrimaryPart.Name, parts = {},
	motors = {}, attachments = {} }
for _, d in model:GetDescendants() do
	if d:IsA("Humanoid") then
		out.hip = d.HipHeight
		out.rig = d.RigType.Name
	elseif d:IsA("BasePart") then
		out.parts[d.Name] = { class = d.ClassName, cframe = cf(d.CFrame), size = { d.Size.X, d.Size.Y, d.Size.Z } }
	elseif d:IsA("Motor6D") then
		out.motors[d.Name] = { c0 = cf(d.C0), c1 = cf(d.C1), part0 = d.Part0.Name, part1 = d.Part1.Name }
	elseif d:IsA("Attachment") then
		out.attachments[d.Parent.Name .. "/" .. d.Name] = cf(d.CFrame)
	end
end
print(serde.encode("json", out))
"""


@unittest.skipUnless(shutil.which("lune"), "Lune is not on PATH (optional rbx-dom cross-check)")
class RbxDomReadbackTest(unittest.TestCase):
    """rbx-dom (Roblox's reference model reader, run through Lune) reads the file back to the same values."""

    def test_rbx_dom_reads_the_same_rig(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = write_body(Path(tmp) / "04-body")
            out = Path(tmp) / "06-roblox"
            self.assertEqual(run_cli("export", "--body", str(body), "--out", str(out))[0], 0)
            script = Path(tmp) / "readback.luau"
            script.write_text(LUNE_READBACK)
            p = subprocess.run(["lune", "run", str(script), str(out / RIG_FILE)], capture_output=True,
                               text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            got = json.loads(p.stdout)
            model = Inst(ET.parse(out / RIG_FILE).getroot().find("Item"), None)
        self.assertEqual((got["name"], got["primary"], got["rig"]), (RIG_NAME, "HumanoidRootPart", "R15"))
        self.assertAlmostEqual(got["hip"], 1.7, places=5)
        self.assertEqual((len(got["parts"]), len(got["motors"]), len(got["attachments"])), (16, 15, 49))
        tol = 1e-5
        for inst in model.walk():
            if inst.cls in ("Part", "MeshPart"):
                g = got["parts"][inst.name]
                self.assertEqual(g["class"], inst.cls)
                self.assertLess(_worst(g["cframe"], inst.props["CFrame"]), tol, inst.name)
                self.assertLess(_worst(g["size"], inst.props["size"]), tol, inst.name)
            elif inst.cls == "Motor6D":
                g = got["motors"][inst.name]
                self.assertEqual((g["part0"], g["part1"]), (_parent_name(model, inst), inst.parent.name))
                self.assertLess(_worst(g["c0"], inst.props["C0"]), tol, inst.name)
                self.assertLess(_worst(g["c1"], inst.props["C1"]), tol, inst.name)
            elif inst.cls == "Attachment":
                key = f"{inst.parent.name}/{inst.name}"
                self.assertLess(_worst(got["attachments"][key], inst.props["CFrame"]), tol, key)


def _worst(a, b) -> float:
    assert len(a) == len(b)
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _parent_name(model: Inst, motor: Inst) -> str:
    ref = motor.props["Part0"][1]
    return next(i.name for i in model.walk() if i.ref == ref)


if __name__ == "__main__":
    unittest.main()
