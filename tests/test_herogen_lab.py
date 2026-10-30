"""Lab server: run index, path safety, content types, read-only serving, and the page's external references."""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import socket
import struct
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zlib
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.lab import runindex  # noqa: E402
from herogen.lab.server import CDN_PREFIX, STATIC_DIR, content_security_policy, make_server, safe_run_file  # noqa: E402,E501


def tiny_png(rgb=(200, 40, 40), size=2) -> bytes:
    """A valid size x size RGB PNG, written by hand."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def tiny_glb(names=runindex.SECTIONS) -> bytes:
    """A GLB with one empty node per section at a pivot; no meshes or binary chunk."""
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(names)))}],
           "nodes": [{"name": n, "translation": [0.0, float(i) * 0.3, 0.0]} for i, n in enumerate(names)]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    return struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js)) + struct.pack("<II", len(js), 0x4E4F534A) + js


def record(stage, name, ok=True, checks=(), seconds=1.5, cost=0.01, details=None):
    return {"stage": stage, "name": name, "ok": ok, "started": "2026-10-28T19:00:00-07:00", "seconds": seconds,
            "cost_usd": cost, "cost_basis": "estimate", "models": ["muse-image-1.0"], "outputs": [],
            "checks": list(checks), "notes": "", "details": details or {}}


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(json.dumps(data))


def make_full_run(root: Path, run_id: str = "full") -> Path:
    run = root / run_id
    write(run / "stage-0.json", record(0, "input", checks=[
        {"name": "min_side_px", "value": 1024.0, "threshold": 256, "op": ">=", "pass": True, "gate": True}],
        details={"label": "synthetic sketch"}))
    write(run / "stage-1.json", record(1, "cover_art"))
    write(run / "stage-2.json", record(2, "turnaround", checks=[
        {"name": "left_profile_width_over_height", "value": 0.34, "threshold": 0.33, "op": "<=", "pass": False,
         "gate": False}]))
    write(run / "stage-2h.json", record("2h", "held-out view"))
    write(run / "stage-3.json", record(3, "base mesh", checks=[
        {"name": "silhouette_iou_front", "value": 0.97, "threshold": 0.9, "pass": True},
        {"name": "triangles", "value": 1000, "threshold": None, "pass": True, "gate": False}]))
    write(run / "stage-4.json", record(4, "game body", ok=False, checks=[
        {"name": "triangles_high", "value": 22768, "threshold": 24000, "pass": True, "rule": "<="},
        {"name": "pivots_in_joint_region", "value": 15, "threshold": 15, "pass": True,
         "rule": "pivot inside its section and within one cut radius of the mask joint"},
        {"name": "texture_ssim_front", "value": 0.69, "threshold": 0.7, "pass": False, "gate": True}]))
    for p in ["00-sketch.png", "01-cover.png", "02-turnaround.png", "heldout/front-left-34.png",
              "03-renders/front.png", "04-body/rest-renders/textured-front.png",
              *[f"views/{v}.png" for v in runindex.TURNAROUND_VIEWS],
              "05-renders/carve/front-left-34.png", "05-renders/hybrid/front-left-34.png"]:
        write(run / p, tiny_png())
    sections = {"units": "studs", "height": 5.5, "mesh_source": "test", "budgets": {
        "triangles_high_max": 24000, "triangles_low_max": 6000},
        "sections": [{"name": n, "parent": None if n == "LowerTorso" else "UpperTorso", "joint": "J",
                      "pivot": [0.0, 0.0, float(i)], "triangles_high": 100, "triangles_low": 25}
                     for i, n in enumerate(runindex.SECTIONS)]}
    for d in ("04-body", "04-body-carve"):
        write(run / d / "high.glb", tiny_glb())
        write(run / d / "low.glb", tiny_glb())
        write(run / d / "sections.json", sections)
    write(run / "04-body-carve" / "stage-4.json", record(4, "game body", ok=True))
    view = {"iou": 0.9, "colour_error": 7.0, "ssim": 0.8, "seen": True}
    held = {"iou": 0.6, "colour_error": 40.0, "ssim": 0.1, "seen": False, "yaw_deg": 28.0, "scale": 3.3}
    write(run / "05-scores.json", {"bodies": [
        {"name": "carve", "glb": "04-body-carve/high.glb", "triangles": 23000, "atlas": 1024,
         "views": {"front": view, "back": view, "left": view, "right": view, "front-left-34": held}},
        {"name": "hybrid", "glb": "04-body/high.glb", "triangles": 22768, "atlas": 1024,
         "views": {"front": view, "front-left-34": held}},
        {"name": "hybrid-variant", "glb": "04-body/high.glb", "variant_of": "hybrid", "views": {}},
        {"name": "../evil", "glb": "x", "views": {}},
    ]})
    return run


class RunIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.root.mkdir()
        self.run = make_full_run(self.root)
        # A partial run: only a body, no records upstream, no scores.
        write(self.root / "partial" / "04-body" / "high.glb", tiny_glb())
        (self.root / "not-a-run").mkdir()
        (self.root / ".hidden").mkdir()
        write(self.root / ".hidden" / "stage-0.json", record(0, "input"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_list(self):
        runs = {r["id"]: r for r in runindex.run_list(self.root)}
        self.assertEqual(set(runs), {"full", "partial"})
        self.assertEqual(runs["full"]["stages"]["4"], "gate failed")
        self.assertEqual(runs["full"]["stages"]["2h"], "pass")
        self.assertTrue(runs["full"]["has_scores"])
        self.assertEqual(runs["partial"]["stages"]["0"], "not run")
        self.assertTrue(runs["partial"]["has_body"])
        self.assertEqual(runindex.run_list(self.root / "missing"), [])

    def test_detail_stages_and_checks(self):
        d = runindex.run_detail(self.run)
        self.assertEqual([s["stage"] for s in d["stages"]], ["0", "1", "2", "2h", "3", "4", "5"])
        by = {s["stage"]: s for s in d["stages"]}
        self.assertEqual(by["5"]["status"], "not run")
        self.assertEqual(by["0"]["label"], "synthetic sketch")
        self.assertEqual(by["0"]["checks"][0]["limit"], ">= 256")
        self.assertEqual(by["2"]["checks"][0]["limit"], "<= 0.33")
        self.assertFalse(by["2"]["checks"][0]["gate"])
        mesh = {c["name"]: c for c in by["3"]["checks"]}
        self.assertEqual(mesh["silhouette_iou_front"]["limit"], ">= 0.9")  # no op: >= as on results.html
        self.assertEqual(mesh["triangles"]["limit"], "–")
        self.assertIsNone(mesh["triangles"]["cmp"])
        body = {c["name"]: c for c in by["4"]["checks"]}
        self.assertEqual(body["triangles_high"]["limit"], "<= 24000")  # Blender `rule`
        pivots = body["pivots_in_joint_region"]
        self.assertEqual((pivots["cmp"], pivots["threshold"], pivots["limit"]), (">=", 15, ">= 15"))
        self.assertTrue(pivots["cmp_assumed"])
        self.assertTrue(any("cut radius" in n and n.startswith("rule: ") for n in pivots["notes"]), pivots["notes"])
        self.assertIn(runindex.ASSUMED_CMP_NOTE, pivots["notes"])
        self.assertFalse(body["triangles_high"]["cmp_assumed"])
        self.assertEqual(body["triangles_high"]["notes"], [])
        self.assertEqual(by["4"]["checks_failed_gate"], 1)
        self.assertAlmostEqual(d["total_seconds"], 9.0)

    def test_detail_chain_bodies_scores(self):
        d = runindex.run_detail(self.run)
        chain = {c["key"]: c for c in d["chain"]}
        self.assertEqual([c["key"] for c in d["chain"]], ["sketch", "cover", "turnaround", "heldout", "mesh", "body"])
        self.assertEqual([i["path"] for i in chain["turnaround"]["images"]],
                         ["views/front.png", "views/back.png", "views/left.png", "views/right.png"])
        self.assertTrue(all(i["exists"] for c in d["chain"] for i in c["images"]))
        self.assertEqual(chain["turnaround"]["sheet"], "02-turnaround.png")
        bodies = d["bodies"]
        self.assertEqual([(b["name"], b["dir"]) for b in bodies], [("hybrid", "04-body"), ("carve", "04-body-carve")])
        self.assertEqual(bodies[0]["high"], "04-body/high.glb")
        self.assertEqual(bodies[0]["low"], "04-body/low.glb")
        self.assertFalse(bodies[0]["stage4_ok"])
        self.assertTrue(bodies[1]["stage4_ok"])
        self.assertEqual(len(bodies[0]["sections"]["sections"]), 15)
        self.assertEqual(bodies[0]["sections"]["budgets"]["triangles_high_max"], 24000)
        sc = d["scores"]
        self.assertEqual(sc["heldout_image"], "heldout/front-left-34.png")
        names = {b["name"]: b for b in sc["bodies"]}
        self.assertFalse(names["carve"]["views"]["front-left-34"]["seen"])
        self.assertEqual(names["carve"]["heldout_render"], "05-renders/carve/front-left-34.png")
        self.assertIsNone(names["../evil"]["heldout_render"])
        self.assertEqual(names["hybrid-variant"]["variant_of"], "hybrid")
        self.assertNotIn("back", names["hybrid"]["views"])

    def test_missing_files_are_not_run(self):
        d = runindex.run_detail(self.root / "partial")
        self.assertTrue(all(s["status"] == "not run" for s in d["stages"]))
        self.assertIsNone(d["scores"])
        self.assertFalse(any(i["exists"] for c in d["chain"] for i in c["images"]))
        self.assertEqual([b["name"] for b in d["bodies"]], ["body"])
        self.assertIsNone(d["bodies"][0]["low"])
        self.assertIsNone(d["bodies"][0]["sections"])

    def test_unreadable_json(self):
        (self.run / "stage-1.json").write_text("{not json")
        (self.run / "05-scores.json").write_text("[]")
        d = runindex.run_detail(self.run)
        self.assertEqual(d["stages"][1]["status"], "unreadable")
        self.assertIn("error", d["scores"])

    def test_normalise_check_rules(self):
        n = runindex.normalise_check
        # A sentence is a note, never an operator; the default comparison is said out loud.
        c = n({"name": "a", "value": 3, "threshold": 2, "rule": "inside the joint", "note": "measured", "pass": True})
        self.assertEqual((c["cmp"], c["limit"], c["cmp_assumed"]), (">=", ">= 2", True))
        self.assertEqual(c["notes"], ["measured", "rule: inside the joint", runindex.ASSUMED_CMP_NOTE])
        # A recorded operator wins over a sentence in the other field.
        c = n({"name": "b", "value": 1, "threshold": 2, "op": "<", "rule": "below two"})
        self.assertEqual((c["cmp"], c["cmp_assumed"], c["notes"]), ("<", False, ["rule: below two"]))
        # No threshold: no comparison and no assumption, whatever the rule says.
        c = n({"name": "c", "value": 1, "threshold": None, "rule": "count only"})
        self.assertEqual((c["cmp"], c["limit"], c["cmp_assumed"], c["notes"]), (None, "–", False, ["rule: count only"]))
        # Wrong types: pass and gate are only true/false when recorded as such; odd values become text.
        c = n({"name": 7, "value": {"x": 1}, "threshold": [1, 2], "op": 5, "pass": "yes", "gate": 0, "note": 3})
        self.assertEqual((c["name"], c["value"], c["pass"], c["gate"]), ("7", '{"x": 1}', False, True))
        self.assertEqual(c["limit"], ">= [1, 2]")

    def test_wrongly_typed_fields(self):
        write(self.run / "stage-1.json", {"name": 3, "ok": "yes", "checks": 5, "models": "muse", "tools": {"a": 1},
                                          "details": ["label"], "notes": {"n": 1}, "seconds": "1", "cost_basis": 2})
        write(self.run / "stage-2.json", {"checks": {"name": "x"}})
        write(self.run / "stage-3.json", {"checks": "abc"})
        write(self.run / "stage-4.json", ["not", "a", "record"])
        write(self.run / "04-body" / "sections.json", {"sections": 5, "budgets": [1], "height": "tall"})
        write(self.run / "04-body-carve" / "sections.json", {"sections": [1, {"name": "Head", "pivot": "up"}]})
        write(self.run / "05-scores.json", {"bodies": [5, {"name": ["x"], "glb": None, "views": [1, 2]},
                                                       {"name": "carve", "views": {"front": [1]}}]})
        d = runindex.run_detail(self.run)
        json.dumps(d, allow_nan=False)
        by = {s["stage"]: s for s in d["stages"]}
        self.assertEqual((by["1"]["status"], by["1"]["name"], by["1"]["checks"]), ("gate failed", "cover art", []))
        self.assertEqual((by["1"]["models"], by["1"]["tools"], by["1"]["label"]), ([], [], ""))
        self.assertIsNone(by["1"]["seconds"])
        self.assertEqual(by["2"]["checks"], [])
        self.assertEqual(by["3"]["checks"], [])
        self.assertEqual((by["4"]["status"], by["4"]["error"]), ("unreadable", "stage-4.json is not a JSON object"))
        bodies = {b["dir"]: b for b in d["bodies"]}
        self.assertEqual(bodies["04-body"]["sections"]["sections"], [])
        self.assertIsNone(bodies["04-body"]["sections"]["budgets"])
        self.assertIsNone(bodies["04-body-carve"]["sections"]["sections"][0]["pivot"])
        self.assertEqual([b["name"] for b in d["scores"]["bodies"]], ['["x"]', "carve"])
        self.assertEqual(d["scores"]["bodies"][1]["views"], {})
        self.assertEqual([r["id"] for r in runindex.run_list(self.root)], ["full", "partial"])

    def test_nan_and_infinity_become_null(self):
        (self.run / "stage-0.json").write_text(
            '{"name": "input", "ok": true, "seconds": NaN, "cost_usd": Infinity, "checks": ['
            '{"name": "a", "value": NaN, "threshold": -Infinity, "pass": true},'
            '{"name": "b", "value": 1e400, "threshold": 2, "op": ">=", "pass": false}]}')
        (self.run / "05-scores.json").write_text(
            '{"bodies": [{"name": "carve", "glb": "04-body-carve/high.glb", "triangles": 1e999,'
            ' "views": {"front": {"iou": NaN, "ssim": -Infinity, "colour_error": 3.5}}}]}')
        write(self.run / "stage-1.json", record(1, "cover", seconds=1e308))
        write(self.run / "stage-2.json", record(2, "turnaround", seconds=1e308))
        d = runindex.run_detail(self.run)
        text = json.dumps(d, allow_nan=False)  # raises if any NaN or infinity is left
        self.assertNotIn("NaN", text)
        s0 = d["stages"][0]
        self.assertEqual((s0["seconds"], s0["cost_usd"]), (None, None))
        self.assertEqual([(c["value"], c["threshold"], c["limit"]) for c in s0["checks"]],
                         [(None, None, "–"), (None, 2, ">= 2")])
        self.assertIsNone(d["total_seconds"])  # 1e308 + 1e308 overflows: no total rather than infinity
        self.assertAlmostEqual(d["total_cost_usd"], 0.05)  # stage 0's infinite cost was dropped, not summed
        front = d["scores"]["bodies"][0]["views"]["front"]
        self.assertEqual((front["iou"], front["ssim"], front["colour_error"]), (None, None, 3.5))
        self.assertIsNone(d["scores"]["bodies"][0]["triangles"])
        self.assertEqual(runindex.parse_json("[NaN, Infinity, -Infinity, 1e400, 1.5]"), [None, None, None, None, 1.5])

    def test_broken_links_and_unreadable_files(self):
        try:
            os.symlink(self.root / "nowhere.json", self.run / "stage-9.json")
            os.symlink(self.root / "nowhere.json", self.run / "stage-1.json.tmp")
            (self.run / "stage-1.json").unlink()
            os.symlink(self.root / "nowhere.json", self.run / "stage-1.json")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        runs = {r["id"]: r for r in runindex.run_list(self.root)}
        self.assertEqual(set(runs), {"full", "partial"})
        self.assertEqual(runs["full"]["stages"]["1"], "not run")
        self.assertGreater(runs["full"]["modified"], 0)
        d = runindex.run_detail(self.run)
        self.assertEqual(d["stages"][1]["status"], "not run")
        if os.name == "posix" and os.geteuid() != 0:
            (self.run / "stage-2.json").chmod(0)
            try:
                self.assertEqual(runindex.run_detail(self.run)["stages"][2]["status"], "unreadable")
                self.assertEqual({r["id"]: r for r in runindex.run_list(self.root)}["full"]["stages"]["2"],
                                 "unreadable")
            finally:
                (self.run / "stage-2.json").chmod(0o644)

    def test_sections_profile_and_budgets(self):
        base = json.loads((self.run / "04-body" / "sections.json").read_text())
        self.assertIsNone(runindex.run_detail(self.run)["bodies"][0]["sections"]["profile"])
        # The default "best" profile: no triangle budget, written as null or left out.
        for budgets in (None, "missing", {}, {"triangles_high_max": None}, [24000]):
            data = {**base, "profile": "best"}
            if budgets == "missing":
                data.pop("budgets")
            else:
                data["budgets"] = budgets
            write(self.run / "04-body" / "sections.json", data)
            sec = runindex.run_detail(self.run)["bodies"][0]["sections"]
            self.assertEqual((sec["profile"], sec["budgets"]), ("best", None), budgets)
            self.assertEqual(len(sec["sections"]), 15)
        # An opt-in budget profile keeps its budgets.
        write(self.run / "04-body" / "sections.json", {**base, "profile": "roblox-ingame"})
        sec = runindex.run_detail(self.run)["bodies"][0]["sections"]
        self.assertEqual((sec["profile"], sec["budgets"]["triangles_high_max"]), ("roblox-ingame", 24000))
        write(self.run / "04-body" / "sections.json", {**base, "profile": 7})
        self.assertIsNone(runindex.run_detail(self.run)["bodies"][0]["sections"]["profile"])

    def test_run_dir_rejects(self):
        self.assertIsNotNone(runindex.run_dir(self.root, "full"))
        for bad in ["", "..", ".hidden", "not-a-run", "missing", "full/04-body", "/etc"]:
            self.assertIsNone(runindex.run_dir(self.root, bad), bad)


class PathSafetyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "runs"
        self.root.mkdir()
        self.run = make_full_run(self.root)
        write(base / "outside.png", tiny_png())
        write(base / "outside" / "stage-0.json", record(0, "input"))
        write(self.run / "notes.html", b"<html></html>")
        write(self.run / "script.py", b"print(1)")
        write(self.run / ".secret.json", {})
        self.symlinks = True
        try:
            os.symlink(base / "outside.png", self.run / "link-out.png")
            os.symlink(base / "outside", self.root / "linked-run")
            os.symlink(self.run / "00-sketch.png", self.run / "link-in.png")
        except (OSError, NotImplementedError):
            self.symlinks = False

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowed(self):
        for rel, ctype in [("full/00-sketch.png", "image/png"), ("full/04-body/high.glb", "model/gltf-binary"),
                           ("full/stage-0.json", "application/json"), ("full/views/front.png", "image/png"),
                           ("full/00%2Dsketch.png", "image/png")]:
            got = safe_run_file(self.root, rel)
            self.assertIsNotNone(got, rel)
            self.assertEqual(got[1], ctype)

    def test_refused(self):
        bad = ["full/../outside.png", "full/%2e%2e/outside.png", "../outside.png", "/etc/passwd", "%2Fetc%2Fpasswd",
               "full//00-sketch.png", "full/./00-sketch.png", "full\\00-sketch.png", "full/notes.html",
               "full/script.py", "full/.secret.json", "full", "full/", "full/missing.png", "missing/00-sketch.png",
               "full/04-body", "full/00-sketch.png%00.json", "outside/stage-0.json"]
        for rel in bad:
            self.assertIsNone(safe_run_file(self.root, rel), rel)

    def test_symlinks(self):
        if not self.symlinks:
            self.skipTest("symlinks unavailable")
        self.assertIsNone(safe_run_file(self.root, "full/link-out.png"))
        self.assertIsNone(safe_run_file(self.root, "linked-run/stage-0.json"))
        self.assertIsNotNone(safe_run_file(self.root, "full/link-in.png"))

    def test_links_stay_inside_their_own_run(self):
        # A link to a sibling run, to a folder of the root that is not a run, or a run folder that is itself a link
        # to another run: all refused, though each target lies inside the root.
        write(self.root / "sibling" / "stage-0.json", record(0, "input"))
        write(self.root / "sibling" / "secret.png", tiny_png())
        write(self.root / "not-a-run" / "secret.png", tiny_png())
        try:
            os.symlink(self.root / "sibling" / "secret.png", self.run / "sib.png")
            os.symlink(self.root / "not-a-run" / "secret.png", self.run / "nar.png")
            os.symlink(self.root / "sibling", self.run / "sibdir")
            os.symlink(self.run, self.root / "alias")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        for rel in ["full/sib.png", "full/nar.png", "full/sibdir/secret.png", "alias/00-sketch.png"]:
            self.assertIsNone(safe_run_file(self.root, rel), rel)
        self.assertIsNone(runindex.run_dir(self.root, "alias"))
        self.assertIsNotNone(safe_run_file(self.root, "sibling/secret.png"))


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "runs"
        cls.root.mkdir()
        make_full_run(cls.root)
        write(cls.root.parent / "outside.png", tiny_png())
        # A run with wrongly typed fields and one whose record holds NaN.
        write(cls.root / "typed" / "stage-0.json", {"name": "input", "ok": True, "checks": 5, "models": 3})
        write(cls.root / "typed" / "05-scores.json", {"bodies": {"carve": 1}})
        (cls.root / "nan").mkdir()
        (cls.root / "nan" / "stage-0.json").write_text('{"ok": true, "seconds": NaN, "checks": '
                                                       '[{"name": "a", "value": Infinity, "threshold": 1}]}')
        # A run whose stage-9.json is a broken link.
        write(cls.root / "broken" / "stage-0.json", record(0, "input"))
        cls.symlinks = True
        try:
            os.symlink(cls.root / "nowhere.json", cls.root / "broken" / "stage-9.json")
        except (OSError, NotImplementedError):
            cls.symlinks = False
        cls.before = sorted((p.relative_to(cls.root).as_posix(), p.lstat().st_mtime_ns)
                            for p in cls.root.rglob("*"))
        cls.server = make_server(cls.root, port=0, quiet=True)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def get(self, path, method="GET", headers=None):
        req = urllib.request.Request(self.base + path, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def raw(self, path, method="GET", host="default"):
        """Send one request exactly as written (no client-side path cleaning) and return every byte of the reply.

        `host=None` sends no Host header at all."""
        port = self.server.server_address[1]
        lines = [f"{method} {path} HTTP/1.1"]
        if host == "default":
            host = f"127.0.0.1:{port}"
        if host is not None:
            lines.append(f"Host: {host}")
        lines += ["Connection: close", "", ""]
        return send_raw(port, "\r\n".join(lines).encode("latin-1"))

    def raw_response(self, path, method="GET", host="default"):
        """(status, headers, body) parsed from `raw`; the body is every byte after the headers."""
        return parse_raw(self, self.raw(path, method, host), method, f"{method} {path}")

    def strict_json(self, body):
        def refuse(name):
            raise ValueError(f"{name} is not JSON")
        return json.loads(body, parse_constant=refuse)

    def test_bound_to_loopback(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_idle_connections_time_out(self):
        from herogen.lab.server import LabHandler
        self.assertEqual(LabHandler.timeout, 30)  # a half-sent request cannot hold a thread for ever

    def test_page_and_static(self):
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/html"))
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertIn(b"importmap", body)
        for name, ctype in [("app.js", "text/javascript"), ("studio.css", "text/css"),
                            ("icons/box.svg", "image/svg+xml")]:
            status, headers, _ = self.get(f"/static/{name}")
            self.assertEqual(status, 200, name)
            self.assertTrue(headers["Content-Type"].startswith(ctype))
        for bad in ["/static/../server.py", "/static/%2e%2e/server.py", "/static/missing.js", "/nowhere"]:
            self.assertEqual(self.get(bad)[0], 404, bad)

    def test_api(self):
        status, headers, body = self.get("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual({r["id"] for r in json.loads(body)["runs"]}, {"r1/full", "r1/typed", "r1/nan", "r1/broken"})
        status, _, body = self.get("/api/runs/r1/full")
        self.assertEqual(status, 200)
        d = json.loads(body)
        self.assertEqual((d["id"], d["root"], d["name"]), ("r1/full", "r1", "full"))
        self.assertEqual(len(d["bodies"]), 2)
        for bad in ["/api/runs/..", "/api/runs/%2e%2e", "/api/runs/r1/missing", "/api/runs/r1/full/x", "/api/runs/full", "/api/runs/r2/full", "/api/runs/r1"]:
            self.assertEqual(self.get(bad)[0], 404, bad)

    def test_files_and_types(self):
        status, headers, body = self.get("/runs/r1/full/04-body/high.glb")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "model/gltf-binary")
        self.assertEqual(body[:4], b"glTF")
        status, headers, body = self.get("/runs/r1/full/00-sketch.png")
        self.assertEqual((status, headers["Content-Type"]), (200, "image/png"))
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        status, headers, body = self.get("/runs/r1/full/00-sketch.png", method="HEAD")
        self.assertEqual((status, body), (200, b""))
        for bad in ["/runs/r1/full/../full/00-sketch.png", "/runs/r1/full/%2e%2e/full/00-sketch.png",
                    "/runs/r1/full/stage-0.py", "/runs/%2Fetc%2Fpasswd"]:
            self.assertEqual(self.get(bad)[0], 404, bad)
        self.assertEqual(self.get("/runs/r1/full/00-sketch.png", method="POST")[0], 501)

    def test_foreign_host_refused(self):
        port = self.server.server_address[1]
        for host in ["evil.example", f"evil.example:{port}", "127.0.0.1", f"127.0.0.1:{port + 1}",
                     f"localhost.evil.example:{port}", f"127.0.0.1:{port}.evil.example", f"0.0.0.0:{port}"]:
            for path in ["/", "/api/runs", "/api/runs/r1/full", "/runs/r1/full/00-sketch.png", "/static/app.js"]:
                status, headers, body = self.raw_response(path, host=host)
                self.assertEqual(status, 421, (host, path))
                self.assertEqual(self.strict_json(body), {"error": "421 Misdirected Request"})
        self.assertEqual(self.raw_response("/api/runs", host=None)[0], 421)
        for host in [f"127.0.0.1:{port}", f"localhost:{port}", f"LOCALHOST:{port}"]:
            self.assertEqual(self.raw_response("/api/runs", host=host)[0], 200, host)
        status, _, body = self.raw_response("/api/runs", method="HEAD", host="evil.example")
        self.assertEqual((status, body), (421, b""))

    def test_security_headers(self):
        for path in ["/", "/api/runs", "/runs/r1/full/00-sketch.png", "/static/app.js", "/nowhere"]:
            headers = self.get(path)[1]
            self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin", path)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff", path)
        csp = self.get("/")[1]["Content-Security-Policy"]
        directives = dict(d.strip().split(" ", 1) for d in csp.split(";"))
        script_src = directives["script-src"].split()
        self.assertIn(CDN_PREFIX, script_src)
        self.assertNotIn("https://cdn.jsdelivr.net", script_src)
        self.assertNotIn("'unsafe-inline'", script_src)
        self.assertTrue(all(s in ("'self'", CDN_PREFIX) or s.startswith("'sha256-") for s in script_src), script_src)
        self.assertEqual(directives["connect-src"].split(), ["'self'", "blob:", "data:", CDN_PREFIX])

    def test_page_only_with_its_csp(self):
        # The page is HTML only at / and /index.html, both with the CSP; /static/ never serves HTML, however the
        # name is spelled (the disk may be case-insensitive, and the name is decoded before the lookup).
        for path in ["/", "/index.html"]:
            status, headers, _ = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertIn("script-src", headers["Content-Security-Policy"])
        for path in ["/static/index.html", "/static/%69ndex.html", "/static/index%2ehtml", "/static/index%2Ehtml",
                     "/static/Index.html", "/static/INDEX.HTML", "/static/index.HTML", "/static/%49ndex.html",
                     "/Index.html", "/%69ndex.html", "/index%2ehtml"]:
            status, headers, body = self.raw_response(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(headers["Content-Type"], "application/json", path)
            self.assertNotIn(b"<html", body.lower(), path)

    def test_absolute_form_must_name_this_server(self):
        port = self.server.server_address[1]
        for target in ["http://evil.example/api/runs", f"http://evil.example:{port}/api/runs",
                       f"https://127.0.0.1:{port}/api/runs", f"http://127.0.0.1:{port + 1}/api/runs",
                       "http:/api/runs", f"ftp://127.0.0.1:{port}/api/runs"]:
            status, _, body = self.raw_response(target)
            self.assertEqual(status, 421, target)
            self.assertNotIn(b'"runs"', body, target)
        for target in [f"http://127.0.0.1:{port}/api/runs", f"http://localhost:{port}/api/runs"]:
            self.assertEqual(self.raw_response(target)[0], 200, target)

    def test_stdlib_errors_are_ours(self):
        port = self.server.server_address[1]
        for data, want in malformed_requests(port):
            status, headers, body = parse_raw(self, send_raw(port, data), what=repr(data[:40]))
            self.assertEqual(status, want, data[:40])
            self.assertEqual(headers["Content-Type"], "application/json", data[:40])
            self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin", data[:40])
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff", data[:40])
            self.assertEqual(self.strict_json(body)["error"].split()[0], str(want))
            self.assertNotIn(b"s3cr3t", body)
        status, _, body = self.raw_response("/api/runs", method="OPTIONS", host="evil.example")
        self.assertEqual(status, 421)  # the Host is checked before "unsupported method"

    def test_bad_runs_do_not_break_the_index(self):
        status, _, body = self.raw_response("/api/runs")
        self.assertEqual(status, 200)
        runs = {r["name"]: r for r in self.strict_json(body)["runs"]}
        self.assertEqual(set(runs), {"full", "typed", "nan", "broken"})
        if self.symlinks:
            self.assertTrue(os.path.islink(self.root / "broken" / "stage-9.json"))
        self.assertEqual(runs["typed"]["stages"]["0"], "pass")
        for run_id in ["r1/full", "r1/typed", "r1/broken", "r1%2Ftyped"]:
            status, _, body = self.raw_response(f"/api/runs/{run_id}")
            self.assertEqual(status, 200, run_id)
            self.assertEqual(self.strict_json(body)["id"], run_id.replace("%2F", "/"))
        typed = self.strict_json(self.raw_response("/api/runs/r1/typed")[2])
        self.assertEqual(typed["stages"][0]["checks"], [])
        self.assertEqual(typed["scores"]["error"], "05-scores.json has no list of bodies")

    def test_nan_is_valid_json(self):
        status, headers, body = self.raw_response("/api/runs/r1/nan")
        self.assertEqual((status, headers["Content-Type"]), (200, "application/json"))
        d = self.strict_json(body)
        self.assertIsNone(d["stages"][0]["seconds"])
        self.assertIsNone(d["stages"][0]["checks"][0]["value"])
        self.assertIn("r1/nan", {r["id"] for r in self.strict_json(self.raw_response("/api/runs")[2])["runs"]})

    def test_traversal_over_http(self):
        # Sent byte for byte: backslashes, encoded slashes and dots, double encoding, NUL.
        for path in ["/runs/r1/full\\00-sketch.png", "/runs/r1/full%5C00-sketch.png", "/runs/r1/full%5c..%5c..%5coutside.png",
                     "/runs/r1/full/..%2F..%2Foutside.png", "/runs/r1/full/%2e%2e/%2e%2e/outside.png",
                     "/runs/r1/full/%2E%2E%2F%2E%2E%2Foutside.png", "/runs/%2e%2e/outside.png", "/runs/../outside.png", "/runs/r1/%2e%2e/outside.png",
                     "/runs/r1/full/..\\..\\outside.png", "/runs/r1/full/%252e%252e/%252e%252e/outside.png",
                     "/runs/r1/full/00-sketch.png%00", "/runs/r1/full/00-sketch.png%00.json", "/runs/%2Fetc%2Fpasswd",
                     "/runs/r1/full/%2Fetc%2Fpasswd", "/api/runs/..%2Ffull", "/api/runs/%2e%2e", "/api/runs/r1/full%5C..",
                     "/static/..%2Fserver.py", "/static/..%5Cserver.py", "/static/%2e%2e%2fserver.py",
                     "/static/..\\server.py", "/../herogen/lab/server.py", "/runs/r1/broken/stage-9.json"]:
            status, _, body = self.raw_response(path)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"PNG", body, path)
            self.assertNotIn(b"import", body, path)
        self.assertEqual(self.raw_response("/runs/r1/full/00-sketch.png")[0], 200)

    def test_head_errors_have_no_body(self):
        for path in ["/nowhere", "/api/runs/r1/missing", "/runs/r1/full/missing.png", "/static/missing.js",
                     "/runs/r1/full/..%2F..%2Foutside.png"]:
            status, headers, body = self.raw_response(path, method="HEAD")
            self.assertEqual((status, body), (404, b""), path)
            self.assertEqual(headers["Content-Type"], "application/json")
            get_status, _, get_body = self.raw_response(path)
            self.assertEqual(get_status, 404)
            self.assertEqual(self.strict_json(get_body), {"error": "404 Not Found"})
            self.assertEqual(int(headers["Content-Length"]), len(get_body))
        status, _, body = self.raw_response("/api/runs", method="HEAD")
        self.assertEqual((status, body), (200, b""))

    def test_unexpected_error_is_a_500(self):
        from herogen.lab import server as server_mod
        original = server_mod.Workspace.runs

        def boom(ws):
            raise RuntimeError("index failed")
        server_mod.Workspace.runs = boom
        try:
            status, headers, body = self.raw_response("/api/runs?token=secret")
            head_status, _, head_body = self.raw_response("/api/runs", method="HEAD")
        finally:
            server_mod.Workspace.runs = original
        self.assertEqual((status, headers["Content-Type"]), (500, "application/json"))
        self.assertEqual(self.strict_json(body), {"error": "500 Internal Server Error"})
        self.assertNotIn(b"index failed", body)
        self.assertEqual((head_status, head_body), (500, b""))
        self.assertEqual(self.get("/api/runs")[0], 200)  # the server is still up

    def test_zz_read_only(self):
        for p in ["/", "/api/runs", "/api/runs/r1/full", "/runs/r1/full/05-scores.json", "/runs/r1/full/04-body/low.glb"]:
            self.get(p)
        after = sorted((p.relative_to(self.root).as_posix(), p.lstat().st_mtime_ns) for p in self.root.rglob("*"))
        self.assertEqual(after, self.before)


def send_raw(port: int, data: bytes) -> bytes:
    """Send `data` byte for byte to the server on `port` and return every byte of its reply."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(data)
        chunks = []
        while chunk := s.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def parse_raw(test: unittest.TestCase, data: bytes, method: str = "GET", what: str = "request"):
    """(status, headers, body) of a raw reply; the body is every byte after the headers."""
    test.assertTrue(data, f"{what}: connection closed with no response")
    head, _, body = data.partition(b"\r\n\r\n")
    resp = http.client.HTTPResponse(_FakeSocket(head + b"\r\n\r\n"), method=method)
    resp.begin()
    return resp.status, resp.headers, body


def malformed_requests(port: int) -> list[tuple[bytes, int]]:
    """Request lines the stdlib refuses before any handler runs, with the status each must get."""
    host = f"Host: 127.0.0.1:{port}\r\n".encode()
    return [
        (b"GET /x?token=s3cr3t HTTP/9\r\n" + host + b"\r\n", 400),
        (b"GET /a b?token=s3cr3t HTTP/1.1\r\n" + host + b"\r\n", 400),
        (b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n", 505),
        (b"GET /" + b"a" * 70000 + b"?token=s3cr3t HTTP/1.1\r\n" + host + b"\r\n", 414),
        (b"PUT /api/runs?token=s3cr3t HTTP/1.1\r\n" + host + b"\r\n", 501),
        (b"DELETE /runs/full/00-sketch.png?token=s3cr3t HTTP/1.1\r\n" + host + b"\r\n", 501),
        (b"PUT /api/runs?token=s3cr3t HTTP/1.1\r\nHost: evil.example\r\n\r\n", 421),
    ]


class _FakeSocket:
    """Lets http.client parse a status line and headers already read from a raw socket."""

    def __init__(self, data: bytes):
        self._file = io.BytesIO(data)

    def makefile(self, *args, **kwargs):
        return self._file


class ErrorLogTest(unittest.TestCase):
    """A 500 is logged to stderr with the method and path only: no query string, no headers (access line too)."""

    def test_log_has_no_query_or_headers(self):
        from herogen.lab import server as server_mod
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        srv = make_server(Path(tmp.name), port=0, quiet=False)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        original = server_mod.Workspace.runs
        captured = io.StringIO()
        real_stderr = sys.stderr

        def boom(ws):
            raise RuntimeError("index failed")
        server_mod.Workspace.runs = boom
        sys.stderr = captured
        try:
            port = srv.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/api/runs?token=s3cr3t", headers={"Authorization": "Bearer h3ad3r"})
            resp = conn.getresponse()
            self.assertEqual(resp.status, 500)
            resp.read()
            conn.close()
        finally:
            sys.stderr = real_stderr
            server_mod.Workspace.runs = original
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=5)
        log = captured.getvalue()
        self.assertIn("lab: GET /api/runs failed", log)
        self.assertIn("RuntimeError: index failed", log)
        self.assertIn('"GET /api/runs" 500', log)  # the access line, without its query string
        for secret in ("s3cr3t", "token=", "h3ad3r", "Authorization"):
            self.assertNotIn(secret, log)

    def test_malformed_requests_logged_without_request_data(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        srv = make_server(Path(tmp.name), port=0, quiet=False)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        captured = io.StringIO()
        real_stderr = sys.stderr
        sys.stderr = captured
        try:
            port = srv.server_address[1]
            for data, want in malformed_requests(port):
                status = parse_raw(self, send_raw(port, data), what=repr(data[:40]))[0]
                self.assertEqual(status, want, data[:40])
        finally:
            sys.stderr = real_stderr
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=5)
        log = captured.getvalue()
        for want in ("400", "414", "421", "501", "505"):
            self.assertIn(f" {want} ", log)
        for bad in ("s3cr3t", "token", "Traceback", "AttributeError", "aaaaaaaa", "evil.example"):
            self.assertNotIn(bad, log)


class PageReferencesTest(unittest.TestCase):
    """The page loads nothing from the network except the pinned three.js modules."""

    URL = re.compile(r"""(?:https?:)?//[^\s"'`)<>]+""")

    def scripts(self):
        return {p.name: p.read_text() for p in STATIC_DIR.glob("*.js")}

    def test_html_references(self):
        html = (STATIC_DIR / "index.html").read_text()
        importmap = re.search(r"<script type=\"importmap\">(.*?)</script>", html, re.S)
        self.assertIsNotNone(importmap)
        imports = json.loads(importmap.group(1))["imports"]
        self.assertEqual(set(imports), {"three", "three/addons/"})
        for url in imports.values():
            self.assertTrue(url.startswith(CDN_PREFIX), url)
        # Every module the page loads from the CDN is pinned by its SHA-384 (import-map integrity).
        integrity = json.loads(importmap.group(1))["integrity"]
        js = "\n".join(self.scripts().values())
        needed = {imports["three"]} | {imports["three/addons/"] + spec[len("three/addons/"):] for spec in
                                        re.findall(r"""from\s+['"](three/addons/[^'"]+)['"]""", js)}
        self.assertGreater(len(needed), 2)
        self.assertTrue(needed <= set(integrity), needed - set(integrity))
        for url, digest in integrity.items():
            self.assertTrue(url.startswith(CDN_PREFIX), url)
            self.assertRegex(digest, r"^sha384-[A-Za-z0-9+/]{64}$", url)
        # The CSP hash covers the import map exactly as served, integrity included.
        self.assertIn("'sha256-", content_security_policy(html))
        for url in self.URL.findall(html):
            self.assertTrue(url.startswith(CDN_PREFIX) or url.startswith("http://www.w3.org/"), url)
        for attr in re.findall(r"""(?:src|href)=["']([^"']+)["']""", html):
            self.assertFalse(re.match(r"^[a-z]+:|^//", attr), attr)
            if not attr.startswith("#"):
                self.assertTrue((STATIC_DIR.parent / attr).is_file(), attr)  # every referenced file ships
        self.assertNotRegex(html, r"<style|style=", "inline styles would need 'unsafe-inline' in the CSP")

    def test_scripts_and_styles(self):
        scripts = self.scripts()
        self.assertIn("app.js", scripts)
        for name, js in scripts.items():
            for spec in re.findall(r"""^\s*import\s+(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]""", js, re.M):
                ok = spec == "three" or spec.startswith("three/addons/") or spec.startswith("./")
                self.assertTrue(ok, (name, spec))
                if spec.startswith("./"):
                    self.assertTrue((STATIC_DIR / spec[2:]).is_file(), (name, spec))
            self.assertFalse(re.search(r"\bimport\s*\(", js), f"{name}: no dynamic imports")
            # Text from a run folder goes on the page as text only.
            self.assertNotRegex(js, r"\.innerHTML|\.outerHTML|insertAdjacentHTML|document\.write", name)
            self.assertNotRegex(js, r"""setAttribute\(\s*['"]style['"]""", name)  # the CSP refuses style attrs
        self.assertIn('href="static/fonts/fonts.css"', (STATIC_DIR / "index.html").read_text())
        for text in [*scripts.values(), *(p.read_text() for p in STATIC_DIR.rglob("*.css"))]:
            for url in self.URL.findall(text):
                self.assertTrue(url.startswith("http://www.w3.org/"), url)


if __name__ == "__main__":
    unittest.main()
