"""Lab server: run index, path safety, content types, read-only serving, and the page's external references."""

from __future__ import annotations

import json
import os
import re
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
from herogen.lab.server import CDN_PREFIX, STATIC_DIR, make_server, safe_run_file  # noqa: E402


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
        self.assertEqual(body["pivots_in_joint_region"]["limit"], ">= 15")
        self.assertIn("cut radius", body["pivots_in_joint_region"]["rule_text"])
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


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "runs"
        cls.root.mkdir()
        make_full_run(cls.root)
        cls.before = sorted((p.relative_to(cls.root).as_posix(), p.stat().st_mtime_ns)
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

    def get(self, path, method="GET"):
        req = urllib.request.Request(self.base + path, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_bound_to_loopback(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_page_and_static(self):
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/html"))
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertIn(b"importmap", body)
        for name, ctype in [("lab.js", "text/javascript"), ("lab.css", "text/css")]:
            status, headers, _ = self.get(f"/static/{name}")
            self.assertEqual(status, 200, name)
            self.assertTrue(headers["Content-Type"].startswith(ctype))
        for bad in ["/static/../server.py", "/static/%2e%2e/server.py", "/static/missing.js", "/nowhere"]:
            self.assertEqual(self.get(bad)[0], 404, bad)

    def test_api(self):
        status, headers, body = self.get("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual([r["id"] for r in json.loads(body)["runs"]], ["full"])
        status, _, body = self.get("/api/runs/full")
        self.assertEqual(status, 200)
        d = json.loads(body)
        self.assertEqual(d["id"], "full")
        self.assertEqual(len(d["bodies"]), 2)
        for bad in ["/api/runs/..", "/api/runs/%2e%2e", "/api/runs/missing", "/api/runs/full/x"]:
            self.assertEqual(self.get(bad)[0], 404, bad)

    def test_files_and_types(self):
        status, headers, body = self.get("/runs/full/04-body/high.glb")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "model/gltf-binary")
        self.assertEqual(body[:4], b"glTF")
        status, headers, body = self.get("/runs/full/00-sketch.png")
        self.assertEqual((status, headers["Content-Type"]), (200, "image/png"))
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        status, headers, body = self.get("/runs/full/00-sketch.png", method="HEAD")
        self.assertEqual((status, body), (200, b""))
        for bad in ["/runs/full/../full/00-sketch.png", "/runs/full/%2e%2e/full/00-sketch.png",
                    "/runs/full/stage-0.py", "/runs/%2Fetc%2Fpasswd"]:
            self.assertEqual(self.get(bad)[0], 404, bad)
        self.assertEqual(self.get("/runs/full/00-sketch.png", method="POST")[0], 501)

    def test_zz_read_only(self):
        for p in ["/", "/api/runs", "/api/runs/full", "/runs/full/05-scores.json", "/runs/full/04-body/low.glb"]:
            self.get(p)
        after = sorted((p.relative_to(self.root).as_posix(), p.stat().st_mtime_ns) for p in self.root.rglob("*"))
        self.assertEqual(after, self.before)


class PageReferencesTest(unittest.TestCase):
    """The page loads nothing from the network except the pinned three.js modules."""

    URL = re.compile(r"""(?:https?:)?//[^\s"'`)<>]+""")

    def test_html_references(self):
        html = (STATIC_DIR / "index.html").read_text()
        importmap = re.search(r"<script type=\"importmap\">(.*?)</script>", html, re.S)
        self.assertIsNotNone(importmap)
        imports = json.loads(importmap.group(1))["imports"]
        self.assertEqual(set(imports), {"three", "three/addons/"})
        for url in imports.values():
            self.assertTrue(url.startswith(CDN_PREFIX), url)
        for url in self.URL.findall(html):
            self.assertTrue(url.startswith(CDN_PREFIX) or url.startswith("http://www.w3.org/"), url)
        for attr in re.findall(r"""(?:src|href)=["']([^"']+)["']""", html):
            self.assertFalse(re.match(r"^[a-z]+:|^//", attr), attr)

    def test_scripts_and_styles(self):
        js = (STATIC_DIR / "lab.js").read_text()
        for spec in re.findall(r"""^\s*import\s+(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]""", js, re.M):
            self.assertTrue(spec == "three" or spec.startswith("three/addons/") or spec.startswith("./"), spec)
        self.assertFalse(re.search(r"\bimport\s*\(", js), "no dynamic imports")
        for text in (js, (STATIC_DIR / "lab.css").read_text()):
            for url in self.URL.findall(text):
                self.assertTrue(url.startswith("http://www.w3.org/"), url)


if __name__ == "__main__":
    unittest.main()
