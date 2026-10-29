"""Stage 5 results rendering on hand-written stage records and synthetic images."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline.results import stage5_results  # noqa: E402
from herogen.pipeline.runfolder import RunFolder, runs_root  # noqa: E402
from unittest import mock  # noqa: E402


def _record(n, name, ok=True, outputs=(), checks=(), cost=0.0, seconds=1.0, notes="", details=None):
    return {
        "stage": n, "name": name, "ok": ok, "started": "2026-10-28T19:00:00-07:00", "seconds": seconds,
        "cost_usd": cost, "cost_basis": "estimate", "models": ["muse-image-1.0"] if cost else [],
        "outputs": list(outputs), "checks": list(checks), "notes": notes, "details": details or {},
    }


class ResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = runs_root(Path(self.tmp.name) / "runs")
        self.run = RunFolder(self.root, "r1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_partial_run(self):
        Image.new("RGB", (8, 8), "white").save(self.run.file("00-sketch.png"))
        Image.new("RGB", (8, 8), "blue").save(self.run.file("01-cover.png"))
        self.run.write_stage(_record(0, "input", outputs=["00-sketch.png"], details={"label": "stand-in sketch"},
                                     checks=[{"name": "min_side_px", "value": 8, "threshold": 256, "op": ">=",
                                              "pass": False, "gate": True}], ok=False))
        self.run.write_stage(_record(1, "cover_art", outputs=["01-cover.png", "missing.png"], cost=0.0125,
                                     seconds=2.5, notes="<b>not html</b>",
                                     checks=[{"name": "aspect_agreement", "value": 0.97, "threshold": 0.85,
                                              "pass": True}]))
        data = stage5_results(self.run)
        status = {s["stage"]: s["status"] for s in data["stages"]}
        self.assertEqual(status, {0: "gate failed", 1: "pass", 2: "not run", 3: "not run", 4: "not run"})
        self.assertAlmostEqual(data["total_cost_usd"], 0.0125)
        self.assertAlmostEqual(data["total_seconds"], 3.5)
        stage1 = data["stages"][1]
        self.assertEqual(stage1["images"], ["01-cover.png"])  # listed outputs that don't exist are skipped

        html = (self.run.path / "results.html").read_text()
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn("src='01-cover.png'", html)
        self.assertIn("&lt;b&gt;not html&lt;/b&gt;", html)
        self.assertIn("stand-in sketch", html)
        self.assertEqual(html.count("not run"), 3)
        self.assertIn("aspect_agreement", html)
        self.assertEqual(json.loads((self.run.path / "results.json").read_text())["run_id"], "r1")
        s5 = self.run.read_stage(5)
        self.assertEqual(s5["outputs"], ["results.html", "results.json"])

    def test_blender_stage_renders_are_shown(self):
        self.run.write_stage(_record(3, "base_mesh", outputs=["03-mesh.glb"]))
        Image.new("RGB", (8, 8)).save(self.run.file("03-renders/front.png"))
        data = stage5_results(self.run)
        self.assertEqual(data["stages"][3]["images"], ["03-renders/front.png"])

    def test_blender_check_rule_is_shown(self):
        self.run.write_stage(_record(4, "game_body", checks=[
            {"name": "triangles_high", "value": 23436, "threshold": 24000, "rule": "<=", "pass": True, "gate": True},
            {"name": "open_edges", "value": 0, "threshold": None, "pass": True, "gate": True}]))
        stage5_results(self.run)
        html = (self.run.path / "results.html").read_text()
        self.assertIn("&lt;= 24000", html)
        self.assertNotIn("&gt;= 24000", html)
        self.assertNotIn("&gt;= None", html)

    def test_runs_root_ignores_itself(self):
        self.assertEqual((self.root / ".gitignore").read_text(), "*\n")

    def test_bad_run_id(self):
        for bad in ("", "../x", ".hidden"):
            with self.assertRaises(ValueError):
                RunFolder(self.root, bad)


class WriteJsonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = RunFolder(runs_root(Path(self.tmp.name) / "runs"), "r1")
        self.run.write_json("stage-9.json", {"v": 1})

    def tearDown(self):
        self.tmp.cleanup()

    def _left(self):
        return sorted(p.name for p in self.run.path.iterdir() if p.name.endswith(".tmp"))

    def test_a_failed_write_leaves_the_old_file(self):
        with self.assertRaises(TypeError):  # not serialisable: fails before the file is touched
            self.run.write_json("stage-9.json", {"v": object()})
        with mock.patch("os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run.write_json("stage-9.json", {"v": 2})
        self.assertEqual(self.run.read_json("stage-9.json"), {"v": 1})
        self.assertEqual(self._left(), [])

    def test_the_new_file_is_synced_before_it_replaces_the_old(self):
        order = []
        real_fsync, real_replace = os.fsync, os.replace
        with mock.patch("os.fsync", side_effect=lambda fd: (order.append("fsync"), real_fsync(fd))), \
                mock.patch("os.replace", side_effect=lambda a, b: (order.append("replace"), real_replace(a, b))):
            self.run.write_json("stage-9.json", {"v": 3})
        self.assertEqual(order, ["fsync", "replace", "fsync"])  # the file, then the folder
        self.assertEqual(self.run.read_json("stage-9.json"), {"v": 3})


if __name__ == "__main__":
    unittest.main()
