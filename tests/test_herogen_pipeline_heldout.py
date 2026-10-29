"""Held-out 3/4 view (stage 2h) and the scores table on the results page, with a fake Model API."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import client as client_mod  # noqa: E402
from herogen.pipeline import heldout, imaging, stages  # noqa: E402
from herogen.pipeline.__main__ import run_pipeline  # noqa: E402
from herogen.pipeline.client import ModelClient  # noqa: E402
from herogen.pipeline.results import stage5_results  # noqa: E402
from herogen.pipeline.runfolder import RunFolder, runs_root  # noqa: E402
from test_herogen_pipeline_flow import DUMMY_KEY, FakeAPI, _img  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place  # noqa: E402

BUILD_FILES = ["00-sketch.png", "01-cover.png", "02-turnaround.png", "stage-0.json", "stage-1.json",
               "stage-2.json", "views/meta.json"] + [f"views/{v}.png" for v in stages.ALL_VIEWS] \
    + [f"masks/{v}.png" for v in stages.ALL_VIEWS]


class HeldoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.sketch = Path(self.tmp.name) / "sketch.png"
        paint(place(figure_mask(200, "front"), (300, 280), (50, 40)), colour=(90, 90, 90)).save(self.sketch)

    def tearDown(self):
        self.tmp.cleanup()

    def _live(self, api: FakeAPI, run_id: str, stage_list, cache: str | None = None) -> int:
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", lambda c, e, b: api.post(c, e, b)), \
                mock.patch.object(client_mod, "decode_sam_output", api.decode):
            return run_pipeline(self.sketch, run_id, stage_list, mode="live", runs=str(self.root), cache=cache)

    def _snapshot(self, run: Path) -> dict[str, tuple[bytes, int]]:
        return {rel: ((run / rel).read_bytes(), (run / rel).stat().st_mtime_ns) for rel in BUILD_FILES}

    def test_heldout_follows_stage2(self):
        api = FakeAPI()
        self.assertEqual(self._live(api, "r", [0, 1, 2, 5]), 0)
        run = self.root / "r"
        meta = json.loads((run / "views/meta.json").read_text())
        img, mask = _img(run / heldout.IMAGE_REL), _img(run / heldout.MASK_REL)
        self.assertEqual(img.size, (meta["image_size"],) * 2)
        self.assertEqual(mask.size, img.size)
        self.assertEqual(mask.mode, "L")
        rec = json.loads((run / "stage-2h.json").read_text())
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["stage"], "2h")
        self.assertEqual(rec["details"]["mask_method"], "sam-3.1")
        checks = {c["name"]: c for c in rec["checks"]}
        self.assertGreaterEqual(checks["heldout_height_agreement_front"]["value"], 0.99)
        self.assertAlmostEqual(checks["heldout_width_over_front_width"]["value"], 0.7, delta=0.02)
        self.assertGreater(checks["heldout_width_over_side_width"]["value"], 1.0)
        self.assertAlmostEqual(rec["cost_usd"], 0.01 + 0.0025)
        # The feet of the held-out figure sit on the front view's feet row.
        front = imaging.bbox(imaging.from_mask_image(_img(run / "masks/front.png")))
        self.assertEqual(imaging.bbox(imaging.from_mask_image(mask))[3], front[3])
        # Stage 2 itself is unchanged by the held-out view.
        s2 = json.loads((run / "stage-2.json").read_text())
        self.assertNotIn("heldout", json.dumps(s2["outputs"]))
        self.assertAlmostEqual(s2["cost_usd"], 0.01 + 0.0025 + 0.01 + 0.0025)
        res = json.loads((run / "results.json").read_text())
        self.assertEqual([s["stage"] for s in res["stages"]], [0, 1, 2, "2h", 3, 4])
        self.assertIn(heldout.IMAGE_REL, (run / "results.html").read_text())

    def test_heldout_only_on_existing_run(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        (run / "stage-2h.json").unlink()
        for rel in (heldout.IMAGE_REL, heldout.MASK_REL, heldout.RAW_REL):
            (run / rel).unlink()
        before = self._snapshot(run)
        api = FakeAPI()
        self.assertEqual(self._live(api, "r", ["2h"], cache=str(Path(self.tmp.name) / "fresh-cache")), 0)
        self.assertEqual(api.prompts, [heldout.HELDOUT_PROMPT])  # one image call, no sheet, cover or top
        self.assertEqual(api.calls, ["images/edits", "responses"])
        self.assertEqual(self._snapshot(run), before)
        self.assertTrue((run / heldout.IMAGE_REL).exists())

    def test_wrong_angle_retries_and_fails_alone(self):
        api = FakeAPI(heldout_squash=1.0)  # as wide as the front view: not turned
        self.assertEqual(self._live(api, "r", [0, 1, 2, 5]), 2)
        run = self.root / "r"
        self.assertEqual(api.prompts.count(heldout.HELDOUT_PROMPT), heldout.ATTEMPTS)
        rec = json.loads((run / "stage-2h.json").read_text())
        self.assertFalse(rec["ok"])
        self.assertEqual(len(rec["details"]["attempts"]), heldout.ATTEMPTS)
        failed = [c["name"] for c in rec["checks"] if c.get("gate", True) and not c["pass"]]
        self.assertEqual(failed, ["heldout_width_over_front_width"])
        self.assertTrue(json.loads((run / "stage-2.json").read_text())["ok"])
        res = json.loads((run / "results.json").read_text())
        self.assertEqual({s["stage"]: s["status"] for s in res["stages"]}["2h"], "gate failed")

    def test_needs_a_passed_stage2(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1]), 0)
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"]), 1)
        self.assertFalse((self.root / "r/stage-2h.json").exists())

    def test_oversized_figure_is_cut_not_shrunk(self):
        raw = paint(place(figure_mask(200, "front"), (300, 280), (50, 40)))
        mask = imaging.threshold_mask(raw)
        _, placed = heldout.place_on_canvas(raw, mask, 2.0, 300, 290)  # a 400 px figure on a 300 px canvas
        h = imaging.box_size(imaging.bbox(placed))[1]
        self.assertEqual(h, 290)  # cut at the top edge, so the height check sees 290 / 400


def _scores():
    def view(iou, seen, **kw):
        return {"iou": iou, "colour_error": 7.0, "ssim": 0.85, "seen": seen, **kw}

    return {"bodies": [
        {"name": "carve", "glb": "04-body/high.glb", "triangles": 23436, "atlas": 1024,
         "views": {"front": view(0.96, True), "back": view(0.95, True), "left": view(0.9, True),
                   "right": view(0.91, True),
                   "front-left-34": view(0.8, False, yaw_deg=41.0, scale=1.0)}},
        {"name": "hybrid", "glb": "04-body-hybrid/high.glb", "triangles": 20000, "atlas": 2048,
         "views": {"front": view(0.97, True), "front-left-34": view(0.85, False, yaw_deg=44.5, scale=0.98)}},
        {"name": "../evil", "glb": "x", "triangles": 1, "atlas": 1, "views": {}},
    ]}


class ScoresPageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = RunFolder(runs_root(Path(self.tmp.name) / "runs"), "r1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_scores_table(self):
        self.run.write_json("05-scores.json", _scores())
        Image.new("RGB", (8, 8), "white").save(self.run.file(heldout.IMAGE_REL))
        Image.new("RGB", (8, 8), "grey").save(self.run.file("05-renders/carve/front-left-34.png"))
        Image.new("RGB", (8, 8), "grey").save(self.run.file("05-renders/evil/front-left-34.png"))
        data = stage5_results(self.run)
        bodies = {b["name"]: b for b in data["scores"]["bodies"]}
        self.assertEqual(bodies["carve"]["heldout_render"], "05-renders/carve/front-left-34.png")
        self.assertIsNone(bodies["hybrid"]["heldout_render"])
        self.assertIsNone(bodies["../evil"]["heldout_render"])
        html = (self.run.path / "results.html").read_text()
        self.assertNotIn("<script", html.lower())
        self.assertEqual(html.count("<tr class='heldout'>"), 3)
        self.assertIn("yaw 41.0°", html)
        self.assertIn("triangles 23436", html)
        self.assertIn("atlas 2048 px", html)
        self.assertIn("src='heldout/front-left-34.png'", html)
        self.assertIn("src='05-renders/carve/front-left-34.png'", html)
        self.assertEqual(html.count("src='heldout/front-left-34.png'"), 1)  # only beside a body with a render
        self.assertNotIn("../evil/", html)
        self.assertIn("<h3>../evil</h3>", html)
        # Missing views of a body are shown as not scored, rows stay in order.
        hybrid = html.split("<h3>hybrid</h3>", 1)[1].split("</table>", 1)[0]
        self.assertEqual(hybrid.count("not scored"), 3)
        firsts = [row.split("<td>", 1)[1].split("</td>", 1)[0].split(" ")[0] for row in hybrid.split("<tr")[2:]]
        self.assertEqual(firsts, ["front", "back", "left", "right", "front-left-34"])
        self.assertIn("unseen", hybrid)

    def test_no_scores_file_no_section(self):
        data = stage5_results(self.run)
        self.assertNotIn("scores", data)
        self.assertNotIn("<h2>Scores</h2>", (self.run.path / "results.html").read_text())

    def test_bad_scores_file(self):
        self.run.file("05-scores.json").write_text("{not json")
        data = stage5_results(self.run)
        self.assertIn("could not be read", data["scores"]["error"])
        self.assertIn("could not be read", (self.run.path / "results.html").read_text())


if __name__ == "__main__":
    unittest.main()
