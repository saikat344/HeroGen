"""Held-out 3/4 views (stage 2h) and the scores table on the results page, with a fake Model API."""

from __future__ import annotations

import json
import os
import shutil
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
from test_herogen_pipeline_flow import DUMMY_KEY, FakeAPI, _decode_data_url, _img  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place  # noqa: E402

BUILD_FILES = ["00-sketch.png", "01-cover.png", "02-turnaround.png", "stage-0.json", "stage-1.json",
               "stage-2.json", "views/meta.json"] + [f"views/{v}.png" for v in stages.ALL_VIEWS] \
    + [f"masks/{v}.png" for v in stages.ALL_VIEWS]


VIEWS = [v.name for v in heldout.VIEWS]
SAM = client_mod.PRICE_PER_IMAGE_USD[client_mod.SAM_MODEL]
IMG = client_mod.PRICE_PER_IMAGE_USD[client_mod.IMAGE_MODEL]
# Per generation: the image, then SAM for the figure, the face and the head. Once per stage: face and head on
# the two profile views.
COST_PER_GENERATION = IMG + 3 * SAM
COST_PROFILE_FACES = 4 * SAM


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

    def _replay(self, run_id: str, stage_list) -> int:
        def no_network(*_a, **_k):
            raise AssertionError("replay must not call the network")

        env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(ModelClient, "_post", no_network):
            return run_pipeline(self.sketch, run_id, stage_list, mode="replay", runs=str(self.root))

    def _snapshot(self, run: Path) -> dict[str, tuple[bytes, int]]:
        return {rel: ((run / rel).read_bytes(), (run / rel).stat().st_mtime_ns) for rel in BUILD_FILES}

    def _rec(self, run_id: str = "r") -> dict:
        return json.loads((self.root / run_id / "stage-2h.json").read_text())

    def _fresh_heldout(self, run: Path) -> None:
        """Forget an earlier stage 2h, keeping stages 0-2."""
        (run / "stage-2h.json").unlink()
        shutil.rmtree(run / "heldout")

    def test_both_views_follow_stage2(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2, 5]), 0)
        run = self.root / "r"
        meta = json.loads((run / "views/meta.json").read_text())
        front = imaging.bbox(imaging.from_mask_image(_img(run / "masks/front.png")))
        rec = self._rec()
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["stage"], "2h")
        self.assertEqual(rec["details"]["accepted_views"], VIEWS)
        self.assertEqual(rec["details"]["reference"], "generated")
        self.assertIn("consistency", rec["details"]["reference_note"])
        self.assertEqual(rec["details"]["build_azimuths_deg"], [0.0, 90.0, 180.0, 270.0])
        self.assertEqual(rec["details"]["azimuth_clearance_deg"], 10.0)
        for view in heldout.VIEWS:
            files = heldout.view_paths(view)
            self.assertEqual(files["image"], f"heldout/{view.name}.png")
            img, mask = _img(run / files["image"]), _img(run / files["mask"])
            self.assertEqual(img.size, (meta["image_size"],) * 2)
            self.assertEqual(mask.size, img.size)
            self.assertEqual(mask.mode, "L")
            self.assertTrue((run / files["raw"]).exists())
            # The feet of the held-out figure sit on the front view's feet row.
            self.assertEqual(imaging.bbox(imaging.from_mask_image(mask))[3], front[3])
            d = rec["details"]["views"][view.name]
            self.assertEqual((d["status"], d["files"], d["kept_attempt"]), ("accepted", files, 0))
            self.assertEqual(d["mask_method"], "sam-3.1")
            self.assertEqual(d["prompt"], view.prompt)
            yaw = d["found_yaw_deg"]
            self.assertTrue(30 <= -view.facing * yaw <= 55, yaw)  # the fake turns the figure 45 degrees
            measured = d["attempts"][0]["measured"]
            self.assertEqual(measured["facing_votes"]["landmarks"], 1)
            self.assertEqual(measured["facing_votes"]["toes"], 1)
            # The fake SAM answers "face" and "head" with the whole figure: the head cue can't vote.
            self.assertIsNone(measured["face_turn"])
            checks = {c["name"]: c for c in rec["checks"] if c["name"].startswith(view.name + ".")}
            self.assertGreaterEqual(checks[f"{view.name}.height_agreement_front"]["value"], 0.99)
            self.assertTrue(all(c["pass"] for c in checks.values()))
        self.assertIsNone(rec["details"]["profile_face_offset"])
        self.assertAlmostEqual(rec["cost_usd"], 2 * COST_PER_GENERATION + COST_PROFILE_FACES)
        self.assertNotEqual(heldout.VIEWS[0].prompt, heldout.VIEWS[1].prompt)
        self.assertFalse((run / "heldout/rejected").exists() or (run / "heldout/review").exists())
        # Stage 2 itself is unchanged by the held-out views.
        s2 = json.loads((run / "stage-2.json").read_text())
        self.assertNotIn("heldout", json.dumps(s2["outputs"]))
        self.assertAlmostEqual(s2["cost_usd"], 2 * (IMG + SAM))
        res = json.loads((run / "results.json").read_text())
        self.assertEqual([s["stage"] for s in res["stages"]], [0, 1, 2, "2h", 3, 4])
        html = (run / "results.html").read_text()
        for name in VIEWS:  # the page shows both, and the first stays where older pages read it
            self.assertIn(f"heldout/{name}.png", html)

    def test_heldout_only_on_existing_run(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        self._fresh_heldout(run)
        before = self._snapshot(run)
        api = FakeAPI()
        inputs = []
        post = api.post

        def record(c, endpoint, body):
            if endpoint == "images/edits":
                inputs.append([_decode_data_url(i["image_url"]).size for i in body["images"]])
            return post(c, endpoint, body)

        api.post = record
        self.assertEqual(self._live(api, "r", ["2h"], cache=str(Path(self.tmp.name) / "fresh-cache")), 0)
        self.assertEqual(api.prompts, [v.prompt for v in heldout.VIEWS])  # no sheet, cover or top
        # Each view is an edit of the cover with the turnaround's front view as the pose.
        cover = _img(run / "01-cover.png").size
        self.assertEqual(inputs, [[cover, cover], [cover, cover]])
        self.assertEqual(self._rec()["details"]["inputs"], ["01-cover.png", "heldout/pose-reference.png"])
        # The pose reference holds the front view's figure, centred, at the set fill of the height or width.
        ref = imaging.bbox(imaging.threshold_mask(_img(run / "heldout/pose-reference.png")))
        w, h = imaging.box_size(ref)
        self.assertTrue(abs(h / cover[1] - heldout.POSE_REFERENCE_FILL) <= 0.01
                        or abs(w / cover[0] - heldout.POSE_REFERENCE_MAX_WIDTH) <= 0.01, (w, h))
        self.assertLessEqual(abs((ref[0] + ref[2]) - cover[0]), 2)
        self.assertEqual(api.calls, ["responses"] * 4 + (["images/edits"] + ["responses"] * 3) * 2)
        self.assertEqual(self._snapshot(run), before)
        for view in heldout.VIEWS:
            self.assertTrue((run / heldout.view_paths(view)["image"]).exists())

    def test_replay_reproduces_the_record(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        self.assertEqual(self._replay("r2", [0, 1, 2]), 0)
        a, b = self._rec("r"), self._rec("r2")
        values = [[(c["name"], c["value"]) for c in r["checks"]] for r in (a, b)]
        self.assertEqual(values[0], values[1])
        self.assertEqual(b["cost_usd"], 0.0)
        for view in heldout.VIEWS:
            rel = heldout.view_paths(view)["image"]
            self.assertEqual((self.root / "r" / rel).read_bytes(), (self.root / "r2" / rel).read_bytes())

    def _assert_rejected(self, api: FakeAPI, failing: str) -> dict:
        self.assertEqual(self._live(api, "r", [0, 1, 2, 5]), 2)
        run = self.root / "r"
        rec = self._rec()
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["details"]["accepted_views"], [])
        for view in heldout.VIEWS:
            self.assertEqual(api.prompts.count(view.prompt), heldout.ATTEMPTS)
            d = rec["details"]["views"][view.name]
            self.assertEqual(d["status"], "rejected")
            self.assertEqual(len(d["attempts"]), heldout.ATTEMPTS)
            self.assertEqual(d["files"], heldout.view_paths(view, "rejected"))
            self.assertTrue((run / d["files"]["image"]).exists())
            self.assertFalse((run / heldout.view_paths(view)["image"]).exists())  # nothing unchecked to score
            self.assertIn(f"{view.name}.{failing}",
                          [c["name"] for c in rec["checks"] if c.get("gate", True) and not c["pass"]])
        self.assertTrue(json.loads((run / "stage-2.json").read_text())["ok"])
        res = json.loads((run / "results.json").read_text())
        self.assertEqual({s["stage"]: s["status"] for s in res["stages"]}["2h"], "gate failed")
        return rec

    def test_wrong_facing_retries_and_fails_alone(self):
        rec = self._assert_rejected(FakeAPI(heldout_yaw=-45.0), "facing_cues_against")
        votes = rec["details"]["views"]["front-left-34"]["attempts"][0]["measured"]["facing_votes"]
        self.assertEqual(votes, {"landmarks": -1, "toes": -1, "head": 0})

    def test_stride_rejected(self):
        self._assert_rejected(FakeAPI(heldout_pose="stride"), "pose_landmark_rms")

    def test_arms_down_rejected(self):
        self._assert_rejected(FakeAPI(heldout_pose="down"), "pose_landmark_rms")

    def test_front_on_view_is_not_held_out(self):
        self._assert_rejected(FakeAPI(heldout_yaw=0.0), "found_yaw_clear_of_build_azimuths_deg")

    def test_grey_zone_view_needs_review(self):
        # The fake A-pose fits to about 0.011; with the pass limit below that, every generation lands between
        # pass and fail (the synthetic tests cover which real poses do).
        with mock.patch.object(heldout, "POSE_RMS_PASS", 0.005):
            self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 2)
        run = self.root / "r"
        rec = self._rec()
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["details"]["needs_review_views"], VIEWS)
        for view in heldout.VIEWS:
            d = rec["details"]["views"][view.name]
            self.assertEqual(d["status"], "needs review")
            self.assertEqual(d["files"]["image"], f"heldout/review/{view.name}.png")
            self.assertTrue((run / d["files"]["image"]).exists())
            self.assertFalse((run / heldout.view_paths(view)["image"]).exists())
            failed = [c["name"] for c in rec["checks"] if c["name"].startswith(view.name) and not c["pass"]]
            self.assertEqual(failed, [f"{view.name}.pose_landmark_rms_clear", f"{view.name}.status_accepted"])
            self.assertEqual(len(d["attempts"]), heldout.ATTEMPTS)

    def test_rerun_replaces_stale_files(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        self.assertEqual(self._live(FakeAPI(heldout_yaw=-45.0), "r", ["2h"],
                                    cache=str(Path(self.tmp.name) / "other-cache")), 2)
        for view in heldout.VIEWS:
            for rel in heldout.view_paths(view).values():
                self.assertFalse((run / rel).exists(), rel)
            self.assertTrue((run / heldout.view_paths(view, "rejected")["image"]).exists())
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"]), 0)  # replays the first, accepted, generations
        for view in heldout.VIEWS:
            self.assertTrue((run / heldout.view_paths(view)["image"]).exists())
        self.assertFalse((run / "heldout/rejected").exists())

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
        Image.new("RGB", (8, 8), "white").save(self.run.file("heldout/front-left-34.png"))
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
