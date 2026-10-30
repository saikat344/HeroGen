"""Held-out 3/4 views (stage 2h) and the scores table on the results page, with a fake Model API."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import client as client_mod  # noqa: E402
from herogen.pipeline import heldout, imaging, stages  # noqa: E402
from herogen.pipeline.__main__ import main, run_pipeline  # noqa: E402
from herogen.pipeline.client import ApiError, ModelClient, ReplayMiss  # noqa: E402
from herogen.pipeline.results import stage5_results  # noqa: E402
from herogen.pipeline.runfolder import RunFolder, runs_root  # noqa: E402
from test_herogen_pipeline_flow import DUMMY_KEY, FakeAPI, _decode_data_url, _img  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place  # noqa: E402
from test_herogen_pipeline_heldout_synth import figure, turnaround  # noqa: E402

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

    def _live(self, api: FakeAPI, run_id: str, stage_list, cache: str | None = None, views=None) -> int:
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", lambda c, e, b: api.post(c, e, b)), \
                mock.patch.object(client_mod, "decode_sam_output", api.decode):
            return run_pipeline(self.sketch, run_id, stage_list, mode="live", runs=str(self.root), cache=cache,
                                views=views)

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

    def test_four_views_follow_stage2(self):
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
        self.assertEqual(rec["details"]["build_azimuths_deg"], [45.0 * i for i in range(8)])
        self.assertEqual(rec["details"]["azimuth_clearance_deg"], 10.0)
        self.assertEqual(rec["details"]["fit_error_deg"], [-2.0, 1.0])
        self.assertEqual(rec["details"]["checks_version"], heldout.checks_version())
        self.assertEqual(heldout.accepted_views(RunFolder(self.root, "r")),
                         {v.name: heldout.view_paths(v)["image"] for v in heldout.VIEWS})
        self.assertEqual(rec["details"]["decisions"], [])
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
            self.assertEqual(d["image_sha256"], heldout.file_sha256(run / files["image"]))
            self.assertEqual(d["stage2_identity"], heldout.stage2_identity(RunFolder(self.root, "r")))
            measured0 = d["attempts"][0]["measured"]
            fit0 = measured0["fit"][f"facing_{view.side}"]
            self.assertEqual(measured0["found_yaw_range_deg"], sorted([fit0["yaw_deg"], fit0["plain_yaw_deg"]]))
            lo, hi = measured0["true_yaw_range_deg"]
            self.assertLess(lo, hi)
            self.assertEqual(d["mask_method"], "sam-3.1")
            first = d["attempts"][0]
            self.assertEqual(first["requested_yaw_deg"], view.yaw_deg)  # the first request is the target
            self.assertEqual(first["pose_reference"],
                             "views/front.png" if abs(view.yaw_deg) < 50 else f"views/{view.side}.png")
            for rel in d["kept_inputs"].values():
                self.assertTrue((run / rel).exists(), rel)
            yaw = d["found_yaw_deg"]
            self.assertEqual(-view.facing * yaw, abs(yaw))  # facing the view's way
            self.assertTrue(view.window_deg[0] <= abs(yaw) <= view.window_deg[1], yaw)  # turned as asked
            self.assertEqual(d["target_yaw_deg"], view.yaw_deg)
            measured = d["attempts"][0]["measured"]
            self.assertEqual(measured["facing_votes"]["landmarks"], 1)
            self.assertEqual(measured["facing_votes"]["toes"], 1)
            # The fake SAM answers "face" and "head" with the whole figure: the head cue can't vote.
            self.assertIsNone(measured["face_turn"])
            checks = {c["name"]: c for c in rec["checks"] if c["name"].startswith(view.name + ".")}
            self.assertGreaterEqual(checks[f"{view.name}.height_agreement_front"]["value"], 0.99)
            self.assertTrue(all(c["pass"] for c in checks.values()))
        self.assertIsNone(rec["details"]["profile_face_offset"])
        self.assertAlmostEqual(rec["cost_usd"], len(VIEWS) * COST_PER_GENERATION + COST_PROFILE_FACES)
        prompts = [rec["details"]["views"][v]["attempts"][0]["prompt"] for v in VIEWS]
        self.assertEqual(len(set(prompts)), len(VIEWS))
        self.assertFalse((run / "heldout/rejected").exists() or (run / "heldout/review").exists())
        self.assertFalse((run / heldout.STAGING_DIR).exists())
        # Stage 2 itself is unchanged by the held-out views.
        s2 = json.loads((run / "stage-2.json").read_text())
        self.assertNotIn("heldout", json.dumps(s2["outputs"]))
        self.assertAlmostEqual(s2["cost_usd"], 2 * (IMG + SAM))
        res = json.loads((run / "results.json").read_text())
        self.assertEqual([s["stage"] for s in res["stages"]], [0, 1, 2, "2h", 3, 4])
        html = (run / "results.html").read_text()
        for name in VIEWS:  # the page shows all four
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
        # One generation per view (no sheet, cover or top), each from the cover, a pose reference and a
        # turn guide, all the cover's size.
        self.assertEqual(len(api.prompts), len(VIEWS))
        cover = _img(run / "01-cover.png").size
        self.assertEqual(inputs, [[cover] * 3] * len(VIEWS))
        for view in heldout.VIEWS:
            # The pose reference and the guide stand at the set fill of the height, centred.
            for rel in self._rec()["details"]["views"][view.name]["kept_inputs"].values():
                if "inputs/" not in rel:
                    continue
                box = imaging.bbox(imaging.threshold_mask(_img(run / rel)))
                w, h = imaging.box_size(box)
                self.assertTrue(abs(h / cover[1] - heldout.POSE_REFERENCE_FILL) <= 0.02
                                or abs(w / cover[0] - heldout.POSE_REFERENCE_MAX_WIDTH) <= 0.01, (rel, w, h))
                self.assertLessEqual(abs((box[0] + box[2]) - cover[0]), 4, rel)
        self.assertEqual(api.calls, ["responses"] * 4 + (["images/edits"] + ["responses"] * 3) * len(VIEWS))
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
            self.assertEqual(len(rec["details"]["views"][view.name]["attempts"]), heldout.ATTEMPTS)
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
        rec = self._assert_rejected(FakeAPI(heldout_yaw=-30.0), "facing_cues_against")
        votes = rec["details"]["views"]["front-left-30"]["attempts"][0]["measured"]["facing_votes"]
        self.assertEqual(votes, {"landmarks": -1, "toes": -1, "head": 0})

    def test_under_turning_model_is_asked_for_more(self):
        # A model that turns three quarters of what it is asked: each attempt adds the measured shortfall.
        self.assertEqual(self._live(FakeAPI(heldout_gain=0.75), "r", [0, 1, 2]), 0)
        rec = self._rec()
        self.assertEqual(rec["details"]["accepted_views"], VIEWS)
        for view in heldout.VIEWS:
            tried = rec["details"]["views"][view.name]["attempts"]
            self.assertGreater(len(tried), 1)
            asked = [abs(a["requested_yaw_deg"]) for a in tried]
            self.assertEqual(asked[0], abs(view.yaw_deg))
            self.assertEqual(asked, sorted(asked))
            for a, b in zip(tried, tried[1:]):  # the next request is the last one plus the shortfall
                want = abs(a["requested_yaw_deg"]) + abs(view.yaw_deg) - abs(a["measured"]["found_yaw_deg"])
                self.assertAlmostEqual(abs(b["requested_yaw_deg"]), min(heldout.MAX_REQUEST_DEG, want))
            self.assertEqual(len({a["sample"] for a in tried}), len(tried))
            if abs(tried[-1]["requested_yaw_deg"]) >= heldout.PROFILE_REFERENCE_FROM_DEG:
                self.assertEqual(tried[-1]["pose_reference"], f"views/{view.side}.png")

    def test_stride_rejected(self):
        self._assert_rejected(FakeAPI(heldout_pose="stride"), "pose_landmark_rms")

    def test_arms_down_never_accepted(self):
        self.assertEqual(self._live(FakeAPI(heldout_pose="down"), "r", [0, 1, 2]), 2)
        rec = self._rec()
        self.assertEqual(rec["details"]["accepted_views"], [])
        for view in heldout.VIEWS:
            d = rec["details"]["views"][view.name]
            self.assertEqual(len(d["attempts"]), heldout.ATTEMPTS)
            failed = [c["name"] for c in d["attempts"][d["kept_attempt"]]["checks"] if not c["pass"]]
            self.assertIn("pose_landmark_rms_clear", failed)
            # The coarse fake figure's arms hang half hidden at 60 degrees (0.0199, just inside review); the
            # synthetic-humanoid tests reject every arms-down pose.
            if abs(view.yaw_deg) == 30:
                self.assertEqual(d["status"], "rejected")
                self.assertIn("pose_landmark_rms", failed)

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
        self.assertEqual(self._live(FakeAPI(heldout_yaw=-30.0), "r", ["2h"],
                                    cache=str(Path(self.tmp.name) / "other-cache")), 2)
        for view in heldout.VIEWS:
            for rel in heldout.view_paths(view).values():
                self.assertFalse((run / rel).exists(), rel)
            self.assertTrue((run / heldout.view_paths(view, "rejected")["image"]).exists())
        # A file an earlier version of the stage wrote and listed (an old view name) goes too.
        Image.new("RGB", (8, 8), "white").save(run / "heldout/front-left-34.png")
        rec = self._rec()
        rec["outputs"].append("heldout/front-left-34.png")
        (run / "stage-2h.json").write_text(json.dumps(rec))
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"]), 0)  # replays the first, accepted, generations
        self.assertFalse((run / "heldout/front-left-34.png").exists())
        for view in heldout.VIEWS:
            self.assertTrue((run / heldout.view_paths(view)["image"]).exists())
        self.assertFalse((run / "heldout/rejected").exists())

    def test_wide_figure_on_a_portrait_canvas(self):
        # A 2:3 portrait sketch with an A-pose figure 0.84 as wide as tall: the pose reference is held to the
        # canvas width, and the guide is drawn at the same scale, so the figure the model draws at the guide's
        # size maps back to the front view's height (it was about 11% too tall, failing the height check).
        paint(place(figure_mask(200, "front"), (300, 450), (66, 125)), colour=(90, 90, 90)).save(self.sketch)
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run, rec = self.root / "r", self._rec()
        self.assertEqual(rec["details"]["accepted_views"], VIEWS)
        cover = _img(run / "01-cover.png").size
        self.assertEqual(cover, (300, 450))
        for view in heldout.VIEWS:
            d = rec["details"]["views"][view.name]
            boxes = {k: imaging.bbox(imaging.threshold_mask(_img(run / d["kept_inputs"][k])))
                     for k in ("pose_reference", "turn_guide")}
            heights = {k: imaging.box_size(b)[1] for k, b in boxes.items()}
            self.assertLessEqual(abs(heights["pose_reference"] - heights["turn_guide"]), 2, (view.name, heights))
            for b in boxes.values():
                self.assertLessEqual(imaging.box_size(b)[0], heldout.POSE_REFERENCE_MAX_WIDTH * cover[0] + 2)
            checks = {c["name"]: c["value"] for c in rec["checks"]}
            self.assertGreaterEqual(checks[f"{view.name}.height_agreement_front"], 0.98, view.name)

    def _drop_cached(self, cache: Path, text: str) -> None:
        for p in sorted(cache.glob("*.json")):
            if json.loads(p.read_text())["request"].get("text") == text:
                p.unlink()
                return
        raise AssertionError(f"no cached {text!r} call")

    def test_partial_cache_is_an_error_not_a_lost_vote(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        self._drop_cached(self.root / "_cache", "face")  # the first one: a profile view's face
        self.assertEqual(self._replay("r2", [0, 1, 2]), 1)
        rec = self._rec("r2")
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["status"], "error")
        self.assertIn("ReplayMiss", rec["error"])

    def test_api_error_on_the_face_needs_review(self):
        class Broken:
            def segment(self, *_a):
                raise ApiError("responses failed: HTTP 500")

        out = heldout.measure_face(Broken(), None, np.ones((8, 8), bool))
        self.assertTrue(out["api_error"])
        self.assertIsNone(out["offset"])

        class Missing:
            def segment(self, *_a):
                raise ReplayMiss("no cached SAM response")

        with self.assertRaises(ReplayMiss):
            heldout.measure_face(Missing(), None, np.ones((8, 8), bool))

        class Bug:
            def segment(self, *_a):
                raise TypeError("a bug, not a measurement")

        with self.assertRaises(TypeError):
            heldout.measure_face(Bug(), None, np.ones((8, 8), bool))
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        from test_herogen_pipeline_heldout_synth import turned_mask  # noqa: PLC0415

        seen = turned_mask(figure("a"), 30)
        for face, profile_error in ((out, False), ({"offset": None}, True)):
            checks, _ = heldout.view_checks(heldout.VIEW_BY_NAME["front-left-30"], seen, model, 0.3, 0.2, face,
                                            None, profile_face_error=profile_error)
            self.assertEqual(heldout.view_status(checks), "needs review")
            self.assertEqual([c["name"] for c in checks if not c["pass"]], ["head_cue_measured"])

    def test_live_face_api_failure_needs_review_and_is_not_cached(self):
        api = FakeAPI()
        post, faces = api.post, []

        def flaky(c, endpoint, body):
            if endpoint == "responses" and body["input"][0]["content"][0]["text"] == "face":
                faces.append(1)
                if len(faces) == 3:  # the first held-out view's face call
                    raise ApiError("responses failed: HTTP 500")
            return post(c, endpoint, body)

        api.post = flaky
        self.assertEqual(self._live(api, "r", [0, 1, 2]), 0)  # the next generation is accepted
        rec = self._rec("r")
        first = rec["details"]["views"]["front-left-30"]["attempts"][0]
        self.assertTrue(first["measured"]["face"]["api_error"])
        self.assertIn("HTTP 500", first["measured"]["face"]["error"])
        self.assertEqual(first["status"], "needs review")  # never accepted without the cue
        for p in (self.root / "_cache").glob("*.json"):
            self.assertNotIn("api_error", json.loads(p.read_text()))
        # The failure is not a result, so a replay has nothing for that call and stops.
        self.assertEqual(self._replay("r2", [0, 1, 2]), 1)
        self.assertIn("ReplayMiss", self._rec("r2")["error"])

    def test_error_mid_stage_keeps_the_earlier_files(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        old = self._rec()
        self.assertTrue(old["ok"])
        before = {rel: (run / rel).read_bytes() for rel in old["outputs"]}
        api = FakeAPI(heldout_yaw=-30.0)  # a re-run that would reject every view, then fails part-way
        post, edits = api.post, []

        def flaky(c, endpoint, body):
            if endpoint == "images/edits":
                edits.append(1)
                if len(edits) == 3:
                    raise RuntimeError("connection reset")
            return post(c, endpoint, body)

        api.post = flaky
        self.assertEqual(self._live(api, "r", ["2h"], cache=str(Path(self.tmp.name) / "fresh")), 1)
        rec = self._rec()
        self.assertEqual((rec["ok"], rec["status"]), (False, "error"))
        self.assertIn("connection reset", rec["error"])
        # The paid calls of the failed run are in the record: 4 profile SAM calls, then front-left-30's two
        # generations before the third image call failed (the second drew the same image, so its 3 SAM calls
        # came from the cache).
        stop = rec["stopped_runs"][-1]
        live = [c for c in stop["calls"] if not c["cached"]]
        self.assertEqual(len(live), 4 + 4 + 1)
        self.assertAlmostEqual(stop["cost_usd"], COST_PROFILE_FACES + COST_PER_GENERATION + IMG)
        # The earlier run's calls and cost are kept, and the total counts both.
        self.assertEqual(rec["calls"], old["calls"])
        self.assertAlmostEqual(rec["run_cost_usd"], old["cost_usd"])
        self.assertAlmostEqual(rec["cost_usd"], old["cost_usd"] + stop["cost_usd"])
        self.assertEqual([r["status"] for r in rec["stopped_runs"]], ["error"])
        res = stage5_results(RunFolder(self.root, "r"))
        self.assertEqual({s["stage"]: s["status"] for s in res["stages"]}["2h"], "error")
        # The next run that finishes counts the stopped one too.
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"], cache=str(Path(self.tmp.name) / "fresh2")), 0)
        done = self._rec()
        self.assertAlmostEqual(done["cost_usd"], done["run_cost_usd"] + stop["cost_usd"])
        self.assertEqual([r["started"] for r in done["stopped_runs"]], [stop["started"]])
        # The earlier run's views and files are untouched, and still what a scorer reads.
        self.assertEqual(rec["details"], old["details"])
        self.assertEqual(rec["started"], old["started"])  # the record still describes the earlier run
        self.assertEqual({rel: (run / rel).read_bytes() for rel in old["outputs"]}, before)
        self.assertEqual(sorted(heldout.accepted_views(RunFolder(self.root, "r"))), sorted(VIEWS))
        self.assertFalse((run / heldout.STAGING_DIR).exists())

    def test_record_marked_running_and_files_kept_until_done(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        old = self._rec()
        files = sorted(p.name for p in (run / "heldout").rglob("*.png"))
        seen = {}

        def stop(*_a):
            seen["rec"] = self._rec()
            seen["files"] = sorted(p.name for p in (run / "heldout").rglob("*.png"))
            seen["accepted"] = heldout.accepted_views(RunFolder(self.root, "r"))
            raise KeyboardInterrupt

        with mock.patch.object(heldout, "_profile_face", stop), self.assertRaises(KeyboardInterrupt):
            heldout.stage2h_heldout(RunFolder(self.root, "r"), ModelClient(self.root / "_cache"))
        self.assertEqual(seen["rec"]["status"], "running")
        self.assertEqual((seen["rec"]["outputs"], seen["rec"]["details"]), (old["outputs"], old["details"]))
        self.assertEqual(seen["files"], files)
        self.assertEqual(seen["accepted"], {})  # nothing is scored while the stage runs
        self.assertEqual(sorted(heldout.accepted_views(RunFolder(self.root, "r"))), sorted(VIEWS))  # after: kept
        self.assertEqual(self._rec()["status"], "error")
        self.assertIn("KeyboardInterrupt", self._rec()["error"])
        self.assertEqual(sorted(p.name for p in (run / "heldout").rglob("*.png")), files)

    def test_time_limit(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        with self.assertRaises(heldout.StageTimeLimit):
            heldout.stage2h_heldout(RunFolder(self.root, "r"), ModelClient(self.root / "_cache"), time_limit_s=0.0)
        rec = self._rec()
        self.assertEqual(rec["status"], "error")
        self.assertIn("time limit", rec["error"])
        self.assertEqual(heldout.TIME_LIMIT_S, 1800.0)

    def test_listed_outputs_outside_heldout_are_never_deleted(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        rec = self._rec()
        rec["outputs"] += ["heldout/../stage-2.json", "heldout/inputs/../../01-cover.png", "/etc/hosts",
                           "views/front.png"]
        (run / "stage-2h.json").write_text(json.dumps(rec))
        before = self._snapshot(run)
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"]), 0)
        self.assertEqual(self._snapshot(run), before)
        self.assertEqual(heldout.heldout_rel("heldout/./a/../b.png"), "heldout/b.png")
        for bad in ("heldout/../x.png", "heldout/a/../../x", "/heldout/x", "heldout", "views/x.png", None, 3,
                    "heldout\\..\\x"):
            self.assertIsNone(heldout.heldout_rel(bad), bad)

    def _review(self, view: str, decision: str, note: str = "", by: str = "tester") -> int:
        return main(["review", "--runs", str(self.root), "--run-id", "r", "--view", view, f"--{decision}",
                     "--note", note, "--by", by])

    def _all_in_review(self) -> None:
        # The fake A-pose fits to about 0.011; with the pass limit below that, every view lands in review. The
        # limit stays patched for the whole test: it is part of the checks version a review must match.
        patch = mock.patch.object(heldout, "POSE_RMS_PASS", 0.005)
        patch.start()
        self.addCleanup(patch.stop)
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 2)

    def test_human_review(self):
        self._all_in_review()
        run = self.root / "r"
        self.assertEqual(self._rec()["details"]["needs_review_views"], VIEWS)
        self.assertEqual(self._review("front-left-30", "accept", "looks right"), 0)
        rec = self._rec()
        d = rec["details"]["views"]["front-left-30"]
        self.assertEqual((d["status"], d["machine_status"], d["files"]),
                         ("accepted", "needs review", heldout.view_paths(heldout.VIEWS[0])))
        for rel in d["files"].values():
            self.assertTrue((run / rel).exists(), rel)
        self.assertFalse((run / "heldout/review/front-left-30.png").exists())
        entry = rec["details"]["decisions"][-1]
        self.assertEqual({k: entry[k] for k in ("view", "decision", "by", "note", "from_status", "to_status")},
                         {"view": "front-left-30", "decision": "accepted", "by": "tester", "note": "looks right",
                          "from_status": "needs review", "to_status": "accepted"})
        self.assertEqual(len(entry["image_sha256"]), 64)
        self.assertTrue(entry["at"])
        self.assertNotIn("front-left-30.status_accepted", [c["name"] for c in rec["checks"]])
        self.assertFalse(rec["ok"])  # three views still wait
        self.assertIn("heldout/front-left-30.png", rec["outputs"])
        self.assertNotIn("heldout/review/front-left-30.png", rec["outputs"])

        self.assertEqual(self._review("front-left-60", "reject", "arms not in A-pose", by="Maintainer"), 0)
        rec = self._rec()
        d = rec["details"]["views"]["front-left-60"]
        self.assertEqual(d["files"], heldout.view_paths(heldout.VIEW_BY_NAME["front-left-60"], "rejected"))
        self.assertTrue((run / d["files"]["image"]).exists())
        failing = {c["name"]: c["note"] for c in rec["checks"] if not c["pass"] and c.get("gate", True)}
        self.assertIn("arms not in A-pose", failing["front-left-60.status_accepted"])
        self.assertEqual(rec["details"]["rejected_views"], ["front-left-60"])
        # A rejection can be reversed by a later decision, and every decision stays in the log.
        self.assertEqual(self._review("front-left-60", "accept", "second look"), 0)
        for v in ("front-right-30", "front-right-60"):
            self.assertEqual(self._review(v, "accept"), 0)
        rec = self._rec()
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["details"]["accepted_views"], VIEWS)
        self.assertEqual([e["decision"] for e in rec["details"]["decisions"]],
                         ["accepted", "rejected", "accepted", "accepted", "accepted"])
        self.assertFalse((run / "heldout/review").exists() or (run / "heldout/rejected").exists())
        # Scoring and the results page see only accepted views.
        self.assertEqual(self._live(FakeAPI(), "r", [5]), 0)
        self.assertIn("src='heldout/front-left-60.png'", (run / "results.html").read_text())

    def test_review_cannot_overrule_a_failed_gate(self):
        self.assertEqual(self._live(FakeAPI(heldout_yaw=0.0), "r", [0, 1, 2]), 2)
        before = self._rec()
        self.assertEqual(self._review("front-left-30", "accept"), 1)
        self.assertEqual(self._rec(), before)
        self.assertEqual(self._review("front-left-30", "reject", "front-on"), 0)  # rejecting it is recorded
        self.assertEqual(self._rec()["details"]["decisions"][-1]["machine_status"], "rejected")
        self.assertEqual(self._review("front-left-99", "reject"), 1)
        self.assertEqual(main(["review", "--runs", str(self.root), "--run-id", "nothing", "--view",
                               "front-left-30", "--reject", "--by", "x"]), 1)

    def test_remake_some_views_keeps_the_others(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        self._rec_started = self._rec()["started"]
        self.assertEqual(self._review("front-left-60", "reject", "arms not in A-pose"), 0)
        self.assertEqual(self._review("front-right-60", "reject", "arms not in A-pose"), 0)
        kept = {rel: (run / rel).read_bytes() for v in heldout.VIEWS[:2] for rel in heldout.view_paths(v).values()}
        rejected = {rel: (run / rel).read_bytes() for v in heldout.VIEWS[2:]
                    for rel in heldout.view_paths(v, "rejected").values()}
        # A remake that fails part-way deletes nothing: the human-rejected files stay where they are.
        broken = FakeAPI()
        broken.post = lambda c, e, b: (_ for _ in ()).throw(RuntimeError("down")) if e == "images/edits" \
            else FakeAPI.post(broken, c, e, b)
        self.assertEqual(self._live(broken, "r", ["2h"], cache=str(Path(self.tmp.name) / "broken"),
                                    views=["front-left-60", "front-right-60"]), 1)
        self.assertEqual({rel: (run / rel).read_bytes() for rel in rejected}, rejected)
        self.assertEqual(self._rec()["details"]["rejected_views"], ["front-left-60", "front-right-60"])
        api = FakeAPI()
        self.assertEqual(self._live(api, "r", ["2h"], cache=str(Path(self.tmp.name) / "fresh"),
                                    views=["front-left-60", "front-right-60"]), 0)
        self.assertEqual(api.calls, ["responses"] * 4 + (["images/edits"] + ["responses"] * 3) * 2)
        rec = self._rec()
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["details"]["accepted_views"], VIEWS)
        self.assertEqual(rec["details"]["made_this_run"], ["front-left-60", "front-right-60"])
        first = self._rec_started
        self.assertEqual(rec["details"]["kept_from_earlier_run"], {"front-left-30": first, "front-right-30": first})
        self.assertEqual([e["decision"] for e in rec["details"]["decisions"]], ["rejected", "rejected"])
        # The rejected images carried a person's decision: archived, not deleted, and the log says where.
        for e in rec["details"]["decisions"]:
            self.assertTrue(e["archived_to"].startswith(heldout.ARCHIVE_DIR + "/"), e)
            self.assertEqual(heldout.file_sha256(run / e["archived_to"]), e["image_sha256"])
        self.assertIsNone(rec["details"]["views"]["front-left-60"]["review"])
        self.assertEqual({rel: (run / rel).read_bytes() for rel in kept}, kept)
        self.assertFalse((run / "heldout/rejected").exists())
        names = {c["name"].split(".", 1)[0] for c in rec["checks"]}
        self.assertEqual(names, set(VIEWS))
        for rel in rec["outputs"]:
            self.assertTrue((run / rel).exists(), rel)
        self.assertEqual(len(rec["outputs"]), 4 * 5)
        # A second remake keeps the first run's start for the views it keeps.
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"], views=["front-right-60"]), 0)
        self.assertEqual(self._rec()["details"]["kept_from_earlier_run"]["front-left-30"], first)
        # Only views the stage knows, and only with an earlier record of the others.
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"], views=["front-left-45"]), 1)
        self._fresh_heldout(run)
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"], views=["front-left-60"]), 1)

    def test_views_from_an_earlier_turnaround_are_not_kept(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        # Stages 0-2 again from another sketch: the kept views would belong to the old turnaround.
        paint(place(figure_mask(240, "front"), (300, 320), (40, 30)), colour=(40, 90, 40)).save(self.sketch)
        cache = str(Path(self.tmp.name) / "c2")
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2], cache=cache, views=["front-left-30"]), 1)
        rec = self._rec()
        self.assertFalse(rec["ok"])
        self.assertIn("earlier turnaround", rec["error"])
        self.assertEqual(heldout.accepted_views(RunFolder(self.root, "r")), {})
        # Remaking every view is the way on.
        self.assertEqual(self._live(FakeAPI(), "r", ["2h"], cache=cache), 0)
        ident = heldout.stage2_identity(RunFolder(self.root, "r"))
        self.assertEqual({d["stage2_identity"] for d in self._rec()["details"]["views"].values()}, {ident})

    def test_kept_views_are_judged_again_under_new_checks(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        old_version = self._rec()["details"]["checks_version"]
        with mock.patch.object(heldout, "POSE_RMS_PASS", 0.005):  # stricter checks since the views were made
            self.assertNotEqual(heldout.checks_version(), old_version)
            self.assertEqual(self._live(FakeAPI(), "r", ["2h"], views=["front-left-30"]), 2)
            rec = self._rec()
            for name in VIEWS[1:]:  # kept, judged again: now waiting for review, and moved there
                d = rec["details"]["views"][name]
                self.assertEqual((d["status"], d["checks_version"]), ("needs review", heldout.checks_version()))
                self.assertEqual(d["rechecked"]["from_checks_version"], old_version)
                self.assertEqual(d["files"]["image"], f"heldout/review/{name}.png")
                self.assertTrue((run / d["files"]["image"]).exists())
                self.assertFalse((run / f"heldout/{name}.png").exists())
            self.assertEqual(heldout.accepted_views(RunFolder(self.root, "r")), {})

    def test_review_needs_the_current_checks_and_image(self):
        self._all_in_review()
        run = self.root / "r"
        with mock.patch.object(heldout, "POSE_RMS_FAIL", 0.03):
            self.assertEqual(self._review("front-left-30", "accept"), 1)  # judged under other checks
            self.assertEqual(main(["recheck", "--runs", str(self.root), "--run-id", "r"]), 0)
            self.assertEqual(self._review("front-left-30", "accept"), 0)
        # An image changed since it was checked can't be decided on.
        Image.new("RGB", (8, 8), "red").save(run / "heldout/review/front-right-30.png")
        self.assertEqual(main(["recheck", "--runs", str(self.root), "--run-id", "r"]), 0)
        before = self._rec()
        self.assertEqual(self._review("front-right-30", "accept"), 1)
        self.assertEqual(self._rec(), before)

    def test_a_stricter_recheck_overrules_an_acceptance(self):
        self._all_in_review()
        run = self.root / "r"
        self.assertEqual(self._review("front-left-30", "accept", "looks right"), 0)
        with mock.patch.object(heldout, "POSE_RMS_FAIL", 0.005):  # a gate the view now fails
            self.assertEqual(main(["recheck", "--runs", str(self.root), "--run-id", "r"]), 0)
            d = self._rec()["details"]["views"]["front-left-30"]
            self.assertEqual((d["status"], d["machine_status"]), ("rejected", "rejected"))
            self.assertEqual(d["review"]["decision"], "accepted")  # the decision stays on record
            self.assertTrue((run / "heldout/rejected/front-left-30.png").exists())
            self.assertFalse((run / "heldout/front-left-30.png").exists())
            self.assertNotIn("front-left-30", heldout.accepted_views(RunFolder(self.root, "r")))
            stale = [a for i, a in enumerate(d["attempts"]) if i != d["kept_attempt"]]
            self.assertTrue(stale and all(a["stale"] for a in stale))

    def test_move_files_rolls_back(self):
        run = RunFolder(self.root, "m")
        for rel in ("heldout/review/a.png", "heldout/review/b.png"):
            Image.new("RGB", (4, 4), "white").save(run.file(rel))
        real, calls = os.replace, []

        def flaky(a, b):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")
            return real(a, b)

        with mock.patch("os.replace", side_effect=flaky), self.assertRaises(OSError):
            heldout.move_files(run, {"heldout/review/a.png": "heldout/a.png", "heldout/review/b.png": "heldout/b.png"})
        for rel in ("heldout/review/a.png", "heldout/review/b.png"):
            self.assertTrue(run.exists(rel), rel)
        self.assertFalse(run.exists("heldout/a.png") or run.exists("heldout/b.png"))
        with self.assertRaises(ValueError):
            heldout.move_files(run, {"heldout/review/a.png": "heldout/../a.png"})

    def test_review_reports_file_errors_in_one_line(self):
        err = io.StringIO()
        with mock.patch("herogen.pipeline.__main__.review_view",
                        side_effect=PermissionError(13, "Permission denied", "heldout/x.png")), \
                contextlib.redirect_stderr(err):
            self.assertEqual(self._review("front-left-30", "accept"), 1)
        self.assertEqual(err.getvalue().count("\n"), 1)
        self.assertIn("Permission denied", err.getvalue())

    def test_recheck_keeps_a_stopped_runs_error(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        api = FakeAPI()
        api.post = lambda c, e, b: (_ for _ in ()).throw(RuntimeError("down")) if e == "images/edits" \
            else FakeAPI.post(api, c, e, b)
        self.assertEqual(self._live(api, "r", ["2h"], cache=str(Path(self.tmp.name) / "broken")), 1)
        stopped = self._rec()
        self.assertEqual(main(["recheck", "--runs", str(self.root), "--run-id", "r"]), 0)
        rec = self._rec()
        self.assertEqual((rec["status"], rec["error"], rec["ok"]), ("error", stopped["error"], False))
        self.assertEqual(rec["stopped_runs"], stopped["stopped_runs"])

    def test_review_moves_all_files_or_none(self):
        self._all_in_review()
        run = self.root / "r"
        before = self._rec()
        (run / "heldout/review/front-left-30-raw.png").unlink()
        self.assertEqual(self._review("front-left-30", "accept"), 1)
        self.assertEqual(self._rec(), before)
        for rel in ("heldout/review/front-left-30.png", "heldout/review/front-left-30-mask.png"):
            self.assertTrue((run / rel).exists(), rel)
        self.assertFalse((run / "heldout/front-left-30.png").exists())
        self.assertEqual(heldout.accepted_views(RunFolder(self.root, "r")), {})
        # A record that can't be written moves the files back.
        with mock.patch.object(RunFolder, "write_stage", side_effect=OSError("disk full")), \
                self.assertRaises(OSError):
            heldout.review_view(RunFolder(self.root, "r"), "front-right-30", "accepted", by="tester")
        self.assertTrue((run / "heldout/review/front-right-30.png").exists())
        self.assertFalse((run / "heldout/front-right-30.png").exists())
        self.assertEqual(self._rec(), before)

    def test_scoring_reads_accepted_views_from_the_record(self):
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = self.root / "r"
        Image.new("RGB", (8, 8), "red").save(run / "heldout/front-left-30.png")  # not the image checked
        Image.new("RGB", (8, 8), "red").save(run / "heldout/front-left-45.png")  # never recorded
        Image.new("RGB", (8, 8), "red").save(run / "heldout/front-left-45-mask.png")
        self.assertEqual(sorted(heldout.accepted_views(RunFolder(self.root, "r"))), sorted(VIEWS[1:]))

    def test_views_on_threshold_masks_are_never_accepted(self):
        # Records written by earlier versions could measure a view on a background-threshold mask.
        self.assertEqual(self._live(FakeAPI(), "r", [0, 1, 2]), 0)
        run = RunFolder(self.root, "r")
        rec = self._rec()
        d = rec["details"]["views"]["front-left-30"]
        d["mask_method"], d["mask_method_detail"] = "threshold", "threshold (sam under half the threshold area)"
        for a in d["attempts"]:
            a["mask_method"] = d["mask_method_detail"]
            a["checks"] = [c for c in a["checks"] if c["name"] != "figure_mask_sam_coverage"]
        run.write_json("stage-2h.json", rec)
        self.assertEqual(sorted(heldout.accepted_views(run)), sorted(VIEWS[1:]))
        self.assertEqual(stages.degraded_masks(rec), ["front-left-30: threshold (sam under half the threshold area)"])
        # Judged again, it is rejected: a measurement on a threshold mask is not one.
        new = heldout.recheck_heldout(run)
        view = new["details"]["views"]["front-left-30"]
        self.assertEqual(view["status"], "rejected")
        failed = [c["name"] for c in view["attempts"][view["kept_attempt"]]["checks"] if not c["pass"]]
        self.assertEqual(failed, ["figure_mask_sam_coverage"])

    def test_checks_version_follows_constants_and_code(self):
        base = heldout.checks_version()
        self.assertEqual(heldout.checks_version(), base)
        with mock.patch.object(heldout, "POSE_RMS_PASS", 0.01):
            self.assertNotEqual(heldout.checks_version(), base)
        with mock.patch.object(heldout, "LEG_BANDS", {**heldout.LEG_BANDS, "knee": (0.67, 0.74)}):
            self.assertNotEqual(heldout.checks_version(), base)

        def view_status(checks):  # same name, other code
            return "accepted"

        with mock.patch.object(heldout, "view_status", view_status):
            self.assertNotEqual(heldout.checks_version(), base)
        self.assertEqual(heldout.checks_version(), base)

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
        # A record written before per-view statuses: one view, accepted when the stage passed.
        self.run.write_stage({"stage": "2h", "name": "heldout", "ok": True, "seconds": 1.0, "cost_usd": 0.0,
                              "outputs": ["heldout/front-left-34.png"], "checks": [],
                              "details": {"view": "front-left-34"}})
        data = stage5_results(self.run)
        bodies = {b["name"]: b for b in data["scores"]["bodies"]}
        self.assertEqual(bodies["carve"]["heldout_renders"], {"front-left-34": "05-renders/carve/front-left-34.png"})
        self.assertEqual(bodies["hybrid"]["heldout_renders"], {})
        self.assertEqual(bodies["../evil"]["heldout_renders"], {})
        html = (self.run.path / "results.html").read_text()
        self.assertNotIn("<script", html.lower())
        self.assertEqual(html.count("<tr class='heldout'>"), 3)
        self.assertIn("yaw 41.0°", html)
        self.assertIn("triangles 23436", html)
        self.assertIn("atlas 2048 px", html)
        self.assertIn("src='heldout/front-left-34.png'", html)
        self.assertIn("src='05-renders/carve/front-left-34.png'", html)
        scores_html = html.split("<h2>Scores</h2>", 1)[1]
        self.assertEqual(scores_html.count("src='heldout/front-left-34.png'"), 1)  # only beside a rendered body
        self.assertNotIn("../evil/", html)
        self.assertIn("<h3>../evil</h3>", html)
        # Missing views of a body are shown as not scored, rows stay in order.
        hybrid = html.split("<h3>hybrid</h3>", 1)[1].split("</table>", 1)[0]
        self.assertEqual(hybrid.count("not scored"), 3)
        firsts = [row.split("<td>", 1)[1].split("</td>", 1)[0].split(" ")[0] for row in hybrid.split("<tr")[2:]]
        self.assertEqual(firsts, ["front", "back", "left", "right", "front-left-34"])
        self.assertIn("unseen", hybrid)

    def _record(self, views: dict) -> None:
        ident = heldout.stage2_identity(self.run)
        out = {}
        for name, (status, img, good_hash) in views.items():
            Image.new("RGB", (8, 8), "white").save(self.run.file(img))
            sha = heldout.file_sha256(self.run.path / img) if good_hash else "0" * 64
            out[name] = {"status": status, "files": {"image": img}, "image_sha256": sha, "stage2_identity": ident}
        self.run.write_stage({"stage": "2h", "name": "heldout", "ok": False, "seconds": 1.0, "cost_usd": 0.0,
                              "outputs": [], "checks": [], "details": {"views": out}})

    def test_every_accepted_heldout_view(self):
        # Accepted in the record with a matching image: shown. Accepted but the image changed, in review, or
        # only on disk (an image and a mask nobody recorded): not shown as held out.
        self._record({"front-left-30": ("accepted", "heldout/front-left-30.png", True),
                      "front-right-30": ("accepted", "heldout/front-right-30.png", True),
                      "front-left-60": ("accepted", "heldout/front-left-60.png", False),
                      "front-right-60": ("needs review", "heldout/review/front-right-60.png", True)})
        for rel in ("heldout/front-left-45.png", "heldout/front-left-45-mask.png"):
            Image.new("RGB", (8, 8), "white").save(self.run.file(rel))
        for v in ("front-left-30", "front-right-30", "front-left-60", "front-left-45"):
            Image.new("RGB", (8, 8), "grey").save(self.run.file(f"05-renders/carve/{v}.png"))

        def row(iou):
            return {"iou": iou, "colour_error": 7.0, "ssim": 0.85, "seen": False, "yaw_deg": 30.0}

        self.run.write_json("05-scores.json", {"bodies": [
            {"name": "carve", "glb": "04-body/high.glb", "triangles": 1, "atlas": 1024,
             "views": {"front": {"iou": 0.9}, "front-left-30": row(0.81), "front-right-30": row(0.82),
                       "front-left-60": row(0.7)}}]})
        data = stage5_results(self.run)
        self.assertEqual(data["scores"]["heldout_views"], ["front-left-30", "front-right-30"])
        self.assertEqual(data["scores"]["other_views"], ["front-left-60"])
        self.assertEqual(data["scores"]["heldout_images"], {"front-left-30": "heldout/front-left-30.png",
                                                           "front-right-30": "heldout/front-right-30.png"})
        self.assertEqual(list(data["scores"]["bodies"][0]["heldout_renders"]), ["front-left-30", "front-right-30"])
        html = (self.run.path / "results.html").read_text()
        self.assertEqual(html.count("<tr class='heldout'>"), 2)
        self.assertIn("front-left-60 (not an accepted held-out view)", html)
        for v in ("front-left-30", "front-right-30"):
            self.assertEqual(html.count(f"src='heldout/{v}.png'"), 1)
            self.assertIn(f"src='05-renders/carve/{v}.png'", html)
        for v in ("front-left-45", "front-right-60"):
            self.assertNotIn(v, html.split("<h2>Scores</h2>", 1)[1])

    def test_stale_heldout_record(self):
        self._record({"front-left-30": ("accepted", "heldout/front-left-30.png", True)})
        Image.new("RGB", (8, 8), "blue").save(self.run.file("01-cover.png"))  # stage 1 ran again
        data = stage5_results(self.run)
        self.assertEqual({s["stage"]: s["status"] for s in data["stages"]}["2h"], "stale: stage 2 ran again")
        self.assertEqual(heldout.accepted_views(self.run), {})

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
