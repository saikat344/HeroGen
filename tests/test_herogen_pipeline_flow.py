"""Stage flow 0-2 and 5 with a fake Model API: record once, then replay with the network blocked."""

from __future__ import annotations

import base64
import io
import json
import os
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
from herogen.pipeline import facesource, imaging, stages  # noqa: E402
from herogen.pipeline.__main__ import parse_stages, run_pipeline  # noqa: E402
from herogen.pipeline.client import ApiError, ClientError, ModelClient, ReplayMiss, SamInstance  # noqa: E402
from test_herogen_pipeline_heldout_synth import REQUEST as HELDOUT_REQUEST  # noqa: E402
from test_herogen_pipeline_heldout_synth import fake_heldout_image  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place, sheet  # noqa: E402
from fake_model_server import NOT_A_PHOTO, is_question, question_answer  # noqa: E402

DUMMY_KEY = "dummy-test-key-not-real-0123456789"


def _b64png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(path) -> Image.Image:
    with Image.open(path) as im:
        return im.copy()


def _decode_data_url(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


FEET_TEXTS = ("foot", "shoe", "boot")  # the held-out turn guide's foot prompts (`heldout.find_feet`)


class FakeAPI:
    """Stands in for Meta's Model API. Draws every image by code; SAM masks come from thresholding."""

    def __init__(self, sheet_order=("front", "right", "front", "left"), heldout_yaw: float | None = None,
                 heldout_pose: str = "a", heldout_gain: float = 1.0) -> None:
        self.sheet_order = sheet_order
        self.heldout_yaw = heldout_yaw  # held-out turn; None: as requested; negative: the wrong way
        self.heldout_gain = heldout_gain  # turn drawn / turn requested, when heldout_yaw is None
        self.heldout_pose = heldout_pose  # "a", "down" or "stride" (see test_herogen_pipeline_heldout_synth)
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.last_sam_image: Image.Image | None = None
        self.photo_answer = dict(NOT_A_PHOTO)  # Muse Spark's answer to the photo question
        self.last_sam_text = ""

    def post(self, _client, endpoint: str, body: dict) -> dict:
        assert os.environ.get("MODEL_API_KEY") == DUMMY_KEY
        if endpoint == "responses" and is_question(body):  # the photo question: a drawing unless told otherwise
            self.calls.append("question")
            return question_answer(self.photo_answer)
        self.calls.append(endpoint)
        if endpoint == "responses":
            self.last_sam_image = _decode_data_url(body["input"][0]["content"][1]["image_url"])
            self.last_sam_text = body["input"][0]["content"][0]["text"]
            return {"status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "fake"}]}]}
        prompt = body["prompt"]
        self.prompts.append(prompt)
        if endpoint == "images/generations":
            img = paint(place(figure_mask(200, "front"), (300, 280), (50, 40)), colour=(90, 90, 90))
        elif prompt == stages.COVER_PROMPT:
            src = _decode_data_url(body["images"][0]["image_url"])
            img = paint(imaging.threshold_mask(src), colour=(30, 80, 160), bg=(235, 235, 235))
        elif prompt == stages.TURNAROUND_PROMPT:
            img, _ = sheet(order=self.sheet_order)
        elif prompt == stages.TOP_PROMPT:
            img = paint(place(figure_mask(200, "front"), (300, 280), (50, 40)))  # not a real top view
        elif prompt in facesource.PROMPTS.values():
            # a faithful upscale of the head crop (stage 2f)
            img = _decode_data_url(body["images"][0]["image_url"]).resize((1024, 1024), Image.LANCZOS)
        elif HELDOUT_REQUEST.search(prompt):
            # The sheet's stick figure turned in 3D, framed like the turn guide (the last image).
            frame = _decode_data_url(body["images"][-1]["image_url"])
            img = fake_heldout_image(prompt, self.heldout_yaw, self.heldout_pose, frame=frame,
                                     gain=self.heldout_gain)
        else:
            raise AssertionError(f"unexpected prompt {prompt[:40]!r}")
        return {"data": [{"b64_json": _b64png(img)}], "usage": {"input_tokens": 1}}

    def decode(self, text: str, size: tuple[int, int]) -> list[SamInstance]:
        mask = imaging.threshold_mask(self.last_sam_image)
        if self.last_sam_text in FEET_TEXTS:  # each figure's feet: its lowest 5% of rows, one per column run
            out = []
            for x0, x1 in imaging.split_columns(mask):
                fig = np.zeros_like(mask)
                fig[:, x0:x1] = mask[:, x0:x1]
                b = imaging.bbox(fig)
                band = fig.copy()
                band[: b[3] - max(1, int(0.05 * (b[3] - b[1])))] = False
                for a, c in imaging.column_runs(band):
                    m = np.zeros_like(mask)
                    m[:, a:c] = band[:, a:c]
                    out.append(SamInstance(mask=m, box=imaging.bbox(m), confidence=0.9))
            return out
        out = []
        for x0, x1 in imaging.split_columns(mask):
            m = np.zeros_like(mask)
            m[:, x0:x1] = mask[:, x0:x1]
            b = imaging.bbox(m)
            out.append(SamInstance(mask=m, box=b, confidence=0.9))
        return out


class _RunsTest(unittest.TestCase):
    """A temporary runs root and a sketch, recorded through a fake API and replayed with the network blocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.sketch = Path(self.tmp.name) / "sketch.png"
        paint(place(figure_mask(200, "front"), (300, 280), (50, 40)), colour=(90, 90, 90)).save(self.sketch)

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, api: FakeAPI, run_id: str) -> int:
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", lambda c, e, b, **_: api.post(c, e, b)), \
                mock.patch.object(client_mod, "decode_sam_output", api.decode):
            return run_pipeline(
                self.sketch, run_id, [0, 1, 2, 5], mode="live", runs=str(self.root), label="test sketch"
            )

    def _replay(self, run_id: str) -> int:
        def no_network(*_a, **_k):
            raise AssertionError("replay must not call the network")

        env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(ModelClient, "_post", no_network):
            return run_pipeline(self.sketch, run_id, [0, 1, 2, 5], mode="replay", runs=str(self.root))


class FlowTest(_RunsTest):
    def test_record_then_replay(self):
        api = FakeAPI()
        self.assertEqual(self._record(api, "rec"), 0)
        run = self.root / "rec"
        for rel in ["00-sketch.png", "01-cover.png", "02-turnaround.png", "views/meta.json", "results.html",
                    "results.json"] + [f"views/{v}.png" for v in ("front", "back", "left", "right", "top")] \
                + [f"masks/{v}.png" for v in ("front", "back", "left", "right", "top")] \
                + ["masks/front-face.png", "masks/front-head.png"]:
            self.assertTrue((run / rel).exists(), rel)
        meta = json.loads((run / "views/meta.json").read_text())
        self.assertEqual(set(meta) >= {"pixels_per_unit", "height_units", "image_size"}, True)
        for v in ("front", "back", "left", "right", "top"):
            view = _img(run / f"views/{v}.png")
            mask = _img(run / f"masks/{v}.png")
            self.assertEqual(view.size, (meta["image_size"], meta["image_size"]))
            self.assertEqual(mask.size, view.size)
            self.assertEqual(mask.mode, "L")
        front = imaging.from_mask_image(_img(run / "masks/front.png"))
        front_h = imaging.box_size(imaging.bbox(front))[1]
        self.assertAlmostEqual(meta["pixels_per_unit"] * meta["height_units"], front_h, delta=0.5)
        self.assertEqual(meta["mask_method"], {"front": "sam-3.1", "right": "sam-3.1", "back": "sam-3.1",
                                               "left": "sam-3.1", "top": "visual_hull"})
        self.assertEqual(meta["mask_method_detail"]["front"], "sam-3.1:character")
        self.assertEqual(meta["top_method"], "visual_hull")
        s2 = json.loads((run / "stage-2.json").read_text())
        self.assertTrue(s2["ok"])
        self.assertEqual(len(s2["details"]["attempts"]), 1)
        # sheet and top view; SAM on the sheet, the top view and the front view's face and head
        self.assertAlmostEqual(s2["cost_usd"], 0.01 + 0.0025 + 0.01 + 0.0025 + 2 * 0.0025)
        # Never write the key anywhere.
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".html"):
                self.assertNotIn(DUMMY_KEY, p.read_text())

        self.assertEqual(self._replay("rep"), 0)
        rep = self.root / "rep"
        for rel in ["01-cover.png", "views/front.png", "masks/left.png", "views/top.png"]:
            self.assertEqual((run / rel).read_bytes(), (rep / rel).read_bytes(), rel)
        c_rec = [c["value"] for c in json.loads((run / "stage-2.json").read_text())["checks"]]
        c_rep = [c["value"] for c in json.loads((rep / "stage-2.json").read_text())["checks"]]
        self.assertEqual(c_rec, c_rep)
        self.assertEqual(json.loads((rep / "stage-2.json").read_text())["cost_usd"], 0.0)
        # stage 2f follows stage 2: the same close-up record on record and on replay, the replay free
        f_rec = json.loads((run / "stage-2f.json").read_text())
        f_rep = json.loads((rep / "stage-2f.json").read_text())
        self.assertTrue(f_rec["ok"])
        self.assertEqual(f_rec["details"]["used_views"], f_rep["details"]["used_views"])
        self.assertEqual(f_rep["cost_usd"], 0.0)
        self.assertEqual(api.prompts.count(facesource.PROMPTS["FC4"]), 1)

    def test_reordered_sheet_with_same_facing_profiles(self):
        api = FakeAPI(sheet_order=("front", "front", "right", "right"))
        self.assertEqual(self._record(api, "reorder"), 0)
        s2 = json.loads((self.root / "reorder/stage-2.json").read_text())
        assign = s2["details"]["view_assignment"]
        want = {"front": "panel0", "back": "panel1", "right": "panel2", "left": "panel3"}
        self.assertEqual(assign["panel_for_view"], want)
        self.assertFalse(assign["requested_order_kept"])
        self.assertTrue(assign["side_fixes"].get("mirrored_left"))
        left = imaging.from_mask_image(_img(self.root / "reorder/masks/left.png"))
        self.assertLess(imaging.facing_offset(left), 0)

    def test_partial_cache_is_not_a_threshold_mask(self):
        # A SAM call missing from the replay cache stops the stage; it never falls back to the threshold mask.
        self.assertEqual(self._record(FakeAPI(), "rec"), 0)
        cache = self.root / "_cache"
        sketch = _img(self.root / "rec" / "00-sketch.png")
        sha = client_mod._sha(client_mod.png_bytes(sketch))
        for p in sorted(cache.glob("*.json")):
            req = json.loads(p.read_text())["request"]
            if req.get("text") == stages.SAM_TEXTS[0] and req.get("image_sha256") == sha:
                p.unlink()
        with self.assertRaises(ReplayMiss):
            stages.sam_figure_mask(ModelClient(cache), sketch)
        self.assertEqual(self._replay("rep"), 1)
        self.assertFalse((self.root / "rep" / "stage-1.json").exists())

    def test_corrupt_cached_mask_is_not_a_threshold_mask(self):
        # A cached SAM mask of the wrong size is a broken cache, not an API failure: the replay stops.
        self.assertEqual(self._record(FakeAPI(), "rec"), 0)
        cache = self.root / "_cache"
        mask = sorted(cache.glob("*-mask-0.png"))[0]
        Image.new("L", (3, 3), 255).save(mask)
        self.assertEqual(self._replay("rep"), 1)

    def test_live_sam_failure_stops_the_stage_and_is_not_cached(self):
        # A failed SAM call is neither a fallback to the next prompt nor a cached result.
        api = FakeAPI()
        post = api.post

        def flaky(c, endpoint, body):
            if endpoint == "responses" and body["input"][0]["content"][0]["text"] == stages.SAM_TEXTS[0]:
                raise ApiError("responses failed: HTTP 503")
            return post(c, endpoint, body)

        api.post = flaky
        self.assertEqual(self._record(api, "rec"), 1)
        s1 = json.loads((self.root / "rec" / "stage-1.json").read_text())
        self.assertEqual((s1["ok"], s1["status"]), (False, "error"))
        self.assertIn("HTTP 503", s1["error"])
        self.assertEqual(api.calls, ["question"])  # the photo question, then the sketch's first SAM prompt failed
        cached = [json.loads(p.read_text())["request"]["model"] for p in (self.root / "_cache").glob("*.json")]
        self.assertEqual(cached, [client_mod.TEXT_MODEL])  # only the photo question's answer, no SAM entry
        self.assertEqual(self._replay("rep"), 1)

    def test_gate_failure_stops_and_still_reports(self):
        api = FakeAPI(sheet_order=("front", "right", "front"))
        self.assertEqual(self._record(api, "bad"), 2)
        run = self.root / "bad"
        s2 = json.loads((run / "stage-2.json").read_text())
        self.assertFalse(s2["ok"])
        self.assertEqual(len(s2["details"]["attempts"]), 4)
        self.assertNotIn(stages.TOP_PROMPT, api.prompts)  # no top-view call after a failed gate
        self.assertEqual(api.prompts.count(stages.TURNAROUND_PROMPT), 4)
        res = json.loads((run / "results.json").read_text())
        status = {s["stage"]: s["status"] for s in res["stages"]}
        self.assertEqual(status, {0: "pass", 1: "pass", 2: "gate failed", 3: "not run", 4: "not run"})
        self.assertIn("gate failed", (run / "results.html").read_text())


class SheetsAPI(FakeAPI):
    """A fake API that answers the n-th turnaround request with ``sheets[n]``."""

    def __init__(self, sheets) -> None:
        super().__init__()
        self.sheets = sheets
        self.n = 0

    def post(self, c, endpoint, body):
        if endpoint == "images/edits" and body["prompt"] == stages.TURNAROUND_PROMPT:
            img = self.sheets[min(self.n, len(self.sheets) - 1)]
            self.n += 1
            self.calls.append(endpoint)
            self.prompts.append(body["prompt"])
            return {"data": [{"b64_json": _b64png(img)}], "usage": {"input_tokens": 1}}
        return super().post(c, endpoint, body)


def _two_gates_sheet() -> Image.Image:
    """Arms down (fails the A-pose gate) and a short left profile (fails the height gate, and more)."""
    figs = [figure_mask(200, "front", "down"), figure_mask(200, "right"), figure_mask(200, "front", "down"),
            figure_mask(150, "left")]
    gap = 40
    mask = np.zeros((280, sum(f.shape[1] for f in figs) + gap * 5), dtype=bool)
    x = gap
    for f in figs:
        mask[240 - f.shape[0] : 240, x : x + f.shape[1]] = f
        x += f.shape[1] + gap
    return paint(mask)


def _one_legged_back_sheet() -> Image.Image:
    """Every view right except the back, drawn with one leg: it passes every silhouette gate (same height and
    width, same profiles, front arms out) but stage 3's carve keeps only what front and back both show."""
    back = figure_mask(200, "front").copy()
    back[110:, back.shape[1] // 2:] = False
    figs = [figure_mask(200, "front"), figure_mask(200, "right"), back, figure_mask(200, "left")]
    gap = 40
    mask = np.zeros((280, sum(f.shape[1] for f in figs) + gap * 5), dtype=bool)
    x = gap
    for f in figs:
        mask[240 - f.shape[0] : 240, x : x + f.shape[1]] = f
        x += f.shape[1] + gap
    return paint(mask)


class APoseRecordedTest(_RunsTest):
    """The A-pose span ratio is recorded on every sheet but no longer decides whether a sheet is kept."""

    def test_arms_down_sheet_that_carves_is_kept(self):
        down, _ = sheet(arms="down")
        api = SheetsAPI([down])
        self._record(api, "d")  # stage 2h, which follows, has its own checks; only stage 2 is under test
        s2 = json.loads((self.root / "d" / "stage-2.json").read_text())
        self.assertTrue(s2["ok"])
        self.assertEqual(api.n, 1)
        ap = next(c for c in s2["checks"] if c["name"] == "a_pose_hand_span_over_shoulders")
        self.assertEqual((ap["pass"], ap["gate"]), (False, False))
        self.assertNotIn("a_pose_hand_span_over_shoulders", stages.SHEET_GATES)


class CarveCheckFlowTest(_RunsTest):
    """A sheet stage 3's carve would refuse triggers another attempt before accepting the best sheet."""

    def test_a_sheet_that_does_not_carve_is_retried(self):
        good, _ = sheet()
        api = SheetsAPI([_one_legged_back_sheet(), good])
        self.assertEqual(self._record(api, "c"), 0)
        s2 = json.loads((self.root / "c" / "stage-2.json").read_text())
        first, second = s2["details"]["attempts"]
        self.assertEqual([c["name"] for c in first["checks"] if not c["pass"] and c.get("gate", True)],
                         [stages.CARVE_GATE])
        self.assertIn("worst: front", next(c for c in first["checks"] if c["name"] == stages.CARVE_GATE)["note"])
        self.assertEqual(s2["details"]["kept_attempt"], 1)
        self.assertTrue(next(c for c in s2["checks"] if c["name"] == stages.CARVE_GATE)["pass"])


class KeepBestSheetTest(_RunsTest):
    """Stage 2 keeps its best sheet (fewest failed gates, then the smallest shortfall), not its last."""

    def _sheets(self, sheets) -> tuple[int, dict, SheetsAPI]:
        api = SheetsAPI(sheets)
        code = self._record(api, "k")
        return code, json.loads((self.root / "k" / "stage-2.json").read_text()), api

    def test_keeps_the_sheet_with_the_fewest_failed_gates(self):
        unsplit, _ = sheet(order=("front", "right", "front"))
        one = _one_legged_back_sheet()
        sheets = [unsplit, one, _two_gates_sheet(), unsplit]
        code, s2, api = self._sheets(sheets)
        self.assertEqual(code, 0)
        self.assertTrue(s2["ok"])
        self.assertEqual(api.n, 4)
        ranks = [a["rank"]["failed_gates"] for a in s2["details"]["attempts"]]
        self.assertEqual(ranks[1], 1)
        self.assertGreaterEqual(ranks[2], 2)
        self.assertEqual(ranks[0], ranks[3])
        self.assertGreater(ranks[0], ranks[2])  # a sheet that can't be split ranks below any that can
        self.assertEqual(s2["details"]["kept_attempt"], 1)
        self.assertIn("generation 2 is kept", s2["notes"])
        kept = s2["details"]["attempts"][1]
        self.assertEqual([(c["name"], c["value"], c["pass"]) for c in s2["checks"]
                          if c["name"] in {k["name"] for k in kept["checks"]}],
                         [(c["name"], c["value"], c["pass"]) for c in kept["checks"]])
        self.assertEqual([c["name"] for c in s2["checks"] if not c["pass"] and c.get("gate", True)],
                         [])
        carve = next(c for c in s2["checks"] if c["name"] == stages.CARVE_GATE)
        self.assertEqual((carve["pass"], carve["gate"]), (False, False))
        self.assertTrue(next(c for c in kept["checks"] if c["name"] == stages.CARVE_GATE)["gate"])
        self.assertIn("carve fallback would be refused", s2["notes"])
        self.assertEqual(s2["details"]["view_assignment"], kept["assignment"])
        run = self.root / "k"
        self.assertEqual(np.asarray(_img(run / "02-turnaround.png")).tobytes(), np.asarray(one).tobytes())
        back = imaging.from_mask_image(_img(run / "masks/back.png"))
        front = imaging.from_mask_image(_img(run / "masks/front.png"))
        self.assertLess(back.sum(), front.sum())  # attempt 2's back, one leg hidden
        self.assertEqual(api.prompts.count(stages.TOP_PROMPT), 1)
        self.assertTrue((run / "views/meta.json").exists())
        self.assertTrue((run / "masks/parts.json").exists())

    def test_a_tie_keeps_the_earlier_sheet(self):
        one = _one_legged_back_sheet()
        code, s2, _ = self._sheets([one, one, one, one])
        self.assertEqual(code, 0)
        self.assertEqual(s2["details"]["kept_attempt"], 0)

    def test_a_passing_sheet_is_kept_and_stops_the_attempts(self):
        unsplit, _ = sheet(order=("front", "right", "front"))
        good, _ = sheet()
        code, s2, api = self._sheets([unsplit, good])
        self.assertEqual(code, 0)
        self.assertEqual((api.n, s2["details"]["kept_attempt"]), (2, 1))
        self.assertTrue(s2["ok"])


class ClientTest(unittest.TestCase):
    def test_replay_miss_and_no_key(self):
        with tempfile.TemporaryDirectory() as d:
            c = ModelClient(d, mode="replay")
            with self.assertRaises(ReplayMiss):
                c.generate_image("anything")
            env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
            with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(ClientError):
                ModelClient(d, mode="live").generate_image("anything")

    def test_request_key_ignores_nothing_that_matters(self):
        a = client_mod.request_key({"prompt": "x", "images_sha256": ["1"]})
        self.assertEqual(a, client_mod.request_key({"images_sha256": ["1"], "prompt": "x"}))
        self.assertNotEqual(a, client_mod.request_key({"prompt": "x", "images_sha256": ["2"]}))

    def test_scrub_removes_key(self):
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}):
            self.assertNotIn(DUMMY_KEY, client_mod._scrub(f"bad key {DUMMY_KEY} Bearer abc.def"))


class CliTest(unittest.TestCase):
    def test_parse_stages(self):
        self.assertEqual(parse_stages("0-2,5"), [0, 1, 2, 5])
        self.assertEqual(parse_stages("2"), [2])
        self.assertEqual(parse_stages("2h"), ["2h"])
        self.assertEqual(parse_stages("5,2H,0-2"), [0, 1, 2, "2h", 5])
        with self.assertRaises(ValueError):
            parse_stages("0-4")


if __name__ == "__main__":
    unittest.main()
