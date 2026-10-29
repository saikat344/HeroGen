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
from herogen.pipeline import heldout, imaging, stages  # noqa: E402
from herogen.pipeline.__main__ import parse_stages, run_pipeline  # noqa: E402
from herogen.pipeline.client import ClientError, ModelClient, ReplayMiss, SamInstance  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place, sheet  # noqa: E402

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


class FakeAPI:
    """Stands in for Meta's Model API. Draws every image by code; SAM masks come from thresholding."""

    def __init__(self, sheet_order=("front", "right", "front", "left"), heldout_squash: float = 0.7) -> None:
        self.sheet_order = sheet_order
        self.heldout_squash = heldout_squash  # held-out figure width / front width (1.0 fails the angle check)
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.last_sam_image: Image.Image | None = None

    def post(self, _client, endpoint: str, body: dict) -> dict:
        assert os.environ.get("MODEL_API_KEY") == DUMMY_KEY
        self.calls.append(endpoint)
        if endpoint == "responses":
            self.last_sam_image = _decode_data_url(body["input"][0]["content"][1]["image_url"])
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "fake"}]}]}
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
        elif prompt == heldout.HELDOUT_PROMPT:
            # A turned figure: the front figure squeezed sideways, same height and framing as the cover.
            front = imaging.to_mask_image(figure_mask(200, "front"))
            w = max(1, round(front.size[0] * self.heldout_squash))
            turned = imaging.from_mask_image(front.resize((w, front.size[1]), Image.BILINEAR))
            img = paint(place(turned, (300, 280), (150 - w // 2, 40)), colour=(30, 80, 160), bg=(255, 255, 255))
        else:
            raise AssertionError(f"unexpected prompt {prompt[:40]!r}")
        return {"data": [{"b64_json": _b64png(img)}], "usage": {"input_tokens": 1}}

    def decode(self, text: str, size: tuple[int, int]) -> list[SamInstance]:
        mask = imaging.threshold_mask(self.last_sam_image)
        out = []
        for x0, x1 in imaging.split_columns(mask):
            m = np.zeros_like(mask)
            m[:, x0:x1] = mask[:, x0:x1]
            b = imaging.bbox(m)
            out.append(SamInstance(mask=m, box=b, confidence=0.9))
        return out


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.sketch = Path(self.tmp.name) / "sketch.png"
        paint(place(figure_mask(200, "front"), (300, 280), (50, 40)), colour=(90, 90, 90)).save(self.sketch)

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, api: FakeAPI, run_id: str) -> int:
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", lambda c, e, b: api.post(c, e, b)), \
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

    def test_record_then_replay(self):
        api = FakeAPI()
        self.assertEqual(self._record(api, "rec"), 0)
        run = self.root / "rec"
        for rel in ["00-sketch.png", "01-cover.png", "02-turnaround.png", "views/meta.json", "results.html",
                    "results.json"] + [f"views/{v}.png" for v in ("front", "back", "left", "right", "top")] \
                + [f"masks/{v}.png" for v in ("front", "back", "left", "right", "top")]:
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
        self.assertAlmostEqual(s2["cost_usd"], 0.01 + 0.0025 + 0.01 + 0.0025)  # sheet+SAM, top+SAM
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
