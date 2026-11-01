"""Fast front-part overlap and request-local receipts, using synthetic responses only."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock

import numpy as np
from PIL import Image

from herogen.pipeline import client as client_module, stages
from herogen.pipeline.client import ApiError, ModelClient, ReplayMiss
from herogen.pipeline.parts import front_part_masks
from herogen.pipeline.retry import RetryPolicy
from herogen.pipeline.report import report
from herogen.pipeline.runfolder import RunFolder
from herogen.pipeline.__main__ import run_pipeline
from fake_model_server import sam_response
import test_herogen_pipeline_flow as flow
from test_herogen_pipeline_synth import figure_mask, paint, place


class ParallelPartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = Image.new("RGB", (16, 16), "white")
        self.figure = np.ones((16, 16), dtype=bool)

    def test_fast_part_calls_overlap_and_keep_separate_receipts(self):
        barrier = threading.Barrier(2)
        client = ModelClient(self.root, mode="live", fast=True)
        client.stage, client.purpose = 2, "front parts"
        def post(c, endpoint, body, **kwargs):
            barrier.wait(timeout=5)
            return sam_response("")
        with mock.patch.object(ModelClient, "_post", post), \
                mock.patch.object(client_module, "decode_sam_output", return_value=[]):
            masks, info = front_part_masks(client, self.image, self.figure)
        self.assertEqual(masks, {})
        self.assertEqual(list(info), ["face", "head"])
        self.assertEqual(len(client.log.calls), 2)
        self.assertEqual({r["purpose"] for r in client.log.calls},
                         {"segment face (front parts)", "segment head (front parts)"})
        self.assertEqual(len({r["key"] for r in client.log.calls}), 2)
        self.assertEqual({r["stage"] for r in client.log.calls}, {2})
        self.assertTrue(all(r["outcome"] == "ok" for r in client.log.calls))
        run = RunFolder(self.root, "overlap")
        run.write_stage({"stage": 2, "name": "parts", "seconds": 1, "calls": client.log.calls})
        data = report(run.path, stream=io.StringIO())
        self.assertEqual(data["stages"][0]["peak_concurrent_calls"], 2)
        # Replaying the same requests still overlaps safely and has no billable cost.
        client.mode = "replay"
        client.log.calls.clear()
        front_part_masks(client, self.image, self.figure)
        self.assertEqual(len(client.log.calls), 2)
        self.assertTrue(all(r["cached"] and r["cost_usd"] == 0 for r in client.log.calls))

    def test_both_failures_are_drained_recorded_and_raised_in_prompt_order(self):
        for first, second in ((ApiError("face failed"), ReplayMiss("head missing")),
                              (ReplayMiss("face missing"), ApiError("head failed"))):
            with self.subTest(first=type(first).__name__):
                barrier = threading.Barrier(2)
                client = ModelClient(self.root, mode="live", fast=True)
                def post(c, endpoint, body, **kwargs):
                    text = body["input"][0]["content"][0]["text"]
                    barrier.wait(timeout=5)
                    raise first if text == "face" else second
                run = RunFolder(self.root, "error")
                stage = stages._Stage(run, client, 2, "front_parts")
                with mock.patch.object(ModelClient, "_post", post):
                    with self.assertRaises(type(first)) as raised:
                        stages._stopping_on_client_error(
                            lambda: front_part_masks(client, self.image, self.figure), stage)
                self.assertIs(raised.exception, first)
                rec = run.read_stage(2)
                self.assertEqual(rec["status"], "error")
                self.assertEqual(len(rec["calls"]), 2)
                self.assertEqual({r["outcome"] for r in rec["calls"]}, {"ApiError", "ReplayMiss"})
                self.assertIn(str(first), rec["error"])

    def test_overlapping_transport_retry_does_not_mix_costs_or_attempts(self):
        barrier = threading.Barrier(2)
        attempts = {"face": 0, "head": 0}
        limiter = mock.Mock()
        limiter.acquire.return_value = ("slot", 0)
        client = ModelClient(self.root, mode="live", fast=True, limiter=limiter,
                             policy=RetryPolicy(sleep=lambda _: None, base_s=0))
        settled = []
        client.after_api_attempt = settled.append
        def send(request, timeout, guard):
            text = json.loads(request.data)["input"][0]["content"][0]["text"]
            attempts[text] += 1
            if attempts[text] == 1:
                barrier.wait(timeout=5)
            if text == "face" and attempts[text] == 1:
                raise urllib.error.HTTPError("synthetic", 429, "busy", {"Retry-After": "0"}, None)
            response = mock.MagicMock()
            response.__enter__.return_value.status = 200
            response.__enter__.return_value.read.return_value = json.dumps(sam_response("")).encode()
            return response
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "placeholder"}), \
                mock.patch.object(client_module, "_open", send), \
                mock.patch.object(client_module, "decode_sam_output", return_value=[]), \
                self.assertLogs("herogen.pipeline.client", "WARNING"):
            front_part_masks(client, self.image, self.figure)
        self.assertEqual(attempts, {"face": 2, "head": 1})
        self.assertEqual(len(client.log.calls), 3)
        face = [r for r in client.log.calls if "segment face" in r["purpose"]]
        head = [r for r in client.log.calls if "segment head" in r["purpose"]]
        self.assertEqual([r["attempt"] for r in face], [1, 2])
        self.assertEqual([r["attempt"] for r in head], [1])
        self.assertEqual([r["cost_usd"] for r in face], [0, .0025])
        self.assertEqual([r["cost_usd"] for r in head], [.0025])
        self.assertEqual(client.log.cost_usd(), .005)
        self.assertEqual(sorted(settled), [0, .0025, .0025])
        rows = [json.loads(line) for line in (self.root / "calls-ledger.jsonl").read_text().splitlines()]
        complete = [r for r in rows if r["outcome"] != "sent"]
        self.assertEqual(len(complete), 3)
        self.assertEqual(sum(r["cost_usd"] for r in complete), .005)
        self.assertTrue(any(c.kwargs["rate_limited"] for c in limiter.release.call_args_list))

    def test_normal_parts_stop_at_replay_miss_before_requesting_head(self):
        client = ModelClient(self.root)
        with mock.patch.object(client, "segment", side_effect=ReplayMiss("no face")) as segment:
            with self.assertRaises(ReplayMiss):
                front_part_masks(client, self.image, self.figure)
        self.assertEqual([c.args[1] for c in segment.call_args_list], ["face"])


class NormalCallSequenceTests(unittest.TestCase):
    def test_stage_zero_to_two_matches_recorded_base_call_sequence(self):
        # Recorded at 42680b7 with the same synthetic input and responses.
        expected = ["compare:muse-spark-1.3", "segment:character", "segment:foot", "segment:face",
                    "segment:head", "cover", "segment:character", "sheet", "segment:character",
                    "top", "segment:face", "segment:head"]
        api = flow.FakeAPI()
        sequence = []
        def post(client, endpoint, body, **kwargs):
            if endpoint == "responses":
                label = ("compare:" + body["model"] if "text" in body
                         else "segment:" + body["input"][0]["content"][0]["text"])
            else:
                label = next(name for name, prompt in (("cover", stages.COVER_PROMPT),
                             ("sheet", stages.TURNAROUND_PROMPT), ("top", stages.TOP_PROMPT))
                             if body["prompt"].startswith(prompt))
            sequence.append(label)
            return api.post(client, endpoint, body)
        with tempfile.TemporaryDirectory() as directory:
            sketch = Path(directory) / "sketch.png"
            paint(place(figure_mask(200, "front"), (300, 280), (50, 40))).save(sketch)
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": flow.DUMMY_KEY}), \
                    mock.patch.object(ModelClient, "_post", post), \
                    mock.patch.object(client_module, "decode_sam_output", api.decode), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = run_pipeline(sketch, "hero", [0, 1, 2], runs=directory, mode="live", stop_after=2)
        self.assertEqual(code, 0)
        self.assertEqual(sequence, expected)
