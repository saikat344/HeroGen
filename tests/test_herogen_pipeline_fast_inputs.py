"""Necessary fast input gates and call ordering with synthetic images and fake clients."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from herogen.pipeline import client as client_module, inputs, stages
from herogen.pipeline.__main__ import run_pipeline
from herogen.pipeline.client import ApiError, ModelClient, ReplayMiss, SamInstance
from herogen.pipeline.runfolder import RunFolder
from herogen.service.runner import CliRunner, StageContext, StageSpec
from test_herogen_pipeline_fast import ConcurrentAPI
from test_herogen_pipeline_prompt import draw
import test_herogen_pipeline_flow as flow


class FastInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = RunFolder(self.root, "hero")
        self.api = ConcurrentAPI()
        self.client = ModelClient(self.root / "cache", mode="live", fast=True)
        self.calls = []
        self.before = lambda label: None
        self.after = lambda label: None
        self.feet = "normal"
        self.failures = {}

        def post(client, endpoint, body, **kwargs):
            if endpoint == "responses":
                label = "photo" if "text" in body else body["input"][0]["content"][0]["text"]
            else:
                label = "cover"
            self.calls.append(label)
            self.before(label)
            if label in self.failures:
                raise self.failures[label]
            answer = self.api.post(client, endpoint, body)
            self.after(label)
            return answer

        def decode(text, size):
            if self.api.last_sam_text == "feet":
                if self.feet == "missing":
                    return []
                if self.feet == "upper":
                    mask = np.zeros((size[1], size[0]), dtype=bool)
                    mask[60:70, 280:300] = True
                    return [SamInstance(mask, (280, 60, 300, 70), .9)] * 2
            return self.api.decode(text, size)

        for patch in (mock.patch.dict(os.environ, {"MODEL_API_KEY": flow.DUMMY_KEY}),
                      mock.patch.object(ModelClient, "_post", post),
                      mock.patch.object(client_module, "decode_sam_output", decode)):
            patch.start()
            self.addCleanup(patch.stop)

    def input(self, scenario="ok", *, generated=False):
        image = draw(scenario)
        path = self.root / "input.png"
        image.save(path)
        origin = None
        if generated:
            image.save(self.run.file("p-full-body.png"))
            self.run.write_stage({"stage": "p", "ok": True, "details": {"user_text": "a knight"}})
            origin = {"stage": "p", "image": "p-full-body.png"}
        stages.stage0_input(self.run, path, origin=origin, fast=True)
        return image

    def reject(self, message):
        with self.assertRaises(stages.InputRejected) as raised:
            stages.stage1_cover(self.run, self.client)
        record = self.run.read_stage(1)
        self.assertEqual(raised.exception.record, record)
        self.assertEqual(record["status"], "input_rejected")
        self.assertFalse(record["ok"])
        failed = [c for c in record["checks"] if c["gate"] and not c["pass"]]
        self.assertTrue(failed)
        self.assertTrue(all(c["gate"] and c["verdict"] == "reject" for c in failed))
        self.assertTrue(all({"name", "value", "threshold"} <= c.keys() for c in failed))
        spec = StageSpec("1", "cover art", "model", "stage-1.json")
        context = mock.Mock(spec=StageContext, run_dir=self.run.path)
        outcome = CliRunner.outcome(spec, context, 0, 3, "", "", False)
        self.assertEqual(outcome.state, "rejected")
        self.assertFalse(outcome.transient)
        if "exactly one" in message:
            service_message = ("This image shows more than one character." if failed[0]["value"] else
                               "We couldn't find a character in this image.")
        elif "head to feet" in message:
            service_message = "The character is cut off; HeroGen needs the full body, head to feet."
        else:
            service_message = message
        self.assertIn(service_message, outcome.message)
        if "photo of a real person" not in message:
            self.assertIn(message, record["error"])
        self.assertFalse(self.run.exists("01-cover.png"))
        return record

    def test_zero_and_two_characters_reject_without_cover_or_retry(self):
        for scenario, count in (("blank", 0), ("two", 2)):
            with self.subTest(scenario=scenario):
                self.calls.clear()
                self.input(scenario)
                record = self.reject(f"We need exactly one character in the picture; we found {count}.")
                check = next(c for c in record["checks"] if c["name"] == "sketch_characters_found")
                self.assertEqual((check["value"], check["threshold"]), (count, 1))
                self.assertCountEqual(self.calls, ["character", "feet", "photo"])

    def test_missing_and_upper_half_feet_reject(self):
        self.input()
        for feet in ("missing", "upper"):
            with self.subTest(feet=feet):
                self.feet = feet
                self.calls.clear()
                # Each variant needs its own fake decode rather than a previous cached response.
                self.client = ModelClient(self.root / feet, mode="live", fast=True)
                record = self.reject("Show the whole character from head to feet;")
                check = next(c for c in record["checks"] if c["name"] == "feet_in_lower_half")
                self.assertEqual((check["value"], check["threshold"]), (0, 2))
                self.assertCountEqual(self.calls, ["character", "feet", "photo"])

    def test_bottom_crop_rejects_even_when_two_feet_are_detected(self):
        self.input("cut")
        record = self.reject("Show the whole character from head to feet; the feet were cut off.")
        checks = {c["name"]: c for c in record["checks"]}
        self.assertTrue(checks["feet_in_lower_half"]["pass"])
        self.assertEqual(checks["figure_bottom_gap_px"]["value"], 0)
        self.assertNotIn("cover", self.calls)

    def test_top_crop_rejects(self):
        image = self.input()
        Image.fromarray(np.roll(np.asarray(image), -50, axis=0)).save(self.run.file("00-sketch.png"))
        self.reject("Show the whole character from head to feet; the head was cut off.")
        self.assertNotIn("cover", self.calls)

    def test_small_extra_character_is_still_another_instance(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 30:70] = True
        small = np.zeros_like(mask)
        small[50, 5] = True
        figures = [SamInstance(mask, (30, 10, 70, 90), .9), SamInstance(small, (5, 50, 6, 51), .9)]
        checks, _ = inputs.fast_figure_checks(figures, [], 100)
        self.assertEqual(checks[0]["value"], 2)
        self.assertFalse(checks[0]["pass"])

    def test_edge_tolerance_and_feet_thresholds_are_unchanged(self):
        for gap in (2, 3):  # 100 px has the existing 2 px minimum tolerance
            mask = np.zeros((100, 100), dtype=bool)
            mask[gap:100-gap, 30:70] = True
            figure = SamInstance(mask, (30, gap, 70, 100-gap), .9)
            foot_mask = np.zeros_like(mask)
            foot_mask[80:90, 40:50] = True
            foot = SamInstance(foot_mask, (40, 80, 50, 90), .9)
            checks, _ = inputs.fast_figure_checks([figure], [foot, foot], 100)
            self.assertEqual(checks[1]["threshold"], 2)
            self.assertTrue(checks[1]["pass"])
            self.assertEqual([c["pass"] for c in checks[2:]], [gap > 2, gap > 2, gap > 2])

    def test_prompt_checks_overlap_cover_and_use_one_feet_call(self):
        self.input(generated=True)
        barrier = threading.Barrier(3)
        self.before = lambda _: barrier.wait(timeout=5)
        record = stages.stage1_cover(self.run, self.client)
        self.assertTrue(record["ok"])
        self.assertCountEqual(self.calls, ["character", "feet", "cover"])
        self.assertTrue(all(c["pass"] for c in record["checks"]))
        self.assertEqual({c["name"] for c in record["checks"] if c["gate"]},
                         {"sketch_characters_found", "sketch_feet_on_figure"})
        self.assertEqual(len(record["calls"]), 3)
        self.assertTrue(self.run.exists("01-cover.png"))

    def test_rejected_prompt_discards_speculative_cover_and_stale_output(self):
        self.input("two", generated=True)
        draw("ok").save(self.run.file("01-cover.png"))
        record = self.reject("We need exactly one character in the picture; we found 2.")
        self.assertCountEqual(self.calls, ["character", "feet", "cover"])
        self.assertNotIn("01-cover.png", record["outputs"])

    def test_upload_checks_overlap_and_photo_passes_before_cover(self):
        self.input()
        barrier = threading.Barrier(3)
        passed = threading.Event()
        def before(label):
            if label == "cover":
                self.assertTrue(passed.is_set())
            else:
                barrier.wait(timeout=5)
        self.before = before
        self.after = lambda label: passed.set() if label == "photo" else None
        record = stages.stage1_cover(self.run, self.client)
        self.assertTrue(record["ok"])
        self.assertCountEqual(self.calls[:3], ["character", "feet", "photo"])
        self.assertEqual(self.calls[3:], ["cover"])

    def test_photo_refusal_prevents_cover(self):
        self.input()
        self.api.photo_answer.update(real_person_photo_probability=.99, image_kind="photo_of_real_person")
        record = self.reject("This looks like a photo of a real person.")
        self.assertCountEqual(self.calls, ["character", "feet", "photo"])
        self.assertIn("Upload a drawing or artwork", record["error"])

    def test_missing_or_mismatched_prompt_provenance_requires_photo_gate(self):
        for provenance in ("missing", "mismatch", "failed"):
            with self.subTest(provenance=provenance):
                self.input(generated=True)
                if provenance == "missing":
                    self.run.file("p-full-body.png").unlink()
                elif provenance == "mismatch":
                    draw("two").save(self.run.file("p-full-body.png"))
                else:
                    self.run.write_stage({"stage": "p", "ok": False})
                self.api.photo_answer.update(real_person_photo_probability=.99, image_kind="photo_of_real_person")
                self.calls.clear()
                self.client = ModelClient(self.root / provenance, mode="live", fast=True)
                self.reject("This looks like a photo of a real person.")
                self.assertCountEqual(self.calls, ["character", "feet", "photo"])

    def test_model_failures_stay_loud_and_drain_concurrent_calls(self):
        for label, failure in (("photo", ApiError("photo failed")),
                               ("character", ReplayMiss("character missing")),
                               ("feet", ApiError("feet failed")),
                               ("cover", ApiError("cover failed"))):
            with self.subTest(label=label):
                self.input(generated=label == "cover")
                self.calls.clear()
                self.failures = {label: failure}
                self.client = ModelClient(self.root / label, mode="live", fast=True)
                with self.assertRaises(type(failure)):
                    stages.stage1_cover(self.run, self.client)
                record = self.run.read_stage(1)
                self.assertEqual(record["status"], "error")
                self.assertEqual(len(record["calls"]), 3)
                self.assertCountEqual(self.calls, ["character", "feet", "cover" if label == "cover" else "photo"])
                self.assertFalse(self.run.exists("01-cover.png"))

    def test_cli_rejection_stops_before_stage_two_without_retry(self):
        self.input("two")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = run_pipeline(self.root / "input.png", "cli", [0, 1, 2], runs=str(self.root),
                                mode="live", fast=True)
        self.assertEqual(code, 3)
        self.assertIn("We need exactly one character in the picture; we found 2.", err.getvalue())
        self.assertFalse(RunFolder(self.root, "cli").exists("stage-2.json"))
        self.assertCountEqual(self.calls, ["character", "feet", "photo"])
