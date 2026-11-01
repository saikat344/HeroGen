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

from herogen import prompts
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
        self.requests = []
        self.before = lambda label: None
        self.after = lambda label: None
        self.feet = "normal"
        self.cover_feet = "normal"
        self.cover_image = None
        self.failures = {}

        def post(client, endpoint, body, **kwargs):
            if endpoint == "responses":
                label = "photo" if "text" in body else body["input"][0]["content"][0]["text"]
            else:
                label = "cover"
            self.calls.append(label)
            self.requests.append((label, body))
            self.before(label)
            if label in self.failures:
                raise self.failures[label]
            completion = prompts.selected("cover_complete_figure").step("cover")["prompt"]
            if label == "cover" and (self.cover_image is not None or body["prompt"].startswith(completion)):
                image = self.cover_image.copy() if self.cover_image is not None else draw("ok")
                image.putpixel((0, 0), (255, 255, 255))
                answer = {"data": [{"b64_json": flow._b64png(image)}]}
            else:
                answer = self.api.post(client, endpoint, body)
            self.after(label)
            return answer

        def decode(text, size):
            if self.api.last_sam_text == "feet":
                upload = np.array_equal(np.asarray(self.api.last_sam_image), np.asarray(self.upload_image))
                feet = self.feet if upload else self.cover_feet
                if feet == "missing":
                    return []
                if feet == "upper":
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
        self.upload_image = image
        path = self.root / "input.png"
        image.save(path)
        origin = None
        if generated:
            image.save(self.run.file("p-full-body.png"))
            self.run.write_stage({"stage": "p", "ok": True, "details": {"user_text": "a knight"}})
            origin = {"stage": "p", "image": "p-full-body.png"}
        stages.stage0_input(self.run, path, origin=origin, fast=True)
        return image

    def reject(self, message, *, cover_exists=False):
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
        self.assertEqual(self.run.exists("01-cover.png"), cover_exists)
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

    def assert_completion(self, record, failed):
        self.assertTrue(record["ok"])
        self.assertTrue(record["details"]["completed_figure"])
        self.assertEqual(record["details"]["upload_failed_checks"], [*failed, "sketch_feet_on_figure"])
        version = prompts.selected("cover_complete_figure")
        self.assertEqual(record["details"]["prompt_version"], version.ref())
        self.assertEqual(record["details"]["prompt"], version.step("cover")["prompt"])
        self.assertEqual([b["prompt"] for label, b in self.requests if label == "cover"],
                         [version.step("cover")["prompt"]])
        checks = {c["name"]: c for c in record["checks"]}
        self.assertEqual({c["name"] for c in record["checks"] if c["gate"]},
                         {"upload_real_person_photo_probability", "sketch_characters_found",
                          "sketch_feet_on_figure"})
        self.assertFalse(checks["upload_feet_on_figure"]["gate"])
        self.assertTrue(checks["sketch_feet_on_figure"]["pass"])
        self.assertCountEqual(self.calls[:3], ["character", "feet", "photo"])
        self.assertEqual(self.calls[3:4], ["cover"])
        self.assertCountEqual(self.calls[4:], ["character", "feet"])
        # The last two SAM requests must measure the generated image, not the cut-off upload.
        for label, body in self.requests[-2:]:
            image = flow._decode_data_url(body["input"][0]["content"][1]["image_url"])
            with Image.open(self.run.file("01-cover.png")) as saved:
                np.testing.assert_array_equal(np.asarray(image), np.asarray(saved))

    def test_missing_and_upper_half_feet_complete(self):
        self.input()
        for feet in ("missing", "upper"):
            with self.subTest(feet=feet):
                self.feet = feet
                self.calls.clear()
                self.requests.clear()
                # Each variant needs its own fake decode rather than a previous cached response.
                self.client = ModelClient(self.root / feet, mode="live", fast=True)
                record = stages.stage1_cover(self.run, self.client)
                self.assert_completion(record, ["feet_in_lower_half"])
                check = next(c for c in record["checks"] if c["name"] == "upload_feet_in_lower_half")
                self.assertEqual((check["value"], check["threshold"]), (0, 2))

    def test_bottom_crop_completes_even_when_two_feet_are_detected(self):
        self.input("cut")
        cover_started = threading.Event()
        barrier = threading.Barrier(2)
        self.after = lambda label: cover_started.set() if label == "cover" else None
        self.before = lambda _: barrier.wait(timeout=5) if cover_started.is_set() else None
        record = stages.stage1_cover(self.run, self.client)
        self.assert_completion(record, ["figure_bottom_gap_px"])
        checks = {c["name"]: c for c in record["checks"]}
        self.assertTrue(checks["upload_feet_in_lower_half"]["pass"])
        self.assertEqual(checks["upload_figure_bottom_gap_px"]["value"], 0)

    def test_top_crop_completes(self):
        image = self.input()
        Image.fromarray(np.roll(np.asarray(image), -50, axis=0)).save(self.run.file("00-sketch.png"))
        record = stages.stage1_cover(self.run, self.client)
        self.assert_completion(record, ["figure_top_gap_px"])

    def test_cover_full_body_failure_uses_existing_plain_message(self):
        for scenario, feet in (("ok", "missing"), ("ok", "upper"), ("cut", "normal"),
                               ("blank", "normal"), ("two", "normal"), ("top", "normal")):
            with self.subTest(scenario=scenario, feet=feet):
                self.input("cut")
                self.cover_image = (Image.fromarray(np.roll(np.asarray(draw("ok")), -50, axis=0))
                                    if scenario == "top" else draw(scenario))
                self.cover_feet = feet
                self.calls.clear()
                self.client = ModelClient(self.root / f"cover-{scenario}-{feet}", mode="live", fast=True)
                record = self.reject("Show the whole character from head to feet;", cover_exists=True)
                self.assertNotIn("completed_figure", record["details"])
                self.assertEqual(self.calls.count("cover"), 1)
                self.assertCountEqual(self.calls[4:], ["character", "feet"])

    def test_full_body_upload_still_requires_full_body_cover(self):
        self.input()
        self.cover_feet = "missing"
        record = self.reject("Show the whole character from head to feet;", cover_exists=True)
        self.assertEqual(record["details"]["prompt"], stages.COVER_PROMPT)
        self.assertEqual(record["details"]["upload_failed_checks"], [])
        self.assertNotIn("completed_figure", record["details"])
        self.assertCountEqual(self.calls[4:], ["character", "feet"])

    def test_completion_preserves_redo_reference_and_appearance_note(self):
        self.input("cut")
        reference = draw("robot")
        reference.save(self.run.file("redo/parent-cover.png"))
        self.run.write_json("redo.json", {"from_stage": 1, "note": "green coat"})
        record = stages.stage1_cover(self.run, self.client)
        request = next(b for label, b in self.requests if label == "cover")
        self.assertEqual(len(request["images"]), 1)
        np.testing.assert_array_equal(np.asarray(flow._decode_data_url(request["images"][0]["image_url"])),
                                      np.asarray(reference))
        expected = (prompts.selected("cover_complete_figure").step("cover")["prompt"] +
                    prompts.selected("cover_redo").data["note_slot"].replace("{note}", "green coat"))
        self.assertEqual(record["details"]["prompt"], expected)
        self.assertEqual(request["prompt"], expected)

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
            elif not passed.is_set():
                barrier.wait(timeout=5)
        self.before = before
        self.after = lambda label: passed.set() if label == "photo" else None
        record = stages.stage1_cover(self.run, self.client)
        self.assertTrue(record["ok"])
        self.assertCountEqual(self.calls[:3], ["character", "feet", "photo"])
        self.assertEqual(self.calls[3:4], ["cover"])
        self.assertCountEqual(self.calls[4:], ["character", "feet"])
        self.assertEqual(record["details"]["prompt"], stages.COVER_PROMPT)
        self.assertNotIn("prompt_version", record["details"])
        self.assertNotIn("completed_figure", record["details"])

    def test_non_fast_recorded_call_sequence_and_cover_wording_unchanged(self):
        for scenario in ("ok", "cut"):
            with self.subTest(scenario=scenario):
                self.input(scenario)
                self.calls.clear()
                self.requests.clear()
                client = ModelClient(self.root / f"full-{scenario}", mode="live", fast=False)
                record = stages.stage1_cover(self.run, client)
                self.assertTrue(record["ok"])
                self.assertEqual(self.calls, ["photo", "character", "foot", "face", "head", "cover",
                                              "character"])
                request = next(b for label, b in self.requests if label == "cover")
                self.assertEqual(request["prompt"], stages.COVER_PROMPT)
                np.testing.assert_array_equal(
                    np.asarray(flow._decode_data_url(request["images"][0]["image_url"])),
                    np.asarray(self.upload_image))
                self.assertNotIn("completed_figure", record["details"])
                self.assertNotIn("upload_failed_checks", record["details"])
                self.assertNotIn("prompt_version", record["details"])

    def test_prompt_crop_still_rejects_without_completion_or_cover_checks(self):
        self.input("cut", generated=True)
        self.reject("Show the whole character from head to feet;")
        self.assertCountEqual(self.calls, ["character", "feet", "cover"])
        request = next(b for label, b in self.requests if label == "cover")
        self.assertTrue(request["prompt"].startswith(stages.COVER_PROMPT))

    def test_completion_cover_checks_propagate_model_errors_and_drain(self):
        for label, failure in (("character", ReplayMiss("cover character missing")),
                               ("feet", ApiError("cover feet failed"))):
            with self.subTest(label=label):
                self.input("cut")
                self.calls.clear()
                self.failures = {}
                self.client = ModelClient(self.root / f"failure-cover-{label}", mode="live", fast=True)
                self.after = lambda done: self.failures.update({label: failure}) if done == "cover" else None
                with self.assertRaises(type(failure)):
                    stages.stage1_cover(self.run, self.client)
                record = self.run.read_stage(1)
                self.assertEqual(record["status"], "error")
                self.assertEqual(len(record["calls"]), 6)
                self.assertCountEqual(self.calls[4:], ["character", "feet"])
                self.assertNotIn("completed_figure", record["details"])

    def test_cli_completes_cut_off_upload_and_continues_only_when_cover_passes(self):
        self.input("cut")
        for scenario, expected in (("ok", 0), ("cut", 3)):
            with self.subTest(scenario=scenario):
                self.cover_image = draw(scenario)
                self.calls.clear()
                with mock.patch("herogen.pipeline.__main__.stage2_turnaround") as stage2, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    stage2.return_value = {"name": "turnaround", "seconds": 0, "cost_usd": 0}
                    code = run_pipeline(self.root / "input.png", f"cli-{scenario}", [0, 1, 2],
                                        runs=str(self.root), cache=str(self.root / f"cache-cli-{scenario}"),
                                        mode="live", fast=True)
                self.assertEqual(code, expected)
                self.assertEqual(stage2.call_count, int(expected == 0))
                self.assertEqual(self.calls.count("cover"), 1)

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
