"""Gate-ordered turnaround calls preserve serial outcomes, failures, replay and spend."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from herogen import prompts
from herogen.pipeline import stages
from herogen.pipeline.client import ApiError, ClientError, IMAGE_MODEL, SAM_MODEL, ModelClient, ReplayMiss
from herogen.pipeline.runfolder import RunFolder
from herogen.service.pipestage import prompt_error_exit


class RecordingClient(ModelClient):
    def __init__(self, root, figures=(4,), errors=None):
        super().__init__(root)
        self.figures = figures
        self.errors = errors or {}
        self.sent = []

    def edit_image(self, prompt, images, *, size="1024x1024", attempt=None):
        self.sent.append((attempt, prompt, size))
        error = self.errors.get(attempt)
        self.log.add(IMAGE_MODEL, "image_edit", f"image-{attempt}", cached=False,
                     **({"outcome": "unknown"} if error else {}))
        if error:
            raise error
        if attempt is None:
            image = Image.new("RGB", (64, 64), (235, 235, 235))
            ImageDraw.Draw(image).ellipse((20, 20, 44, 44), fill=(40, 50, 70))
        else:
            image = Image.new("RGB", (400, 160), (235, 235, 235))
            draw = ImageDraw.Draw(image)
            count = self.figures[min(attempt, len(self.figures) - 1)]
            for i in range(count):
                cx = (i * 400 + 200) // count
                draw.rectangle((cx - 18, 24, cx + 18, 136), fill=(40, 50, 70 + attempt))
        return image


def figure(client, image):
    mask = (np.asarray(image) < 200).any(axis=-1)
    client.log.add(SAM_MODEL, "segment", f"sam-{image.size}-{hashlib.sha256(image.tobytes()).hexdigest()}",
                   cached=False)
    return mask, "sam-3.1", len(stages.imaging.column_runs(mask))


class TurnaroundConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.next_id = 0
        self.addCleanup(patch.stopall)
        self.original_figure = stages.sam_figure_mask
        patch.object(stages, "sam_figure_mask", side_effect=figure).start()
        patch.object(stages, "carve_check", return_value=(
            {"name": stages.CARVE_GATE, "pass": True, "value": .95, "gate": True}, {})).start()
        def empty_parts(run, client):
            run.write_json("masks/parts.json", {})
            return {}

        patch("herogen.pipeline.parts.write_part_masks", side_effect=empty_parts).start()

    def run_stage(self, concurrency, *, figures=(4,), errors=None, note=""):
        self.next_id += 1
        run = RunFolder(self.root, f"r{self.next_id}")
        Image.new("RGB", (64, 96), "white").save(run.file("01-cover.png"))
        if note:
            run.write_json("redo.json", {"from_stage": 2, "note": note})
        client = RecordingClient(self.root / f"cache{self.next_id}", figures, errors)
        caught = None
        try:
            record = stages.stage2_turnaround(run, client, attempts=4, concurrency=concurrency)
        except Exception as error:
            caught = error
            record = run.read_stage(2) if run.exists("stage-2.json") else None
        return run, client, record, caught

    def assert_same(self, serial, parallel):
        self.assertEqual(serial[1].sent, parallel[1].sent)
        self.assertEqual(serial[1].log.calls, parallel[1].log.calls)
        self.assertEqual(serial[1].log.cost_usd(), parallel[1].log.cost_usd())
        a, b = serial[2], parallel[2]
        if a is None:
            self.assertIsNone(b)
        else:
            a, b = dict(a), dict(b)
            for record in (a, b):
                record.pop("started")
                record.pop("seconds")
            self.assertEqual(a, b)
        self.assertIs(serial[3], parallel[3])

    def test_early_acceptance_never_sends_unused_candidates(self):
        for error in (ApiError("unused timeout", transient=True), ReplayMiss("unused cache miss")):
            with self.subTest(error=error):
                serial = self.run_stage(1, errors={3: error})
                parallel = self.run_stage(4, errors={3: error})
                self.assert_same(serial, parallel)
                self.assertIsNone(parallel[3])
                self.assertTrue(parallel[2]["ok"])
                self.assertEqual([a for a, _, _ in parallel[1].sent], [0, None])
                self.assertEqual(parallel[2]["cost_usd"], .025)

    def test_ineligible_sheets_never_send_or_charge_a_top(self):
        for error in (ApiError("unused top", http_status=503, transient=True), ReplayMiss("unused top")):
            with self.subTest(error=error):
                serial = self.run_stage(1, figures=(3,), errors={None: error})
                parallel = self.run_stage(4, figures=(3,), errors={None: error})
                self.assertIsInstance(serial[3], stages.GateFailed)
                self.assertIsInstance(parallel[3], stages.GateFailed)
                # GateFailed carries each run's own record; compare the measured records instead.
                serial = (*serial[:3], None)
                parallel = (*parallel[:3], None)
                self.assert_same(serial, parallel)
                self.assertEqual([a for a, _, _ in parallel[1].sent], [0, 1, 2, 3])
                self.assertEqual(parallel[2]["cost_usd"], .05)

    def test_failed_first_sheet_reaches_second_before_any_top(self):
        primary = ApiError("refused", http_status=400, error_code="content_policy_violation", transient=False)
        secondary = ApiError("unused top overload", http_status=503, transient=True)
        serial = self.run_stage(1, figures=(3,), errors={1: primary, None: secondary})
        parallel = self.run_stage(4, figures=(3,), errors={1: primary, None: secondary})
        self.assert_same(serial, parallel)
        self.assertIs(parallel[3], primary)
        self.assertIsNone(primary.__cause__)
        self.assertEqual(prompt_error_exit(primary), 4)
        self.assertEqual([a for a, _, _ in parallel[1].sent], [0, 1])
        self.assertEqual(parallel[2]["cost_usd"], .0225)
        self.assertIn("refused", parallel[2]["error"])

    def test_local_failures_do_not_send_later_calls_or_acquire_transient_causes(self):
        for primary in (ClientError("local client"), prompts.RegistryError("local registry"), ValueError("local state")):
            with self.subTest(primary=primary):
                fields = vars(primary).copy()
                errors = {None: ApiError("unused overload", http_status=503, transient=True),
                          1: ReplayMiss("unused candidate")}
                with patch.object(stages, "split_sheet", side_effect=primary):
                    serial = self.run_stage(1, errors=errors)
                    parallel = self.run_stage(4, errors=errors)
                self.assert_same(serial, parallel)
                self.assertIs(parallel[3], primary)
                self.assertEqual(vars(primary), fields)
                self.assertIsNone(primary.__cause__)
                self.assertEqual(prompt_error_exit(primary), 1)
                self.assertEqual([a for a, _, _ in parallel[1].sent], [0])

    def test_every_reached_model_failure_propagates_with_original_fields_and_spend(self):
        for error in (ApiError("temporary", http_status=503, transient=True),
                      ApiError("refused", http_status=400, error_code="content_policy_violation"),
                      ApiError("HTTP 503 content_policy_violation", http_status=400), ReplayMiss("cache miss")):
            for source in ("sheet", "top", "sheet-mask", "top-mask"):
                with self.subTest(error=error, source=source):
                    fields = vars(error).copy()
                    errors = {0: error} if source == "sheet" else {None: error} if source == "top" else {}

                    def failing_figure(client, image):
                        if source == "sheet-mask" and image.width == 400 or source == "top-mask" and image.width == 64:
                            raise error
                        return figure(client, image)

                    with patch.object(stages, "sam_figure_mask", side_effect=failing_figure):
                        serial = self.run_stage(1, errors=errors)
                        parallel = self.run_stage(4, errors=errors)
                    self.assert_same(serial, parallel)
                    self.assertIs(parallel[3], error)
                    self.assertEqual(vars(error), fields)
                    self.assertEqual(parallel[2]["status"], "error")
                    self.assertFalse(parallel[0].exists("02-turnaround.png"))

    def test_retry_selection_and_output_bytes_match_serial(self):
        serial = self.run_stage(1, figures=(3, 3, 4))
        parallel = self.run_stage(4, figures=(3, 3, 4))
        self.assert_same(serial, parallel)
        self.assertIsNone(parallel[3])
        self.assertEqual(parallel[2]["details"]["kept_attempt"], 2)
        self.assertEqual([a for a, _, _ in parallel[1].sent], [0, 1, 2, None])
        for output in serial[2]["outputs"]:
            self.assertEqual(serial[0].file(output).read_bytes(), parallel[0].file(output).read_bytes())

    def test_redo_note_selected_version_and_parameters_reach_calls_and_record(self):
        note = "green coat"
        data = copy.deepcopy(prompts.selected("turnaround_redo").data)
        data["name"] = "turnaround-redo-v2"
        data["steps"][0]["prompt"] += " Keep the coat fastened."
        data["steps"][0]["params"]["size"] = "2048x1024"
        data["sha256"] = prompts.content_hash(data)
        versions = self.root / "versions"
        versions.mkdir()
        (versions / "turnaround-redo-v2.json").write_text(json.dumps(data))
        with patch.object(prompts, "VERSIONS", versions), \
                patch.dict("os.environ", {"HEROGEN_TURNAROUND_REDO_VERSION": "turnaround-redo-v2"}):
            serial = self.run_stage(1, note=note)
            parallel = self.run_stage(4, note=note)
        self.assert_same(serial, parallel)
        self.assertIsNone(parallel[3])
        sent = parallel[1].sent[0]
        self.assertIn(note, sent[1])
        self.assertEqual(sent[2], "2048x1024")
        self.assertEqual(parallel[2]["details"]["prompt"], sent[1])
        self.assertEqual(parallel[2]["details"]["prompt_version"], {"name": data["name"], "sha256": data["sha256"]})

    def test_cli_retains_concurrency_argument_for_existing_callers(self):
        from herogen.pipeline import __main__ as cli
        with patch.object(cli, "run_pipeline", return_value=0) as run:
            self.assertEqual(cli.main(["run", "--run-id", "r", "--stages", "2"]), 0)
            self.assertEqual(run.call_args.kwargs["turnaround_concurrency"], 4)
            self.assertEqual(cli.main(["run", "--run-id", "r", "--stages", "2", "--turnaround-concurrency", "1"]), 0)
            self.assertEqual(run.call_args.kwargs["turnaround_concurrency"], 1)

    def test_redo_replays_across_concurrency_settings_without_extra_cache_entries(self):
        from fake_model_server import FakeModelServer
        from herogen.pipeline import client as client_module
        from tests.test_herogen_pipeline_flow import DUMMY_KEY, FakeAPI

        # Use the real client cache and SAM decoding path; the API draws its own synthetic inputs.
        for recorded_width, replay_width in ((1, 4), (4, 1)):
            with self.subTest(recorded_width=recorded_width):
                api = FakeAPI()
                cache = self.root / f"replay-cache-{recorded_width}"
                runs = [RunFolder(self.root, f"replay-{recorded_width}-{width}") for width in (1, 4)]
                for run in runs:
                    Image.new("RGB", (64, 96), "white").save(run.file("01-cover.png"))
                    run.write_json("redo.json", {"from_stage": 2, "note": "green coat"})
                before_calls, after_calls = [], []
                client = ModelClient(cache, mode="live", before_api_attempt=before_calls.append)
                client.after_api_attempt = after_calls.append
                server = FakeModelServer(lambda e, b: (200, {}, api.post(None, e, b)), DUMMY_KEY)
                with server.serving(str(cache / "limiter")), \
                        patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                        patch.object(client_module, "decode_sam_output", api.decode), \
                        patch.object(stages, "sam_figure_mask", wraps=self.original_figure):
                    recorded = stages.stage2_turnaround(runs[0], client, concurrency=recorded_width)
                with patch.object(ModelClient, "_post", side_effect=AssertionError("network during replay")), \
                        patch.object(stages, "sam_figure_mask", wraps=self.original_figure):
                    replayed = stages.stage2_turnaround(runs[1], ModelClient(cache), concurrency=replay_width)
                self.assertTrue(recorded["ok"])
                self.assertTrue(replayed["ok"])
                self.assertEqual([c["key"] for c in recorded["calls"]], [c["key"] for c in replayed["calls"]])
                self.assertTrue(all(c["cached"] for c in replayed["calls"]))
                self.assertEqual(replayed["cost_usd"], 0)
                self.assertEqual(recorded["details"], replayed["details"])
                self.assertEqual(len(before_calls), len(after_calls))
                self.assertGreater(len(before_calls), 0)

    def test_invalid_limits_are_rejected_before_work(self):
        run = RunFolder(self.root, "invalid")
        client = ModelClient(self.root / "cache")
        for kwargs in ({"concurrency": 0}, {"concurrency": True}, {"attempts": 0}):
            with self.assertRaisesRegex(ValueError, "positive integers"):
                stages.stage2_turnaround(run, client, **kwargs)
        self.assertEqual(client.log.calls, [])


if __name__ == "__main__":
    unittest.main()
