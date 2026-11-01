"""Bounded turnaround speculation, ordered consumption and loud background failures, without network."""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from herogen.pipeline import stages
from herogen.pipeline.client import ApiError, IMAGE_MODEL, SAM_MODEL, ModelClient, ReplayMiss, SamInstance
from herogen.pipeline.runfolder import RunFolder


class ConcurrentClient(ModelClient):
    def __init__(self, root, error=None):
        super().__init__(root)
        self.error = error
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.state = {"active": 0, "peak": 0, "finished": []}

    def edit_image(self, prompt, images, *, size="1024x1024", attempt=0):
        top = prompt == stages.TOP_PROMPT
        with self.lock:
            self.state["active"] += 1
            self.state["peak"] = max(self.state["peak"], self.state["active"])
        try:
            if top or attempt == 0:
                self.barrier.wait(timeout=5)
            self.log.add(IMAGE_MODEL, "edit", "top" if top else str(attempt), cached=False)
            if attempt == 3 and not top and self.error:
                raise self.error("candidate failed")
            return Image.new("RGB", (32, 32), (attempt, 128 if top else 0, 0))
        finally:
            with self.lock:
                self.state["active"] -= 1
                self.state["finished"].append("top" if top else attempt)


class TurnaroundConcurrencyTest(unittest.TestCase):
    def requests(self, client, cover, attempts, concurrency):
        step = stages.prompt_registry.selected("turnaround").step("sheet")
        return stages._TurnaroundRequests(client, cover, attempts, concurrency,
                                          prompt=step["prompt"], size=step["params"]["size"])

    def test_bounded_overlap_and_ordered_logs(self):
        with tempfile.TemporaryDirectory() as root:
            client = ConcurrentClient(root)
            cover = Image.new("RGB", (32, 32))
            mask = np.ones((32, 32), bool)
            with patch.object(stages, "sam_figure_mask", return_value=(mask, "sam-3.1", 1)), \
                    patch.object(stages.imaging, "complete_attached_foreground", return_value=mask), \
                    self.requests(client, cover, 4, 2) as requests:
                for i in range(4):
                    sheet, _ = requests.sheet(i)
                    self.assertEqual(sheet.getpixel((0, 0))[0], i)
                    requests.top()
            self.assertEqual(client.state["peak"], 2)
            self.assertEqual(client.state["active"], 0)
            self.assertEqual(len(client.state["finished"]), 5)
            self.assertEqual([c["key"] for c in client.log.calls],
                             ["0", "top", "1", "top", "2", "top", "3", "top"])
            self.assertEqual(client.log.cost_usd(), .05)

    def test_cached_top_reuse_does_not_replay_the_first_calls_retry_history(self):
        with tempfile.TemporaryDirectory() as root:
            client = ConcurrentClient(root)
            mask = np.ones((32, 32), bool)

            def figure(branch, image):
                if image.getpixel((0, 0))[1] == 128:
                    branch.log.add(SAM_MODEL, "segment", "top-mask", cached=False, outcome="unknown")
                    branch.log.add(SAM_MODEL, "segment", "top-mask", cached=False)
                return mask, "sam-3.1", 1

            with patch.object(stages, "sam_figure_mask", side_effect=figure), \
                    patch.object(stages.imaging, "complete_attached_foreground", return_value=mask), \
                    self.requests(client, Image.new("RGB", (32, 32)), 4, 2) as requests:
                for i in range(4):
                    requests.sheet(i)
                    requests.top()
            top_masks = [c for c in client.log.calls if c["key"] == "top-mask"]
            self.assertEqual(len(top_masks), 5)  # unknown + successful first call, then three cache hits
            self.assertEqual(sum(c.get("outcome") == "unknown" for c in top_masks), 1)
            self.assertEqual(sum(c["cached"] for c in top_masks), 3)
            self.assertFalse(any(c.get("outcome") for c in top_masks if c["cached"]))
            self.assertEqual(client.log.cost_usd(), .055)

    def test_identical_candidate_pixels_send_one_sam_request(self):
        class SameImageClient(ConcurrentClient):
            def edit_image(self, *args, **kwargs):
                super().edit_image(*args, **kwargs)
                return Image.new("RGB", (32, 32), "white")

        with tempfile.TemporaryDirectory() as root:
            client = SameImageClient(root)
            client.mode = "live"
            mask = np.ones((32, 32), bool)
            posts = []

            def post(branch, endpoint, body, *, model, kind, key, **kwargs):
                posts.append(key)
                threading.Event().wait(.05)  # Other workers reach the same initially absent SAM key.
                branch.log.add(model, kind, key, cached=False)
                return {"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": "mask"}]}]}

            with patch.object(ModelClient, "_post", post), \
                    patch("herogen.pipeline.client.decode_sam_output", return_value=[
                        SamInstance(mask, (0, 0, 32, 32), 1.0)]), \
                    patch.object(stages.imaging, "complete_attached_foreground", return_value=mask), \
                    self.requests(client, Image.new("RGB", (32, 32)), 4, 4) as requests:
                for i in range(4):
                    requests.sheet(i)
                requests.top()
            self.assertEqual(len(posts), 1)
            sam_calls = [c for c in client.log.calls if c["model"] == SAM_MODEL]
            self.assertEqual(len(sam_calls), 5)
            self.assertEqual(sum(c["cached"] for c in sam_calls), 4)
            self.assertEqual(client.log.cost_usd(), .0525)

    def test_unused_failed_candidate_is_never_ignored(self):
        for error in (ApiError, ReplayMiss):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as root:
                client = ConcurrentClient(root, error)
                mask = np.ones((32, 32), bool)
                with patch.object(stages, "sam_figure_mask", return_value=(mask, "sam-3.1", 1)), \
                        patch.object(stages.imaging, "complete_attached_foreground", return_value=mask):
                    with self.assertRaisesRegex(error, "candidate failed"):
                        with self.requests(client, Image.new("RGB", (32, 32)), 4, 2) as requests:
                            requests.sheet(0)  # Even an early accepted sheet must drain unused samples.
                self.assertEqual(client.state["active"], 0)
                self.assertEqual(len(client.log.calls), 5)

    def test_stage2_preserves_original_errors_from_every_concurrent_call(self):
        errors = (
            ApiError("temporary", http_status=503, transient=True),
            ApiError("refused", http_status=400, error_code="content_policy_violation"),
            ApiError("HTTP 503 content_policy_violation", http_status=400),
            ReplayMiss("missing result"),
        )
        for source in ("sheet", "top", "sheet-mask", "top-mask", "unused-sheet"):
            for error in errors:
                with self.subTest(source=source, error=error), tempfile.TemporaryDirectory() as root:
                    fields = vars(error).copy()
                    run = RunFolder(Path(root), "r")
                    Image.new("RGB", (32, 32)).save(run.file("01-cover.png"))
                    run.write_json("redo.json", {"from_stage": 2, "note": "green coat"})
                    prompt, version = stages.turnaround_prompt(run)
                    test = self

                    class FailingClient(ConcurrentClient):
                        def edit_image(self, sent_prompt, images, *, size="1024x1024", attempt=0):
                            image = super().edit_image(sent_prompt, images, size=size, attempt=attempt)
                            top = sent_prompt == stages.TOP_PROMPT
                            if not top:
                                test.assertEqual(sent_prompt, prompt)
                                test.assertEqual(size, version.step("sheet")["params"]["size"])
                            if (source == "sheet" and not top and attempt == 0 or
                                    source == "top" and top or
                                    source == "unused-sheet" and not top and attempt == 1):
                                raise error
                            return image

                    client = FailingClient(root)
                    mask = np.ones((32, 32), bool)

                    def figure(branch, image):
                        top = image.getpixel((0, 0))[1] == 128
                        if source == "top-mask" and top or source == "sheet-mask" and not top:
                            raise error
                        return mask, "sam-3.1", 1

                    with patch.object(stages, "sam_figure_mask", side_effect=figure), \
                            patch.object(stages.imaging, "complete_attached_foreground", return_value=mask), \
                            patch.object(stages, "split_sheet", return_value={
                                "views": {}, "runs": [], "sam_coverage": {}, "split_source": "sam"}):
                        with self.assertRaises(type(error)) as raised:
                            stages.stage2_turnaround(run, client, attempts=2, concurrency=2)
                    self.assertIs(raised.exception, error)
                    self.assertEqual(vars(raised.exception), fields)
                    self.assertEqual(client.state["peak"], 2)
                    self.assertEqual(client.state["active"], 0)
                    self.assertEqual(len(client.state["finished"]), 3)
                    self.assertEqual(len(client.log.calls), 3)
                    self.assertEqual(run.read_stage(2)["status"], "error")
                    self.assertFalse(run.exists("02-turnaround.png"))

    def test_stage2_raises_first_failure_in_serial_order_after_draining_all_calls(self):
        for first_source in ("sheet", "top"):
            with self.subTest(first_source=first_source), tempfile.TemporaryDirectory() as root:
                first = ApiError("first failed", http_status=503, transient=True)
                later = ReplayMiss("later failed")
                last_sheet = threading.Event()

                class FailingClient(ConcurrentClient):
                    def edit_image(self, prompt, images, *, size="1024x1024", attempt=0):
                        image = super().edit_image(prompt, images, size=size, attempt=attempt)
                        top = prompt == stages.TOP_PROMPT
                        if not top and attempt > 0:
                            if attempt == 3:
                                last_sheet.set()
                            raise later
                        if top == (first_source == "top"):
                            if not last_sheet.wait(timeout=5):
                                raise AssertionError("later calls did not finish before the first failure")
                            raise first
                        if top:
                            raise later
                        return image

                run = RunFolder(Path(root), "r")
                Image.new("RGB", (32, 32)).save(run.file("01-cover.png"))
                client = FailingClient(root)
                mask = np.ones((32, 32), bool)
                with patch.object(stages, "sam_figure_mask", return_value=(mask, "sam-3.1", 1)), \
                        patch.object(stages, "split_sheet", return_value={
                            "views": {}, "runs": [], "sam_coverage": {}, "split_source": "sam"}):
                    with self.assertRaises(ApiError) as raised:
                        stages.stage2_turnaround(run, client, attempts=4, concurrency=4)
                self.assertIs(raised.exception, first)
                self.assertEqual(client.state["active"], 0)
                self.assertEqual(len(client.state["finished"]), 5)
                self.assertEqual(len(client.log.calls), 5)
                self.assertEqual(run.read_stage(2)["status"], "error")

    def test_cli_defaults_to_four_and_can_request_serial_scheduling(self):
        from herogen.pipeline import __main__ as cli
        with patch.object(cli, "run_pipeline", return_value=0) as run:
            self.assertEqual(cli.main(["run", "--run-id", "r", "--stages", "2"]), 0)
            self.assertEqual(run.call_args.kwargs["turnaround_concurrency"], 4)
            self.assertEqual(cli.main(["run", "--run-id", "r", "--stages", "2",
                                       "--turnaround-concurrency", "1"]), 0)
            self.assertEqual(run.call_args.kwargs["turnaround_concurrency"], 1)

    def test_invalid_limits_and_single_request_budget_callbacks_are_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunFolder(Path(root), "r")
            client = ModelClient(Path(root) / "cache")
            for kwargs in ({"concurrency": 0}, {"concurrency": True}, {"attempts": 0}):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    stages.stage2_turnaround(run, client, **kwargs)
            client.before_api_attempt = lambda model: None
            with self.assertRaisesRegex(ValueError, "per-attempt callbacks"):
                stages.stage2_turnaround(run, client, concurrency=2)
            self.assertEqual(client.log.calls, [])


if __name__ == "__main__":
    unittest.main()
