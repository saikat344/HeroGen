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
from herogen.pipeline.client import ApiError, IMAGE_MODEL, ModelClient, ReplayMiss
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
    def test_bounded_overlap_and_ordered_logs(self):
        with tempfile.TemporaryDirectory() as root:
            client = ConcurrentClient(root)
            cover = Image.new("RGB", (32, 32))
            mask = np.ones((32, 32), bool)
            with patch.object(stages, "sam_figure_mask", return_value=(mask, "sam-3.1", 1)), \
                    patch.object(stages.imaging, "complete_attached_foreground", return_value=mask), \
                    stages._TurnaroundRequests(client, cover, 4, 2) as requests:
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

    def test_unused_failed_candidate_is_never_ignored(self):
        for error in (ApiError, ReplayMiss):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as root:
                client = ConcurrentClient(root, error)
                mask = np.ones((32, 32), bool)
                with patch.object(stages, "sam_figure_mask", return_value=(mask, "sam-3.1", 1)), \
                        patch.object(stages.imaging, "complete_attached_foreground", return_value=mask):
                    with self.assertRaisesRegex(error, "candidate failed"):
                        with stages._TurnaroundRequests(client, Image.new("RGB", (32, 32)), 4, 2) as requests:
                            requests.sheet(0)  # Even an early accepted sheet must drain unused samples.
                self.assertEqual(client.state["active"], 0)
                self.assertEqual(len(client.log.calls), 5)

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
