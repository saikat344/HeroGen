"""Call receipts and report arithmetic without live model or GPU work."""
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import urllib.error
from unittest import mock

from PIL import Image
from herogen.pipeline import client as client_module
from herogen.pipeline.client import ApiError, ModelClient, ReplayMiss
from herogen.pipeline.report import report
from herogen.pipeline.runfolder import RunFolder
from herogen.blender.remote import client as remote_module


class CallReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_env_switch_stops_after_one_recorded_transport_attempt(self):
        limiter = mock.Mock()
        limiter.acquire.return_value = ("slot", 0)
        from herogen.pipeline.retry import RetryPolicy
        with mock.patch.dict(os.environ, {"HEROGEN_FAST_LANE": "1", "MODEL_API_KEY": "placeholder"}):
            client = ModelClient(self.root / "cache", mode="live", limiter=limiter,
                                 policy=RetryPolicy(max_attempts=8, sleep=lambda s: None, base_s=0))
            self.assertEqual(client.policy.max_attempts, 1)
            with mock.patch.object(client_module, "_open", side_effect=urllib.error.HTTPError(
                    "url", 429, "busy", {}, None)) as send:
                with self.assertRaises(ApiError) as raised:
                    client.generate_image("a knight")
        self.assertFalse(raised.exception.transient)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(len(client.log.calls), 1)
        self.assertEqual([c["attempt"] for c in client.log.calls], [1])
        self.assertEqual(client.log.cost_usd(), 0)
        self.assertTrue(all(c["wall_seconds"] >= 0 for c in client.log.calls))

    def test_failed_answer_is_resent_once_and_keeps_both_unknown_billing_receipts(self):
        from herogen.pipeline.retry import RetryPolicy
        limiter = mock.Mock()
        limiter.acquire.return_value = ("slot", 0)
        client = ModelClient(self.root / "cache", mode="live", fast=True, limiter=limiter,
                             policy=RetryPolicy(max_attempts=8, sleep=lambda s: None, base_s=0))
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200
        response.__enter__.return_value.read.return_value = json.dumps(
            {"status": "failed", "error": {"code": "server_error"}}).encode()
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": "placeholder"}), \
                mock.patch.object(client_module, "_open", return_value=response) as send:
            with self.assertRaises(ApiError) as raised:
                client.generate_image("a knight")
        # A "failed" answer with code server_error is a 500: the fast lane resends it once, and no more.
        self.assertFalse(raised.exception.transient)
        self.assertEqual(raised.exception.http_status, 500)
        self.assertEqual(send.call_count, 2)
        self.assertEqual([(c["attempt"], c.get("resend")) for c in client.log.calls], [(1, None), (2, True)])
        self.assertEqual(client.log.cost_usd(), 2 * client_module.PRICE_PER_IMAGE_USD[client_module.IMAGE_MODEL])
        rows = [json.loads(line) for line in (self.root / "cache/calls-ledger.jsonl").read_text().splitlines()]
        complete = [r for r in rows if r["outcome"] != "sent"]
        self.assertEqual(len(complete), 2)
        self.assertTrue(all(r["unknown"] for r in complete))

    def test_cached_calls_and_replay_misses_have_receipts(self):
        client = ModelClient(self.root / "cache", fast=True)
        image = Image.new("RGB", (8, 8))
        with self.assertRaises(ReplayMiss):
            client.segment(image, "face")
        self.assertEqual(client.log.calls[-1]["outcome"], "ReplayMiss")
        self.assertEqual(client.log.cost_usd(), 0)
        with mock.patch.object(client, "_load", return_value={"instances": []}):
            self.assertEqual(client.segment(image, "face"), [])
        self.assertTrue(client.log.calls[-1]["cached"])
        self.assertIn("segment face", client.log.calls[-1]["purpose"])

    def test_report_sums_all_stages_and_orders_cached_and_failed_calls(self):
        run = RunFolder(self.root, "hero")
        for stage, start, seconds, calls in [
            (3, "2026-11-01T01:00:02+00:00", 7, [
                {"model": "sam-3d-objects", "kind": "gpu", "purpose": "shape", "cached": False,
                 "started": "2026-11-01T01:00:02+00:00", "wall_seconds": 2, "cost_usd": .05,
                 "outcome": "RemoteError"}]),
            (1, "2026-11-01T01:00:01+00:00", 4, [
                {"model": "muse-image-1.0", "kind": "edit", "purpose": "cover art", "cached": True,
                 "started": "2026-11-01T01:00:01+00:00", "wall_seconds": .2}]),
            (5, "2026-11-01T01:00:03+00:00", 1, [])]:
            run.write_stage({"stage": stage, "name": "synthetic", "started": start,
                             "seconds": seconds, "calls": calls})
        output = io.StringIO()
        result = report(run.path, stream=output)
        self.assertEqual([c["stage"] for c in result["calls"]], [1, 3])
        self.assertEqual(result["total"], {"calls": 2, "call_seconds": 2.2, "seconds": 12,
                                          "wall_seconds": 12, "peak_concurrent_calls": 1,
                                          "cost_usd": .05, "unknown_cost_calls": 0})
        self.assertIn("RemoteError", output.getvalue())
        self.assertIn("TOTAL", output.getvalue())
        self.assertEqual(result, json.loads((run.path / "fast-report.json").read_text()))

    def test_parallel_calls_show_critical_path_sum_and_peak_without_endpoint_overlap(self):
        run = RunFolder(self.root, "parallel")
        run.write_stage({"stage": 2, "name": "parts", "seconds": 7, "calls": [
            {"model": "sam-3.1", "started": f"2026-11-01T01:00:0{start}+00:00",
             "wall_seconds": seconds, "cached": False}
            for start, seconds in ((0, 4), (1, 4), (4, 2))]})
        output = io.StringIO()
        data = report(run.path, stream=output)
        stage = data["stages"][0]
        self.assertEqual(stage["wall_seconds"], 7)
        self.assertEqual(stage["call_seconds"], 10)
        self.assertEqual(stage["peak_concurrent_calls"], 2)
        self.assertEqual(data["total"]["peak_concurrent_calls"], 2)
        self.assertIn("wall seconds  sum call seconds  peak concurrent calls", output.getvalue())
        self.assertIn("2  3  7.00  10.000000  2", output.getvalue())

    def test_sequential_retries_cache_hits_and_legacy_missing_timings(self):
        run = RunFolder(self.root, "sequential")
        run.write_stage({"stage": 1, "name": "cover", "seconds": 4, "calls": [
            {"model": "sam-3.1", "started": "2026-11-01T01:00:00+00:00", "wall_seconds": 1,
             "outcome": "failed", "cost_usd": 0},
            {"model": "sam-3.1", "started": "2026-11-01T01:00:02+00:00", "wall_seconds": 1},
            {"model": "sam-3.1", "started": "2026-11-01T01:00:03+00:00", "wall_seconds": .1,
             "cached": True}]})
        run.write_stage({"stage": 0, "name": "legacy", "calls": [
            {"model": "sam-3.1"},
            {"model": "sam-3.1", "started": "unavailable", "wall_seconds": 0}]})
        data = report(run.path, stream=io.StringIO())
        by_stage = {s["stage"]: s for s in data["stages"]}
        self.assertEqual(by_stage[1]["peak_concurrent_calls"], 1)
        self.assertEqual(by_stage[0]["peak_concurrent_calls"], 0)
        self.assertEqual(data["total"]["peak_concurrent_calls"], 1)

    def test_report_labels_unknown_gpu_prices(self):
        run = RunFolder(self.root, "unknown")
        run.write_stage({"stage": 3, "name": "shape", "calls": [
            {"model": "sam-3d-objects", "cost_usd": None, "cached": False}]})
        output = io.StringIO()
        data = report(run.path, stream=output)
        self.assertEqual(data["total"]["unknown_cost_calls"], 1)
        self.assertIn("known costs only", output.getvalue())

    def test_timed_out_gpu_call_has_one_failure_receipt(self):
        image = self.root / "image.png"
        Image.new("RGB", (8, 8)).save(image)
        remote = remote_module.RemoteClient(self.root / "remote", mode="live", timeout=1, fast=True)
        with mock.patch.object(remote_module, "find_modal", return_value="synthetic"), \
                mock.patch.object(remote_module.subprocess, "run", side_effect=subprocess.TimeoutExpired(
                    "synthetic", 1)) as send:
            with self.assertRaises(remote_module.RemoteError):
                remote.objects(image, image)
        send.assert_called_once()
        self.assertEqual(len(remote.log.calls), 1)
        self.assertEqual(remote.log.calls[0]["outcome"], "RemoteError")
        self.assertEqual(remote.log.calls[0]["kind"], "gpu")
        self.assertGreaterEqual(remote.log.calls[0]["cost_usd"],
                                round(remote_module.PRICE_PER_SECOND[remote.objects_gpu] * remote.timeout, 4))
        self.assertGreaterEqual(remote.log.calls[0]["wall_seconds"], 0)
