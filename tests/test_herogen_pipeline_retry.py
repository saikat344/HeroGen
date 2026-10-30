"""The Model API retry policy and the shared per-key limiter, through a local fake server and the real client."""

from __future__ import annotations

import email.utils
import http.client
import json
import math
import os
import random
import ssl
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import client as client_mod  # noqa: E402
from herogen.pipeline import retry  # noqa: E402
from herogen.pipeline.client import ApiError, ModelClient  # noqa: E402
from fake_model_server import FakeModelServer, error_body, sam_response  # noqa: E402
from test_herogen_pipeline_flow import DUMMY_KEY, _b64png  # noqa: E402

IMAGE = Image.new("RGB", (16, 16), "white")
OK_SAM = (200, {}, sam_response(""))
OK_IMAGE = (200, {}, {"data": [{"b64_json": _b64png(IMAGE)}]})


class FakeTime:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self, wall: float = 1_800_000_000.0) -> None:
        self.t = 0.0
        self.wall0 = wall
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def wall(self) -> float:
        return self.wall0 + self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s

    def pass_time(self, s: float) -> None:
        """The limiter's sleep: time passes, not counted as a retry wait."""
        self.t += s


class Script:
    """A responder that answers request n with ``answers[n]`` (the last one repeats)."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.n = 0

    def __call__(self, endpoint: str, body: dict):
        a = self.answers[min(self.n, len(self.answers) - 1)]
        self.n += 1
        return a


def err(status: int, **headers):
    return status, {k.replace("_", "-"): str(v) for k, v in headers.items()}, error_body(status)


class RetryHeaderTest(unittest.TestCase):
    def test_retryable_statuses(self):
        for s in (408, 409, 425, 429, 500, 502, 503, 504, 599, None):
            self.assertTrue(retry.retryable(s), s)
        for s in (400, 401, 403, 404, 405, 410, 413, 422, 501, 505):
            self.assertFalse(retry.retryable(s), s)

    def test_retry_after_seconds_or_http_date(self):
        now = 1_800_000_000.0
        self.assertEqual(retry.retry_after_s({"Retry-After": "7"}, now), 7.0)
        self.assertEqual(retry.retry_after_s({"retry-after": "2.5"}, now), 2.5)
        date = email.utils.formatdate(now + 30, usegmt=True)
        self.assertAlmostEqual(retry.retry_after_s({"Retry-After": date}, now), 30.0, delta=1.0)
        past = email.utils.formatdate(now - 30, usegmt=True)
        self.assertEqual(retry.retry_after_s({"Retry-After": past}, now), 0.0)
        self.assertIsNone(retry.retry_after_s({"Retry-After": "soon"}, now))
        self.assertIsNone(retry.retry_after_s({}, now))

    def test_reset_headers_take_the_largest(self):
        now = 1_800_000_000.0
        h = {"x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "6m0s", "retry-after-ms": "250"}
        self.assertEqual(retry.reset_wait_s(h, now), 360.0)
        self.assertEqual(retry.reset_wait_s({"X-RateLimit-Reset": str(now + 12)}, now), 12.0)  # an epoch time
        self.assertEqual(retry.reset_wait_s({"RateLimit-Reset": "4"}, now), 4.0)
        self.assertEqual(retry.reset_wait_s({"retry-after-ms": "1500"}, now), 1.5)
        self.assertEqual(retry.reset_wait_s({"x-ratelimit-reset-tokens": "20ms"}, now), 0.02)
        self.assertIsNone(retry.reset_wait_s({"x-ratelimit-reset": "later"}, now))
        self.assertIsNone(retry.reset_wait_s({}, now))

    def test_wait_prefers_retry_after_then_reset_then_full_jitter(self):
        p = retry.RetryPolicy(rand=random.Random(1), wall=lambda: 1_800_000_000.0)
        w, why = p.wait(0, {"Retry-After": "5", "x-ratelimit-reset": "50"})
        self.assertTrue(5.0 <= w < 6.0 and why == "Retry-After")
        w, why = p.wait(0, {"x-ratelimit-reset": "50", "ratelimit-reset": "9"})
        self.assertTrue(50.0 <= w < 51.0 and why == "rate-limit reset header")
        for n in range(12):
            cap = min(60.0, 2.0 * 2 ** n)
            waits = [p.wait(n, {})[0] for _ in range(400)]
            self.assertTrue(all(0.0 <= x <= cap for x in waits), n)
            self.assertGreater(max(waits), 0.9 * cap)  # full jitter spans the whole range
            self.assertLess(min(waits), 0.1 * cap)


class RetryThroughServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.time = FakeTime()
        self.policy = retry.RetryPolicy(sleep=self.time.sleep, clock=self.time.clock, wall=self.time.wall,
                                        rand=random.Random(7))
        self.limiter = retry.SharedLimiter(Path(self.tmp.name) / "limiter", "k", clock=self.time.wall,
                                           sleep=self.time.pass_time)
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def _call(self, responder, *, muse: bool = False):
        server = FakeModelServer(responder, DUMMY_KEY)
        self.calls += 1  # a fresh cache each call, so nothing is answered from an earlier one
        self.cache = Path(self.tmp.name) / f"cache{self.calls}"
        client = ModelClient(self.cache, mode="live", policy=self.policy, limiter=self.limiter)
        with server.serving(self.tmp.name + "/limiter"), mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                self.assertLogs("herogen.pipeline.client", "WARNING") as logs:
            client_mod.log.warning("start")
            try:
                out = client.generate_image("p") if muse else client.segment(IMAGE, "character")
                error = None
            except ApiError as e:
                out, error = None, e
        self.server, self.client, self.logs = server, client, logs.output[1:]
        for line in logs.output + [self.ledger_text()]:
            self.assertNotIn(DUMMY_KEY, line)
            self.assertNotIn("earer", line)
        self.assertTrue(all(h["auth_ok"] == "True" for _, h in server.requests))
        return out, error

    def ledger_text(self) -> str:
        p = self.cache / client_mod.LEDGER
        return p.read_text() if p.exists() else ""

    def ledger(self) -> list[dict]:
        return [json.loads(x) for x in self.ledger_text().splitlines()]

    def test_retry_after_seconds_is_honoured(self):
        out, e = self._call(Script(err(429, Retry_After=3), OK_SAM))
        self.assertIsNone(e)
        self.assertEqual(out, [])
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(len(self.time.sleeps), 1)
        self.assertTrue(3.0 <= self.time.sleeps[0] < 4.0)
        rows = self.ledger()
        self.assertEqual([(r["attempt"], r["status"], r["outcome"]) for r in rows], [(1, 429, "failed"), (2, 200, "ok")])
        self.assertEqual(rows[0]["wait_reason"], "Retry-After")
        self.assertEqual(len(self.logs), 1)
        self.assertIn("attempt 1/8 failed (HTTP 429", self.logs[0])
        self.assertIn("Retry-After", self.logs[0])

    def test_retry_after_http_date_is_honoured(self):
        date = email.utils.formatdate(self.time.wall() + 20, usegmt=True)
        _, e = self._call(Script(err(503, Retry_After=date), OK_SAM))
        self.assertIsNone(e)
        self.assertTrue(19.0 <= self.time.sleeps[0] < 21.0, self.time.sleeps)

    def test_largest_rate_limit_reset_header(self):
        _, e = self._call(Script(err(429, x_ratelimit_reset_requests="2s", x_ratelimit_reset_tokens="1m30s"),
                                 OK_SAM))
        self.assertIsNone(e)
        self.assertTrue(90.0 <= self.time.sleeps[0] < 91.0, self.time.sleeps)

    def test_retryable_statuses_are_retried(self):
        for status in (408, 409, 425, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                _, e = self._call(Script(err(status), err(status), OK_SAM))
                self.assertIsNone(e)
                self.assertEqual(len(self.server.requests), 3)

    def test_other_4xx_and_501_are_never_retried(self):
        for status in (400, 401, 403, 404, 413, 422, 501):
            with self.subTest(status=status):
                _, e = self._call(Script(err(status), OK_SAM))
                self.assertIsNotNone(e)
                self.assertIn(f"HTTP {status}", str(e))
                self.assertIn("not retried", str(e))
                self.assertEqual(len(self.server.requests), 1)
                self.assertEqual(self.logs, [])

    def test_at_most_eight_attempts(self):
        _, e = self._call(Script(err(500)))
        self.assertIn("failed after 8 attempts: HTTP 500", str(e))
        self.assertEqual(len(self.server.requests), retry.MAX_ATTEMPTS)
        self.assertEqual(len(self.time.sleeps), retry.MAX_ATTEMPTS - 1)
        for n, s in enumerate(self.time.sleeps):  # full-jitter backoff, base 2 s, cap 60 s
            self.assertTrue(0.0 <= s <= min(60.0, 2.0 * 2 ** n), (n, s))
        self.assertEqual(len(self.logs), retry.MAX_ATTEMPTS - 1)

    def test_a_wait_past_the_deadline_is_not_started(self):
        _, e = self._call(Script(err(429, Retry_After=700), OK_SAM))  # SAM's deadline is 600 s
        self.assertIn("would pass the 600 s deadline", str(e))
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.time.sleeps, [])

    def test_the_deadline_bounds_the_total_wait(self):
        _, e = self._call(Script(err(503, Retry_After=250)))
        self.assertIn("deadline", str(e))
        self.assertEqual(len(self.server.requests), 3)  # waits of 250 and 250 fit in 600 s, a third does not
        self.assertLessEqual(self.time.t, client_mod.DEADLINE_S[client_mod.SAM_MODEL])

    def test_muse_gets_the_longer_deadline(self):
        _, e = self._call(Script(err(429, Retry_After=700), OK_IMAGE), muse=True)  # Muse: 900 s
        self.assertIsNone(e)
        self.assertEqual(len(self.server.requests), 2)

    def test_a_dropped_connection_is_retried_as_an_unknown_outcome(self):
        out, e = self._call(Script((0, {}, None), OK_IMAGE), muse=True)
        self.assertIsNone(e)
        rows = self.ledger()
        self.assertEqual([r["outcome"] for r in rows], ["unknown", "ok"])
        self.assertEqual([c.get("outcome", "ok") for c in self.client.log.calls], ["unknown", "ok"])

    def test_a_refused_connection_is_retried_as_failed(self):
        self.cache = Path(self.tmp.name) / "refused"
        client = ModelClient(self.cache, mode="live", policy=self.policy, limiter=self.limiter)
        with mock.patch.object(client_mod, "API_BASE", "http://127.0.0.1:9/v1"), \
                mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY, "no_proxy": "127.0.0.1"}), \
                self.assertLogs("herogen.pipeline.client", "WARNING"), self.assertRaises(ApiError) as cm:
            client.segment(IMAGE, "character")
        self.assertIn("after 8 attempts", str(cm.exception))
        self.assertEqual({r["outcome"] for r in self.ledger()}, {"failed"})
        self.assertEqual(client.log.calls, [])  # nothing reached the model, nothing may be billed

    def test_muse_timeout_on_our_side_is_an_unknown_outcome_and_counted(self):
        with mock.patch.object(client_mod, "TIMEOUT_S", 0.3):
            out, e = self._call(Script(OK_IMAGE + (1.0,), OK_IMAGE), muse=True)
        self.assertIsNone(e)
        self.assertEqual(out.size, IMAGE.size)
        rows = self.ledger()
        self.assertEqual([(r["attempt"], r["outcome"], r["status"]) for r in rows], [(1, "unknown", None), (2, "ok", 200)])
        self.assertIn("TimeoutError", rows[0]["error"])
        calls = self.client.log.calls
        self.assertEqual([(c["model"], c["cached"], c.get("outcome")) for c in calls],
                         [(client_mod.IMAGE_MODEL, False, "unknown"), (client_mod.IMAGE_MODEL, False, None)])
        self.assertAlmostEqual(self.client.log.cost_usd(), 2 * client_mod.PRICE_PER_IMAGE_USD[client_mod.IMAGE_MODEL])

    def test_a_transiently_failed_answer_is_retried(self):
        failed = (200, {}, {**sam_response(""), "status": "failed", "output": [],
                            "error": {"code": "rate_limit_exceeded", "message": "slow down"}})
        out, e = self._call(Script(failed, OK_SAM))
        self.assertIsNone(e)
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual([(r["status"], r["outcome"]) for r in self.ledger()], [(429, "failed"), (200, "ok")])
        self.assertEqual(json.loads(self.limiter.state_path.read_text())["limit"], retry.LIMIT_START // 2)
        self.assertEqual([c.get("outcome") for c in self.client.log.calls], ["unknown", None])  # may be billed

    def test_an_incomplete_answer_is_not_retried_and_stops(self):
        incomplete = (200, {}, {**sam_response(""), "status": "incomplete", "output": [],
                                "incomplete_details": {"reason": "max_output_tokens"}})
        _, e = self._call(Script(incomplete, OK_SAM))
        self.assertIn("status is 'incomplete'", str(e))
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(list(self.cache.glob("*.json")), [])  # not a result, never cached

    def test_an_unreadable_200_is_not_retried(self):
        _, e = self._call(Script((200, {}, ["not", "an", "object"]), OK_SAM))
        self.assertIn("not a JSON object", str(e))
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.ledger()[0]["outcome"], "unreadable")
        self.assertEqual([c.get("outcome") for c in self.client.log.calls], ["unknown"])  # answered: counted
        self.assertAlmostEqual(self.client.log.cost_usd(), client_mod.PRICE_PER_IMAGE_USD[client_mod.SAM_MODEL])

    def _hold_left(self) -> float:
        return json.loads(self.limiter.state_path.read_text())["hold_until"] - self.time.wall()

    def _next_call_is_prompt(self) -> None:
        """A later call on the key, from a fresh limiter object (another process), is held off at most
        HOLD_MAX_S."""
        self.limiter = retry.SharedLimiter(Path(self.tmp.name) / "limiter", "k", clock=self.time.wall,
                                           sleep=self.time.pass_time)
        t0 = self.time.t
        _, e = self._call(Script(OK_SAM))
        self.assertIsNone(e)
        self.assertLessEqual(self.time.t - t0, retry.HOLD_MAX_S + 1.0)

    def test_a_huge_or_bad_server_wait_never_blocks_the_key(self):
        cases = {
            "Retry-After 99999999": {"Retry-After": "99999999"},
            "Retry-After inf": {"Retry-After": "inf"},
            "Retry-After nan": {"Retry-After": "nan"},
            "Retry-After in the year 9999": {"Retry-After": "Fri, 31 Dec 9999 23:59:59 GMT"},
            "x-ratelimit-reset inf": {"x-ratelimit-reset": "inf"},
            "retry-after-ms inf": {"retry-after-ms": "inf"},
            "x-ratelimit-reset 1e300": {"x-ratelimit-reset": "1e300"},
            "ratelimit-reset 99999999": {"ratelimit-reset": "99999999"},
        }
        for name, headers in cases.items():
            with self.subTest(name):
                self.setUp()
                self._call(Script((429, headers, error_body(429)), OK_SAM))
                for s in self.time.sleeps:
                    self.assertTrue(math.isfinite(s) and s <= client_mod.DEADLINE_S[client_mod.SAM_MODEL], s)
                self.assertLessEqual(self._hold_left(), retry.HOLD_MAX_S)
                self.assertLessEqual(self.time.t, client_mod.DEADLINE_S[client_mod.SAM_MODEL])
                self._next_call_is_prompt()
                self.tearDown()

    def test_a_wait_not_started_holds_no_one(self):
        _, e = self._call(Script(err(429, Retry_After=99999999), OK_SAM))
        self.assertIn("would pass the 600 s deadline", str(e))
        self.assertLessEqual(self._hold_left(), 0.0)
        self.assertEqual(json.loads(self.limiter.state_path.read_text())["limit"], retry.LIMIT_START // 2)

    def test_a_long_wait_holds_others_at_most_the_cap(self):
        _, e = self._call(Script(err(429, Retry_After=300), OK_SAM))
        self.assertIsNone(e)
        self.assertTrue(300.0 <= self.time.sleeps[0] < 301.0)  # the caller honours the server
        state = json.loads(self.limiter.state_path.read_text())
        self.assertLessEqual(state["hold_until"] - (self.time.wall0 + self.time.sleeps[0]), 0.0)  # over by now
        lim = retry.SharedLimiter(self.tmp.name + "/other", "k", clock=self.time.wall, sleep=self.time.pass_time)
        slot, _ = lim.acquire(self.time.wall() + 1)
        self.assertEqual(lim.release(slot, rate_limited=True, hold_s=1e9), 2)
        self.assertLessEqual(json.loads(lim.state_path.read_text())["hold_until"] - self.time.wall(),
                             retry.HOLD_MAX_S)

    def test_an_epoch_milliseconds_reset_is_read_as_one(self):
        reset_ms = str(int((self.time.wall() + 30) * 1000))
        _, e = self._call(Script(err(429, x_ratelimit_reset=reset_ms), OK_SAM))
        self.assertIsNone(e)
        self.assertTrue(30.0 <= self.time.sleeps[0] < 32.0, self.time.sleeps)

    def test_a_damaged_hold_in_the_state_file_is_capped(self):
        for bad in (1e18, float("inf"), "soon"):
            with self.subTest(bad=bad):
                self.limiter.dir.mkdir(parents=True, exist_ok=True)
                self.limiter.state_path.write_text(json.dumps({"limit": 4, "slots": {}, "hold_until": bad,
                                                               "successes": 0}))
                self._next_call_is_prompt()

    def test_a_muse_gateway_error_may_be_billed(self):
        # The service may have generated (and billed) the image before the gateway gave up.
        out, e = self._call(Script(err(504), err(502), err(500), OK_IMAGE), muse=True)
        self.assertIsNone(e)
        self.assertEqual([(r["status"], r["outcome"]) for r in self.ledger()],
                         [(504, "unknown"), (502, "unknown"), (500, "unknown"), (200, "ok")])
        self.assertAlmostEqual(self.client.log.cost_usd(), 4 * client_mod.PRICE_PER_IMAGE_USD[client_mod.IMAGE_MODEL])

    def test_a_muse_503_is_refused_unprocessed(self):
        _, e = self._call(Script(err(503), OK_IMAGE), muse=True)
        self.assertIsNone(e)
        self.assertEqual([r["outcome"] for r in self.ledger()], ["failed", "ok"])
        self.assertAlmostEqual(self.client.log.cost_usd(), client_mod.PRICE_PER_IMAGE_USD[client_mod.IMAGE_MODEL])

    def test_a_sam_5xx_is_not_counted(self):
        _, e = self._call(Script(err(504), OK_SAM))
        self.assertIsNone(e)
        self.assertEqual([r["outcome"] for r in self.ledger()], ["failed", "ok"])

    def test_a_muse_409_is_never_retried(self):
        _, e = self._call(Script(err(409), OK_IMAGE), muse=True)
        self.assertIn("HTTP 409", str(e))
        self.assertIn("not retried", str(e))
        self.assertEqual(len(self.server.requests), 1)

    def test_an_error_while_reading_the_answer_may_be_billed(self):
        class Answer:
            status = 200

            def __init__(self, error):
                self.error = error

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def read(self):
                raise self.error

        errors = [ssl.SSLError("bad record mac"), ConnectionAbortedError("aborted"),
                  http.client.IncompleteRead(b"{", 100)]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.cache = Path(self.tmp.name) / f"read-{type(error).__name__}"
                client = ModelClient(self.cache, mode="live", policy=self.policy, limiter=self.limiter)
                answers = iter([Answer(error)])

                def urlopen(req, timeout):
                    return next(answers, None) or _OkAnswer()

                with mock.patch.object(client_mod.urllib.request, "urlopen", urlopen), \
                        mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                        self.assertLogs("herogen.pipeline.client", "WARNING"):
                    img = client.generate_image("p")
                self.assertEqual(img.size, IMAGE.size)
                self.assertEqual([r["outcome"] for r in self.ledger()], ["unknown", "ok"])
                self.assertAlmostEqual(client.log.cost_usd(),
                                       2 * client_mod.PRICE_PER_IMAGE_USD[client_mod.IMAGE_MODEL])

    def test_a_429_lowers_the_shared_limit(self):
        _, e = self._call(Script(err(429, Retry_After=1), OK_SAM))
        self.assertIsNone(e)
        state = json.loads(self.limiter.state_path.read_text())
        self.assertEqual(state["limit"], retry.LIMIT_START // 2)
        self.assertEqual(state["slots"], {})
        self.assertGreater(state["hold_until"], self.time.wall0)


class _OkAnswer:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def read(self):
        return json.dumps(OK_IMAGE[2]).encode()


@unittest.skipUnless(os.name == "nt", "Windows process API")
class WindowsProcessProbeTest(unittest.TestCase):
    def test_open_process_results(self):
        for error, alive in ((8, True), (5, True), (87, False), (9999, True)):
            with self.subTest(error=error), mock.patch.object(retry, "_kernel32") as kernel, \
                    mock.patch.object(retry.ctypes, "get_last_error", return_value=error):
                kernel.OpenProcess.return_value = None
                self.assertEqual(retry._alive(os.getpid()), alive)
                kernel.OpenProcess.assert_called_once_with(0x00100000, False, os.getpid())
                kernel.WaitForSingleObject.assert_not_called()
                kernel.CloseHandle.assert_not_called()

    def test_wait_results_close_the_handle(self):
        for result, alive in ((0xFFFFFFFF, True), (0x102, True), (0, False), (1, True)):
            with self.subTest(result=result), mock.patch.object(retry, "_kernel32") as kernel:
                kernel.OpenProcess.return_value = 123
                kernel.WaitForSingleObject.return_value = result
                self.assertEqual(retry._alive(os.getpid()), alive)
                kernel.WaitForSingleObject.assert_called_once_with(123, 0)
                kernel.CloseHandle.assert_called_once_with(123)

    def test_wait_exception_closes_the_handle(self):
        with mock.patch.object(retry, "_kernel32") as kernel:
            kernel.OpenProcess.return_value = 123
            kernel.WaitForSingleObject.side_effect = OSError("probe failed")
            with self.assertRaises(OSError):
                retry._alive(os.getpid())
            kernel.CloseHandle.assert_called_once_with(123)

    def test_failed_probe_keeps_the_only_slot(self):
        for open_failed in (True, False):
            with self.subTest(open_failed=open_failed), tempfile.TemporaryDirectory() as tmp:
                clock = FakeTime()
                options = dict(start=1, maximum=1, clock=clock.wall, sleep=clock.pass_time)
                owner = retry.SharedLimiter(tmp, "k", **options)
                claimer = retry.SharedLimiter(tmp, "k", **options)
                slot, _ = owner.acquire(clock.wall() + 1)
                self.assertIsNotNone(slot)
                with mock.patch.object(retry, "_kernel32") as kernel, \
                        mock.patch.object(retry.ctypes, "get_last_error", return_value=8):
                    kernel.OpenProcess.return_value = None if open_failed else 123
                    kernel.WaitForSingleObject.return_value = 0xFFFFFFFF
                    second, waited = claimer.acquire(clock.wall() + 1)
                self.assertIsNone(second)
                self.assertGreaterEqual(waited, 1)
                self.assertEqual(list(json.loads(owner.state_path.read_text())["slots"]), [slot])
                owner.release(slot)


class SharedLimiterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.time = FakeTime()

    def tearDown(self):
        self.tmp.cleanup()

    def _limiter(self, **kw) -> retry.SharedLimiter:
        return retry.SharedLimiter(self.tmp.name, "k", clock=self.time.wall, sleep=self.time.pass_time, **kw)

    def test_limit_holds_and_adapts(self):
        a, b = self._limiter(start=2, maximum=3), self._limiter(start=2, maximum=3)  # two "processes"
        s1, _ = a.acquire(self.time.wall() + 10)
        s2, _ = b.acquire(self.time.wall() + 10)
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        s3, waited = a.acquire(self.time.wall() + 5)
        self.assertIsNone(s3)  # both slots taken until the deadline
        self.assertGreater(waited, 4.0)
        self.assertEqual(b.release(s2, rate_limited=True, hold_s=30.0), 1)  # a 429 halves the limit
        slot, waited = b.acquire(self.time.wall() + 100)
        self.assertIsNone(slot)  # a's slot is still held and the limit is 1
        a.release(s1, ok=True)
        t0 = self.time.wall()
        state = json.loads(a.state_path.read_text())
        slot, waited = b.acquire(self.time.wall() + 100)
        self.assertIsNotNone(slot)
        self.assertGreaterEqual(self.time.wall(), state["hold_until"])  # never before the hold ends
        self.assertGreaterEqual(self.time.wall(), t0)
        for _ in range(retry.SUCCESSES_PER_STEP - 2):
            b.release(slot, ok=True)
            slot, _ = b.acquire(self.time.wall() + 10)
        self.assertEqual(b.release(slot, ok=True), 2)  # enough successes since the 429 raise it again

    def test_a_dead_processs_slot_is_freed(self):
        lim = self._limiter(start=1, maximum=1)
        child = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                               text=True, check=True)
        dead = int(child.stdout)
        lim._update(lambda st, now: st["slots"].update({"gone": {"pid": dead, "since": now}}))
        slot, _ = lim.acquire(self.time.wall() + 1)
        self.assertIsNotNone(slot)

    def test_the_key_never_names_the_state_file(self):
        with mock.patch.dict(os.environ, {retry.LIMITER_DIR_ENV: self.tmp.name}):
            lim = retry.SharedLimiter.for_key(DUMMY_KEY)
        self.assertNotIn(DUMMY_KEY, str(lim.state_path))
        self.assertEqual(lim.state_path.parent, Path(self.tmp.name))

    def test_processes_sharing_a_key_never_exceed_the_limit(self):
        def slow(endpoint, body):
            return OK_SAM + (0.15,)

        server = FakeModelServer(slow, DUMMY_KEY)
        child = textwrap.dedent("""
            import os, sys
            sys.path[:0] = [sys.argv[1]]
            from PIL import Image
            from herogen.pipeline import client, retry
            client.API_BASE = sys.argv[2]
            lim = retry.SharedLimiter(sys.argv[3], "k", start=2, maximum=2)
            c = client.ModelClient(sys.argv[4], mode="live", limiter=lim)
            for i in range(6):
                c.segment(Image.new("RGB", (8, 8), (i, int(sys.argv[5]), 0)), "character")
        """)
        root = os.path.dirname(_HERE)
        env = {**os.environ, "MODEL_API_KEY": DUMMY_KEY, "no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}
        with server.serving(self.tmp.name):
            procs = [subprocess.Popen([sys.executable, "-c", child, root, server.base, self.tmp.name,
                                       f"{self.tmp.name}/cache{n}", str(n)], env=env) for n in range(4)]
            codes = [p.wait(timeout=120) for p in procs]
        self.assertEqual(codes, [0] * 4)
        self.assertEqual(len(server.requests), 24)
        self.assertLessEqual(server.max_in_flight, 2)
        self.assertEqual(server.max_in_flight, 2)  # and the limit is used

    def _children(self, code: str, n: int = 4) -> None:
        """Run `code` in `n` processes at once (argv: repo root, this test's folder, child number)."""
        root = os.path.dirname(_HERE)
        procs = []
        try:
            for i in range(n):
                procs.append(subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), root,
                                                self.tmp.name, str(i)]))
            self.assertEqual([p.wait(timeout=120) for p in procs], [0] * n)
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()

    def test_children_timeout_reaps_every_child(self):
        procs = [mock.Mock() for _ in range(4)]
        for p in procs:
            p.poll.return_value = None
            p.wait.return_value = 0
        procs[0].wait.side_effect = [subprocess.TimeoutExpired("child", 120), 0]
        procs[1].wait.side_effect = [subprocess.TimeoutExpired("child", 5), 0]
        with mock.patch.object(subprocess, "Popen", side_effect=procs), \
                self.assertRaises(subprocess.TimeoutExpired):
            self._children("pass")
        for p in procs:
            p.terminate.assert_called_once_with()
            self.assertIn(mock.call(timeout=5), p.wait.call_args_list)
        procs[1].kill.assert_called_once_with()
        self.assertEqual(procs[1].wait.call_args_list[-1], mock.call())

    def test_processes_never_lose_a_state_update(self):
        # Every read-modify-write of the state file happens under the lock, so none overwrites another's.
        self._children("""
            import sys
            sys.path[:0] = [sys.argv[1]]
            from herogen.pipeline import retry
            lim = retry.SharedLimiter(sys.argv[2], "k", start=3, maximum=5)
            for _ in range(50):
                lim._update(lambda st, now: st.__setitem__("n", st.get("n", 0) + 1))
        """)
        state = json.loads((Path(self.tmp.name) / "k.json").read_text())
        self.assertEqual((state["n"], state["limit"], state["slots"]), (200, 3, {}))

    def test_parallel_ledger_rows_never_interleave(self):
        # Force partial raw writes and yield between them, so only the lock prevents torn rows.
        self._children("""
            import io, sys, time
            sys.path[:0] = [sys.argv[1]]
            from herogen.pipeline import client

            class ShortWriteFile(io.FileIO):
                calls = 0
                short_writes = 0

                def write(self, data):
                    n = super().write(data[:4096])
                    type(self).calls += 1
                    type(self).short_writes += n < len(data)
                    time.sleep(0.001)
                    return n

            def split_open(path, mode):
                return io.TextIOWrapper(io.BufferedWriter(ShortWriteFile(path, mode)))

            client.open = split_open
            c = client.ModelClient(sys.argv[2] + "/cache", mode="replay")
            for i in range(40):
                c._ledger({"child": int(sys.argv[3]), "i": i, "pad": sys.argv[3] * 200000})
            assert ShortWriteFile.calls > 40, ShortWriteFile.calls
            assert ShortWriteFile.short_writes >= 40, ShortWriteFile.short_writes
        """)
        rows = [json.loads(line) for line in (Path(self.tmp.name) / "cache" / client_mod.LEDGER).read_text()
                .splitlines()]
        self.assertEqual(sorted((r["child"], r["i"]) for r in rows), [(c, i) for c in range(4) for i in range(40)])
        self.assertTrue(all(r["pad"] == str(r["child"]) * 200000 for r in rows))


if __name__ == "__main__":
    unittest.main()
