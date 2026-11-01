"""Fast-mode admission limits with fake clocks and local processes only."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from herogen.pipeline import retry
from herogen.pipeline.client import ModelClient
from test_herogen_pipeline_retry import FakeTime


class FastLimiterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.time = FakeTime()

    def limiter(self, model="sam-3.1", **kwargs):
        return retry.SharedRPMLimiter(self.tmp.name, "key", model=model,
                                      clock=self.time.wall, sleep=self.time.pass_time, **kwargs)

    def test_default_limits_and_environment_overrides(self):
        for model, rpm in retry.FAST_RPM.items():
            lim = self.limiter(model)
            self.assertEqual((lim.rpm, lim.max_in_flight), (rpm, 32))
        with mock.patch.dict(os.environ, {"HEROGEN_RPM_MUSE_IMAGE_1_0": "123",
                                          "HEROGEN_MAX_IN_FLIGHT": "19"}):
            lim = self.limiter("muse-image-1.0")
            self.assertEqual((lim.rpm, lim.max_in_flight), (123, 19))
        for name in ("HEROGEN_RPM_SAM_3_1", "HEROGEN_MAX_IN_FLIGHT"):
            for bad in ("0", "-1", "bad", "1.5"):
                with self.subTest(name=name, bad=bad), mock.patch.dict(os.environ, {name: bad}):
                    with self.assertRaisesRegex(ValueError, name):
                        self.limiter()

    def test_rolling_minute_counts_released_attempts_and_models_independently(self):
        a, b = self.limiter(rpm=2), self.limiter(rpm=2)
        for lim in (a, b):
            slot, _ = lim.acquire(self.time.wall())
            lim.release(slot)
        slot, _ = a.acquire(self.time.wall() + 59)
        self.assertIsNone(slot)
        other = self.limiter("muse-image-1.0", rpm=1)
        slot, _ = other.acquire(self.time.wall())
        self.assertIsNotNone(slot)
        other.release(slot)
        slot, waited = b.acquire(self.time.wall() + 60)
        self.assertIsNotNone(slot)
        self.assertEqual(waited, 60)
        b.release(slot)

    def test_429_halves_model_rpm_and_holds_the_whole_key(self):
        a, b = self.limiter(rpm=8), self.limiter("muse-image-1.0", rpm=8)
        slot, _ = a.acquire(self.time.wall())
        self.assertEqual(a.release(slot, rate_limited=True, hold_s=5), 4)
        slot, waited = b.acquire(self.time.wall() + 10)
        self.assertEqual(waited, 5)
        b.release(slot)
        for _ in range(retry.SUCCESSES_PER_STEP):
            slot, _ = a.acquire(self.time.wall() + 60)
            limit = a.release(slot, ok=True)
        self.assertEqual(limit, 5)
        slot, _ = a.acquire(self.time.wall() + 60)
        a.release(slot, rate_limited=True, hold_s=float("inf"))
        state = json.loads(a.state_path.read_text())
        self.assertLessEqual(state["hold_until"], self.time.wall())

    def test_in_flight_ceiling_is_shared_across_models_and_legacy_slots(self):
        legacy = retry.SharedLimiter(self.tmp.name, "key", clock=self.time.wall)
        first, _ = legacy.acquire(self.time.wall())
        a = self.limiter(rpm=100, max_in_flight=2)
        second, _ = a.acquire(self.time.wall())
        b = self.limiter("muse-image-1.0", rpm=100, max_in_flight=2)
        self.assertIsNone(b.acquire(self.time.wall())[0])
        a.release(second)
        third, _ = b.acquire(self.time.wall())
        self.assertIsNotNone(third)
        legacy.release(first)
        b.release(third)

    def test_threads_share_rpm_and_in_flight_limits(self):
        for rpm, ceiling in ((7, 32), (100, 3)):
            with self.subTest(rpm=rpm, ceiling=ceiling), tempfile.TemporaryDirectory() as directory:
                limiters = [retry.SharedRPMLimiter(directory, "key", model="sam-3.1", rpm=rpm,
                                                  max_in_flight=ceiling, clock=lambda: 1000)
                            for _ in range(16)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                    slots = list(pool.map(lambda lim: lim.acquire(1000)[0], limiters))
                self.assertEqual(sum(s is not None for s in slots), min(rpm, ceiling))

    def test_processes_and_their_threads_share_rpm_and_in_flight_limits(self):
        code = textwrap.dedent('''
            import concurrent.futures, pathlib, sys, time
            from herogen.pipeline.retry import SharedRPMLimiter
            root, number, rpm, ceiling = sys.argv[1:]
            lim = SharedRPMLimiter(root, "key", model="sam-3.1", rpm=int(rpm),
                                   max_in_flight=int(ceiling), clock=lambda: 1000)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                slots = list(pool.map(lambda _: lim.acquire(1000)[0], range(8)))
            pathlib.Path(root, "ready-" + number).write_text("ready")
            deadline = time.monotonic() + 15
            while len(list(pathlib.Path(root).glob("ready-*"))) < 4:
                if time.monotonic() > deadline:
                    raise RuntimeError("process barrier timed out")
                time.sleep(.005)
            print(sum(s is not None for s in slots))
        ''')
        for rpm, ceiling in ((7, 32), (100, 3)):
            with self.subTest(rpm=rpm, ceiling=ceiling), tempfile.TemporaryDirectory() as directory:
                procs = []
                try:
                    for n in range(4):
                        procs.append(subprocess.Popen([sys.executable, "-c", code, directory, str(n),
                                                       str(rpm), str(ceiling)], stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE, text=True))
                    outputs = [p.communicate(timeout=30) for p in procs]
                    self.assertEqual([p.returncode for p in procs], [0] * 4, outputs)
                    self.assertEqual(sum(int(out) for out, _ in outputs), min(rpm, ceiling))
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

    def test_deadline_and_dead_process_slots(self):
        lim = self.limiter(rpm=1, max_in_flight=1)
        self.assertIsNone(lim.acquire(self.time.wall() - 1)[0])
        slot, _ = lim.acquire(self.time.wall())
        with mock.patch.object(retry, "_alive", return_value=False):
            # Process death frees its slot, but its request still consumes RPM for the full minute.
            self.assertIsNone(lim.acquire(self.time.wall())[0])
            self.time.pass_time(60)
            self.assertIsNotNone(lim.acquire(self.time.wall())[0])

    def test_client_selects_fast_limiter_only_in_fast_mode(self):
        for fast in (False, True):
            client = ModelClient(Path(self.tmp.name) / str(fast), mode="live", fast=fast)
            client.policy.max_attempts = 1
            factory = retry.SharedRPMLimiter if fast else retry.SharedLimiter
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": "placeholder"}), \
                    mock.patch.object(factory, "for_key") as create:
                create.return_value.acquire.return_value = (None, 0)
                with self.assertRaisesRegex(RuntimeError, "no call slot"):
                    client._post("responses", {}, model="sam-3.1", kind="segment", key="synthetic")
            create.assert_called_once_with("placeholder", **({"model": "sam-3.1"} if fast else {}))
