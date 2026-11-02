"""The service's side of the build log: job-level events from the orchestrator (queue, requeue, cancel, model
ready, end), step boundaries from each group, the event file reaching stage subprocesses, and `EventPump`, which
mirrors a stage's events to the data volume while the stage runs on a local copy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_jobs import JobCase  # noqa: E402

from herogen.pipeline import events  # noqa: E402
from herogen.service import worker  # noqa: E402
from herogen.service.fakes import FakeRunner  # noqa: E402
from herogen.service.runner import PLAN, CliRunner, StageContext  # noqa: E402
from herogen.service.snapshot import LocalVolume  # noqa: E402
from herogen.service.worker import CopyWorkdir, EventPump, make_context, run_group, run_job  # noqa: E402


class CountingVolume(LocalVolume):
    def __init__(self) -> None:
        super().__init__()
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class BuildLogCase(JobCase):
    def log(self, job) -> list[dict]:
        path = self.storage.run_dir(self.user["id"], job["id"]) / events.FILE
        return [events.validate(json.loads(x)) for x in path.read_text().splitlines()]


class TestServiceEvents(BuildLogCase):
    def test_a_requeued_attempt_logs_why_and_keeps_finished_steps(self):
        self.runner.fail = {"3": "transient"}
        job = self.new_job()
        self.assertEqual(self.go(job), "queued")
        self.runner.fail = {}
        self.assertEqual(self.go(job, call="call-2"), "succeeded")
        got = self.log(job)
        requeue = next(e for e in got if e["kind"] == "requeue")
        self.assertEqual((requeue["a"], requeue["next"], requeue["of"], requeue["step"], requeue["code"]),
                         (1, 2, self.s.max_attempts, "3", "model_unavailable"))
        failed = next(e for e in got if e["kind"] == "step_end" and e["step"] == "3" and e["a"] == 1)
        self.assertEqual((failed["outcome"], failed["code"]), ("failed", "model_unavailable"))
        second = got[got.index(requeue) + 1:]
        self.assertEqual(second[0]["kind"], "job_start")
        self.assertEqual(second[0]["a"], 2)
        self.assertNotIn("queue", [e["kind"] for e in second], "queue time is logged for the first attempt")
        kept = [e["step"] for e in second if e["kind"] == "step_end" and e["outcome"] == "kept"]
        self.assertEqual(kept, ["0", "1", "2"])
        self.assertEqual([e["kind"] for e in got].count("model_ready"), 1)
        self.assertEqual((got[-1]["kind"], got[-1]["state"], got[-1]["export"]), ("job_end", "succeeded", "ready"))

    def test_cancel_is_logged_after_the_running_stage(self):
        job = self.new_job()
        store, user = self.store, self.user

        class CancelAfterCover(FakeRunner):
            def run(self, spec, ctx):
                out = super().run(spec, ctx)
                if spec.key == "1":
                    store.job_cancel(ctx.job_id, user["id"])
                return out

        self.runner = CancelAfterCover()
        self.assertEqual(self.go(job), "canceled")
        got = self.log(job)
        self.assertEqual([e["kind"] for e in got[-2:]], ["cancel", "job_end"])
        self.assertEqual(got[-1]["state"], "canceled")
        self.assertEqual([e["step"] for e in got if e["kind"] == "step_start"], ["0", "1"])

    def test_a_rejection_ends_the_log_with_its_step(self):
        self.runner.fail = {"2": "reject"}
        job = self.new_job()
        self.assertEqual(self.go(job), "rejected")
        got = self.log(job)
        end = next(e for e in got if e["kind"] == "step_end" and e["step"] == "2")
        self.assertEqual((end["outcome"], end["code"]), ("rejected", "rejected"))
        self.assertEqual((got[-1]["kind"], got[-1]["state"]), ("job_end", "rejected"))

    def test_a_deleted_hero_gets_no_folder_back(self):
        job = self.new_job()
        run = self.storage.run_dir(self.user["id"], job["id"])
        worker.job_event(self.storage, job, None, "cancel", a=1)
        self.assertFalse(run.exists())
        worker.job_event(self.storage, job, None, "job_start", create=True, a=1)
        self.assertTrue((run / events.FILE).is_file())


class TestEventPump(BuildLogCase):
    def test_copies_and_commits_only_when_the_file_changed(self):
        src, dst = self.tmp / "local" / events.FILE, self.tmp / "remote" / events.FILE
        src.parent.mkdir()
        dst.parent.mkdir()
        vol, current = CountingVolume(), {"ok": True}
        pump = EventPump(src, dst, vol, lambda: current["ok"])
        self.assertFalse(pump.pump(), "nothing to copy yet")
        events.emit("local", file=src, step="4", what="part_split")
        self.assertTrue(pump.pump())
        self.assertEqual(dst.read_bytes(), src.read_bytes())
        self.assertFalse(pump.pump(), "unchanged: no copy, no commit")
        self.assertEqual(vol.commits, 1)
        events.emit("local", file=src, step="4", what="decimated")
        current["ok"] = False
        self.assertFalse(pump.pump(), "a stale attempt never writes")
        self.assertTrue(pump.stopped)
        current["ok"] = True
        self.assertFalse(pump.pump(), "and stays stopped")
        self.assertEqual(len(dst.read_text().splitlines()), 1)
        self.assertEqual(vol.commits, 1)
        self.assertFalse(any(p.name.endswith(".part") for p in dst.parent.iterdir()))

    def test_never_creates_a_missing_run_folder(self):
        src, dst = self.tmp / "local" / events.FILE, self.tmp / "gone" / events.FILE
        src.parent.mkdir()
        events.emit("local", file=src, step="4", what="part_split")
        vol = CountingVolume()
        self.assertFalse(EventPump(src, dst, vol, lambda: True).pump())
        self.assertFalse(dst.parent.exists())
        self.assertEqual(vol.commits, 0)

    def test_a_stage_on_a_local_copy_shows_its_events_before_it_ends(self):
        job = self.new_job()
        seen = {}
        test = self

        class Watching(FakeRunner):
            def run(self, spec, ctx):
                if spec.key == "1":
                    events.emit("local", file=ctx.run_dir / events.FILE, step="1", a=ctx.attempt, what="probe")
                    remote = test.storage.run_dir(test.user["id"], ctx.job_id) / events.FILE
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and b'"what":"probe"' not in (
                            remote.read_bytes() if remote.exists() else b""):
                        time.sleep(0.02)
                    seen["live"] = b'"what":"probe"' in remote.read_bytes()
                return super().run(spec, ctx)

        def execute(j, attempt, group, resume):
            ctx = make_context(self.s, self.storage, j)
            wd = CopyWorkdir(ctx, self.tmp / "work" / group, CountingVolume(),
                             lambda: self.store.job_current(j["id"], attempt))
            return run_group(self.store, self.s, j, attempt, group, resume, Watching(), wd)

        with mock.patch.object(worker, "PUMP_EVERY_S", 0.05):
            self.assertEqual(run_job(self.store, self.s, self.storage, job["id"], "c", execute), "succeeded")
        self.assertTrue(seen["live"], "the pump mirrored the event while stage 1 still ran")
        kinds = [e["kind"] for e in self.log(job)]
        self.assertEqual(kinds.count("step_start"), 8)
        self.assertEqual(kinds[-1], "job_end")


class TestStageEnvironment(BuildLogCase):
    def test_every_stage_subprocess_gets_its_own_event_file_step_and_attempt(self):
        ctx = StageContext(job_id="j", kind="sketch", profile="best", name="", prompt="", run_dir=self.tmp / "run",
                           input_path=self.tmp / "in.png", cache_dir=self.tmp / "c", remote_cache_dir=self.tmp / "r",
                           attempt=2)
        ctx.run_dir.mkdir()
        seen = []

        def fake_run(argv, **kw):
            seen.append(kw["env"])
            return subprocess.CompletedProcess(argv, 0, "", "")

        runner = CliRunner(model_api_key="k")
        runner._flags["herogen.blender"] = ""
        with mock.patch("herogen.service.runner.subprocess.run", side_effect=fake_run), \
                mock.patch.dict(os.environ, {events.ENV_FILE: "/elsewhere/events.jsonl"}):
            for key in ("1", "3", "6"):
                runner.run(next(s for s in PLAN if s.key == key), ctx)
        for env, key in zip(seen, ("1", "3", "6")):
            self.assertEqual(env[events.ENV_FILE], str(ctx.run_dir / events.FILE))
            self.assertEqual((env[events.ENV_STEP], env[events.ENV_ATTEMPT]), (key, "2"))
