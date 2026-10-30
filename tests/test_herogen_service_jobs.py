"""Jobs end to end with the fake runner: success, rejection with its reason, retries within limits, refunds,
cancellation, duplicate workers, resuming, the copy-and-fence work folder, the sweep, retention, stage commands."""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import Clock, TempDirCase, make_user, png, settings

from herogen.service.fakes import FakeRunner
from herogen.service.runner import (
    PLAN,
    CliRunner,
    Outcome,
    StageContext,
    describe_error,
    describe_rejection,
    plan_for,
    resume_index,
    write_record,
)
from herogen.service.snapshot import LocalVolume
from herogen.service.storage import Storage
from herogen.service.store import Store, new_id
from herogen.service.worker import (
    CopyWorkdir,
    GroupResult,
    InPlaceWorkdir,
    make_context,
    purge,
    reconcile,
    run_group,
    run_job,
)


class JobCase(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.clock = Clock()
        self.s = settings(self.tmp)
        self.store = Store(self.s.meta_path, clock=self.clock)
        self.storage = Storage(self.s.data_dir)
        self.user = make_user(self.store)
        self.runner = FakeRunner()
        self.n = 0

    def tearDown(self) -> None:
        self.store.close()
        super().tearDown()

    def new_job(self, kind="sketch", key=None, **kw):
        jid = new_id("j")
        if kind == "sketch":
            self.storage.save_input(self.user["id"], jid, png(400, 600))
        self.n += 1
        job, _ = self.store.job_create(job_id=jid, user_id=self.user["id"], client_key=key or f"key-{self.n:05d}",
                                       kind=kind, profile="roblox-ingame", jobs_per_day=10, jobs_total=10,
                                       storage_max_bytes=10 ** 10, prompt="a knight" if kind == "prompt" else "",
                                       **kw)
        return job

    def execute(self, job, attempt, group, resume):
        return run_group(self.store, self.s, job, attempt, group, resume, self.runner,
                         InPlaceWorkdir(make_context(self.s, self.storage, job)))

    def go(self, job, call="call-1", **kw):
        return run_job(self.store, self.s, self.storage, job["id"], call, self.execute, **kw)

    def stages(self, job):
        return {r["stage"]: r for r in self.store.job_stages(job["id"])}


class TestJobFlow(JobCase):
    def test_success_records_every_stage(self):
        job = self.new_job()
        self.assertEqual(self.go(job), "succeeded")
        j = self.store.job_get(job["id"])
        self.assertEqual((j["state"], j["attempts"], j["counts_quota"]), ("succeeded", 1, 1))
        self.assertGreater(j["bytes"], 0)
        self.assertAlmostEqual(j["cost_usd"], 0.0375, places=6, msg="the job's own stages; 2h runs beside it")
        self.assertIn("basis", json.loads(j["cost_detail"]))
        st = self.stages(job)
        self.assertEqual(set(st), {"0", "1", "2", "3", "4", "6", "5"})
        self.assertTrue(all(r["state"] == "passed" for r in st.values()))
        self.assertEqual(self.runner.calls, ["0", "1", "2", "3", "4", "6", "5"])

    def test_prompt_job_draws_its_sketch_first(self):
        job = self.new_job("prompt")
        self.assertEqual(self.go(job), "succeeded")
        self.assertEqual(self.runner.calls[0], "p")
        self.assertTrue(self.storage.input_path(self.user["id"], job["id"]).is_file())

    def test_rejection_shows_the_reason_and_stops(self):
        self.runner.fail = {"2": "reject"}
        job = self.new_job()
        self.assertEqual(self.go(job), "rejected")
        j = self.store.job_get(job["id"])
        self.assertEqual((j["error_stage"], j["error_code"], j["counts_quota"]), ("2", "rejected", 1))
        self.assertIn("turnaround", j["error_message"])
        self.assertIn("height_agreement_4_views measured 0.81, needs >= 0.95", j["error_message"])
        self.assertIn("full body", j["error_message"], "with a tip for the next sketch")
        self.assertEqual(self.runner.calls, ["0", "1", "2"])
        self.assertEqual(self.stages(job)["2"]["state"], "rejected")

    def test_permanent_error_after_spending_fails_and_counts(self):
        self.runner.fail = {"3": "error"}
        job = self.new_job()
        self.assertEqual(self.go(job), "failed")
        j = self.store.job_get(job["id"])
        self.assertEqual((j["error_code"], j["counts_quota"], j["attempts"]), ("stage_error", 1, 1),
                         "stages 1-2 spent money: the job counts")
        self.assertNotIn("today's heroes", j["error_message"])

    def test_an_error_before_any_spend_is_refunded(self):
        self.runner.fail = {"0": "error"}
        job = self.new_job()
        self.assertEqual(self.go(job), "failed")
        j = self.store.job_get(job["id"])
        self.assertEqual(j["counts_quota"], 0)
        self.assertIn("does not use up one of today's heroes, but it still counts toward your total",
                      j["error_message"])
        self.assertNotIn("Traceback", j["error_message"])

    def test_a_crashing_runner_is_recorded_not_raised(self):
        self.runner.fail = {"1": "raise"}
        job = self.new_job()
        self.assertEqual(self.go(job), "failed")
        self.assertEqual(self.stages(job)["1"]["state"], "failed")

    def test_transient_error_retries_and_resumes_after_passed_stages(self):
        self.runner.fail = {"3": "transient"}
        job = self.new_job()
        redispatched = []
        self.assertEqual(self.go(job, redispatch=redispatched.append), "queued")
        self.assertEqual(redispatched, [job["id"]])
        self.runner.fail = {}
        self.runner.calls.clear()
        self.assertEqual(self.go(job, call="call-2"), "succeeded")
        self.assertEqual(self.runner.calls, ["3", "4", "6", "5"], "stages 0-2 kept from attempt 1")
        self.assertEqual(self.stages(job)["1"]["message"], "kept from an earlier attempt")
        self.assertEqual(self.store.job_get(job["id"])["attempts"], 2)

    def test_retries_stop_at_the_attempt_limit(self):
        self.runner.fail = {"1": "transient"}
        job = self.new_job()
        self.assertEqual(self.go(job), "queued")
        self.assertEqual(self.go(job, call="call-2"), "failed")
        j = self.store.job_get(job["id"])
        self.assertEqual((j["attempts"], j["counts_quota"]), (2, 0))
        self.assertIsNone(self.go(job, call="call-3"), "a finished job is never run again")

    def test_duplicate_worker_does_nothing(self):
        job = self.new_job()
        claim = self.store.job_claim(job["id"], "first", 60, 2)
        self.assertIsNotNone(claim)
        self.assertIsNone(self.go(job, call="second"))
        self.assertEqual(self.runner.calls, [])

    def test_cancel_stops_at_the_next_stage(self):
        job = self.new_job()
        store = self.store
        user = self.user

        class CancelAfterCover(FakeRunner):
            def run(self, spec, ctx):
                out = super().run(spec, ctx)
                if spec.key == "1":
                    store.job_cancel(ctx.job_id, user["id"])
                return out

        self.runner = CancelAfterCover()
        self.assertEqual(self.go(job), "canceled")
        self.assertEqual(self.runner.calls, ["0", "1"])

    def test_missing_upload_fails_with_refund(self):
        job = self.new_job()
        self.storage.input_path(self.user["id"], job["id"]).unlink()
        self.assertEqual(self.go(job), "failed")
        self.assertEqual(self.store.job_get(job["id"])["counts_quota"], 0)

    def test_restarted_group_stops_after_the_limit(self):
        job = self.new_job()
        claim = self.store.job_claim(job["id"], "c", 60, 2)
        for _ in range(2):
            self.store.group_enter(job["id"], claim["attempt"], "model", 2)
        res = self.execute(claim["job"], claim["attempt"], "model", 0)
        self.assertEqual((res.status, res.code, res.retry), ("error", "restarts", False))
        self.assertEqual(self.runner.calls, [], "nothing paid for")

    def test_a_failed_roblox_export_still_delivers_the_glb(self):
        for mode in ("reject", "error"):
            self.runner = FakeRunner(fail={"6": mode})
            job = self.new_job()
            self.assertEqual(self.go(job), "succeeded", mode)
            six = self.stages(job)["6"]
            self.assertEqual(six["state"], {"reject": "rejected", "error": "failed"}[mode])
            self.assertIn("Roblox rig", six["message"])
            self.assertEqual(self.stages(job)["5"]["state"], "passed", "the stages after it still run")

    def test_roblox_export_missing_still_delivers_the_glb(self):
        self.runner = FakeRunner(roblox=False)
        job = self.new_job()
        self.assertEqual(self.go(job), "succeeded")
        six = self.stages(job)["6"]
        self.assertEqual(six["state"], "skipped")
        self.assertIn("not part of this version", six["message"])


class TestHeldoutBeside(JobCase):
    """Stage 2h runs beside the job once stage 2 passes; nothing that happens to it fails, delays, retries or
    refunds the job."""

    def heldout_now(self, runner=None):
        """A start hook running 2h at once, inside the job's flow (2h finishes before the job)."""
        def start(job):
            self.run_heldout(job, runner)
        return start

    def run_heldout(self, job, runner=None, root=None):
        from herogen.service.snapshot import LocalVolume
        from herogen.service.worker import HeldoutWorkdir, run_heldout

        ctx = make_context(self.s, self.storage, job)
        work = (root or self.tmp / "heldout-work") / job["id"]

        def workdir_for(claim):
            return HeldoutWorkdir(ctx, work, LocalVolume(), lambda: self.store.side_live(job["id"], "2h", claim))

        return run_heldout(self.store, self.s, job, runner or self.runner, workdir_for)

    def assert_delivered(self, job, attempts=1):
        j = self.store.job_get(job["id"])
        self.assertEqual((j["state"], j["error_code"], j["error_message"], j["attempts"], j["counts_quota"],
                          j["refunded_at"]), ("succeeded", None, None, attempts, 1, None))
        self.assertTrue((self.storage.run_dir(self.user["id"], job["id"]) / "04-body" / "high.glb").is_file())
        return j

    def test_every_kind_of_2h_failure_leaves_the_job_delivered(self):
        class Fails(FakeRunner):
            def __init__(self, how):
                super().__init__()
                self.how = how

            def run(self, spec, ctx):
                if spec.key == "2h":
                    self.calls.append("2h")
                    if self.how == "timeout":
                        return Outcome("failed", 9.0, 0.0, "slow", "timeout", True, "timed out")
                    if self.how == "rate_limit":
                        return Outcome("failed", 1.0, 0.0, "busy", "model_unavailable", True, "HTTP 429: slow down")
                    if self.how == "api_error":   # the stage stops with status: error in its record
                        write_record(ctx.run_dir, {"stage": "2h", "name": "heldout", "ok": False,
                                                   "status": "error", "cost_usd": 0.0025, "checks": []})
                        return CliRunner.outcome(spec, ctx, 1.0, 1, "", "ClientError", False)
                    if self.how == "every_view_rejected":
                        write_record(ctx.run_dir, {"stage": "2h", "name": "heldout", "ok": False, "cost_usd": 0.05,
                                                   "checks": [{"name": "views_accepted", "value": 0, "threshold": 1,
                                                               "pass": False}]})
                        return CliRunner.outcome(spec, ctx, 1.0, 2, "", "", False)
                    if self.how == "crash":
                        raise RuntimeError("container lost")
                    self.calls.pop()       # recorded again by the base class
                return super().run(spec, ctx)

        for how in ("timeout", "rate_limit", "api_error", "every_view_rejected", "crash"):
            runner = Fails(how)
            self.runner = runner
            job = self.new_job()
            self.assertEqual(self.go(job, heldout=self.heldout_now(runner)), "succeeded", how)
            j = self.assert_delivered(job)
            side = self.stages(job)["2h"]
            self.assertEqual(side["state"], "unavailable", how)
            self.assertIn("The hero is not affected.", side["message"], how)
            self.assertEqual(runner.calls.count("2h"), 1, f"{how}: never retried")
            if how == "every_view_rejected":
                self.assertIn("did not accept", side["message"])
                self.assertAlmostEqual(j["cost_usd"], 0.0375 + 0.05, places=6, msg="its spend still counts")

    def test_the_job_finishes_first_and_2h_attaches_later(self):
        started = []
        job = self.new_job()
        self.assertEqual(self.go(job, heldout=started.append), "succeeded")
        j = self.assert_delivered(job)
        self.assertAlmostEqual(j["cost_usd"], 0.0375, places=6)
        self.assertEqual(self.stages(job).get("2h"), None, "not started yet: the job did not wait for it")
        self.assertEqual(self.run_heldout(started[0]), "passed")
        j = self.assert_delivered(job)
        self.assertAlmostEqual(j["cost_usd"], 0.05, places=6, msg="2h's cost joins the job's")
        self.assertAlmostEqual(json.loads(j["cost_detail"])["heldout_usd"], 0.0125, places=6)
        self.assertTrue((self.storage.run_dir(self.user["id"], job["id"]) / "heldout" / "front-left-34.png").is_file())
        self.assertIsNone(self.run_heldout(started[0]), "at most once per job")

    def test_2h_finishes_first(self):
        job = self.new_job()
        self.assertEqual(self.go(job, heldout=self.heldout_now()), "succeeded")
        j = self.assert_delivered(job)
        self.assertEqual(self.stages(job)["2h"]["state"], "passed")
        self.assertAlmostEqual(j["cost_usd"], 0.05, places=6)

    def test_canceled_while_2h_runs(self):
        store, user = self.store, self.user

        class CancelDuring2h(FakeRunner):
            def run(self, spec, ctx):
                if spec.key == "2h":
                    store.job_cancel(ctx.job_id, user["id"])
                return super().run(spec, ctx)

        self.runner = CancelDuring2h()
        job = self.new_job()
        self.assertEqual(self.go(job, heldout=self.heldout_now()), "canceled")
        j = self.store.job_get(job["id"])
        self.assertEqual(j["state"], "canceled", "2h's late finish does not reopen the job")
        self.assertEqual(self.stages(job)["2h"]["state"], "passed", "its record is still attached")
        self.assertNotIn("3", self.runner.calls, "the job's own path stopped at the next stage")

    def test_a_lost_2h_worker_is_recorded_as_unavailable(self):
        job = self.new_job()
        self.assertEqual(self.go(job, heldout=lambda j: self.store.side_claim(j["id"], "2h", "lost", 60)), "succeeded")
        self.clock.advance(61)
        self.store.reconcile_stale(self.s.max_attempts)
        self.assertEqual(self.stages(job)["2h"]["state"], "unavailable")
        self.assert_delivered(job)

    def test_starting_2h_failing_is_not_fatal(self):
        def broken(job):
            raise RuntimeError("could not spawn")

        job = self.new_job()
        self.assertEqual(self.go(job, heldout=broken), "succeeded")
        self.assert_delivered(job)

    def test_a_job_rejected_at_stage_2_starts_no_2h(self):
        self.runner = FakeRunner(fail={"2": "reject"})
        started = []
        job = self.new_job()
        self.assertEqual(self.go(job, heldout=started.append), "rejected")
        self.assertEqual(started, [])


class TestHeldoutFiles(JobCase):
    """The job's own path and 2h never write the same file."""

    def test_the_jobs_copy_back_leaves_2h_files_alone(self):
        job = self.new_job()
        claim = self.store.job_claim(job["id"], "c", 60, 2)
        ctx = make_context(self.s, self.storage, claim["job"])
        ctx.run_dir.mkdir(parents=True)
        wd = CopyWorkdir(ctx, self.tmp / "main", LocalVolume(), lambda: True)
        loc = wd.open()
        (ctx.run_dir / "heldout").mkdir()                        # 2h writes while the job's stage runs
        (ctx.run_dir / "heldout" / "front-left-34.png").write_bytes(b"view")
        (ctx.run_dir / "stage-2h.json").write_text("{}")
        (loc.run_dir / "stage-3.json").write_text("{}")
        wd.sync()                                                # mirrors with deletion, except 2h's paths
        self.assertTrue((ctx.run_dir / "heldout" / "front-left-34.png").is_file())
        self.assertTrue((ctx.run_dir / "stage-2h.json").is_file())
        self.assertTrue((ctx.run_dir / "stage-3.json").is_file())
        wd.close()

    def test_2h_writes_only_its_own_paths_and_never_into_a_deleted_run(self):
        from herogen.service.worker import HeldoutWorkdir

        job = self.new_job()
        self.store.job_claim(job["id"], "c", 60, 2)
        self.assertTrue(self.store.side_claim(job["id"], "2h", "k", 600))
        ctx = make_context(self.s, self.storage, job)
        ctx.run_dir.mkdir(parents=True)
        (ctx.run_dir / "stage-3.json").write_text('{"by": "job"}')
        wd = HeldoutWorkdir(ctx, self.tmp / "h", LocalVolume(), lambda: self.store.side_live(job["id"], "2h", "k"))
        loc = wd.open()
        (loc.run_dir / "heldout").mkdir()
        (loc.run_dir / "heldout" / "v.png").write_bytes(b"v")
        (loc.run_dir / "stage-2h.json").write_text("{}")
        (loc.run_dir / "stage-3.json").write_text('{"by": "2h copy"}')   # a stale copy of the job's file
        (loc.run_dir / "04-body").mkdir()
        (loc.run_dir / "04-body" / "high.glb").write_bytes(b"stale")
        wd.sync()
        self.assertEqual((ctx.run_dir / "stage-3.json").read_text(), '{"by": "job"}', "the job's file is untouched")
        self.assertFalse((ctx.run_dir / "04-body").exists())
        self.assertTrue((ctx.run_dir / "heldout" / "v.png").is_file())
        self.storage.delete_job(self.user["id"], job["id"])     # the user deletes the hero while 2h runs
        (loc.run_dir / "heldout" / "w.png").write_bytes(b"w")
        wd.sync()
        self.assertFalse(ctx.run_dir.exists(), "a deleted run folder is never brought back")
        wd.close()


class TestSweep(JobCase):
    def test_dead_worker_is_requeued_and_dispatched(self):
        job = self.new_job()
        self.store.job_claim(job["id"], "dead", 60, 2)
        self.clock.advance(61)
        sent = []
        out = reconcile(self.store, self.s, lambda jid: sent.append(jid) or f"call-{len(sent)}")
        self.assertEqual(out["stale"], [(job["id"], "queued")])
        self.assertEqual(sent, [job["id"]])
        self.assertEqual(self.go(job, call="call-1"), "succeeded")

    def test_queue_respects_the_running_cap(self):
        jobs = [self.new_job() for _ in range(4)]
        sent = []
        reconcile(self.store, self.s, lambda jid: sent.append(jid) or "c")
        self.assertEqual(sent, [jobs[0]["id"], jobs[1]["id"]])

    def test_retention_deletes_files_and_inputs(self):
        job = self.new_job()
        self.go(job)
        self.clock.advance(self.s.retention_days * 86400 + 1)
        out = purge(self.store, self.s, self.storage)
        self.assertEqual(out["jobs"], [job["id"]])
        self.assertFalse(self.storage.run_dir(self.user["id"], job["id"]).exists())
        self.assertFalse(self.storage.input_path(self.user["id"], job["id"]).exists())
        self.assertEqual(self.store.job_get(job["id"])["prompt"], "")

    def test_retention_expires_cache_entries(self):
        import time as _time

        cache = self.storage.cache_dir(self.user["id"])
        (cache / "ab").mkdir(parents=True)
        old, new = cache / "ab" / "old.json", cache / "new.json"
        old.write_text("{}")
        new.write_text("{}")
        past = _time.time() - (self.s.retention_days + 1) * 86400
        os.utime(old, (past, past))
        out = purge(self.store, self.s, self.storage)
        self.assertEqual(out["cache_files"], 1)
        self.assertFalse(old.exists())
        self.assertFalse((cache / "ab").exists())
        self.assertTrue(new.exists())


class TestCopyWorkdir(JobCase):
    def test_copies_in_and_out_and_fences_stale_attempts(self):
        job = self.new_job()
        claim = self.store.job_claim(job["id"], "c", 60, 2)
        ctx = make_context(self.s, self.storage, claim["job"])
        ctx.run_dir.mkdir(parents=True)
        (ctx.run_dir / "old.json").write_text("{}")
        current = {"ok": True}
        wd = CopyWorkdir(ctx, self.tmp / "local", LocalVolume(), lambda: current["ok"])
        loc = wd.open()
        self.assertNotEqual(loc.run_dir, ctx.run_dir)
        self.assertTrue((loc.run_dir / "old.json").exists())
        self.assertTrue(loc.input_path.is_file())
        (loc.run_dir / "old.json").unlink()
        (loc.run_dir / "new.json").write_text("{}")
        (loc.cache_dir / "entry.json").write_text("{}")
        wd.sync()
        self.assertFalse((ctx.run_dir / "old.json").exists(), "the run folder mirrors the local copy")
        self.assertTrue((ctx.run_dir / "new.json").exists())
        self.assertTrue((ctx.cache_dir / "entry.json").exists())
        current["ok"] = False
        (loc.run_dir / "late.json").write_text("{}")
        wd.sync()
        self.assertFalse((ctx.run_dir / "late.json").exists(), "a stale attempt writes nothing back")
        wd.close()
        self.assertFalse((self.tmp / "local").exists())

    def test_full_job_through_copies(self):
        job = self.new_job()

        def execute(j, attempt, group, resume):
            ctx = make_context(self.s, self.storage, j)
            wd = CopyWorkdir(ctx, self.tmp / "work" / group, LocalVolume(),
                             lambda: self.store.job_current(j["id"], attempt))
            return run_group(self.store, self.s, j, attempt, group, resume, self.runner, wd)

        self.assertEqual(run_job(self.store, self.s, self.storage, job["id"], "c", execute), "succeeded")
        self.assertTrue((self.storage.run_dir(self.user["id"], job["id"]) / "04-body" / "high.glb").is_file())


class TestStageCommands(TempDirCase):
    def ctx(self, **kw) -> StageContext:
        base = dict(job_id="j_" + "a" * 26, kind="sketch", profile="roblox-avatar", name="Sir Bob", prompt="",
                    run_dir=self.tmp / "runs" / ("j_" + "a" * 26), input_path=self.tmp / "in.png",
                    cache_dir=self.tmp / "cache", remote_cache_dir=self.tmp / "cache-remote",
                    stage3_args=("--mesh-source", "sam3d-objects"))
        base.update(kw)
        (self.tmp / "in.png").write_bytes(png(300, 300))
        return StageContext(**base)

    def test_commands_follow_the_stage_clis(self):
        r = CliRunner(python="py")
        r._flags["herogen.blender"] = "usage: --run --stages --profile PROFILE --remote-cache DIR"
        ctx = self.ctx()
        specs = {s.key: s for s in PLAN}
        argv, stdin = r.command(specs["0"], ctx, {}, self.tmp)
        self.assertEqual(argv[:4], ["py", "-m", "herogen.pipeline", "run"])
        self.assertTrue(argv[4].endswith("Sir-Bob.png"), "stage 0 sees the hero's name, not the upload's")
        self.assertIn("--live", argv)
        self.assertEqual(argv[argv.index("--label") + 1], "uploaded sketch")
        self.assertEqual(argv[argv.index("--cache") + 1], str(ctx.cache_dir))
        argv, _ = r.command(specs["3"], ctx, {}, self.tmp)
        self.assertIn("sam3d-objects", argv)
        self.assertEqual(argv[argv.index("--remote-cache") + 1], str(ctx.remote_cache_dir))
        argv, _ = r.command(specs["4"], ctx, {}, self.tmp)
        self.assertEqual(argv[argv.index("--profile") + 1], "roblox-avatar")
        r._flags["herogen.blender"] = "usage: --run --stages"
        argv, _ = r.command(specs["4"], ctx, {}, self.tmp)
        self.assertNotIn("--profile", argv, "a pipeline without profiles gets no --profile")
        argv, stdin = r.command(specs["p"], self.ctx(kind="prompt", prompt="a knight"), {}, self.tmp)
        self.assertEqual(stdin, "a knight", "the prompt goes on stdin, never on the command line")
        self.assertNotIn("a knight", " ".join(argv))

    def test_stage_2_runs_alone_and_2h_on_its_own(self):
        r = CliRunner(python="py")
        ctx = self.ctx()
        specs = {s.key: s for s in PLAN}
        argv, _ = r.command(specs["2"], ctx, {}, self.tmp)
        self.assertEqual(argv[:4], ["py", "-m", "herogen.service.pipestage", "2"])
        argv, _ = r.command(specs["2h"], ctx, {}, self.tmp)
        self.assertEqual(argv[argv.index("--stages") + 1], "2h")
        self.assertEqual(specs["2h"].group, "heldout")

    def test_pipestage_exit_codes(self):
        from unittest import mock

        from herogen.pipeline import stages as pstages
        from herogen.service import pipestage

        args = ["2", "--run-id", "r", "--runs", str(self.tmp / "runs"), "--cache", str(self.tmp / "cache")]
        with mock.patch.object(pstages, "stage2_turnaround", return_value={"name": "turnaround", "seconds": 1.0,
                                                                           "cost_usd": 0.02}), \
                mock.patch("sys.stdout"):
            self.assertEqual(pipestage.main(args), 0)
        failed = pstages.GateFailed(2, {"checks": [{"name": "figures_found", "pass": False}]})
        with mock.patch.object(pstages, "stage2_turnaround", side_effect=failed), \
                mock.patch("sys.stderr"):
            self.assertEqual(pipestage.main(args), 2)
        with mock.patch.object(pstages, "stage2_turnaround", side_effect=RuntimeError("x")), \
                mock.patch("sys.stderr"):
            self.assertEqual(pipestage.main(args), 1)

    def test_stage_env_holds_no_secrets(self):
        from herogen.service.runner import base_env

        os.environ["HEROGEN_SESSION_KEY"] = "x" * 40
        os.environ["HEROGEN_EMAIL_API_KEY"] = "y" * 40
        try:
            env = base_env({"HOME": "/tmp"})
        finally:
            del os.environ["HEROGEN_SESSION_KEY"], os.environ["HEROGEN_EMAIL_API_KEY"]
        self.assertNotIn("HEROGEN_SESSION_KEY", env)
        self.assertNotIn("HEROGEN_EMAIL_API_KEY", env)
        self.assertNotIn("MODEL_API_KEY", env)

    def test_outcomes_from_records(self):
        ctx = self.ctx()
        ctx.run_dir.mkdir(parents=True)
        spec = next(s for s in PLAN if s.key == "1")
        self.assertEqual(CliRunner.outcome(spec, ctx, 1.0, 1, "", "ClientError: HTTP 503", False).state, "failed")
        out = CliRunner.outcome(spec, ctx, 1.0, 1, "", "HTTP 503 from api", False)
        self.assertTrue(out.transient)
        self.assertEqual(out.code, "model_unavailable")
        (ctx.run_dir / "stage-1.json").write_text(json.dumps({"ok": True, "cost_usd": 0.02}))
        out = CliRunner.outcome(spec, ctx, 1.0, 0, "", "", False)
        self.assertEqual((out.state, out.cost_usd), ("passed", 0.02))
        (ctx.run_dir / "stage-1.json").write_text(json.dumps({"ok": False, "checks": [
            {"name": "sketch_mask_area_frac", "value": 0.0123, "threshold": 0.05, "pass": False}]}))
        out = CliRunner.outcome(spec, ctx, 1.0, 2, "", "", False)
        self.assertEqual(out.state, "rejected")
        self.assertIn("No single full-body figure", out.message)
        self.assertIn("0.0123", out.message)
        out = CliRunner.outcome(spec, ctx, 9.0, -1, "", "", True)
        self.assertEqual((out.state, out.code, out.transient), ("failed", "timeout", True))

    def test_transient_failures_are_recognised_by_status(self):
        from herogen.service.runner import transient_failure

        for text in ("ClientError: images/edits failed: HTTP 503: busy", "failed: HTTP 429: slow down",
                     "failed: URLError: <urlopen error [Errno 61] Connection refused>", "TimeoutError: timed out",
                     "RemoteError: sam3d-objects timed out after 1800 s"):
            self.assertTrue(transient_failure(text), text)
        for text in ("ClientError: images/edits failed: HTTP 400: bad request", "HTTP 401: unauthorized",
                     "ClientError: responses: response is not a JSON object", "ClientError: MODEL_API_KEY is not set",
                     "ValueError: bad mask", "ClientError: images/edits failed: HTTP 404: model 5000 not found"):
            self.assertFalse(transient_failure(text), text)
        spec = next(s for s in PLAN if s.key == "1")
        self.assertFalse(describe_error(spec, "ClientError: images/edits failed: HTTP 400: nope", False)[2])
        self.assertTrue(describe_error(spec, "ClientError: images/edits failed: HTTP 502: gateway", False)[2])

    def test_messages_never_leak_internals(self):
        spec = next(s for s in PLAN if s.key == "4")
        code, msg, _ = describe_error(spec, "Traceback ... /home/user/x.py line 3 Bearer abc", False)
        self.assertEqual(code, "stage_error")
        self.assertNotIn("Traceback", msg)
        self.assertNotIn("/home", msg)
        rec = {"checks": [{"name": f"max_joint_gap_high_{i}", "value": 0.3, "threshold": 0.12, "op": "<=",
                           "pass": False} for i in range(6)]}
        text = describe_rejection(spec, rec)
        self.assertIn("gaps at the joints", text)
        self.assertIn("and 2 more", text)

    def test_resume_index_and_plans(self):
        run = self.tmp / "run"
        run.mkdir()
        plan = plan_for("sketch")
        self.assertNotIn("p", [s.key for s in plan])
        self.assertEqual(plan_for("prompt")[0].key, "p")
        self.assertEqual(resume_index(run, plan), 0)
        for k in ("0", "1"):
            (run / f"stage-{k}.json").write_text(json.dumps({"ok": True}))
        (run / "stage-3.json").write_text(json.dumps({"ok": True}))
        self.assertEqual(resume_index(run, plan), 2, "only a leading run of passed stages counts")

    def test_group_result_round_trips(self):
        g = GroupResult("error", "3", "timeout", "slow", True, 1.0, 0.1, 0.01)
        self.assertEqual(GroupResult(**g.to_dict()), g)
        self.assertIsInstance(Outcome("passed"), Outcome)
