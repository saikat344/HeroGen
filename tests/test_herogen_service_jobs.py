"""Jobs end to end with the fake runner: success, rejection with its reason, retries within limits,
cancellation, duplicate workers, resuming, the copy-and-fence work folder, the sweep, retention, stage commands."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
        self.store = Store(self.s.meta_path, clock=self.clock, max_running=self.s.max_running_jobs)
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
                                       kind=kind, profile="roblox-ingame", storage_max_bytes=10 ** 10, prompt="a knight" if kind == "prompt" else "",
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
    def test_restarted_group_keeps_the_cost_of_both_executions(self):
        job = self.new_job("prompt")

        def execute(j, attempt, group, resume):
            if group == "model":
                self.assertEqual(self.execute(j, attempt, group, resume).status, "ok")
            return self.execute(j, attempt, group, resume)

        self.assertEqual(run_job(self.store, self.s, self.storage, job["id"], "restart", execute), "succeeded")
        self.assertAlmostEqual(self.store.job_get(job["id"])["cost_usd"], 2 * 0.0475, places=6)
        self.assertAlmostEqual(sum(r["cost_usd"] for r in self.stages(job).values()), 2 * 0.0475, places=6)

    def test_success_records_every_stage(self):
        job = self.new_job()
        self.assertEqual(self.go(job), "succeeded")
        j = self.store.job_get(job["id"])
        self.assertEqual((j["state"], j["attempts"]), ("succeeded", 1))
        self.assertGreater(j["bytes"], 0)
        self.assertAlmostEqual(j["cost_usd"], 0.0375, places=6, msg="the job's main stages")
        self.assertIn("basis", json.loads(j["cost_detail"]))
        st = self.stages(job)
        self.assertEqual(set(st), {"0", "1", "2", "3", "3f", "4", "6", "5"})
        self.assertTrue(all(r["state"] == "passed" for r in st.values()))
        self.assertEqual(self.runner.calls, ["0", "1", "2", "3", "3f", "4", "6", "5"])

    def test_prompt_job_draws_its_sketch_first(self):
        job = self.new_job("prompt")
        self.assertEqual(self.go(job), "succeeded")
        self.assertEqual(self.runner.calls[0], "p")
        self.assertTrue((self.storage.run_dir(self.user["id"], job["id"]) / "p-full-body.png").is_file())

    def test_rejection_shows_the_reason_and_stops(self):
        self.runner.fail = {"2": "reject"}
        job = self.new_job()
        self.assertEqual(self.go(job), "rejected")
        j = self.store.job_get(job["id"])
        self.assertEqual((j["error_stage"], j["error_code"]), ("2", "rejected"))
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
        self.assertEqual((j["error_code"], j["attempts"]), ("stage_error", 1))
        self.assertNotIn("today's heroes", j["error_message"])


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
        self.assertEqual(self.runner.calls, ["3", "3f", "4", "6", "5"], "stages 0-2 kept from attempt 1")
        self.assertEqual(self.stages(job)["1"]["message"], "kept from an earlier attempt")
        self.assertEqual(self.store.job_get(job["id"])["attempts"], 2)
        self.assertAlmostEqual(self.store.job_get(job["id"])["cost_usd"], 0.0375, places=6,
                               msg="the earlier attempt's completed model stages still count")

    def test_retries_stop_at_the_attempt_limit(self):
        self.runner.fail = {"1": "transient"}
        job = self.new_job()
        self.assertEqual(self.go(job), "queued")
        self.assertEqual(self.go(job, call="call-2"), "failed")
        j = self.store.job_get(job["id"])
        self.assertEqual(j["attempts"], 2)
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
            self.assertEqual(six["state"], "skipped")
            self.assertIn("Roblox rig", six["message"])
            self.assertEqual(self.stages(job)["5"]["state"], "passed", "the stages after it still run")

    def test_roblox_export_missing_still_delivers_the_glb(self):
        self.runner = FakeRunner(roblox=False)
        job = self.new_job()
        self.assertEqual(self.go(job), "succeeded")
        six = self.stages(job)["6"]
        self.assertEqual(six["state"], "skipped")
        self.assertIn("not part of this version", six["message"])


class TestLeases(JobCase):
    """Heartbeats: a worker that is alive keeps its lease however long a copy or a stage takes. One that died loses
    it LEASE_TTL_S after its last renewal, or, for a job, when the running stage's own lease ends if that is later
    (the held-out views hold no other lease), so the sweep finds it."""

    def setUp(self) -> None:
        super().setUp()
        from unittest import mock

        from herogen.service import worker

        patch = mock.patch.object(worker, "RENEW_EVERY_S", 0.01)
        patch.start()
        self.addCleanup(patch.stop)

    def lease(self, sql, *args):
        return self.store._one(sql, args)["lease_until"]

    def outlast(self, sql, *args, check, total_s=None):
        """Advance the clock in steps just under LEASE_TTL_S, `total_s` in all (by default past any lease the
        stage timeouts alone gave), waiting each time for the heartbeat to renew, then run the sweep and call
        `check`."""
        import time as _time

        from herogen.service.worker import LEASE_MARGIN_S, LEASE_TTL_S

        total_s = total_s or self.s.stage_timeout_s + 2 * LEASE_MARGIN_S
        step = LEASE_TTL_S - 1
        for _ in range(int(total_s // step) + 1):
            self.clock.advance(step)
            deadline = _time.monotonic() + 5
            while self.lease(sql, *args) < self.clock.t + LEASE_TTL_S - 1:
                self.assertLess(_time.monotonic(), deadline, "the lease was not renewed")
                _time.sleep(0.005)
            self.store.reconcile_stale(self.s.max_attempts)
            check()

    def test_a_long_gap_between_groups_does_not_cost_the_job_its_lease(self):
        test, job = self, self.new_job()
        sql = "SELECT lease_until FROM jobs WHERE id=?"

        def execute(j, attempt, group, resume):
            if group == "blender":   # a slow copy back, a cold start and a slow copy in, past stage 2's lease
                test.outlast(sql, job["id"], check=lambda: test.assertEqual(
                    test.store.job_get(job["id"])["state"], "running", "the sweep took a live job's lease"))
            return self.execute(j, attempt, group, resume)

        self.assertEqual(run_job(self.store, self.s, self.storage, job["id"], "c1", execute), "succeeded")
        self.assertEqual(self.store.job_get(job["id"])["attempts"], 1)

    def test_a_dead_orchestrator_loses_the_jobs_lease_within_the_ttl(self):
        from herogen.service.worker import LEASE_TTL_S

        test, job = self, self.new_job()
        sql = "SELECT lease_until FROM jobs WHERE id=?"

        def execute(j, attempt, group, resume):
            if group == "blender":
                test.outlast(sql, job["id"], check=lambda: None)   # stage 2's own lease has run out by now
                raise SystemExit("the container died")
            return self.execute(j, attempt, group, resume)

        with self.assertRaises(SystemExit):
            run_job(self.store, self.s, self.storage, job["id"], "c1", execute)
        self.clock.advance(LEASE_TTL_S - 2)
        self.assertEqual(self.store.reconcile_stale(self.s.max_attempts), [])
        self.clock.advance(3)
        self.assertEqual(self.store.reconcile_stale(self.s.max_attempts), [(job["id"], "failed")])

    def test_a_dead_orchestrator_mid_stage_holds_the_job_until_the_stages_lease_ends(self):
        from herogen.service.worker import LEASE_MARGIN_S, LEASE_TTL_S

        class DiesInStage3(FakeRunner):
            def run(self, spec, ctx):
                if spec.key == "3":
                    raise SystemExit("the container died")
                return super().run(spec, ctx)

        self.runner = DiesInStage3()
        job = self.new_job()
        with self.assertRaises(SystemExit):
            self.go(job)
        started = self.stages(job)["3"]["started_at"]
        self.clock.advance(LEASE_TTL_S + 1)
        self.assertEqual(self.store.reconcile_stale(self.s.max_attempts), [], "stage 3's own lease still holds it")
        self.clock.t = started + self.s.blender_timeout_s + LEASE_MARGIN_S + 1
        self.assertEqual(self.store.reconcile_stale(self.s.max_attempts), [(job["id"], "failed")])


class TestDeleteDuringCopyBack(JobCase):
    """A delete that lands between a copy back's check and its commit never brings the hero's files back."""

    def setUp(self) -> None:
        super().setUp()
        import shutil

        from test_herogen_service_store import SimVolume

        self.job = self.new_job()
        self.started = []
        self.assertEqual(self.go(self.job), "succeeded")
        self.shared = self.tmp / "shared"
        shutil.copytree(self.s.data_dir, self.shared)
        self.vol_worker = SimVolume(self.shared, self.tmp / "view-worker")
        self.vol_web = SimVolume(self.shared, self.tmp / "view-web")
        self.vol_worker.reload()
        self.vol_web.reload()
        self.st_worker, self.st_web = Storage(self.tmp / "view-worker"), Storage(self.tmp / "view-web")

    def web(self, delete_account: bool):
        def act():
            self.vol_web.reload()
            if delete_account:
                self.store.user_delete(self.user["id"])
                self.st_web.delete_user(self.user["id"])
            else:
                self.store.job_delete(self.job["id"], self.user["id"])
                self.st_web.delete_job(self.user["id"], self.job["id"])
            self.vol_web.commit()
        return act


    def files(self, under: Path) -> list[str]:
        return sorted(p.relative_to(under).as_posix() for p in under.rglob("*") if p.is_file()) if under.exists() else []


    def test_the_jobs_own_copy_back_is_undone_too(self):
        job = self.new_job()
        claim = self.store.job_claim(job["id"], "c", 60, 2)
        ctx = make_context(self.s, self.st_worker, claim["job"])
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        self.vol_worker.reload()
        wd = CopyWorkdir(ctx, self.tmp / "main", self.vol_worker, lambda: True,
                         still_wanted=lambda: (self.store.job_get(job["id"]) or {}).get("deleted_at") is None)
        loc = wd.open()
        (loc.run_dir / "stage-3.json").write_text("{}")

        def delete():
            self.vol_web.reload()
            self.store.job_delete(job["id"], self.user["id"])
            self.st_web.delete_job(self.user["id"], job["id"])
            self.vol_web.commit()

        self.vol_worker.before_commit = delete
        wd.sync()
        run = self.shared / "users" / self.user["id"] / "runs" / job["id"]
        self.assertEqual(self.files(run), [])
        wd.close()


class TestSweep(JobCase):
    def test_dead_worker_ends_and_next_queued_job_dispatches(self):
        job = self.new_job()
        self.store.job_claim(job['id'], 'dead', 60, 2)
        next_job = self.new_job()
        self.clock.advance(61)
        sent = []
        result = reconcile(self.store, self.s, lambda jid: sent.append(jid) or 'call-next')
        self.assertEqual(result['stale'], [(job['id'], 'failed')])
        self.assertEqual(sent, [next_job['id']])
        self.assertEqual(self.go(next_job, call='call-next'), 'succeeded')

    def test_queue_respects_the_running_cap(self):
        jobs = []
        for i in range(4):
            self.user = make_user(self.store, f"queue{i}@example.com")
            jobs.append(self.new_job())
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

    def test_default_stage3_uses_live_hybrid_with_body_joints(self):
        from herogen.service.config import Settings
        from herogen.service.runner import stage3_args

        self.assertEqual(stage3_args(Settings().stage3_args),
                         ("--mesh-source", "sam3d-objects", "--joints", "sam3d-body", "--remote", "live"))
        self.assertEqual(stage3_args("--mesh-source carve --joints silhouette"),
                         ("--mesh-source", "carve", "--joints", "silhouette"))

    def test_commands_follow_the_stage_clis(self):
        r = CliRunner(python="py")
        r._flags["herogen.blender"] = "usage: --run --stages --profile PROFILE --remote-cache DIR --quality-cache DIR"
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
        self.assertEqual(argv[argv.index("--quality-cache") + 1], str(ctx.cache_dir))
        argv, _ = r.command(specs["4"], ctx, {}, self.tmp)
        self.assertEqual(argv[argv.index("--profile") + 1], "roblox-avatar")
        argv, _ = r.command(specs["4"], self.ctx(profile="best"), {}, self.tmp)
        self.assertEqual(argv[argv.index("--profile") + 1], "roblox-ingame",
                         "a best hero's game body gets the in-game budget the Roblox export needs")
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
        self.assertNotIn("2h", specs)

    def test_pipestage_exit_codes(self):
        from unittest import mock

        from herogen.pipeline import stages as pstages
        from herogen.service import pipestage

        args = ["2", "--run-id", "r", "--runs", str(self.tmp / "runs"), "--cache", str(self.tmp / "cache")]
        with mock.patch.object(pstages, "stage2_turnaround", return_value={"name": "turnaround", "seconds": 1.0,
                                                                           "cost_usd": 0.02}), \
                mock.patch("sys.stdout"):
            self.assertEqual(pipestage.main(args), 0)
        failed = pstages.GateFailed(2, {"stage": 2, "ok": False,
                                      "checks": [{"name": "figures_found", "pass": False}]})
        with mock.patch.object(pstages, "stage2_turnaround", side_effect=failed), \
                mock.patch("sys.stderr"):
            self.assertEqual(pipestage.main(args), 2)
        with mock.patch.object(pstages, "stage2_turnaround", side_effect=RuntimeError("x")), \
                mock.patch("sys.stderr"):
            self.assertEqual(pipestage.main(args), 1)

    def test_turnaround_adapter_defaults_to_four_and_allows_serial_calls(self):
        from unittest import mock
        from herogen.pipeline import stages as pstages
        from herogen.service import pipestage
        args = ["2", "--run-id", "r", "--runs", str(self.tmp / "runs"), "--cache", str(self.tmp / "cache")]
        result = {"name": "turnaround", "seconds": 1.0, "cost_usd": 0.02}
        with mock.patch.object(pstages, "stage2_turnaround", return_value=result) as stage, \
                mock.patch("sys.stdout"):
            self.assertEqual(pipestage.main(args), 0)
            self.assertEqual(stage.call_args.kwargs["concurrency"], 4)
            self.assertEqual(pipestage.main(args + ["--turnaround-concurrency", "1"]), 0)
            self.assertEqual(stage.call_args.kwargs["concurrency"], 1)

    def test_model_key_reaches_stage3_but_not_stage4_or_export(self):
        from unittest import mock
        import subprocess

        runner = CliRunner(python="py", model_api_key="synthetic-key")
        runner._flags["herogen.blender"] = "--quality-cache"
        specs = {s.key: s for s in PLAN}
        for key in ("1", "3", "3f", "4", "6"):
            with mock.patch.object(runner, "available", return_value=None), \
                    mock.patch("herogen.service.runner.subprocess.run",
                               return_value=subprocess.CompletedProcess([], 1, "", "")) as execute:
                runner.run(specs[key], self.ctx())
            self.assertEqual("MODEL_API_KEY" in execute.call_args.kwargs["env"], key in ("1", "3", "3f"))

    def test_stage_env_forwards_fast_lane_and_request_budgets(self):
        from unittest import mock
        from herogen.service.runner import base_env

        values = {"HEROGEN_FAST_LANE": "1", "HEROGEN_MAX_IN_FLIGHT": "6",
                  "HEROGEN_RPM_IMAGE": "30", "HEROGEN_RPM_TEXT": "120",
                  "HEROGEN_REFINE_NOTE": "private per-job text"}
        with mock.patch.dict(os.environ, values, clear=True):
            env = base_env()
        for key in values:
            if key != "HEROGEN_REFINE_NOTE":
                self.assertEqual(env[key], values[key])
        self.assertNotIn("HEROGEN_REFINE_NOTE", env)
        self.assertNotIn(values["HEROGEN_REFINE_NOTE"], env.values())

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
        self.assertFalse(out.transient, "model error text cannot select retry behavior")
        out = CliRunner.outcome(spec, ctx, 1.0, 5, "", "", False)
        self.assertTrue(out.transient)
        self.assertEqual(out.code, "model_unavailable")
        (ctx.run_dir / "stage-1.json").write_text(json.dumps({"ok": True, "cost_usd": 0.02}))
        out = CliRunner.outcome(spec, ctx, 1.0, 0, "", "", False)
        self.assertEqual((out.state, out.cost_usd), ("passed", 0.02))
        (ctx.run_dir / "stage-1.json").write_text(json.dumps({"ok": False, "status": "gate_failed", "checks": [
            {"name": "sketch_mask_area_frac", "value": 0.005, "threshold": 0.01, "pass": False}]}))
        out = CliRunner.outcome(spec, ctx, 1.0, 2, "", "", False)
        self.assertEqual(out.state, "rejected")
        self.assertIn("drawn too small within the image", out.message)
        self.assertIn("0.005", out.message)
        out = CliRunner.outcome(spec, ctx, 9.0, -1, "", "", True)
        self.assertEqual((out.state, out.code, out.transient), ("failed", "timeout", True))

    def rejected_input_outcome(self, stage: str, checks: list[dict]) -> Outcome:
        """The outcome of a record like the pipeline writes when an input check rejects the upload."""
        ctx = self.ctx()
        ctx.run_dir.mkdir(parents=True)
        spec = next(s for s in PLAN if s.key == stage)
        (ctx.run_dir / spec.record).write_text(json.dumps(
            {"stage": stage, "name": spec.label, "ok": False, "status": "input_rejected",
             "error": "rejected", "checks": checks}))
        return CliRunner.outcome(spec, ctx, 5.0, 3, "stage: input rejected; stopping",
                                 "stage: input rejected; stopping", False)

    def test_input_rejection_without_a_character(self):
        message = ("no character found in the sketch: SAM 3.1 found nothing for any of 'person', 'full body', "
                   "'character'. HeroGen needs one clearly drawn, full-body character")
        out = self.rejected_input_outcome("1", [
            {"name": "upload_real_person_photo_probability", "value": 0.01, "threshold": 0.5, "op": "<",
             "pass": True, "gate": True, "verdict": "pass"},
            {"name": "sketch_characters_found", "value": 0, "threshold": 1, "op": "==", "pass": False,
             "gate": True, "verdict": "reject", "message": message},
        ])
        self.assertEqual((out.state, out.code), ("rejected", "rejected"))
        self.assertTrue(out.message.startswith("We couldn't find a character in this image."), out.message)
        self.assertIn("full body from head to feet", out.message, "the general tips follow")
        for hidden in ("sketch_characters_found", "measured", "SAM 3.1"):
            self.assertNotIn(hidden, out.message)
        self.assertIn("sketch_characters_found measured 0, needs == 1", out.detail)

    def test_input_rejection_of_several_characters(self):
        message = ("the sketch shows 2 characters: SAM 3.1 found 1 more at least 25% the size of the largest "
                   "(the next is 66% of it). Draw one character per sketch")
        out = self.rejected_input_outcome("1", [
            {"name": "sketch_characters_found", "value": 2, "threshold": 1, "op": "==", "pass": False,
             "gate": True, "verdict": "reject", "message": message},
        ])
        self.assertEqual((out.state, out.code), ("rejected", "rejected"))
        self.assertTrue(out.message.startswith("This image shows more than one character."), out.message)
        self.assertIn("full body from head to feet", out.message)
        for hidden in ("sketch_characters_found", "measured", "SAM 3.1"):
            self.assertNotIn(hidden, out.message)
        self.assertIn("sketch_characters_found measured 2, needs == 1", out.detail)

    def test_input_rejection_of_a_real_person_photo(self):
        message = ("this looks like a photograph of a real person (Muse Spark: photo_of_real_person, "
                   "probability 0.97). Upload a drawing or artwork of a character, not a photo of a real person")
        out = self.rejected_input_outcome("1", [
            {"name": "upload_real_person_photo_probability", "value": 0.97, "threshold": 0.5, "op": "<",
             "pass": False, "gate": True, "verdict": "reject", "message": message},
        ])
        self.assertEqual((out.state, out.code), ("rejected", "rejected"))
        self.assertTrue(out.message.startswith("This looks like a photo of a real person."), out.message)
        for hidden in ("upload_real_person_photo_probability", "measured", "Muse Spark", "0.97", "probability"):
            self.assertNotIn(hidden, out.message, "neither the judging model nor its probability is shown")
        self.assertIn("upload_real_person_photo_probability measured 0.97, needs < 0.5", out.detail,
                      "the measured check stays in the logs")

    def test_input_rejection_of_a_cropped_figure(self):
        message = ("the figure looks cropped: SAM 3.1 found its legs but no feet. HeroGen needs the whole body "
                   "from the top of the head to the feet; draw or scan the full figure")
        out = self.rejected_input_outcome("1", [
            {"name": "sketch_feet_on_figure", "value": 0, "threshold": 1, "op": ">=", "pass": False,
             "gate": True, "verdict": "reject", "message": message},
        ])
        self.assertEqual((out.state, out.code), ("rejected", "rejected"))
        self.assertTrue(out.message.startswith(
            "The character is cut off; HeroGen needs the full body, head to feet."), out.message)
        for hidden in ("sketch_feet_on_figure", "measured", "SAM 3.1"):
            self.assertNotIn(hidden, out.message)
        self.assertIn("sketch_feet_on_figure measured 0, needs >= 1", out.detail)

    def test_input_rejection_of_a_too_small_too_faint_sketch(self):
        out = self.rejected_input_outcome("0", [
            {"name": "min_side_px", "value": 100, "threshold": 256, "op": ">=", "pass": False, "gate": True,
             "verdict": "reject", "message": "the sketch is too small: its shorter side is 100 px and HeroGen "
             "needs at least 256 px. Scan or export it at a higher resolution"},
            {"name": "sketch_line_contrast", "value": 12.3, "threshold": 32, "op": ">=", "pass": False,
             "gate": True, "verdict": "reject", "message": "the lines are too faint: the darkest lines are only "
             "12.3 grey levels darker than the paper and HeroGen needs at least 32. Darken the lines, or scan "
             "or photograph the sketch with more contrast"},
        ])
        self.assertEqual((out.state, out.code), ("rejected", "rejected"))
        self.assertTrue(out.message.startswith("The sketch is too small: its shorter side is 100 px"), out.message)
        self.assertIn("The lines are too faint:", out.message, "each failed check's message is shown as a sentence")
        self.assertIn("full body from head to feet", out.message)
        for hidden in ("min_side_px", "sketch_line_contrast", "measured"):
            self.assertNotIn(hidden, out.message)
        self.assertIn("min_side_px measured 100, needs >= 256", out.detail)

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

    def test_prompt_resume_checks_the_legacy_stage_p_to_input_boundary(self):
        run = self.tmp / "prompt-run"
        run.mkdir()
        plan = plan_for("prompt")
        legacy = {"stage": "p", "name": "sketch_from_description", "ok": True}
        write_record(run, legacy)
        self.assertEqual(resume_index(run, plan), 0, "a legacy image cannot feed the new stage 0")
        (run / "p-full-body.png").write_bytes(png(300, 300))
        self.assertEqual(resume_index(run, plan), 0, "the version reference is required too")
        write_record(run, {**legacy, "name": "prompt_sketch", "details": {
            "prompt_version": {"name": "hero-sketch-v0", "sha256": "a" * 64}}})
        self.assertEqual(resume_index(run, plan), 1, "a valid earlier pipeline version remains reusable")
        (run / "p-full-body.png").unlink()
        self.assertEqual(resume_index(run, plan), 0, "a missing canonical image is regenerated")
        write_record(run, legacy)
        write_record(run, {"stage": 0, "ok": True})
        self.assertEqual(resume_index(run, plan), 2, "a completed legacy conversion is not rebuilt")

    def test_group_result_round_trips(self):
        g = GroupResult("error", "3", "timeout", "slow", True, 1.0, 0.1, 0.01)
        self.assertEqual(GroupResult(**g.to_dict()), g)
        self.assertIsInstance(Outcome("passed"), Outcome)
