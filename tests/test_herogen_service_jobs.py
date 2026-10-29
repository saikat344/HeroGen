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
        self.assertAlmostEqual(j["cost_usd"], 0.05, places=6)
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
