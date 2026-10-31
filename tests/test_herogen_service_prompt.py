"""Prompt jobs through the service runner and the pipeline, with synthetic Model API answers only."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
from unittest import mock

from tests.test_herogen_pipeline_prompt import DUMMY_KEY, FakeAPI
from tests.test_herogen_service_jobs import JobCase

from herogen import prompts
from herogen.pipeline import client as client_mod
from herogen.pipeline import prompt_sketch
from herogen.pipeline.__main__ import main as pipeline_main
from herogen.pipeline.client import ApiError, ModelClient
from herogen.pipeline.runfolder import RunFolder
from herogen.service import bundle, labapi, pipestage
from herogen.service.runner import PLAN, CliRunner, INPUT_LABELS


def _write_json(folder, rel, data):
    """Fake record storage for these orchestration tests; directory fsync is unavailable on Windows.

    Stage functions, record content, image files and the model cache remain real. RunFolder durability has its
    own tests; this boundary lets the service tests exercise the same path on Windows and POSIX.
    """
    path = folder.file(rel)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


class StructuredAPI(FakeAPI):
    """A fake transport failure with independent machine fields and untrusted message text."""

    def __init__(self, *, error, **kw):
        super().__init__(**kw)
        self.error = error

    def post(self, client, endpoint, body, **kw):
        if self.fail == endpoint:
            self.calls.append(endpoint)
            raise self.error
        return super().post(client, endpoint, body, **kw)


class PromptJobTest(JobCase):
    def setUp(self):
        super().setUp()
        self.api = FakeAPI()
        self.cli = CliRunner(model_api_key=DUMMY_KEY)
        self.commands = []
        self.stage_outcomes = []
        self.description = "a chevalier with a violet shield"
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(ModelClient, "_post", lambda c, e, b, **kw: self.post(c, e, b, **kw)).start()
        mock.patch.object(client_mod, "decode_sam_output", lambda *a: self.api.decode(*a)).start()
        mock.patch.object(RunFolder, "write_json", _write_json).start()
        mock.patch("herogen.service.runner.subprocess.run", side_effect=self.subprocess).start()
        fake = self.runner
        test = self

        class PromptRunner:
            def available(self, spec):
                return fake.available(spec)

            def run(self, spec, ctx):
                if spec.key in ("p", "0"):
                    out = test.cli.run(spec, ctx)
                    test.stage_outcomes.append((spec.key, out))
                    return out
                return fake.run(spec, ctx)

        self.runner = PromptRunner()

    def post(self, client, endpoint, body, **kw):
        return self.api.post(client, endpoint, body, **kw)

    def subprocess(self, argv, **kw):
        """Execute the actual stage entry points in process so the fake transport cannot escape to a network."""
        self.commands.append(list(argv))
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, kw["env"], clear=True), \
                mock.patch("sys.stdin", io.StringIO(kw["input"] or "")), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if argv[2] == "herogen.service.pipestage":
                code = pipestage.main(argv[3:])
            elif argv[2] == "herogen.pipeline":
                code = pipeline_main(argv[3:])
            else:
                raise AssertionError(f"unexpected stage module {argv[2]}")
        self.subprocess_exit_code = code
        return subprocess.CompletedProcess(argv, code, stdout.getvalue(), stderr.getvalue())

    def prompt_job(self, text=None, name=""):
        job = self.new_job("prompt", name=name)
        # JobCase supplies a default description; use the store to exercise older, already queued jobs too.
        self.store.db.execute("UPDATE jobs SET prompt=? WHERE id=?", (text or self.description, job["id"]))
        return self.store.job_get(job["id"])

    def record(self, job, stage):
        run = self.storage.run_dir(self.user["id"], job["id"])
        return json.loads((run / f"stage-{stage}.json").read_text(encoding="utf-8"))

    def test_prompt_uses_pipeline_stage_p_then_its_image_and_origin(self):
        job = self.prompt_job(name="Sir Chevalier")
        with self.assertLogs("herogen.service", "INFO") as logs:
            self.assertEqual(self.go(job), "succeeded")
        run = self.storage.run_dir(self.user["id"], job["id"])
        rec = self.record(job, "p")
        self.assertEqual(rec["name"], "prompt_sketch")
        self.assertEqual(rec["details"]["prompt_version"], prompts.selected("prompt_sketch").ref())
        self.assertEqual(rec["details"]["same_character_version"], prompts.selected("same_character").ref())
        self.assertEqual(self.api.calls[:2], ["images/generations", "images/edits"])
        self.assertIn("compare", self.api.calls)
        self.assertEqual(self.commands[0][2:4], ["herogen.service.pipestage", "p"])
        self.assertEqual(self.commands[1][2:5], ["herogen.pipeline", "run", "--run-id"])
        self.assertEqual((run / "00-sketch.png").read_bytes(), (run / prompt_sketch.SKETCH).read_bytes())
        s0 = self.record(job, "0")["details"]
        self.assertEqual(s0["origin"], {"stage": "p", "image": prompt_sketch.SKETCH,
                                       "prompt_version": rec["details"]["prompt_version"]})
        self.assertEqual(s0["label"], INPUT_LABELS["prompt"])
        self.assertFalse(self.storage.input_path(self.user["id"], job["id"]).exists())
        self.assertNotIn(self.description, json.dumps(self.commands))
        self.assertNotIn(self.description, "\n".join(logs.output))
        self.assertEqual(self.commands[0][self.commands[0].index("--cache") + 1],
                         str(self.storage.cache_dir(self.user["id"])))
        self.assertIn("sketch", labapi.run_detail(run.parent, run.name)["images"])
        workspace = labapi.workspace(run.parent, [run.name], job_names={run.name: job["name"]})
        self.assertEqual(workspace["characters"][0]["title"], job["name"])
        self.assertTrue(bundle.glb_path(run, "high").is_file())
        self.assertTrue(bundle.roblox_bundle(run))

    def test_job_cost_includes_the_pipeline_stage_p_record(self):
        job = self.prompt_job()
        self.assertEqual(self.go(job), "succeeded")
        cost = self.record(job, "p")["cost_usd"]
        self.assertGreater(cost, 0.03, "design, full-body edit, segmentation and comparison all count")
        self.assertAlmostEqual(self.store.job_get(job["id"])["cost_usd"], cost + 0.0375, places=6)
        self.assertAlmostEqual(self.stages(job)["p"]["cost_usd"], cost, places=6)

    def test_retry_replays_stage_p_without_another_generation_or_cost(self):
        job = self.prompt_job()
        identity = prompt_sketch.identity_checks

        def interrupted(*a, **kw):
            identity(*a, **kw)  # every model answer is cached, but the stage has not finished
            raise ApiError(f"HTTP 503: {self.description}", http_status=503, transient=True)

        with self.assertLogs("herogen.service", "INFO") as logs:
            with mock.patch.object(prompt_sketch, "identity_checks", side_effect=interrupted):
                self.assertEqual(self.go(job), "queued")
            calls = list(self.api.calls)
            first_cost = self.record(job, "p")["cost_usd"]
            self.assertEqual(self.go(job, call="retry"), "succeeded")
        self.assertNotIn(self.description, "\n".join(logs.output))
        self.assertEqual(self.api.calls, calls, "every stage-p call is cached, including its checks")
        self.assertEqual(self.record(job, "p")["cost_usd"], 0)
        self.assertTrue(all(c["cached"] for c in self.record(job, "p")["calls"]))
        self.assertEqual([o.cost_usd for k, o in self.stage_outcomes if k == "p"], [first_cost, 0])
        self.assertAlmostEqual(self.store.job_get(job["id"])["cost_usd"], first_cost + 0.0375, places=6)

    def test_gate_failure_rejects_with_checks_and_keeps_spend(self):
        self.api = FakeAPI(edits=("two",))
        job = self.prompt_job()
        self.assertEqual(self.go(job), "rejected")
        saved = self.store.job_get(job["id"])
        self.assertEqual((saved["error_stage"], saved["error_code"]), ("p", "rejected"))
        self.assertIn("character_instances measured 2, needs == 1", saved["error_message"])
        self.assertIn("rewording", saved["error_message"])
        self.assertAlmostEqual(saved["cost_usd"], self.record(job, "p")["cost_usd"], places=6)
        self.assertGreater(saved["cost_usd"], 0)
        self.assertEqual(len(self.commands), 1)

    def test_invalid_queued_prompt_rejects_before_any_model_call(self):
        job = self.prompt_job("123 !!!")
        self.assertEqual(self.go(job), "rejected")
        saved = self.store.job_get(job["id"])
        self.assertEqual(saved["error_stage"], "p")
        self.assertIn("Describe one character in words", saved["error_message"])
        self.assertEqual(self.record(job, "p")["status"], "input_rejected")
        self.assertEqual(self.api.calls, [])

    def test_content_policy_refusal_rejects_without_echoing_the_description(self):
        self.api = StructuredAPI(fail="images/generations", error=ApiError(
            self.description, http_status=400, error_code="content_policy_violation"))
        job = self.prompt_job()
        with self.assertLogs("herogen.service", "INFO") as logs:
            self.assertEqual(self.go(job), "rejected")
        saved = self.store.job_get(job["id"])
        self.assertIn("content policy", saved["error_message"])
        self.assertNotIn(self.description, saved["error_message"])
        self.assertNotIn(self.description, "\n".join(logs.output))
        self.assertEqual(saved["attempts"], 1)

    def test_recorded_transient_failure_retries_and_permanent_failure_does_not(self):
        for status, expected in ((503, "queued"), (400, "failed")):
            with self.subTest(status=status):
                self.api = StructuredAPI(fail="images/edits", error=ApiError(
                    self.description, http_status=status, transient=status == 503))
                job = self.prompt_job(f"{self.description} number {status}")
                with self.assertLogs("herogen.service", "INFO") as logs:
                    self.assertEqual(self.go(job), expected)
                saved = self.store.job_get(job["id"])
                self.assertEqual(saved["error_code"], "model_unavailable" if status == 503 else "stage_error")
                self.assertNotIn(self.description, saved["error_message"])
                self.assertNotIn(self.description, "\n".join(logs.output))
                self.assertEqual(self.record(job, "p")["status"], "error")
                self.assertGreater(self.stages(job)["p"]["cost_usd"], 0, "the design already spent money")

    def test_exhausted_transient_failure_retains_stage_p_cost(self):
        self.api = StructuredAPI(fail="images/edits", error=ApiError(
            "unavailable", http_status=503, transient=True))
        job = self.prompt_job()
        self.assertEqual(self.go(job), "queued")
        first_cost = self.record(job, "p")["cost_usd"]
        self.assertGreater(first_cost, 0)
        self.assertAlmostEqual(self.store.job_get(job["id"])["cost_usd"], first_cost, places=6)
        self.assertEqual(self.go(job, call="retry"), "failed")
        saved = self.store.job_get(job["id"])
        self.assertAlmostEqual(saved["cost_usd"], first_cost, places=6)
        self.assertEqual(saved["counts_quota"], 1)
        self.assertEqual(self.api.calls.count("images/generations"), 1)

    def test_unexpected_error_output_and_missing_record_never_log_the_description(self):
        spec = next(s for s in PLAN if s.key == "p")
        job = self.prompt_job()
        from herogen.service.worker import make_context

        ctx = make_context(self.s, self.storage, job)
        with mock.patch("herogen.service.runner.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 1, self.description, f"HTTP 503: {self.description}")):
            with self.assertLogs("herogen.service", "INFO") as logs:
                self.assertEqual(self.go(job), "failed")
        self.assertNotIn(self.description, "\n".join(logs.output))
        out = CliRunner.outcome(spec, ctx, 1, 1, self.description, self.description, False)
        self.assertEqual(out.state, "failed")
        self.assertIn("exit_code=1", out.detail)
        out = CliRunner.outcome(spec, ctx, 1, -1, self.description, self.description, True)
        self.assertEqual((out.state, out.code), ("failed", "timeout"))
        self.assertIn("timed_out=True", out.detail)

    def test_structured_outcomes_ignore_description_and_output_markers(self):
        from herogen.pipeline.stages import GateFailed, InputRejected
        from herogen.service.runner import describe_error, describe_rejection
        from herogen.service.worker import make_context

        spec = next(s for s in PLAN if s.key == "p")
        job = self.prompt_job()
        ctx = make_context(self.s, self.storage, job)
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        descriptions = ("a plain knight", "a content_policy_violation knight", "a HTTP 503 knight",
                        "a timed out knight", "a ReplayMiss knight", "a TimeoutError URLError knight")
        for case, exit_code, expected in (
                ("pass", 0, ("passed", "", False)),
                ("input", 3, ("rejected", "rejected", False)),
                ("policy", 4, ("rejected", "rejected", False)),
                ("transient", 5, ("failed", "model_unavailable", True)),
                ("permanent", 1, ("failed", "stage_error", False)),
                ("gate", 6, ("rejected", "rejected", False))):
            reference = None
            for description in descriptions:
                with self.subTest(case=case, description=description):
                    ctx.prompt = description

                    def stage(run, client, text):
                        rec = {"ok": case == "pass", "status": "input_rejected" if case == "input" else
                               ("passed" if case == "pass" else "error"), "error": text,
                               "details": {"user_text": text}, "name": "prompt_sketch",
                               "stage": "p", "checks": [], "seconds": 0, "cost_usd": 0}
                        if case in ("pass", "gate"):
                            rec.pop("status")
                            rec.pop("error")
                        if case == "gate":
                            rec["checks"] = [{"name": "character_instances", "pass": False}]
                        _write_json(run, "stage-p.json", rec)
                        if case == "pass":
                            return rec
                        if case == "input":
                            raise InputRejected("p", rec)
                        if case == "gate":
                            raise GateFailed("p", rec)
                        try:
                            raise ApiError(text, http_status=503 if case == "transient" else 400,
                                           error_code="content_policy_violation" if case == "policy" else None,
                                           transient=case == "transient")
                        except ApiError as e:
                            raise prompt_sketch.PromptStageError(text) from e

                    with mock.patch.object(prompt_sketch, "stagep_prompt", side_effect=stage), \
                            self.assertNoLogs("herogen.service"):
                        outcome = self.cli.run(spec, ctx)
                    self.assertEqual((outcome.state, outcome.code, outcome.transient), expected)
                    self.assertNotIn(description, outcome.message + outcome.detail)
                    if reference is None:
                        reference = outcome.message
                    self.assertEqual(outcome.message, reference)
                    self.assertEqual(self.subprocess_exit_code, exit_code)
                    # Even direct callers of the helpers cannot classify stage p from error text.
                    self.assertFalse(describe_error(spec, description, False)[2])
                    self.assertNotIn("content policy", describe_rejection(spec, {"error": description}))

    def test_exit_codes_work_without_a_record_and_timeout_takes_priority(self):
        from herogen.service.worker import make_context

        spec = next(s for s in PLAN if s.key == "p")
        job = self.prompt_job()
        ctx = make_context(self.s, self.storage, job)
        for code in (1, 2, 3, 4, 5, 6):
            with self.subTest(code=code):
                outcome = CliRunner.outcome(spec, ctx, 1, code, self.description, self.description, False)
                self.assertEqual(outcome.state, "failed")
                self.assertEqual(outcome.transient, code == 5)
                if code != 5:
                    self.assertIn(f"exit_code={code}", outcome.detail)
                self.assertNotIn(self.description, outcome.message)
                timeout = CliRunner.outcome(spec, ctx, 1, code, self.description, self.description, True)
                self.assertEqual((timeout.state, timeout.code, timeout.transient), ("failed", "timeout", True))

    def test_unicode_description_travels_on_utf8_stdin_only(self):
        job = self.prompt_job("a knight called \u96ea with a caf\u00e9 shield")
        real = self.subprocess

        def execute(argv, **kw):
            self.assertEqual(kw["encoding"], "utf-8")
            self.assertEqual(kw["env"]["PYTHONIOENCODING"], "utf-8")
            return real(argv, **kw)

        with mock.patch("herogen.service.runner.subprocess.run", side_effect=execute):
            self.assertEqual(self.go(job), "succeeded")
        self.assertEqual(self.record(job, "p")["details"]["user_text"], job["prompt"])
        self.assertNotIn(job["prompt"], str(self.commands))

    def test_pipeline_version_overrides_reach_the_stage_subprocess(self):
        same = prompts.load("same-character-v1")
        self.api = FakeAPI(same=same)
        job = self.prompt_job()
        with mock.patch.dict(os.environ, {"HEROGEN_PROMPT_SKETCH_VERSION": "hero-sketch-v1",
                                          "HEROGEN_SAME_CHARACTER_VERSION": same.name}):
            self.assertEqual(self.go(job), "succeeded")
        self.assertEqual(self.record(job, "p")["details"]["same_character_version"], same.ref())
        self.assertEqual(self.record(job, "p")["details"]["prompt_version"], prompts.load("hero-sketch-v1").ref())
