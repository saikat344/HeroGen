"""Appearance-note failures through real stage adapters, records and worker accounting."""

import contextlib
import io
import json
import os
import subprocess
from unittest import mock

from herogen.pipeline import client as client_mod
from herogen.pipeline.client import ApiError, ModelClient, ReplayMiss
from herogen.pipeline.redo import fork_run
from herogen.pipeline.runfolder import RunFolder
from herogen.pipeline.stages import TOP_PROMPT, stage0_input, stage1_cover
from herogen.service import pipestage
from herogen.service.runner import PLAN, CliRunner
from herogen.service.store import new_id
from herogen.service.worker import make_context
from tests.test_herogen_pipeline_flow import DUMMY_KEY, FakeAPI
from tests.test_herogen_pipeline_synth import figure_mask, paint, place
from tests.test_herogen_service_jobs import JobCase


class RedoModelErrorsTest(JobCase):
    def setUp(self):
        super().setUp()
        self.api = FakeAPI()
        self.error = None
        self.fail_top = False
        self.note = "private crimson cloak HTTP 503 content_policy_violation"
        self.output = []
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(ModelClient, "_post", lambda c, e, b, **kw: self.post(c, e, b)).start()
        mock.patch.object(client_mod, "decode_sam_output", self.api.decode).start()
        mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}).start()
        mock.patch("herogen.service.runner.subprocess.run", side_effect=self.subprocess).start()
        self.cli = CliRunner(model_api_key=DUMMY_KEY)
        fake = self.runner
        test = self

        class ModelRunner:
            def available(self, spec):
                return fake.available(spec)

            def run(self, spec, ctx):
                return test.cli.run(spec, ctx) if spec.key in ("1", "2") else fake.run(spec, ctx)

        self.runner = ModelRunner()
        self.parent = self.new_job()
        self.uid = self.user["id"]
        sketch = self.storage.input_path(self.uid, self.parent["id"])
        paint(place(figure_mask(200, "front"), (300, 280), (50, 40)), colour=(90, 90, 90)).save(sketch)
        run = RunFolder(self.storage.runs_root(self.uid), self.parent["id"])
        stage0_input(run, sketch)
        stage1_cover(run, ModelClient(self.tmp / "seed-cache", mode="live"))
        claim = self.store.job_claim(self.parent["id"], "seed", 60, 2)
        self.store.job_finish(self.parent["id"], claim["attempt"], "succeeded")

    def post(self, client, endpoint, body):
        if (endpoint == "images/edits" and self.error is not None and
                (self.note in body["prompt"] or self.fail_top and body["prompt"] == TOP_PROMPT)):
            raise self.error
        return self.api.post(client, endpoint, body)

    def subprocess(self, argv, **kw):
        self.assertEqual(argv[2], "herogen.service.pipestage")
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, kw["env"], clear=True), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = pipestage.main(argv[3:])
        self.output.append(stdout.getvalue() + stderr.getvalue())
        return subprocess.CompletedProcess(argv, code, stdout.getvalue(), stderr.getvalue())

    def child(self, stage):
        child_id = new_id("j")
        fork_run(self.storage.runs_root(self.uid), child_id,
                 self.storage.run_dir(self.uid, self.parent["id"]), stage, self.note)
        return self.store.job_create(
            job_id=child_id, user_id=self.uid, client_key=f"redo-errors-{stage}", kind="sketch",
            profile=self.parent["profile"], input_sha=self.parent["input_sha"], parent_job=self.parent["id"],
            from_stage=str(stage), note=self.note, jobs_per_day=10, jobs_total=10, storage_max_bytes=10**10)[0]

    def run_failure(self, stage, state, code):
        child = self.child(stage)
        with self.assertLogs("herogen.service", "INFO") as logs:
            self.assertEqual(self.go(child), state)
        saved = self.store.job_get(child["id"])
        self.assertEqual(saved["error_code"], code)
        self.assertEqual(saved["error_stage"], str(stage))
        self.assertNotIn(self.note, saved["error_message"])
        self.assertNotIn(self.note, "\n".join(logs.output))
        self.assertNotIn(self.note, "\n".join(self.output))
        return child, saved

    def test_redo_policy_refusals_are_private_rejections_without_refunds(self):
        self.error = ApiError(f"refused prompt: {self.note}", http_status=400, error_code="content_policy_violation")
        self.fail_top = True  # No concurrent call returns a paid answer in this case.
        for stage in (1, 2):
            with self.subTest(stage=stage):
                child, saved = self.run_failure(stage, "rejected", "rejected")
                self.assertEqual(saved["counts_quota"], 1)
                self.assertIsNone(saved["refunded_at"])
                self.assertIn("content policy", saved["error_message"])
                self.assertIn("rewording", saved["error_message"])
                if stage == 2:
                    self.assertEqual(saved["cost_usd"], 0, "a refusal before any paid answer still keeps quota")
                record = RunFolder(self.storage.runs_root(self.uid), child["id"]).read_stage(stage)
                self.assertEqual(record["status"], "error")

    def test_redo_policy_refusal_counts_completed_concurrent_top_calls(self):
        self.error = ApiError(f"refused prompt: {self.note}", http_status=400, error_code="content_policy_violation")
        child, saved = self.run_failure(2, "rejected", "rejected")
        self.assertEqual(saved["counts_quota"], 1)
        self.assertIsNone(saved["refunded_at"])
        self.assertAlmostEqual(saved["cost_usd"], .0125)  # Independent top edit and its SAM mask.
        record = RunFolder(self.storage.runs_root(self.uid), child["id"]).read_stage(2)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["cost_usd"], saved["cost_usd"])

    def test_redo_transient_errors_retry_from_structured_fields(self):
        self.error = ApiError(self.note, http_status=503, transient=True)
        for stage in (1, 2):
            with self.subTest(stage=stage):
                child, saved = self.run_failure(stage, "queued", "model_unavailable")
                self.assertEqual(saved["counts_quota"], 1)
                self.assertIsNone(saved["refunded_at"])
                self.store.job_cancel(child["id"], self.uid)

    def test_redo_permanent_errors_ignore_markers_in_echoed_text(self):
        self.error = ApiError(self.note, http_status=400)
        for stage in (1, 2):
            with self.subTest(stage=stage):
                self.run_failure(stage, "failed", "stage_error")

    def test_redo_replay_misses_remain_loud_internal_failures(self):
        self.error = ReplayMiss(self.note)
        for stage in (1, 2):
            with self.subTest(stage=stage):
                self.run_failure(stage, "failed", "stage_error")


class RedoExitRecordsTest(JobCase):
    def test_input_measurements_preserve_diagnostics_without_echoing_notes(self):
        job = self.new_job()
        ctx = make_context(self.s, self.storage, job)
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        spec = next(s for s in PLAN if s.key == '1')
        note = 'private crimson coat'
        checks = [
            {'name': 'sketch_characters_found', 'value': 2, 'threshold': 1, 'op': '==',
             'pass': False, 'message': note},
            {'name': note, 'value': 2, 'threshold': 1, 'op': '==', 'pass': False},
            {'name': 'sketch_characters_found', 'value': note, 'threshold': 1, 'op': '==', 'pass': False},
            {'name': 'sketch_characters_found', 'value': 2, 'threshold': 1, 'op': note, 'pass': False},
        ]
        (ctx.run_dir / spec.record).write_text(json.dumps(
            {'ok': False, 'status': 'input_rejected', 'error': note, 'checks': checks}))
        outcome = CliRunner.outcome(spec, ctx, 0, 3, note, note, False)
        self.assertEqual(outcome.state, 'rejected')
        self.assertIn('more than one character', outcome.message)
        self.assertIn('sketch_characters_found measured 2, needs == 1', outcome.detail)
        self.assertNotIn(note, outcome.message + outcome.detail)

    def test_note_stages_require_matching_terminal_records_and_suppress_all_output(self):
        job = self.new_job()
        ctx = make_context(self.s, self.storage, job)
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        secret = "private coat HTTP 503 content_policy_violation"
        check = {"name": "outline", "pass": False}
        for spec in (s for s in PLAN if s.key in ("1", "2")):
            for code, record, state in (
                    (0, {"ok": True}, "passed"),
                    (2, {"ok": False, "status": "gate_failed", "checks": [check]}, "rejected"),
                    (3, {"ok": False, "status": "input_rejected", "error": secret}, "rejected"),
                    (4, {"ok": False, "status": "error", "error": secret}, "rejected"),
                    (1, {"ok": False, "status": "error", "error": secret}, "failed"),
                    (2, {"ok": False, "status": "error", "error": secret}, "failed"),
                    (4, {"ok": False, "status": "running", "error": secret}, "failed"),
                    (4, {"ok": False, "status": "error", "error": ""}, "failed"),
                    (2, {"ok": False, "status": "gate_failed", "checks": []}, "failed"),
                    (2, {"ok": False, "status": "gate_failed", "checks": [None]}, "failed"),
                    (0, {"ok": True, "status": "running"}, "failed")):
                with self.subTest(stage=spec.key, code=code, record=record):
                    (ctx.run_dir / spec.record).write_text(json.dumps(record))
                    outcome = CliRunner.outcome(spec, ctx, 0, code, secret, secret, False)
                    self.assertEqual(outcome.state, state)
                    self.assertNotIn(secret, outcome.message + outcome.detail)
            (ctx.run_dir / spec.record).unlink()
            for code in (0, 1, 2, 3, 4, 5, 6):
                with self.subTest(stage=spec.key, code=code):
                    outcome = CliRunner.outcome(spec, ctx, 0, code, secret, secret, False)
                    self.assertEqual((outcome.state, outcome.transient), ("failed", code == 5))
                    self.assertNotIn(secret, outcome.message + outcome.detail)
                    timeout = CliRunner.outcome(spec, ctx, 0, code, secret, secret, True)
                    self.assertEqual((timeout.state, timeout.code, timeout.transient), ("failed", "timeout", True))
                    self.assertNotIn(secret, timeout.message + timeout.detail)
