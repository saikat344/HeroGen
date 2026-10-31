"""Stage-p exit and record agreement, using local usage errors and fake pipeline records."""

import io
import json
from unittest import mock

from tests.test_herogen_service_jobs import JobCase

from herogen.pipeline.stages import GateFailed
from herogen.service import pipestage
from herogen.service.runner import PLAN, CliRunner
from herogen.service.worker import make_context


class TestPromptExitRecords(JobCase):
    def setUp(self):
        super().setUp()
        self.job = self.new_job("prompt")
        self.ctx = make_context(self.s, self.storage, self.job)
        self.ctx.run_dir.mkdir(parents=True, exist_ok=True)
        self.spec = next(s for s in PLAN if s.key == "p")
        self.description = "a private knight HTTP 503 content_policy_violation"

    def outcome(self, code, record=None):
        path = self.ctx.run_dir / "stage-p.json"
        path.unlink(missing_ok=True)
        if record is not None:
            path.write_text(json.dumps(record), encoding="utf-8")
        return CliRunner.outcome(self.spec, self.ctx, 0, code, self.description, self.description, False)

    def assert_internal(self, outcome, code):
        self.assertEqual((outcome.state, outcome.code, outcome.transient), ("failed", "stage_error", False))
        self.assertIn("internal error", outcome.message)
        self.assertIn(f"exit_code={code}", outcome.detail)
        self.assertNotIn(self.description, outcome.message + outcome.detail)

    def test_real_argparse_error_is_logged_and_refunded_as_an_internal_failure(self):
        class UsageErrorRunner(CliRunner):
            def command(self, *args):
                argv, stdin = super().command(*args)
                return [*argv, "--bogus"], stdin

        self.runner = UsageErrorRunner()
        with self.assertLogs("herogen.service", "WARNING") as logs:
            self.assertEqual(self.go(self.job), "failed")
        self.assertIn("exit_code=2", "\n".join(logs.output))
        self.assertNotIn(self.job["prompt"], "\n".join(logs.output))
        saved = self.store.job_get(self.job["id"])
        self.assertEqual((saved["error_code"], saved["attempts"], saved["counts_quota"]), ("stage_error", 1, 0))
        self.assertIsNotNone(saved["refunded_at"])
        self.assertEqual(saved["cost_usd"], 0)
        self.assertFalse((self.ctx.run_dir / "stage-p.json").exists())

    def test_exit_two_without_record_is_internal(self):
        self.assert_internal(self.outcome(2), 2)

    def test_exit_two_with_running_record_is_internal(self):
        self.assert_internal(self.outcome(2, {"ok": False, "status": "running", "error": self.description}), 2)

    def test_adapter_gate_exit_six_and_terminal_record_preserve_failed_checks(self):
        record = {"stage": "p", "ok": False, "checks": [
            {"name": "character_instances", "gate": True, "pass": False, "value": 2,
             "op": "==", "threshold": 1}], "cost_usd": 0.25}
        args = ["p", "--run-id", self.job["id"], "--runs", str(self.ctx.runs_root),
                "--cache", str(self.ctx.cache_dir)]

        def write_stage(run, rec):
            (run.path / "stage-p.json").write_text(json.dumps(rec), encoding="utf-8")

        with mock.patch("herogen.pipeline.client.ModelClient"), \
                mock.patch("herogen.pipeline.prompt_sketch.stagep_prompt", side_effect=GateFailed("p", record)), \
                mock.patch("herogen.pipeline.runfolder.RunFolder.write_stage", write_stage), \
                mock.patch("sys.stdin", io.StringIO(self.description)), mock.patch("sys.stderr", io.StringIO()):
            code = pipestage.main(args)
        self.assertEqual(code, 6)
        saved = json.loads((self.ctx.run_dir / "stage-p.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "gate_failed")
        self.assertNotIn("error", saved)
        outcome = self.outcome(code, saved)
        self.assertEqual((outcome.state, outcome.cost_usd), ("rejected", 0.25))
        self.assertIn("character_instances measured 2, needs == 1", outcome.message)

    def test_exit_six_without_consistent_gate_record_is_internal(self):
        check = {"name": "character_instances", "pass": False}
        for record in (None, {}, {"ok": False, "status": "running"},
                       {"ok": True, "status": "gate_failed", "checks": [check]},
                       {"ok": False, "status": "gate_failed", "checks": [check], "error": ""},
                       {"ok": False, "status": "error", "checks": [check]},
                       {"ok": False, "status": "gate_failed", "checks": []},
                       {"ok": False, "status": "gate_failed", "checks": 1},
                       {"ok": False, "status": "gate_failed", "checks": [None]}):
            with self.subTest(record=record):
                self.assert_internal(self.outcome(6, record), 6)

    def test_input_and_policy_rejections_require_matching_error_records(self):
        for code, status in ((3, "input_rejected"), (4, "error")):
            record = {"ok": False, "status": status, "error": self.description}
            with self.subTest(code=code):
                self.assertEqual(self.outcome(code, record).state, "rejected")
                for inconsistent in (None, {**record, "ok": True}, {**record, "status": "running"},
                                     {**record, "status": "gate_failed"}, {**record, "error": ""},
                                     {"ok": False, "status": status}):
                    with self.subTest(record=inconsistent):
                        self.assert_internal(self.outcome(code, inconsistent), code)
                self.assert_internal(self.outcome(1, record), 1)
                self.assert_internal(self.outcome(2, record), 2)
        for inconsistent in ({"ok": True, "status": "running"}, {"ok": True, "error": self.description}):
            with self.subTest(record=inconsistent):
                self.assert_internal(self.outcome(0, inconsistent), 0)
