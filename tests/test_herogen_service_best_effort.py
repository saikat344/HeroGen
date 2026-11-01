"""Visitor success at model readiness, with explicit best-effort export outcomes."""
import json
import threading
from unittest.mock import patch

from test_herogen_service_jobs import JobCase
from herogen.pipeline.client import ApiError, ReplayMiss, ModelClient, IMAGE_MODEL
from herogen.service.bundle import glb_path
from herogen.service.fakes import FakeRunner
from herogen.service.runner import CliRunner, PLAN
from herogen.service.worker import ThreadBackend


class BestEffortTest(JobCase):
    def test_later_errors_deliver_model_and_explicit_notes(self):
        for stage, mode in [('3f', 'error'), ('4', 'reject'), ('4', 'error'),
                            ('6', 'error'), ('5', 'error'), ('3', 'reject')]:
            with self.subTest(stage=stage, mode=mode):
                self.runner = FakeRunner(fail={stage: mode})
                job = self.new_job()
                self.assertEqual(self.go(job), 'succeeded')
                saved = self.store.job_get(job['id'])
                self.assertEqual(saved['state'], 'succeeded')
                rows = self.stages(job)
                self.assertEqual(rows[stage]['state'], 'skipped')
                self.assertTrue(rows[stage]['message'])
                run = self.storage.run_dir(self.user['id'], job['id'])
                self.assertEqual(glb_path(run).name, '03-source.glb' if stage in ('3', '3f') else '03-refined.glb')
                rec = json.loads((run / f'stage-{stage}.json').read_text())
                self.assertFalse(rec['ok'])
                if stage == '4':
                    self.assertEqual(rows['6']['state'], 'skipped')
                    self.assertNotIn('6', self.runner.calls)
                    self.assertEqual(saved['export_state'], 'unavailable')
                if stage == '6':
                    self.assertEqual(saved['export_state'], 'unavailable')
                if stage == '3':
                    self.assertIn('rough', rows['3']['message'])
                    self.assertEqual(rows['3f']['state'], 'skipped')

    def test_rejected_shape_without_mesh_fails_clearly(self):
        class NoMesh(FakeRunner):
            def _stage_3(self, ctx, run):
                super()._stage_3(ctx, run)
                (run / '03-mesh.glb').unlink()
                (run / '03-source.glb').unlink()
        self.runner = NoMesh(fail={'3': 'reject'})
        job = self.new_job()
        self.assertEqual(self.go(job), 'failed')
        self.assertIn('no mesh was saved', self.store.job_get(job['id'])['error_message'])

    def test_early_permanent_error_keeps_its_message(self):
        self.runner = FakeRunner(fail={'1': 'error'})
        job = self.new_job()
        self.assertEqual(self.go(job), 'failed')
        self.assertIn('cover art', self.store.job_get(job['id'])['error_message'])
        self.assertNotIn('3', self.runner.calls)

    def test_model_errors_after_mesh_are_recorded_and_logged_in_full(self):
        for error in (ApiError('synthetic model error details'), ReplayMiss('synthetic cache details')):
            class ModelFailure(FakeRunner):
                def run(self, spec, ctx):
                    if spec.key == '3f':
                        raise error
                    return super().run(spec, ctx)
            self.runner = ModelFailure()
            job = self.new_job()
            with self.assertLogs('herogen.service', 'WARNING') as captured:
                self.assertEqual(self.go(job), 'succeeded')
            self.assertIn(str(error), '\n'.join(captured.output))
            run = self.storage.run_dir(self.user['id'], job['id'])
            rec = json.loads((run / 'stage-3f.json').read_text())
            self.assertIn(str(error), rec['error'])
            self.assertEqual(self.stages(job)['3f']['state'], 'skipped')

    def test_gate_off_full_hero_never_invokes_spend_checks_or_budget_hooks(self):
        from herogen.blender import refine
        from herogen.blender.refine import character
        test = self
        class RealRefine(FakeRunner):
            def run(self, spec, ctx):
                if spec.key == '3f':
                    test.assertIsNone(ctx.refine_cap_usd)
                    client = ModelClient(ctx.cache_dir, mode='replay')
                    refine.run_stage3f(ctx.run_dir, client, cap_usd=ctx.refine_cap_usd)
                return super().run(spec, ctx)
        def setup(source, work, client, **kw):
            self.assertIsNone(client.before_api_attempt)
            self.assertIsNone(client.after_api_attempt)
            client.log.add(IMAGE_MODEL, 'edit', 'synthetic', False)
            raise ReplayMiss('synthetic setup miss')
        self.store.generation_update(actor='fixture', spend_limits_enabled=False, limit_usd=0)
        self.runner = RealRefine()
        with patch.object(self.store, '_spend_exhausted', side_effect=AssertionError('spend check')), \
                patch.object(refine, 'Budget', side_effect=AssertionError('budget installed')), \
                patch.object(character, 'setup', side_effect=setup):
            job = self.new_job()
            with self.assertLogs('herogen.service', 'WARNING'):
                self.assertEqual(self.go(job), 'succeeded')
        rec = json.loads((self.storage.run_dir(self.user['id'], job['id']) / 'stage-3f.json').read_text())
        self.assertGreater(rec['cost_usd'], 0)
        self.assertNotIn('budget', rec)
        self.assertAlmostEqual(self.store.job_get(job['id'])['cost_usd'], .0375 + rec['cost_usd'])

    def test_gate_on_passes_configured_cap_to_refine_command(self):
        self.store.generation_update(actor='fixture', spend_limits_enabled=True)
        from herogen.service.worker import make_context
        ctx = make_context(self.s, self.storage, self.new_job(),
                           spend_limits_enabled=self.store.generation_get()['spend_limits_enabled'])
        spec = next(s for s in PLAN if s.key == '3f')
        runner = CliRunner()
        runner._flags['herogen.blender'] = ''
        argv, _ = runner.command(spec, ctx, {}, self.tmp)
        self.assertEqual(argv[argv.index('--refine-cap-usd')+1], str(self.s.refine_cap_usd))

    def test_success_precedes_stage4_and_frees_dispatch_slot(self):
        self.store.generation_update(actor='fixture', max_running=1)
        waiting, release, next_started = threading.Event(), threading.Event(), threading.Event()
        first = self.new_job()
        second = self.new_job()
        test = self
        class DelayedBody(FakeRunner):
            def run(self, spec, ctx):
                if ctx.job_id == second['id'] and spec.key == '0':
                    next_started.set()
                if ctx.job_id == first['id'] and spec.key == '4':
                    saved = test.store.job_get(first['id'])
                    test.assertEqual(saved['state'], 'succeeded')
                    test.assertEqual(saved['export_state'], 'running')
                    test.assertIsNotNone(glb_path(ctx.run_dir))
                    waiting.set()
                    if not release.wait(10):
                        raise RuntimeError('test export gate timed out')
                return super().run(spec, ctx)
        backend = ThreadBackend(self.store, self.s, self.storage, DelayedBody())
        try:
            from herogen.service.worker import dispatch_queued
            dispatch_queued(self.store, self.s, backend.submit)
            self.assertTrue(waiting.wait(10))
            self.assertTrue(next_started.wait(10), 'next hero blocked by export')
        finally:
            release.set()
            backend.wait()
            backend.pool.shutdown()
        self.assertEqual(self.store.job_get(first['id'])['export_state'], 'ready')
        self.assertEqual(self.store.job_get(second['id'])['state'], 'succeeded')

    def test_refine_worker_crash_after_saved_mesh_preserves_success(self):
        job = self.new_job()
        def execute(j, attempt, group, resume):
            if group == 'blender':
                original = self.runner.run
                def fail(spec, ctx):
                    if spec.key == '3f':
                        raise SystemExit('synthetic worker death')
                    return original(spec, ctx)
                self.runner.run = fail
                try:
                    self.execute(j, attempt, group, resume)
                except SystemExit:
                    raise RuntimeError('synthetic transport failure after mesh')
            return self.execute(j, attempt, group, resume)
        from herogen.service.worker import run_job
        with self.assertLogs('herogen.service', 'ERROR'):
            self.assertEqual(run_job(self.store, self.s, self.storage, job['id'], 'call', execute), 'succeeded')
        self.assertEqual(self.stages(job)['3f']['state'], 'skipped')
        record = json.loads((self.storage.run_dir(self.user['id'], job['id']) / 'stage-3f.json').read_text())
        self.assertIn('synthetic transport failure', record['error'])

    def test_analytics_separates_time_to_model_from_export_completion(self):
        job = self.new_job()
        self.store.job_claim(job['id'], 'call', 60, 2)
        self.clock.advance(12)
        self.store.job_model_ready(job['id'], 1, bytes_=123)
        self.store.job_stage_set(job['id'], 1, '4', 'passed', seconds=5, cost_usd=.2)
        self.clock.advance(7)
        self.store.job_export_finish(job['id'], 1, state='ready')
        totals = self.store.analytics_report(7)['stages']['job_totals']
        self.assertEqual(totals['time_to_model']['median_seconds'], 12)
        self.assertEqual(totals['time_to_export']['median_seconds'], 19)
        self.assertEqual(totals['export_duration']['median_seconds'], 7)
        self.assertEqual(totals['total_cost_usd'], .2)

    def test_orchestrator_loss_during_refine_recovers_saved_mesh(self):
        class LostRefine(FakeRunner):
            def run(self, spec, ctx):
                if spec.key == '3f':
                    raise SystemExit('synthetic worker loss')
                return super().run(spec, ctx)
        self.runner = LostRefine()
        job = self.new_job()
        with self.assertRaises(SystemExit):
            self.go(job)
        self.clock.advance(self.s.blender_timeout_s + 1000)
        self.store.reconcile_stale(2)
        saved = self.store.job_get(job['id'])
        self.assertEqual((saved['state'], saved['export_state']), ('succeeded', 'unavailable'))
        self.assertEqual(glb_path(self.storage.run_dir(self.user['id'], job['id'])).name, '03-source.glb')
        self.assertEqual(self.stages(job)['3f']['state'], 'skipped')
        self.assertTrue(self.stages(job)['3f']['message'])

    def test_export_worker_loss_preserves_success_and_marks_unavailable(self):
        class LostExport(FakeRunner):
            def run(self, spec, ctx):
                if spec.key == '4':
                    raise SystemExit('synthetic worker loss')
                return super().run(spec, ctx)
        self.runner = LostExport()
        job = self.new_job()
        with self.assertRaises(SystemExit):
            self.go(job)
        self.clock.advance(self.s.blender_timeout_s + 1000)
        self.store.reconcile_stale(2)
        saved = self.store.job_get(job['id'])
        self.assertEqual((saved['state'], saved['export_state']), ('succeeded', 'unavailable'))
        self.assertEqual(self.stages(job)['4']['state'], 'skipped')
        self.assertFalse(self.store.job_current(job['id'], 1))


from test_herogen_service_web import WebCase


class BestEffortHTTPTest(WebCase):
    def test_downloads_publish_at_their_output_boundaries(self):
        browser = self.signin()
        job = self.make_job(browser).json()
        test = self
        class InspectBoundaries(FakeRunner):
            def run(self, spec, ctx):
                if spec.key in ('4', '5'):
                    view = browser.get('/api/jobs/' + job['id']).json()
                    test.assertEqual(view['state'], 'succeeded')
                    test.assertEqual(view['export_state'], 'running')
                    test.assertEqual(browser.get(view['downloads']['glb']['url']).status_code, 200)
                    if spec.key == '4':
                        test.assertEqual(view['roblox_state'], 'running')
                        test.assertFalse(view['downloads']['roblox']['available'])
                    else:
                        test.assertEqual(view['roblox_state'], 'ready')
                        test.assertEqual(browser.get(view['downloads']['roblox']['url']).status_code, 200)
                return super().run(spec, ctx)
        self.assertEqual(self.finish(job['id'], InspectBoundaries()), 'succeeded')

    def test_degraded_glb_and_full_viewer_use_source_and_no_stale_roblox_download(self):
        browser = self.signin()
        job = self.make_job(browser).json()
        self.assertEqual(self.finish(job['id'], FakeRunner(fail={'3f': 'error', '4': 'reject'})), 'succeeded')
        view = browser.get('/api/jobs/' + job['id']).json()
        self.assertEqual(view['roblox_state'], 'unavailable')
        self.assertTrue(view['downloads']['glb']['available'])
        self.assertEqual(browser.get(view['downloads']['glb']['url']).status_code, 200)
        detail = browser.get('/api/runs/r1/' + job['id']).json()
        self.assertEqual(detail['bodies'][0]['high'], '03-source.glb')
        from herogen.service.worker import make_context
        saved = self.store.job_get(job['id'])
        ctx = make_context(self.s, self.svc.storage, saved)
        FakeRunner()._stage_6(ctx, ctx.run_dir)  # Unaccepted leftover files cannot become downloads.
        self.assertEqual(browser.get('/api/jobs/' + job['id'] + '/download/roblox').status_code, 404)

    def test_post_jobs_skips_spend_with_gate_off_even_after_recorded_spend(self):
        browser = self.signin()
        self.store.generation_update(actor='fixture', spend_limits_enabled=False, limit_usd=0)
        first = self.make_job(browser).json()
        self.assertEqual(self.finish(first['id']), 'succeeded')
        self.assertGreater(self.store.generation_get()['spent_usd'], 0)
        self.assertEqual(self.make_job(browser, key='second-spend-free').status_code, 201)
