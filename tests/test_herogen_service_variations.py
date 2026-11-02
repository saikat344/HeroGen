"""Immutable complete variations, input resubmission and real-run timing hints."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from unittest import mock

from test_herogen_service_support import web_test
from test_herogen_service_web import WebCase
from herogen.pipeline.redo import note_for
from herogen.pipeline.runfolder import RunFolder
from herogen.service.config import from_env
from herogen.service.fakes import FakeRunner


class RefusedRunner(FakeRunner):
    def run(self, spec, ctx):
        outcome = super().run(spec, ctx)
        if spec.key == '1':
            from herogen.service.runner import CliRunner
            return CliRunner.outcome(spec, ctx, .01, 3, '', '', False)
        return outcome

    def _stage_1(self, ctx, run):
        super()._stage_1(ctx, run)
        path = run / 'stage-1.json'
        rec = json.loads(path.read_text())
        rec.update(ok=False, status='input_rejected', error='Input check refused the sketch',
                   checks=[{'name': 'sketch_feet_on_figure', 'value': 0, 'threshold': 1, 'op': '>=', 'pass': False}])
        path.write_text(json.dumps(rec))


@web_test
class TestVariations(WebCase):
    def configure(self, **values):
        from test_herogen_service_web import Browser
        from herogen.service.web import create_app
        self.svc.settings = replace(self.s, **values)
        old = self.b
        self.b = Browser(create_app(self.svc))
        self.b.c.cookies.update(old.c.cookies)
        self.b.session()

    def setUp(self):
        super().setUp()
        self.b = self.signin()
        self.parent = self.make_job(self.b).json()['id']
        self.finish(self.parent)
        self.uid = self.store.job_get(self.parent)['user_id']
        self.run = self.svc.storage.run_dir(self.uid, self.parent)

    def redo(self, stage='texture', note='', parent=None, key='variation-0001', **extra):
        return self.b.post(f'/api/jobs/{parent or self.parent}/redo',
            json={'stage': stage, 'note': note, 'build_all': True, **extra}, headers={'Idempotency-Key': key})

    def snapshot(self):
        return {str(p.relative_to(self.run)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.run.rglob('*') if p.is_file()}

    def test_only_redos_that_rebuild_the_shape_warm_the_3d_models(self):
        warm = mock.Mock()
        self.svc.prewarm = warm
        self.assertEqual(self.redo('texture').status_code, 201)
        warm.assert_not_called()
        self.assertEqual(self.redo('cover', key='variation-0002').status_code, 201)
        self.assertEqual(warm.call_count, 1)

    def test_texture_continuation_keeps_shape_and_runs_remaining_stages(self):
        before = self.snapshot()
        response = self.redo()
        self.assertEqual(response.status_code, 201, response.text)
        child = response.json()
        self.assertEqual((child['parent_job'], child['from_stage'], child['stop_after']), (self.parent, '3f', None))
        runner = FakeRunner()
        self.assertEqual(self.finish(child['id'], runner), 'succeeded')
        self.assertEqual(runner.calls, ['3f','4','6','5'])
        folder = self.svc.storage.run_dir(self.uid, child['id'])
        for rel in ('03-source.glb', '03-mesh.glb', 'views/front.png', 'views/meta.json'):
            self.assertEqual((folder / rel).read_bytes(), (self.run / rel).read_bytes())
        rec = json.loads((folder / 'stage-3.json').read_text())
        self.assertEqual((rec['reused_from'], rec['seconds'], rec['cost_usd']), (self.parent, 0, 0))
        self.assertEqual(before, self.snapshot())

    def test_texture_note_is_cleaned_stored_and_feature_gated(self):
        self.assertFalse(self.b.session()['limits']['texture_notes'])
        self.assertEqual(self.redo(note='green coat').status_code, 400)
        self.configure(texture_notes=True)
        self.assertTrue(self.b.session()['limits']['texture_notes'])
        response = self.redo(note='«green»\ncoat')
        self.assertEqual(response.status_code, 201, response.text)
        child = response.json()
        self.assertEqual(child['note'], 'green coat')
        run = RunFolder(self.svc.storage.runs_root(self.uid), child['id'])
        self.assertEqual(note_for(run, '3f'), 'green coat')
        self.assertEqual(note_for(run, 2), '')
        self.assertTrue(from_env({'HEROGEN_TEXTURE_NOTES': '1'}, require_key=False).texture_notes)

    def test_texture_continuation_passes_its_note_to_the_cli(self):
        from unittest import mock
        from herogen.service.runner import PLAN, CliRunner, base_env

        self.configure(texture_notes=True)
        child = self.redo(note='«Blue»\ncoat 雪').json()
        commands = []
        case = self

        class InspectingRunner(FakeRunner):
            def _stage_3f(self, ctx, run):
                cli = CliRunner(python='py')
                cli._flags['herogen.blender'] = '--refine-profile --refine-note'
                spec = next(s for s in PLAN if s.key == '3f')
                env = base_env(ctx.env)
                argv, stdin = cli.command(spec, ctx, env, case.tmp)
                commands.append(argv)
                case.assertIsNone(stdin)
                case.assertEqual(argv[-2:], ['--refine-note', 'Blue coat 雪'])
                case.assertNotIn('Blue coat 雪', env.values())
                super()._stage_3f(ctx, run)

        runner = InspectingRunner()
        with mock.patch.dict('os.environ', {'HEROGEN_FAST_LANE': '1', 'HEROGEN_SERVICE_REFINE_PROFILE': ''}):
            self.assertEqual(self.finish(child['id'], runner), 'succeeded')
        self.assertEqual(runner.calls, ['3f', '4', '6', '5'])
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index('--refine-profile') + 1], 'fast')

    def test_texture_requires_inactive_parent_and_passing_stage_three(self):
        self.store.db.execute("UPDATE jobs SET state='running' WHERE id=?", (self.parent,))
        self.assertEqual(self.redo().status_code, 409)
        self.store.db.execute("UPDATE jobs SET state='succeeded' WHERE id=?", (self.parent,))
        path = self.run / 'stage-3.json'
        rec = json.loads(path.read_text()); rec['ok'] = False; path.write_text(json.dumps(rec))
        self.assertEqual(self.redo().status_code, 409)
        self.assertEqual(len(self.store.jobs_list(self.uid)), 1)

    def test_full_retry_and_refine_from_views_are_new_versions(self):
        before = self.snapshot()
        for stage, expected in (('cover', ['1','2','3','3f','4','6','5']), ('turnaround', ['2','3','3f','4','6','5'])):
            response = self.redo(stage=stage, note='' if stage == 'cover' else 'Weathered brass', key=f'variation-{stage}')
            self.assertEqual(response.status_code, 201, response.text)
            runner = FakeRunner(); child = response.json()
            self.assertNotEqual(child['id'], self.parent)
            self.assertIsNone(child['stop_after'])
            self.assertEqual(self.finish(child['id'], runner), 'succeeded')
            self.assertEqual(runner.calls, expected)
        self.assertEqual(self.snapshot(), before)

    def test_failed_variation_can_repeat_parameters_as_another_version(self):
        child = self.redo('turnaround', 'blue coat').json()['id']
        self.assertEqual(self.finish(child, FakeRunner(fail={'2': 'error'})), 'failed')
        again = self.redo('turnaround', 'blue coat', parent=child, key='variation-again').json()
        self.assertEqual((again['parent_job'], again['note'], again['from_stage']), (child, 'blue coat', '2'))
        self.assertEqual(self.finish(again['id']), 'succeeded')
        self.assertEqual(self.store.job_get(child)['state'], 'failed')
        self.assertEqual(self.store.job_get(self.parent)['state'], 'succeeded')

    def test_retry_of_selected_refinement_reruns_its_step_without_note(self):
        child = self.redo('turnaround', 'Blue coat').json()['id']
        self.finish(child)
        retry = self.redo('turnaround', '', parent=child, key='retry-selected-step').json()
        self.assertEqual((retry['parent_job'], retry['from_stage'], retry['note']), (child, '2', ''))
        runner = FakeRunner(); self.finish(retry['id'], runner)
        self.assertEqual(runner.calls[0], '2')
        self.assertEqual(self.store.job_get(child)['note'], 'Blue coat')

    def test_tree_data_carries_recorded_metrics_and_model_for_comparison(self):
        child = self.redo().json()['id']; self.finish(child)
        versions = self.b.get(f'/api/jobs/{child}/versions').json()['versions']
        self.assertEqual(len(versions), 2)
        for v in versions:
            self.assertIsInstance(v['seconds'], (int, float))
            self.assertIsInstance(v['cost_usd'], (int, float))
            self.assertTrue(v['model_url'].startswith(f"/runs/r1/{v['id']}/"))
        selected = self.b.get(f'/api/jobs/{child}').json()
        self.assertEqual(selected['recipe']['from_stage'], '3f')
        self.assertEqual(selected['recipe']['profile'], 'roblox-ingame')

    def test_images_are_published_only_after_the_stage_finishes_writing(self):
        jid = self.make_job(self.b, key='publishing-input').json()['id']
        folder = self.svc.storage.run_dir(self.uid, jid)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / '00-sketch.png').write_bytes(b'')
        self.assertNotIn('sketch', self.b.get(f'/api/jobs/{jid}').json()['images'])
        (folder / '00-sketch.png').write_bytes((self.run / '00-sketch.png').read_bytes())
        (folder / 'stage-0.json').write_bytes((self.run / 'stage-0.json').read_bytes())
        self.assertIn('sketch', self.b.get(f'/api/jobs/{jid}').json()['images'])
        self.store.db.execute("UPDATE job_stages SET state='running' WHERE job_id=? AND stage='0'", (self.parent,))
        self.assertNotIn('sketch', self.b.get(f'/api/jobs/{self.parent}').json()['images'])

    def test_all_new_routes_and_files_are_owner_only(self):
        other = self.signin('other@example.com')
        for path in (f'/api/jobs/{self.parent}/redo', f'/api/jobs/{self.parent}/retry-input'):
            response = other.post(path, json={'stage': 'texture'}, headers={'Idempotency-Key': 'other-variation'})
            self.assertEqual(response.status_code, 404, response.text)
        for path in (f'/api/jobs/{self.parent}', f'/api/jobs/{self.parent}/versions', f'/runs/r1/{self.parent}/03-source.glb'):
            self.assertEqual(other.get(path).status_code, 404)
        self.assertEqual(self.b.post(f'/api/jobs/{self.parent}/retry-input', csrf=False).status_code, 403)

    def test_full_build_request_is_idempotent_and_different_mode_conflicts(self):
        first = self.redo('turnaround')
        self.assertEqual(self.redo('turnaround').json()['id'], first.json()['id'])
        self.assertEqual(self.redo('turnaround', build_all=False).status_code, 409)
        self.assertEqual(self.redo('turnaround', build_all='yes').status_code, 400)

    def test_note_validation_and_job_rate_limits_apply(self):
        for note in ('a' * 301, None, {}, '123!'):
            response = self.redo('turnaround', note)
            self.assertEqual((response.status_code, response.json()['error']['code']), (400, 'bad_note'))
        self.configure(rate_jobs_user=(1,3600))
        self.assertEqual(self.redo().status_code, 429)

    def test_refused_input_try_again_creates_fresh_hero_with_same_input(self):
        rejected = self.make_job(self.b, key='refused-input').json()['id']
        self.assertEqual(self.finish(rejected, RefusedRunner()), 'rejected')
        view = self.b.get(f'/api/jobs/{rejected}').json()
        self.assertTrue(view['input_rejected']); self.assertIn('cut off', view['error']['message'])
        path = f'/api/jobs/{rejected}/retry-input'; headers = {'Idempotency-Key': 'refused-try-again'}
        response = self.b.post(path, json={}, headers=headers)
        self.assertEqual(response.status_code, 201, response.text)
        child = response.json()
        self.assertIsNone(child['parent_job'])
        self.assertEqual(self.b.post(path, json={}, headers=headers).json()['id'], child['id'])
        self.assertEqual(self.finish(child['id']), 'succeeded')
        new_input = self.svc.storage.run_dir(self.uid, child['id']) / '00-sketch.png'
        self.assertEqual(new_input.read_bytes(), (self.svc.storage.run_dir(self.uid, rejected) / '00-sketch.png').read_bytes())
        self.assertEqual(self.store.job_get(rejected)['state'], 'rejected')
        self.assertEqual(self.b.post(f'/api/jobs/{self.parent}/retry-input', json={}, headers=headers).status_code, 409)

    def test_typical_hints_ignore_cached_remote_and_model_calls(self):
        for n in range(3):
            jid = self.parent if n == 0 else self.make_job(self.b, key=f'measured-{n}').json()['id']
            if n: self.finish(jid)
            run = self.svc.storage.run_dir(self.uid, jid)
            self.store.db.execute("UPDATE job_stages SET seconds=17 WHERE job_id=? AND stage='3'", (jid,))
            rec = json.loads((run / 'stage-3.json').read_text())
            rec.update(cost_usd=0, mesh={'source': {'remote': {'cached': True}}})
            (run / 'stage-3.json').write_text(json.dumps(rec))
        view = self.b.get(f'/api/jobs/{self.parent}').json()
        self.assertIsNone(next(s['typical_seconds'] for s in view['stages'] if s['key'] == '3'))
        for j in self.store.jobs_list(self.uid):
            path = self.svc.storage.run_dir(self.uid, j['id']) / 'stage-3.json'
            rec = json.loads(path.read_text()); rec['mesh']['source']['remote']['cached'] = False
            path.write_text(json.dumps(rec))
        view = self.b.get(f'/api/jobs/{self.parent}').json()
        self.assertEqual(next(s['typical_seconds'] for s in view['stages'] if s['key'] == '3'), 17)
        from herogen.service.analytics import real_timing
        self.assertFalse(real_timing({'cost_usd': 0, 'calls': [{'cached': True}]}))
        self.assertTrue(real_timing({'cost_usd': 0, 'calls': [{'cached': True}, {'cached': False}]}))
        self.assertFalse(real_timing({'cost_usd': 0, 'reused_from': 'parent', 'calls': [{'cached': False}]}))
