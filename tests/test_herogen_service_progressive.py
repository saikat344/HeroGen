"""Owner-only progressive files, tolerant publication reads and measured stage timing."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest

from test_herogen_service_support import web_test
from test_herogen_service_web import Browser, WebCase
from herogen.service.fakes import FakeRunner, box_glb
from herogen.service.progressive import read_progress
from herogen.service.runner import write_record


@web_test
class ProgressiveFiles(WebCase):
    def test_mid_run_shape_versions_images_and_owner_isolation(self):
        owner = self.signin()
        other = self.signin('bob@example.com')
        job = self.make_job(owner).json()
        jid = job['id']
        seen = []
        case = self

        class Publishing(FakeRunner):
            def _stage_3(self, ctx, run):
                super()._stage_3(ctx, run)
                view = owner.get(f'/api/jobs/{jid}').json()
                case.assertEqual(view['models'], [])
                case.assertEqual(owner.get(f'/runs/r1/{jid}/03-source.glb').status_code, 404)

            def _stage_3f(self, ctx, run):
                view = owner.get(f'/api/jobs/{jid}').json()
                case.assertEqual([m['n'] for m in view['models']], [0])
                case.assertEqual(view['refinement'], {'versions': [], 'done': False})
                case.assertEqual(owner.get(f'/api/jobs/{jid}/download/glb').status_code, 200)
                case.assertEqual(owner.get(f'/api/jobs/{jid}/download/roblox').status_code, 409)
                case.check_private(owner, other, view)
                seen.append(view)
                super()._stage_3f(ctx, run)
                case.assertEqual((run / '03-refined.glb').read_bytes(), (run / '03-refined-v2.glb').read_bytes())

            def publish_version(self, run, versions, n, accepted):
                super().publish_version(run, versions, n, accepted)
                view = owner.get(f'/api/jobs/{jid}').json()
                case.assertEqual(view['state'], 'running')
                case.assertEqual([m['n'] for m in view['models']], list(range(n + 1)))
                case.assertEqual(view['refinement']['versions'][-1]['n'], n)
                case.check_private(owner, other, view)
                seen.append(view)

        self.assertEqual(self.finish(jid, Publishing()), 'succeeded')
        final = owner.get(f'/api/jobs/{jid}').json()
        self.assertTrue(final['refinement']['done'])
        self.assertEqual(final['models'][-1]['n'], 2)
        self.assertEqual(len(seen), 3)
        self.assertNotEqual(seen[1]['models'][-1]['url'], seen[2]['models'][-1]['url'])
        self.assertEqual(owner.get(seen[0]['models'][0]['url']).status_code, 200)
        self.assertEqual(owner.get(final['downloads']['glb']['url']).headers['content-type'], 'model/gltf-binary')

    def check_private(self, owner, other, view):
        for url in [*view['images'].values(), *(m['url'] for m in view['models']), view['downloads']['glb']['url']]:
            self.assertIn('?v=', url.replace('&v=', '?v='))
            response = owner.get(url)
            self.assertEqual(response.status_code, 200, url)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            self.assertEqual(other.get(url).status_code, 404, url)
            self.assertEqual(Browser(self.app).get(url).status_code, 401, url)
            self.assertEqual(owner.get(url + '-wrong').status_code, 404)

    def test_malformed_progress_never_breaks_job_or_exposes_unlisted_versions(self):
        owner = self.signin()
        jid = self.make_job(owner).json()['id']
        case = self

        class BrokenPublisher(FakeRunner):
            def _stage_3f(self, ctx, run):
                (run / '03-refined-v1.glb').write_bytes(box_glb())
                (run / '03-refined.glb').write_bytes(box_glb())
                for content in ('{', 'null', '[]', '{"versions":{},"done":false}',
                                '{"versions":[{"n":true}],"done":false}'):
                    (run / 'stage-3f-progress.json').write_text(content)
                    response = owner.get(f'/api/jobs/{jid}')
                    case.assertEqual(response.status_code, 200)
                    case.assertEqual(response.json()['refinement'], {'versions': [], 'done': False})
                    case.assertEqual([m['n'] for m in response.json()['models']], [0])
                    for rel in ('03-refined-v1.glb', '03-refined.glb', '03-Refined.glb', '03-Refined-v1.glb'):
                        case.assertEqual(owner.get(f'/runs/r1/{jid}/{rel}').status_code, 404)
                    case.assertEqual(owner.get(f'/api/jobs/{jid}/download/glb?artifact=03-refined-v1.glb').status_code, 404)
                super()._stage_3f(ctx, run)

        self.assertEqual(self.finish(jid, BrokenPublisher()), 'succeeded')

    def test_partially_written_versions_are_never_listed_or_served(self):
        owner = self.signin()
        jid = self.make_job(owner).json()['id']
        case = self

        class TruncatedPublisher(FakeRunner):
            def _stage_3f(self, ctx, run):
                versions = []
                self.publish_version(run, versions, 1, ['front'])
                self.publish_version(run, versions, 2, ['front', 'back'])
                data = (run / '03-refined-v2.glb').read_bytes()
                (run / '03-refined-v2.glb').write_bytes(data[:len(data) // 2])  # a producer crash mid-write
                view = owner.get(f'/api/jobs/{jid}').json()
                case.assertEqual([m['n'] for m in view['models']], [0, 1])
                case.assertEqual(view['refinement']['versions'][-1]['n'], 1)
                case.assertEqual(owner.get(view['models'][-1]['url']).status_code, 200)
                for url in (f'/runs/r1/{jid}/03-refined-v2.glb',
                            f'/api/jobs/{jid}/download/glb?artifact=03-refined-v2.glb'):
                    case.assertEqual(owner.get(url).status_code, 404, url)
                (run / '03-source.glb').write_bytes(b'glTF')
                view = owner.get(f'/api/jobs/{jid}').json()
                case.assertEqual([m['n'] for m in view['models']], [1])
                case.assertEqual(owner.get(f'/runs/r1/{jid}/03-source.glb').status_code, 404)
                (run / '03-source.glb').write_bytes(box_glb())
                super()._stage_3f(ctx, run)

        self.assertEqual(self.finish(jid, TruncatedPublisher()), 'succeeded')

    def test_contract_rejects_unsafe_names_bad_types_links_and_non_iso_times(self):
        run = self.tmp / 'publication'
        run.mkdir()
        versions = []
        FakeRunner().publish_version(run, versions, 1, ['front'])
        valid = {'versions': versions, 'done': False}
        self.assertEqual(len(read_progress(run)['versions']), 1)
        linked = self.tmp / 'linked-publication'
        linked.symlink_to(run, target_is_directory=True)
        self.assertEqual(read_progress(linked), {'versions': [], 'done': False})
        for field, value in (('n', True), ('n', 0), ('glb', '../03-refined-v1.glb'), ('glb', '03-refined.glb'),
                             ('accepted', 'front'), ('pending', ['../back']), ('time', 'yesterday'), ('time', None)):
            broken = json.loads(json.dumps(valid))
            broken['versions'][0][field] = value
            (run / 'stage-3f-progress.json').write_text(json.dumps(broken))
            self.assertEqual(read_progress(run), {'versions': [], 'done': False}, (field, value))
        (run / 'stage-3f-progress.json').write_text(json.dumps(valid))
        (run / '03-refined-v1.glb').unlink()
        self.assertEqual(read_progress(run)['versions'], [])
        (run / '03-refined-v1.glb').symlink_to(self.tmp / 'outside.glb')
        (self.tmp / 'outside.glb').write_bytes(box_glb())
        self.assertEqual(read_progress(run)['versions'], [])

    def test_signed_view_names_and_final_version_metadata(self):
        run = self.tmp / 'signed-views'
        run.mkdir()
        versions = []
        FakeRunner().publish_version(run, versions, 1, ['head+0', 'body-90', 'hair-upper+180'])
        valid = {'profile': 'fast', 'versions': versions, 'done': True, 'final_n': 1}
        (run / 'stage-3f-progress.json').write_text(json.dumps(valid))
        self.assertEqual(read_progress(run)['final_n'], 1)
        self.assertEqual(read_progress(run)['versions'][0]['accepted'], versions[0]['accepted'])
        for field, value in (('final_n', True), ('final_n', 2), ('final_n', None),
                             ('done', False), ('profile', '../fast')):
            broken = {**valid, field: value}
            (run / 'stage-3f-progress.json').write_text(json.dumps(broken))
            self.assertEqual(read_progress(run), {'versions': [], 'done': False}, (field, value))
        for view in ('../head+0', 'body/+90', 'body\\+90', '<head+0>'):
            broken = json.loads(json.dumps(valid))
            broken['versions'][0]['accepted'] = [view]
            (run / 'stage-3f-progress.json').write_text(json.dumps(broken))
            self.assertEqual(read_progress(run), {'versions': [], 'done': False}, view)

    @unittest.skipUnless(os.environ.get('HEROGEN_RECORDED_REFINE_SEQUENCES'),
                         'set HEROGEN_RECORDED_REFINE_SEQUENCES to both recorded sequence folders')
    def test_recorded_sequences_replay_through_reader_and_owner_job_api(self):
        roots = [Path(p) for p in os.environ['HEROGEN_RECORDED_REFINE_SEQUENCES'].split(os.pathsep)]
        self.assertEqual({p.name for p in roots}, {'s023', 's037'})
        owner = self.signin()
        other = self.signin('sequence-other@example.com')
        case = self
        for root in roots:
            with self.subTest(sequence=root.name):
                manifest = json.loads((root / 'sequence.json').read_text())
                for name, digest in manifest['files'].items():
                    self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, name)
                jid = self.make_job(owner, key=f'recorded-{root.name}').json()['id']
                seen = []

                class RecordedPublisher(FakeRunner):
                    def _stage_3f(self, ctx, run):
                        def publish(name, expected, done):
                            shutil.copyfile(root / name, run / 'stage-3f-progress.json')
                            raw = json.loads((run / 'stage-3f-progress.json').read_text())
                            progress = read_progress(run)
                            case.assertEqual([v['n'] for v in progress['versions']], expected)
                            case.assertEqual(progress['profile'], 'fast')
                            case.assertEqual(progress['done'], done)
                            case.assertEqual(progress['final_n'], expected[-1] if done else None)
                            view = owner.get(f'/api/jobs/{jid}').json()
                            case.assertEqual(view['refinement'], progress)
                            case.assertEqual([m['n'] for m in view['models']], [0, *expected])
                            for version, original in zip(progress['versions'], raw['versions']):
                                case.assertEqual(version, {k: original[k] for k in version})
                            case.check_private(owner, other, view)
                            case.assertEqual(other.get(f'/api/jobs/{jid}').status_code, 404)
                            case.assertEqual(Browser(case.app).get(f'/api/jobs/{jid}').status_code, 401)
                            seen.append((expected.copy(), done, progress['final_n']))

                        publish('stage-3f-progress-initial.json', [], False)
                        expected = []
                        for event in manifest['timeline']:
                            shutil.copyfile(root / event['glb'], run / event['glb'])
                            case.assertEqual(owner.get(f"/runs/r1/{jid}/{event['glb']}").status_code, 404)
                            expected.append(event['n'])
                            publish(event['progress'], expected, False)
                        shutil.copyfile(run / manifest['timeline'][-1]['glb'], run / '03-refined.glb')
                        publish('stage-3f-progress.json', expected, True)
                        case.assertEqual((run / '03-refined.glb').read_bytes(),
                                         (run / f"03-refined-v{expected[-1]}.glb").read_bytes())
                        write_record(run, {'stage': '3f', 'ok': True, 'status': 'complete',
                                           'outputs': ['03-refined.glb'], 'cost_usd': 0})

                self.assertEqual(self.finish(jid, RecordedPublisher()), 'succeeded')
                self.assertEqual(seen, [([], False, None), ([1], False, None), ([1, 2], False, None),
                                        ([1, 2, 3], False, None), ([1, 2, 3], True, 3)])
                final = owner.get(f'/api/jobs/{jid}').json()
                self.assertEqual(final['refinement']['final_n'], 3)
                self.assertTrue(final['refinement']['done'])
                self.assertEqual(owner.get(final['downloads']['glb']['url']).content,
                                 (root / '03-refined-v3.glb').read_bytes())

    def test_cancel_keeps_already_published_versions_available_to_owner(self):
        owner = self.signin()
        jid = self.make_job(owner).json()['id']
        case = self
        class CancelAfterVersion(FakeRunner):
            def _stage_3f(self, ctx, run):
                self.publish_version(run, [], 1, ['front'])
                response = owner.post(f'/api/jobs/{jid}/cancel', json={})
                case.assertEqual(response.status_code, 200)
                view = response.json()
                case.assertEqual(view['state'], 'canceled')
                case.assertEqual([m['n'] for m in view['models']], [0, 1])
                case.assertEqual(owner.get(view['models'][-1]['url']).status_code, 200)
        self.assertEqual(self.finish(jid, CancelAfterVersion()), 'canceled')

    def test_typical_times_need_three_measured_stage_finishes(self):
        owner = self.signin()
        for n in range(1, 4):
            jid = self.make_job(owner, key=f'timing-0000000{n}').json()['id']
            self.assertEqual(self.finish(jid), 'succeeded')
            view = owner.get(f'/api/jobs/{jid}').json()
            artwork = next(s for s in view['stages'] if s['key'] == '1')
            self.assertEqual(artwork['typical_seconds'], .01 if n == 3 else None)
            self.assertEqual(artwork['cost_usd'], .0125)
        self.assertEqual(self.store.analytics_stage_times()['1'],
                         next(s['median_seconds'] for s in self.store.analytics_report()['stages']['main'] if s['stage'] == '1'))
        self.store.db.execute("UPDATE job_stages SET state='running', seconds=999 WHERE job_id=? AND stage='1'", (jid,))
        self.assertNotIn('1', self.store.analytics_stage_times())

    def test_progressive_analytics_allow_only_version_numbers_and_keep_csrf(self):
        owner = self.signin()
        events = [{'name': 'model_first_shown'}, {'name': 'refined_version_shown', 'props': {'n': '1'}},
                  {'name': 'update_clicked', 'props': {'n': '2'}}]
        payload = {'visitor_id': 'a' * 32, 'device': 'desktop', 'events': events}
        self.assertEqual(owner.post('/api/events', json=payload, csrf=False).status_code, 403)
        self.assertEqual(owner.post('/api/events', json=payload).status_code, 200)
        for value in (0, '0', '-1', 'path.glb', '1000000000'):
            payload['events'] = [{'name': 'refined_version_shown', 'props': {'n': value}}]
            self.assertEqual(owner.post('/api/events', json=payload).status_code, 400)
