"""Frozen-wave concurrency, accounting, cache isolation and opt-in plumbing."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from PIL import Image

from herogen.blender import refine as stage
from herogen.blender.refine import fast
from herogen.pipeline.client import ApiError, IMAGE_MODEL, ModelClient
from tests import test_herogen_blender_refine as fixture
from tests.test_herogen_blender_refine import FakeClient, HAVE_REFINE


class ConcurrentBudgetTests(unittest.TestCase):
    def test_estimate_reserves_all_parallel_requests(self):
        self.assertEqual(stage.cost_estimate(refine_profile='fast'), .35)
        self.assertEqual(stage.cost_estimate(True, refine_profile='fast'), .35)

    def test_atomic_limit_and_thread_local_settlement(self):
        client = ModelClient('unused')
        budget = stage.Budget(client, .03)
        barrier = threading.Barrier(3)

        def attempt():
            budget.reserve(IMAGE_MODEL)
            token = budget.active
            barrier.wait(timeout=5)
            budget.settle(.005)
            return token

        with ThreadPoolExecutor(max_workers=3) as pool:
            tokens = list(pool.map(lambda _: attempt(), range(3)))
        self.assertEqual(len(set(tokens)), 3)
        self.assertEqual(budget.spent(), .015)
        self.assertEqual(budget.state['pending_attempt_usd'], 0)
        self.assertEqual(len(budget.state['attempts']), 3)

    def test_racing_reservations_cannot_overspend(self):
        budget = stage.Budget(ModelClient('unused'), .01)
        def attempt(_):
            try:
                budget.reserve(IMAGE_MODEL)
                return True
            except stage.BudgetStop:
                return False
        with ThreadPoolExecutor(max_workers=6) as pool:
            self.assertEqual(sum(pool.map(attempt, range(30))), 1)
        self.assertEqual(budget.spent(), .01)

    def test_private_cache_reuses_root_and_publishes_complete_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = ModelClient(tmp)
            parent._save_png('k.png', Image.new('RGB', (4, 4), 'blue'))
            parent._save_entry('k', {'image': 'k.png'})
            a, b = fast.isolated_client(parent, 'a'), fast.isolated_client(parent, 'b')
            self.assertEqual(a._load('k'), {'image': 'k.png'})
            self.assertTrue((a.cache_dir / 'k.png').is_file())
            self.assertFalse((b.cache_dir / 'k.png').exists())
            self.assertIsNot(a.log, b.log)
            self.assertIs(a.limiter, parent.limiter)
            self.assertIs(a.before_api_attempt, parent.before_api_attempt)
            a.log.add(IMAGE_MODEL, 'edit', 'n', True)
            a._save_png('n.png', Image.new('RGB', (4, 4), 'red'))
            a._save_entry('n', {'image': 'n.png'})
            fast.publish(parent, a)
            self.assertEqual(parent._load('n'), {'image': 'n.png'})
            self.assertEqual(len(parent.log.calls), 1)


@unittest.skipUnless(HAVE_REFINE, 'texture-refine dependencies are required')
class FastStageTests(unittest.TestCase):
    def setUp(self):
        fixture.StageTests.setUp(self)
        def save(path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (4, 4), 'blue').save(path)
            return path
        self.tex.save.side_effect = save
        import numpy as np
        self.tex.render_view.return_value = (np.zeros((64, 64, 3)), np.zeros((64, 64), bool))
    identity = staticmethod(fixture.StageTests.identity)

    def render_many(self, mode, args, log, **kw):
        if mode == 'render':
            for cam in args['cams']:
                self.blender(mode, {**args, 'cams': [cam]}, log, **kw)
                (Path(args['out']) / (cam['name'] + '.pos.exr')).write_bytes(b'position fixture')
        else:
            self.blender(mode, args, log, **kw)

    def test_parallel_edits_frozen_inputs_and_ordered_projection(self):
        barrier = threading.Barrier(5)
        lock = threading.Lock()
        started = []
        original = FakeClient.edit_image
        main = threading.get_ident()
        projected = []
        def edit(child, *args, **kw):
            with lock:
                started.append(threading.get_ident())
                index = len(started)
            if index <= 5:
                barrier.wait(timeout=5)
            return original(child, *args, **kw)
        def project(cam, *args):
            self.assertEqual(threading.get_ident(), main)
            projected.append(cam['name'])
            return 17
        self.tex.apply.side_effect = project
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(rec['status'], 'complete')
        self.assertEqual(len(set(started[:5])), 5)
        self.assertEqual(projected[:5], ['head+0', 'body+0', 'body+90', 'body-90', 'body+180'])
        self.assertEqual(projected[-6:], ['head+0', 'body+0', 'body+90', 'body-90', 'body+180', 'upper+180'])
        self.assertEqual(len(set(self.renders[:5])), 1)
        self.assertTrue(self.renders[5].endswith('texture-v1.png'))
        self.assertEqual(sum(v['accepted'] for v in rec['views'].values()), 6)
        self.assertEqual(rec['thresholds']['outline'], fixture.refine.GATE)
        self.assertGreaterEqual(len(rec['calls']), 31)  # concurrent attempts can finish before cancellation

    def test_third_head_attempt_is_preserved(self):
        original = FakeClient.edit_image
        def edit(child, prompt, images, **kw):
            if 'head+0' in child.cache_dir.parts and kw['attempt'] < 2:
                child._call(IMAGE_MODEL, 'edit')
                return Image.new('RGB', images[0].size, (115, 115, 115))
            return original(child, prompt, images, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertTrue(rec['views']['head+0']['accepted'])
        self.assertEqual(rec['views']['head+0']['tries'], 3)


    def test_parallel_attempts_choose_lowest_passing_index(self):
        original = FakeClient.edit_image
        barrier = threading.Barrier(3)
        seen = []
        def edit(child, prompt, images, **kw):
            if 'head+0' in child.cache_dir.parts:
                seen.append(kw['attempt'])
                barrier.wait(timeout=5)
                if kw['attempt'] == 0:
                    child._call(IMAGE_MODEL, 'edit')
                    return Image.new('RGB', images[0].size, (115, 115, 115))
            return original(child, prompt, images, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertCountEqual(seen, [0, 1, 2])
        self.assertEqual(rec['views']['head+0']['selected_attempt'], 1)
        self.assertEqual(rec['views']['head+0']['attempts'][0]['reason'], 'outline moved')

    def test_default_recipe_and_feather_reach_ordered_projection(self):
        from herogen.blender.refine import imaging
        from herogen.blender.refine import prompts
        import numpy as np

        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(imaging, 'feather_map', side_effect=lambda *a: np.ones((64, 64), np.float32)), \
             mock.patch.object(imaging, 'closeup_feather', return_value=np.full((64, 64), .25)), \
             mock.patch.object(imaging, 'rebake_views'):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(rec['recipes'], ['RF2R', 'RF3H'])
        self.assertEqual(rec['repaint_recipe'], 'RF2R')
        sent = [text for text, _, _ in self.client.inputs]
        self.assertEqual(set(sent), {prompts.prompt(v['prompt'], v['camera']['angle'])
                                     for v in rec['views'].values()})
        self.assertGreaterEqual(sum('Only make it sharp' in text for text in sent), 5)
        self.assertFalse(any('scratches, pitting, rust' in text for text in sent))
        for call in self.tex.apply.call_args_list:
            cam, _, feather = call.args[:3]
            expected = .25 if cam['name'] in ('head+0', 'upper+180') else 1
            np.testing.assert_array_equal(feather, expected)
        self.assertEqual(rec['thresholds']['closeup_feather_px'], imaging.CLOSEUP_FEATHER_PX)
        self.assertEqual(rec['thresholds']['exposure_mode'], 'off')

    def test_frozen_renders_support_final_feather_rebake(self):
        from herogen.blender.refine import imaging
        import numpy as np

        self.tex.U = np.ones(1)
        self.tex.render_view.return_value = (np.zeros((64, 64, 3)), np.zeros((64, 64), bool))
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(imaging, 'load_accepted_views', wraps=imaging.load_accepted_views) as load:
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(rec['status'], 'complete')
        self.assertEqual(load.call_count, 2)
        self.assertEqual(self.tex.reset.call_count, 2)
        self.assertEqual(self.tex.apply.call_count, 11)  # five first-wave views, then the full six-view version
        for key, view in rec['views'].items():
            folder = self.run / 'blender-work' / 'texture-refine' / key
            self.assertTrue((folder / (view['camera']['name'] + '.png')).is_file())
            self.assertEqual((folder / (view['camera']['name'] + '.pos.exr')).read_bytes(), b'position fixture')

    def test_missing_head_retains_body_and_upper_back_passes(self):
        self.info['head_centre'] = None
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(rec['views']['head+0']['reason'], 'head_not_found')
        self.assertEqual(sum(v['accepted'] for v in rec['views'].values()), 5)

    def test_model_error_drains_other_passes_and_stays_loud(self):
        original = FakeClient.edit_image
        def edit(child, *args, **kw):
            if 'body+90' in child.cache_dir.parts:
                raise ApiError('offline fixture')
            return original(child, *args, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            with self.assertRaises(ApiError):
                stage.run_stage3f(self.run, self.client, refine_profile='fast')
        rec = json.loads((self.run / 'stage-3f.json').read_text())
        self.assertEqual(rec['status'], 'error')
        self.assertFalse((self.run / '03-refined.glb').exists())
        self.assertEqual(rec['views']['hair-upper+180']['reason'], 'not attempted: model error')
        self.assertGreater(len(rec['calls']), 0)

    def test_zero_cap_skips_all_paid_attempts_without_overspend(self):
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast', cap_usd=0)
        self.assertEqual(rec['status'], 'cap_reached')
        self.assertEqual(rec['budget']['spent_usd_total'], 0)
        self.assertFalse(rec['views']['hair-upper+180']['accepted'])

    def test_first_attempt_selection_does_not_wait_for_earlier_pass(self):
        original_edit = FakeClient.edit_image
        original_collect = fast.Pass.collect
        body_selected = threading.Event()
        def edit(child, prompt, images, **kw):
            if 'head+0' in child.cache_dir.parts and kw['attempt'] == 0:
                self.assertTrue(body_selected.wait(5))
            return original_edit(child, prompt, images, **kw)
        def collect(job, index):
            original_collect(job, index)
            if job.key == 'body+0' and index == 0:
                self.assertTrue(job.cancelled.is_set())
                body_selected.set()
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit), \
             mock.patch.object(fast.Pass, 'collect', collect):
            stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertTrue(body_selected.is_set())

    def test_progress_contract_atomic_order_and_final_copy(self):
        import os
        replaced = []
        replace = os.replace
        def checked(source, target):
            source, target = Path(source), Path(target)
            if target.name == 'stage-3f-progress.json':
                progress = json.loads(source.read_text())
                for version in progress['versions']:
                    self.assertTrue((self.run / version['glb']).is_file())
            replaced.append(target.name)
            return replace(source, target)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(fast.os, 'replace', side_effect=checked):
            stage.run_stage3f(self.run, self.client, refine_profile='fast')
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertEqual(set(progress), {'profile', 'versions', 'done', 'final_n'})
        self.assertEqual(progress['profile'], 'fast')
        self.assertIs(progress['done'], True)
        self.assertEqual(progress['final_n'], 2)
        previous = set()
        for n, version in enumerate(progress['versions'], 1):
            self.assertEqual(set(version), {'n', 'glb', 'accepted', 'rejected', 'pending', 'time',
                                            'atlas_sha256', 'code_sha', 'wave'})
            self.assertEqual(version['n'], n)
            self.assertEqual(version['wave'], n)
            self.assertGreater(set(version['accepted']), previous)
            previous = set(version['accepted'])
            self.assertLess(replaced.index(version['glb']), replaced.index('stage-3f-progress-v%d.json' % n))
            self.assertEqual(len(version['atlas_sha256']), 64)
            snapshot = json.loads((self.run / ('stage-3f-progress-v%d.json' % n)).read_text())
            self.assertFalse(snapshot['done'])
            self.assertEqual(snapshot['versions'][-1], version)
        last = self.run / progress['versions'][-1]['glb']
        self.assertEqual((self.run / '03-refined.glb').read_bytes(), last.read_bytes())

    def test_failed_export_keeps_previous_progress_pointer(self):
        exports = []
        def blender(mode, args, log, **kw):
            if mode == 'export':
                exports.append(args['out'])
                if len(exports) == 2:
                    Path(args['out']).write_bytes(b'partial')
                    raise RuntimeError('export fixture')
            return self.render_many(mode, args, log, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=blender):
            with self.assertRaisesRegex(RuntimeError, 'export fixture'):
                stage.run_stage3f(self.run, self.client, refine_profile='fast')
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertFalse(progress['done'])
        self.assertEqual(len(progress['versions']), 1)
        self.assertTrue((self.run / progress['versions'][0]['glb']).is_file())
        self.assertFalse((self.run / '03-refined-v2.part.glb').exists())
        self.assertFalse((self.run / '03-refined.glb').exists())
        record = json.loads((self.run / 'stage-3f.json').read_text())
        self.assertGreater(len(record['calls']), 0)  # export failures still drain/publish paid call evidence

    def test_all_rejected_has_no_version_and_finished_fallback(self):
        self.client.reject = True
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            record = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(record['final_model'], '03-source.glb')
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertEqual(progress, {'profile': 'fast', 'versions': [], 'done': True, 'final_n': None})
        self.assertFalse(list(self.run.glob('03-refined*.glb')))

    def test_retry_publication_cannot_change_wave_two_frozen_input(self):
        original = FakeClient.edit_image
        wave_two_started = threading.Event()
        def edit(child, prompt, images, **kw):
            if 'head+0' in child.cache_dir.parts:
                if kw['attempt'] == 0:
                    child._call(IMAGE_MODEL, 'edit')
                    return Image.new('RGB', images[0].size, (115, 115, 115))
                self.assertTrue(wave_two_started.wait(5))
            elif 'hair-upper+180' in child.cache_dir.parts:
                wave_two_started.set()
            return original(child, prompt, images, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            record = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        first = progress['versions'][0]
        self.assertNotIn('head+0', first['accepted'])
        self.assertIn('head+0', first['pending'])
        self.assertIn('hair-upper+180', first['pending'])
        self.assertIn('head+0', progress['versions'][-1]['accepted'])
        self.assertEqual(record['views']['head+0']['selected_attempt'], 1)
        self.assertTrue(self.renders[5].endswith('texture-v1.png'))
        self.assertEqual(record['waves'][1]['frozen_atlas'], 'texture-v1.png')
        for before, after in zip(progress['versions'], progress['versions'][1:]):
            self.assertLess(set(before['accepted']), set(after['accepted']))

    def test_bad_profile_and_upscale_combination_fail_before_output_deletion(self):
        final = self.run / '03-refined.glb'
        final.write_bytes(b'keep')
        for kw in ({'refine_profile': 'unknown'}, {'refine_profile': 'fast', 'upscale_texture': True}):
            with self.assertRaises(ValueError):
                stage.run_stage3f(self.run, self.client, **kw)
            self.assertEqual(final.read_bytes(), b'keep')

    def test_default_omits_fast_metadata(self):
        rec = stage.run_stage3f(self.run, self.client)
        self.assertNotIn('refine_profile', rec)
        self.assertNotIn('waves', rec)
        self.assertEqual(len(rec['views']), 20)


@unittest.skipUnless(HAVE_REFINE, 'texture-refine dependencies are required')
class ReprojectionTests(unittest.TestCase):
    def test_reprojection_resets_base_and_reproduces_atlas_hash(self):
        import hashlib
        import numpy as np
        from herogen.blender.refine import cameras, character, imaging as I
        xs, zs = np.meshgrid(np.linspace(-.2, .2, 16), np.linspace(.1, .9, 24))
        pos = np.stack([xs, np.zeros_like(xs), zs], -1)
        normal = np.broadcast_to([0., -1., 0.], pos.shape).copy()
        base = np.full(pos.shape, .1, np.float32)
        cam = cameras.camera('body+0', [0., 0., .5], 0, 1., (64, 64))
        ones = np.ones((64, 64), np.float32)
        view = {'key': 'body+0', 'cam': cam, 'hair': False, 'closeup': False,
                'alpha': ones, 'refined': np.full((64, 64, 3), .7, np.float32),
                'base_feather': ones, 'weight': ones, 'depth': ones * 2,
                'valid': ones.astype(bool)}
        with mock.patch.object(character, 'BODY_RES', (64, 64)), \
             mock.patch.object(I, 'load_accepted_views', return_value=[view]):
            texture = I.Texture.from_arrays(pos, normal, np.ones(pos.shape[:2], bool), base, 1.)
            records = {'body+0': {'accepted': True}}
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'texture.png'
                fast.reproject(texture, Path(tmp), records, cam)
                texture.save(path)
                first = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertGreater(float(texture.T.mean()), .6)
                texture.T[:] = 0  # rebuilding must not depend on the previous blended atlas
                fast.reproject(texture, Path(tmp), records, cam)
                texture.save(path)
                self.assertEqual(first, hashlib.sha256(path.read_bytes()).hexdigest())


class PlumbingTests(unittest.TestCase):
    def test_service_setting_and_cost_binding(self):
        from herogen.service.config import ConfigError, Settings, from_env
        default = Settings()
        selected = from_env({'HEROGEN_SESSION_KEY': 'x' * 40, 'HEROGEN_SERVICE_REFINE_PROFILE': 'fast'})
        self.assertEqual(selected.refine_profile, 'fast')
        self.assertNotEqual(default.cost_signature, default.with_(refine_profile='fast').cost_signature)
        with self.assertRaises(ConfigError):
            from_env({'HEROGEN_SESSION_KEY': 'x' * 40, 'HEROGEN_SERVICE_REFINE_PROFILE': 'unknown'})

    def test_service_runner_passes_opt_in_only_when_selected(self):
        from herogen.service.runner import CliRunner, PLAN, StageContext
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = StageContext(job_id='job', kind='sketch', profile='best', name='', prompt='',
                               run_dir=root, input_path=root/'input.png', cache_dir=root/'cache',
                               remote_cache_dir=root/'remote')
            runner = CliRunner(python='py')
            runner._flags['herogen.blender'] = '--refine-profile'
            spec = next(s for s in PLAN if s.key == '3f')
            default, _ = runner.command(spec, ctx, {}, root)
            self.assertNotIn('--refine-profile', default)
            ctx.refine_profile = 'fast'
            selected, _ = runner.command(spec, ctx, {}, root)
            self.assertEqual(selected[selected.index('--refine-profile') + 1], 'fast')

    def test_cli_opt_in(self):
        from herogen.blender.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'stage-3.json').write_text('{"ok":true}')
            with mock.patch.object(stage, 'run_stage3f', return_value={'ok': True}) as called:
                self.assertEqual(main(['run', '--run', tmp, '--stages', '3f', '--refine-profile', 'fast']), 0)
            self.assertEqual(called.call_args.kwargs['refine_profile'], 'fast')
