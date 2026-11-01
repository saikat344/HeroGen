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
            a.log.add(IMAGE_MODEL, 'edit', 'n', True)
            a._save_png('n.png', Image.new('RGB', (4, 4), 'red'))
            a._save_entry('n', {'image': 'n.png'})
            fast.publish(parent, a)
            self.assertEqual(parent._load('n'), {'image': 'n.png'})
            self.assertEqual(len(parent.log.calls), 1)


@unittest.skipUnless(HAVE_REFINE, 'texture-refine dependencies are required')
class FastStageTests(unittest.TestCase):
    setUp = fixture.StageTests.setUp
    identity = staticmethod(fixture.StageTests.identity)

    def render_many(self, mode, args, log, **kw):
        if mode == 'render':
            for cam in args['cams']:
                self.blender(mode, {**args, 'cams': [cam]}, log, **kw)
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
        self.assertEqual(projected, ['head+0', 'body+0', 'body+90', 'body-90', 'body+180', 'upper+180'])
        self.assertEqual(len(set(self.renders[:5])), 1)
        self.assertTrue(self.renders[5].endswith('body+180/texture.png'))
        self.assertEqual(sum(v['accepted'] for v in rec['views'].values()), 6)
        self.assertEqual(rec['thresholds']['outline'], fixture.refine.GATE)
        self.assertEqual(len(rec['calls']), 31)  # six edits, four masks each, plus hair head mask

    def test_third_head_attempt_is_preserved(self):
        original = FakeClient.edit_image
        def edit(child, prompt, images, **kw):
            if child.cache_dir.name == 'head+0' and kw['attempt'] < 2:
                child._call(IMAGE_MODEL, 'edit')
                return Image.new('RGB', images[0].size, (115, 115, 115))
            return original(child, prompt, images, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertTrue(rec['views']['head+0']['accepted'])
        self.assertEqual(rec['views']['head+0']['tries'], 3)

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
        self.assertCountEqual(sent, [prompts.prompt(v['prompt'], v['camera']['angle'])
                                     for v in rec['views'].values()])
        self.assertEqual(sum('Only make it sharp' in text for text in sent), 5)
        self.assertFalse(any('scratches, pitting, rust' in text for text in sent))
        for call in self.tex.apply.call_args_list:
            cam, _, feather = call.args[:3]
            expected = .25 if cam['name'] in ('head+0', 'upper+180') else 1
            np.testing.assert_array_equal(feather, expected)
        self.assertEqual(rec['thresholds']['closeup_feather_px'], imaging.CLOSEUP_FEATHER_PX)
        self.assertEqual(rec['thresholds']['exposure_mode'], 'off')

    def test_missing_head_retains_body_and_upper_back_passes(self):
        self.info['head_centre'] = None
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(rec['views']['head+0']['reason'], 'head_not_found')
        self.assertEqual(sum(v['accepted'] for v in rec['views'].values()), 5)

    def test_model_error_drains_other_passes_and_stays_loud(self):
        original = FakeClient.edit_image
        def edit(child, *args, **kw):
            if child.cache_dir.name == 'body+90':
                raise ApiError('offline fixture')
            return original(child, *args, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            with self.assertRaises(ApiError):
                stage.run_stage3f(self.run, self.client, refine_profile='fast')
        rec = json.loads((self.run / 'stage-3f.json').read_text())
        self.assertEqual(rec['status'], 'error')
        self.assertFalse((self.run / '03-refined.glb').exists())
        self.assertNotIn('hair-upper+180', rec['views'])
        self.assertGreater(len(rec['calls']), 0)

    def test_zero_cap_skips_all_paid_attempts_without_overspend(self):
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast', cap_usd=0)
        self.assertEqual(rec['status'], 'cap_reached')
        self.assertEqual(rec['budget']['spent_usd_total'], 0)
        self.assertFalse(rec['views']['hair-upper+180']['accepted'])

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


class PlumbingTests(unittest.TestCase):
    def test_service_setting_and_deploy_selector(self):
        from herogen.service.config import ConfigError, Settings, env_for_deploy, from_env
        default = Settings()
        selected = from_env({'HEROGEN_SESSION_KEY': 'x' * 40, 'HEROGEN_SERVICE_REFINE_PROFILE': 'fast'})
        self.assertEqual(selected.refine_profile, 'fast')
        self.assertEqual(default.refine_profile, 'best')
        self.assertEqual(env_for_deploy({'HEROGEN_SERVICE_REFINE_PROFILE': 'fast'}),
                         {'HEROGEN_SERVICE_REFINE_PROFILE': 'fast'})
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
