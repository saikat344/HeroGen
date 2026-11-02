"""Frozen-wave concurrency, accounting, cache isolation and opt-in plumbing."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from PIL import Image

from herogen.blender import refine as stage
from herogen.blender.refine import fast
from herogen.pipeline.client import ApiError, IMAGE_MODEL, SAM_MODEL, PRICE_PER_IMAGE_USD, ModelClient
from tests import test_herogen_blender_refine as fixture
from tests.test_herogen_blender_refine import FakeClient, HAVE_REFINE


class ConcurrentBudgetTests(unittest.TestCase):
    def test_estimate_reserves_all_parallel_requests(self):
        self.assertEqual(stage.cost_estimate(refine_profile='fast'), .25)
        self.assertEqual(stage.cost_estimate(True, refine_profile='fast'), .25)

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

    def test_cached_family_is_not_replaced_but_changed_content_is_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = ModelClient(tmp)
            parent._save_entry('k', {'value': 1})
            child = fast.isolated_client(parent, 'child')
            self.assertEqual(child._load('k'), {'value': 1})
            with mock.patch.object(fast.os, 'replace', side_effect=OSError('shared file is open')):
                fast.publish(parent, child)
            self.assertEqual(parent._load('k'), {'value': 1})
            child._save_entry('k', {'value': 2})
            fast.publish(parent, child)
            self.assertEqual(parent._load('k'), {'value': 2})

    def test_nested_siblings_wait_for_a_complete_shared_parent_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = ModelClient(tmp)
            parent._save_entry('k', {'value': 1})
            outer = fast.isolated_client(parent, 'outer')
            a = fast.isolated_client(outer, 'a')
            b = fast.isolated_client(outer, 'b')
            copying, release = threading.Event(), threading.Event()
            real_copy = fast.shutil.copy2
            def slow_copy(source, target, *args, **kwargs):
                target = Path(target)
                if target == outer.cache_dir / 'k.json':
                    target.write_bytes(b'')
                    copying.set()
                    if not release.wait(5):
                        raise TimeoutError('copy was not released')
                return real_copy(source, target, *args, **kwargs)
            with mock.patch.object(fast.shutil, 'copy2', side_effect=slow_copy), \
                    ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(a._load, 'k')
                self.assertTrue(copying.wait(5))
                second = pool.submit(b._load, 'k')
                try:
                    with self.assertRaises(TimeoutError):
                        second.result(timeout=.05)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5), {'value': 1})
                self.assertEqual(second.result(timeout=5), {'value': 1})

    def test_nested_cache_keeps_images_and_new_entries_with_their_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = ModelClient(tmp)
            parent._save_png('k.png', Image.new('RGB', (4, 4), 'blue'))
            parent._save_entry('k', {'image': 'k.png'})
            outer = fast.isolated_client(parent, 'outer')
            nested = fast.isolated_client(outer, 'nested')
            self.assertEqual(nested._load('k'), {'image': 'k.png'})
            self.assertEqual(Image.open(nested.cache_dir / 'k.png').getpixel((0, 0)), (0, 0, 255))
            nested._save_png('n.png', Image.new('RGB', (4, 4), 'red'))
            nested._save_entry('n', {'image': 'n.png'})
            self.assertTrue((nested.cache_dir / 'n.json').is_file())
            self.assertFalse((outer.cache_dir / 'n.json').exists())
            fast.publish(outer, nested)
            fast.publish(parent, outer)
            self.assertEqual(parent._load('n'), {'image': 'n.png'})


@unittest.skipUnless(HAVE_REFINE, 'texture-refine dependencies are required')
class FastStageTests(unittest.TestCase):
    def test_auto_palette_is_built_from_this_invocations_preparation(self):
        with mock.patch('herogen.blender.refine.region_palette.prepare_manifest', return_value='built.json') as build, \
                mock.patch('herogen.blender.refine.region_palette.export_texture',
                           side_effect=lambda manifest, prep, texture, *a, **kw: (texture, {})) as grade, \
                mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            stage.run_stage3f(self.run, self.client, refine_profile='fast', region_palette=True)
        self.assertEqual(build.call_args.args[0], self.run.resolve())
        self.assertTrue(all(c.args[0] == 'built.json' for c in grade.call_args_list))

    def test_palette_flag_off_never_calls_calibration(self):
        import os
        with mock.patch.dict(os.environ, {'HEROGEN_REFINE_REGION_PALETTE': ''}), \
                mock.patch('herogen.blender.refine.region_palette.export_texture') as grade, \
                mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        grade.assert_not_called()
        self.assertNotIn('region_palette', rec)

    def test_palette_flag_grades_each_published_version_without_model_calls(self):
        import os
        def graded(manifest, prep, texture, camera, folder, **kwargs):
            return texture, {'skin': {'cap_lab': 2}}
        with mock.patch.dict(os.environ, {'HEROGEN_REFINE_REGION_PALETTE': 'manifest.json'}), \
                mock.patch('herogen.blender.refine.region_palette.export_texture', side_effect=graded) as grade, \
                mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertEqual(grade.call_count, len(progress['versions']))
        self.assertEqual(rec['region_palette']['skin']['cap_lab'], 2)

    def test_palette_export_does_not_feed_the_next_wave(self):
        exports = []
        def grade(manifest, prep, texture, camera, folder, **kwargs):
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / 'graded.png'
            target.write_bytes(texture.read_bytes())
            return target, {}
        def blender(mode, args, log, **kwargs):
            if mode == 'export':
                exports.append(args['texture'])
            return self.render_many(mode, args, log, **kwargs)
        with mock.patch('herogen.blender.refine.region_palette.prepare_manifest', return_value='auto.json'), \
                mock.patch('herogen.blender.refine.region_palette.export_texture', side_effect=grade), \
                mock.patch('herogen.blender.refine.backend.blender', side_effect=blender):
            record = stage.run_stage3f(self.run, self.client, refine_profile='fast', region_palette=True)
        self.assertTrue(exports)
        self.assertTrue(all(path.endswith('graded.png') for path in exports))
        self.assertTrue(self.renders[5].endswith('texture-v1.png'))
        self.assertEqual(record['waves'][1]['frozen_atlas'], 'texture-v1.png')

    def test_palette_failure_keeps_fast_progress_unpublished(self):
        import os
        with mock.patch.dict(os.environ, {'HEROGEN_REFINE_REGION_PALETTE': 'bad.json'}), \
                mock.patch('herogen.blender.refine.region_palette.export_texture', side_effect=ValueError('wrong palette layout')), \
                mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            with self.assertRaisesRegex(ValueError, 'wrong palette layout'):
                stage.run_stage3f(self.run, self.client, refine_profile='fast')
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertEqual(progress['versions'], [])
        self.assertFalse((self.run / '03-refined.glb').exists())

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

    def test_waiting_limiter_allows_settle_sam_and_selection_to_cancel_before_payment(self):
        waiting, release, first_started = (threading.Event() for _ in range(3))
        paid, settled = [], []
        count = 0
        count_lock = threading.Lock()

        def limiter(model):
            nonlocal count
            if model == IMAGE_MODEL:
                with count_lock:
                    count += 1
                    number = count
                if number == 2:
                    waiting.set()
                    if not release.wait(5):
                        raise RuntimeError('test limiter timed out')

        self.client.before_api_attempt = limiter
        self.client.after_api_attempt = settled.append

        def view(child, prep, current, tex, cam, hair, folder, record, *args, attempt_start=0, **kwargs):
            if attempt_start:
                self.assertTrue(first_started.wait(5))
            child.before_api_attempt(IMAGE_MODEL)
            paid.append((attempt_start, IMAGE_MODEL))
            if not attempt_start:
                first_started.set()
                self.assertTrue(waiting.wait(5))
            child.after_api_attempt(PRICE_PER_IMAGE_USD[IMAGE_MODEL])
            child.before_api_attempt(SAM_MODEL)
            paid.append((attempt_start, SAM_MODEL))
            child.after_api_attempt(PRICE_PER_IMAGE_USD[SAM_MODEL])
            folder.mkdir(parents=True, exist_ok=True)
            record.update(accepted=True, reason='accepted', attempts=[{'try': attempt_start + 1}])

        def waves(client, prep, current, tex, cams, work, data, timeout, cover, head_found, recipe,
                  save, state_lock, **kwargs):
            job = fast.Pass('body+0', 'body+0', False, {'attempts': []}, work / 'body+0')
            with ThreadPoolExecutor(max_workers=2) as pool:
                try:
                    job.submit(pool, client, prep, current, tex, cams['body+0'], work, timeout,
                               cover, head_found, cams['body+0'], recipe, threading.Lock())
                    self.assertTrue(waiting.wait(5))
                    # First settlement and the independent SAM quota must finish with the limiter still blocked.
                    job.tasks[0][0].result(timeout=2)
                    with state_lock:
                        job.collect(0)
                    self.assertTrue(job.cancelled.is_set())
                    self.assertEqual(paid, [(0, IMAGE_MODEL), (0, SAM_MODEL)])
                    self.assertFalse(release.is_set())
                finally:
                    release.set()
                with self.assertRaises(CancelledError):
                    job.tasks[1][0].result(timeout=2)
                self.assertEqual(paid, [(0, IMAGE_MODEL), (0, SAM_MODEL)])
                # Reusing the same worker must not retain the cancelled attempt's context.
                self.assertIsNone(pool.submit(stage._attempt_cancelled.get).result(timeout=2))
            return current, False

        with mock.patch.object(stage, '_view', side_effect=view), mock.patch.object(fast, 'waves', side_effect=waves):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(settled, [PRICE_PER_IMAGE_USD[IMAGE_MODEL], PRICE_PER_IMAGE_USD[SAM_MODEL], 0])
        self.assertEqual(rec['budget']['pending_attempt_usd'], 0)
        self.assertEqual(rec['budget']['spent_usd_total'], sum(settled))

    def test_paired_limiter_hooks_release_without_holding_the_budget_lock(self):
        slot = threading.Semaphore(1)
        waiting = threading.Event()
        count = 0
        count_lock = threading.Lock()

        def acquire(model):
            nonlocal count
            with count_lock:
                count += 1
                number = count
            if number == 2:
                waiting.set()
            if not slot.acquire(timeout=3):
                raise RuntimeError('paired limiter deadlocked')

        def release(cost):
            slot.release()

        self.client.before_api_attempt = acquire
        self.client.after_api_attempt = release

        def waves(client, prep, current, *args, **kwargs):
            entered = threading.Event()
            def attempt(index):
                if index:
                    self.assertTrue(entered.wait(3))
                client.before_api_attempt(IMAGE_MODEL)
                if not index:
                    entered.set()
                    self.assertTrue(waiting.wait(3))
                client.after_api_attempt(0)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = pool.submit(attempt, 0), pool.submit(attempt, 1)
                first.result(timeout=4)
                second.result(timeout=4)
            return current, False

        with mock.patch.object(fast, 'waves', side_effect=waves):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(rec['budget']['pending_attempt_usd'], 0)
        self.assertTrue(slot.acquire(blocking=False))
        self.assertFalse(slot.acquire(blocking=False))

    def test_previous_settle_can_wait_without_blocking_state_changes(self):
        entered, release = threading.Event(), threading.Event()
        def settle(cost):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('settle hook timed out')
        self.client.after_api_attempt = settle

        def waves(client, prep, current, tex, cams, work, data, timeout, cover, head_found, recipe,
                  save, state_lock, **kwargs):
            def attempt():
                client.before_api_attempt(IMAGE_MODEL)
                client.after_api_attempt(0)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(attempt)
                try:
                    self.assertTrue(entered.wait(3))
                    acquired = state_lock.acquire(timeout=2)
                    self.assertTrue(acquired)
                    if acquired:
                        state_lock.release()
                finally:
                    release.set()
                future.result(timeout=2)
            return current, False

        with mock.patch.object(fast, 'waves', side_effect=waves):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(rec['budget']['pending_attempt_usd'], 0)

    def test_limiter_failure_releases_reservation_and_stays_loud(self):
        self.client.before_api_attempt = mock.Mock(side_effect=ApiError('limiter failed'))
        self.client.after_api_attempt = mock.Mock()
        def waves(client, prep, current, *args, **kwargs):
            client.before_api_attempt(IMAGE_MODEL)
        with mock.patch.object(fast, 'waves', side_effect=waves), self.assertRaisesRegex(ApiError, 'limiter failed'):
            stage.run_stage3f(self.run, self.client, refine_profile='fast')
        rec = json.loads((self.run / 'stage-3f.json').read_text())
        self.assertEqual(rec['status'], 'error')
        self.assertEqual(rec['budget']['pending_attempt_usd'], 0)
        self.assertEqual(rec['budget']['spent_usd_total'], 0)
        self.client.after_api_attempt.assert_not_called()

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

    def test_without_spend_cap_fast_runs_and_records_no_budget(self):
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast', cap_usd=None)
        self.assertEqual(rec['status'], 'complete')
        self.assertNotIn('budget', rec)
        self.assertGreater(rec['cost_usd'], 0)
        self.assertEqual(sum(v['accepted'] for v in rec['views'].values()), 6)

    def test_every_fast_pass_stops_after_two_failed_attempts(self):
        original = FakeClient.edit_image
        def edit(child, prompt, images, **kw):
            if kw['attempt'] < 2:
                child._call(IMAGE_MODEL, 'edit')
                return Image.new('RGB', images[0].size, (115, 115, 115))
            return original(child, prompt, images, **kw)
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(len(rec['views']), 6)
        for view in rec['views'].values():
            self.assertFalse(view['accepted'])
            self.assertEqual(view['tries'], 2)
            self.assertEqual([trial['try'] for trial in view['attempts']], [1, 2])


    def test_parallel_attempts_choose_lowest_passing_index(self):
        original = FakeClient.edit_image
        barrier = threading.Barrier(2)
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
        self.assertCountEqual(seen, [0, 1])
        self.assertEqual(rec['views']['head+0']['selected_attempt'], 1)
        self.assertEqual(rec['views']['head+0']['attempts'][0]['reason'], 'outline moved')

    def test_side_beauty_conditions_side_and_back_attempts_and_keeps_notes_and_versions(self):
        from herogen.blender.refine import prompts
        fixture.StageTests.closeup_artwork(self)
        captured = []
        original = FakeClient.edit_image

        def edit(child, text, images, **kwargs):
            captured.append((child.cache_dir.parts, text, images, kwargs['attempt']))
            return original(child, text, images, **kwargs)

        note = 'Keep the leather brown.'
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast', side_beauty=True,
                                    user_note=note)
        expected = {'head+0': None, 'body+0': None, 'body+90': 'right',
                    'body-90': 'left', 'body+180': 'back', 'hair-upper+180': 'back'}
        self.assertTrue(rec['side_beauty'])
        for key, view in rec['views'].items():
            if expected[key] is None:
                self.assertNotIn('art_views', view)
                self.assertNotIn('art_framing', view)
            else:
                self.assertEqual(view['art_views'], [expected[key]])
            self.assertLessEqual(view['tries'], 2)
            if key == 'hair-upper+180':
                self.assertIn('silhouette_alignment', view['art_framing'][expected[key]])
        for parts, text, images, attempt in captured:
            key = next(k for k in expected if k in parts)
            recipe = 'RF13H' if key.startswith('hair-') else 'RF12R' if expected[key] else 'RF2R'
            self.assertEqual(text, prompts.prompt(recipe, rec['views'][key]['camera']['angle'], note))
            self.assertEqual(len(images), 3 if expected[key] else 2)
            self.assertEqual(images[-1].getpixel((0, 0)), (0, 0, 255))
            self.assertIn(attempt, (0, 1))
            if expected[key]:
                self.assertIn('number and material', text)
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertTrue(progress['done'])
        self.assertEqual(progress['final_n'], len(progress['versions']))
        self.assertEqual((self.run / '03-refined.glb').read_bytes(),
                         (self.run / progress['versions'][-1]['glb']).read_bytes())

    def test_side_beauty_off_ignores_existing_art_and_environment_can_opt_in(self):
        import os
        fixture.StageTests.closeup_artwork(self)
        with mock.patch.dict(os.environ, {'HEROGEN_SIDE_BEAUTY': '1'}), \
             mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast', side_beauty=False)
        self.assertNotIn('side_beauty', rec)
        self.assertEqual(set(rec['prompt_sha256']), {'RF2R', 'RF2W', 'RF3H', 'RF4U', 'RF5U'})
        self.assertTrue(all(len(images) == 2 for _, images, _ in self.client.inputs))
        self.client.inputs.clear()
        with mock.patch.dict(os.environ, {'HEROGEN_SIDE_BEAUTY': '1'}), \
             mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast', cap_usd=5)
        self.assertTrue(rec['side_beauty'])
        self.assertEqual({len(images) for _, images, _ in self.client.inputs}, {2, 3})

    def test_cli_side_beauty_flag_reaches_stage(self):
        from herogen.blender.__main__ import main
        with mock.patch.object(stage, 'run_stage3f', return_value={'ok': True}) as run:
            self.assertEqual(main(['run', '--run', str(self.run), '--stages', '3f',
                                   '--refine-profile', 'fast', '--side-beauty']), 0)
        self.assertTrue(run.call_args.kwargs['side_beauty'])

    def test_side_beauty_missing_art_stays_loud_before_model_calls(self):
        fixture.StageTests.closeup_artwork(self)
        (self.run / 'views/back.png').unlink()
        with self.assertRaises(FileNotFoundError):
            stage.run_stage3f(self.run, self.client, refine_profile='fast', side_beauty=True)
        self.assertEqual(self.client.log.calls, [])
        self.assertEqual(json.loads((self.run / 'stage-3f.json').read_text())['status'], 'error')

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

    def test_skipped_fast_runs_finish_empty_progress(self):
        for missing_source in (False, True):
            with self.subTest(missing_source=missing_source):
                if missing_source:
                    (self.run / '03-source.glb').unlink()
                record = stage.run_stage3f(self.run, self.client, refine_profile='fast',
                                          profile='best' if missing_source else 'preview')
                self.assertEqual(record['status'], 'skipped')
                progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
                self.assertEqual(progress, {'profile': 'fast', 'versions': [], 'done': True, 'final_n': None})
        self.assertFalse(self.client.inputs)

    def test_preparation_cap_replaces_stale_progress_with_finished_empty_fallback(self):
        progress = self.run / 'stage-3f-progress.json'
        progress.write_text('{"profile":"fast","versions":[{"n":1}],"done":true,"final_n":1}')
        with mock.patch.object(fixture.character, 'setup',
                               side_effect=stage.BudgetStop('preparation cap')):
            record = stage.run_stage3f(self.run, self.client, refine_profile='fast', cap_usd=0)
        self.assertEqual(record['status'], 'cap_reached')
        self.assertEqual(json.loads(progress.read_text()),
                         {'profile': 'fast', 'versions': [], 'done': True, 'final_n': None})
        self.assertFalse(json.loads((self.run / 'stage-3f-progress-initial.json').read_text())['done'])
        self.assertFalse((self.run / '03-refined.glb').exists())

    def test_bad_profile_and_upscale_combination_fail_before_output_deletion(self):
        final = self.run / '03-refined.glb'
        final.write_bytes(b'keep')
        for kw in ({'refine_profile': 'unknown'}, {'refine_profile': 'fast', 'upscale_texture': True}):
            with self.assertRaises(ValueError):
                stage.run_stage3f(self.run, self.client, **kw)
            self.assertEqual(final.read_bytes(), b'keep')

    def test_default_fast_and_best_keep_the_frozen_request_sequences(self):
        from collections import Counter
        from herogen.blender.refine import prompts
        self.client.reject = True  # Exercise every sample, without cancellation after a pass.
        expected = {
            'fast': [('RF2R', angle, 2) for angle in (0, 0, 90, -90, 180)] + [('RF3H', 180, 2)],
            'best': [('RF2R', angle, 3) for angle in (0, 40, -40, 90, -90, 180, 0, 45, -45, 90, -90, 180)]
                    + [('RF3H', angle, 2) for angle in (90, 135, 180, -135, -90, 135, 180, -135)],
        }
        for profile, passes in expected.items():
            with self.subTest(profile=profile), \
                 mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
                self.client.inputs.clear()
                stage.run_stage3f(self.run, self.client, refine_profile=profile, cap_usd=None)
                actual = [(text, options['attempt']) for text, _, options in self.client.inputs]
                frozen = [(prompts.prompt(recipe, angle), attempt)
                          for recipe, angle, attempts in passes for attempt in range(attempts)]
                if profile == 'fast':
                    self.assertEqual(Counter(actual), Counter(frozen))
                else:
                    self.assertEqual(actual, frozen)

    def test_default_omits_fast_metadata(self):
        rec = stage.run_stage3f(self.run, self.client)
        self.assertNotIn('refine_profile', rec)
        self.assertNotIn('waves', rec)
        self.assertEqual(len(rec['views']), 20)

    # ---- build log ------------------------------------------------------------------------------------------

    VIEWS = ('head+0', 'body+0', 'body+90', 'body-90', 'body+180', 'hair-upper+180')

    def run_logged(self, edit=None, **kwargs):
        """Run the fast stage with a build-log file; every Muse Image repaint goes through the call tracker."""
        from herogen.pipeline import events
        from herogen.pipeline.fast import tracked
        log = self.run / 'events.jsonl'
        inner = edit or FakeClient.edit_image
        with mock.patch.dict('os.environ', {'HEROGEN_EVENTS_FILE': str(log)}), \
             mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', tracked(IMAGE_MODEL, 'edit', isolate=True)(inner)), \
             events.step('3f'):
            rec = stage.run_stage3f(self.run, self.client, refine_profile='fast', **kwargs)
        lines = log.read_text().splitlines()
        # Every line is inside the schema exactly as written.
        self.assertEqual([events.validate(json.loads(line)) for line in lines], [json.loads(line) for line in lines])
        return rec, [json.loads(line) for line in lines]

    @staticmethod
    def decisions(log, what):
        return [e for e in log if e['kind'] == 'decision' and e['what'] == what]

    def test_build_log_labels_each_try_and_both_waves(self):
        rec, log = self.run_logged()
        self.assertTrue(all(e['step'] == '3f' for e in log))
        waves = self.decisions(log, 'wave')
        self.assertEqual([(w['n'], w['m']) for w in waves],
                         [(10, {'wave': 1, 'views': 5, 'tries': 2}),
                          (2, {'wave': 2, 'views': 1, 'tries': 2, 'from_version': 1})])
        self.assertEqual([e['m'] for e in log if e.get('what') == 'views_rendered' and e['kind'] == 'local'],
                         [{'wave': 1, 'cameras': 5}, {'wave': 2, 'cameras': 1}])
        # Each repaint request is one call row with its view and try, even from the worker threads.
        starts = [e for e in log if e['kind'] == 'call_start']
        ends = {e['id']: e for e in log if e['kind'] == 'call_end'}
        self.assertTrue(starts)
        for start in starts:
            self.assertEqual((start['who'], start['what']), ('muse-image', 'repaint'))
            self.assertIn(start['view'], self.VIEWS)
            self.assertIn(start['try'], (1, 2))
            self.assertEqual({k: ends[start['id']][k] for k in ('view', 'try')},
                             {'view': start['view'], 'try': start['try']})
        # Exactly one result per try: judged against its gates, or cancelled as no longer needed.
        results = self.decisions(log, 'view_result')
        self.assertCountEqual([(e['view'], e['try']) for e in results],
                              [(view, n) for view in self.VIEWS for n in (1, 2)])
        for e in results:
            if e['ok']:
                self.assertGreaterEqual(e['m']['overlap'], e['m']['overlap_min'])
            else:
                self.assertEqual(e['why'], 'lower_try_accepted')
                self.assertIn(e['m']['paid'], (0, 1))
        selected = self.decisions(log, 'view_selected')
        self.assertCountEqual([e['view'] for e in selected], self.VIEWS)
        for e in selected:
            self.assertEqual(e['try'], rec['views'][e['view']]['selected_attempt'] + 1)
        self.assertEqual(self.decisions(log, 'view_kept'), [])
        versions = [e for e in log if e['kind'] == 'version']
        self.assertEqual([(v['n'], v['passed'], v['of'], v['m']['wave']) for v in versions],
                         [(1, 5, 6, 1), (2, 6, 6, 2)])
        # The record's numbers are the log's numbers.
        hair = next(e for e in results if e['view'] == 'hair-upper+180' and e['ok'])
        trial = next(t for t in rec['views']['hair-upper+180']['attempts'] if t['try'] == hair['try'])
        self.assertAlmostEqual(hair['m']['agreement'], trial['checks']['agreement'], places=4)
        self.assertEqual(hair['m']['agreement_min'], fixture.refine.HAIR_GATE['agree_min'])

    def test_build_log_labels_the_cancelled_hedge(self):
        original = FakeClient.edit_image
        selected = {view: threading.Event() for view in self.VIEWS}
        original_collect = fast.Pass.collect

        def collect(job, index):
            original_collect(job, index)
            if job.selected is not None:
                selected[job.key].set()

        def edit(child, prompt, images, **kw):
            view = child.cache_dir.parts[-2]
            if kw['attempt'] == 1:
                # The second try asks only after the first was accepted, so the guard cancels it unpaid.
                self.assertTrue(selected[view].wait(5))
            return original(child, prompt, images, **kw)

        with mock.patch.object(fast.Pass, 'collect', collect):
            rec, log = self.run_logged(edit)
        results = {(e['view'], e['try']): e for e in self.decisions(log, 'view_result')}
        self.assertEqual(len(results), 12)
        for view in self.VIEWS:
            self.assertTrue(results[view, 1]['ok'])
            cancelled = results[view, 2]
            self.assertEqual((cancelled['ok'], cancelled['why'], cancelled['m']),
                             (False, 'lower_try_accepted', {'paid': 0, 'paid_usd': 0}))
            self.assertEqual(rec['views'][view]['selected_attempt'], 0)
        ends = [e for e in log if e['kind'] == 'call_end' and e['try'] == 2]
        self.assertEqual(len(ends), 6)
        self.assertTrue(all(e['outcome'] == 'cancelled' and e['usd'] == 0 for e in ends))

    def test_build_log_reports_rejected_tries_with_their_gate_numbers(self):
        def edit(child, prompt, images, **kw):
            child._call(IMAGE_MODEL, 'edit')
            return Image.new('RGB', images[0].size, (115, 115, 115))

        rec, log = self.run_logged(edit)
        results = self.decisions(log, 'view_result')
        self.assertEqual(len(results), 12)
        for e in results:
            trial = rec['views'][e['view']]['attempts'][e['try'] - 1]
            self.assertFalse(e['ok'])
            self.assertEqual(e['why'], {'outline moved': 'outline_moved',
                                        'hair outline, scale or colour agreement': 'hair_mismatch'}[trial['reason']])
            self.assertAlmostEqual(e['m']['overlap'], trial['checks']['iou'], places=4)
            if e['why'] == 'outline_moved':
                self.assertEqual((e['m']['median_limit'], e['m']['p90_limit']),
                                 (trial['checks']['median_limit'], trial['checks']['p90_limit']))
        kept = self.decisions(log, 'view_kept')
        self.assertCountEqual([e['view'] for e in kept], self.VIEWS)
        self.assertTrue(all(e['m'] == {'tries': 2} for e in kept))
        self.assertEqual([e for e in log if e['kind'] == 'version'], [])
        # Without a version 1, the hair wave repaints from the unrefined texture.
        self.assertNotIn('from_version', self.decisions(log, 'wave')[1]['m'])
        self.assertEqual([(e['what'], e['why']) for e in log if e['kind'] == 'skip'],
                         [('texture_refine', 'views_rejected')])

    def test_build_log_skips_the_face_close_up_without_a_head(self):
        self.info['head_centre'] = None
        rec, log = self.run_logged()
        self.assertIn({'what': 'head_probe', 'ok': False},
                      [{k: e[k] for k in ('what', 'ok')} for e in log if e['kind'] == 'local' and 'ok' in e])
        self.assertEqual([(e['view'], e['why']) for e in log if e['kind'] == 'skip'], [('head+0', 'head_not_found')])
        self.assertEqual(self.decisions(log, 'wave')[0]['n'], 8)

    def test_build_log_says_the_spending_cap_was_reached(self):
        rec, log = self.run_logged(cap_usd=0)
        self.assertEqual(rec['status'], 'cap_reached')
        # Wave 1's tries stop at the cap, the hair wave never starts, and the step keeps its unrefined texture.
        self.assertCountEqual([(e['view'], e['why']) for e in self.decisions(log, 'view_kept')],
                              [(view, 'spend_cap') for view in self.VIEWS[:5]])
        self.assertEqual([(e['what'], e.get('view'), e['why']) for e in log if e['kind'] == 'skip'],
                         [('view', 'hair-upper+180', 'spend_cap'), ('texture_refine', None, 'spend_cap')])

    def test_without_a_build_log_file_nothing_is_written(self):
        with mock.patch.dict('os.environ', {}, clear=False) as env, \
             mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many):
            env.pop('HEROGEN_EVENTS_FILE', None)
            stage.run_stage3f(self.run, self.client, refine_profile='fast')
        self.assertEqual(list(self.run.rglob('events.jsonl')), [])


class BuildLogTests(unittest.TestCase):
    def test_a_cancelled_try_says_whether_its_repaint_was_already_paid(self):
        from herogen.blender.refine import buildlog
        from herogen.pipeline import events
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'events.jsonl'
            private = {'attempts': [{'try': 2, 'checks': {'iou': 0.79, 'median_px': 10.0, 'median_limit': 6.0}}]}
            with mock.patch.dict('os.environ', {events.ENV_FILE: str(log)}):
                buildlog.try_cancelled('body-90', 2, private, [{'model': IMAGE_MODEL, 'cached': False,
                                                                'cost_usd': 0.01}])
                buildlog.try_cancelled('body-90', 2, {'attempts': []}, [{'model': IMAGE_MODEL, 'cached': True,
                                                                         'cost_usd': 0.0}])
            first, second = (json.loads(line) for line in log.read_text().splitlines())
        self.assertEqual((first['view'], first['try'], first['why'], first['m']),
                         ('body-90', 2, 'lower_try_accepted',
                          {'overlap': 0.79, 'median_px': 10.0, 'median_limit': 6.0, 'paid': 1, 'paid_usd': 0.01}))
        self.assertEqual((second['try'], second['m']), (2, {'paid': 0, 'paid_usd': 0}))

    def test_a_full_lane_client_logs_no_call_costs_but_its_cancelled_try_is_still_paid(self):
        from herogen.blender.refine import buildlog
        from herogen.pipeline import events
        from herogen.pipeline.client import CallLog, call_cost
        log = CallLog()   # a client outside the fast lane records live calls without their own cost
        log.add(IMAGE_MODEL, 'edit', 'k', cached=False)
        log.add(SAM_MODEL, 'segment', 'k2', cached=False, outcome='unknown')   # a timeout that may be billed
        log.add(SAM_MODEL, 'segment', 'k3', cached=True)
        self.assertNotIn('cost_usd', log.calls[0])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'events.jsonl'
            with mock.patch.dict('os.environ', {events.ENV_FILE: str(out)}):
                buildlog.try_cancelled('body+0', 2, {'attempts': []}, log.calls)
            event = json.loads(out.read_text())
        self.assertEqual(event['m'], {'paid': 1, 'paid_usd': round(call_cost(log.calls[0]) + call_cost(log.calls[1]), 6)})
        self.assertEqual(event['m']['paid_usd'], PRICE_PER_IMAGE_USD[IMAGE_MODEL] + PRICE_PER_IMAGE_USD[SAM_MODEL])

    def test_a_line_that_cannot_be_worked_out_never_stops_the_stage(self):
        from herogen.blender.refine import buildlog
        from herogen.pipeline import events
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'events.jsonl'
            buildlog.try_result('body+0', {'checks': object()}, False, 'outline moved')   # log off: nothing
            self.assertFalse(out.exists())
            with mock.patch.dict('os.environ', {events.ENV_FILE: str(out)}):
                buildlog.try_result('body+0', {'checks': object()}, False, 'outline moved')
            event = json.loads(out.read_text())
        self.assertEqual((event['kind'], event['what'], event['why']), ('skip', 'build_log_detail', 'unreadable'))

    def test_record_reasons_become_fixed_identifiers(self):
        from herogen.blender.refine import buildlog
        self.assertEqual([buildlog.why(r) for r in (
            'outline moved', 'new face outside the render face', 'hair outline, scale or colour agreement',
            'cancelled: lower attempt accepted', 'API cap: projected $0.0100 exceeds $0.0000',
            'missing facial features: mouth', 'ApiError')],
            ['outline_moved', 'second_face', 'hair_mismatch', 'lower_try_accepted', 'spend_cap',
             'missing_features', 'model_error'])


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
