"""Beauty face ownership, independent fallback and progressive repaint contracts."""
from concurrent.futures import Future
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender import refine as stage
try:
    from herogen.blender.refine import beauty, cameras, fast, fast_beauty as B, imaging as I
    HAVE_REFINE = True
except ImportError:
    HAVE_REFINE = False
from herogen.pipeline.client import ApiError, ReplayMiss, ModelClient
from herogen.refine_budget import BudgetStop
from tests import test_herogen_blender_refine as fixture
from tests import test_herogen_blender_refine_note as note_fixture
from tests.test_herogen_blender_refine_note import NOTE


class TrackedLock:
    def __init__(self, lock=None):
        self.lock = lock or threading.RLock()
        self.local = threading.local()

    @property
    def owned(self):
        return getattr(self.local, 'depth', 0) > 0

    def __enter__(self):
        self.lock.acquire()
        self.local.depth = getattr(self.local, 'depth', 0) + 1
        return self

    def __exit__(self, *args):
        self.local.depth -= 1
        self.lock.release()


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class FaceRuleTests(unittest.TestCase):
    def test_fast_beauty_estimate_includes_frontal_feature_calls(self):
        self.assertEqual(stage.cost_estimate(refine_profile='fast'), .25)
        self.assertEqual(stage.cost_estimate(refine_profile='fast', beauty=True), .5725)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.data = {'views': {k: {'accepted': False} for k in ('head+0', 'body+0')}}
        self.control = B.FaceControl(self.work, {}, data=self.data)
        self.control.default_work = self.work / 'ordinary'
        self.default = {'accepted': True, 'prompt': 'RF2R', 'attempts': []}
        folder = self.work / 'head+0/attempt-1'
        folder.mkdir(parents=True)
        Image.new('L', (8, 8), 255).save(folder / 'face.png')
        self.head = SimpleNamespace(name='head+0', record={'attempts': [], 'tries': 2},
                                    tasks=[None, (None, None, self.default, folder)])

    def frontal(self, key='head+0', *, both=True, mouth=True):
        for folder in (self.work / key, self.control.default_work / key):
            folder.mkdir(parents=True, exist_ok=True)
            EyeQualityTests.face().save(folder / 'refined.png')
        self.data['views'][key] = {'accepted': True, 'prompt': 'RF6B', 'selected_attempt': 0,
            'attempts': [{'accepted': True, 'checks': {'facial_features': {
                'repaint': {'both_eyes': both, 'mouth': mouth, 'eye_boxes': EyeQualityTests.boxes}}}}]}

    def test_complete_beauty_head_wins_over_default(self):
        self.frontal()
        self.control.choose([self.head], self.work)
        self.assertEqual(self.data['face_source'], {'source': 'beauty', 'view': 'head+0', 'attempt': 0})
        self.assertNotIn('face-default-head+0', self.data['views'])

    def test_soft_eye_rejects_beauty_and_selects_the_whole_default_head(self):
        self.frontal()
        EyeQualityTests.face(3).save(self.work / 'head+0/refined.png')
        self.control.choose([self.head], self.work)
        self.assertEqual(self.data['face_source']['source'], 'default')
        self.assertTrue(self.data['views']['head+0']['head_only'])
        check = self.data['beauty_head_candidate']['attempts'][0]['checks']['eye_quality']
        self.assertFalse(check['accepted'])
        self.assertLess(check['eyes'][0]['ratios']['laplacian_variance'], 1)

    def test_complete_beauty_body_can_anchor_the_face(self):
        self.frontal('body+0')
        self.control.choose([self.head], self.work)
        self.assertEqual(self.data['face_source']['view'], 'body+0')

    def test_missing_eye_or_mouth_or_rejected_frontal_uses_default(self):
        for both, mouth, accepted in ((False, True, True), (True, False, True), (True, True, False)):
            with self.subTest(both=both, mouth=mouth, accepted=accepted):
                self.control.key = None
                self.frontal(both=both, mouth=mouth)
                self.data['views']['head+0']['accepted'] = accepted
                self.control.choose([self.head], self.work)
                self.assertEqual(self.data['face_source']['source'], 'default')
                self.assertEqual((self.work / 'head+0/face.png').read_bytes(),
                                 self.head.tasks[1][3].joinpath('face.png').read_bytes())
                self.assertTrue(self.data['views']['head+0']['head_only'])
                self.assertFalse(np.asarray(Image.open(self.work / 'head+0/weight.png')).any())

    def test_oblique_beauty_does_not_qualify_and_requests_default_fast_fallback(self):
        self.frontal('body+90')
        self.default['accepted'] = False
        self.control.choose([self.head], self.work)
        self.assertEqual(self.data['face_source']['source'], 'default-fast-fallback')

    def test_missing_head_requests_default_fast_fallback(self):
        self.control.choose([], self.work)
        self.assertEqual(self.data['face_fallback']['reason'], 'head_not_found')
        self.control.fallback_texture = np.zeros((8, 8, 3))
        with mock.patch.object(self.control, 'paint_default_head') as paint:
            self.control.paint(mock.Mock(), self.work, self.data['views'])
        paint.assert_called_once()

    def test_head_slots_never_cancel_each_other_in_either_completion_order(self):
        for order in ((0, 1), (1, 0)):
            job = B.FacePass('head+0', 'head+0', False, {'attempts': []}, self.work / 'selected')
            for index in range(2):
                future = Future()
                folder = self.work / f'candidate-{index}'
                folder.mkdir(exist_ok=True)
                (folder / 'refined.png').write_bytes(bytes([index]))
                record = {'accepted': True, 'prompt': ('RF6B', 'RF2R')[index],
                          'attempts': [{'accepted': True}]}
                future.set_result(None)
                job.tasks.append((future, None, record, folder))
            for index in order:
                job.collect(index)
            self.assertFalse(job.cancelled.is_set())
            self.assertEqual(job.record['prompt'], 'RF6B')
            self.assertEqual(job.record['tries'], 2)
            self.assertEqual([a['face_candidate'] for a in job.record['attempts']], ['beauty', 'default'])

    def test_conditional_slot_is_due_only_after_two_gate_rejections(self):
        def job(*outcomes):
            job = fast.Pass('body+90', 'body+90', False, {'attempts': []}, self.work / 'selected')
            job.slots, job.conditional = 2, 1
            for index, outcome in enumerate(outcomes):
                future = Future()
                record = {'accepted': outcome == 'accepted', 'reason': outcome,
                          'attempts': [{'accepted': outcome == 'accepted', 'reason': outcome}]}
                if outcome == 'budget':
                    future.set_exception(BudgetStop('cap'))
                elif outcome == 'cancelled':
                    future.cancel()
                else:
                    future.set_result(None)
                job.tasks.append((future, None, record, self.work / f'candidate-{index}'))
                job.completed[index] = record
            return job
        self.assertTrue(job('outline moved', 'outline moved').conditional_due())
        for outcomes in (('accepted', 'outline moved'), ('outline moved', 'accepted'),
                         ('outline moved', 'budget'), ('outline moved', 'cancelled'),
                         ('outline moved', 'incomplete'), ('outline moved',),
                         ('outline moved', 'outline moved', 'outline moved')):
            with self.subTest(outcomes=outcomes):
                self.assertFalse(job(*outcomes).conditional_due())

    def test_two_eye_regions_and_mouth_are_required(self):
        image = Image.new('RGB', (20, 20))
        face = np.ones((20, 20), bool)
        for eye_count in (0, 1, 2):
            for mouth in (False, True):
                def segment(image, text):
                    mask = np.zeros((20, 20), bool)
                    if text == 'eyes':
                        for x in (4, 14)[:eye_count]:
                            mask[5:7, x:x+2] = True
                    elif mouth:
                        mask[14:16, 8:12] = True
                    return [SimpleNamespace(mask=mask)]
                features = beauty.frontal_features(SimpleNamespace(segment=segment), image, face)
                self.assertEqual(features['both_eyes'], eye_count == 2)
                self.assertEqual(features['mouth'], mouth)
                self.assertEqual(features['eye_regions'], eye_count)
        for error in (ApiError('failure'), ReplayMiss('missing')):
            with self.assertRaises(type(error)):
                beauty.frontal_features(SimpleNamespace(segment=mock.Mock(side_effect=error)), image, face)

    def test_two_fragments_on_one_side_do_not_prove_both_eyes(self):
        face = np.ones((20, 20), bool)
        eyes = np.zeros_like(face)
        eyes[5:7, 2:4] = True
        eyes[5:7, 6:8] = True
        client = SimpleNamespace(segment=lambda image, text: [SimpleNamespace(mask=eyes)])
        result = beauty.frontal_features(client, Image.new('RGB', (20, 20)), face)
        self.assertEqual(result['eye_regions'], 2)
        self.assertFalse(result['both_eyes'])

    def test_default_candidate_uses_the_original_atlas_cover_and_sample_index(self):
        self.control.default_base = self.work / 'ungraded.png'
        cover = Image.new('RGB', (8, 8), 'blue')
        with mock.patch('herogen.blender.refine.backend.blender') as render:
            self.control.render_default(self.work, {'name': 'head+0'}, self.work / 'wave-1', 900)
        self.assertEqual(render.call_args.args[1]['texture'], str(self.control.default_base))
        reference, options = self.control.options(SimpleNamespace(name='head+0', hair=False), 1, cover)
        self.assertIs(reference, cover)
        self.assertNotIn('recipe_name', options)
        self.assertEqual(options['attempt_start'], 0)
        self.assertEqual(options['rendered'], self.work / 'ordinary/wave-1')

    def test_every_default_slot_uses_ordinary_frozen_render_and_no_art(self):
        cover = Image.new('RGB', (8, 8), 'blue')
        for name, hair in [('body+0', False), ('body+90', False), ('body-90', False),
                           ('body+180', False), ('upper+180', True)]:
            reference, options = self.control.options(SimpleNamespace(name=name, hair=hair), 1, cover)
            self.assertIs(reference, cover)
            self.assertEqual(options['rendered'], self.control.default_work / ('wave-2' if hair else 'wave-1'))
            self.assertEqual(options['attempt_start'], 0)
            self.assertEqual(options['art'], ())
            self.assertIsNone(options['art_masks'])
            self.assertNotIn('recipe_name', options)
            self.assertNotIn('complete_face', options)

    def test_selected_face_interior_has_zero_graded_base_contribution(self):
        cam = cameras.camera('head+0', [0, 0, 1], 0, 2, (8, 8))
        yy, xx = np.mgrid[:8, :8]
        pos = np.stack(((xx + .5 - 4) / 4, np.zeros_like(xx), (4 - yy - .5) / 4 + 1), -1)
        normal = np.zeros_like(pos)
        normal[..., 1] = -1
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        rgb = np.full((8, 8, 3), [.8, .3, .1], np.float32)
        key = self.control.key = 'head+0'
        folder = self.work / key
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / 'face.png')
        lock = self.control.state_lock = TrackedLock()
        class GuardedSource(dict):
            def __setitem__(self, key, value):
                self_test.assertTrue(lock.owned, 'face metadata must share the record writer lock')
                super().__setitem__(key, value)
        self_test = self
        self.data['face_source'] = GuardedSource(source='beauty')
        view = {'cam': cam, 'valid': np.ones((8, 8), bool), 'depth': np.zeros((8, 8)), 'refined': rgb}
        outputs = []
        for base in (np.zeros((8, 8, 3)), np.ones((8, 8, 3))):
            tex = I.Texture.from_arrays(pos, normal, np.ones((8, 8), bool), base, 2)
            with mock.patch.object(I, 'load_accepted_views', return_value=[view]):
                self.control.paint(tex, self.work, {key: {'accepted': True}})
            np.testing.assert_array_equal(tex.T[3:5, 3:5], rgb[3:5, 3:5])
            np.testing.assert_array_equal(tex.T[~mask], base[~mask])
            outputs.append(tex.T[3:5, 3:5])
        np.testing.assert_array_equal(*outputs)

    def test_empty_default_face_still_selects_the_complete_default_head(self):
        self.frontal(both=False, mouth=False)
        Image.new('L', (8, 8), 0).save(self.head.tasks[1][3] / 'face.png')
        self.control.choose([self.head], self.work)
        self.assertEqual(self.data['face_source']['source'], 'default')
        self.assertTrue(self.data['views']['head+0']['head_only'])
        self.assertFalse(np.asarray(Image.open(self.work / 'head+0/weight.png')).any())

    def test_head_skin_mask_extends_the_face_without_replacing_clothing(self):
        self.control.key = 'head+0'
        folder = self.work / 'head+0'
        face = np.zeros((8, 8), bool)
        face[2:6, 2:6] = True
        Image.fromarray(face.astype(np.uint8) * 255).save(folder / 'face.png')
        Image.new('RGB', (8, 8), 'blue').save(folder / 'input.png')
        self.data['views']['head+0'] = {'camera': {'name': 'head+0'}}
        self.data['face_source'] = {'source': 'beauty'}
        skin = face.copy()
        skin[:2, 2:6] = True  # exposed bald scalp outside the narrower face segmentation
        client = SimpleNamespace(segment=mock.Mock(return_value=[SimpleNamespace(mask=skin)]))
        self.control.prepare_region(client, self.work)
        region = np.asarray(Image.open(folder / 'face-paint.png')) > 127
        np.testing.assert_array_equal(region, skin)
        self.assertFalse(region[:, :2].any())  # surrounding costume/hair remains excluded
        self.assertEqual(client.segment.call_args.args[1], 'skin')

    def test_skin_edge_matches_surroundings_and_preserves_interior(self):
        cam = cameras.camera('head+0', [0, 0, 1], 0, 2, (8, 8))
        yy, xx = np.mgrid[:8, :8]
        pos = np.stack(((xx + .5 - 4) / 4, np.zeros_like(xx), (4 - yy - .5) / 4 + 1), -1)
        normal = np.zeros_like(pos)
        normal[..., 1] = -1
        face = np.zeros((8, 8), bool)
        face[3:6, 2:6] = True
        region = face.copy()
        region[1:3, 2:6] = True
        folder = self.work / 'head+0'
        for name, mask in [('face', face), ('face-paint', region)]:
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / (name + '.png'))
        Image.new('L', (8, 8), 0).save(folder / 'source-weight.png')
        original = np.full((8, 8, 3), .2, np.float32)
        self.control.default_work = None
        self.control.default_base = self.work / 'original.png'
        Image.fromarray((original * 255).astype(np.uint8)).save(self.control.default_base)
        tex = I.Texture.from_arrays(pos, normal, np.ones((8, 8), bool), np.full((8, 8, 3), .9), 2)
        rgb = np.full((8, 8, 3), .6, np.float32)
        self.control.key = 'head+0'
        self.data['face_source'] = {'source': 'beauty'}
        view = {'cam': cam, 'valid': np.ones((8, 8), bool), 'depth': np.zeros((8, 8)),
                'refined': rgb, 'base_feather': np.ones((8, 8)), 'alpha': np.ones((8, 8))}
        with mock.patch.object(I, 'load_accepted_views', return_value=[view]):
            self.control.paint(tex, self.work, {'head+0': {'accepted': True}})
        np.testing.assert_allclose(tex.T[4, 3:5], .6)
        self.assertTrue((tex.T[1, 2:6] > .6).all())
        self.assertTrue((tex.T[1, 2:6] < .9).all())
        np.testing.assert_allclose(tex.T[~region], .9)

    def resolve_default_atlas(self, *, mask_error=None, wave_error=None):
        cam = cameras.camera('head+0', [0, 0, 1], 0, 2, (8, 8))
        yy, xx = np.mgrid[:8, :8]
        pos = np.stack(((xx + .5 - 4) / 4, np.zeros_like(xx), (4 - yy - .5) / 4 + 1), -1)
        normal = np.zeros_like(pos)
        normal[..., 1] = -1
        tex = I.Texture.from_arrays(pos, normal, np.ones((8, 8), bool),
                                    np.full((8, 8, 3), .9), 2)
        self.control.default_work = None
        self.control.default_base = self.work / 'original.png'
        Image.new('RGB', (8, 8), (51, 51, 51)).save(self.control.default_base)
        self.control.source_info = {'height': 2, 'body_centre': [0, 0, 1],
                                    'boxes_px': {'face': [384, 384, 640, 1152]}}
        self.data['face_source'] = {'source': 'default-fast-fallback'}
        def ordinary(client, prep, current, default, cams, work, data, *args, **kwargs):
            self.assertNotIn('beauty_context', kwargs)
            self.assertEqual(kwargs['user_note'], NOTE)
            self.assertEqual(kwargs['attempts'], 1)
            np.testing.assert_allclose(default.T, .2)
            default.T[3:5] = [.3, .4, .5]
            data['views']['body+90'] = {'accepted': True}
            checkpoint = args[4]
            checkpoint(ok=False, status='running')
            self.assertIsNot(data, self.data['default_fast_fallback'])
            self.assertIsNot(data['views'], self.data['default_fast_fallback']['views'])
            render = work / 'wave-1'
            render.mkdir(exist_ok=True)
            Image.new('RGBA', (8, 8), (51, 51, 51, 255)).save(render / 'head+0.png')
            return current, False
        mask = np.ones((8, 8), bool)
        with mock.patch.object(fast, 'waves', side_effect=wave_error or ordinary), \
             mock.patch.object(self.control, 'prepare_hair_views'), \
             mock.patch.object(beauty, 'union', return_value=mask, side_effect=mask_error), \
             mock.patch.object(cameras, 'read_exr', return_value=None), \
             mock.patch.object(cameras, 'decode_pos', return_value=(pos, mask)):
            stopped = self.control.default_fast(mock.Mock(), self.work, tex, {'head+0': cam},
                                                self.work, 10, mock.Mock(), True, 'RF2R',
                                                mock.Mock(), NOTE)
        np.testing.assert_allclose(tex.T, .9)
        return stopped

    def test_default_fast_resolution_keeps_original_base_and_ordinary_side_paint(self):
        self.assertFalse(self.resolve_default_atlas())
        atlas = self.control.fallback_texture
        np.testing.assert_array_equal((atlas[:3] * 255).round(), 51)
        np.testing.assert_array_equal((atlas[3, 0] * 255).round(), [76, 102, 128])
        self.assertTrue(self.data['default_fast_fallback']['views']['body+90']['accepted'])
        self.assertFalse(self.data['face_source']['cap_reached'])

    def test_default_resolution_needs_no_additional_face_or_skin_model_call(self):
        self.assertFalse(self.resolve_default_atlas(mask_error=AssertionError('unexpected mask call')))

    def test_default_wave_transport_and_replay_errors_stay_loud(self):
        for error in (ApiError('wave failed'), ReplayMiss('missing wave')):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                self.resolve_default_atlas(wave_error=error)

    def test_whole_head_is_exact_on_all_sides_and_only_the_neck_blends(self):
        # The synthetic head has front/back/side texels at identical heights,
        # an ear outside the face, a bald scalp and a hole in the hair mask.
        h, w = 1536, 1024
        mask = np.zeros((h, w), np.uint8)
        mask[100:400, 420:600] = 255
        mask[170:230, 470:530] = 0
        (self.work / 'probe').mkdir()
        Image.fromarray(mask).save(self.work / 'probe/head.png')
        self.control.source_info = {'height': 2., 'body_centre': [0, 0, 0],
            'boxes_px': {'head': [420, 100, 600, 400], 'face': [450, 200, 570, 400]}}
        cam = cameras.camera('probe0', [0, 0, 0], 0, 2 * 1.06, (1024, 1536))
        k = cameras.px_per_unit(cam)
        uv = np.array([[500, 110], [425, 270], [500, 210], [500, 350],
                       [500, 420], [500, 450], [500, 500], [150, 300]])
        pos = np.zeros((2, 4, 3))
        pos.reshape(-1, 3)[:, 0] = (uv[:, 0] - 512) / k
        pos.reshape(-1, 3)[:, 2] = (768 - uv[:, 1]) / k
        pos.reshape(-1, 3)[:, 1] = [-.2, .2, 0, .3, 0, 0, 0, 0]
        normals = np.zeros_like(pos)
        normals.reshape(-1, 3)[:] = [[0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0]] * 2
        default = np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 255
        self.control.key = 'head+0'
        self.control.fallback_texture = default
        for source in ('default', 'default-fast-fallback'):
            for grade in (.1, .9):
                self.data['face_source'] = {'source': source}
                tex = I.Texture.from_arrays(pos, normals, np.ones((2, 4), bool),
                                            np.full((2, 4, 3), grade), 2)
                self.control.paint(tex, self.work, {})
                pixels = tex.T.reshape(-1, 3)
                np.testing.assert_array_equal(pixels[:4], default.reshape(-1, 3)[:4])
                self.assertTrue((pixels[4:6] > default.reshape(-1, 3)[4:6]).all())
                self.assertTrue((pixels[4:6] < grade).all())
                np.testing.assert_allclose(pixels[6:], grade)
                self.assertFalse(self.data['face_source']['head_region']['grade_inside'])

    def test_connected_hat_uses_default_at_all_depths_but_separate_prop_does_not(self):
        probe = self.work / 'probe'
        probe.mkdir()
        rgba = np.zeros((1536, 1024, 4), np.uint8)
        rgba[20:120, 300:750, 3] = 255  # hat extends above and beside the semantic head box
        rgba[100:450, 450:570, 3] = 255
        rgba[20:180, 90:110, 3] = 255  # separate raised prop
        Image.fromarray(rgba).save(probe / 'probe0.png')
        self.control.source_info = {'height': 2., 'body_centre': [0, 0, 0],
            'boxes_px': {'head': [420, 150, 600, 450], 'face': [450, 200, 570, 400]}}
        cam = cameras.camera('probe0', [0, 0, 0], 0, 2.12, (1024, 1536))
        k = cameras.px_per_unit(cam)
        uv = np.array([[320, 50], [320, 50], [100, 50], [500, 550]])
        pos = np.zeros((1, 4, 3))
        pos[0, :, 0] = (uv[:, 0] - 512) / k
        pos[0, :, 2] = (768 - uv[:, 1]) / k
        pos[0, :2, 1] = [-.2, .2]
        normals = np.zeros_like(pos)
        self.control.key = 'head+0'
        self.control.fallback_texture = np.full((1, 4, 3), .2, np.float32)
        self.data['face_source'] = {'source': 'default'}
        tex = I.Texture.from_arrays(pos, normals, np.ones((1, 4), bool),
                                    np.full((1, 4, 3), .9, np.float32), 2)
        self.control.paint(tex, self.work, {})
        np.testing.assert_array_equal(tex.T[0, :2], self.control.fallback_texture[0, :2])
        np.testing.assert_allclose(tex.T[0, 2:], .9)
        self.assertGreater(self.data['face_source']['head_region']['headgear_pixels'], 0)

    def test_long_hair_is_preserved_without_filling_the_torso_opening(self):
        # Hair ring encloses a torso-sized hole; only the cranial hole is filled.
        mask = np.zeros((1536, 1024), np.uint8)
        mask[100:700, 400:620] = 255
        mask[200:680, 440:580] = 0
        (self.work / 'probe').mkdir()
        Image.fromarray(mask).save(self.work / 'probe/head.png')
        self.control.source_info = {'height': 2., 'body_centre': [0, 0, 0],
            'boxes_px': {'head': [400, 100, 620, 700], 'face': [440, 200, 580, 350]}}
        k = 1536 / 2.12
        uv = np.array([[420, 550], [510, 220], [510, 550], [420, 550]])
        pos = np.zeros((1, 4, 3))
        pos[0, 3, 1] = .2  # torso behind hanging hair must keep beauty paint
        pos[0, :, 0] = (uv[:, 0] - 512) / k
        pos[0, :, 2] = (768 - uv[:, 1]) / k
        tex = I.Texture.from_arrays(pos, np.ones_like(pos), np.ones((1, 4), bool),
                                    np.full((1, 4, 3), .9), 2)
        self.control.key = 'head+0'
        self.control.fallback_texture = np.full((1, 4, 3), .2, np.float32)
        self.data['face_source'] = {'source': 'default'}
        self.control.paint(tex, self.work, {})
        np.testing.assert_allclose(tex.T[0, :2], .2)
        np.testing.assert_allclose(tex.T[0, 2:], .9)

    def test_hanging_hair_reuses_existing_masks_in_replay_mode_and_misses_stay_loud(self):
        image = Image.new('RGB', (8, 8), 'brown')
        image.save(self.work / 'head+0/input.png')
        mask = np.ones((8, 8), bool)
        view = {'key': 'head+0', 'cam': {}, 'valid': mask, 'depth': np.zeros((8, 8))}
        self.data['face_source'] = {'source': 'default'}
        owner = self
        class Cached:
            mode = 'live'
            def segment(self, image, text):
                owner.assertEqual(self.mode, 'replay')
                owner.assertEqual(text, 'hair')
                return [SimpleNamespace(mask=mask)]
        client = Cached()
        with mock.patch.object(I, 'load_accepted_views', return_value=[view]):
            self.control.prepare_hair_views(client, self.work, {}, 2)
        self.assertEqual(client.mode, 'live')
        np.testing.assert_array_equal(self.control.hair_views[0]['head_hair'], mask)
        self.assertEqual(self.data['face_source']['cached_hair_views'], ['head+0'])
        for error in (ReplayMiss('missing accepted hair mask'), ApiError('failed mask')):
            with mock.patch.object(I, 'load_accepted_views', return_value=[view]), \
                 mock.patch.object(Cached, 'segment', side_effect=error), self.assertRaises(type(error)):
                self.control.prepare_hair_views(client, self.work, {}, 2)

    def test_hair_mask_does_not_copy_a_collar_detail_inside_the_setup_head_mask(self):
        self.control.source_info = {'height': 2., 'body_centre': [0, 0, 0],
            'boxes_px': {'head': [400, 100, 620, 700], 'face': [440, 200, 580, 350]}}
        (self.work / 'probe').mkdir()
        Image.new('L', (1024, 1536), 255).save(self.work / 'probe/head.png')
        cam = cameras.camera('probe0', [0, 0, 0], 0, 2.12, (1024, 1536))
        k = cameras.px_per_unit(cam)
        pos = np.array([[[(420-512)/k, 0, (768-550)/k], [(480-512)/k, 0, (768-550)/k]]])
        mask = np.zeros((1536, 1024), bool)
        mask[550, 420] = True  # hair; nearby collar detail is absent from the semantic hair mask
        self.control.hair_views = [{'cam': cam, 'head_hair': mask, 'valid': np.ones_like(mask),
                                    'depth': np.zeros_like(mask, dtype=float)}]
        self.control.key = 'head+0'
        self.control.fallback_texture = np.full((1, 2, 3), .2, np.float32)
        self.data['face_source'] = {'source': 'default'}
        tex = I.Texture.from_arrays(pos, np.ones_like(pos), np.ones((1, 2), bool),
                                    np.full((1, 2, 3), .9), 2)
        self.control.paint(tex, self.work, {})
        np.testing.assert_allclose(tex.T[0, 0], .2)
        np.testing.assert_allclose(tex.T[0, 1], .9)

    def test_no_head_or_face_bounds_remains_exportable_and_records_the_limit(self):
        self.control.key = 'body+0'
        self.control.fallback_texture = np.ones((1, 1, 3), np.float32)
        self.data['face_source'] = {'source': 'default-fast-fallback'}
        self.control.paint(SimpleNamespace(), self.work, {})
        self.assertEqual(self.data['face_source']['texels'], 0)
        self.assertEqual(self.data['face_source']['head_region'], 'not detected')

    def test_no_beauty_acceptance_still_publishes_a_fallback_version(self):
        records = {key: {'accepted': False, 'reason': 'outline moved'} for key in self.data['views']}
        data = {'views': records, 'face_source': {'source': 'default-fast-fallback'}}
        run = self.work / 'run'
        run.mkdir()
        tex = mock.Mock()
        tex.save.side_effect = lambda path: Path(path)
        control = mock.Mock()
        def export(mode, args, *rest, **kw):
            Path(args['out']).write_bytes(b'export')
        texture = self.work / 'texture-v1.png'
        texture.write_bytes(b'atlas')
        with mock.patch.object(fast, 'reproject'), \
             mock.patch('herogen.blender.refine.backend.blender', side_effect=export):
            publisher = fast.Publisher(run, self.work, tex, self.work, data, {}, 10, face_control=control)
            self.assertEqual(publisher.publish(1), texture)
            self.assertIsNone(publisher.publish(1))
            publisher.finish()
        self.assertTrue(publisher.progress['done'])
        self.assertEqual((run / '03-refined.glb').read_bytes(), b'export')
        self.assertEqual(publisher.progress['versions'][0]['accepted'], [])

    def test_default_head_does_not_add_a_skin_request(self):
        Image.new('L', (8, 8), 0).save(self.head.tasks[1][3] / 'face.png')
        self.control.choose([self.head], self.work)
        client = SimpleNamespace(segment=mock.Mock(side_effect=AssertionError('unexpected mask call')))
        self.control.prepare_region(client, self.work)
        client.segment.assert_not_called()


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class EyeQualityTests(unittest.TestCase):
    boxes = [[10, 16, 46, 40], [74, 16, 110, 40]]

    @staticmethod
    def face(blur=0):
        from PIL import ImageDraw, ImageFilter
        image = Image.new('RGB', (120, 64), (190, 155, 130))
        draw = ImageDraw.Draw(image)
        for x in (10, 74):
            draw.ellipse((x, 16, x + 36, 40), fill='white', outline='black', width=2)
            draw.ellipse((x + 11, 17, x + 25, 39), fill=(30, 90, 130), outline='black', width=2)
            draw.ellipse((x + 16, 22, x + 21, 35), fill='black')
        return image.filter(ImageFilter.GaussianBlur(blur))

    def test_identical_eyes_pass_and_soft_blobs_fail(self):
        image = self.face()
        self.assertTrue(B.eye_quality(image, image, self.boxes)['accepted'])
        result = B.eye_quality(self.face(3), image, self.boxes)
        self.assertFalse(result['accepted'])
        self.assertTrue(all(not eye['accepted'] for eye in result['eyes']))

    def test_sharper_eyes_pass_and_each_eye_is_required(self):
        image, soft = self.face(), self.face(3)
        self.assertTrue(B.eye_quality(image, soft, self.boxes)['accepted'])
        image.paste(soft.crop((70, 0, 120, 64)), (70, 0))
        result = B.eye_quality(image, self.face(), self.boxes)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['eyes'][0]['accepted'])
        self.assertFalse(result['eyes'][1]['accepted'])

    def test_flat_ordinary_eyes_are_ineligible_even_against_flat_beauty(self):
        flat = Image.new('RGB', self.face().size, 'grey')
        for candidate in (flat, self.face()):
            with self.subTest(flat_beauty=candidate is flat):
                result = B.eye_quality(candidate, flat, self.boxes)
                self.assertFalse(result['accepted'])
                self.assertEqual(result['reason'], 'ordinary eye detail unavailable')
                self.assertTrue(all(not eye['eligible'] for eye in result['eyes']))
                self.assertTrue(all(eye['ratios']['laplacian_variance'] is None for eye in result['eyes']))

    def test_missing_or_invalid_boxes_cannot_pass(self):
        image = self.face()
        self.assertFalse(B.eye_quality(image, image, self.boxes[:1])['accepted'])
        with self.assertRaises(ValueError):
            B.eye_quality(image, image.resize((60, 32)), self.boxes)
        with self.assertRaises(ValueError):
            B.eye_quality(image, image, [[10, 10, 10, 20], self.boxes[1]])



@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class FastBeautyStageTests(unittest.TestCase):
    setUp = note_fixture.FastNoteTests.setUp
    identity = staticmethod(fixture.StageTests.identity)
    def render_many(self, mode, args, log, **kwargs):
        note_fixture.FastNoteTests.render_many(self, mode, args, log, **kwargs)
        if mode == 'render':
            from PIL import ImageDraw
            for cam in args['cams']:
                path = Path(args['out']) / (cam['name'] + '.png')
                with Image.open(path) as image:
                    image = image.copy()
                draw = ImageDraw.Draw(image)
                for x in (20, 36):
                    draw.ellipse((x, 24, x + 8, 32), fill='white', outline='black')
                    draw.ellipse((x + 3, 26, x + 5, 30), fill='black')
                image.save(path)
    run_fast = note_fixture.FastNoteTests.run_fast
    beauty_folder = fixture.StageTests.beauty_folder

    def test_body_reference_uses_surface_prompt_and_retained_mask_without_a_model_call(self):
        import hashlib
        folder, _ = self.beauty_folder()
        references = json.loads((folder / 'references.json').read_text())
        for view in references['views'].values():
            view['image_sha256'] = hashlib.sha256((folder / view['image']).read_bytes()).hexdigest()
        control = B.FaceControl(folder, references, data={'views': {}})
        before = len(self.client.log.calls)
        reference, options = control.options(SimpleNamespace(name='body+0', record={}), 0,
                                             Image.new('RGB', (64, 64)))
        self.assertEqual(options['recipe_name'], 'RF17B')
        self.assertEqual(options['reference_record']['yaw'], 0)
        self.assertEqual(options['surface_reference_mask'].size, reference.size)
        self.assertEqual(len(self.client.log.calls), before)
        _, head = control.options(SimpleNamespace(name='head+0', record={}), 0, reference)
        self.assertEqual(head['recipe_name'], 'RF6B')
        self.assertNotIn('surface_reference_mask', head)

    @staticmethod
    def slot_kind(kw):
        return 'beauty' if kw.get('recipe_name') else kw.get('attempt_start', 0)

    @classmethod
    def rejecting(cls, original, reject):
        """Make the real request, then reject the sample at the gate when `reject(view, kind)`."""
        def view(*args, **kw):
            result = original(*args, **kw)
            record = args[7]
            if reject(args[4]['name'], cls.slot_kind(kw)) and record['accepted']:
                record.update(accepted=False, reason='outline moved')
                record['attempts'][-1].update(accepted=False, reason='outline moved')
            return result
        return view

    def run_beauty(self, *, complete=True, default=True, reject=None, **kwargs):
        folder, _ = self.beauty_folder()
        self.tex.idx = np.arange(16)
        prep = self.run / 'prep'
        prep.mkdir(exist_ok=True)
        Image.new('RGB', (4, 4), 'blue').save(prep / 'base.png')
        features = {'both_eyes': complete, 'mouth': complete, 'eye_regions': 2 if complete else 0,
                    'eye_boxes': [[20, 24, 28, 32], [36, 24, 44, 32]] if complete else []}
        original = stage._view
        def view(*args, **kw):
            if not default and kw.get('recipe_name') in (None, 'RF2R') and args[4]['name'] == 'head+0':
                args[7].update(accepted=False, reason='outline moved')
                return args[2]
            return original(*args, **kw)
        if reject is not None:
            view = self.rejecting(view, reject)
        with mock.patch.object(stage, '_artwork_grade', return_value=None), \
             mock.patch.object(beauty, 'frontal_features', return_value=features), \
             mock.patch.object(B.FaceControl, 'paint') as paint, \
             mock.patch.object(stage, '_view', side_effect=view):
            record = self.run_fast(beauty_dir=folder, **kwargs)
        self.assertGreater(paint.call_count, 0)
        return record

    def paid_run(self, *, all_flags=False, reject=None, beauty=True):
        """Run with the real request-key cache and fake only the paid transport.

        The fixture's usual edit override counts cache replays as paid, so it cannot prove this cap.
        Every edit is recorded with its view, slot folder, run (nested default-head or main) and key.
        """
        self.client.mode = 'live'
        paid = Counter()
        ordinary = Counter()
        edits = []
        lock = threading.Lock()
        render = self.render_many
        def distinct_render(mode, args, log, **kwargs):
            render(mode, args, log, **kwargs)
            if mode == 'render':
                for cam in args['cams']:
                    path = Path(args['out']) / (cam['name'] + '.png')
                    with Image.open(path) as image:
                        image = image.copy()
                    image.putpixel((32, 32), (40 + sum(map(ord, cam['name'])) % 100, 80, 100, 255))
                    image.save(path)
        def post(client, endpoint, body, **kwargs):
            self.assertEqual(endpoint, 'images/edits')
            view = client.cache_dir.parts[-2]
            with lock:
                paid[view] += 1
                if 'face-source' in client.cache_dir.parts:
                    ordinary[view] += 1
                self.assertLessEqual(paid[view], 3, f'fourth paid sample for {view}')
            png = body['images'][0]['image_url'].split(',', 1)[1]
            return {'data': [{'b64_json': png}]}
        def edit(client, *args, **kwargs):
            result = ModelClient.edit_image(client, *args, **kwargs)
            call = client.log.calls[-1]
            parts = client.cache_dir.parts
            with lock:
                edits.append({'view': parts[-2], 'slot': parts[-1], 'nested': 'face-source' in parts,
                              'attempt': kwargs.get('attempt', 0), 'key': call['key'], 'cached': call['cached']})
            return result
        self.client.cache_dir.mkdir()
        self.assertFalse(list(self.client.cache_dir.iterdir()))
        fixture.StageTests.closeup_artwork(self)
        with mock.patch.object(fixture.FakeClient, 'edit_image', edit), \
             mock.patch.object(fixture.FakeClient, '_post', post), \
             mock.patch.object(self, 'render_many', side_effect=distinct_render), \
             mock.patch.object(stage, 'art_references', side_effect=lambda run, cam, *a:
                 ((('back', Image.new('RGB', (64, 64), 'green')),), {}, {})
                 if stage.side_art_yaw(cam['angle']) else ((), {}, {})), \
             mock.patch('herogen.blender.refine.region_palette.prepare_manifest', return_value='manifest.json'), \
             mock.patch('herogen.blender.refine.region_palette.export_texture',
                        side_effect=lambda m, p, t, *a, **kw: (t, {})):
            if beauty:
                record = self.run_beauty(cap_usd=None, side_beauty=all_flags, region_palette=all_flags,
                                         reject=reject)
            else:
                with mock.patch.object(stage, '_view', side_effect=self.rejecting(stage._view, reject)):
                    record = self.run_fast(cap_usd=None)
        self.assertEqual(sum(paid.values()), sum(c['kind'] == 'edit' and not c['cached']
                                               for c in self.client.log.calls))
        return record, paid, ordinary, edits

    expected_views = property(lambda self: {('hair-' if hair else '') + name
                                            for wave in fast.WAVES for name, hair in wave})

    def assert_paid_samples(self, *, all_flags=False):
        record, paid, ordinary, edits = self.paid_run(all_flags=all_flags)
        expected = self.expected_views
        self.assertEqual(paid, Counter({view: 2 for view in expected}))
        self.assertEqual(ordinary, Counter({view: 1 for view in expected}))
        self.assertTrue(all(view['tries'] == 1 for view in record['default_fast_fallback']['views'].values()))
        self.assertTrue(any(c['kind'] == 'edit' and c['cached'] for c in self.client.log.calls))
        self.assertFalse([e for e in edits if e['slot'] == 'attempt-2'])

    def test_uncached_beauty_pays_at_most_two_samples_per_view(self):
        self.assert_paid_samples()

    def test_uncached_all_three_pays_at_most_two_samples_per_view(self):
        self.assert_paid_samples(all_flags=True)

    def test_accepted_default_slot_pays_exactly_two_samples_per_view(self):
        record, paid, ordinary, edits = self.paid_run(reject=lambda view, kind: kind == 'beauty')
        expected = self.expected_views
        self.assertEqual(paid, Counter({view: 2 for view in expected}))
        self.assertEqual(ordinary, Counter({view: 1 for view in expected}))
        self.assertFalse([e for e in edits if e['slot'] == 'attempt-2'])
        self.assertEqual({k: v['winning_slot'] for k, v in record['views'].items()},
                         {view: 2 for view in expected})
        self.assertEqual(record['face_source'], {'source': 'default', 'view': 'head+0', 'attempt': 0,
                                                 **{k: record['face_source'][k] for k in record['face_source']
                                                    if k not in ('source', 'view', 'attempt')}})

    def assert_double_rejection(self, all_flags):
        record, paid, ordinary, edits = self.paid_run(
            all_flags=all_flags, reject=lambda view, kind: kind in ('beauty', 0))
        expected = self.expected_views
        self.assertEqual(paid, Counter({view: 3 for view in expected}))
        # The default-head run pays ordinary attempts 0 and 1; the main slots replay both.
        self.assertEqual(ordinary, Counter({view: 2 for view in expected}))
        for view in expected:
            nested = {e['attempt']: e['key'] for e in edits if e['nested'] and e['view'] == view}
            main = {e['slot']: e for e in edits if not e['nested'] and e['view'] == view}
            self.assertEqual(set(nested), {0, 1})
            self.assertEqual(main['attempt-1']['key'], nested[0])
            self.assertEqual(main['attempt-2']['key'], nested[1])
            self.assertEqual(main['attempt-2']['attempt'], 1)
            self.assertTrue(main['attempt-1']['cached'] and main['attempt-2']['cached'])
            self.assertEqual(record['views'][view]['winning_slot'], 3)
            fallback = record['default_fast_fallback']['views'][view]
            self.assertEqual((fallback['accepted'], fallback['tries'], fallback['selected_attempt']),
                             (True, 2, 1))
        self.assertEqual(record['face_source']['source'], 'default')
        self.assertEqual(record['face_source']['attempt'], 1)
        self.assertEqual(record['views']['head+0']['selected_attempt'], 2)
        atlas = self.run / 'blender-work/texture-refine/default-fast/blender-work/texture-refine/texture.png'
        import hashlib
        self.assertEqual(record['face_source']['atlas_sha256'],
                         hashlib.sha256(atlas.read_bytes()).hexdigest())

    def test_double_rejection_pays_one_shared_third_sample(self):
        self.assert_double_rejection(False)

    def test_double_rejection_with_all_flags_pays_one_shared_third_sample(self):
        self.assert_double_rejection(True)

    def test_every_rejection_never_pays_a_fourth_sample(self):
        record, paid, ordinary, edits = self.paid_run(reject=lambda view, kind: True)
        expected = self.expected_views
        self.assertEqual(paid, Counter({view: 3 for view in expected}))
        self.assertEqual(ordinary, Counter({view: 2 for view in expected}))
        self.assertFalse(any(v['accepted'] for v in record['views'].values()))
        self.assertTrue(all(v['tries'] == 3 for v in record['views'].values()))
        self.assertEqual(record['face_source']['source'], 'default-fast-fallback')
        self.assertFalse([e for e in edits if e['slot'] not in ('attempt-0', 'attempt-1', 'attempt-2')])

    def test_plain_fast_never_launches_a_conditional_slot(self):
        record, paid, ordinary, edits = self.paid_run(beauty=False, reject=lambda view, kind: True)
        expected = self.expected_views
        self.assertEqual(paid, Counter({view: 2 for view in expected}))
        self.assertFalse(ordinary)
        self.assertEqual(sorted((e['slot'], e['attempt']) for e in edits),
                         sorted([('attempt-0', 0), ('attempt-1', 1)] * len(expected)))
        self.assertTrue(all(v['tries'] == 2 and not v['accepted'] for v in record['views'].values()))

    def test_complete_beauty_front_is_kept_and_progress_and_note_survive(self):
        record = self.run_beauty(user_note=NOTE, cap_usd=5)
        self.assertEqual(record['face_source']['source'], 'beauty')
        self.assertTrue(record['face_fallback']['accepted'])
        self.assertLessEqual(record['views']['head+0']['tries'], fast.ATTEMPTS)
        self.assertTrue(all(text.endswith(f'«{NOTE}»') for text, _, _ in self.client.inputs))
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertTrue(progress['done'])
        self.assertGreaterEqual(len(progress['versions']), 2)
        previous = set()
        for version in progress['versions']:
            self.assertGreater(set(version['accepted']), previous)
            previous = set(version['accepted'])
            self.assertEqual(version['face_source']['source'], 'beauty')
            self.assertTrue((self.run / version['glb']).is_file())
        self.assertEqual((self.run / '03-refined.glb').read_bytes(),
                         (self.run / progress['versions'][-1]['glb']).read_bytes())

    def test_incomplete_beauty_front_uses_accepted_default_face(self):
        record = self.run_beauty(complete=False, cap_usd=5)
        self.assertEqual(record['face_source']['source'], 'default')
        self.assertTrue(record['views']['head+0']['accepted'])
        self.assertEqual(record['views']['head+0']['prompt'], 'RF2R')
        self.assertFalse(record['beauty_head_candidate']['accepted'])
        self.assertTrue(record['views']['body+0']['accepted'])
        self.assertEqual(record['views']['body+0']['winning_slot'], 2)
        self.assertEqual(len(record['views']), 6)

    def test_no_accepted_frontal_exports_with_default_fast_fallback(self):
        record = self.run_beauty(complete=False, default=False, cap_usd=5)
        self.assertEqual(record['status'], 'complete')
        self.assertEqual(record['face_source']['source'], 'default-fast-fallback')
        self.assertTrue((self.run / '03-refined.glb').exists())
        self.assertIn('default_fast_fallback', record)
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertTrue(progress['done'])
        self.assertTrue(progress['versions'])
        self.assertTrue(all(v['face_source']['source'] == 'default-fast-fallback'
                            for v in progress['versions']))

    def test_model_errors_propagate_and_never_substitute_the_base(self):
        for error in (ApiError('failed'), ReplayMiss('missing')):
            with mock.patch.object(stage, '_view', side_effect=error):
                with self.assertRaises(type(error)):
                    self.run_beauty(cap_usd=5)
            self.assertFalse((self.run / '03-refined.glb').exists())

    def test_missing_face_drains_and_propagates_a_late_model_error(self):
        import time
        original = stage._view
        def view(*args, **kwargs):
            if args[4]['name'] == 'body+90' and kwargs.get('attempt_start') == 0:
                time.sleep(.2)
                raise ApiError('late side repaint failure')
            return original(*args, **kwargs)
        with mock.patch.object(stage, '_view', side_effect=view):
            with self.assertRaisesRegex(ApiError, 'late side repaint failure'):
                self.run_beauty(complete=False, default=False, cap_usd=5)
        record = json.loads((self.run / 'stage-3f.json').read_text())
        self.assertEqual(record['status'], 'error')
        self.assertEqual(record['default_fast_fallback']['views']['body+90']['reason'], 'ApiError')
        self.assertFalse((self.run / '03-refined.glb').exists())

    def test_face_selection_shares_the_checkpoint_writer_lock(self):
        budgets = []
        initialize, select = stage.Budget.__init__, B.FaceControl.choose
        def init(budget, *args, **kwargs):
            initialize(budget, *args, **kwargs)
            budget.lock = TrackedLock(budget.lock)
            budgets.append(budget)
        def choose(control, *args):
            self.assertTrue(budgets[0].lock.owned, 'selection must exclude concurrent record writes')
            return select(control, *args)
        with mock.patch.object(stage.Budget, '__init__', new=init), \
             mock.patch.object(B.FaceControl, 'choose', new=choose):
            record = self.run_beauty(cap_usd=5)
        self.assertEqual(record['face_source']['source'], 'beauty')

    def test_skin_mask_model_failure_stays_loud(self):
        original = beauty.union
        def union(client, image, text):
            if text == 'skin':
                raise ApiError('skin mask unavailable')
            return original(client, image, text)
        with mock.patch.object(beauty, 'union', side_effect=union):
            with self.assertRaisesRegex(ApiError, 'skin mask unavailable'):
                self.run_beauty(cap_usd=5)
        record = json.loads((self.run / 'stage-3f.json').read_text())
        self.assertEqual(record['status'], 'error')
        self.assertEqual(record['face_source']['region_error'], 'ApiError')
        self.assertFalse((self.run / '03-refined.glb').exists())
