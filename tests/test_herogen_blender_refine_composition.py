"""Opt-in repaint composition and saved side-orientation requirements."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender import refine as stage
from herogen.blender.refine import fast, fast_beauty, region_palette
from tests.test_herogen_blender_fast_beauty import FastBeautyStageTests
from tests import test_herogen_blender_refine as fixture


class OrientationTests(unittest.TestCase):
    def test_absent_uncertain_swapped_and_corrected_records_use_default_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertFalse(stage.side_orientation(run)['enabled'])
            for data in ({}, {'checks': []}, self.record(-.3, .2), self.record(0, -.2),
                         self.record(.3, -.2, {'mirrored_left': True})):
                with self.subTest(data=data):
                    (run / 'stage-2.json').write_text(json.dumps(data))
                    result = stage.side_orientation(run)
                    self.assertFalse(result['enabled'])
                    self.assertTrue(result['reason'])

    @staticmethod
    def record(right=.3, left=-.2, fixes=None):
        return {'checks': [
            {'name': 'right_view_toe_offset_as_drawn', 'value': right, 'pass': right > 0},
            {'name': 'left_view_toe_offset_as_drawn', 'value': left, 'pass': left < 0}],
            'details': {'view_assignment': {'side_fixes': fixes or {}}}}

    def test_saved_unswapped_directions_enable_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'stage-2.json').write_text(json.dumps(self.record()))
            self.assertTrue(stage.side_orientation(run)['enabled'])


class CompositionTests(unittest.TestCase):
    setUp = FastBeautyStageTests.setUp
    identity = staticmethod(FastBeautyStageTests.identity)
    render_many = FastBeautyStageTests.render_many
    run_fast = FastBeautyStageTests.run_fast
    beauty_folder = FastBeautyStageTests.beauty_folder
    run_beauty = FastBeautyStageTests.run_beauty

    def artwork(self):
        for folder in ('views', 'masks'):
            (self.run / folder).mkdir(exist_ok=True)
        for name in ('front', 'right', 'back', 'left'):
            Image.new('RGB', (64, 64), 'green').save(self.run / 'views' / (name + '.png'))

    def test_beauty_and_side_references_compose_without_changing_frontal_candidates(self):
        self.artwork()
        with mock.patch.object(stage, 'art_references', side_effect=lambda run, cam, *a:
                ((('back', Image.new('RGB', (64, 64), 'green')),), {}, {})
                if stage.side_art_yaw(cam['angle']) else ((), {}, {})):
            rec = self.run_beauty(side_beauty=True, cap_usd=None)
        self.assertEqual(rec['face_source']['source'], 'beauty')
        self.assertEqual(rec['views']['head+0']['prompt'], 'RF6B')
        self.assertEqual(rec['face_fallback']['prompt'], 'RF2R')
        self.assertEqual(rec['views']['body+90']['prompt'], 'RF16B')
        self.assertEqual(rec['views']['body+90']['art_views'], ['back'])
        self.assertNotIn('budget', rec)

    def test_beauty_side_prompt_preserves_the_surface_only_contract(self):
        from herogen.blender.refine import prompts
        text = prompts.prompt('RF16B', 180)
        self.assertTrue(text.startswith(prompts.prompt('RF15B', 180)))
        self.assertIn(prompts.DESIGN_ONLY, text)
        self.assertIn('The first image alone determines the shape and position', text)
        self.assertIn('never copy them', text)
        self.assertEqual(prompts.ART_VARIANTS['RF6B'], 'RF16B')

    def test_beauty_and_palette_apply_after_face_ownership(self):
        events = []
        def paint(control, *args):
            events.append('face')
            control.default_head_weight = np.ones((4, 4), np.float32)
        def grade(manifest, prep, texture, *a, **kw):
            events.append('palette')
            np.testing.assert_array_equal(kw['protected_head'], np.ones((4, 4)))
            return texture, {'skin': {'status': 'fitted'}}
        with mock.patch.object(region_palette, 'prepare_manifest', return_value='manifest.json'), \
             mock.patch.object(region_palette, 'export_texture', side_effect=grade), \
             mock.patch.object(fast_beauty.FaceControl, 'paint', autospec=True, side_effect=paint):
            folder, _ = self.beauty_folder()
            self.tex.idx = np.arange(16)
            (self.run / 'prep').mkdir(exist_ok=True)
            Image.new('RGB', (4, 4), 'blue').save(self.run / 'prep/base.png')
            with mock.patch.object(stage, '_artwork_grade', return_value=None), \
                 mock.patch('herogen.blender.refine.beauty.frontal_features',
                            return_value={'both_eyes': True, 'mouth': True}):
                self.run_fast(beauty_dir=folder, region_palette=True, cap_usd=None)
        self.assertTrue(events)
        self.assertEqual(events, ['face', 'palette'] * (len(events) // 2))

    def test_side_and_palette_keep_independent_reference_and_export_steps(self):
        fixture.StageTests.closeup_artwork(self)
        with mock.patch.object(region_palette, 'prepare_manifest', return_value='manifest.json'), \
             mock.patch.object(region_palette, 'export_texture', side_effect=lambda m, p, t, *a, **kw: (t, {})) as grade:
            rec = self.run_fast(side_beauty=True, region_palette=True, cap_usd=None)
        self.assertTrue(rec['side_beauty_orientation']['enabled'])
        self.assertEqual(rec['views']['body+90']['prompt'], 'RF12R')
        self.assertEqual(rec['views']['body+0']['prompt'], 'RF2R')
        self.assertTrue(grade.called)
        self.assertNotIn('protected_head', grade.call_args.kwargs)

    def test_unverified_orientation_never_loads_artwork_and_records_why(self):
        (self.run / 'stage-2.json').unlink()
        rec = self.run_fast(side_beauty=True, cap_usd=None)
        self.assertFalse(rec['side_beauty_orientation']['enabled'])
        self.assertEqual(rec['views']['body+90']['prompt'], 'RF2R')
        self.assertNotIn('side_beauty', rec)
        self.assertEqual(rec['side_beauty_orientation']['reason'], 'stage-2 orientation record missing')

    def test_all_three_keep_the_default_head_and_orientation_fallback(self):
        (self.run / 'stage-2.json').write_text(json.dumps(OrientationTests.record(-.3, .2)))
        with mock.patch.object(region_palette, 'prepare_manifest', return_value='manifest.json'), \
             mock.patch.object(region_palette, 'export_texture', side_effect=lambda m, p, t, *a, **kw: (t, {})) as grade:
            rec = self.run_beauty(complete=False, side_beauty=True, region_palette=True, cap_usd=None)
        self.assertFalse(rec['side_beauty_orientation']['enabled'])
        self.assertEqual(rec['face_source']['source'], 'default')
        self.assertEqual(rec['views']['body+90']['prompt'], 'RF6B')
        self.assertTrue(grade.called)
        self.assertNotIn('budget', rec)


class PaletteProtectionTests(unittest.TestCase):
    def test_default_head_is_byte_exact_while_unprotected_clothing_is_graded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pos.exr').write_bytes(b'layout')
            import hashlib
            mask = np.ones((32, 32), bool)
            Image.fromarray(mask.astype(np.uint8) * 255).save(root / 'mask.png')
            Image.new('L', (32, 32), 255).save(root / 'cov.png')
            for name, colour in [('art', (180, 140, 110)), ('input', (90, 70, 55)), ('body+0', (90, 70, 55))]:
                Image.new('RGB', (32, 32), colour).save(root / (name + '.png'))
            data = {'atlas_layout_sha256': hashlib.sha256(b'layout').hexdigest(), 'artwork': 'art.png',
                    'artwork_masks': {'rest': 'mask.png'}, 'render_masks': {'rest': 'mask.png'},
                    'atlas_masks': {'rest': 'mask.png'}}
            (root / 'manifest.json').write_text(json.dumps(data))
            protected = np.zeros((32, 32), np.float32)
            protected[:16] = 1
            with mock.patch('herogen.blender.refine.backend.blender'):
                target, grades = region_palette.export_texture(root / 'manifest.json', root, root / 'input.png',
                    {'name': 'body+0'}, root, protected_head=protected)
            before, after = np.asarray(Image.open(root / 'input.png')), np.asarray(Image.open(target))
            np.testing.assert_array_equal(before[:16], after[:16])
            self.assertGreater(np.abs(before[16:].astype(int) - after[16:]).max(), 0)
            self.assertEqual(grades['rest']['protected_head_pixels'], 512)


del FastBeautyStageTests
