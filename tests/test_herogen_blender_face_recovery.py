"""Fast-lane face-only paint from a head+0 sample that only the outline check rejected."""
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender import refine as stage
from herogen.pipeline.client import SamInstance
from tests import test_herogen_blender_refine as fixture
from tests import test_herogen_blender_refine_fast as fast_fixture
from tests.test_herogen_blender_refine import FakeClient, HAVE_REFINE, disc

if HAVE_REFINE:
    from herogen.blender.refine import face_recovery as R

FIGURE = (60, 80, 100)
FACE = disc((64, 64), 32, 32, 8)
MARK = (5, 30)  # a pixel only the widened head repaint covers
INNER = (32, 46)  # a figure pixel, outside the face, that marks a same-outline repaint


def widened(image):
    """A repaint whose silhouette grows to the left (hair wider than the mesh) with the same face."""
    edit = image.copy()
    pixels = np.asarray(edit).copy()
    pixels[10:54, 2:14] = FIGURE
    return Image.fromarray(pixels)


def marked(image):
    """A repaint with the render's outline, marked for the SAM stand-in."""
    edit = image.copy()
    edit.putpixel(INNER, (200, 10, 10))
    return edit


def trial(tries, reason, checks=True):
    out = {'try': tries, 'accepted': False, 'reason': reason}
    if checks:
        out['checks'] = {'iou': .8, 'median_px': 30, 'p90_px': 60}
    return out


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class RecoveryRuleTests(unittest.TestCase):
    def records(self, head_reason='outline moved', body_accepted=False, head_trials=None):
        head = {'accepted': False, 'reason': head_reason,
                'attempts': head_trials if head_trials is not None else [trial(1, 'outline moved')]}
        body = {'accepted': body_accepted, 'reason': 'accepted' if body_accepted else 'outline moved',
                'attempts': []}
        return {'head+0': head, 'body+0': body}

    def test_due_only_when_both_frontals_resolved_and_rejected(self):
        self.assertTrue(R.due(self.records()))
        self.assertFalse(R.due(self.records(body_accepted=True)))
        accepted = self.records()
        accepted['head+0'].update(accepted=True, reason='accepted')
        self.assertFalse(R.due(accepted))
        self.assertFalse(R.due(self.records(head_reason='incomplete')))
        pending = self.records()
        pending['body+0']['reason'] = 'incomplete'
        self.assertFalse(R.due(pending))
        self.assertFalse(R.due(self.records(head_reason='head_not_found')))

    def test_only_outline_rejections_are_candidates_lowest_first(self):
        record = {'attempts': [trial(2, 'outline moved'), trial(1, 'new face outside the render face'),
                               trial(3, 'cancelled: lower attempt accepted', checks=False),
                               trial(4, 'outline moved', checks=False), trial(5, 'outline moved')]}
        self.assertEqual(R.candidates(record), [1, 4])

    def test_view_never_paints_with_an_accepted_frontal(self):
        recovery = {'applied': True, 'camera': {'name': 'head+0'}}
        records = self.records(body_accepted=True)
        self.assertIsNone(R.view(Path('unused'), records, recovery, 2, None))
        self.assertIsNone(R.view(Path('unused'), self.records(), {'applied': False}, 2, None))


class Segmenter:
    """SAM stand-in: a face disc on the render, a chosen face on the widened repaint, optional hair."""
    def __init__(self, edit_face=FACE, hair=None):
        self.edit_face, self.hair, self.queries = edit_face, hair, []
        self.render_face = FACE

    def __call__(self, client, image, text):
        client._call(fixture.SAM_MODEL, 'segment')
        repaint = image.size == (64, 64) and (image.getpixel(MARK) == FIGURE
                                              or image.getpixel(INNER) == (200, 10, 10))
        self.queries.append((text, repaint))
        if text == 'face':
            return [SamInstance(mask=self.edit_face if repaint else self.render_face, box=(0, 0, 1, 1))]
        if text == 'hair' and self.hair is not None and not repaint:
            return [SamInstance(mask=self.hair, box=(0, 0, 1, 1))]
        return []


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class FastRecoveryStageTests(unittest.TestCase):
    setUp = fast_fixture.FastStageTests.setUp
    render_many = fast_fixture.FastStageTests.render_many
    identity = staticmethod(fixture.StageTests.identity)

    def run_stage(self, *, head=widened, body='grey', segmenter=None, profile='fast'):
        original = FakeClient.edit_image
        def edit(child, prompt, images, **kw):
            parts = child.cache_dir.parts
            if 'head+0' in parts and head is not None:
                child.inputs.append((prompt, images, kw))
                child._call(fixture.IMAGE_MODEL, 'edit')
                return head(images[0])
            if 'body+0' in parts and body == 'grey':
                child.inputs.append((prompt, images, kw))
                child._call(fixture.IMAGE_MODEL, 'edit')
                return Image.new('RGB', images[0].size, (115, 115, 115))
            return original(child, prompt, images, **kw)
        segmenter = segmenter or Segmenter()
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(FakeClient, 'edit_image', edit), \
             mock.patch.object(FakeClient, 'segment', lambda client, image, text: segmenter(client, image, text)):
            if profile == 'best':
                with mock.patch.object(stage, 'REFINE_ORDER', ('head+0', 'body+0')), \
                     mock.patch.object(stage, 'HAIR_ORDER', ()):
                    record = stage.run_stage3f(self.run, self.client)
            else:
                record = stage.run_stage3f(self.run, self.client, refine_profile='fast')
        return record, segmenter

    work = property(lambda self: self.run / 'blender-work/texture-refine')

    def test_outline_only_head_paints_the_face_inside_the_render_face_mask(self):
        record, _ = self.run_stage()
        self.assertFalse(record['views']['head+0']['accepted'])
        self.assertFalse(record['views']['body+0']['accepted'])
        self.assertEqual({t['reason'] for t in record['views']['head+0']['attempts']}, {'outline moved'})
        recovery = record['face_recovery']
        self.assertTrue(recovery['applied'])
        self.assertEqual((recovery['view'], recovery['attempt']), ('head+0', 0))
        self.assertEqual(recovery['reason'], R.REASON)
        with Image.open(self.work / 'face-recovery/weight.png') as image:
            weight = np.asarray(image.convert('L')) / 255
        self.assertEqual(recovery['mask_area_px'], int((weight > 0).sum()))
        # Written only inside the render's SAM face mask, feathered inward and never outward.
        self.assertFalse(weight[~FACE].any())
        self.assertEqual(weight[32, 32], 1)
        edge = FACE & ~np.pad(FACE, 1)[2:, 1:-1]
        self.assertTrue((weight[edge] < 1).all())
        # The paint is projected at head+0's place, first, in the final version.
        cams = [call.args[0]['name'] for call in self.tex.apply.call_args_list]
        final = cams[-5:]
        self.assertEqual(final[0], 'head+0')
        self.assertNotIn('body+0', final)
        applied = self.tex.apply.call_args_list[-5].args[5]
        np.testing.assert_allclose(applied, weight, atol=1 / 255)
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertEqual(progress['versions'][-1]['face_recovery'], 0)
        self.assertEqual(record['final_model'], '03-refined.glb')

    def test_no_recovery_with_an_accepted_frontal(self):
        with mock.patch.object(R, 'evaluate', wraps=R.evaluate) as evaluate:
            record, _ = self.run_stage(body=None)
        self.assertTrue(record['views']['body+0']['accepted'])
        self.assertNotIn('face_recovery', record)
        evaluate.assert_not_called()
        progress = json.loads((self.run / 'stage-3f-progress.json').read_text())
        self.assertTrue(all('face_recovery' not in v for v in progress['versions']))
        self.assertFalse((self.work / 'face-recovery').exists())

    def test_no_recovery_when_the_face_outside_check_fails(self):
        moved = disc((64, 64), 32, 52, 8)  # the repaint's face slid down onto the neck
        record, _ = self.run_stage(segmenter=Segmenter(edit_face=moved))
        recovery = record['face_recovery']
        self.assertFalse(recovery['applied'])
        self.assertEqual([row['reason'] for row in recovery['evaluated']],
                         ['new face outside the render face'] * 2)
        self.assertFalse((self.work / 'face-recovery').exists())
        cams = [call.args[0]['name'] for call in self.tex.apply.call_args_list]
        self.assertNotIn('head+0', cams)

    def test_oversized_render_face_mask_skips_recovery(self):
        whole = np.ones((64, 64), bool)  # SAM returned the whole frame as "face"
        segmenter = Segmenter(edit_face=whole)
        segmenter.render_face = whole
        record, _ = self.run_stage(segmenter=segmenter)
        recovery = record['face_recovery']
        self.assertFalse(recovery['applied'])
        self.assertEqual(recovery['reason'], R.OVERSIZED_REASON)
        row, = recovery['evaluated']  # the shared render mask stops further attempts
        self.assertEqual(row['reason'], R.OVERSIZED_REASON)
        self.assertGreater(row['face_fraction'], R.FACE_FRACTION_MAX)
        self.assertFalse((self.work / 'face-recovery').exists())
        cams = [call.args[0]['name'] for call in self.tex.apply.call_args_list]
        self.assertNotIn('head+0', cams)

    def test_non_outline_rejection_never_recovers(self):
        # The outline passes but the face moved: only the face-outside check rejects the head.
        moved = disc((64, 64), 32, 52, 8)
        record, segmenter = self.run_stage(head=marked,
                                           segmenter=Segmenter(edit_face=moved))
        reasons = {t['reason'] for t in record['views']['head+0']['attempts']}
        self.assertEqual(reasons, {'new face outside the render face'})
        self.assertEqual(record['face_recovery'], {
            'applied': False, 'view': 'head+0', 'evaluated': [],
            'reason': 'no head+0 sample was rejected only by the outline check'})

    def test_hair_and_outline_band_are_excluded(self):
        hair = np.zeros((64, 64), bool)
        hair[:30] = True
        record, _ = self.run_stage(segmenter=Segmenter(hair=hair))
        with Image.open(self.work / 'face-recovery/weight.png') as image:
            weight = np.asarray(image.convert('L')) / 255
        self.assertTrue(record['face_recovery']['applied'])
        self.assertFalse(weight[hair].any())
        self.assertTrue(weight[~hair & FACE].any())

    def test_recovery_reuses_paid_samples_only(self):
        record, _ = self.run_stage()
        edits = [c for c in record['calls'] if c['kind'] == 'edit']
        self.assertEqual(len(edits), 2 * 6)  # two attempts per pass, as without recovery
        recovery_calls = [c for c in record['calls'] if c['kind'] == 'segment']
        self.assertTrue(recovery_calls)

    def test_best_is_unchanged(self):
        with mock.patch.object(R, 'resolve', wraps=R.resolve) as resolve:
            record, _ = self.run_stage(profile='best')
        resolve.assert_not_called()
        self.assertNotIn('face_recovery', record)


if __name__ == '__main__':
    unittest.main()
