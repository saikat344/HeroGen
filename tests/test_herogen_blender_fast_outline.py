"""The fast outline measurement keeps cloth close to the background grey; best keeps its own."""
import unittest
from unittest import mock

import numpy as np

from herogen.blender import refine as stage
from tests import test_herogen_blender_refine as fixture
from tests import test_herogen_blender_refine_fast as fast_fixture
from tests.test_herogen_blender_refine import HAVE_REFINE, disc

if HAVE_REFINE:
    from herogen.blender.refine import character as C, imaging as I


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class NearGreyClothTests(unittest.TestCase):
    """A grey sleeve close to the background grey, touching the exterior, as in s037's head view."""
    def scene(self):
        size = 96
        figure = disc((size, size), 48, 48, 24)
        sleeve = np.zeros((size, size), bool)
        sleeve[40:60, 4:40] = True
        figure |= sleeve
        bg = np.array([C.BG] * 3, np.float32)
        render = np.broadcast_to(bg, (size, size, 3)).copy()
        render[figure] = (.25, .3, .35)
        render[sleeve] = bg + (-.04, -.03, -.07)   # the render's grey cloth
        return figure, sleeve, render, bg

    def test_kept_grey_sleeve_stays_in_the_fast_measurement(self):
        figure, sleeve, render, bg = self.scene()
        edit = render.copy()
        edit[sleeve] = bg + (.01, .02, -.03)        # the repaint lightened the sleeve toward the backdrop
        old = I.figure_mask(edit, render_figure=figure)
        new = I.figure_mask(edit, render_figure=figure, render_rgb=render)
        self.assertLess(old[sleeve].mean(), .5)     # the established mask floods the sleeve
        self.assertGreater(new[sleeve].mean(), .95)
        self.assertFalse(I.outline_gate(figure, old, closeup=True)[0])
        self.assertTrue(I.outline_gate(figure, new, closeup=True)[0])
        # Nothing changes outside the render's figure.
        np.testing.assert_array_equal(old & ~figure, new & ~figure)

    def test_a_real_outline_move_is_still_measured(self):
        figure, sleeve, render, bg = self.scene()
        render[sleeve] = (.7, .2, .2)               # coloured cloth in the render
        edit = render.copy()
        edit[sleeve] = bg                           # the repaint removed the sleeve
        new = I.figure_mask(edit, render_figure=figure, render_rgb=render)
        self.assertLess(new[sleeve].mean(), .05)
        self.assertFalse(I.outline_gate(figure, new, closeup=True)[0])

    def test_limits_are_unchanged(self):
        self.assertEqual(I.GATE, {"iou_min": 0.90, "median_px": 6.0, "p90_px": 16.0})
        self.assertEqual(I.CLOSEUP_GATE, {"iou_min": 0.90, "median_px": 10.0, "p90_px": 24.0})
        self.assertEqual(I.BG_TOL, .07)

    def test_without_render_colour_the_mask_is_the_established_one(self):
        figure, sleeve, render, bg = self.scene()
        edit = render.copy()
        edit[sleeve] = bg + (.01, .02, -.03)
        np.testing.assert_array_equal(I.figure_mask(edit, render_figure=figure),
                                      I.figure_mask(edit, render_figure=figure, render_rgb=None))


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class OutlineProfileTests(unittest.TestCase):
    setUp = fast_fixture.FastStageTests.setUp
    render_many = fast_fixture.FastStageTests.render_many
    identity = staticmethod(fixture.StageTests.identity)

    def measured(self, **kwargs):
        with mock.patch('herogen.blender.refine.backend.blender', side_effect=self.render_many), \
             mock.patch.object(I, 'figure_mask', wraps=I.figure_mask) as mask:
            stage.run_stage3f(self.run, self.client, **kwargs)
        self.assertTrue(mask.call_args_list)
        return [call for call in mask.call_args_list if call.kwargs.get('render_rgb') is not None]

    def test_fast_outline_gate_uses_the_render_colour(self):
        self.assertTrue(self.measured(refine_profile='fast'))

    def test_best_measurement_is_unchanged(self):
        self.assertFalse(self.measured())


if __name__ == '__main__':
    unittest.main()
