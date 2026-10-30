"""herogen.pipeline.checks: the math of every check, on synthetic masks."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import checks as C  # noqa: E402
from herogen.pipeline import imaging  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, place  # noqa: E402


def scaled(mask: np.ndarray, f: float) -> np.ndarray:
    img = imaging.to_mask_image(mask)
    return imaging.from_mask_image(img.resize((round(img.size[0] * f), round(img.size[1] * f)), Image.NEAREST))


def view_set(h: int = 200, arms: str = "a", size: int = 300) -> dict[str, np.ndarray]:
    """Same-scale square view masks, figures centred with feet on one row."""
    out = {}
    for name, pose in (("front", "front"), ("back", "front"), ("right", "right"), ("left", "left")):
        m = figure_mask(h, pose, arms)
        out[name] = place(m, (size, size), ((size - m.shape[1]) // 2, (size - h) // 2))
    return out


class BasicsTest(unittest.TestCase):
    def test_check_and_gate(self):
        a = C.check("x", 0.9, 0.8)
        b = C.check("y", 3, 4, op="==")
        c = C.check("z", 0.1, 0.5, gate=False)
        self.assertTrue(a["pass"])
        self.assertFalse(b["pass"])
        self.assertFalse(c["pass"])
        self.assertTrue(C.gate_ok([a, c]))
        self.assertFalse(C.gate_ok([a, b]))
        self.assertEqual(set(a), {"name", "value", "threshold", "op", "pass", "gate"})

    def test_agreement(self):
        self.assertEqual(C.agreement([2.0, 4.0]), 0.5)
        self.assertEqual(C.agreement([3.0, 3.0, 3.0]), 1.0)
        self.assertEqual(C.agreement([0.0, 3.0]), 0.0)
        self.assertEqual(C.agreement([]), 0.0)

    def test_aspect(self):
        m = np.zeros((50, 50), dtype=bool)
        m[10:40, 20:30] = True
        self.assertEqual(C.aspect(m), 3.0)
        self.assertEqual(C.aspect(np.zeros((5, 5), dtype=bool)), 0.0)


class AttemptRankTest(unittest.TestCase):
    EXPECTED = ("figures_found", "a", "b")

    def test_fewest_failed_gates_then_smallest_shortfall(self):
        one_far = [C.check("figures_found", 4, 4, op="=="), C.check("a", 0.5, 0.9), C.check("b", 0.2, 0.33, op="<=")]
        one_near = [C.check("figures_found", 4, 4, op="=="), C.check("a", 0.89, 0.9), C.check("b", 0.2, 0.33, op="<=")]
        two = [C.check("figures_found", 4, 4, op="=="), C.check("a", 0.89, 0.9), C.check("b", 0.34, 0.33, op="<=")]
        r_far, r_near, r_two = (C.attempt_rank(x, self.EXPECTED) for x in (one_far, one_near, two))
        self.assertEqual(r_near, (1, round(0.01 / 0.9, 6)))
        self.assertEqual(r_far[0], 1)
        self.assertLess(r_near, r_far)
        self.assertLess(r_far, r_two)  # one failed gate, however far, beats two
        self.assertEqual(r_two, (2, round(0.01 / 0.9 + 0.01 / 0.33, 6)))

    def test_unmeasured_gates_count_as_failed(self):
        unsplit = [C.check("figures_found", 3, 4, op="==")]
        self.assertEqual(C.attempt_rank(unsplit, self.EXPECTED), (3, 2.25))
        two = [C.check("figures_found", 4, 4, op="=="), C.check("a", 0.1, 0.9), C.check("b", 0.9, 0.33, op="<=")]
        self.assertLess(C.attempt_rank(two, self.EXPECTED), C.attempt_rank(unsplit, self.EXPECTED))

    def test_non_gating_checks_do_not_count(self):
        checks = [C.check("figures_found", 4, 4, op="=="), C.check("a", 0.95, 0.9), C.check("b", 0.2, 0.33, op="<="),
                  C.check("info", 0.0, 1.0, gate=False)]
        self.assertEqual(C.attempt_rank(checks, self.EXPECTED), (0, 0.0))


class CoverChecksTest(unittest.TestCase):
    def test_same_figure_at_another_scale_passes(self):
        sketch = place(figure_mask(200, "front"), (400, 400), (60, 100))
        cover = place(scaled(figure_mask(200, "front"), 1.6), (500, 500), (80, 120))
        checks, extra = C.cover_checks(sketch, cover)
        by = {c["name"]: c for c in checks}
        self.assertGreater(by["aspect_agreement"]["value"], 0.97)
        self.assertGreater(by["silhouette_iou_normalised"]["value"], 0.9)
        self.assertTrue(C.gate_ok(checks))
        self.assertIn("sketch_aspect_h_over_w", extra)

    def test_changed_pose_fails(self):
        sketch = place(figure_mask(200, "front", "a"), (400, 400), (60, 100))
        cover = place(figure_mask(200, "front", "down"), (400, 400), (60, 100))
        checks, _ = C.cover_checks(sketch, cover)
        self.assertFalse(C.gate_ok(checks))


class SideViewChecksTest(unittest.TestCase):
    def test_consistent_a_pose_set_passes(self):
        v = view_set()
        checks = C.side_view_checks(v, figures_found=4)
        by = {c["name"]: c for c in checks}
        self.assertEqual(by["height_agreement_4_views"]["value"], 1.0)
        self.assertEqual(by["front_back_width_agreement"]["value"], 1.0)
        self.assertGreater(by["left_right_mirror_iou"]["value"], 0.99)
        self.assertGreater(by["a_pose_hand_span_over_shoulders"]["value"], C.A_POSE_SPREAD_MIN)
        self.assertTrue(C.gate_ok(checks), checks)

    def test_arms_down_a_pose_ratio_is_recorded_not_gated(self):
        checks = C.side_view_checks(view_set(arms="down"), figures_found=4)
        by = {c["name"]: c for c in checks}
        self.assertFalse(by["a_pose_hand_span_over_shoulders"]["pass"])
        self.assertFalse(by["a_pose_hand_span_over_shoulders"]["gate"])
        self.assertIn("not gated", by["a_pose_hand_span_over_shoulders"]["note"])
        self.assertTrue(C.gate_ok(checks), checks)

    def test_mismatched_views_fail(self):
        v = view_set()
        v["back"] = place(scaled(figure_mask(200, "front"), 0.85), (300, 300), (60, 60))
        v["left"] = np.fliplr(v["left"])  # both profiles now face the same way
        by = {c["name"]: c for c in C.side_view_checks(v, figures_found=4)}
        self.assertFalse(by["height_agreement_4_views"]["pass"])
        self.assertFalse(by["front_back_width_agreement"]["pass"])
        self.assertLess(by["left_right_mirror_iou"]["value"], 0.7)

    def test_wrong_figure_count_stops_early(self):
        checks = C.side_view_checks({}, figures_found=3)
        self.assertEqual([c["name"] for c in checks], ["figures_found"])
        self.assertFalse(C.gate_ok(checks))

    def test_arm_reaching_forward_fails_profile_depth(self):
        v = view_set()
        m = v["right"].copy()
        ys, xs = np.nonzero(m)
        m[int(ys.min() + 0.2 * 200) : int(ys.min() + 0.25 * 200), xs.max() : xs.max() + 60] = True
        v["right"] = m
        by = {c["name"]: c for c in C.side_view_checks(v, figures_found=4)}
        self.assertFalse(by["right_profile_width_over_height"]["pass"])


class TopChecksTest(unittest.TestCase):
    def test_top_extents(self):
        v = view_set()
        top = imaging.top_from_hull(v["front"], v["back"], v["right"], v["left"])
        by = {c["name"]: c for c in C.top_view_checks(v, top)}
        self.assertGreater(by["top_width_vs_front_width"]["value"], 0.98)  # 1 px centring offset
        self.assertGreater(by["top_depth_vs_side_width"]["value"], 0.95)
        squashed = np.zeros_like(top)
        squashed[140:150, 100:200] = True
        by = {c["name"]: c for c in C.top_view_checks(v, squashed, gate=False)}
        self.assertFalse(by["top_width_vs_front_width"]["pass"])
        self.assertFalse(by["top_width_vs_front_width"]["gate"])

    def test_mirror_iou_and_silhouette_iou(self):
        r = figure_mask(200, "right")
        self.assertAlmostEqual(C.mirror_iou(np.fliplr(r), r), 1.0)
        self.assertAlmostEqual(C.silhouette_iou(r, scaled(r, 2.0)), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
