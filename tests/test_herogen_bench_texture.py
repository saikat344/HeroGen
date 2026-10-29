"""Ground-truth bench texture sampling: Cycles' texel-centre, interpolation and extension conventions."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

try:
    import numpy as np

    from herogen.bench import texture as TEX

    HAVE_DEPS = True
except ImportError:  # numpy missing
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy is required")
class SampleTest(unittest.TestCase):
    def setUp(self):
        # 2 x 4 texture; row 0 is v = 0. Columns: 0, 1/3, 2/3, 1 alpha
        self.t = np.array([[0.0, 1 / 3, 2 / 3, 1.0], [0.0, 1 / 3, 2 / 3, 1.0]])

    def test_texel_centres_read_exactly(self):
        for i in range(4):
            u = (i + 0.5) / 4
            self.assertAlmostEqual(TEX.sample(self.t, u, 0.25), self.t[0, i])
            self.assertAlmostEqual(TEX.sample(self.t, u, 0.25, "Closest"), self.t[0, i])

    def test_linear_blends_between_centres_and_closest_does_not(self):
        u = 0.25                      # halfway between the centres of columns 0 and 1
        self.assertAlmostEqual(TEX.sample(self.t, u, 0.25), 1 / 6)
        self.assertAlmostEqual(TEX.sample(self.t, u + 1e-9, 0.25, "Closest"), 1 / 3)
        self.assertAlmostEqual(TEX.sample(self.t, u - 1e-9, 0.25, "Closest"), 0.0)

    def test_extension_at_the_border(self):
        u = 0.0                       # half a texel left of column 0's centre: blends with the column beyond
        self.assertAlmostEqual(TEX.sample(self.t, u, 0.25, "Linear", "REPEAT"), 0.5)       # wraps to column 3
        self.assertAlmostEqual(TEX.sample(self.t, u, 0.25, "Linear", "EXTEND"), 0.0)       # repeats column 0
        self.assertAlmostEqual(TEX.sample(self.t, u, 0.25, "Linear", "CLIP"), 0.0)         # outside reads 0
        self.assertAlmostEqual(TEX.sample(self.t, 1.0, 0.25, "Linear", "CLIP"), 0.5)       # half of column 3
        self.assertAlmostEqual(TEX.sample(self.t, 1.0, 0.25, "Linear", "MIRROR"), 1.0)     # mirrors column 3
        self.assertAlmostEqual(TEX.sample(self.t, 1.1, 0.25, "Closest", "CLIP"), 0.0)

    def test_unsupported_modes_are_refused(self):
        with self.assertRaises(ValueError):
            TEX.sample(self.t, 0.5, 0.5, "Cubic")
        with self.assertRaises(ValueError):
            TEX.wrap(0, 4, "SPHERE")

    def test_barycentric_uv(self):
        uv = TEX.barycentric_uv([0, 0, 0], [1, 0, 0], [0, 0, 1], [0.25, 0, 0.5], [0, 0], [1, 0], [0, 1])
        np.testing.assert_allclose(uv, [0.25, 0.5])
        flat = TEX.barycentric_uv([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0], [1, 0], [0, 1])
        np.testing.assert_allclose(flat, [1 / 3, 1 / 3])


if __name__ == "__main__":
    unittest.main()
