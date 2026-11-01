"""Head density changes preserve existing UV continuity and relative island shape."""
import unittest

import numpy as np

from herogen.blender.uv_density import weight_front


class FrontDensityTests(unittest.TestCase):
    def test_front_island_is_enlarged_without_tearing_shared_edges(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], float)
        faces = np.array([[0, 1, 2], [2, 1, 3], [0, 2, 1]])
        uv = np.array([[[0, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 1]],
                       [[2, 0], [2, 1], [3, 0]]], float)
        weighted = weight_front(verts, faces, uv, 2)
        np.testing.assert_allclose(weighted[0, 1], weighted[1, 1])
        np.testing.assert_allclose(weighted[0, 2], weighted[1, 0])
        np.testing.assert_allclose(weighted[0, 1] - weighted[0, 0], [2, 0])
        np.testing.assert_allclose(weighted[0, 2] - weighted[0, 0], [0, 2])
        np.testing.assert_array_equal(weighted[2], uv[2])
        np.testing.assert_array_equal(weight_front(verts, faces, uv, 1), uv)
        np.testing.assert_array_equal(uv[0, 0], [0, 0])


if __name__ == "__main__":
    unittest.main()
