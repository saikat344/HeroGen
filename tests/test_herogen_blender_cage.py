"""Continuous refined colour transfer on synthetic coarse/fine surfaces, without Blender."""
import unittest

import numpy as np

from herogen.blender import texture
from herogen.blender.split import Section


def strip(xs, height):
    verts = np.array([[x, y, height(x)] for x in xs for y in (-1., 1.)])
    faces = np.array([[2*i, 2*i+2, 2*i+3] for i in range(len(xs)-1)]
                     + [[2*i, 2*i+3, 2*i+1] for i in range(len(xs)-1)])
    return verts, faces


class TestCageProjection(unittest.TestCase):
    def setUp(self):
        self.v, self.f = strip(np.linspace(-1, 1, 201), lambda x: .5*abs(x) + .2*x*x)
        uv = (self.v[self.f, :2] + 1) / 2
        n = 512
        gradient = np.broadcast_to((np.arange(n)+.5)[None, :, None]/n*255, (n, n, 3)).copy()
        self.source = texture.SourceColour(self.v, self.f, uv, gradient, exact=True)

    def test_concave_crease_removes_closest_point_jump_and_gradient_duplication(self):
        coarse, faces = strip(np.array([-1., 0., 1.]), lambda x: .3 + .04*abs(x))
        # Samples cross the middle of both coarse facets; interpolate the same vertex normals as a bake.
        x = np.linspace(-.08, .08, 161)
        p = np.column_stack((x, np.zeros_like(x), .3 + .04*abs(x)))
        vn = texture.vertex_normals(coarse, faces).reshape(3, 2, 3).mean(axis=1)
        normals = np.column_stack([np.interp(x, [-1, 0, 1], vn[:, k]) for k in range(3)])
        ids, bary, _ = self.source.closest(p)
        old_uv = np.einsum('ni,nij->nj', bary, self.source.corner_uv[ids])
        ids, bary, _, fallback = self.source.project(p, normals, .8)
        new_uv = np.einsum('ni,nij->nj', bary, self.source.corner_uv[ids])
        spacing = np.diff(x) / 2
        self.assertGreater(float(np.max(np.linalg.norm(np.diff(old_uv, axis=0), axis=1) / spacing)), 8)
        ratios = np.linalg.norm(np.diff(new_uv, axis=0), axis=1) / spacing
        self.assertGreater(ratios.min(), .8)
        self.assertLess(ratios.max(), 1.3)
        self.assertFalse(fallback.any())
        colours, _, stats = self.source.cage(p, normals)
        expected = (x + 1) / 2 * 255
        np.testing.assert_allclose(colours[:, 0], expected, atol=1.5)
        self.assertTrue((np.diff(colours[:, 0]) > 0).all())
        self.assertEqual(stats['primary_projection']['fallback_fraction'], 0)

    def test_nearest_direction_distance_limit_and_zero_normal_fallback(self):
        p = np.array([[.2, 0, .5], [.2, 0, -.1], [3, 0, 0], [.2, 0, .5]])
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]], float)
        ids, bary, dist, fallback = self.source.project(p, normals, .45)
        np.testing.assert_array_equal(fallback, [False, False, True, True])
        hit = np.einsum('ni,nij->nj', bary[:2], self.v[self.f[ids[:2]]])
        np.testing.assert_allclose(hit[:, :2], [[.2, 0], [.2, 0]], atol=1e-12)
        np.testing.assert_allclose(dist[:2], [.392, .208], atol=1e-12)
        expected = self.source.closest(p[2:])
        np.testing.assert_array_equal(ids[2:], expected[0])
        np.testing.assert_allclose(bary[2:], expected[1])
        np.testing.assert_allclose(dist[2:], expected[2])
        self.assertTrue(self.source.project(p[:1], normals[:1], .01)[3][0])

    def test_bake_records_measured_limit_and_exact_fallback_fraction(self):
        verts = np.array([[2, -1, 0], [4, -1, 0], [2, 1, 0]], float)
        faces = np.array([[0, 1, 2]])
        uv = np.array([[[.1, .1], [.9, .1], [.1, .9]]])
        section = Section('Head', verts, faces, np.zeros(3), [], 0)
        atlas, stats, _ = texture.bake({'Head': section}, {'Head': uv}, {}, 32, primary=self.source)
        projection = stats['primary_projection']
        self.assertEqual(projection['closest_distance_max'], texture.CLOSEST_DISTANCE)
        self.assertEqual(projection['closest_normal_cos_min'], texture.CLOSEST_NORMAL_COS)
        self.assertEqual(projection['fallback_fraction'], 1)
        self.assertEqual(projection['fallback_count'], projection['samples'])
        self.assertEqual(projection['max_distance'],
                         projection['distance_factor'] * projection['nearest_distance_p95'])
        covered = texture._texels(verts[faces], uv, 32)[0]
        self.assertTrue((atlas.reshape(-1, 3)[covered] > 250).all())

    def test_area_weighted_normals_and_shared_edge_interpolation(self):
        verts = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]], float)
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        normals = texture.vertex_normals(verts, faces)
        expected = np.array([0., 6., 2.]) / np.sqrt(40)
        np.testing.assert_allclose(normals[:2], np.broadcast_to(expected, (2, 3)))
        a = np.array([.3, .7, 0]) @ normals[faces[0]]
        b = np.array([.7, .3, 0]) @ normals[faces[1]]
        np.testing.assert_allclose(a, b, atol=1e-15)

    def test_head_surface_field_is_continuous_scale_invariant_and_preserves_gradient(self):
        top, top_faces = strip(np.array([-1., 0., 1.]), lambda x: .3 + .04*abs(x))
        verts = np.concatenate((top, top * [1, 1, 0] + [0, 0, -.7]))
        faces = np.concatenate((top_faces, top_faces[:, ::-1] + len(top)))
        field, method = texture.cage_normals('Head', verts, faces)
        self.assertEqual(method, 'area-weighted vertex normals')
        moved, _ = texture.cage_normals('Head', verts * 7 + [4, -3, 6], faces)
        np.testing.assert_allclose(field, moved, atol=1e-14)
        body, method = texture.cage_normals('UpperTorso', verts, faces)
        self.assertEqual(method, 'area-weighted vertex normals')
        np.testing.assert_allclose(body, texture.vertex_normals(verts, faces))
        x = np.linspace(-.08, .08, 161)
        p = np.column_stack((x, np.zeros_like(x), .3 + .04*abs(x)))
        vn = field[:len(top)].reshape(3, 2, 3).mean(axis=1)
        normals = np.column_stack([np.interp(x, [-1, 0, 1], vn[:, k]) for k in range(3)])
        ids, bary, _, fallback = self.source.project(p, normals, .8)
        uv = np.einsum('ni,nij->nj', bary, self.source.corner_uv[ids])
        self.assertFalse(fallback.any())
        ratio = np.linalg.norm(np.diff(uv, axis=0), axis=1) / (np.diff(x) / 2)
        self.assertGreater(ratio.min(), .5)
        self.assertLess(ratio.max(), 2)
        self.assertTrue((np.diff(uv[:, 0]) > 0).all())
        # Head points without three nonzero extents retain the mesh normal field.
        flat = top * [1, 1, 0]
        actual, method = texture.cage_normals('Head', flat, top_faces)
        self.assertEqual(method, 'area-weighted vertex normals')
        np.testing.assert_allclose(actual, texture.vertex_normals(flat, top_faces))

    def test_overhang_coincident_sheet_and_interior_cap(self):
        from types import SimpleNamespace
        from unittest import mock

        face = np.array([[-.3, -.3, 0], [.3, -.3, 0], [.3, .3, 0], [-.3, .3, 0]])
        brim = np.array([[-.3, .15, .08], [.3, .15, .08], [.3, .32, .05], [-.3, .32, .05]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]])
        verts = np.vstack([face, brim])
        uv = np.array([[[.05, .1], [.45, .1], [.45, .9]], [[.05, .1], [.45, .9], [.05, .9]],
                       [[.55, .1], [.95, .9], [.95, .1]], [[.55, .1], [.55, .9], [.95, .9]]])
        image = np.zeros((64, 64, 3))
        image[:, :32] = [220, 175, 145]
        dark = [35, 20, 15]
        image[:, 32:] = dark
        source = texture.SourceColour(verts, faces, uv, image, exact=True)
        cap = np.array([[-.15, -.3, -.25], [.15, -.3, -.25], [0, -.2, -.25]])
        exact = SimpleNamespace(verts=np.vstack([verts, cap]), faces=np.vstack([faces, [8, 10, 9]]))
        cap_uv = np.vstack([uv * [.8, 1], [[[.82, .1], [.98, .1], [.90, .5]]]])
        atlas, _, _ = texture.bake({'Head': exact}, {'Head': cap_uv}, {}, 256, primary=source)
        self.assertLessEqual(float(np.abs(atlas[40:216, 120:190].astype(float) - dark).mean()), 5)
        exact.cap_triangles = 1
        with mock.patch.object(source, 'project', side_effect=AssertionError('reliable sheet needs no ray')):
            _, stats, _ = texture.bake({'Head': exact}, {'Head': cap_uv}, {}, 256, primary=source)
        self.assertEqual(stats['primary_projection']['ray_count'], 0)
        self.assertLess(stats['primary_projection']['nearest_distance_p95'], 1e-12)
        self.assertGreater(stats['primary_projection']['interior_count'], 0)
        remeshed = SimpleNamespace(verts=np.vstack([face + [0, 0, .025], brim]), faces=faces)
        atlas, _, _ = texture.bake({'Head': remeshed}, {'Head': uv}, {}, 256, primary=source)
        self.assertLessEqual(float(np.abs(atlas[40:216, 150:235].astype(float) - dark).mean()), 5)
        # Even the opposite normal must retain the coincident brim, rather than ray through to the face.
        point = np.array([[0., .22, .08 - (.22 - .15) * .03 / .17]])
        ids, bary, dist, fallback = source.project(point, np.array([[0., 0., 1.]]), .3)
        self.assertIn(ids[0], [2, 3])
        self.assertLess(dist[0], 1e-12)
        self.assertFalse(fallback[0])
        colour, _ = source(point)
        np.testing.assert_allclose(colour, [dark])



if __name__ == '__main__':
    unittest.main()
