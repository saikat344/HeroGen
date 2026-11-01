"""Allocate more Head texels to frontal UV islands without splitting existing islands."""
from __future__ import annotations

import numpy as np


def weight_front(verts: np.ndarray, faces: np.ndarray, uv: np.ndarray, factor: float) -> np.ndarray:
    """Scale front-facing islands before packing. Shared UV edges stay continuous.

    The character faces -Y. An island qualifies when its area-weighted normal has a frontal cosine above
    0.5 (within 60 degrees of the face). Uniform island scaling preserves its unwrap and local aspect ratio.
    Packing afterwards restores the unit square and the requested gutter.
    """
    parent = list(range(len(faces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = {}
    for i, face in enumerate(faces):
        for j, k in ((0, 1), (1, 2), (2, 0)):
            a, b = sorted(((int(face[j]), *np.round(uv[i, j], 6)),
                           (int(face[k]), *np.round(uv[i, k], 6))))
            key = (a, b)
            if key in edges:
                parent[root(i)] = root(edges[key])
            else:
                edges[key] = i
    groups = {}
    for i in range(len(faces)):
        groups.setdefault(root(i), []).append(i)
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1)
    result = uv.copy()
    for ids in groups.values():
        if -normal[ids, 1].sum() > .5 * area[ids].sum():
            centre = uv[ids].reshape(-1, 2).mean(axis=0)
            result[ids] = centre + factor * (uv[ids] - centre)
    return result
