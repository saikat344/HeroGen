"""herogen.blender.glb: the one GLB reader (contract frame, node transforms, strides, short indices, texture
coordinates, materials and images) on files written by the tests themselves. No Blender, no model calls."""
from __future__ import annotations

import io
import json
import math
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from herogen.blender import glb as G

BOX_FACES = np.array([
    [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
    [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5],
])


def box(centre, half):
    corners = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], float) * 2 - 1
    return np.asarray(centre, float) + corners * np.asarray(half, float), BOX_FACES.copy()


def png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def pack_glb(doc: dict, binary: bytes) -> bytes:
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (struct.pack("<III", G.GLB_MAGIC, 2, 28 + len(js) + len(binary))
            + struct.pack("<II", len(js), G.CHUNK_JSON) + js + struct.pack("<II", len(binary), G.CHUNK_BIN) + binary)


class GlbReader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="herogen-glb-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip_in_the_contract_frame(self):
        v, f = box((0.2, -0.4, 1.0), (0.5, 0.3, 1.0))
        g = G.load_glb(G.write_glb(self.tmp / "b.glb", [("Body", v, f)]))
        self.assertEqual([m.name for m in g.meshes], ["Body"])
        np.testing.assert_allclose(g.meshes[0].verts, v, atol=1e-6)
        np.testing.assert_array_equal(g.meshes[0].faces, f)
        self.assertIsNone(g.meshes[0].uv)
        self.assertIsNone(g.merged_uv())
        self.assertIsNone(g.texture_rgb())

    def test_texture_coordinates_material_and_image(self):
        v, f = box((0, 0, 1), (0.5, 0.5, 1))
        uv = np.random.default_rng(0).random((len(v), 2))
        tex = np.zeros((4, 6, 3), np.uint8)
        tex[:, :3] = (200, 10, 20)
        g = G.load_glb(G.write_glb(self.tmp / "t.glb", [("A", v, f, uv), ("B", v + 2, f)], texture_png=png(tex)))
        np.testing.assert_allclose(g.meshes[0].uv, uv, atol=1e-6)
        self.assertIsNone(g.meshes[1].uv)
        self.assertEqual(g.image_sizes, [(6, 4)])
        self.assertEqual(len(g.materials), 1)
        self.assertEqual(g.materials[0].image, 0)
        np.testing.assert_array_equal(g.meshes[0].face_material, np.zeros(12))
        np.testing.assert_array_equal(g.merged_face_material(), np.r_[np.zeros(12), -np.ones(12)])
        merged = g.merged_uv()
        self.assertEqual(merged.shape, (16, 2))
        self.assertTrue((merged[8:] == 0).all())
        rgb = g.texture_rgb()
        self.assertEqual(rgb.shape, (4, 6, 3))
        np.testing.assert_array_equal(rgb[0, 0], (200, 10, 20))
        self.assertEqual(g.image(0).shape, (4, 6, 4))
        self.assertIsNone(g.image(5))

    def test_node_transforms_strides_and_short_indices(self):
        # One triangle stored interleaved (position + a second VEC3, stride 24) with uint16 indices, on a child
        # node translated and scaled under a parent rotated 90 degrees about glTF +Y.
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        inter = np.concatenate([pos, np.zeros_like(pos)], axis=1).tobytes()
        idx = np.array([0, 1, 2], np.uint16).tobytes()
        doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
               "nodes": [{"rotation": [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)], "children": [1]},
                         {"mesh": 0, "translation": [0, 0, 2], "scale": [2, 2, 2], "name": "tri", "skin": 0}],
               "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
               "buffers": [{"byteLength": len(inter) + 8}],
               "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(inter), "byteStride": 24},
                               {"buffer": 0, "byteOffset": len(inter), "byteLength": 6}],
               "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                             {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}]}
        p = self.tmp / "t.glb"
        p.write_bytes(pack_glb(doc, inter + idx))
        g = G.load_glb(p)
        # glTF point p -> child (2p + (0,0,2)) -> rotate +90 about +Y ((x,y,z) -> (z,y,-x)) -> contract (x,-z,y).
        local = pos.astype(float) * 2 + [0, 0, 2]
        rot = np.stack([local[:, 2], local[:, 1], -local[:, 0]], axis=1)
        expect = np.stack([rot[:, 0], -rot[:, 2], rot[:, 1]], axis=1)
        np.testing.assert_allclose(g.meshes[0].verts, expect, atol=1e-6)
        self.assertEqual(g.triangles, 1)
        self.assertTrue(any("bind pose" in n for n in g.notes))     # skinned meshes are flagged, not posed

    def test_mirroring_node_flips_the_winding(self):
        v, f = box((0, 0, 0), (1, 1, 1))
        p = G.write_glb(self.tmp / "m.glb", [("M", v, f)])
        data = p.read_bytes()
        n = struct.unpack_from("<I", data, 12)[0]
        doc = json.loads(data[20:20 + n])
        binary = data[20 + n + 8:]
        doc["nodes"][0]["scale"] = [-1, 1, 1]
        p.write_bytes(pack_glb(doc, binary))
        g = G.load_glb(p)
        np.testing.assert_array_equal(g.meshes[0].faces, f[:, ::-1])
        np.testing.assert_allclose(g.meshes[0].verts[:, 0], -v[:, 0], atol=1e-6)

    def test_refuses_compressed_and_non_glb_files(self):
        doc = {"asset": {"version": "2.0"}, "extensionsRequired": ["KHR_draco_mesh_compression"]}
        p = self.tmp / "d.glb"
        p.write_bytes(pack_glb(doc, b""))
        with self.assertRaisesRegex(G.GlbError, "compressed"):
            G.load_glb(p)
        (self.tmp / "x.glb").write_text('{"asset": {}}')
        with self.assertRaises(G.GlbError):
            G.load_glb(self.tmp / "x.glb")


if __name__ == "__main__":
    unittest.main()
