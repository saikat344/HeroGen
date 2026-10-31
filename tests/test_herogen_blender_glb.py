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


class GlbNormals(unittest.TestCase):
    setUp = GlbReader.setUp
    tearDown = GlbReader.tearDown

    # Keep these fixtures independent of Blender and exercise local glTF coordinates directly.
    def fixture(self, indexed=True):
        pos = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0],
                        [0.4e-6, 0, 0], [0, 0, 1], [2, 0, 0]], dtype="<f4")
        uv = np.arange(12, dtype="<f4").reshape(6, 2) / 12
        interleaved = np.concatenate([np.ones((6, 1), dtype="<f4"), pos, uv], axis=1).astype("<f4")
        binary = interleaved.tobytes() + np.arange(6, dtype="<u2").tobytes()
        doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
               "nodes": [{"mesh": 0, "translation": [3, 5, 7], "scale": [-1, 2, 3]},
                         {"mesh": 0, "rotation": [0, 1, 0, 0]}],
               "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                                            "material": 0, "extras": {"keep": True}}]}],
               "buffers": [{"byteLength": len(binary)}],
               "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": interleaved.nbytes, "byteStride": 24},
                               {"buffer": 0, "byteOffset": interleaved.nbytes, "byteLength": 12}],
               "accessors": [{"bufferView": 0, "byteOffset": 4, "componentType": 5126, "count": 6, "type": "VEC3"},
                             {"bufferView": 0, "byteOffset": 16, "componentType": 5126, "count": 6, "type": "VEC2"},
                             {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}],
               "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                              "extensions": {"KHR_materials_unlit": {}}}],
               "textures": [{"source": 0}], "images": [{"bufferView": 1, "mimeType": "image/png"}],
               "extensionsUsed": ["KHR_materials_unlit"], "extensions": {"example": {"value": 42}}}
        if indexed:
            doc["meshes"][0]["primitives"][0]["indices"] = 2
        return doc, binary

    def rewrite(self, doc, binary):
        source, out = self.tmp / "source.glb", self.tmp / "smooth.glb"
        source.write_bytes(pack_glb(doc, binary))
        count = G.add_missing_normals(source, out)
        return count, *G._chunks(out.read_bytes())

    def test_area_weighted_normals_across_uv_seams_preserve_everything_else(self):
        for indexed in (True, False):
            with self.subTest(indexed=indexed):
                doc, binary = self.fixture(indexed)
                count, after, payload = self.rewrite(doc, binary)
                self.assertEqual(count, 1)
                prim = after["meshes"][0]["primitives"][0]
                normal_index = prim["attributes"].pop("NORMAL")
                normals = G._accessor(after, payload, normal_index)
                # Face cross products are (0,0,2) and (0,2,0), with a tiny positional seam offset.
                np.testing.assert_allclose(normals[[0, 1, 3, 5]], np.tile([0, 1, 1], (4, 1)) / np.sqrt(2), atol=2e-7)
                np.testing.assert_allclose(normals[2], [0, 0, 1])
                np.testing.assert_allclose(normals[4], [0, 1, 0])
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7)
                self.assertEqual(payload[:len(binary)], binary)
                after["accessors"].pop()
                after["bufferViews"].pop()
                after["buffers"][0]["byteLength"] = doc["buffers"][0]["byteLength"]
                self.assertEqual(after, doc)
                # A second pass keeps the whole GLB byte-identical.
                out = self.tmp / "smooth.glb"
                before = out.read_bytes()
                self.assertEqual(G.add_missing_normals(out, out), 0)
                self.assertEqual(out.read_bytes(), before)

    def test_unequal_areas_and_grid_boundaries(self):
        doc, binary = self.fixture(False)
        pos = G._accessor(doc, binary, 0).copy()
        pos[4] = [0, 0, 3]  # second face has three times the area
        pos[3] = [0, 0, 0]
        raw = bytearray(binary)
        for i, point in enumerate(pos):
            raw[i * 24 + 4:i * 24 + 16] = point.astype("<f4").tobytes()
        _, after, payload = self.rewrite(doc, bytes(raw))
        normals = G._accessor(after, payload, after["meshes"][0]["primitives"][0]["attributes"]["NORMAL"])
        np.testing.assert_allclose(normals[0], [0, 3 / np.sqrt(10), 1 / np.sqrt(10)], atol=1e-7)
        raw[3 * 24 + 4:3 * 24 + 8] = np.array([0.6e-6], dtype="<f4").tobytes()
        _, after, payload = self.rewrite(doc, bytes(raw))
        normals = G._accessor(after, payload, after["meshes"][0]["primitives"][0]["attributes"]["NORMAL"])
        np.testing.assert_array_equal(normals[0], [0, 0, 1])
        np.testing.assert_array_equal(normals[3], [0, 1, 0])

    def test_cross_primitive_smoothing_preserves_authored_normals(self):
        doc, binary = self.fixture()
        prim = doc["meshes"][0]["primitives"][0]
        # Each primitive reads one triangle and one half of the interleaved vertex array.
        doc["accessors"][0]["count"] = doc["accessors"][1]["count"] = 3
        prim.pop("indices")
        doc["accessors"].extend([{**doc["accessors"][0], "byteOffset": 76},
                                 {**doc["accessors"][1], "byteOffset": 88},
                                 {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3"}])
        authored = np.tile([1, 0, 0], (3, 1)).astype("<f4").tobytes()
        doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(authored)})
        binary += authored
        doc["buffers"][0]["byteLength"] = len(binary)
        doc["meshes"][0]["primitives"].append({"attributes": {"POSITION": 3, "TEXCOORD_0": 4, "NORMAL": 5}})
        # A different mesh at the same coordinates must not contribute.
        doc["meshes"].append({"primitives": [{"attributes": {"POSITION": 3, "NORMAL": 5}}]})
        count, after, payload = self.rewrite(doc, binary)
        self.assertEqual(count, 1)
        normal = after["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        np.testing.assert_allclose(G._accessor(after, payload, normal)[0], [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], atol=2e-7)
        self.assertEqual(after["meshes"][0]["primitives"][1], doc["meshes"][0]["primitives"][1])
        self.assertEqual(payload[:len(binary)], binary)
        np.testing.assert_array_equal(G._accessor(after, payload, 5), [[1, 0, 0]] * 3)

    def test_every_mesh_and_missing_primitive_gets_its_own_accessor(self):
        doc, binary = self.fixture()
        primitive = doc["meshes"][0]["primitives"][0]
        doc["meshes"][0]["primitives"].append(json.loads(json.dumps(primitive)))
        doc["meshes"].append({"primitives": [json.loads(json.dumps(primitive))]})
        count, after, payload = self.rewrite(doc, binary)
        self.assertEqual(count, 3)
        indices = [p["attributes"]["NORMAL"] for m in after["meshes"] for p in m["primitives"]]
        self.assertEqual(len(set(indices)), 3)
        for index in indices:
            np.testing.assert_allclose(np.linalg.norm(G._accessor(after, payload, index), axis=1), 1, atol=1e-7)
        self.assertEqual(payload[:len(binary)], binary)

    def test_degenerate_isolated_and_cancelling_faces_have_unit_normals(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], float)
        source = G.write_glb(self.tmp / "degenerate.glb", [("M", pos, np.array([[0, 1, 2], [0, 2, 1], [0, 0, 0]]))])
        doc, binary = G._chunks(source.read_bytes())
        _, after, payload = self.rewrite(doc, binary)
        normal = after["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        normals = G._accessor(after, payload, normal)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
        # The isolated vertex uses the local glTF +Z fallback.
        np.testing.assert_array_equal(normals[3], [0, 0, 1])
        np.testing.assert_array_equal(normals[:3], np.tile([0, 1, 0], (3, 1)))

    def test_non_triangle_primitive_is_copied_byte_for_byte(self):
        doc, binary = self.fixture()
        doc["meshes"][0]["primitives"][0]["mode"] = 1
        count, after, payload = self.rewrite(doc, binary)
        self.assertEqual(count, 0)
        self.assertEqual(after, doc)
        self.assertEqual((self.tmp / "source.glb").read_bytes(), (self.tmp / "smooth.glb").read_bytes())

    def test_unknown_chunks_are_preserved(self):
        doc, binary = self.fixture()
        data = pack_glb(doc, binary)
        extra = struct.pack("<II", 4, 12345) + b"keep"
        data = data[:8] + struct.pack("<I", len(data) + len(extra)) + data[12:] + extra
        source, out = self.tmp / "source.glb", self.tmp / "smooth.glb"
        source.write_bytes(data)
        self.assertEqual(G.add_missing_normals(source, out), 1)
        self.assertTrue(out.read_bytes().endswith(extra))

    def test_malformed_glbs_raise_without_writing_destination(self):
        mutations = {
            "sparse": lambda d: d["accessors"][0].update(sparse={}),
            "external": lambda d: d["buffers"][0].update(uri="outside.bin"),
            "draco": lambda d: d.update(extensionsRequired=["KHR_draco_mesh_compression"]),
            "nested_compression": lambda d: d["meshes"][0]["primitives"][0].update(extensions={"KHR_draco_mesh_compression": {}}),
            "overrun": lambda d: d["accessors"][0].update(count=999),
            "bad_reference": lambda d: d["meshes"][0]["primitives"][0]["attributes"].update(POSITION=-1),
            "bad_component": lambda d: d["accessors"][0].update(componentType=123),
            "bad_view": lambda d: d["accessors"][0].update(bufferView=99),
            "bad_stride": lambda d: d["bufferViews"][0].update(byteStride=4),
            "short_view": lambda d: d["bufferViews"][0].update(byteLength=9999),
            "bad_indices": lambda d: d["accessors"][2].update(componentType=5126),
            "bad_counts": lambda d: d["accessors"][1].update(count=2),
            "negative_count": lambda d: d["accessors"][0].update(count=-1),
            "bad_structure": lambda d: d["accessors"].__setitem__(0, []),
            "bad_position_type": lambda d: d["accessors"][0].update(type="VEC2"),
        }
        source, out = self.tmp / "bad.glb", self.tmp / "out.glb"
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                doc, binary = self.fixture()
                mutate(doc)
                source.write_bytes(pack_glb(doc, binary))
                out.write_bytes(b"previous output")
                with self.assertRaises(G.GlbError):
                    G.add_missing_normals(source, out)
                self.assertEqual(out.read_bytes(), b"previous output")
        for value in (float("nan"), float("inf"), 1e20):
            with self.subTest(position=value):
                doc, binary = self.fixture()
                raw = bytearray(binary)
                raw[4:8] = np.array([value], dtype="<f4").tobytes()
                source.write_bytes(pack_glb(doc, bytes(raw)))
                with self.assertRaises(G.GlbError):
                    G.add_missing_normals(source, out)
                self.assertEqual(out.read_bytes(), b"previous output")
        doc, binary = self.fixture()
        raw = bytearray(binary)
        raw[-2:] = np.array([99], dtype="<u2").tobytes()
        source.write_bytes(pack_glb(doc, bytes(raw)))
        with self.assertRaisesRegex(G.GlbError, "index out of range"):
            G.add_missing_normals(source, out)
        self.assertEqual(out.read_bytes(), b"previous output")
        data = pack_glb(*self.fixture())
        for broken in (data[:-1], data[:20] + b"!" + data[21:], data[:12] + struct.pack("<I", len(data)) + data[16:]):
            source.write_bytes(broken)
            with self.assertRaises(G.GlbError):
                G.add_missing_normals(source, out)
            self.assertEqual(out.read_bytes(), b"previous output")


if __name__ == "__main__":
    unittest.main()
