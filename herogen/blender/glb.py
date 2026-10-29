"""Read triangle meshes from binary glTF (`.glb`) with numpy only, so scoring runs without Blender.

The one GLB reader of herogen/blender: our own bodies, SAM 3D Objects' source mesh and other tools' outputs are
read the same way. Node transforms are applied, and positions are converted from glTF axes (+Y up, front along
+Z) to the contract's Blender frame (+Z up, the character facing -Y): X = x, Y = -z, Z = y.

Supported: triangle primitives (mode 4), indexed or not, float or normalised-integer attributes, buffer views
with a stride; per vertex the base-colour texture coordinates and `COLOR_0`; per material the base-colour factor,
the alpha mode and cutoff, and the texture (embedded PNG, JPEG or WebP). Compressed meshes (Draco, meshopt),
sparse accessors and buffers outside the GLB are refused with a clear error rather than read wrongly. Skinned
meshes are read in their bind pose (joints and morph targets are not applied) and say so in `notes`;
`KHR_texture_transform` is not applied.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
TRIANGLES = 4

# componentType -> (numpy dtype, divisor for normalised integers)
_COMPONENTS = {
    5120: (np.int8, 127.0), 5121: (np.uint8, 255.0),
    5122: (np.int16, 32767.0), 5123: (np.uint16, 65535.0),
    5125: (np.uint32, None), 5126: (np.float32, None),
}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_UNSUPPORTED = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_meshopt_compression"}

# glTF (x, y, z) -> contract (X, Y, Z) = (x, -z, y).
GLTF_TO_CONTRACT = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


class GlbError(ValueError):
    pass


@dataclass
class Material:
    base_color: tuple[float, float, float, float]   # baseColorFactor, linear 0-1 as stored
    image: int | None                                # index into the GLB's images of the base-colour texture
    texcoord: int                                    # which TEXCOORD_n the texture reads
    alpha_mode: str = "OPAQUE"                       # glTF alphaMode: OPAQUE, MASK or BLEND
    alpha_cutoff: float = 0.5                        # glTF alphaCutoff (MASK only)


@dataclass
class Mesh:
    name: str
    verts: np.ndarray                    # float64 [N, 3], contract frame
    faces: np.ndarray                    # int64 [M, 3]
    uv: np.ndarray | None = None         # float64 [N, 2] glTF texture coordinates (v down), or None
    colors: np.ndarray | None = None     # float64 [N, 4] COLOR_0 in 0-1, or None
    face_material: np.ndarray | None = None   # int64 [M], index into Glb.materials, -1 = none


@dataclass
class Glb:
    meshes: list[Mesh]
    image_sizes: list[tuple[int, int]]   # (width, height) of each embedded image that could be decoded
    notes: list[str]
    materials: list[Material] = field(default_factory=list)
    images: list[bytes | None] = field(default_factory=list, repr=False)   # encoded bytes per glTF image

    @property
    def triangles(self) -> int:
        return int(sum(len(m.faces) for m in self.meshes))

    def merged(self) -> tuple[np.ndarray, np.ndarray]:
        """All meshes as one vertex and face array (faces re-indexed)."""
        verts, faces, base = [], [], 0
        for m in self.meshes:
            verts.append(m.verts)
            faces.append(m.faces + base)
            base += len(m.verts)
        if not verts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(verts), np.concatenate(faces)

    def merged_uv(self) -> np.ndarray | None:
        """Per-vertex texture coordinates in `merged()` order (zeros for meshes without any); None if no mesh has
        texture coordinates."""
        if not any(m.uv is not None for m in self.meshes):
            return None
        return np.concatenate([m.uv if m.uv is not None else np.zeros((len(m.verts), 2)) for m in self.meshes])

    def merged_face_material(self) -> np.ndarray:
        """Per-face material index in `merged()` order (-1 = none)."""
        if not self.meshes:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([m.face_material if m.face_material is not None
                               else np.full(len(m.faces), -1, dtype=np.int64) for m in self.meshes])

    def image(self, index: int) -> np.ndarray | None:
        """Embedded image `index` decoded to float RGBA [H, W, 4] in 0-255, or None when it cannot be decoded."""
        raw = self.images[index] if 0 <= index < len(self.images) else None
        if raw is None:
            return None
        from PIL import Image

        try:
            with Image.open(io.BytesIO(raw)) as im:
                return np.asarray(im.convert("RGBA"), dtype=np.float64)
        except Exception:  # noqa: BLE001  (an unreadable texture is reported by the caller, not fatal)
            return None

    def texture_rgb(self) -> np.ndarray | None:
        """The first material's base-colour texture as float RGB [H, W, 3] in 0-255 (None when there is none)."""
        for mat in self.materials:
            if mat.image is not None:
                img = self.image(mat.image)
                if img is not None:
                    return img[..., :3]
        return None


def _chunks(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 20:
        raise GlbError("file too short to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GlbError("not a binary glTF file (bad magic); export as .glb")
    if version != 2:
        raise GlbError(f"glTF version {version} is not supported (2 only)")
    doc, binary, off = None, b"", 12
    while off + 8 <= min(length, len(data)):
        size, kind = struct.unpack_from("<II", data, off)
        body = data[off + 8 : off + 8 + size]
        if kind == CHUNK_JSON:
            doc = json.loads(body.decode("utf-8"))
        elif kind == CHUNK_BIN and not binary:
            binary = body
        off += 8 + size
    if doc is None:
        raise GlbError("GLB has no JSON chunk")
    return doc, binary


def _view_bytes(doc: dict, binary: bytes, index: int) -> tuple[bytes, int | None]:
    view = doc["bufferViews"][index]
    if view.get("buffer", 0) != 0 or doc["buffers"][view.get("buffer", 0)].get("uri"):
        raise GlbError("buffers outside the GLB's own binary chunk are not supported")
    start = view.get("byteOffset", 0)
    return binary[start : start + view["byteLength"]], view.get("byteStride")


def _accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    if "sparse" in acc:
        raise GlbError("sparse accessors are not supported")
    if "bufferView" not in acc:
        raise GlbError("accessor without a buffer view")
    dtype, norm = _COMPONENTS[acc["componentType"]]
    width = _WIDTH[acc["type"]]
    count = acc["count"]
    raw, stride = _view_bytes(doc, binary, acc["bufferView"])
    item = np.dtype(dtype).itemsize * width
    stride = stride or item
    start = acc.get("byteOffset", 0)
    if start + stride * (count - 1) + item > len(raw):
        raise GlbError(f"accessor {index} runs past its buffer view")
    if stride == item:
        out = np.frombuffer(raw, dtype=dtype, count=count * width, offset=start).reshape(count, width)
    else:
        flat = np.frombuffer(raw, dtype=np.uint8)[start : start + stride * count]
        flat = np.pad(flat, (0, stride * count - len(flat)))  # the last element needs only `item` bytes
        out = np.ascontiguousarray(flat.reshape(count, stride)[:, :item]).view(dtype).reshape(count, width)
    out = out.astype(np.float64) if dtype == np.float32 else out
    if acc.get("normalized") and norm is not None:
        out = np.maximum(out.astype(np.float64) / norm, -1.0)
    return out


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF stores column-major
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]))
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]))
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


def _primitive(doc: dict, binary: bytes, prim: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = _accessor(doc, binary, prim["attributes"]["POSITION"])[:, :3].astype(np.float64)
    if "indices" in prim:
        idx = _accessor(doc, binary, prim["indices"]).astype(np.int64).ravel()
    else:
        idx = np.arange(len(pos), dtype=np.int64)
    if len(idx) % 3:
        raise GlbError("triangle primitive whose index count is not a multiple of 3")
    if len(idx) and (idx.min() < 0 or idx.max() >= len(pos)):
        raise GlbError("triangle index out of range")
    return pos, idx.reshape(-1, 3)


def _float_attr(doc: dict, binary: bytes, prim: dict, name: str) -> np.ndarray | None:
    if name not in prim["attributes"]:
        return None
    a = _accessor(doc, binary, prim["attributes"][name])
    if a.dtype.kind in "iu":   # a non-normalised integer attribute: read as stored
        a = a.astype(np.float64)
    return a


def _materials(doc: dict, notes: list[str]) -> list[Material]:
    out = []
    textures = doc.get("textures", [])
    for i, mat in enumerate(doc.get("materials", [])):
        pbr = mat.get("pbrMetallicRoughness", {})
        factor = tuple(float(x) for x in pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]))
        image, texcoord = None, 0
        ref = pbr.get("baseColorTexture")
        if ref is not None:
            texcoord = int(ref.get("texCoord", 0))
            if "KHR_texture_transform" in ref.get("extensions", {}):
                notes.append(f"material {i}: KHR_texture_transform is not applied")
            ti = ref.get("index", -1)
            tex = textures[ti] if 0 <= ti < len(textures) else {}
            src = tex.get("source")
            if src is None:   # e.g. EXT_texture_webp keeps its source under the extension
                src = next((e.get("source") for e in tex.get("extensions", {}).values()
                            if isinstance(e, dict) and "source" in e), None)
            if src is None:
                notes.append(f"material {i}: base-colour texture has no image source this reader can use")
            image = src
        elif "KHR_materials_pbrSpecularGlossiness" in mat.get("extensions", {}):
            notes.append(f"material {i}: KHR_materials_pbrSpecularGlossiness is not read; base colour only")
        mode = str(mat.get("alphaMode", "OPAQUE")).upper()
        if mode not in ("OPAQUE", "MASK", "BLEND"):
            notes.append(f"material {i}: alphaMode {mode!r} is not glTF; read as OPAQUE")
            mode = "OPAQUE"
        out.append(Material(factor[:4] + (1.0,) * (4 - len(factor[:4])), image, texcoord, mode,
                            float(mat.get("alphaCutoff", 0.5))))
    return out


def _images(doc: dict, binary: bytes, notes: list[str]) -> tuple[list[bytes | None], list[tuple[int, int]]]:
    from PIL import Image

    raws, sizes = [], []
    for i, img in enumerate(doc.get("images", [])):
        if "bufferView" not in img:
            notes.append(f"image {i} is not embedded; its size is unknown")
            raws.append(None)
            continue
        raw, _ = _view_bytes(doc, binary, img["bufferView"])
        try:
            with Image.open(io.BytesIO(raw)) as im:
                sizes.append(im.size)
            raws.append(bytes(raw))
        except Exception:  # noqa: BLE001  (an unreadable texture is reported, not fatal)
            notes.append(f"image {i} could not be decoded")
            raws.append(None)
    return raws, sizes


def load_glb(path: Path | str) -> Glb:
    """Every triangle mesh instance in the GLB's default scene, in the contract frame."""
    data = Path(path).read_bytes()
    doc, binary = _chunks(data)
    blocked = _UNSUPPORTED & set(doc.get("extensionsRequired", []) + doc.get("extensionsUsed", []))
    if blocked:
        raise GlbError(f"compressed meshes are not supported ({', '.join(sorted(blocked))}); "
                       "export an uncompressed GLB")
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    roots = scenes[doc.get("scene", 0)]["nodes"] if scenes else list(range(len(nodes)))
    notes: list[str] = []
    materials = _materials(doc, notes)
    meshes: list[Mesh] = []
    stack = [(i, np.eye(4)) for i in reversed(roots)]
    seen = 0
    while stack:
        i, parent = stack.pop()
        seen += 1
        if seen > 100_000:
            raise GlbError("node hierarchy has a cycle")
        node = nodes[i]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            mdoc = doc["meshes"][node["mesh"]]
            name = node.get("name") or mdoc.get("name") or f"mesh{node['mesh']}"
            if "skin" in node:
                notes.append(f"{name}: skinned mesh read in its bind pose (joint transforms not applied)")
            parts_v, parts_f, parts_uv, parts_c, parts_m, base = [], [], [], [], [], 0
            for prim in mdoc.get("primitives", []):
                if prim.get("mode", TRIANGLES) != TRIANGLES:
                    notes.append(f"{name}: skipped a non-triangle primitive (mode {prim.get('mode')})")
                    continue
                if prim.get("targets"):
                    notes.append(f"{name}: morph targets not applied")
                pos, faces = _primitive(doc, binary, prim)
                pos = pos @ world[:3, :3].T + world[:3, 3]
                mi = prim.get("material", -1)
                mi = mi if 0 <= mi < len(materials) else -1
                tc = materials[mi].texcoord if mi >= 0 else 0
                uv = _float_attr(doc, binary, prim, f"TEXCOORD_{tc}")
                col = _float_attr(doc, binary, prim, "COLOR_0")
                if col is not None and col.shape[1] == 3:
                    col = np.concatenate([col, np.ones((len(col), 1))], axis=1)
                parts_v.append(pos @ GLTF_TO_CONTRACT.T)
                parts_f.append(faces + base)
                parts_uv.append(uv[:, :2] if uv is not None else None)
                parts_c.append(col)
                parts_m.append(np.full(len(faces), mi, dtype=np.int64))
                base += len(pos)
            if parts_v:
                faces = np.concatenate(parts_f)
                if np.linalg.det(world[:3, :3]) < 0:
                    faces = faces[:, ::-1]  # a mirroring transform flips the winding
                n = [len(v) for v in parts_v]
                uv = (np.concatenate([u if u is not None else np.zeros((k, 2)) for u, k in zip(parts_uv, n)])
                      if any(u is not None for u in parts_uv) else None)
                col = (np.concatenate([c if c is not None else np.ones((k, 4)) for c, k in zip(parts_c, n)])
                       if any(c is not None for c in parts_c) else None)
                meshes.append(Mesh(name, np.concatenate(parts_v), faces, uv, col, np.concatenate(parts_m)))
        for c in reversed(node.get("children", [])):
            stack.append((c, world))
    raws, sizes = _images(doc, binary, notes)
    return Glb(meshes, sizes, notes, materials, raws)


def write_glb(path: Path | str, meshes: list[tuple], texture_png: bytes | None = None,
              alpha_mode: str | None = None, alpha_cutoff: float | None = None) -> Path:
    """A minimal GLB of triangle meshes given in the contract frame (for tests and fixtures).

    Each mesh is `(name, verts, faces)` or `(name, verts, faces, uv)`; with `texture_png`, every mesh that has
    texture coordinates uses that one embedded image as its base colour, in a material with `alpha_mode` and
    `alpha_cutoff` when they are given."""
    binary = bytearray()
    doc: dict = {"asset": {"version": "2.0", "generator": "herogen.blender.glb"},
                 "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
                 "accessors": [], "bufferViews": [], "buffers": []}

    def add(arr: np.ndarray, target: int | None, kind: str, comp: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": arr.nbytes}
        if target is not None:
            view["target"] = target
        doc["bufferViews"].append(view)
        binary.extend(arr.tobytes())
        a = {"bufferView": len(doc["bufferViews"]) - 1, "componentType": comp, "count": int(len(arr)),
             "type": kind}
        if kind == "VEC3":
            a["min"], a["max"] = arr.min(axis=0).tolist(), arr.max(axis=0).tolist()
        doc["accessors"].append(a)
        return len(doc["accessors"]) - 1

    material = None
    if texture_png is not None:
        while len(binary) % 4:
            binary.append(0)
        doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(texture_png)})
        binary.extend(texture_png)
        doc["images"] = [{"bufferView": len(doc["bufferViews"]) - 1, "mimeType": "image/png"}]
        doc["textures"] = [{"source": 0}]
        doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        if alpha_mode is not None:
            doc["materials"][0]["alphaMode"] = alpha_mode
        if alpha_cutoff is not None:
            doc["materials"][0]["alphaCutoff"] = alpha_cutoff
        material = 0
    for item in meshes:
        name, verts, faces = item[:3]
        uv = item[3] if len(item) > 3 else None
        pos = (np.asarray(verts, dtype=np.float64) @ GLTF_TO_CONTRACT).astype(np.float32)  # inverse = transpose
        idx = np.asarray(faces, dtype=np.uint32).ravel()
        attrs = {"POSITION": add(pos, 34962, "VEC3", 5126)}
        if uv is not None:
            attrs["TEXCOORD_0"] = add(np.asarray(uv, dtype=np.float32), 34962, "VEC2", 5126)
        prim = {"attributes": attrs, "indices": add(idx, 34963, "SCALAR", 5125), "mode": TRIANGLES}
        if uv is not None and material is not None:
            prim["material"] = material
        doc["meshes"].append({"name": name, "primitives": [prim]})
        doc["nodes"].append({"name": name, "mesh": len(doc["meshes"]) - 1})
        doc["scenes"][0]["nodes"].append(len(doc["nodes"]) - 1)
    while len(binary) % 4:
        binary.append(0)
    doc["buffers"].append({"byteLength": len(binary)})
    js = json.dumps(doc).encode("utf-8")
    js += b" " * (-len(js) % 4)
    out = struct.pack("<III", GLB_MAGIC, 2, 12 + 8 + len(js) + 8 + len(binary))
    out += struct.pack("<II", len(js), CHUNK_JSON) + js + struct.pack("<II", len(binary), CHUNK_BIN) + bytes(binary)
    p = Path(path)
    p.write_bytes(out)
    return p
