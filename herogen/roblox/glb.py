"""A small glTF 2.0 binary (`.glb`) reader: the JSON document, and each mesh node's vertex positions and
triangle count. Only what the export needs; it refuses layouts it would read wrongly instead of guessing."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import RobloxExportError

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT = 5126
TRIANGLES = 4
IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)
IDENTITY_SCALE = (1.0, 1.0, 1.0)
IDENTITY_MATRIX = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def read_chunks(path: Path) -> tuple[dict, bytes | None]:
    """The JSON document and the BIN chunk (None when absent)."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise RobloxExportError(f"cannot read {path}: {exc.strerror or exc}") from None
    if len(data) < 20:
        raise RobloxExportError(f"{path}: too short for a glTF binary file")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise RobloxExportError(f"{path}: not a glTF 2.0 binary file")
    doc, binchunk, off = None, None, 12
    end = min(length, len(data))
    while off + 8 <= end:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if len(chunk) != clen:
            raise RobloxExportError(f"{path}: truncated chunk at byte {off}")
        if ctype == CHUNK_JSON and doc is None:
            try:
                doc = json.loads(chunk)
            except ValueError:
                raise RobloxExportError(f"{path}: the JSON chunk is not valid JSON") from None
        elif ctype == CHUNK_BIN and binchunk is None:
            binchunk = chunk
        off += 8 + clen
    if not isinstance(doc, dict):
        raise RobloxExportError(f"{path}: no JSON chunk")
    return doc, binchunk


def _close(values, want) -> bool:
    return len(values) == len(want) and all(abs(float(a) - b) <= 1e-9 for a, b in zip(values, want))


def mesh_nodes(path: Path) -> dict[str, dict]:
    """{node name: {"translation": [3], "positions": (n, 3) float64 in G with the translation applied,
    "triangles": int}} for every node that holds a mesh.

    Refused: a mesh node that is rotated, scaled, given a matrix, has children or sits under another node (the
    body is a flat scene of translated sections); non-triangle primitives; positions that are not float VEC3;
    sparse accessors; two mesh nodes with one name.
    """
    doc, binchunk = read_chunks(path)
    if binchunk is None:
        raise RobloxExportError(f"{path}: no BIN chunk (external buffers are not supported)")
    try:
        nodes = doc["nodes"]
        accessors = doc["accessors"]
        views = doc["bufferViews"]
        meshes = doc["meshes"]
    except KeyError as exc:
        raise RobloxExportError(f"{path}: glTF document has no {exc.args[0]!r}") from None
    def positions(idx: int) -> np.ndarray:
        acc = accessors[idx]
        if acc.get("componentType") != FLOAT or acc.get("type") != "VEC3" or "sparse" in acc:
            raise RobloxExportError(f"{path}: accessor {idx} is not a dense float VEC3")
        if "bufferView" not in acc:
            raise RobloxExportError(f"{path}: accessor {idx} has no buffer view")
        bv = views[acc["bufferView"]]
        if bv.get("buffer", 0) != 0:
            raise RobloxExportError(f"{path}: accessor {idx} reads an external buffer")
        base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride", 12)
        count = acc["count"]
        if count < 1 or base + (count - 1) * stride + 12 > len(binchunk):
            raise RobloxExportError(f"{path}: accessor {idx} runs past the BIN chunk")
        arr = np.ndarray((count, 3), dtype="<f4", buffer=binchunk, offset=base, strides=(stride, 4))
        return arr.astype(np.float64)

    try:
        return _read_nodes(path, nodes, meshes, accessors, positions)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RobloxExportError(f"{path}: malformed glTF document ({exc.__class__.__name__}: {exc})") \
            from None


def _read_nodes(path, nodes, meshes, accessors, positions) -> dict[str, dict]:
    child_of = {c for n in nodes for c in n.get("children", ())}
    out: dict[str, dict] = {}
    for i, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        name = node.get("name", f"node{i}")
        if "rotation" in node and not _close(node["rotation"], IDENTITY_ROTATION):
            raise RobloxExportError(f"{path}: node {name} is rotated; only translated sections are supported")
        if "scale" in node and not _close(node["scale"], IDENTITY_SCALE):
            raise RobloxExportError(f"{path}: node {name} is scaled; only translated sections are supported")
        if "matrix" in node and not _close(node["matrix"], IDENTITY_MATRIX):
            raise RobloxExportError(f"{path}: node {name} has a matrix; only translated sections are "
                                    "supported")
        if node.get("children"):
            raise RobloxExportError(f"{path}: node {name} has children; a flat scene of sections is expected")
        if i in child_of:
            raise RobloxExportError(f"{path}: node {name} sits under another node; a flat scene is expected")
        if name in out:
            raise RobloxExportError(f"{path}: two mesh nodes are named {name}")
        t = np.array(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
        pts, tris = [], 0
        for prim in meshes[node["mesh"]]["primitives"]:
            if prim.get("mode", TRIANGLES) != TRIANGLES:
                raise RobloxExportError(f"{path}: node {name} has a primitive that is not triangles")
            p = positions(prim["attributes"]["POSITION"])
            pts.append(p + t)
            n = accessors[prim["indices"]]["count"] if "indices" in prim else len(p)
            tris += n // 3
        if not pts:
            raise RobloxExportError(f"{path}: node {name} has no primitives")
        out[name] = {"translation": t.tolist(), "positions": np.concatenate(pts), "triangles": int(tris)}
    return out
