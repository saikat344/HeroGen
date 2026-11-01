"""The 3D viewer's material routing, executed as production code on the CPU.

`herogen/lab/static/viewer.js` runs in a `node` vm under a stub three.js (no WebGL, no DOM, no network), with
its `import` lines stripped the way the page's bundler-free module loading leaves it. A best-profile run's
refined or SAM-source body is one mesh named for the whole body, under no R15 section node, and it is still
the body on screen: every view control — Lit and Art, clay, wireframe, per-section colours, isolation — and
the triangle count must reach it, exactly as they reach each section of a sectioned body.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "herogen" / "lab" / "static" / "viewer.js"
HAVE_NODE = shutil.which("node") is not None

# The whole-body bucket: meshes under no R15 section node, named as viewer.js names it.
WHOLE = "body"

HARNESS = r"""
// CPU-only: runs the production viewer.js under a stub three.js. No WebGL, no DOM, no network.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = process.argv[2];
const mode = process.argv[3];
const source = fs.readFileSync(path.join(root, 'herogen/lab/static/viewer.js'), 'utf8');
const poses = fs.readFileSync(path.join(root, 'herogen/lab/static/poses.js'), 'utf8');
const sections = vm.runInNewContext(poses.match(/export const SECTIONS = (\[[\s\S]*?\]);/)[1]);
class Vector3 { constructor(x = 0, y = 0, z = 0) { Object.assign(this, { x, y, z }); } }
class Box3 { setFromObject() { return this; } }
class Color { constructor(v) { this.value = v; } clone() { return new Color(this.value); } }
class MeshBasicMaterial { constructor(o) { Object.assign(this, { type: 'MeshBasicMaterial' }, o); } }
class RoomEnvironment { constructor() {} }
class PMREMGenerator { fromScene() { return { texture: {} }; } }
class DirectionalLight { constructor() {} }
const THREE = { Vector3, Box3, Color, MeshBasicMaterial, PMREMGenerator, DirectionalLight, DoubleSide: 2,
  SRGBColorSpace: 'srgb', ACESFilmicToneMapping: 4, NoToneMapping: 0,
  MathUtils: { clamp: (v, a, b) => Math.max(a, Math.min(b, v)) } };
const scope = { THREE, RoomEnvironment, GLTFLoader: class {}, SECTIONS: sections, R15_PARENT: {},
  boneDirs: () => ({}), poseLibrary: () => ({}), console };
vm.createContext(scope);
// WHOLE is a late export: an older viewer has no such bucket, so it is read through a typeof guard.
const EXPORTS = vm.runInContext(source.replace(/^import .*;\s*$/gm, '').replace(/\bexport /g, '')
  + '\n({ Viewer, instance, artMaterial, WHOLE: typeof WHOLE === "undefined" ? null : WHOLE });', scope);
const { Viewer, instance, artMaterial } = EXPORTS;
// The whole-body bucket's name, as the current viewer names it (an older viewer has none: no such bucket).
const WHOLE = EXPORTS.WHOLE ?? 'body';
const CLAY = { clay: true }; const WIRE = { wire: true }; const GHOST = { ghost: true };
const SEC = { Head: { section: 'Head' }, [WHOLE]: { section: WHOLE } };
const material = () => ({ type: 'MeshStandardMaterial', color: new Color(0x6699aa), map: { isTexture: true } });
// A Viewer with everything but the renderer and DOM: the state applyShading reads and writes.
const viewer = (inst) => Object.assign(Object.create(Viewer.prototype), { inst, shading: 'textured',
  lighting: 'lit', hot: null, renderer: { toneMapping: 4, toneMappingExposure: 1 },
  mat: { clay: CLAY, wire: WIRE, ghost: GHOST }, secMat: SEC });
const sectionNode = (name, mesh) => ({ name, traverse(fn) { fn(mesh); }, matrix: { clone: () => ({}) },
  getWorldPosition: () => new Vector3() });
const out = {};
if (mode === 'whole-body') {
  // The best profile's refined body: one mesh named for the whole body, no R15 section nodes anywhere.
  const original = material();
  const mesh = { name: 'body', isMesh: true, geometry: { index: { count: 6 } }, material: original, userData: {} };
  const scene = { clone() { return this; }, updateMatrixWorld() {}, getObjectByName: () => null,
    traverse(fn) { fn(mesh); } };
  const inst = instance({ scene });
  const v = viewer(inst);
  out.registered = { sectionCount: inst.sectionCount, buckets: Object.keys(inst.meshes),
    meshInWhole: inst.meshes[WHOLE] && inst.meshes[WHOLE][0] === mesh, sectionTag: mesh.userData.section,
    total: inst.total };
  v.setLighting('art');
  const art = mesh.material;
  out.art = { type: art.type, mapKept: art.map === original.map, colourCloned: art.color !== original.color,
    toneMapping: v.renderer.toneMapping, originalReplaced: art !== original };
  v.setLighting('lit');
  out.lit = { originalRestored: mesh.material === original, toneMapping: v.renderer.toneMapping };
  v.setLighting('art');
  out.artStandInReused = mesh.material === art;
  v.setShading('clay');
  out.clay = mesh.material === CLAY;
  v.setShading('wire');
  out.wire = { material: mesh.material === WIRE, toneMapping: v.renderer.toneMapping };
  v.setShading('sections');
  out.sectionsView = mesh.material === SEC[WHOLE];
  v.setShading('textured');
  out.texturedAgain = mesh.material === art;
  v.setHot(['Head']);
  out.ghostedWhenNotHot = mesh.material === GHOST;
  v.setHot(null);
  out.restoredWhenHotCleared = mesh.material === art;
} else if (mode === 'sectioned') {
  // A sectioned body: the positive control, one mesh under its R15 section node.
  const original = material();
  const mesh = { isMesh: true, geometry: { index: { count: 3 } }, material: original, userData: {} };
  const head = sectionNode('Head', mesh);
  const scene = { clone() { return this; }, updateMatrixWorld() {},
    getObjectByName(n) { return n === 'Head' ? head : null; }, traverse(fn) { fn(head); fn(mesh); } };
  const inst = instance({ scene });
  const v = viewer(inst);
  out.registered = { sectionCount: inst.sectionCount, buckets: Object.keys(inst.meshes),
    meshInSection: inst.meshes.Head[0] === mesh, sectionTag: mesh.userData.section, total: inst.total };
  v.setLighting('art');
  const art = mesh.material;
  out.art = { type: art.type, mapKept: art.map === original.map, toneMapping: v.renderer.toneMapping };
  out.artSharedPerOriginal = artMaterial([original, original]).every((m) => m === art);
  v.setLighting('lit');
  out.lit = { originalRestored: mesh.material === original, toneMapping: v.renderer.toneMapping };
  v.setLighting('art');
  out.artAgain = mesh.material === art;
  v.setShading('clay');
  out.clay = { material: mesh.material === CLAY, toneMapping: v.renderer.toneMapping };
  v.setShading('wire');
  out.wire = mesh.material === WIRE;
  v.setShading('sections');
  out.sectionsView = mesh.material === SEC.Head;
  v.setShading('textured');
  v.setHot(['Other']);
  out.ghostedWhenNotHot = mesh.material === GHOST;
  v.setHot(null);
  out.restoredWhenHotCleared = mesh.material === art;
  out.exposure = {};
  for (const [input, expected] of [[0.5, 0.5], [2, 2], [-1, 0.5], [3, 2], [NaN, 1]]) {
    v.setExposure(input);
    out.exposure[String(input)] = v.renderer.toneMappingExposure === expected;
  }
} else if (mode === 'mixed') {
  // A section node nested in another, plus a mesh under no section: each mesh registered exactly once.
  const inner = { isMesh: true, geometry: { index: { count: 9 } }, material: material(), userData: {} };
  const stray = { isMesh: true, geometry: { attributes: { position: { count: 12 } } }, material: material(),
    userData: {} };
  const head = sectionNode('Head', inner);
  const torso = { name: 'UpperTorso', traverse(fn) { fn(head); fn(inner); }, matrix: { clone: () => ({}) },
    getWorldPosition: () => new Vector3() };
  const scene = { clone() { return this; }, updateMatrixWorld() {},
    getObjectByName(n) { return n === 'Head' ? head : n === 'UpperTorso' ? torso : null; },
    traverse(fn) { fn(torso); fn(head); fn(inner); fn(stray); } };
  const inst = instance({ scene });
  out.sectionCount = inst.sectionCount;
  out.buckets = Object.keys(inst.meshes);
  out.innerInSectionOnly = inst.meshes.Head[0] === inner && !(inst.meshes.UpperTorso || []).includes(inner);
  out.strayInWhole = (inst.meshes[WHOLE] || [])[0] === stray;
  out.sectionTags = { inner: inner.userData.section, stray: stray.userData.section };
  out.tris = { Head: inst.tris.Head, UpperTorso: inst.tris.UpperTorso, whole: inst.tris[WHOLE] ?? null,
    total: inst.total };
} else throw Error('expected whole-body, sectioned, or mixed');
console.log(JSON.stringify(out));
"""


@unittest.skipUnless(HAVE_NODE, "node is required to execute the viewer's JavaScript")
class ViewerRoutingTest(unittest.TestCase):
    """instance() registers every mesh of the loaded body, and the view controls reach each one."""

    def probe(self, mode: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".cjs", delete=False, encoding="utf-8") as f:
            f.write(HARNESS)
            script = f.name
        self.addCleanup(os.unlink, script)
        run = subprocess.run(["node", script, str(ROOT), mode], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, f"{mode}: {run.stderr}")
        return json.loads(run.stdout)

    def test_whole_body_mesh_answers_the_view_controls(self):
        """The Art view of a best-profile refined body swaps its one mesh to the unlit base colour.

        The mesh is under no R15 section node, so it registers under the whole-body bucket with no section
        tag; Lit restores the original material and tone mapping, and clay, wireframe, per-section colours,
        isolation and the triangle count all reach it too."""
        out = self.probe("whole-body")
        self.assertEqual(out["registered"], {"sectionCount": 0, "buckets": [WHOLE], "meshInWhole": True,
                                            "sectionTag": None, "total": 2})
        self.assertEqual(out["art"], {"type": "MeshBasicMaterial", "mapKept": True, "colourCloned": True,
                                      "toneMapping": 0, "originalReplaced": True})
        self.assertEqual(out["lit"], {"originalRestored": True, "toneMapping": 4})
        self.assertTrue(out["artStandInReused"])
        self.assertTrue(out["clay"])
        self.assertEqual(out["wire"], {"material": True, "toneMapping": 0})
        self.assertTrue(out["sectionsView"])
        self.assertTrue(out["texturedAgain"])
        self.assertTrue(out["ghostedWhenNotHot"])
        self.assertTrue(out["restoredWhenHotCleared"])

    def test_sectioned_body_material_switching(self):
        """A sectioned body keeps its switching: Art and back, shared stand-ins, shading, isolation, exposure."""
        out = self.probe("sectioned")
        self.assertEqual(out["registered"], {"sectionCount": 1, "buckets": ["Head"], "meshInSection": True,
                                            "sectionTag": "Head", "total": 1})
        self.assertEqual(out["art"], {"type": "MeshBasicMaterial", "mapKept": True, "toneMapping": 0})
        self.assertTrue(out["artSharedPerOriginal"])
        self.assertEqual(out["lit"], {"originalRestored": True, "toneMapping": 4})
        self.assertTrue(out["artAgain"])
        self.assertEqual(out["clay"], {"material": True, "toneMapping": 4})
        self.assertTrue(out["wire"])
        self.assertTrue(out["sectionsView"])
        self.assertTrue(out["ghostedWhenNotHot"])
        self.assertTrue(out["restoredWhenHotCleared"])
        self.assertEqual(out["exposure"], {"0.5": True, "2": True, "-1": True, "3": True, "NaN": True})

    def test_each_mesh_is_registered_exactly_once(self):
        """A mesh nested under two section nodes belongs to its own section; an unsectioned mesh to the whole."""
        out = self.probe("mixed")
        self.assertEqual(out["sectionCount"], 2)
        self.assertEqual(out["buckets"], ["Head", "UpperTorso", WHOLE])
        self.assertTrue(out["innerInSectionOnly"])
        self.assertTrue(out["strayInWhole"])
        self.assertEqual(out["sectionTags"], {"inner": "Head", "stray": None})
        self.assertEqual(out["tris"], {"Head": 3, "UpperTorso": 0, "whole": 4, "total": 7})


class ViewerSourceTest(unittest.TestCase):
    """What the CPU probes cannot build: the Viewer's own materials (a renderer needs WebGL)."""

    def test_whole_bucket_has_a_section_material(self):
        source = VIEWER.read_text()
        self.assertIn("this.secMat[WHOLE] = new THREE.MeshStandardMaterial", source)
        self.assertIn("export const WHOLE = 'body';", source)


if __name__ == "__main__":
    unittest.main()
