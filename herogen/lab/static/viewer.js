// The lab's 3D views of a body: the main viewer, the two-up compare view and offscreen snapshots. Each GLB is
// loaded once per URL and shown as an instance whose sections turn rigidly about their pivots.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { cssToken, reducedMotion } from './dom.js';
import { R15_PARENT, SECTIONS, boneDirs, poseLibrary, worldMatrices } from './poses.js';

const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
const loader = new GLTFLoader();
const cache = new Map(); // url -> Promise<gltf>
const CACHE_MAX = 8;

function disposeGltf(gltf) {
  const seen = new Set();
  gltf.scene.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    for (const m of [].concat(o.material || [])) {
      if (seen.has(m)) continue;
      seen.add(m);
      for (const v of Object.values(m)) if (v && v.isTexture) v.dispose();
      m.dispose();
    }
  });
}

// One parsed GLB per URL; the oldest is freed when more than CACHE_MAX are held (instances still on screen keep
// their own geometry references until the page replaces them).
export function loadGlb(url) {
  if (cache.has(url)) {
    const p = cache.get(url);
    cache.delete(url);
    cache.set(url, p);
    return p;
  }
  const p = loader.loadAsync(url).then((g) => {
    g.scene.traverse((o) => { if (o.isMesh) { o.userData.orig = o.material; o.material.side = THREE.DoubleSide; } });
    return g;
  });
  p.catch(() => cache.delete(url));
  cache.set(url, p);
  while (cache.size > CACHE_MAX) {
    const [oldUrl, old] = cache.entries().next().value;
    cache.delete(oldUrl);
    old.then(disposeGltf, () => {});
  }
  return p;
}

// An instance of a loaded body: its section nodes, meshes, rest transforms, pivots and triangle counts.
export function instance(gltf, info = {}) {
  const root = gltf.scene.clone(true);
  root.updateMatrixWorld(true);
  const nodes = {}; const meshes = {}; const rest = {}; const pivots = {}; const tris = {};
  for (const n of SECTIONS) {
    const node = root.getObjectByName(n);
    if (!node) continue;
    nodes[n] = node;
    rest[n] = node.matrix.clone();
    pivots[n] = node.getWorldPosition(new THREE.Vector3());
    meshes[n] = [];
    tris[n] = 0;
    node.traverse((o) => {
      if (!o.isMesh) return;
      o.userData.section = n;
      o.userData.orig = o.material;
      meshes[n].push(o);
      const g = o.geometry;
      tris[n] += (g.index ? g.index.count : g.attributes.position.count) / 3;
    });
  }
  const parent = {};
  for (const n of Object.keys(nodes)) parent[n] = (info.parents && info.parents[n] !== undefined ? info.parents[n] : R15_PARENT[n]) || null;
  const box = new THREE.Box3().setFromObject(root);
  const inst = { root, nodes, meshes, rest, pivots, tris, parent, box,
    total: Object.values(tris).reduce((a, b) => a + b, 0), sectionCount: Object.keys(nodes).length };
  inst.poses = poseLibrary(boneDirs(info.joints, info.ends, pivots));
  return inst;
}

// Section transforms for `pose` on an instance: {name: {pos, q, scale}} plus the world matrices.
export function poseTargets(inst, pose) {
  const W = worldMatrices(inst.poses[pose] || {}, inst.pivots, inst.parent);
  const out = {};
  for (const [n, node] of Object.entries(inst.nodes)) {
    const m = W[n].clone().multiply(inst.rest[n]);
    const pos = new THREE.Vector3(); const q = new THREE.Quaternion(); const sc = new THREE.Vector3();
    m.decompose(pos, q, sc);
    out[n] = { pos, q, sc };
  }
  return { out, W };
}

export function applyPoseNow(inst, pose) {
  const { out } = poseTargets(inst, pose);
  for (const [n, node] of Object.entries(inst.nodes)) {
    node.position.copy(out[n].pos); node.quaternion.copy(out[n].q); node.scale.copy(out[n].sc);
  }
  inst.root.updateMatrixWorld(true);
}

export function lights(scene) {
  scene.add(new THREE.HemisphereLight(0xffffff, 0x4a4f58, 1.5));
  const d1 = new THREE.DirectionalLight(0xffffff, 1.7); d1.position.set(3, 8, 6); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.6); d2.position.set(-5, 3, -4); scene.add(d2);
}

// Camera presets: the direction from the target to the camera. Front is +Z; the character's left is +X.
export const VIEW_DIRS = {
  front: V3(0, 0.06, 1), q: V3(Math.sin(0.6), 0.22, Math.cos(0.6)), side: V3(1, 0.06, 0), back: V3(0, 0.06, -1),
};

function fitDistance(inst, fov, aspect) {
  const size = inst ? inst.box.getSize(new THREE.Vector3()) : V3(3, 5.5, 1);
  const half = Math.max(size.y / 2, size.x / 2 / Math.max(aspect, 0.2)) * 1.18;
  return half / Math.tan(THREE.MathUtils.degToRad(fov / 2));
}

function centreOf(inst) {
  return inst ? inst.box.getCenter(new THREE.Vector3()) : V3(0, 2.75, 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Main viewer

export class Viewer {
  constructor({ canvas, gizmo, sectionColours }) {
    this.canvas = canvas;
    this.gizmo = gizmo.getContext('2d');
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    lights(this.scene);
    this.persp = new THREE.PerspectiveCamera(30, 1, 0.05, 200);
    this.ortho = new THREE.OrthographicCamera(-3, 3, 3, -3, 0.05, 200);
    this.cam = this.persp;
    this.controls = new OrbitControls(this.persp, canvas);
    this.controls.enableDamping = !reducedMotion();
    this.controls.dampingFactor = 0.12;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 60;
    this.controls.autoRotateSpeed = 1.2;
    this.mat = {
      clay: new THREE.MeshStandardMaterial({ color: 0xb9b4ab, roughness: 0.85, metalness: 0, side: THREE.DoubleSide }),
      wire: new THREE.MeshBasicMaterial({ color: 0xa3b4ff, wireframe: true }),
      ghost: new THREE.MeshBasicMaterial({ color: 0x8090a0, transparent: true, opacity: 0.07, depthWrite: false }),
    };
    this.secMat = Object.fromEntries(SECTIONS.map((n) => [n, new THREE.MeshStandardMaterial({
      color: new THREE.Color(sectionColours[n] || '#888888'), roughness: 0.7, side: THREE.DoubleSide })]));
    this.inst = null;
    this.shading = 'textured';
    this.pose = 'rest';
    this.hot = null; // null: every section; else the set of sections drawn normally
    this.anim = null;
    this.camAnim = null;
    this.view = 'q';
    this.mode = '3d';
    this.onPick = () => {};
    this.dirty = true; // draw only when something changed: the camera, the body, its pose or its shading
    this.controls.addEventListener('change', () => { this.dirty = true; });
    this.theme();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.bindPicking();
    this.bindKeys();
    requestAnimationFrame((t) => this.loop(t));
  }

  theme() {
    this.scene.background = new THREE.Color(cssToken('--viewer-bg') || '#11161b');
    if (this.grid) { this.scene.remove(this.grid); this.grid.geometry.dispose(); this.grid.material.dispose(); }
    // 6 studs square, 12 cells: one cell is half a stud.
    this.grid = new THREE.GridHelper(6, 12, new THREE.Color(cssToken('--grid-a') || '#2c3640'),
      new THREE.Color(cssToken('--grid-b') || '#1f262d'));
    this.scene.add(this.grid);
    this.mat.wire.color.set(cssToken('--wire') || '#a3b4ff');
    this.mat.clay.color.set(cssToken('--clay') || '#b9b4ab');
    this.dirty = true;
  }

  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    this.renderer.setSize(r.width, r.height, false);
    this.persp.aspect = r.width / r.height;
    this.persp.updateProjectionMatrix();
    if (this.mode === 'side' && this.frame) this.applyFrame(this.frame);
    this.dirty = true;
  }

  show(inst) {
    if (this.inst) this.scene.remove(this.inst.root);
    this.inst = inst;
    this.hot = null;
    if (inst) {
      this.scene.add(inst.root);
      this.applyShading();
      this.setPose(this.pose, true);
    }
    this.dirty = true;
  }

  clear() { this.show(null); }

  applyShading() {
    if (!this.inst) return;
    for (const [n, list] of Object.entries(this.inst.meshes)) {
      for (const m of list) {
        if (this.hot && !this.hot.has(n)) { m.material = this.mat.ghost; continue; }
        m.material = this.shading === 'textured' ? m.userData.orig : this.shading === 'clay' ? this.mat.clay
          : this.shading === 'wire' ? this.mat.wire : this.secMat[n];
      }
    }
    this.dirty = true;
  }

  setShading(s) { this.shading = s; this.applyShading(); }
  setHot(names) { this.hot = names && names.length ? new Set(names) : null; this.applyShading(); }

  setPose(pose, instant = false) {
    this.pose = pose;
    if (!this.inst) return;
    const { out } = poseTargets(this.inst, pose);
    const from = {};
    for (const [n, o] of Object.entries(this.inst.nodes)) from[n] = { pos: o.position.clone(), q: o.quaternion.clone() };
    this.anim = { t0: performance.now(), dur: instant || reducedMotion() ? 0 : 450, from, to: out };
    if (instant) this.stepAnim(performance.now());
  }

  // Where a section's pivot is in the current pose (glTF frame, studs).
  posedPivot(name) {
    if (!this.inst || !this.inst.pivots[name]) return null;
    const { W } = poseTargets(this.inst, this.pose);
    return this.inst.pivots[name].clone().applyMatrix4(W[name]);
  }

  stepAnim(now) {
    const a = this.anim;
    if (!a || !this.inst) return;
    const k = a.dur ? Math.min(1, (now - a.t0) / a.dur) : 1;
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    for (const [n, o] of Object.entries(this.inst.nodes)) {
      const f = a.from[n]; const t = a.to[n];
      if (!f || !t) continue;
      o.position.lerpVectors(f.pos, t.pos, e);
      o.quaternion.slerpQuaternions(f.q, t.q, e);
      o.scale.copy(t.sc);
    }
    if (k >= 1) this.anim = null;
    this.dirty = true;
  }

  // Cameras ---------------------------------------------------------------------------------------------------

  flyTo(pos, target, instant = false) {
    if (instant || reducedMotion()) {
      this.camAnim = null;
      this.persp.position.copy(pos);
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    this.camAnim = { t0: performance.now(), dur: 650, p0: this.persp.position.clone(), q0: this.controls.target.clone(),
      p1: pos.clone(), q1: target.clone() };
  }

  stepCam(now) {
    const c = this.camAnim;
    if (!c) return;
    const k = Math.min(1, (now - c.t0) / c.dur);
    const e = 1 - Math.pow(1 - k, 3);
    this.persp.position.lerpVectors(c.p0, c.p1, e);
    this.controls.target.lerpVectors(c.q0, c.q1, e);
    if (k >= 1) this.camAnim = null;
    this.dirty = true;
  }

  preset(view, instant = false) {
    this.view = view;
    if (this.mode === 'side') return;
    const centre = centreOf(this.inst);
    const dist = fitDistance(this.inst, this.persp.fov, this.persp.aspect);
    this.flyTo(centre.clone().add(VIEW_DIRS[view].clone().normalize().multiplyScalar(dist)), centre, instant);
  }

  orbitBy(dAzimuth, dPolar = 0) {
    const off = this.persp.position.clone().sub(this.controls.target);
    const s = new THREE.Spherical().setFromVector3(off);
    s.theta += dAzimuth;
    s.phi = THREE.MathUtils.clamp(s.phi + dPolar, 0.1, Math.PI - 0.1);
    off.setFromSpherical(s);
    this.flyTo(this.controls.target.clone().add(off), this.controls.target.clone(), true);
  }

  zoomBy(f) {
    const off = this.persp.position.clone().sub(this.controls.target).multiplyScalar(f);
    const len = THREE.MathUtils.clamp(off.length(), this.controls.minDistance, this.controls.maxDistance);
    this.flyTo(this.controls.target.clone().add(off.setLength(len)), this.controls.target.clone(), true);
  }

  // 3D or side by side. A frame is {dir, left, right, top, bottom} in studs about the body's origin.
  setMode(mode, frame) {
    this.mode = mode;
    this.frame = frame || null;
    if (mode === 'side' && frame) {
      this.cam = this.ortho;
      this.controls.enabled = false;
      this.controls.autoRotate = false;
      this.applyFrame(frame);
    } else {
      this.cam = this.persp;
      this.controls.enabled = true;
    }
    this.resize();
  }

  applyFrame(f) {
    const c = this.ortho;
    c.left = f.left; c.right = f.right; c.top = f.top; c.bottom = f.bottom;
    c.updateProjectionMatrix();
    const dir = { front: V3(0, 0, 1), back: V3(0, 0, -1), left: V3(1, 0, 0), right: V3(-1, 0, 0) }[f.dir];
    c.position.copy(dir.clone().multiplyScalar(40));
    c.up.set(0, 1, 0);
    c.lookAt(0, 0, 0);
    this.dirty = true;
  }

  // Input -----------------------------------------------------------------------------------------------------

  bindPicking() {
    const ray = new THREE.Raycaster();
    let down = null;
    this.canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    this.canvas.addEventListener('pointerup', (e) => {
      if (!down || !this.inst) return;
      const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
      down = null;
      if (moved > 5) return;
      const r = this.canvas.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.cam);
      const targets = Object.values(this.inst.meshes).flat().filter((m) => m.material !== this.mat.ghost);
      const hit = ray.intersectObjects(targets, false)[0];
      this.onPick(hit ? hit.object.userData.section : null);
    });
  }

  bindKeys() {
    this.canvas.addEventListener('keydown', (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey || this.mode === 'side') return;
      const step = Math.PI / 12;
      const keys = { ArrowLeft: () => this.orbitBy(-step), ArrowRight: () => this.orbitBy(step),
        ArrowUp: () => this.orbitBy(0, -step / 2), ArrowDown: () => this.orbitBy(0, step / 2),
        '+': () => this.zoomBy(0.85), '=': () => this.zoomBy(0.85), '-': () => this.zoomBy(1 / 0.85) };
      if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
    });
  }

  // Frame -----------------------------------------------------------------------------------------------------

  drawGizmo() {
    const g = this.gizmo; const W = 168; const c = W / 2; const L = 50;
    g.clearRect(0, 0, W, W);
    const qi = this.cam.quaternion.clone().invert();
    const axes = [['X', V3(1, 0, 0), cssToken('--axis-x')], ['Y up', V3(0, 1, 0), cssToken('--axis-y')],
      ['Z front', V3(0, 0, 1), cssToken('--axis-z')]]
      .map(([l, v, col]) => ({ l, v: v.applyQuaternion(qi), col })).sort((a, b) => a.v.z - b.v.z);
    g.font = '600 19px ui-monospace, "SF Mono", Menlo, monospace';
    g.lineCap = 'round';
    for (const a of axes) {
      const x = c + a.v.x * L; const y = c - a.v.y * L;
      g.strokeStyle = a.col; g.globalAlpha = a.v.z < -0.2 ? 0.45 : 1; g.lineWidth = 4;
      g.beginPath(); g.moveTo(c, c); g.lineTo(x, y); g.stroke();
      g.fillStyle = a.col; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill();
      g.textAlign = a.v.x > 0.3 ? 'left' : a.v.x < -0.3 ? 'right' : 'center';
      g.textBaseline = 'middle';
      const tw = g.measureText(a.l).width; const k = { left: 0, center: 0.5, right: 1 }[g.textAlign];
      const tx = c + a.v.x * (L + 14); const ty = c - a.v.y * (L + 14);
      g.fillText(a.l, Math.max(2 + tw * k, Math.min(W - 2 - tw * (1 - k), tx)), Math.max(10, Math.min(W - 10, ty)));
    }
    g.globalAlpha = 1;
  }

  loop(now) {
    requestAnimationFrame((t) => this.loop(t));
    if (this.canvas.offsetParent === null) return; // hidden: nothing to draw
    this.stepAnim(now);
    this.stepCam(now);
    if (this.mode !== 'side' && this.controls.update()) this.dirty = true;
    if (!this.dirty) return;
    this.dirty = false;
    this.renderer.render(this.scene, this.cam);
    this.drawGizmo();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Offscreen snapshots: a body rendered orthographically at a yaw, optionally framed to a figure's box in an image.

export class Snapshots {
  constructor(size = 360) {
    this.size = size;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(size, size, false);
    this.cam = new THREE.OrthographicCamera(-3, 3, 3, -3, 0.05, 200);
  }

  // yawDeg: camera azimuth from the front toward the character's left (+X). box: {w, h, x0, y0, x1, y1} of the
  // figure in a reference image; the body's projected box is fitted onto it, as the scorer fits held-out views.
  render(inst, yawDeg, box = null) {
    const scene = new THREE.Scene();
    lights(scene);
    const root = inst.root.clone(true);
    scene.add(root);
    root.updateMatrixWorld(true);
    const y = THREE.MathUtils.degToRad(yawDeg);
    const dir = V3(Math.sin(y), 0, Math.cos(y));
    const right = V3(Math.cos(y), 0, -Math.sin(y));
    const bb = new THREE.Box3().setFromObject(root);
    let minR = Infinity; let maxR = -Infinity;
    for (const x of [bb.min.x, bb.max.x]) for (const z of [bb.min.z, bb.max.z]) {
      const r = x * right.x + z * right.z; minR = Math.min(minR, r); maxR = Math.max(maxR, r);
    }
    const bh = bb.max.y - bb.min.y; const bw = maxR - minR;
    let aspect = 1; let left; let rightE; let top; let bottom;
    if (box && box.y1 > box.y0) {
      const s = (box.y1 - box.y0) / bh; // pixels per stud
      left = (minR + maxR) / 2 - (box.x0 + box.x1) / 2 / s; rightE = left + box.w / s;
      top = bb.max.y + box.y0 / s; bottom = top - box.h / s;
      aspect = box.w / box.h;
    } else {
      const half = Math.max(bh, bw) * 0.56;
      const cy = (bb.max.y + bb.min.y) / 2; const cr = (minR + maxR) / 2;
      left = cr - half; rightE = cr + half; top = cy + half; bottom = cy - half;
    }
    const w = aspect >= 1 ? this.size : Math.round(this.size * aspect);
    const hgt = aspect >= 1 ? Math.round(this.size / aspect) : this.size;
    this.renderer.setSize(w, hgt, false);
    const c = this.cam;
    c.left = left; c.right = rightE; c.top = top; c.bottom = bottom;
    c.near = 0.05; c.far = 200;
    c.updateProjectionMatrix();
    c.position.copy(dir.clone().multiplyScalar(60));
    c.up.set(0, 1, 0);
    c.lookAt(0, 0, 0);
    this.renderer.render(scene, c);
    return this.renderer.domElement.toDataURL('image/png');
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Compare: two bodies side by side, one camera.

export class CompareView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setScissorTest(true);
    this.cam = new THREE.PerspectiveCamera(30, 1, 0.05, 200);
    this.controls = new OrbitControls(this.cam, canvas);
    this.controls.enableDamping = !reducedMotion();
    this.scenes = [new THREE.Scene(), new THREE.Scene()];
    this.scenes.forEach(lights);
    this.items = [null, null];
    this.view = 'q';
    this.alive = true;
    this.dirty = true;
    this.controls.addEventListener('change', () => { this.dirty = true; });
    this.theme();
    this.ro = new ResizeObserver(() => this.size());
    this.ro.observe(canvas);
    this.size();
    requestAnimationFrame(() => this.loop());
  }

  theme() {
    this.scenes.forEach((s) => { s.background = new THREE.Color(cssToken('--viewer-bg') || '#11161b'); });
    this.dirty = true;
  }

  size() {
    const w = this.canvas.clientWidth; const h = this.canvas.clientHeight;
    if (w < 2 || h < 2) return;
    this.renderer.setSize(w, h, false);
    this.cam.aspect = w / 2 / h;
    this.cam.updateProjectionMatrix();
    this.dirty = true;
  }

  set(i, inst) {
    if (this.items[i]) this.scenes[i].remove(this.items[i].root);
    this.items[i] = inst;
    if (inst) this.scenes[i].add(inst.root);
    this.dirty = true;
  }

  preset(view) {
    this.view = view;
    this.size();
    const insts = this.items.filter(Boolean);
    const big = insts.sort((a, b) => b.box.getSize(V3()).y - a.box.getSize(V3()).y)[0] || null;
    const centre = centreOf(big);
    const dist = fitDistance(big, this.cam.fov, this.cam.aspect);
    this.cam.position.copy(centre.clone().add(VIEW_DIRS[view].clone().normalize().multiplyScalar(dist)));
    this.controls.target.copy(centre);
    this.controls.update();
    this.dirty = true;
  }

  dispose() {
    this.alive = false;
    this.ro.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }

  loop() {
    if (!this.alive) return;
    requestAnimationFrame(() => this.loop());
    const w = this.canvas.clientWidth; const h = this.canvas.clientHeight;
    if (w < 2 || this.canvas.offsetParent === null) return;
    if (this.controls.update()) this.dirty = true;
    if (!this.dirty) return;
    this.dirty = false;
    for (let i = 0; i < 2; i += 1) {
      this.renderer.setViewport(i * w / 2, 0, w / 2, h);
      this.renderer.setScissor(i * w / 2, 0, w / 2, h);
      this.renderer.render(this.scenes[i], this.cam);
    }
  }
}
