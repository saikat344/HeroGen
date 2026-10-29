// HeroGen lab page: one run end to end. Data from /api/runs; files from /runs/<id>/<path>; three.js pinned
// in the import map. Everything from a run folder is put on the page as text (never as HTML).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const SECTIONS = [
  'Head', 'UpperTorso', 'LowerTorso',
  'LeftUpperArm', 'LeftLowerArm', 'LeftHand',
  'RightUpperArm', 'RightLowerArm', 'RightHand',
  'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
  'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
];
const PARENT = {
  Head: 'UpperTorso', UpperTorso: 'LowerTorso', LowerTorso: null,
  LeftUpperArm: 'UpperTorso', LeftLowerArm: 'LeftUpperArm', LeftHand: 'LeftLowerArm',
  RightUpperArm: 'UpperTorso', RightLowerArm: 'RightUpperArm', RightHand: 'RightLowerArm',
  LeftUpperLeg: 'LowerTorso', LeftLowerLeg: 'LeftUpperLeg', LeftFoot: 'LeftLowerLeg',
  RightUpperLeg: 'LowerTorso', RightLowerLeg: 'RightUpperLeg', RightFoot: 'RightLowerLeg',
};
const HEIGHT_STUDS = 5.5;

// ---------------------------------------------------------------------------------------------------------
// Small DOM helpers

const $ = (sel) => document.querySelector(sel);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// replaceChildren() would print null as text; skip empty children.
function fill(target, ...kids) {
  const el = typeof target === 'string' ? $(target) : target;
  el.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
}

function fileUrl(runId, path) {
  return '/runs/' + encodeURIComponent(runId) + '/' + path.split('/').map(encodeURIComponent).join('/');
}

function fmtNum(v, digits = 4) {
  if (v === null || v === undefined) return '–';
  if (typeof v === 'boolean') return String(v);
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  return Number(v.toPrecision(digits)).toString();
}
const fmtInt = (v) => (v === null || v === undefined ? '–' : Math.round(v).toLocaleString('en-US'));
const fmtSec = (v) => (v === null || v === undefined ? '–' : v < 60 ? `${v.toFixed(1)} s` : `${(v / 60).toFixed(1)} min`);
const fmtCost = (v) => (v === null || v === undefined ? '–' : `$${v.toFixed(v < 0.01 && v > 0 ? 4 : 3)}`);

function statusBadge(status) {
  const cls = status === 'pass' ? 'pass' : status === 'gate failed' ? 'fail' : status === 'unreadable' ? 'warn' : 'none';
  return h('span', { class: `badge ${cls}`, text: status });
}

async function getJson(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------------------------------------------------
// Lightbox

function openLightbox(src, caption) {
  $('#lightbox-img').src = src;
  $('#lightbox-img').alt = caption;
  $('#lightbox-cap').textContent = caption;
  $('#lightbox').hidden = false;
}
$('#lightbox').addEventListener('click', () => { $('#lightbox').hidden = true; });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#lightbox').hidden = true; });

function thumb(runId, path, exists, tag) {
  if (!exists) return h('div', { class: 'thumb missing', text: 'not run' });
  const src = fileUrl(runId, path);
  return h('button', { type: 'button', class: 'thumb', title: path, onclick: (e) => { e.stopPropagation(); openLightbox(src, path); } },
    h('img', { src, alt: path, loading: 'lazy' }), tag ? h('span', { class: 'tag', text: tag }) : null);
}

// ---------------------------------------------------------------------------------------------------------
// Run header, chain and checks

const state = { runs: [], run: null, stageKey: null };

function renderRunMeta(run) {
  const ran = run.stages.filter((s) => s.status !== 'not run');
  const failed = run.stages.filter((s) => s.status === 'gate failed').map((s) => s.stage);
  fill('#run-meta', 
    h('span', {}, 'stages run ', h('b', { text: `${ran.length}/${run.stages.length}` })),
    h('span', {}, 'time ', h('b', { text: fmtSec(run.total_seconds) })),
    h('span', { title: 'estimate from list prices; replayed calls cost 0' }, 'cost ', h('b', { text: fmtCost(run.total_cost_usd) }), ' (est.)'),
    failed.length ? h('span', {}, 'gate failed: ', h('b', { text: failed.map((k) => `stage ${k}`).join(', ') })) : null,
  );
}

function renderChain(run) {
  const stages = Object.fromEntries(run.stages.map((s) => [s.stage, s]));
  const cards = run.chain.map((step, i) => {
    const s = stages[step.stage] || { status: 'not run' };
    const anyImage = step.images.some((im) => im.exists);
    const status = s.status === 'not run' && anyImage ? 'no record' : s.status;
    const four = step.images.length === 4;
    const tags = four ? step.images.map((im) => im.path.split('/').pop().replace('.png', '')) : [];
    const thumbs = h('div', { class: four ? 'thumbs four' : 'thumbs' },
      step.images.map((im, j) => thumb(run.id, im.path, im.exists, tags[j])));
    const checks = s.checks || [];
    const failedGate = s.checks_failed_gate || 0;
    const foot = h('div', { class: 'step-foot' },
      checks.length
        ? h('span', { class: failedGate ? 'bad' : 'ok', text: `${s.checks_passed}/${checks.length} checks pass` })
        : h('span', { text: s.status === 'not run' ? 'no stage record' : 'no checks' }),
      h('span', { text: s.seconds !== undefined ? `${fmtSec(s.seconds)} · ${fmtCost(s.cost_usd)}` : '' }));
    const card = h('div', {
      class: 'step' + (state.stageKey === step.stage ? ' selected' : ''), role: 'listitem', tabindex: 0,
      'data-stage': step.stage,
      onclick: () => selectStage(step.stage),
      onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectStage(step.stage); } },
    },
    h('div', { class: 'step-head' },
      h('span', {}, h('span', { class: 'step-num', text: String(i + 1) }), h('span', { class: 'step-title', text: step.label })),
      statusBadge(status === 'no record' ? 'not run' : status)),
    thumbs,
    step.sheet ? h('button', { type: 'button', class: 'sheet-link', onclick: (e) => { e.stopPropagation(); openLightbox(fileUrl(run.id, step.sheet), step.sheet); } }, 'Open the full turnaround sheet') : null,
    foot);
    if (status === 'no record') card.querySelector('.badge').textContent = 'no record';
    return card;
  });
  fill('#chain', ...cards);
}

function selectStage(key) {
  state.stageKey = key;
  for (const el of document.querySelectorAll('.step')) el.classList.toggle('selected', el.dataset.stage === key);
  renderChecks();
}

function renderChecks() {
  const run = state.run;
  const s = run.stages.find((x) => x.stage === state.stageKey);
  const step = run.chain.find((c) => c.stage === state.stageKey);
  const title = `Stage ${state.stageKey}${step ? ` · ${step.label}` : ''}`;
  if (!s || s.status === 'not run') {
    fill('#checks', h('div', { class: 'checks-head' }, h('h3', { text: title }), statusBadge('not run')),
      h('p', { class: 'empty', text: `No stage-${state.stageKey}.json in this run.` }));
    return;
  }
  if (s.status === 'unreadable') {
    fill('#checks', h('div', { class: 'checks-head' }, h('h3', { text: title }), statusBadge('unreadable')),
      h('p', { class: 'empty', text: s.error }));
    return;
  }
  const rows = s.checks.map((c) => {
    const res = c.pass ? 'pass' : (c.gate ? 'fail' : 'info');
    return h('tr', {},
      h('td', {}, h('span', { class: 'mono', text: c.name }),
        (c.notes || []).map((n) => h('span', { class: 'small', text: n }))),
      h('td', { class: 'num', text: fmtNum(c.value) }),
      h('td', {
        class: 'num mono', title: c.cmp_assumed ? 'No comparison recorded; >= is the default for a check with a threshold' : null,
        text: c.threshold === null || c.threshold === undefined ? '–' : `${c.cmp} ${fmtNum(c.threshold)}`,
      }),
      h('td', { class: `res-${res}`, text: c.pass ? 'pass' : (c.gate ? 'fail' : 'fail (not gated)') }),
      h('td', { text: c.gate ? 'yes' : 'no' }));
  });
  const models = [...(s.models || []), ...(s.tools || [])];
  fill('#checks', 
    h('div', { class: 'checks-head' },
      h('h3', { text: title }), statusBadge(s.status),
      h('span', { class: 'meta', text: `${fmtSec(s.seconds)} · ${fmtCost(s.cost_usd)} · ${models.length ? models.join(', ') : 'no models'}` })),
    s.label ? h('p', { class: 'notes', text: s.label }) : null,
    rows.length
      ? h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', { text: 'check' }), h('th', { class: 'num', text: 'value' }),
          h('th', { class: 'num', text: 'threshold' }), h('th', { text: 'result' }), h('th', { text: 'gate' }))),
        h('tbody', {}, rows)))
      : h('p', { class: 'empty', text: 'This stage records no checks.' }),
    s.notes ? h('p', { class: 'notes', text: s.notes }) : null,
    s.cost_basis ? h('p', { class: 'notes', text: `Cost: ${s.cost_basis}` }) : null);
}

// ---------------------------------------------------------------------------------------------------------
// Poses. Same rotations as herogen/blender/pose.py, converted from Blender (Z up, faces -Y) to glTF (Y up,
// faces +Z): (x, y, z) -> (x, z, -y), a proper rotation, so axes convert like vectors.

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const FWD = V(0, 0, 1);
const BACK = V(0, 0, -1);
const UP = V(0, 1, 0);
const X_AXIS = V(1, 0, 0);
const OUT_L_UP = V(1, 1, 0);
const OUT_R_UP = V(-1, 1, 0);

function rot(axis, deg) {
  return new THREE.Quaternion().setFromAxisAngle(axis.clone().normalize(), THREE.MathUtils.degToRad(deg));
}
function swing(dir, target, deg) {
  let axis = new THREE.Vector3().crossVectors(dir, target);
  if (axis.lengthSq() < 1e-12) axis = new THREE.Vector3().crossVectors(dir, X_AXIS);
  return rot(axis, deg);
}

const PRESETS = {
  rest: { label: 'Rest', make: () => ({}) },
  walk: { label: 'Walk', make: (d) => walkPose(d, 'Left', 'Right') },
  arms_up: {
    label: 'Arms up',
    make: (d) => ({
      LeftUpperArm: swing(d.LeftUpperArm, OUT_L_UP, 75), RightUpperArm: swing(d.RightUpperArm, OUT_R_UP, 75),
      LeftLowerArm: swing(d.LeftLowerArm, UP, 30), RightLowerArm: swing(d.RightLowerArm, UP, 30),
    }),
  },
  crouch: {
    label: 'Crouch',
    make: (d) => ({
      LeftUpperLeg: swing(d.LeftUpperLeg, FWD, 75), RightUpperLeg: swing(d.RightUpperLeg, FWD, 75),
      LeftLowerLeg: swing(d.LeftLowerLeg, BACK, 100), RightLowerLeg: swing(d.RightLowerLeg, BACK, 100),
      LeftFoot: rot(X_AXIS, 25), RightFoot: rot(X_AXIS, 25),
      UpperTorso: swing(UP, FWD, 25), Head: swing(UP, BACK, 20),
      LeftUpperArm: swing(d.LeftUpperArm, FWD, 40), RightUpperArm: swing(d.RightUpperArm, FWD, 40),
    }),
  },
  arms_forward: {
    label: 'Arms forward',
    make: (d) => ({
      LeftUpperArm: swing(d.LeftUpperArm, FWD, 80), RightUpperArm: swing(d.RightUpperArm, FWD, 80),
      LeftLowerArm: swing(d.LeftLowerArm, FWD, 20), RightLowerArm: swing(d.RightLowerArm, FWD, 20),
      LeftHand: rot(d.LeftLowerArm, 30), RightHand: rot(d.RightLowerArm, -30),
    }),
  },
  twist: {
    label: 'Twist',
    make: (d) => ({
      UpperTorso: rot(UP, 30), Head: rot(UP, 45),
      LeftUpperArm: swing(d.LeftUpperArm, FWD, 30), RightUpperArm: swing(d.RightUpperArm, BACK, 30),
    }),
  },
};

function walkPose(d, s1, s2) {
  return {
    [s1 + 'UpperLeg']: swing(d[s1 + 'UpperLeg'], FWD, 30), [s1 + 'LowerLeg']: swing(d[s1 + 'LowerLeg'], BACK, 15),
    [s2 + 'UpperLeg']: swing(d[s2 + 'UpperLeg'], BACK, 25), [s2 + 'LowerLeg']: swing(d[s2 + 'LowerLeg'], BACK, 40),
    [s2 + 'Foot']: rot(X_AXIS, -15),
    [s1 + 'UpperArm']: swing(d[s1 + 'UpperArm'], BACK, 25), [s2 + 'UpperArm']: swing(d[s2 + 'UpperArm'], FWD, 25),
    [s1 + 'LowerArm']: swing(d[s1 + 'LowerArm'], FWD, 20), [s2 + 'LowerArm']: swing(d[s2 + 'LowerArm'], FWD, 30),
  };
}

const SLIDERS = [
  { key: 'arms', label: 'Arms raise', min: -30, max: 150 },
  { key: 'elbows', label: 'Elbows', min: 0, max: 130 },
  { key: 'stride', label: 'Stride', min: -45, max: 45 },
  { key: 'knees', label: 'Knees', min: 0, max: 120 },
  { key: 'twist', label: 'Torso twist', min: -60, max: 60 },
  { key: 'head', label: 'Head turn', min: -80, max: 80 },
];

function sliderPose(d, v) {
  const p = {};
  const add = (name, q) => { p[name] = p[name] ? p[name].multiply(q) : q; };
  if (v.arms) { add('LeftUpperArm', swing(d.LeftUpperArm, OUT_L_UP, v.arms)); add('RightUpperArm', swing(d.RightUpperArm, OUT_R_UP, v.arms)); }
  if (v.elbows) { add('LeftLowerArm', swing(d.LeftLowerArm, FWD, v.elbows)); add('RightLowerArm', swing(d.RightLowerArm, FWD, v.elbows)); }
  if (v.stride) { add('LeftUpperLeg', swing(d.LeftUpperLeg, FWD, v.stride)); add('RightUpperLeg', swing(d.RightUpperLeg, FWD, -v.stride)); }
  if (v.knees) { add('LeftLowerLeg', swing(d.LeftLowerLeg, BACK, v.knees)); add('RightLowerLeg', swing(d.RightLowerLeg, BACK, v.knees)); }
  if (v.twist) add('UpperTorso', rot(UP, v.twist));
  if (v.head) add('Head', rot(UP, v.head));
  return p;
}

// ---------------------------------------------------------------------------------------------------------
// Viewer: one renderer, one scene; compare mode draws each body in its own viewport with the shared camera.

const CAMERAS = {
  front: V(0, 3.0, 1), quarter: V(0.72, 3.0, 0.72), side: V(1, 3.0, 0), back: V(0, 3.0, -1),
};
const TARGET = V(0, 2.6, 0);
const DIST = 12.5;

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function textSprite(text) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.font = '600 30px system-ui, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillStyle = cssVar('--muted') || '#9aa3b2';
  g.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set(1.6, 0.4, 1);
  return s;
}

function scaleReference() {
  // A 5.5-stud pole with a tick every stud, beside the body.
  const g = new THREE.Group();
  const pts = [V(0, 0, 0), V(0, HEIGHT_STUDS, 0)];
  for (let y = 0; y <= HEIGHT_STUDS + 1e-6; y += 1) pts.push(V(-0.12, y, 0), V(0.12, y, 0));
  pts.push(V(-0.22, HEIGHT_STUDS, 0), V(0.22, HEIGHT_STUDS, 0));
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar('--warn') || '#f2c14e') });
  g.add(new THREE.LineSegments(geom, mat));
  const label = textSprite('5.5 studs');
  label.position.set(0, HEIGHT_STUDS + 0.35, 0);
  g.add(label);
  g.position.set(-2.6, 0, -0.6);
  return g;
}

function disposeMaterial(mat, seen) {
  if (!mat || seen.has(mat)) return;
  seen.add(mat);
  for (const v of Object.values(mat)) if (v && v.isTexture && !seen.has(v)) { seen.add(v); v.dispose(); }
  mat.dispose();
}

// Free the GPU memory of everything under `root`, plus any materials kept outside it.
function disposeTree(root, extraMaterials = []) {
  const seen = new Set();
  root.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    for (const m of [].concat(o.material || [])) disposeMaterial(m, seen);
  });
  for (const m of extraMaterials) disposeMaterial(m, seen);
}

class Viewer {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setScissorTest(true);
    el.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 40;
    this.setCamera('front');

    this.grid = new THREE.GridHelper(12, 12, 0x6b7385, 0x3a404c);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.7;
    this.scene.add(this.grid, scaleReference());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x50555f, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(3, 7, 6);
    this.scene.add(sun);

    this.ghost = new THREE.MeshBasicMaterial({ color: 0x8fa3c0, transparent: true, opacity: 0.12, depthWrite: false });
    this.marker = new THREE.Group();
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xff3b6b, depthTest: false }));
    dot.renderOrder = 10;
    const axes = new THREE.AxesHelper(0.45);
    axes.material.depthTest = false;
    axes.renderOrder = 10;
    this.marker.add(dot, axes);
    this.marker.visible = false;
    this.scene.add(this.marker);
    this.boxHelper = null;

    this.slots = [];        // [{label, entry}] drawn left to right
    this.lit = false;
    this.selected = null;
    this.isolate = false;
    this.cache = new Map(); // url -> Promise<entry>, for the current run only (see releaseAll)
    this.entries = [];      // every loaded body root, shown only while it is in a slot
    this.loaded = [];       // every prepared entry, so releaseAll can free its GPU memory
    this.gen = 0;           // bumped by releaseAll; a load that finishes after it is thrown away
    this.loader = new GLTFLoader();
    this.pose = { from: null, to: {}, t0: 0, dur: 450, walk: false, phase: 0 };
    this.clock = new THREE.Clock();
    this.onPick = () => {};

    this.applyTheme();
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => this.applyTheme());
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    this.bindPicking();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  applyTheme() {
    this.renderer.setClearColor(new THREE.Color(cssVar('--viewer-bg') || '#1b1f27'), 1);
  }

  resize() {
    const w = this.el.clientWidth;
    const hgt = this.el.clientHeight;
    this.renderer.setSize(w, hgt, false);
  }

  setCamera(name) {
    const dir = CAMERAS[name].clone();
    const flat = V(dir.x, 0, dir.z).normalize();
    this.camera.position.copy(TARGET).addScaledVector(flat, DIST).setY(3.2);
    this.controls.target.copy(TARGET);
    this.controls.update();
  }

  load(url) {
    if (!this.cache.has(url)) {
      const gen = this.gen;
      this.cache.set(url, this.loader.loadAsync(url).then((gltf) => {
        if (gen !== this.gen) {
          disposeTree(gltf.scene);
          throw new Error('superseded by another run');
        }
        return this.prepare(gltf);
      }));
    }
    return this.cache.get(url);
  }

  // Drop every body: out of the slots and the scene, geometry, materials and textures freed, cache emptied.
  releaseAll() {
    this.gen += 1;
    this.selected = null;
    this.setSlots([]);
    this.pose.from = null;
    for (const entry of this.loaded) {
      this.scene.remove(entry.root);
      disposeTree(entry.root, entry.meshes.flatMap((m) => [m.lit, m.basic]));
    }
    this.loaded = [];
    this.entries = [];
    this.cache.clear();
  }

  prepare(gltf) {
    const root = gltf.scene;
    root.updateMatrixWorld(true);
    const nodes = {};
    for (const name of SECTIONS) {
      const n = root.getObjectByName(name);
      if (n) nodes[name] = n;
    }
    // Rebuild the R15 hierarchy so a parent's turn carries its children (attach keeps world transforms).
    for (const name of SECTIONS) {
      const parent = PARENT[name];
      if (nodes[name] && parent && nodes[parent]) nodes[parent].attach(nodes[name]);
    }
    root.updateMatrixWorld(true);
    const meshes = [];
    const tris = {};
    root.traverse((o) => {
      if (!o.isMesh) return;
      let p = o;
      while (p && !SECTIONS.includes(p.name)) p = p.parent;
      const section = p ? p.name : null;
      const lit = o.material;
      lit.side = THREE.DoubleSide;
      const basic = new THREE.MeshBasicMaterial({ map: lit.map || null, color: lit.map ? 0xffffff : lit.color, side: THREE.DoubleSide });
      const g = o.geometry;
      const n = g.index ? g.index.count / 3 : g.attributes.position.count / 3;
      if (section) tris[section] = (tris[section] || 0) + n;
      o.userData.section = section;
      meshes.push({ mesh: o, section, lit, basic });
    });
    const rest = {};
    const pivots = {};
    for (const [name, n] of Object.entries(nodes)) {
      rest[name] = n.quaternion.clone();
      pivots[name] = n.getWorldPosition(new THREE.Vector3());
    }
    const dir = (a, b, fallback) => (pivots[a] && pivots[b]
      ? new THREE.Vector3().subVectors(pivots[b], pivots[a]).normalize() : fallback.clone());
    const dirs = {};
    for (const side of ['Left', 'Right']) {
      const out = side === 'Left' ? V(1, -1, 0).normalize() : V(-1, -1, 0).normalize();
      dirs[side + 'UpperArm'] = dir(side + 'UpperArm', side + 'LowerArm', out);
      dirs[side + 'LowerArm'] = dir(side + 'LowerArm', side + 'Hand', out);
      dirs[side + 'Hand'] = dirs[side + 'LowerArm'];
      dirs[side + 'UpperLeg'] = dir(side + 'UpperLeg', side + 'LowerLeg', V(0, -1, 0));
      dirs[side + 'LowerLeg'] = dir(side + 'LowerLeg', side + 'Foot', V(0, -1, 0));
      dirs[side + 'Foot'] = V(0, -1, 0);
    }
    const totalTris = Object.values(tris).reduce((a, b) => a + b, 0);
    root.visible = false;
    this.scene.add(root);
    this.entries.push(root);
    const entry = { root, nodes, meshes, tris, totalTris, rest, pivots, dirs, sectionCount: Object.keys(nodes).length };
    this.loaded.push(entry);
    this.applyMaterials(entry);
    this.applyPoseTo(entry);
    return entry;
  }

  setSlots(slots) {
    this.slots = slots;
    this.refreshMaterials();
    this.updateMarker();
  }

  setLit(lit) { this.lit = lit; this.refreshMaterials(); }
  setIsolate(on) { this.isolate = on; this.refreshMaterials(); }
  select(name) { this.selected = name; this.refreshMaterials(); this.updateMarker(); }

  refreshMaterials() { for (const s of this.slots) this.applyMaterials(s.entry); }

  applyMaterials(entry) {
    for (const m of entry.meshes) {
      const base = this.lit ? m.lit : m.basic;
      if (!this.selected || m.section === this.selected) {
        m.mesh.material = base;
        m.mesh.visible = true;
      } else {
        m.mesh.material = this.ghost;
        m.mesh.visible = !this.isolate;
      }
    }
  }

  updateMarker() {
    if (this.boxHelper) { this.scene.remove(this.boxHelper); this.boxHelper.dispose(); this.boxHelper = null; }
    this.marker.visible = false;
    if (!this.selected || this.slots.length !== 1) return;
    const node = this.slots[0].entry.nodes[this.selected];
    if (!node) return;
    // The section's own meshes only (its node now also carries its child sections).
    this.boxMeshes = this.slots[0].entry.meshes.filter((m) => m.section === this.selected).map((m) => m.mesh);
    this.box = new THREE.Box3();
    this.boxHelper = new THREE.Box3Helper(this.box, 0xff3b6b);
    this.boxHelper.material.depthTest = false;
    this.boxHelper.renderOrder = 9;
    this.scene.add(this.boxHelper);
    this.marker.visible = true;
  }

  // Poses -------------------------------------------------------------------------------------------------

  setPose(make) {
    this.pose.walk = false;
    this.pose.from = this.snapshot();
    this.pose.make = make;
    this.pose.t0 = performance.now();
  }

  setPoseNow(make) {
    this.pose.walk = false;
    this.pose.from = null;
    this.pose.make = make;
  }

  setWalk(on) {
    this.pose.from = this.snapshot();
    this.pose.walk = on;
    this.pose.phase = 0;
    this.pose.t0 = performance.now();
    if (!on) this.pose.make = PRESETS.rest.make;
  }

  snapshot() {
    const snap = new Map();
    for (const e of this.allEntries()) {
      const q = {};
      for (const [name, n] of Object.entries(e.nodes)) q[name] = n.quaternion.clone();
      snap.set(e, q);
    }
    return snap;
  }

  allEntries() { return new Set(this.slots.map((s) => s.entry)); }

  targetFor(entry) {
    if (this.pose.walk) {
      const a = walkPose(entry.dirs, 'Left', 'Right');
      const b = walkPose(entry.dirs, 'Right', 'Left');
      const w = (Math.sin(this.pose.phase) + 1) / 2;
      const out = {};
      for (const name of SECTIONS) {
        const qa = a[name] || new THREE.Quaternion();
        const qb = b[name] || new THREE.Quaternion();
        out[name] = qa.clone().slerp(qb, w);
      }
      return out;
    }
    const make = this.pose.make || PRESETS.rest.make;
    return make(entry.dirs);
  }

  applyPoseTo(entry, now = performance.now()) {
    const target = this.targetFor(entry);
    const from = this.pose.from && this.pose.from.get(entry);
    let t = 1;
    if (from) {
      t = Math.min(1, (now - this.pose.t0) / this.pose.dur);
      t = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    }
    for (const [name, n] of Object.entries(entry.nodes)) {
      // The pose turn is in the parent's frame, applied after the rest orientation.
      const goal = (target[name] || new THREE.Quaternion()).clone().multiply(entry.rest[name]);
      if (from && from[name]) n.quaternion.copy(from[name]).slerp(goal, t);
      else n.quaternion.copy(goal);
    }
    this.groundSnap(entry);
  }

  groundSnap(entry) {
    // Keep the lowest vertex on the ground (a rigid move of the whole body; no geometry changes).
    const root = entry.root;
    root.position.y = 0;
    root.updateMatrixWorld(true);
    let minY = Infinity;
    for (const m of entry.meshes) {
      const pos = m.mesh.geometry.attributes.position;
      const mw = m.mesh.matrixWorld;
      const e = mw.elements;
      for (let i = 0; i < pos.count; i += 1) {
        const y = e[1] * pos.getX(i) + e[5] * pos.getY(i) + e[9] * pos.getZ(i) + e[13];
        if (y < minY) minY = y;
      }
    }
    if (Number.isFinite(minY)) root.position.y = -minY;
    root.updateMatrixWorld(true);
  }

  // Picking -----------------------------------------------------------------------------------------------

  bindPicking() {
    const canvas = this.renderer.domElement;
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
      down = null;
      const rect = canvas.getBoundingClientRect();
      const n = Math.max(1, this.slots.length);
      const slotW = rect.width / n;
      const i = Math.min(n - 1, Math.floor((e.clientX - rect.left) / slotW));
      const slot = this.slots[i];
      if (!slot) return;
      const x = ((e.clientX - rect.left - i * slotW) / slotW) * 2 - 1;
      const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      this.camera.aspect = slotW / rect.height;
      this.camera.updateProjectionMatrix();
      const ray = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2(x, y), this.camera);
      slot.entry.root.visible = true;
      const targets = slot.entry.meshes.filter((m) => m.mesh.visible && m.mesh.material !== this.ghost).map((m) => m.mesh);
      const hit = ray.intersectObjects(targets, false)[0];
      this.onPick(hit ? hit.object.userData.section : null);
    });
  }

  // Frame -------------------------------------------------------------------------------------------------

  frame() {
    const dt = this.clock.getDelta();
    const now = performance.now();
    this.controls.update();
    // While a turn is easing in, every frame; then once more at its end (frames can be skipped when hidden).
    const tweening = Boolean(this.pose.from) && now - this.pose.t0 < this.pose.dur;
    if (this.pose.walk) this.pose.phase += Math.min(dt, 0.1) * 3.2;
    if (this.pose.walk || this.pose.from || this.poseDirty) {
      for (const e of this.allEntries()) this.applyPoseTo(e, now);
      this.poseDirty = false;
      if (!this.pose.walk && !tweening) this.pose.from = null;
    }
    if (this.boxHelper) {
      this.box.makeEmpty();
      const b = new THREE.Box3();
      for (const m of this.boxMeshes) {
        if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
        this.box.union(b.copy(m.geometry.boundingBox).applyMatrix4(m.matrixWorld));
      }
    }
    if (this.marker.visible && this.slots[0]) {
      const node = this.slots[0].entry.nodes[this.selected];
      if (node) { node.getWorldPosition(this.marker.position); node.getWorldQuaternion(this.marker.quaternion); }
    }

    const w = this.el.clientWidth;
    const hgt = this.el.clientHeight;
    const n = Math.max(1, this.slots.length);
    const slotW = w / n;
    const aspect = slotW / hgt;
    // Fit about 7.2 studs tall and 6 wide at the default distance, whatever the viewport shape.
    const half = Math.max(3.6, 3.0 / aspect);
    this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(half / DIST));
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    for (const r of this.entries) r.visible = false;
    if (!this.slots.length) {
      this.renderer.setViewport(0, 0, w, hgt);
      this.renderer.setScissor(0, 0, w, hgt);
      this.renderer.render(this.scene, this.camera);
      return;
    }
    this.slots.forEach((s, i) => {
      for (const o of this.slots) o.entry.root.visible = o === s;
      this.renderer.setViewport(i * slotW, 0, slotW, hgt);
      this.renderer.setScissor(i * slotW, 0, slotW, hgt);
      this.renderer.render(this.scene, this.camera);
    });
  }
}

// ---------------------------------------------------------------------------------------------------------
// Body panel

const ui = { bodyIndex: 0, lod: 'high', compare: false, section: null, preset: 'rest', sliders: {} };
let viewer = null;
let loadToken = 0;

function currentBody() { return state.run && state.run.bodies[ui.bodyIndex]; }

function viewerMessage(text) {
  $('#viewer-msg').hidden = !text;
  $('#viewer-msg').textContent = text || '';
}

function renderBodyControls() {
  const run = state.run;
  const bodies = run ? run.bodies : [];
  fill('#body-pick', ...bodies.map((b, i) => h('button', {
    type: 'button', class: i === ui.bodyIndex ? 'on' : '', title: b.dir,
    onclick: () => { ui.bodyIndex = i; ui.compare = false; renderBodyControls(); showBodies(); },
  }, b.name)));
  $('#body-pick').hidden = bodies.length < 2;
  const body = currentBody();
  for (const b of document.querySelectorAll('[data-lod]')) {
    b.classList.toggle('on', b.dataset.lod === ui.lod);
    b.disabled = !body || !body[b.dataset.lod];
  }
  const cmp = $('#compare-btn');
  cmp.disabled = bodies.length < 2;
  cmp.title = bodies.length < 2 ? 'This run has one body' : 'Show every body side by side with the same camera';
  cmp.setAttribute('aria-pressed', String(ui.compare));
}

async function showBodies() {
  const run = state.run;
  const token = ++loadToken;
  if (!run) return;
  const bodies = run.bodies;
  renderSectionList();
  if (!bodies.length) {
    viewer.setSlots([]);
    fill('#viewer-labels');
    fill('#tri-totals');
    viewerMessage('Stage 4 not run: this run has no 04-body/high.glb or low.glb.');
    return;
  }
  const list = ui.compare ? bodies : [currentBody()];
  viewerMessage('Loading the body…');
  try {
    const entries = await Promise.all(list.map((b) => {
      const path = b[ui.lod] || b.high || b.low;
      return viewer.load(fileUrl(run.id, path)).then((entry) => ({ body: b, path, entry }));
    }));
    if (token !== loadToken) return;
    viewer.setSlots(entries.map((e) => ({ label: e.body.name, entry: e.entry })));
    viewer.poseDirty = true;
    viewerMessage('');
    fill('#viewer-labels', ...entries.map((e) => h('div', { class: 'vl' },
      h('div', { class: 'vl-name', text: e.body.name }),
      h('div', { class: 'vl-sub', text: `${e.path} · ${fmtInt(e.entry.totalTris)} triangles · ${e.entry.sectionCount} sections` }))));
    renderSectionList(entries[0].entry);
    renderTriTotals(entries);
  } catch (err) {
    if (token !== loadToken) return;
    viewer.setSlots([]);
    viewerMessage(`Could not load the body: ${err.message || err}`);
  }
}

function renderTriTotals(entries) {
  const rows = [];
  for (const e of entries) {
    const parts = [];
    const sec = e.body.sections && !e.body.sections.error ? e.body.sections : null;
    // budgets is null when the body was built without a triangle budget (the default "best" profile).
    const budgets = sec ? sec.budgets : null;
    const noBudget = sec && !budgets
      ? (!sec.profile || sec.profile === 'best' ? 'no budget (best quality)' : 'no budget recorded') : null;
    const sum = (k) => (sec ? sec.sections.reduce((a, s) => a + (s[k] || 0), 0) : null);
    const lodRow = (lod, key, budgetKey) => {
      const n = sum(key);
      const budget = budgets ? budgets[budgetKey] : undefined;
      if (n === null) return null;
      const ok = budget === undefined ? null : n <= budget;
      return h('span', {}, `${lod} `, h('b', { text: fmtInt(n) }),
        budget !== undefined ? ` / ${fmtInt(budget)} ` : ' ',
        ok === null ? null : h('span', { class: ok ? 'ok' : 'bad', text: ok ? 'within budget' : 'over budget' }));
    };
    parts.push(h('span', {}, h('b', { text: e.body.name }), ':'));
    if (sec && sec.profile) parts.push(h('span', {}, 'profile ', h('b', { class: 'mono', text: sec.profile })));
    if (sec) {
      parts.push(lodRow('high', 'triangles_high', 'triangles_high_max'), lodRow('low', 'triangles_low', 'triangles_low_max'));
      if (noBudget) parts.push(h('span', { class: 'nobudget', text: noBudget }));
    } else {
      parts.push(h('span', { text: 'no sections.json' }));
    }
    parts.push(h('span', { text: `in viewer (${e.body[ui.lod] ? ui.lod : 'loaded'}): ${fmtInt(e.entry.totalTris)} counted from the GLB` }));
    rows.push(h('div', { class: 'tt-row' }, parts));
  }
  fill('#tri-totals', rows);
}

function renderSectionList(entry) {
  const body = currentBody();
  const sec = body && body.sections && !body.sections.error ? body.sections : null;
  const byName = Object.fromEntries((sec ? sec.sections : []).map((s) => [s.name, s]));
  const names = entry ? SECTIONS.filter((n) => entry.nodes[n]) : (sec ? sec.sections.map((s) => s.name) : []);
  if (!names.length) {
    fill('#section-list', h('li', { class: 'empty', text: 'not run' }));
    fill('#section-detail');
    return;
  }
  fill('#section-list', ...names.map((n) => {
    const t = entry ? entry.tris[n] : (byName[n] || {})[`triangles_${ui.lod}`];
    return h('li', {}, h('button', {
      type: 'button', class: ui.section === n ? 'on' : '', 'data-section': n,
      onclick: () => pickSection(ui.section === n ? null : n),
    }, h('span', { text: n }), h('span', { class: 'tri', text: fmtInt(t) })));
  }));
  renderSectionDetail(entry, byName);
}

function renderSectionDetail(entry, byName) {
  const n = ui.section;
  if (!n) {
    fill('#section-detail', h('p', { class: 'hint', text: 'Click a section (here or on the body) to highlight it, see its triangles and its pivot. Isolate hides the rest.' }));
    return;
  }
  const s = byName[n] || {};
  const glbPivot = entry && entry.pivots[n];
  // sections.json pivots are in studs, Blender frame (Z up, faces -Y); the GLB node sits at the same point.
  const pivot = s.pivot ? s.pivot : glbPivot ? [glbPivot.x, -glbPivot.z, glbPivot.y] : null;
  const rows = [
    ['parent', s.parent || PARENT[n] || 'none (root)'],
    ['joint', s.joint || '–'],
    ['triangles (high)', fmtInt(s.triangles_high)],
    ['triangles (low)', fmtInt(s.triangles_low)],
    [`in viewer (${ui.lod})`, entry ? fmtInt(entry.tris[n]) : '–'],
    ['pivot (studs, Z up)', pivot ? pivot.map((x) => x.toFixed(3)).join(', ') : '–'],
  ];
  fill('#section-detail', h('h3', { text: n }),
    h('dl', {}, rows.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', { text: v })])));
}

function pickSection(name) {
  ui.section = name;
  viewer.select(name);
  const entry = viewer.slots[0] && viewer.slots[0].entry;
  renderSectionList(entry);
}

function renderPoseControls() {
  fill('#presets', ...Object.entries(PRESETS).map(([key, p]) => h('button', {
    type: 'button', class: ui.preset === key ? 'on' : '', 'data-preset': key,
    onclick: () => applyPreset(key),
  }, p.label)));
  fill('#sliders', ...SLIDERS.map((s) => {
    const out = h('output', { text: '0°' });
    const input = h('input', {
      type: 'range', min: s.min, max: s.max, step: 1, value: ui.sliders[s.key] || 0, 'aria-label': s.label,
      oninput: (e) => {
        ui.sliders[s.key] = Number(e.target.value);
        out.textContent = `${ui.sliders[s.key]}°`;
        ui.preset = null;
        setWalkButton(false);
        for (const b of document.querySelectorAll('[data-preset]')) b.classList.remove('on');
        const values = { ...ui.sliders };
        viewer.setPoseNow((d) => sliderPose(d, values));
        viewer.poseDirty = true;
      },
    });
    out.textContent = `${input.value}°`;
    return h('label', { class: 'slider' }, h('span', { text: s.label }), input, out);
  }));
}

function setWalkButton(on) { $('#walk-btn').setAttribute('aria-pressed', String(on)); }

function applyPreset(key) {
  ui.preset = key;
  ui.sliders = {};
  setWalkButton(false);
  renderPoseControls();
  viewer.setPose(PRESETS[key].make);
}

// ---------------------------------------------------------------------------------------------------------
// Scores

function renderScores(run) {
  const sc = run.scores;
  const el = $('#scores');
  if (!sc) {
    fill(el, h('p', { class: 'empty', text: 'Scoring not run: this run has no 05-scores.json.' }),
      compareActions(run));
    return;
  }
  if (sc.error && !sc.bodies.length) {
    fill(el, h('p', { class: 'empty', text: sc.error }), compareActions(run));
    return;
  }
  // Best value per view and metric across bodies (higher IoU / SSIM, lower colour error).
  const best = {};
  for (const v of sc.views) {
    for (const [k, better] of [['iou', 1], ['ssim', 1], ['colour_error', -1]]) {
      const vals = sc.bodies.map((b) => b.views[v] && b.views[v][k]).filter((x) => typeof x === 'number');
      if (vals.length > 1) best[`${v}:${k}`] = better > 0 ? Math.max(...vals) : Math.min(...vals);
    }
  }
  const cards = sc.bodies.map((b) => {
    const rows = sc.views.map((v) => {
      const r = b.views[v];
      const held = v === sc.heldout_view;
      if (!r) return h('tr', { class: held ? 'heldout' : '' }, h('td', { text: v }), h('td', { colspan: 4, class: 'empty', text: 'not scored' }));
      const cell = (k, digits) => h('td', {
        class: 'num' + (best[`${v}:${k}`] !== undefined && r[k] === best[`${v}:${k}`] ? ' best' : ''),
        text: r[k] === null ? '–' : r[k].toFixed(digits),
      });
      const label = held
        ? h('td', {}, 'held-out', h('span', { class: 'small', text: [r.yaw_deg !== null ? `yaw ${r.yaw_deg.toFixed(0)}°` : null, r.scale !== null ? `scale ${r.scale.toFixed(2)}` : null].filter(Boolean).join(' · ') }))
        : h('td', { text: v });
      const seen = r.seen === false ? h('td', { class: 'unseen', text: 'unseen' }) : h('td', { class: 'seen', text: r.seen ? 'seen' : '–' });
      return h('tr', { class: held ? 'heldout' : '' }, label, cell('iou', 3), cell('colour_error', 1), cell('ssim', 3), seen);
    });
    return h('div', { class: 'card score-card' },
      h('h3', {}, b.name, b.variant_of ? h('span', { class: 'badge none', text: `variant of ${b.variant_of}` }) : null),
      h('p', { class: 'meta', text: [b.glb, b.mesh_source ? `mesh ${b.mesh_source}` : null, b.joints_source ? `joints ${b.joints_source}` : null, b.triangles !== null ? `${fmtInt(b.triangles)} tris` : null].filter(Boolean).join(' · ') }),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', { text: 'view' }), h('th', { class: 'num', text: 'IoU' }),
          h('th', { class: 'num', text: 'colour err' }), h('th', { class: 'num', text: 'SSIM' }), h('th', { text: '' }))),
        h('tbody', {}, rows))));
  });
  const strip = [];
  if (sc.heldout_image) strip.push(h('figure', {}, thumb(run.id, sc.heldout_image, true), h('figcaption', { text: 'held-out image (never used to build)' })));
  for (const b of sc.bodies) {
    if (b.heldout_render) strip.push(h('figure', {}, thumb(run.id, b.heldout_render, true), h('figcaption', { text: `${b.name}, same view` })));
  }
  fill(el, 
    compareActions(run),
    sc.error ? h('p', { class: 'empty', text: sc.error }) : null,
    h('div', { class: 'score-grid' }, cards),
    strip.length ? h('div', { class: 'heldout-strip' }, strip) : null,
    h('p', { class: 'hint', text: 'IoU: silhouette overlap with the view mask. Colour error and SSIM are measured inside the masks. Bold is the best value in each row across bodies. The highlighted row is the held-out view: no body saw it.' }));
}

function compareActions(run) {
  const n = run.bodies.length;
  return h('div', { class: 'scores-actions' },
    h('button', {
      type: 'button', class: 'toggle', disabled: n < 2 ? true : null,
      onclick: () => { setCompare(true); $('#body-panel').scrollIntoView({ behavior: 'smooth' }); },
    }, 'Show the bodies side by side in 3D'),
    h('span', { text: n < 2 ? (n ? 'This run has one body.' : 'No body: stage 4 not run.') : `${n} bodies: ${run.bodies.map((b) => b.name).join(', ')}` }));
}

function setCompare(on) {
  if (!state.run) return;
  ui.compare = on && state.run.bodies.length > 1;
  renderBodyControls();
  showBodies();
}

// ---------------------------------------------------------------------------------------------------------
// Boot

let runToken = 0;

// Empty every panel that shows a run, so nothing from the previous run stays on the page.
function clearRun(message) {
  state.run = null;
  state.stageKey = null;
  ui.bodyIndex = 0;
  ui.compare = false;
  ui.section = null;
  loadToken += 1; // a body still loading for the previous run must not land
  viewer.releaseAll();
  fill('#run-meta');
  fill('#chain', message ? h('p', { class: 'empty error', role: 'alert', text: message }) : null);
  fill('#checks');
  fill('#viewer-labels');
  fill('#tri-totals');
  fill('#section-list');
  fill('#section-detail');
  fill('#scores');
  renderBodyControls();
  viewerMessage(message ? 'No run loaded.' : 'Loading…');
}

async function loadRun(id) {
  const token = ++runToken;
  $('#run-select').value = id;
  history.replaceState(null, '', '#run=' + encodeURIComponent(id));
  clearRun(null);
  let run;
  try {
    run = await getJson('/api/runs/' + encodeURIComponent(id));
  } catch (err) {
    if (token !== runToken) return;
    clearRun(`Could not load run ${id}: ${err.message || err}`);
    return;
  }
  if (token !== runToken) return;
  try {
    showRun(run);
  } catch (err) {
    // A record the page cannot draw: say so for this run rather than show part of it.
    console.error(err);
    clearRun(`Could not show run ${id}: ${err.message || err}`);
    return;
  }
  await showBodies();
}

function showRun(run) {
  state.run = run;
  const firstWithRecord = run.chain.find((c) => (run.stages.find((s) => s.stage === c.stage) || {}).status !== 'not run');
  state.stageKey = (firstWithRecord || run.chain[0]).stage;
  ui.bodyIndex = 0;
  ui.compare = false;
  ui.section = null;
  viewer.select(null);
  renderRunMeta(run);
  renderChain(run);
  renderChecks();
  renderBodyControls();
  renderScores(run);
}

async function boot() {
  viewer = new Viewer($('#viewer'));
  viewer.onPick = (name) => pickSection(name && name === ui.section ? null : name);
  window.heroLab = { viewer, state, ui };

  for (const b of document.querySelectorAll('[data-lod]')) {
    b.addEventListener('click', () => { ui.lod = b.dataset.lod; renderBodyControls(); showBodies(); });
  }
  for (const b of document.querySelectorAll('[data-shade]')) {
    b.addEventListener('click', () => {
      for (const o of document.querySelectorAll('[data-shade]')) o.classList.toggle('on', o === b);
      viewer.setLit(b.dataset.shade === 'lit');
    });
  }
  for (const b of document.querySelectorAll('[data-cam]')) b.addEventListener('click', () => viewer.setCamera(b.dataset.cam));
  $('#compare-btn').addEventListener('click', () => setCompare(!ui.compare));
  $('#isolate').addEventListener('change', (e) => viewer.setIsolate(e.target.checked));
  $('#walk-btn').addEventListener('click', () => {
    const on = $('#walk-btn').getAttribute('aria-pressed') !== 'true';
    setWalkButton(on);
    ui.preset = on ? null : 'rest';
    ui.sliders = {};
    renderPoseControls();
    viewer.setWalk(on);
  });
  renderPoseControls();

  let data;
  try {
    data = await getJson('/api/runs');
  } catch (err) {
    fill('#chain', h('p', { class: 'empty', text: `Could not reach the lab server: ${err.message}` }));
    return;
  }
  state.runs = data.runs;
  const sel = $('#run-select');
  fill(sel, ...data.runs.map((r) => h('option', { value: r.id, text: r.id })));
  if (!data.runs.length) {
    fill('#chain', h('p', { class: 'empty', text: 'No runs found in the runs folder.' }));
    viewerMessage('No runs.');
    return;
  }
  const fromHash = new URLSearchParams(location.hash.slice(1)).get('run');
  const preferred = data.runs.find((r) => r.id === fromHash)
    || data.runs.find((r) => r.has_body && r.has_scores) || data.runs.find((r) => r.has_body) || data.runs[0];
  sel.value = preferred.id;
  sel.addEventListener('change', () => loadRun(sel.value));
  await loadRun(preferred.id);
}

boot();
