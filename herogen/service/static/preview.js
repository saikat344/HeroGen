// Independently owned viewers keep the landing page and workspace canvases isolated.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// Camera presets for the stage dock: azimuth in degrees around the model, 0 = facing the hero.
export const PRESETS = {front: 0, side: 90, back: 180};
const ELEVATION = 8 * Math.PI / 180;          // a slightly raised eye line for every preset
const IDLE_MS = 4000;                          // idle time before the auto-orbit starts
const ORBIT_RADIANS_PER_MS = 2 * Math.PI / 40000; // one idle revolution per 40 s
const REVEAL_MS = 1200;
const ENV_LIGHT = 0.35;                       // share of the studio room's light on the model's materials
const SWAP_MS = 260;                           // cross-fade between the static model and the animated body
const CLIP_NAMES = ['idle', 'walk', 'wave', 'jump'];
const CONVERGE_MS = 1600;                      // the figure of dots gathering onto the first model
const FADE_MS = 400;                           // the same hand-off with reduced motion: a fade, no movement
const DOT_SIZE = 0.0042;                       // a dot's world size, in figure heights
const DOT_COLOUR = 0xf3efe8;                   // --stage-text: the stage is dark in both themes
const FIGURE_ROOM = 1.3, FIGURE_LIFT = 0.1;   // the figure's framing: further out, and raised by part of its height
const STAGE_LIFT = 0.1;                        // the stage framing sits this share of the height above the dock
const SWEEP_MS = 4500;                         // one pass of the repaint light down the model while the texture refines
const SWEEP_COLOUR = '1.0, 0.36, 0.12';        // --ember-300 in linear light
const easeOut = t => 1 - Math.pow(1 - t, 4);
const easeInOut = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const now = () => globalThis.performance?.now?.() ?? Date.now();
function reducedMotion() {
  return Boolean(globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
}
// Durations come from tokens.css so reduced motion (which collapses them) applies here too.
function duration(name, fallback) {
  try {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const value = parseFloat(raw);
    if (Number.isFinite(value)) return raw.endsWith('ms') ? value : value * 1000;
  } catch { /* outside a browser */ }
  return fallback;
}
function release(model) {
  const geometries = new Set(), materials = new Set(), textures = new Set();
  model?.traverse(node => {
    if (node.geometry) geometries.add(node.geometry);
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (!material) continue;
      materials.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    }
  });
  textures.forEach(v => v.dispose()); materials.forEach(v => v.dispose()); geometries.forEach(v => v.dispose());
}
// A soft elliptical contact shadow drawn once into a small canvas texture.
function contactShadow() {
  const canvas = globalThis.document?.createElement?.('canvas');
  const context = canvas?.getContext?.('2d');
  if (!context) return null;
  canvas.width = canvas.height = 128;
  const gradient = context.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(0,0,0,0.55)'); gradient.addColorStop(0.55, 'rgba(0,0,0,0.22)'); gradient.addColorStop(1, 'rgba(0,0,0,0)');
  context.fillStyle = gradient; context.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(canvas);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({map: texture, transparent: true, depthWrite: false, toneMapped: false}));
  mesh.rotation.x = -Math.PI / 2; mesh.renderOrder = -1; mesh.visible = false;
  return mesh;
}

/**
 * Options (defaults keep the plain framing that thumbnail and comparison viewers use):
 * - floor: show the floor grid under a model (on by default).
 * - stage: the lit workspace stage. Front framing lifted above the dock, a contact shadow instead of the grid,
 *   an eased reveal on first load, an idle auto-orbit after 4 s without input, eased camera presets.
 */
export function createPreview({damping = true, floor = true, stage = false} = {}) {
  let state = null, request = 0, mode = 'textured', paused = false, viewListener = null, syncing = false;
  let orbitWanted = true, presetListener = null, preset = null;
  // Animation: `anim` is the animated body (loaded once per body URL), `body` the in-flight load of it.
  let anim = null, body = null, animRequest = 0, animListener = null, animState = {clip: null, status: 'idle'};
  let splitRequest = 0;
  function emitPreset(value) { if (preset !== value) { preset = value; presetListener?.(value); } }
  function emitAnim(clip, status) { animState = {clip, status}; animListener?.(clip, status); }
  function step(time) {
    if (!state) return;
    stepAnimation(time);
    const tween = state.tween;
    if (tween) {
      const t = tween.ms <= 0 ? 1 : Math.min(1, (time - tween.start) / tween.ms);
      tween.apply(tween.ease(t));
      if (t >= 1) { state.tween = null; state.lastInput = time; tween.done?.(); }
    }
    // The idle orbit turns the camera directly (not OrbitControls.autoRotate), so input stops it without any coast.
    // A playing animation holds the camera still so the hero keeps facing the way the person chose.
    stepCloud(time); stepSweep(time);
    const idle = stage && orbitWanted && (state.model || state.cloud) && !state.tween && !anim?.active && !reducedMotion() && time - state.lastInput >= IDLE_MS;
    if (idle && !state.orbiting) emitPreset(null);
    if (idle && state.orbiting) {
      const o = orbitOf();
      place(state.controls.target, o.azimuth + Math.min(100, time - state.lastFrame) * ORBIT_RADIANS_PER_MS, o.elevation, o.distance);
    }
    state.orbiting = Boolean(idle); state.lastFrame = time;
  }
  function render() {
    if (!state) return;
    step(now());
    state.controls.update();
    if (state.split) renderSplit(); else state.renderer.render(state.scene, state.camera);
  }
  // ---- the split stage: SAM 3D's raw shape (`state.split.model`) and the model on show, one camera for both ----
  // The camera's aspect and lift for a viewport of w x h (the whole canvas, or one panel of it).
  function frameCamera(w, h, lift = true) {
    state.camera.aspect = w / h;
    if (stage) state.camera.setViewOffset?.(w, h, 0, lift ? Math.round(h * STAGE_LIFT) : 0, w, h);
    state.camera.updateProjectionMatrix();
  }
  // The two panels in canvas pixels from the top left: SAM 3D first. Stacked, only the lower panel sits over the dock.
  function panes() {
    const {w, h} = state.size, s = state.split;
    if (s.axis === 'rows') {
      // Stacked panels share the height above the dock, so neither model stands behind it.
      const room = Math.max(2, h - s.inset), top = Math.max(1, Math.min(room - 1, Math.round(room * s.ratio)));
      return [{x: 0, y: 0, w, h: top, lift: false, raw: true}, {x: 0, y: top, w, h: room - top, lift: false, raw: false}];
    }
    const left = Math.max(1, Math.min(w - 1, Math.round(w * s.ratio)));
    return [{x: 0, y: 0, w: left, h, lift: true, raw: true}, {x: left, y: 0, w: w - left, h, lift: true, raw: false}];
  }
  function renderSplit() {
    const r = state.renderer, {h} = state.size, raw = state.split.model, hero = state.model;
    r.setScissorTest?.(false); r.setViewport?.(0, 0, state.size.w, h); r.clear?.();   // the strip under stacked panels
    r.setScissorTest?.(true);
    for (const p of panes()) {
      const y = h - p.y - p.h;   // WebGL counts from the bottom
      r.setViewport?.(p.x, y, p.w, p.h); r.setScissor?.(p.x, y, p.w, p.h);
      frameCamera(p.w, p.h, p.lift);
      if (raw) raw.visible = p.raw;
      if (hero) hero.visible = !p.raw;
      r.render(state.scene, state.camera);
    }
    r.setScissorTest?.(false);
    if (hero) hero.visible = true;
    if (raw) raw.visible = false;
  }
  // The aspect the framing is fitted to: a half panel while split, so the hero fits either panel at 50/50.
  function viewAspect() {
    const {w, h} = state.size || {w: 1, h: 1};
    if (!state.split) return w / h;
    return state.split.axis === 'rows' ? w / (Math.max(2, h - state.split.inset) / 2) : (w / 2) / h;
  }
  const frameDistance = radius => radius * (stage ? Math.max(2.35, 1.75 / viewAspect()) : Math.max(2.2, 1.5 / viewAspect()));
  // A layout change (split on or off, stacked or side by side) keeps the angle and the person's zoom, refitted.
  function refit() {
    if (!state?.frame?.radius) return;
    const was = state.frame.distance, next = frameDistance(state.frame.radius);
    state.frame.distance = next;
    if (!(was > 0) || state.cloud) return;
    const o = orbitOf();
    place(state.controls.target, o.azimuth, o.elevation, o.distance * next / was);
  }
  function layout() {
    const {w, h} = state.size;
    state.renderer.setSize(w, h, false);
    frameCamera(w, h);
    render();
  }
  // SAM 3D's shape stands where HeroGen's model stands: same height, same feet line, same centre.
  function alignRaw() {
    const raw = state?.split?.model, hero = state?.model;
    if (!raw) return;
    raw.position.set(0, 0, 0); raw.scale.setScalar(1); raw.updateMatrixWorld?.(true);
    if (!hero) return;
    const to = new THREE.Box3().setFromObject(hero), from = new THREE.Box3().setFromObject(raw);
    const h = from.max.y - from.min.y;
    if (to.isEmpty?.() || from.isEmpty?.() || !(h > 0)) return;
    const k = (to.max.y - to.min.y) / h;
    if (!(k > 0)) return;
    raw.scale.setScalar(k);
    raw.position.set((to.min.x + to.max.x) / 2 - k * (from.min.x + from.max.x) / 2, to.min.y - k * from.min.y,
      (to.min.z + to.max.z) / 2 - k * (from.min.z + from.max.z) / 2);
  }
  function dropSplit() {
    if (!state?.split) return;
    ++splitRequest;
    restore();   // while the raw shape is still listed, so its own materials come back before it is freed
    const raw = state.split.model; state.split = null;
    if (raw) { state.scene.remove(raw); release(raw); }
    if (state.model) state.model.visible = true;
    shading(mode);
  }
  /** Split the stage (`info` {left, waiting, ratio, axis}) or join it again (null). `left` is SAM 3D's raw shape;
   *  the other panel shows the model from show(). `waiting`: the texture is still being repainted, so that panel
   *  shows the shape in clay under a slow light sweep. Resolves true once the raw shape is on show. */
  function split(canvas, info) {
    if (!info) {
      if (!state?.split || state.canvas !== canvas) return Promise.resolve(true);
      dropSplit(); refit(); layout();
      return Promise.resolve(true);
    }
    ensure(canvas);
    paused = false; loop();
    const was = state.split, axis = info.axis === 'rows' ? 'rows' : 'columns';
    const ratio = Number.isFinite(info.ratio) ? Math.min(0.8, Math.max(0.2, info.ratio)) : 0.5, waiting = Boolean(info.waiting);
    const inset = axis === 'rows' && info.inset > 0 ? info.inset : 0;
    if (!was) { stopAnimation(true); state.split = {url: null, model: null, ratio, axis, waiting, inset}; }
    const s = state.split, relaid = !was || was.axis !== axis || was.inset !== inset, repaint = !was || was.waiting !== waiting;
    s.ratio = ratio; s.axis = axis; s.waiting = waiting; s.inset = inset;
    if (relaid) { layout(); refit(); }
    if (repaint) shading(mode);
    render();
    if (s.url === info.left) return Promise.resolve(Boolean(s.model));
    const token = ++splitRequest, url = info.left;
    s.url = url;
    return new Promise(resolve => new GLTFLoader().load(url, gltf => {
      if (token !== splitRequest || state?.split !== s) { release(gltf.scene); resolve(false); return; }
      restore();
      if (s.model) { state.scene.remove(s.model); release(s.model); }
      gltf.scene.traverse(o => { for (const m of [].concat(o.material || [])) if ('envMapIntensity' in m) m.envMapIntensity = ENV_LIGHT; });
      s.model = gltf.scene; s.model.visible = false; state.scene.add(s.model);
      alignRaw(); shading(mode); render();
      resolve(true);
    }, undefined, () => { resolve(false); }));
  }
  // The repaint light: a warm line that runs down the model, leaving a faint glow where it has passed.
  function sweepMaterial(material) {
    const u = state.sweep;
    material.customProgramCacheKey = () => 'hg-repaint-sweep';
    material.onBeforeCompile = shader => {
      Object.assign(shader.uniforms, {hgPhase: u.phase, hgLow: u.low, hgHigh: u.high});
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying float hgY;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nhgY = (modelMatrix * vec4(transformed, 1.0)).y;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nuniform float hgPhase, hgLow, hgHigh;\nvarying float hgY;')
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          float hgD = (hgY - hgLow) / max(1e-4, hgHigh - hgLow) - hgPhase;
          float hgGlow = 1.5 * exp(-hgD * hgD * 700.0) + 0.22 * step(0.0, hgD) * exp(-hgD * 5.0);
          totalEmissiveRadiance += vec3(${SWEEP_COLOUR}) * hgGlow;`);
    };
    return material;
  }
  function stepSweep(time) {
    const u = state.sweep;
    if (!state.split?.waiting) return;
    if (reducedMotion()) { u.phase.value = -10; return; }
    // A steady scan from above the head to below the feet over most of the cycle, then a short rest out of sight.
    const t = (time % SWEEP_MS) / SWEEP_MS;
    u.phase.value = t < 0.85 ? 1.08 - 1.16 * t / 0.85 : -10;
  }
  function loop() { state?.renderer.setAnimationLoop(paused ? null : render); }
  function theme() {
    if (!state) return;
    const dark = document.documentElement.dataset.theme === 'dark';
    (Array.isArray(state.grid.material) ? state.grid.material : [state.grid.material]).forEach(m => { m.color.set(dark ? 0x50504b : 0xc7beb0); m.opacity = 0.32; m.transparent = true; });
    render();
  }
  const shown = () => [state?.model, anim?.body, state?.split?.model].filter(Boolean);
  function restore() {
    finishFade(); finishConverge();
    for (const model of shown()) model.traverse(node => {
      if (node.userData?.previewOriginal) {
        node.material.dispose(); node.material = node.userData.previewOriginal; delete node.userData.previewOriginal;
      }
    });
  }
  function shading(value) {
    if (!['textured', 'clay', 'wire'].includes(value)) return;
    mode = value; restore();
    // While the texture refines, HeroGen's panel shows the shape in clay (or wire) under the repaint light.
    const swept = state?.split?.waiting ? state.model : null;
    if (swept) {
      const box = new THREE.Box3().setFromObject(swept);
      state.sweep.low.value = box.min.y; state.sweep.high.value = box.max.y;
    }
    for (const model of shown()) {
      if (value === 'textured' && model !== swept) continue;
      model.traverse(node => {
        if (!node.isMesh) return;
        node.userData.previewOriginal = node.material;
        // The swept clay is a shade darker, so the warm repaint light reads against it.
        const clay = model === swept ? 0x8d867d : 0xb4aca1;
        const material = new THREE.MeshStandardMaterial({color: value === 'wire' ? 0xe29a63 : clay, roughness: 0.85, wireframe: value === 'wire'});
        node.material = model === swept ? sweepMaterial(material) : material;
      });
    }
    render();
  }
  // Camera position from an orbit around the target: azimuth 0 looks at the hero's front (+Z).
  function place(target, azimuth, elevation, distance) {
    state.camera.position.set(target.x + distance * Math.sin(azimuth) * Math.cos(elevation),
      target.y + distance * Math.sin(elevation), target.z + distance * Math.cos(azimuth) * Math.cos(elevation));
  }
  function orbitOf() {
    const p = state.camera.position, t = state.controls.target;
    const x = p.x - t.x, y = p.y - t.y, z = p.z - t.z, distance = Math.hypot(x, y, z) || 1;
    return {azimuth: Math.atan2(x, z), elevation: Math.asin(Math.max(-1, Math.min(1, y / distance))), distance};
  }
  function animate(from, to, ms, ease, done) {
    const target = state.controls.target;
    const apply = k => {
      target.set(from.target.x + (to.target.x - from.target.x) * k, from.target.y + (to.target.y - from.target.y) * k,
        from.target.z + (to.target.z - from.target.z) * k);
      place(target, from.azimuth + (to.azimuth - from.azimuth) * k, from.elevation + (to.elevation - from.elevation) * k,
        from.distance + (to.distance - from.distance) * k);
    };
    state.orbiting = false;
    state.tween = {start: now(), ms, ease, apply, done};
    if (ms <= 0) { apply(1); state.tween = null; state.lastInput = now(); done?.(); }
    render();
  }
  function moveTo(name, {reset = false} = {}) {
    if (!state?.model || !(name in PRESETS)) return false;
    const from = {...orbitOf(), target: {...state.controls.target}};
    let azimuth = PRESETS[name] * Math.PI / 180;
    // Turn the short way round.
    while (azimuth - from.azimuth > Math.PI) azimuth -= 2 * Math.PI;
    while (azimuth - from.azimuth < -Math.PI) azimuth += 2 * Math.PI;
    const frame = state.frame;
    const to = {azimuth, elevation: ELEVATION, distance: reset ? frame.distance : from.distance,
      target: reset ? frame.centre : from.target};
    emitPreset(null);
    animate(from, to, reducedMotion() ? 0 : duration('--dur-slow', 360), easeInOut, () => emitPreset(name));
    return true;
  }
  function markInput() {
    if (!state) return;
    state.lastInput = now();
    state.tween = null; state.orbiting = false;
  }
  function pause() { paused = true; ++request; stopAnimation(true); dropCloud(); loop(); }
  function resume() { paused = false; loop(); }
  function dispose() {
    ++request;
    if (!state) return;
    state.renderer.setAnimationLoop(null); dropSplit(); restore(); dropCloud(); dropAnimation(); release(state.model);
    for (const [type, fn] of state.inputs) state.canvas.removeEventListener?.(type, fn);
    state.observer.disconnect(); state.controls.dispose(); state.scene.environment?.dispose(); state.grid.geometry.dispose();
    (Array.isArray(state.grid.material) ? state.grid.material : [state.grid.material]).forEach(m => m.dispose());
    if (state.shadow) { state.shadow.geometry.dispose(); state.shadow.material.map?.dispose(); state.shadow.material.dispose(); }
    state.renderer.dispose();
    document.removeEventListener('hg:theme', theme); state = null; preset = null;
  }
  function ensure(canvas) {
    if (state && state.canvas !== canvas) dispose();
    if (!state) {
      const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      // Transparent clear colour: the stage backdrop is the CSS gradient behind the canvas.
      renderer.setClearColor?.(0x000000, 0);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
      const scene = new THREE.Scene();
      // Room lighting: a neutral studio room as image-based light, so every side of the model is lit evenly.
      const pmrem = new THREE.PMREMGenerator(renderer);
      scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; pmrem.dispose();
      scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 0.6));
      // Warm key from the upper left and a cool rim from behind right, matching the stage backdrop.
      const sun = new THREE.DirectionalLight(0xffd2a8, 1.2); sun.position.set(-3, 6, 4); scene.add(sun);
      const rim = new THREE.DirectionalLight(0x9fc4ff, 0.6); rim.position.set(4, 3, -5); scene.add(rim);
      // A white light from straight overhead, a little in front, so dark suits still read on the dark stage.
      const top = new THREE.DirectionalLight(0xffffff, 2.2); top.position.set(0, 10, 4); scene.add(top);
      const grid = new THREE.GridHelper(8, 16, 0x8899aa, 0xccd5de); grid.visible = false; scene.add(grid);
      const shadow = stage ? contactShadow() : null; if (shadow) scene.add(shadow);
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      const controls = new OrbitControls(camera, canvas); controls.enableDamping = damping;
      controls.minDistance = 0.1; controls.maxDistance = 100;
      state = {canvas, renderer, scene, camera, controls, grid, shadow, sun, rim, top, model: null, tween: null,
        lastInput: now(), lastFrame: now(), orbiting: false, frame: null, inputs: [], split: null, size: {w: 1, h: 1},
        sweep: {phase: {value: -10}, low: {value: 0}, high: {value: 1}}};
      controls.addEventListener?.('change', () => {
        if (!syncing) viewListener?.(view());
        // A drag that turns the camera away from a preset clears the dock selection; zooming keeps it.
        if (state && preset && !state.tween) {
          const off = Math.abs(((orbitOf().azimuth - PRESETS[preset] * Math.PI / 180) % (2 * Math.PI) + 3 * Math.PI) % (2 * Math.PI) - Math.PI);
          if (off > 0.035) emitPreset(null);
        }
      });
      controls.addEventListener?.('start', markInput);
      controls.addEventListener?.('end', markInput);
      for (const type of ['pointerdown', 'wheel', 'keydown']) {
        const fn = () => markInput();
        canvas.addEventListener?.(type, fn, {passive: true}); state.inputs.push([type, fn]);
      }
      // Lift the framing so the hero stands above the stage dock (frameCamera; each panel of a split has its own).
      const resize = () => {
        if (!state) return;
        state.size = {w: Math.max(1, canvas.clientWidth), h: Math.max(1, canvas.clientHeight)};
        layout();
      };
      state.observer = new ResizeObserver(resize); state.observer.observe(canvas); resize();
      document.addEventListener('hg:theme', theme); theme();
    }
  }
  async function show(canvas, url, {preserveCamera = false, canReplace = () => true} = {}) {
    if (state && state.canvas !== canvas) dispose();
    const token = ++request;
    ensure(canvas);
    paused = false; loop();
    if (state.url === url && state.model) return true;
    const keepModel = preserveCamera && Boolean(state.model);
    if (!keepModel) {
      restore(); stopAnimation(true);
      if (state.model) { state.scene.remove(state.model); release(state.model); state.model = null; }
      state.grid.visible = false; if (state.shadow) state.shadow.visible = false; state.url = null; render();
    }
    return new Promise((resolve, reject) => new GLTFLoader().load(url, gltf => {
      if (token !== request || !state || keepModel && !canReplace()) { release(gltf.scene); resolve(false); return; }
      restore(); stopAnimation(true);
      const first = !state.model;
      if (state.model) { state.scene.remove(state.model); release(state.model); }
      // The studio room lights every side evenly; at full strength it adds up with the lamps and bleaches light,
      // pastel textures toward white under the tone mapping, so the model takes a third of it.
      gltf.scene.traverse(o => { for (const m of [].concat(o.material || [])) if ('envMapIntensity' in m) m.envMapIntensity = ENV_LIGHT; });
      state.model = gltf.scene; state.url = url; state.grid.visible = floor && !stage; state.scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
      // The figure of dots hands over to the first model: the view keeps its angle and its framing.
      const cloud = !keepModel && state.cloud, before = cloud && state.frame ? orbitOf() : null, was = state.frame;
      const {radius, d} = frameBox(size, centre, Boolean(before) || !preserveCamera);
      if (before) {
        const k = d / was.distance, t = state.controls.target;
        t.set(centre.x + (t.x - was.centre.x) * k, centre.y + (t.y - was.centre.y) * k, centre.z + (t.z - was.centre.z) * k);
        place(t, before.azimuth, before.elevation, before.distance * k);
        state.lastInput = now();
      } else if (keepModel && was?.radius) {
        // Another mesh of the same hero in its own units (the Roblox body is in studs): the same view of it.
        const k = radius / was.radius, t = state.controls.target, p = state.camera.position;
        const moved = Math.hypot(centre.x - was.centre.x, centre.y - was.centre.y, centre.z - was.centre.z) > radius * 1e-3;
        if (Math.abs(k - 1) > 1e-3 || moved) {
          const off = [(p.x - t.x) * k, (p.y - t.y) * k, (p.z - t.z) * k];
          t.set(centre.x + (t.x - was.centre.x) * k, centre.y + (t.y - was.centre.y) * k, centre.z + (t.z - was.centre.z) * k);
          p.set(t.x + off[0], t.y + off[1], t.z + off[2]);
          state.controls.minDistance = radius * 0.35; state.controls.maxDistance = radius * 10;
          state.tween = null;
        }
      } else if (!preserveCamera) {
        state.controls.target.copy(centre);
        if (!stage) state.camera.position.set(centre.x + d * 0.45, centre.y + d * 0.15, centre.z + d);
        else reveal(centre, d, first);
      }
      state.grid.position.y = box.min.y;
      state.grid.scale.setScalar(radius / 2);
      if (state.shadow) {
        state.shadow.visible = true; state.shadow.position.set(centre.x, box.min.y + radius * 0.002, centre.z);
        state.shadow.scale.set(Math.max(size.x, size.z) * 1.6, Math.max(size.x, size.z) * 1.1, 1);
      }
      alignRaw(); shading(mode);
      if (cloud) converge(cloud, gltf.scene);
      resolve(true);
    }, undefined, error => {
      if (token === request && state) {
        if (!keepModel && !state.cloud) state.renderer.setAnimationLoop(null);
        render(); reject(error);
      } else resolve(false);
    }));
  }
  // Near and far planes, zoom limits and the stage framing for an object of `size` at `centre`.
  function frameBox(size, centre, limits = true) {
    const radius = Math.max(size.x, size.y, size.z, 0.01);
    const d = frameDistance(radius);
    state.camera.near = radius / 1000; state.camera.far = radius * 100;
    state.camera.updateProjectionMatrix();
    if (limits) { state.controls.minDistance = radius * 0.35; state.controls.maxDistance = radius * 10; }
    state.frame = {centre: {x: centre.x, y: centre.y, z: centre.z}, distance: d, radius};
    return {radius, d};
  }
  // ---- the figure of dots shown while the 3D shape is built (shape-build.js samples it) ---------------------------
  const CLOUD_VERTEX = `
    attribute vec3 target;
    attribute float seed;
    uniform float time, converge, size, scale, drift;
    varying float vAlpha;
    void main() {
      float s = seed * 6.28318;
      vec3 wobble = drift * 0.006 * vec3(sin(time * 0.7 + s * 3.0), sin(time * 0.53 + s * 5.0), cos(time * 0.61 + s * 7.0));
      float k = clamp(converge * 1.6 - fract(seed * 7.13) * 0.6, 0.0, 1.0);
      k = k * k * (3.0 - 2.0 * k);
      vec4 mv = modelViewMatrix * vec4(mix(position + wobble, target, k), 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = max(1.0, size * (0.55 + 0.9 * fract(seed * 13.37)) * scale / max(0.001, -mv.z));
      float twinkle = drift > 0.5 ? 0.5 + 0.5 * sin(time * (0.8 + 2.2 * fract(seed * 3.7)) + s * 11.0) : 0.6;
      vAlpha = 0.3 + 0.7 * twinkle;
    }`;
  const CLOUD_FRAGMENT = `
    uniform vec3 colour;
    uniform float opacity;
    varying float vAlpha;
    void main() {
      float d = length(gl_PointCoord - 0.5);
      if (d > 0.5) discard;
      gl_FragColor = vec4(colour, smoothstep(0.5, 0.1, d) * vAlpha * opacity);
    }`;
  function dropCloud() {
    if (!state?.cloud) return;
    const c = state.cloud, g = state.converge; state.cloud = null; state.converge = null;
    // A hand-off cut short (another tab, another hero) leaves the model as it would have ended: fully shown.
    for (const [m, opacity, transparent] of g?.saved || []) { m.opacity = opacity; m.transparent = transparent; m.needsUpdate = true; }
    c.group.removeFromParent(); c.points.geometry.dispose(); c.material.dispose();
  }
  // An empty stage on `canvas` (no model, no figure) while the next figure is prepared, so the last hero never shows.
  // Triangles in the model on screen (the still model, not the animated body).
  function triangles() {
    let n = 0;
    state?.model?.traverse(o => {
      const g = o.isMesh && o.geometry;
      if (g) n += Math.floor((g.index ? g.index.count : g.attributes.position?.count || 0) / 3);
    });
    return n;
  }
  function clear(canvas) {
    if (!state || state.canvas !== canvas) return;
    ++request; dropSplit(); dropCloud(); restore(); stopAnimation(true);
    if (state.model) { state.scene.remove(state.model); release(state.model); state.model = null; state.url = null; }
    state.grid.visible = false; if (state.shadow) state.shadow.visible = false; layout();
  }
  /** Show `figure` ({key, positions, seeds} from shape-build.js) on `canvas` in place of any model: small dots that
   *  drift and twinkle, framed like a model and turning slowly until the person moves the view. */
  function building(canvas, figure) {
    ensure(canvas);
    paused = false; loop();
    if (state.cloud?.key === figure?.key && state.cloud) return true;
    const count = figure?.positions?.length / 3;
    if (!(count >= 1) || figure.seeds?.length !== count) return false;
    ++request; if (state.split) { dropSplit(); layout(); } dropCloud(); restore(); stopAnimation(true);
    if (state.model) { state.scene.remove(state.model); release(state.model); state.model = null; state.url = null; }
    state.grid.visible = false; if (state.shadow) state.shadow.visible = false;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(figure.positions, 3));
    geometry.setAttribute('target', new THREE.BufferAttribute(new Float32Array(figure.positions), 3));
    geometry.setAttribute('seed', new THREE.BufferAttribute(figure.seeds, 1));
    geometry.computeBoundingBox();
    const material = new THREE.ShaderMaterial({vertexShader: CLOUD_VERTEX, fragmentShader: CLOUD_FRAGMENT, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: {time: {value: 0}, converge: {value: 0}, size: {value: DOT_SIZE}, scale: {value: 1}, drift: {value: 1},
        opacity: {value: 0}, colour: {value: new THREE.Color(DOT_COLOUR)}}});
    const points = new THREE.Points(geometry, material); points.frustumCulled = false;
    const group = new THREE.Group(); group.name = 'hg-shape-figure'; group.add(points); state.scene.add(group);
    state.cloud = {key: figure.key, group, points, material, box: geometry.boundingBox.clone(), born: now(), time: 0};
    const size = state.cloud.box.getSize(new THREE.Vector3()), centre = state.cloud.box.getCenter(new THREE.Vector3());
    // A little smaller and higher than a model, so the title and progress under it stay clear of the figure.
    const {d} = frameBox(size, centre);
    state.controls.target.set(centre.x, centre.y - size.y * FIGURE_LIFT, centre.z);
    place(state.controls.target, 20 * Math.PI / 180, ELEVATION, d * FIGURE_ROOM);
    state.tween = null; state.lastInput = now() - IDLE_MS;   // it turns from the start
    emitPreset(null); render();
    return true;
  }
  function stepCloud(time) {
    const c = state.cloud;
    if (!c) return;
    const still = reducedMotion(), u = c.material.uniforms;
    if (!still) c.time += Math.min(100, Math.max(0, time - (c.last ?? time))) / 1000;
    c.last = time;
    u.time.value = c.time; u.drift.value = still ? 0 : 1;
    u.scale.value = (state.canvas.height || state.canvas.clientHeight || 1) / (2 * Math.tan(state.camera.fov * Math.PI / 360 || 0.3));
    if (!state.converge) { u.opacity.value = still ? 0.9 : 0.9 * Math.min(1, (time - c.born) / 600); return; }
    const g = state.converge, k = g.ms <= 0 ? 1 : Math.min(1, (time - g.start) / g.ms);
    const ramp = (a, b) => easeInOut(Math.min(1, Math.max(0, (k - a) / (b - a))));
    u.converge.value = g.still ? 0 : ramp(0, 0.8);
    u.opacity.value = 0.9 * (1 - ramp(g.still ? 0 : 0.55, 1));
    for (const [m, opacity] of g.saved) m.opacity = opacity * ramp(g.still ? 0 : 0.3, g.still ? 1 : 0.9);
    if (k >= 1) finishConverge();
  }
  // The dots gather onto points of the model's surface while the model fades in, then the dots go.
  function converge(cloud, model) {
    const still = reducedMotion(), box = new THREE.Box3().setFromObject(model);
    const to = box.getSize(new THREE.Vector3()), from = cloud.box.getSize(new THREE.Vector3());
    const at = box.getCenter(new THREE.Vector3()), mid = cloud.box.getCenter(new THREE.Vector3());
    // Lay the figure over the model's box first, so the dots start where the model will be.
    const k = new THREE.Vector3(to.x / (from.x || 1), to.y / (from.y || 1), to.z / (from.z || 1));
    cloud.group.scale.copy(k);
    cloud.group.position.set(at.x - mid.x * k.x, at.y - mid.y * k.y, at.z - mid.z * k.z);
    cloud.group.updateMatrixWorld(true);
    const surface = [];
    model.updateMatrixWorld?.(true);
    model.traverse(node => { if (node.isMesh && node.geometry?.attributes?.position) surface.push(node); });
    const target = cloud.points.geometry.attributes.target, n = target.count, v = new THREE.Vector3();
    const inverse = cloud.group.matrixWorld.clone().invert();
    for (let i = 0; i < n && surface.length; i++) {
      const mesh = surface[i % surface.length], p = mesh.geometry.attributes.position;
      v.fromBufferAttribute(p, Math.floor(Math.random() * p.count)).applyMatrix4(mesh.matrixWorld).applyMatrix4(inverse);
      target.setXYZ(i, v.x, v.y, v.z);
    }
    target.needsUpdate = true;
    const saved = materialsOf(model).map(m => [m, m.opacity, m.transparent]);
    for (const [m] of saved) { m.transparent = true; m.opacity = 0; m.needsUpdate = true; }
    state.converge = {start: now(), ms: still ? duration('--dur-reveal', FADE_MS) : CONVERGE_MS, saved, still, input: state.lastInput};
  }
  function finishConverge() {
    if (!state?.converge) return;
    const g = state.converge; state.converge = null;
    for (const [m, opacity, transparent] of g.saved) { m.opacity = opacity; m.transparent = transparent; m.needsUpdate = true; }
    dropCloud();
    // Untouched since the hand-off: settle on the model's own framing, from the same side.
    if (state.lastInput === g.input && state.frame && !state.tween) {
      const from = {...orbitOf(), target: {...state.controls.target}};
      animate(from, {...from, distance: state.frame.distance, target: state.frame.centre}, reducedMotion() ? 0 : duration('--dur-slow', 360) * 2, easeInOut);
    }
  }
  // First model: ease in from 15 degrees off-axis and 8% further out. Later models snap to the front framing.
  function reveal(centre, d, first) {
    const to = {azimuth: 0, elevation: ELEVATION, distance: d, target: {x: centre.x, y: centre.y, z: centre.z}};
    const motion = first && !reducedMotion();
    if (motion) {
      state.canvas.classList?.remove('reveal'); void state.canvas.offsetWidth; state.canvas.classList?.add('reveal');
    }
    emitPreset(null);
    animate({...to, azimuth: 15 * Math.PI / 180, distance: d * 1.08}, to, motion ? REVEAL_MS : 0, easeOut, () => emitPreset('front'));
  }
  // ---- animation: the Roblox-ready body, re-parented at its R15 joints and posed by animate.js ----------------
  let fade = null;
  function materialsOf(model) {
    const list = new Set();
    model.traverse(node => {
      if (node.isMesh) for (const m of Array.isArray(node.material) ? node.material : [node.material]) if (m) list.add(m);
    });
    return [...list];
  }
  // Fade `to` in over `from` (the two stand in the same place), then hide `from`.
  function crossFade(from, to, done) {
    finishFade();
    to.visible = true;
    const ms = reducedMotion() ? 0 : SWAP_MS;
    if (!from || from === to || ms <= 0) { if (from && from !== to) from.visible = false; done?.(); return; }
    const saved = materialsOf(to).map(m => [m, m.transparent, m.opacity, m.polygonOffset, m.polygonOffsetFactor, m.polygonOffsetUnits]);
    for (const [m] of saved) {
      m.transparent = true; m.opacity = 0; m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -4; m.needsUpdate = true;
    }
    fade = {from, to, start: now(), ms, saved, done};
  }
  function finishFade() {
    if (!fade) return;
    const f = fade; fade = null;
    for (const [m, transparent, opacity, offset, factor, units] of f.saved) {
      m.transparent = transparent; m.opacity = opacity; m.polygonOffset = offset; m.polygonOffsetFactor = factor; m.polygonOffsetUnits = units;
      m.needsUpdate = true;
    }
    f.from.visible = false; f.done?.();
  }
  function shadowRest() {
    if (!state?.shadow || !anim?.shadowScale) return;
    state.shadow.scale.copy(anim.shadowScale); state.shadow.material.opacity = 1; anim.shadowScale = null;
  }
  function stepAnimation(time) {
    if (fade) {
      const k = Math.min(1, (time - fade.start) / fade.ms);
      for (const [m, , opacity] of fade.saved) m.opacity = opacity * easeInOut(k);
      if (k >= 1) finishFade();
    }
    if (!anim?.active) return;
    let lift = 0;
    try {
      const p = anim.player.sample(time);
      anim.rig.apply(p); lift = p.lift;
    } catch { stopAnimation(true); emitAnim(null, 'error'); return; }
    if (state.shadow && anim.shadowScale) {
      const k = 1 - 0.45 * Math.min(1, lift / anim.rig.maxLift);
      state.shadow.scale.set(anim.shadowScale.x * k, anim.shadowScale.y * k, 1); state.shadow.material.opacity = 0.4 + 0.6 * k;
    }
    // Stopping: once the pose has eased back to rest, fade the static model back in.
    if (anim.stopping && !anim.player.busy(time)) {
      anim.stopping = false; anim.active = false; shadowRest();
      crossFade(anim.body, state.model);
    }
  }
  // The body fills the static model's box: same height, same feet line, same centre.
  function fit(a) {
    const box = new THREE.Box3().setFromObject(state.model), rest = a.rig.restBox;
    const height = rest.max.y - rest.min.y;
    if (box.isEmpty() || !(height > 0)) return;
    const k = (box.max.y - box.min.y) / height;
    a.body.scale.setScalar(k);
    a.body.position.set((box.min.x + box.max.x) / 2 - k * (rest.min.x + rest.max.x) / 2, box.min.y - k * rest.min.y,
      (box.min.z + box.max.z) / 2 - k * (rest.min.z + rest.max.z) / 2);
  }
  // Load (once per URL) the body GLB and build its joint tree.
  function loadBody(url, parts) {
    if (body?.url === url) return body.promise;
    const promise = Promise.all([import('./animate.js'),
      new Promise((resolve, reject) => new GLTFLoader().load(url, resolve, undefined, reject))]).then(([module, gltf]) => {
      try {
        const rig = module.createAnimator(THREE, gltf.scene, parts), wrap = new THREE.Group();
        wrap.name = 'hg-animated-body'; wrap.visible = false; wrap.add(gltf.scene);
        return {url, body: wrap, rig, player: null, active: false, stopping: false, shadowScale: null};
      } catch (error) { release(gltf.scene); throw error; }
    });
    body = {url, promise};
    promise.catch(() => { if (body?.promise === promise) body = null; });
    return promise;
  }
  function dropAnimation() {
    finishFade();
    if (anim) { shadowRest(); anim.body.removeFromParent(); release(anim.body); anim = null; }
    const pending = body; body = null;
    pending?.promise.then(b => { if (b !== anim) release(b.body); }, () => {});
  }
  // Back to the static model: eased (the pose returns to rest, then a cross-fade), or at once.
  function stopAnimation(instant = false) {
    ++animRequest;
    if (anim && (anim.active || anim.stopping || fade)) {
      if (instant) {
        finishFade();
        anim.active = false; anim.stopping = false; shadowRest();
        anim.player = anim.rig.player(); anim.rig.apply(anim.player.sample(0));
        anim.body.visible = false; if (state?.model) state.model.visible = true;
      } else if (anim.active) {
        anim.player.play('rest', now()); anim.stopping = true;
      }
    }
    if (animState.clip || animState.status !== 'idle') emitAnim(null, 'idle');
  }
  /** Play an animation clip ('idle', 'walk', 'wave', 'jump') on the hero's body, or stop with anything else.
   *  `info` is the job's `animation` entry ({model, parts}). Resolves true when the clip is playing. */
  async function playAnimation(name, info) {
    if (!state?.model) return false;
    if (!CLIP_NAMES.includes(name)) { stopAnimation(); return true; }
    if (state.split) return false;   // the split stage shows two models; animation is a single-view tool
    const url = info?.model;
    if (typeof url !== 'string' || !Array.isArray(info.parts)) return false;
    const token = ++animRequest, model = state.model;
    if (!anim || anim.url !== url) {
      emitAnim(name, 'loading');
      let built;
      try { built = await loadBody(url, info.parts); }
      catch { if (token === animRequest) emitAnim(null, 'error'); return false; }
      if (token !== animRequest || !state || state.model !== model) return false;
      if (anim !== built) { dropAnimation(); anim = built; body = {url, promise: Promise.resolve(built)}; }
      state.scene.add(anim.body); shading(mode);
    }
    if (!anim.active) {
      fit(anim);
      anim.player = anim.rig.player(); anim.active = true; anim.stopping = false;
      if (state.shadow) { anim.shadowScale = state.shadow.scale.clone(); }
      crossFade(model, anim.body);
    }
    anim.stopping = false;
    anim.player.play(name, now());
    state.lastInput = now(); state.orbiting = false;
    emitAnim(name, 'playing'); render();
    return true;
  }
  function prefetchAnimation(info) {
    if (state && typeof info?.model === 'string' && Array.isArray(info.parts)) loadBody(info.model, info.parts).catch(() => {});
  }
  function snapshot() { render(); return state?.canvas.toDataURL('image/png'); }
  function view() {
    return state ? {position: state.camera.position.toArray(), target: state.controls.target.toArray(), zoom: state.camera.zoom} : null;
  }
  function setView(value) {
    if (!state || !value) return;
    syncing = true;
    state.camera.position.set(...value.position); state.controls.target.set(...value.target);
    state.camera.zoom = value.zoom; state.camera.updateProjectionMatrix();
    render(); syncing = false;
  }
  function lighting(on) {
    if (!state) return;
    state.sun.intensity = on ? 2.4 : 1.2; state.rim.intensity = on ? 1.1 : 0.6; state.top.intensity = on ? 3.2 : 2.2;
    state.renderer.toneMappingExposure = on ? 1.3 : 1.1; render();
  }
  // The dock's Orbit toggle: false stops the idle orbit for good, true allows it again after 4 s idle.
  function orbit(on) {
    if (orbitWanted === Boolean(on)) return;
    orbitWanted = Boolean(on);
    if (state) { state.lastInput = now(); state.orbiting = false; render(); }
  }
  return {show, split, building, clear, triangles, pause, resume, dispose, shading, lighting, snapshot, view, setView, orbit,
    animate: playAnimation, prefetchAnimation,
    onAnimation(callback) { animListener = callback; callback?.(animState.clip, animState.status); },
    camera: name => moveTo(name), reset: () => moveTo('front', {reset: true}),
    onView(callback) { viewListener = callback; },
    onPreset(callback) { presetListener = callback; callback?.(preset); }};
}
const preview = createPreview({stage: true});
export const show = (canvas, url, options) => preview.show(canvas, url, options);
export const split = (canvas, info) => preview.split(canvas, info);
export const building = (canvas, figure) => preview.building(canvas, figure);
export const clear = canvas => preview.clear(canvas);
export const triangles = () => preview.triangles();
export const hide = () => preview.pause();
export const dispose = () => preview.dispose();
export const view = () => preview.view();
export const shading = mode => preview.shading(mode);
export const lighting = on => preview.lighting(on);
export const camera = name => preview.camera(name);
export const reset = () => preview.reset();
export const orbit = on => preview.orbit(on);
export const onPreset = callback => preview.onPreset(callback);
export const animate = (name, info) => preview.animate(name, info);
export const prefetchAnimation = info => preview.prefetchAnimation(info);
export const onAnimation = callback => preview.onAnimation(callback);
