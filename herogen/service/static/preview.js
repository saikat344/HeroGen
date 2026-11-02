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
  function emitPreset(value) { if (preset !== value) { preset = value; presetListener?.(value); } }
  function step(time) {
    if (!state) return;
    const tween = state.tween;
    if (tween) {
      const t = tween.ms <= 0 ? 1 : Math.min(1, (time - tween.start) / tween.ms);
      tween.apply(tween.ease(t));
      if (t >= 1) { state.tween = null; state.lastInput = time; tween.done?.(); }
    }
    // The idle orbit turns the camera directly (not OrbitControls.autoRotate), so input stops it without any coast.
    const idle = stage && orbitWanted && state.model && !state.tween && !reducedMotion() && time - state.lastInput >= IDLE_MS;
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
    state.controls.update(); state.renderer.render(state.scene, state.camera);
  }
  function loop() { state?.renderer.setAnimationLoop(paused ? null : render); }
  function theme() {
    if (!state) return;
    const dark = document.documentElement.dataset.theme === 'dark';
    (Array.isArray(state.grid.material) ? state.grid.material : [state.grid.material]).forEach(m => { m.color.set(dark ? 0x50504b : 0xc7beb0); m.opacity = 0.32; m.transparent = true; });
    render();
  }
  function restore() {
    state?.model?.traverse(node => {
      if (node.userData?.previewOriginal) {
        node.material.dispose(); node.material = node.userData.previewOriginal; delete node.userData.previewOriginal;
      }
    });
  }
  function shading(value) {
    if (!['textured', 'clay', 'wire'].includes(value)) return;
    mode = value; restore();
    if (value !== 'textured') state?.model?.traverse(node => {
      if (!node.isMesh) return;
      node.userData.previewOriginal = node.material;
      node.material = new THREE.MeshStandardMaterial({color: value === 'clay' ? 0xb4aca1 : 0xe29a63, roughness: 0.85, wireframe: value === 'wire'});
    });
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
  function pause() { paused = true; ++request; loop(); }
  function resume() { paused = false; loop(); }
  function dispose() {
    ++request;
    if (!state) return;
    state.renderer.setAnimationLoop(null); restore(); release(state.model);
    for (const [type, fn] of state.inputs) state.canvas.removeEventListener?.(type, fn);
    state.observer.disconnect(); state.controls.dispose(); state.scene.environment?.dispose(); state.grid.geometry.dispose();
    (Array.isArray(state.grid.material) ? state.grid.material : [state.grid.material]).forEach(m => m.dispose());
    if (state.shadow) { state.shadow.geometry.dispose(); state.shadow.material.map?.dispose(); state.shadow.material.dispose(); }
    state.renderer.dispose();
    document.removeEventListener('hg:theme', theme); state = null; preset = null;
  }
  async function show(canvas, url, {preserveCamera = false, canReplace = () => true} = {}) {
    if (state && state.canvas !== canvas) dispose();
    const token = ++request;
    if (!state) {
      const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      // Transparent clear colour: the stage backdrop is the CSS gradient behind the canvas.
      renderer.setClearColor?.(0x000000, 0);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
      const scene = new THREE.Scene();
      // Room lighting: a neutral studio room as image-based light, so every side of the model is lit evenly.
      const pmrem = new THREE.PMREMGenerator(renderer);
      scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; pmrem.dispose();
      scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 0.6));
      // Warm key from the upper left and a cool rim from behind right, matching the stage backdrop.
      const sun = new THREE.DirectionalLight(0xffd2a8, 1.2); sun.position.set(-3, 6, 4); scene.add(sun);
      const rim = new THREE.DirectionalLight(0x9fc4ff, 0.6); rim.position.set(4, 3, -5); scene.add(rim);
      const grid = new THREE.GridHelper(8, 16, 0x8899aa, 0xccd5de); grid.visible = false; scene.add(grid);
      const shadow = stage ? contactShadow() : null; if (shadow) scene.add(shadow);
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      const controls = new OrbitControls(camera, canvas); controls.enableDamping = damping;
      controls.minDistance = 0.1; controls.maxDistance = 100;
      state = {canvas, renderer, scene, camera, controls, grid, shadow, sun, rim, model: null, tween: null,
        lastInput: now(), lastFrame: now(), orbiting: false, frame: null, inputs: []};
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
      const resize = () => {
        if (!state) return;
        const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
        renderer.setSize(w, h, false); camera.aspect = w / h;
        // Lift the framing so the hero stands above the stage dock.
        if (stage) camera.setViewOffset?.(w, h, 0, Math.round(h * 0.1), w, h);
        camera.updateProjectionMatrix(); render();
      };
      state.observer = new ResizeObserver(resize); state.observer.observe(canvas); resize();
      document.addEventListener('hg:theme', theme); theme();
    }
    paused = false; loop();
    if (state.url === url && state.model) return true;
    const keepModel = preserveCamera && Boolean(state.model);
    if (!keepModel) {
      restore();
      if (state.model) { state.scene.remove(state.model); release(state.model); state.model = null; }
      state.grid.visible = false; if (state.shadow) state.shadow.visible = false; state.url = null; render();
    }
    return new Promise((resolve, reject) => new GLTFLoader().load(url, gltf => {
      if (token !== request || !state || keepModel && !canReplace()) { release(gltf.scene); resolve(false); return; }
      restore();
      const first = !state.model;
      if (state.model) { state.scene.remove(state.model); release(state.model); }
      state.model = gltf.scene; state.url = url; state.grid.visible = floor && !stage; state.scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.01);
      const d = radius * (stage ? Math.max(2.35, 1.75 / state.camera.aspect) : Math.max(2.2, 1.5 / state.camera.aspect));
      state.camera.near = radius / 1000; state.camera.far = radius * 100;
      state.camera.updateProjectionMatrix();
      state.frame = {centre: {x: centre.x, y: centre.y, z: centre.z}, distance: d};
      if (!preserveCamera) {
        state.controls.minDistance = radius * 0.35; state.controls.maxDistance = radius * 10;
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
      shading(mode); resolve(true);
    }, undefined, error => {
      if (token === request && state) {
        if (!keepModel) state.renderer.setAnimationLoop(null);
        render(); reject(error);
      } else resolve(false);
    }));
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
    state.sun.intensity = on ? 2.4 : 1.2; state.rim.intensity = on ? 1.1 : 0.6;
    state.renderer.toneMappingExposure = on ? 1.2 : 1.0; render();
  }
  // The dock's Orbit toggle: false stops the idle orbit for good, true allows it again after 4 s idle.
  function orbit(on) {
    if (orbitWanted === Boolean(on)) return;
    orbitWanted = Boolean(on);
    if (state) { state.lastInput = now(); state.orbiting = false; render(); }
  }
  return {show, pause, resume, dispose, shading, lighting, snapshot, view, setView, orbit,
    camera: name => moveTo(name), reset: () => moveTo('front', {reset: true}),
    onView(callback) { viewListener = callback; },
    onPreset(callback) { presetListener = callback; callback?.(preset); }};
}
const preview = createPreview({stage: true});
export const show = (canvas, url, options) => preview.show(canvas, url, options);
export const hide = () => preview.pause();
export const dispose = () => preview.dispose();
export const view = () => preview.view();
export const shading = mode => preview.shading(mode);
export const lighting = on => preview.lighting(on);
export const camera = name => preview.camera(name);
export const reset = () => preview.reset();
export const orbit = on => preview.orbit(on);
export const onPreset = callback => preview.onPreset(callback);
