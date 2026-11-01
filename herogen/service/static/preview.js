// Independently owned viewers keep the landing page and workspace canvases isolated.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

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
export function createPreview({damping = true} = {}) {
  let state = null, request = 0, mode = 'textured', paused = false, viewListener = null, syncing = false;
  function render() { if (state) { state.controls.update(); state.renderer.render(state.scene, state.camera); } }
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
  function pause() { paused = true; ++request; loop(); }
  function resume() { paused = false; loop(); }
  function dispose() {
    ++request;
    if (!state) return;
    state.renderer.setAnimationLoop(null); restore(); release(state.model);
    state.observer.disconnect(); state.controls.dispose(); state.grid.geometry.dispose();
    (Array.isArray(state.grid.material) ? state.grid.material : [state.grid.material]).forEach(m => m.dispose()); state.renderer.dispose();
    document.removeEventListener('hg:theme', theme); state = null;
  }
  async function show(canvas, url, {preserveCamera = false, canReplace = () => true} = {}) {
    if (state && state.canvas !== canvas) dispose();
    const token = ++request;
    if (!state) {
      const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      const scene = new THREE.Scene();
      scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 2.2));
      const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(3, 6, 4); scene.add(sun);
      const grid = new THREE.GridHelper(8, 16, 0x8899aa, 0xccd5de); grid.visible = false; scene.add(grid);
      const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
      const controls = new OrbitControls(camera, canvas); controls.enableDamping = damping;
      controls.minDistance = 0.1; controls.maxDistance = 100;
      state = {canvas, renderer, scene, camera, controls, grid, sun, model: null};
      controls.addEventListener?.('change', () => { if (!syncing) viewListener?.(view()); });
      const resize = () => {
        if (!state) return;
        const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
        renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); render();
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
      state.grid.visible = false; state.url = null; render();
    }
    return new Promise((resolve, reject) => new GLTFLoader().load(url, gltf => {
      if (token !== request || !state || keepModel && !canReplace()) { release(gltf.scene); resolve(false); return; }
      restore();
      if (state.model) { state.scene.remove(state.model); release(state.model); }
      state.model = gltf.scene; state.url = url; state.grid.visible = true; state.scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.01);
      const d = radius * Math.max(2.2, 1.5 / state.camera.aspect);
      state.camera.near = radius / 1000; state.camera.far = radius * 100;
      state.camera.updateProjectionMatrix();
      if (!preserveCamera) {
        state.controls.minDistance = radius * 0.35; state.controls.maxDistance = radius * 10;
        state.camera.position.set(centre.x + d * 0.45, centre.y + d * 0.15, centre.z + d);
        state.controls.target.copy(centre);
      }
      state.grid.position.y = box.min.y;
      state.grid.scale.setScalar(radius / 2); shading(mode); resolve(true);
    }, undefined, error => {
      if (token === request && state) {
        if (!keepModel) state.renderer.setAnimationLoop(null);
        render(); reject(error);
      } else resolve(false);
    }));
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
  function lighting(on) { if (state) { state.sun.intensity = on ? 3.2 : 1.6; render(); } }
  return {show, pause, resume, dispose, shading, lighting, snapshot, view, setView,
    onView(callback) { viewListener = callback; }};
}
const preview = createPreview();
export const show = (canvas, url, options) => preview.show(canvas, url, options);
export const hide = () => preview.pause();
export const dispose = () => preview.dispose();
export const view = () => preview.view();
export const shading = mode => preview.shading(mode);

export const lighting = on => preview.lighting(on);
