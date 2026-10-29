// A small orbit preview of a body GLB (three.js 0.160.0, pinned with integrity in the page's import map).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let state = null;

export function show(canvas, url) {
  if (!state) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, 6, 4);
    scene.add(sun);
    scene.add(new THREE.GridHelper(8, 16, 0x8899aa, 0xccd5de));
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    state = { renderer, scene, camera, controls, model: null };
    const resize = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(canvas);
    resize();
    renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
  }
  new GLTFLoader().load(url, (gltf) => {
    if (state.model) state.scene.remove(state.model);
    state.model = gltf.scene;
    state.scene.add(gltf.scene);
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
    const d = Math.max(size.x, size.y, size.z) * 2.2;
    state.camera.position.set(centre.x + d * 0.6, centre.y + d * 0.25, centre.z + d);
    state.controls.target.copy(centre);
  });
}
