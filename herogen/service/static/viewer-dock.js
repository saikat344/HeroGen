// Stage dock and viewer tools: camera presets, idle orbit and shading toggles, lighting, reset and fullscreen.
import { track } from './analytics.js';

const viewer = () => import('./preview.js');
// Front/Side/Back camera thumbnails use the stage-2 views; the artwork stands in until those exist.
export const CAMERA_VIEWS = [['front', 'Front', ['front']], ['side', 'Side', ['left', 'right']], ['back', 'Back', ['back']]];
export function cameraImages(images) {
  const all = images || {};
  return CAMERA_VIEWS.map(([key, label, sources]) =>
    ({key, label, url: sources.map(s => all[s]).find(Boolean) || all.cover || all.sketch || null}));
}
export function thumbButton(label, url, data = {}) {
  const button = document.createElement('button'); button.type = 'button'; button.className = 'thumb';
  button.setAttribute('aria-pressed', 'false');
  for (const [key, value] of Object.entries(data)) button.dataset[key] = value;
  const frame = document.createElement('span'); frame.className = 'thumb__img';
  if (url) { const image = document.createElement('img'); image.src = url; image.alt = ''; image.loading = 'lazy'; image.decoding = 'async'; frame.append(image); }
  const text = document.createElement('span'); text.className = 'thumb__label'; text.textContent = label;
  button.append(frame, text);
  return button;
}
const reducedMotion = () => Boolean(globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
function storedOrbit() {
  try { return sessionStorage.getItem('hg:orbit') !== 'off'; } catch { return true; }
}

export function viewerDock($) {
  let model = false, views = '', shade = 'textured', orbitOn = storedOrbit(), expanded = false;
  const dock = $('#viewerDock'), stage = $('#viewer');
  function setPreset(name) {
    for (const b of $('#cameraViews').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.camera === name));
  }
  function syncOrbit() {
    const button = $('#orbitToggle'), blocked = reducedMotion();
    button.setAttribute('aria-pressed', String(orbitOn && !blocked));
    button.setAttribute('aria-disabled', String(blocked));
    button.title = blocked ? 'Idle orbit is off because your device asks for reduced motion' : orbitOn ? 'Stop the idle orbit' : 'Turn on the idle orbit';
    if (model) viewer().then(m => m.orbit(orbitOn && !blocked));
  }
  function syncDock() {
    const images = $('#images').children.length > 0;
    $('#images').hidden = model;
    $('#cameraViews').hidden = !model; $('#workbenchShading').hidden = !model; $('#dockSep').hidden = !model;
    dock.hidden = !model && !images;
    stage.classList.toggle('has-dock', !dock.hidden);
  }
  // Show the 3D or the image tools. Called on every selection change.
  function mode(isModel) {
    const entering = isModel && !model;
    model = isModel;
    $('#resetView').hidden = !model; $('#previewLighting').hidden = !model;
    syncDock();
    if (entering) viewer().then(m => { m.onPreset(setPreset); syncOrbit(); }).catch(() => {});
  }
  // Rebuild the Front/Side/Back thumbnails only when their images change.
  function cameras(images) {
    const items = cameraImages(images), signature = JSON.stringify(items);
    if (signature === views) return;
    views = signature;
    $('#cameraViews').replaceChildren(...items.map(item => {
      const button = thumbButton(item.label, item.url, {camera: item.key});
      button.setAttribute('aria-label', `${item.label} view`);
      button.addEventListener('click', () => viewer().then(m => m.camera(item.key)));
      return button;
    }));
  }
  function images(buttons) { $('#images').replaceChildren(...buttons); syncDock(); }

  $('#orbitToggle').addEventListener('click', () => {
    if (reducedMotion()) return;
    orbitOn = !orbitOn;
    try { sessionStorage.setItem('hg:orbit', orbitOn ? 'on' : 'off'); } catch { /* storage blocked */ }
    syncOrbit();
  });
  // Clay and Wireframe toggle; pressing the active one returns to the textured model.
  for (const button of $('#workbenchShading').querySelectorAll('[data-work-shade]')) button.addEventListener('click', () => {
    shade = shade === button.dataset.workShade ? 'textured' : button.dataset.workShade;
    for (const b of $('#workbenchShading').querySelectorAll('[data-work-shade]')) b.setAttribute('aria-pressed', String(b.dataset.workShade === shade));
    viewer().then(m => m.shading(shade));
    track('shade_toggle', {mode: shade});
  });
  $('#resetView').addEventListener('click', () => viewer().then(m => m.reset()));
  $('#previewLighting').addEventListener('click', () => {
    const on = $('#previewLighting').getAttribute('aria-pressed') !== 'true';
    $('#previewLighting').setAttribute('aria-pressed', String(on)); stage.classList.toggle('is-lit', on);
    viewer().then(m => m.lighting(on));
  });
  // Fullscreen on the stage; an in-page expanded stage where the Fullscreen API is unavailable.
  function setExpanded(on) {
    expanded = on; stage.classList.toggle('viewer--expanded', on); document.body.classList.toggle('has-expanded-viewer', on);
    $('#fullscreenPreview').setAttribute('aria-pressed', String(on));
  }
  $('#fullscreenPreview').addEventListener('click', async () => {
    if (document.fullscreenElement) { await document.exitFullscreen().catch(() => {}); return; }
    if (expanded) { setExpanded(false); return; }
    try { await stage.requestFullscreen(); } catch { setExpanded(true); }
  });
  document.addEventListener('fullscreenchange', () => {
    $('#fullscreenPreview').setAttribute('aria-pressed', String(document.fullscreenElement === stage));
    $('#fullscreenPreview').setAttribute('aria-label', document.fullscreenElement === stage ? 'Exit fullscreen' : 'Fullscreen');
  });
  stage.addEventListener('keydown', event => { if (event.key === 'Escape' && expanded) setExpanded(false); });
  // Arrow keys move between dock buttons; Tab still visits each one.
  dock.addEventListener('keydown', event => {
    const keys = {ArrowRight: 1, ArrowLeft: -1, Home: 'first', End: 'last'};
    if (!(event.key in keys)) return;
    const buttons = [...dock.querySelectorAll('button')].filter(b => b.offsetParent !== null);
    const at = buttons.indexOf(document.activeElement);
    if (at < 0 || !buttons.length) return;
    const move = keys[event.key];
    const next = move === 'first' ? 0 : move === 'last' ? buttons.length - 1 : (at + move + buttons.length) % buttons.length;
    event.preventDefault(); buttons[next].focus(); buttons[next].scrollIntoView({block: 'nearest', inline: 'nearest'});
  });
  // The orbit hint fades after the first interaction with the model.
  $('#preview').addEventListener('pointerdown', () => $('#orbitHint').classList.add('is-faded'), {once: true, passive: true});
  globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').addEventListener?.('change', syncOrbit);
  syncOrbit();
  return {mode, cameras, images};
}
