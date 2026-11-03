// Public gallery detail: the saved 3D result on the lit stage, the artwork beside it, and the facts list.
// Every value from the server reaches the page through textContent or an attribute.
import { show, hide, shading } from '/assets/preview.js';
import { bindOrbitDrag, primeAnalytics, track } from './analytics.js';
const $ = (s) => document.querySelector(s);
let character = null, sketch = false, loading = false, referenceRun = 0, thumbFailed = false;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
// Small images use the 512 px thumbnail when the catalog has one, else the full-size file.
const small = (asset) => asset.thumb_url || asset.url;

function duration(seconds) {
  if (!Number.isFinite(seconds)) return '';
  return seconds < 90 ? `${Math.round(seconds)} seconds` : `about ${Math.round(seconds / 60)} min`;
}

// ---- stage -------------------------------------------------------------------------------------------

function stageError(text) {
  $('#publicStageError').hidden = !text;
  if (text) $('#publicStageErrorText').textContent = text;
}
async function model() {
  if (!character || loading) return;
  loading = true; $('#reloadPublicModel').disabled = true; $('#retryPublicModel').disabled = true;
  stageError(''); $('#publicGhost').hidden = false;
  $('#publicModelStatus').textContent = 'Loading the saved 3D model…';
  try {
    await show($('#publicCanvas'), character.assets.model.url);
    $('#publicShade').hidden = false;
    $('#publicModelStatus').textContent = 'Drag to orbit · Scroll to zoom';
  } catch {
    $('#publicModelStatus').textContent = '';
    stageError('The 3D model could not be loaded. Use Reload 3D to try again.');
  } finally {
    loading = false; $('#publicGhost').hidden = true;
    $('#reloadPublicModel').disabled = false; $('#retryPublicModel').disabled = false;
  }
}

// ---- artwork -----------------------------------------------------------------------------------------

function reference() {
  const asset = character.assets[sketch ? 'sketch' : 'artwork'], img = $('#publicReference'), token = ++referenceRun;
  thumbFailed = false;
  $('#referenceStatus').textContent = '';
  img.alt = `${sketch ? 'Generated starting sketch' : 'Generated character artwork'} for ${character.title}`;
  img.src = small(asset); img.hidden = false;
  // Show the thumbnail at once, then swap in the full-size file when it has loaded.
  if (asset.thumb_url) {
    const full = new Image();
    full.onload = () => { if (token === referenceRun) img.src = asset.url; };
    full.src = asset.url;
  }
  $('#referenceLabel').textContent = sketch ? 'Starting sketch' : 'Character artwork';
  $('#referenceToggle').textContent = sketch ? 'Show artwork' : 'Show starting sketch';
}

// ---- facts and record --------------------------------------------------------------------------------

function fact(label, value, cls) {
  const wrap = el('div', 'fact'), dd = el('dd', cls, value);
  wrap.append(el('dt', '', label), dd);
  return wrap;
}
function renderFacts() {
  const facts = $('#publicFacts'), record = character.refinement;
  facts.replaceChildren(fact('Made from', character.input_kind));
  if (!record) {
    $('#publicFactsStatus').textContent = 'No published refinement measurements are available for this character.';
    $('#publicRecordStatus').textContent = 'No published refinement measurements are available for this character.';
    return;
  }
  facts.append(fact('Refinement time', duration(record.seconds)),
    fact('Views accepted', `${record.accepted_views} of ${record.attempted_views}`),
    fact('Meta models', record.models.join(', ')));
  const entries = [['Meta models', record.models.join(', ')], ['Refinement views accepted', `${record.accepted_views} / ${record.attempted_views}`],
    ['Refinement duration', `${record.seconds} seconds`], ['Refinement cost estimate', `$${record.estimated_cost_usd.toFixed(3)} · stage only, not a provider invoice`],
    ['Refinement record version', character.code_commit.slice(0, 12)]];
  for (const [label, value] of entries) $('#publicRecord').append(el('dt', '', label), el('dd', '', value));
}

// ---- more from the gallery ---------------------------------------------------------------------------

async function loadMore(currentId) {
  try {
    const response = await fetch('/api/gallery', { credentials: 'omit', cache: 'no-store' });
    if (!response.ok) return;
    const items = (await response.json()).items.filter((item) => item.id !== currentId).slice(0, 3);
    if (!items.length) return;
    $('#moreGrid').replaceChildren(...items.map((item) => {
      const li = el('li'), link = el('a', 'g-card card card--interactive');
      link.href = `/gallery/${encodeURIComponent(item.id)}`;
      const thumb = el('span', 'g-card__thumb'), img = el('img');
      img.alt = ''; img.loading = 'lazy'; img.decoding = 'async'; img.src = small(item.assets.artwork);
      thumb.append(img);
      const body = el('span', 'g-card__body');
      body.append(el('h3', '', item.title), el('p', 'small muted', item.description));
      link.append(thumb, body); li.append(link);
      return li;
    }));
    $('#moreHeroes').hidden = false;
  } catch { /* the rest of the page does not depend on it */ }
}

// ---- states ------------------------------------------------------------------------------------------

function state({ title, text, icon: name, retry }) {
  $('#galleryDetail').hidden = true; $('#galleryState').hidden = false;
  document.title = `${title} · HeroGen`;
  $('#galleryStateTitle').textContent = title; $('#galleryStateText').textContent = text;
  $('#galleryStateIcon').setAttribute('href', `/assets/icons/sprite.svg#i-${name}`);
  $('#galleryStateRetry').hidden = !retry;
  $('#main').focus({ preventScroll: true });
}

async function boot() {
  $('#galleryDetail').hidden = false; $('#galleryState').hidden = true;
  const id = decodeURIComponent(location.pathname.split('/').filter(Boolean).at(-1) || '');
  let response;
  try { response = await fetch(`/api/gallery/${encodeURIComponent(id)}`, { credentials: 'omit', cache: 'no-store' }); }
  catch {
    return state({ title: 'The gallery is resting.', text: 'The public gallery could not be reached. Check your connection and try again.', icon: 'circle-dashed', retry: true });
  }
  if (response.status === 404) {
    return state({ title: 'This hero is not in the gallery.', text: 'It may have been renamed or removed. The gallery has other saved heroes to explore.', icon: 'circle-dashed', retry: false });
  }
  if (!response.ok) {
    return state({ title: 'The gallery is resting.', text: 'The public gallery is temporarily unavailable. Try again in a moment.', icon: 'circle-dashed', retry: true });
  }
  character = await response.json();
  $('#publicTitle').textContent = character.title; $('#publicDescription').textContent = character.description;
  document.title = `${character.title} · HeroGen`;
  $('#publicCanvas').setAttribute('aria-label', `3D model of ${character.title}. Drag to orbit and scroll to zoom.`);
  renderFacts(); reference(); loadMore(character.id);
  await model();
}

$('#reloadPublicModel').addEventListener('click', model);
$('#retryPublicModel').addEventListener('click', model);
$('#referenceToggle').addEventListener('click', () => { if (character) { sketch = !sketch; reference(); } });
$('#publicReference').addEventListener('error', () => {
  const asset = character?.assets[sketch ? 'sketch' : 'artwork'], img = $('#publicReference');
  if (asset?.thumb_url && !thumbFailed) { thumbFailed = true; img.src = asset.url; return; }   // the thumbnail failed: try the full file
  img.hidden = true; $('#referenceStatus').textContent = 'This reference image could not be loaded. Toggle the reference to retry.';
});
$('#publicShade').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-shade]');
  if (!button) return;
  shading(button.dataset.shade);
  for (const other of $('#publicShade').querySelectorAll('button')) other.setAttribute('aria-pressed', String(other === button));
});
if (document.fullscreenEnabled) {
  $('#publicFullscreen').hidden = false;
  $('#publicFullscreen').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('.gd-stage').requestFullscreen(); }
    catch { $('#publicModelStatus').textContent = 'Fullscreen is unavailable in this browser.'; }
  });
}
$('#galleryStateRetry').addEventListener('click', () => boot().catch(fatal));
window.addEventListener('pagehide', hide);

function fatal(error) {
  state({ title: 'The gallery is resting.', text: error?.message || 'This public hero could not be opened. Try again in a moment.', icon: 'circle-dashed', retry: true });
}
primeAnalytics();
track('page_view', { page: 'gallery_detail' });
bindOrbitDrag($('#publicCanvas'), 'gallery_detail');
boot().catch(fatal);
