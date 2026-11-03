// HeroGen service page: start a guest hero from a sketch or description, follow its
// stages, download the GLB and the Roblox bundle. Every value from the server is shown with textContent.
import { developerMode } from '/assets/developer.js';
import { entryPage } from '/assets/entry.js';
import { landingGallery } from '/assets/landing.js';
let landing;
import { buildLog } from '/assets/buildlog.js';
import { generationView, generationState, TERMINAL } from '/assets/generation.js';
import { descriptionDraft, rewordLabel, rewordNotice, usd } from '/assets/buildlog-copy.js';
import { variationsView } from '/assets/variations.js';
import { renderDownloads } from '/assets/downloads.js';
import { finishedNote, heroesView, kindName } from '/assets/heroes.js';
import { bindOrbitDrag, knownCreateResult, setCsrf, track, trackOnce } from './analytics.js';

const $ = (s) => document.querySelector(s);
const S = { csrf: null, session: null, current: null, timer: null, revision: 0, job: null, shown: null, lostSince: 0 };
const LOST_BANNER_MS = 30000;
let entry;
let screen = 'entry';   // 'entry' (landing), 'heroes' (#heroes), or 'generation' (a hero's workspace)
function setScreen(next) {
  const enteringWorkbench = next === 'generation' && screen !== 'generation';
  screen = next; show('#entry', next === 'entry'); show('#detail', next === 'generation');
  $('#appBar').dataset.screen = next;   // the bottom bar shows the hero's status and actions on the workbench only
  heroes.setVisible(next === 'heroes');
  if (next === 'heroes') show('#app');
  document.title = next === 'heroes' ? 'Your heroes · HeroGen' : 'HeroGen';
  landing?.setVisible(next === 'entry');
  $('#app').classList.toggle('generation-screen', next === 'generation');
  document.body.classList.toggle('workspace-screen', next === 'generation');   // the fixed, non-scrolling workspace frame
  $('#app').classList.toggle('heroes-screen', next === 'heroes');
  if (enteringWorkbench) track('page_view', { page: 'workbench' });
}
const heroes = heroesView($, { api, open: (id) => openJob(id, true), create: () => goCreate(),
  refresh: () => refreshList().catch((e) => heroes.fail(e)) });
const liveLog = buildLog($, api);
const generation = generationView($, (url, options) => preview(url, options), (info) => buildShape(info));
const variations = variationsView($, { api, select: openJob, key: idemKey,
  textureNotes: () => S.session?.limits?.texture_notes === true,
  created: j => { ++S.revision; followJob(j, true); refreshList().catch(e => banner(e.message)); },
  changeIdea: async j => {
    try {
      let file = null;
      if (j.kind === 'sketch') {
        const response = await fetch(j.images.sketch, {credentials: 'same-origin'});
        if (!response.ok) throw new Error('The saved input could not be loaded. Try again.');
        file = new File([await response.blob()], 'character.png', {type: 'image/png'});
      }
      returnToEntry(); history.pushState(null, '', '/'); entry.setFile(file);
      $('#prompt').value = descriptionDraft(j); $('#prompt').dispatchEvent(new Event('input'));
      $('#name').value = j.name || ''; $('#profile').value = j.profile; $('#go').focus();
    } catch (e) { banner(e.message); }
  }
});
function leaveWorkbench() {
  ++S.revision; clearTimeout(S.timer); S.current = null; S.job = null; S.shown = null; S.lostSince = 0;
  if ($('#developerMode').getAttribute('aria-pressed') === 'true') $('#developerMode').click();
  developer.update(null); generation.reset(); liveLog.reset(); variations.reset(); show('#jobLoading', false);
}
function returnToEntry() { leaveWorkbench(); setScreen('entry'); }
// #heroes: the card grid. Without a session there is nothing to list, so it shows the empty state.
async function showHeroes({ push = true } = {}) {
  leaveWorkbench(); setScreen('heroes');
  if (push && location.hash !== '#heroes') history.pushState(null, '', '/#heroes');
  window.scrollTo({ top: 0, behavior: 'auto' }); $('#heroesH').focus({ preventScroll: true });
  if (!S.session?.signed_in) { heroes.update([]); return; }
  heroes.loading();
  try { await refreshList(); } catch (e) { heroes.fail(e); }
}
function goCreate() {
  history.pushState(null, '', '/'); returnToEntry();
  $('#prompt').focus();
}
function signIn() {
  if (location.pathname === '/admin/signin') { show('#signin'); $('#signinDialog').showModal(); }
  else {
    $('#accessIntro').textContent = 'No account needed. Extra heroes wait for a free worker.';
    $('#guestGo').disabled = false;
    $('#guestDialog').showModal();
  }
}
function show(id, on = true) { $(id).hidden = !on; }
// The strip under the header: network, session and page-level problems. `title` says what happened, `text` what is
// safe, and an optional `action` {label, run} what to do. Danger announces itself; the other kinds are polite.
let bannerAction = null;
function banner(text, { title = 'Something went wrong', kind = 'danger', action = null } = {}) {
  const b = $('#banner');
  b.hidden = !text; bannerAction = text ? action : null;
  if (!text) return;
  b.className = `banner banner--${kind} banner--global`;
  b.setAttribute('role', kind === 'danger' ? 'alert' : 'status');
  $('#bannerTitle').textContent = title; $('#bannerText').textContent = text;
  b.querySelector('use').setAttribute('href', `/assets/icons/sprite.svg#i-${kind === 'info' ? 'info' : 'triangle-alert'}`);
  $('#bannerAction').hidden = !action; if (action) $('#bannerAction').textContent = action.label;
}
$('#bannerClose').addEventListener('click', () => banner(''));
$('#bannerAction').addEventListener('click', () => bannerAction?.run());
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'text') e.textContent = v; else if (k === 'cls') e.className = v; else e.setAttribute(k, v);
  }
  for (const k of kids) if (k) e.append(k);
  return e;
}
function fmtTime(t) { return t ? new Date(t * 1000).toLocaleString() : ''; }
function fmtSecs(v) { return typeof v === 'number' ? (v < 90 ? `${v.toFixed(1)} s` : `${(v / 60).toFixed(1)} min`) : ''; }
function idemKey() { const a = new Uint8Array(18); crypto.getRandomValues(a); return Array.from(a, (x) => x.toString(16).padStart(2, '0')).join(''); }

async function api(path, { method = 'GET', json, form, headers = {} } = {}) {
  const opt = { method, headers: { ...headers }, credentials: 'same-origin', cache: 'no-store' };
  if (method !== 'GET') opt.headers['X-CSRF-Token'] = S.csrf || '';
  if (json !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(json); }
  if (form) opt.body = form;
  const r = await fetch(path, opt);
  let data = null;
  try { data = await r.json(); } catch { data = null; }
  if (!r.ok) {
    if (r.status === 401 && S.session?.signed_in) expireAccess();
    const msg = (data && data.error && data.error.message) || `Request failed (${r.status}).`;
    const err = new Error(msg); err.status = r.status; err.code = data && data.error && data.error.code;
    err.retryAfter = r.headers.get('Retry-After'); throw err;
  }
  return data;
}

async function loadSession() {
  S.session = await api('/api/session');
  S.csrf = S.session.csrf;
  setCsrf(S.csrf);
  entry?.generation(S.session.generation);
  return S.session;
}

$('#heroesNav').addEventListener('click', () => showHeroes());
$('#workspaceNav').addEventListener('click', () => $('#backToIdeas').click());

const developer = developerMode($('#developerMode'), $('#developerPanel'),
  (id) => api(`/api/runs/r1/${encodeURIComponent(id)}`));

// ---- sign in -------------------------------------------------------------------------------------------

function initSignin() {
  $('#signinForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const msg = $('#signinMsg');
    msg.textContent = 'Sending…';
    try {
      const r = await api('/api/auth/request', { method: 'POST', json: { email: $('#email').value } });
      msg.textContent = r.message;
    } catch (e) { msg.textContent = e.message; }
  });
}

async function initVerify() {
  $('#signinDialog').close();
  const token = new URLSearchParams(location.hash.slice(1)).get('t');
  history.replaceState(null, '', '/admin/signin');   // the token leaves the address bar and history
  show('#verify');
  if (!token) { $('#verifyWho').textContent = 'This sign-in link is incomplete. Ask for a new one.'; show('#verifyForm', false); return; }
  let info;
  try { info = await api('/api/auth/link', { method: 'POST', json: { token } }); }
  catch (e) { $('#verifyWho').textContent = e.message; show('#verifyForm', false); return; }
  $('#verifyWho').textContent = `Sign in as ${info.email}.`;
  show('#verifyForm');
  $('#verifyMsg').textContent = '';
  $('#verifyForm').onsubmit = async (ev) => {
    ev.preventDefault();
    try {
      const r = await api('/api/auth/verify', { method: 'POST', json: { token } });
      S.csrf = r.csrf;
      setCsrf(S.csrf);
      location.replace('/admin');
    } catch (e) { $('#verifyMsg').textContent = e.message; }
  };
}

// ---- jobs ----------------------------------------------------------------------------------------------

async function refreshList() {
  const { jobs } = await api('/api/jobs');
  heroes.update(jobs);
  return jobs;
}

function renderJob(j) {
  developer.update(j);
  show('#detail', screen === 'generation');
  $('#detailH').textContent = j.name || kindName(j);
  $('#state').textContent = j.state;
  const bits = [`started ${fmtTime(j.created_at)}`, `profile ${j.profile}`];
  if (Number.isFinite(j.seconds)) bits.push(`recorded duration ${fmtSecs(j.seconds)}`);
  // The cost is tallied when the run ends; while it runs the Build log shows the running total instead.
  if (TERMINAL.has(j.state) && Number.isFinite(j.cost_usd)) bits.push(`${j.cost_is_estimate === false ? 'recorded' : 'estimated'} cost ${usd(j.cost_usd)}`);
  $('#detailSub').textContent = bits.join(' · ') + (j.prompt ? ` · “${j.prompt}”` : '');
  S.job = j; S.shown = j.id;
  jobBanner(j); rewordBanner(j);
  generation.update(j); liveLog.update(j);
  variations.update(j);
  // The downloads lead once there is something to download; Retry and Refine then step down to secondary.
  const offer = renderDownloads($, j, { track });
  const lead = !!(offer?.roblox?.url || offer?.glb);
  // On a finished hero the downloads take the subtitle's place under the title.
  $('#generationSubtitle').hidden = j.state === 'succeeded' && !!offer;
  $('#refineVersion').classList.toggle('btn--primary', !lead);
  $('#refineVersion').classList.toggle('btn--secondary', lead);
  for (const id of ['#refineVersion', '#retryVersion']) $(id).classList.toggle('btn--l', !lead);
  const g = j.downloads.glb, r = j.downloads.roblox;
  show('#studio', r.available);
  $('#readme').textContent = j.roblox_readme || '';
  $('#viewerLink').href = j.viewer;
  show('#cancelJob', j.state === 'queued' || j.state === 'running');
  $('#viewerLink').hidden = !g.available;
  renderStatus();
}

// The job-level banner under the progress bar. A failed hero says what stopped, what is safe and offers Retry; a
// cancelled one says it stopped; a requeued one says it will resume. Variation failures and refused inputs have their
// own banners (#variationFailure, #inputRefusal), so none is shown for them here.
function jobBanner(j) {
  const box = $('#detailErr');
  const kind = j.parent_job || j.input_rejected ? null : ['failed', 'rejected'].includes(j.state) ? 'danger'
    : j.state === 'canceled' ? 'info' : j.state === 'queued' && j.error ? 'warning' : null;
  box.hidden = !kind;
  if (!kind) return;
  const label = j.error?.stage_label || j.stage_label || '';
  const where = label ? label[0].toUpperCase() + label.slice(1) : 'This step';
  const stopped = generationState(j).steps.find((x) => x.state === 'canceled')?.title.toLowerCase();
  const saved = j.models?.length || j.downloads?.glb?.available ? 'Your 3D model is saved.'
    : Object.keys(j.images || {}).length ? 'Your saved images are kept.' : '';
  const copy = { danger: [`${where} ${j.state === 'rejected' ? 'needs attention' : 'did not finish'}`, j.error?.message || 'The step could not be completed.', saved],
    info: [stopped ? `Cancelled during ${stopped}` : 'Cancelled before it finished', 'You stopped this hero before it was done.', saved],
    warning: ['Waiting for a free worker', j.error?.message || '', 'Your hero is saved and will start again on its own.'] }[kind];
  box.className = `banner banner--${kind}`; box.setAttribute('role', kind === 'danger' ? 'alert' : 'status');
  box.querySelector('use').setAttribute('href', `/assets/icons/sprite.svg#i-${kind === 'info' ? 'info' : 'triangle-alert'}`);
  $('#detailErrTitle').textContent = copy[0]; $('#detailErrText').textContent = copy[1]; $('#detailErrSafe').textContent = copy[2];
  $('#detailErrText').hidden = !copy[1]; $('#detailErrSafe').hidden = !copy[2];
  $('#detailRetry').hidden = kind === 'warning';
}

// The description reworded after a content-policy refusal: a one-line chip in the toolbar that opens what it now
// says and, when the run stopped on it, a button that opens the description box filled with the rewrite. A dismissed
// note stays dismissed for that hero in this tab.
const REWORD_DISMISSED = 'herogen.rewordDismissed';
const rewordDismissed = new Set((() => { try { return JSON.parse(sessionStorage.getItem(REWORD_DISMISSED) || '[]'); } catch { return []; } })());
function rewordOpen(open) { $('#rewordToggle').setAttribute('aria-expanded', String(open)); $('#rewordDetail').hidden = !open; }
function rewordBanner(j) {
  const note = rewordNotice(j), box = $('#rewordNote'), key = note ? `${j.id}:${j.reword.status}` : '';
  if (box.dataset.key !== key) rewordOpen(false);
  box.dataset.key = key;
  box.hidden = !note || rewordDismissed.has(key);
  if (box.hidden) return;
  box.className = `banner banner--${note.tone}`;
  box.querySelector('use').setAttribute('href', `/assets/icons/sprite.svg#i-${note.tone === 'info' ? 'info' : 'triangle-alert'}`);
  $('#rewordLabel').textContent = rewordLabel(j);
  $('#rewordNoteTitle').textContent = note.title; $('#rewordNoteText').textContent = note.text;
  $('#rewordEdit').hidden = note.tone === 'info' || !TERMINAL.has(j.state);
}
$('#rewordEdit').addEventListener('click', () => { if (S.job) $('#changeIdea').click(); });
$('#rewordToggle').addEventListener('click', () => rewordOpen($('#rewordDetail').hidden));
$('#rewordClose').addEventListener('click', () => {
  const box = $('#rewordNote');
  rewordDismissed.add(box.dataset.key); rewordOpen(false); box.hidden = true;
  try { sessionStorage.setItem(REWORD_DISMISSED, JSON.stringify([...rewordDismissed])); } catch { /* private mode: this page only */ }
  $('#detail').focus({ preventScroll: true });
});
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || $('#rewordDetail').hidden) return;
  rewordOpen(false); $('#rewordToggle').focus();
});
document.addEventListener('click', (e) => { if (!$('#rewordDetail').hidden && !$('#rewordNote').contains(e.target)) rewordOpen(false); });

// ---- status bar ------------------------------------------------------------------------------------------

function fmtClock(v) {
  const s = Math.max(0, Math.round(v)), m = Math.floor(s / 60), h = Math.floor(m / 60);
  return h ? `${h} h ${String(m % 60).padStart(2, '0')} min` : m ? `${m} min ${String(s % 60).padStart(2, '0')} s` : `${s} s`;
}
function elapsedText(j) {
  const now = Date.now() / 1000;
  if (j.state === 'queued') return j.created_at ? `Queued · ${fmtClock(now - j.created_at)} so far` : '';
  const start = j.started_at, end = j.finished_at;   // run time counts from the start, not from when it was queued
  if (!start) return '';
  if (j.state === 'running' || !end) return polls(j) ? `Running · ${fmtClock(now - start)} elapsed` : '';
  if (j.state === 'succeeded') return finishedNote(j, fmtClock, start);
  const took = fmtClock(end - start);
  return { failed: `Stopped after ${took}`, rejected: `Stopped after ${took}`,
    canceled: `Cancelled after ${took}`, stopped: `Saved after ${took}` }[j.state] || '';
}
// Green when everything is saved, ember while results are still arriving, red when the page lost the service.
function renderStatus() {
  const j = S.job, bar = $('#saveStatus');
  const mode = S.lostSince ? 'offline' : j && (j.state === 'running' || j.export_state === 'running') ? 'saving' : 'saved';
  if (bar.dataset.mode !== mode) {
    const copy = { offline: 'Not saved, reconnecting', saving: 'Saving each result as it arrives', saved: 'Progress saved' };
    bar.classList.toggle('is-saving', mode === 'saving'); bar.classList.toggle('is-offline', mode === 'offline');
    bar.classList.toggle('is-just-saved', mode === 'saved' && bar.dataset.mode === 'saving');
    $('#savedText').textContent = copy[mode]; bar.dataset.mode = mode;
  }
  $('#elapsed').textContent = j ? elapsedText(j) : '';
}
setInterval(() => { if (S.job && !document.hidden && screen === 'generation') renderStatus(); }, 1000);
// The page lost (or regained) the service while polling: red status at once, a banner once it lasts.
function connection(ok) {
  if (ok) S.lostSince = 0;   // the caller clears the banner with every good read
  else {
    S.lostSince ||= Date.now();
    if (Date.now() - S.lostSince >= LOST_BANNER_MS) banner('HeroGen is not reachable right now. Your hero is saved, and this page keeps trying.',
      { title: 'Connection lost', kind: 'warning', action: { label: 'Try now', run: () => S.current && openJob(S.current) } });
  }
  renderStatus();
}

// ---- More menu and delete confirmation -------------------------------------------------------------------

function initMenu() {
  const button = $('#moreMenu'), list = $('#moreList');
  const items = () => [...list.querySelectorAll('[role=menuitem]')].filter((i) => !i.hidden && !i.disabled);
  const close = (refocus) => { list.hidden = true; button.setAttribute('aria-expanded', 'false'); if (refocus) button.focus(); };
  const open = (last) => {
    $('#moreRule').hidden = $('#cancelJob').hidden; list.hidden = false; button.setAttribute('aria-expanded', 'true');
    (last ? items().at(-1) : items()[0])?.focus();
  };
  button.addEventListener('click', () => (list.hidden ? open() : close(true)));
  button.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); if (list.hidden) open(e.key === 'ArrowUp'); }
  });
  list.addEventListener('keydown', (e) => {
    const all = items(), at = all.indexOf(document.activeElement);
    if (e.key === 'Escape') { e.preventDefault(); close(true); }
    else if (e.key === 'Tab') close(false);
    else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
      e.preventDefault();
      all[e.key === 'Home' ? 0 : e.key === 'End' ? all.length - 1 : (at + (e.key === 'ArrowDown' ? 1 : -1) + all.length) % all.length]?.focus();
    }
  });
  list.addEventListener('click', (e) => { if (e.target.closest('[role=menuitem]')) close(false); });
  document.addEventListener('click', (e) => { if (!list.hidden && !e.target.closest('.more')) close(false); });
  $('#deleteDialog').addEventListener('close', () => { if ($('#moreMenu').isConnected && $('#appBar').dataset.screen === 'generation') $('#moreMenu').focus(); });
  $('#closeDelete').addEventListener('click', () => $('#deleteDialog').close());
  $('#cancelDelete').addEventListener('click', () => $('#deleteDialog').close());
  // The Retry on the failure banner presses the real Retry button, so it is allowed exactly when that one is.
  const retry = $('#retryVersion'), sync = () => { $('#detailRetry').disabled = retry.disabled; };
  new MutationObserver(sync).observe(retry, { attributes: true, attributeFilter: ['disabled'] }); sync();
  $('#detailRetry').addEventListener('click', () => retry.click());
  // Compare explains itself while it is off.
  const compare = $('#compareVersions'), hint = () => { compare.title = compare.disabled ? 'Needs two finished versions' : 'Compare two finished versions'; };
  new MutationObserver(hint).observe(compare, { attributes: true, attributeFilter: ['disabled'] }); hint();
}

let previewUrl = null, previewRequest = 0, previewModule = null;
async function preview(url, options) {
  if (url && screen !== 'generation') return false;
  if (!url) { ++previewRequest; previewUrl = null; previewModule?.hide(); show('#orbitHint', false); show('#previewError', false); return; }
  if (previewUrl === url) return true;
  const previousUrl = previewUrl;
  previewUrl = url;
  const token = ++previewRequest;
  show('#previewError', false); show('#orbitHint', false);
  try {
    previewModule ||= await import('/assets/preview.js');
    if (token !== previewRequest) return false;
    const replaced = await previewModule.show($('#preview'), url, options);
    if (token !== previewRequest) return false;
    if (replaced === false) { previewUrl = previousUrl; show('#orbitHint', Boolean(previousUrl)); return false; }
    show('#orbitHint'); return true;
  } catch {
    if (token !== previewRequest) return false;
    previewUrl = null; show('#previewError');
    return false;
  }
}

// While the 3D shape is built: the hero's figure of dots in the 3D viewer, until the first model takes its place.
let shapeModule = null;
async function buildShape(info) {
  if (!info || screen !== 'generation') return false;
  const token = ++previewRequest;
  previewUrl = null; show('#previewError', false);
  try {
    [previewModule, shapeModule] = await Promise.all([previewModule || import('/assets/preview.js'), shapeModule || import('/assets/shape-build.js')]);
    const figure = await shapeModule.figurePoints(info);
    if (token !== previewRequest) return false;
    const shown = previewModule.building($('#preview'), figure);
    show('#orbitHint', shown); return shown;
  } catch { return false; }
}

// Poll while the job is queued or running.
function polls(j) {
  return j.state === 'queued' || j.state === 'running' || j.export_state === 'running';
}

function followJob(j, reveal = false) {
  clearTimeout(S.timer);
  if (S.current !== j.id) { developer.update(null); generation.reset(); liveLog.reset(); }
  S.current = j.id;
  if (reveal) {
    setScreen('generation');
    const hash = `#job=${encodeURIComponent(j.id)}`;
    if (location.hash !== hash) history.pushState(null, '', hash);
  }
  renderJob(j); show('#jobLoading', false);
  if (reveal) {
    show('#heroLibrary', false); $('#showLibrary').setAttribute('aria-expanded', 'false');
    window.scrollTo({ top: 0, behavior: 'auto' }); $('#detail').focus({ preventScroll: true });
  }
  if (polls(j) && !document.hidden) S.timer = setTimeout(() => openJob(j.id)
    .then(() => { if (!document.hidden) return refreshList(); }).catch((e) => banner(e.message)), 3000);
}
document.addEventListener('visibilitychange', () => {
  clearTimeout(S.timer);
  if (!document.hidden && S.current && screen === 'generation') openJob(S.current).catch(e => banner(e.message));
});
async function openJob(id, reveal = false) {
  const revision = ++S.revision;
  clearTimeout(S.timer);
  if (S.current !== id) { developer.update(null); generation.reset(); liveLog.reset(); show('#detail', false); S.job = null; S.shown = null; S.lostSince = 0; }
  S.current = id;
  if (reveal) show('#jobLoading');
  try {
    const j = await api(`/api/jobs/${encodeURIComponent(id)}`);
    if (S.current !== id || revision !== S.revision) return;
    banner(''); connection(true); followJob(j, reveal);
  } catch (e) {
    if (revision !== S.revision) return;
    // A hero already on screen that cannot be reached right now is kept as it is, and the page keeps trying.
    if (S.shown === id && !reveal && (!e.status || e.status >= 500)) {
      connection(false);
      if (!document.hidden) S.timer = setTimeout(() => openJob(id), 3000);
    } else if (e.status === 404) {
      banner('This hero is not in your workspace. It may have been deleted.', { title: 'Hero not found', kind: 'warning',
        action: { label: 'Go to your heroes', run: () => { banner(''); $('#heroesNav').click(); } } });
    } else banner(e.message);
  }
  finally { if (revision === S.revision) show('#jobLoading', false); }
}

function initJobs() {
  const prompt = $('#prompt');
  const count = $('#promptCount');
  const limit = S.session.limits.prompt_max_chars;
  // Allow excess text so pasting a long description shows the problem without silently truncating it.
  function updatePromptCount() {
    const length = Array.from(prompt.value).length;
    const over = length > limit;
    count.textContent = `${length} / ${limit} characters${over ? ' (over limit)' : ''}`;
    count.dataset.level = over ? 'over' : length >= limit * 0.9 ? 'near' : '';
    prompt.setAttribute('aria-invalid', String(over));
  }
  prompt.addEventListener('input', updatePromptCount);
  entry = entryPage($, updatePromptCount);
  updatePromptCount();
  for (const r of document.querySelectorAll('input[name=kind]')) r.addEventListener('change', () => {
    const p = document.querySelector('input[name=kind]:checked').value === 'prompt';
    show('#promptIn', p); show('#sketchIn', !p);
  });
  const sel = $('#profile');
  const names = { 'roblox-ingame': 'Roblox in-game', 'roblox-avatar': 'Roblox avatar', best: 'Best quality (GLB only)' };
  sel.replaceChildren(...S.session.limits.profiles.map((p) => el('option', { value: p, text: names[p] || p })));
  sel.value = S.session.limits.default_profile;
  sel.dispatchEvent(new Event('change'));
  entry.generation(S.session.generation);
  $('#jobForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if ($('#go').disabled) return;
    track('create_clicked');
    const say = entry.say;
    const fd = new FormData();
    const kind = document.querySelector('input[name=kind]:checked').value;
    fd.append('kind', kind);
    fd.append('profile', sel.value);
    fd.append('name', $('#name').value);
    let file = null;
    if (kind === 'sketch') {
      file = entry.file();
      if (!file) { track('create_result', { result: 'no_file' }); say('Choose an image to upload.'); $('#browseUpload').focus(); return; }
      if (file.size > S.session.limits.upload_max_bytes) { track('create_result', { result: 'upload_too_large' }); say(`Choose an image smaller than ${Math.round(S.session.limits.upload_max_bytes / 1024 / 1024)} MB.`); return; }
      fd.append('sketch', file, 'sketch');
    } else {
      if (!prompt.value.trim()) { track('create_result', { result: 'empty_prompt' }); say('Describe your hero or choose an image to start.'); prompt.focus(); return; }
      if (Array.from(prompt.value).length > limit) { track('create_result', { result: 'prompt_too_long' }); say(`Keep your idea within ${limit} characters.`); prompt.focus(); return; }
      fd.append('prompt', prompt.value);
    }
    $('#go').disabled = true; $('#go').setAttribute('aria-busy', 'true');
    const key = idemKey();
    const done = () => { $('#go').disabled = false; $('#go').removeAttribute('aria-busy'); };
    try {
      // While live generation is closed, re-check once and say why instead of opening a workspace for nothing.
      if (S.session.generation?.accepting === false) await loadSession();
      if (S.session.generation?.accepting === false) {
        track('create_result', { result: 'generation_closed' });
        say(S.session.generation.sentence || 'Live generation is paused right now.', 'closed'); done(); return;
      }
      // The first create opens a private guest workspace silently; there is no guest dialog.
      if (!S.session.signed_in) { await openGuestWorkspace(); await signedIn(); }
    } catch (e) { say(e.message); done(); return; }
    say('Starting your hero…', 'info');
    try {
      const submit = () => api('/api/jobs', { method: 'POST', form: fd, headers: { 'Idempotency-Key': key } });
      let j;
      try { j = await submit(); }
      catch (e) {
        if (e instanceof TypeError || e.code === 'busy') j = await submit();
        else throw e;
      }
      track('create_result', { result: 'accepted' });
      ++S.revision; followJob(j, true);
      say('');
      refreshList().catch((e) => banner(e.message));
    } catch (e) {
      track('create_result', { result: knownCreateResult(e.code, e.status) });
      say(e.message);
      if (e.code === 'generation_paused' || e.code === 'generation_budget_exhausted') loadSession().catch(() => {});
    }
    finally { done(); }
  });
  $('#cancelJob').addEventListener('click', async () => {
    const id = S.current; ++S.revision; clearTimeout(S.timer);
    $('#cancelJob').disabled = true;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST', json: {} });
      if (S.current === id) { renderJob(job); await openJob(id); }
      await refreshList();
    } catch (e) { banner(e.message); if (S.current === id) openJob(id); }
    finally { $('#cancelJob').disabled = false; }
  });
  $('#deleteJob').addEventListener('click', () => {
    $('#deleteHeading').textContent = `Delete ${S.job ? S.job.name || kindName(S.job) : 'this hero'}?`;
    $('#deleteError').hidden = true; $('#confirmDelete').disabled = false; $('#deleteDialog').showModal();
  });
  $('#confirmDelete').addEventListener('click', async () => {
    const id = S.current; ++S.revision; clearTimeout(S.timer);
    $('#confirmDelete').disabled = true;
    try {
      await api(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE', json: {} });
      $('#deleteDialog').close();
      if (S.current === id) {
        show('#detail', false); developer.update(null); generation.reset(); liveLog.reset(); S.current = null; S.job = null; S.shown = null;
        history.replaceState(null, '', '/#heroes'); await showHeroes({ push: false });
      } else await refreshList();
    } catch (e) {
      $('#deleteError').textContent = e.message; $('#deleteError').hidden = false; $('#confirmDelete').disabled = false;
      if (S.current === id) openJob(id);
    }
  });

}

async function signedIn() {
  const guest = S.session.user.kind === 'guest';
  show('#signOut', !guest);
  updateAccessSummary();
  show('#developerMode'); show('#app');
  const jobs = await refreshList();
  const want = new URLSearchParams(location.hash.slice(1)).get('job');
  if (want) await openJob(want, true);
}
function updateAccessSummary() {
  // The composer shows the generation-closed notice (with the server's sentence) for guests and visitors alike.
  show('#guestAccessSummary', false);
  entry?.generation(S.session?.generation);
}
function expireAccess() {
  if (S.session) S.session.signed_in = false;
  returnToEntry(); show('#app', false); show('#signOut', false); show('#developerMode', false);
  heroes.update([]);
  show('#guestAccessSummary', false); show('#headerSignIn', false);
  entry?.say('Your workspace session has expired. Create your hero to open a new private workspace.', 'info');
}
async function refreshAccess() {
  await loadSession();
  if (S.session.signed_in) { updateAccessSummary(); return; }
  expireAccess();
}
let guestOpening = null;
async function openGuestWorkspace() {
  if (!guestOpening) guestOpening = (async () => {
    const result = await api('/api/demo/access', {method:'POST',json:{}});
    S.csrf = result.csrf; setCsrf(S.csrf); await loadSession();
  })();
  try { await guestOpening; } finally { guestOpening = null; }
}
function initGuestAccess() {
  $('#checkGeneration').addEventListener('click', async () => {
    $('#checkGeneration').disabled = true; $('#checkGeneration').setAttribute('aria-busy', 'true');
    try { await loadSession(); if (S.session.generation?.accepting !== false) { entry.say(''); $('#go').focus(); } }
    catch (e) { entry.say(e.message); }
    finally { $('#checkGeneration').disabled = false; $('#checkGeneration').removeAttribute('aria-busy'); }
  });
  $('#closeGuest').addEventListener('click', () => $('#guestDialog').close());
  $('#checkAvailability').addEventListener('click', async () => {
    $('#checkAvailability').disabled=true;
    try { await refreshAccess(); } catch (e) { banner(e.message); }
    finally { $('#checkAvailability').disabled=false; }
  });
  $('#guestForm').addEventListener('submit', async (event) => {
    event.preventDefault(); $('#guestGo').disabled=true; $('#guestMsg').textContent='Opening your private workspace…';
    try {
      await openGuestWorkspace();
      await loadSession(); $('#guestDialog').close(); $('#guestMsg').textContent='';
      await signedIn();
    } catch(e) { $('#guestMsg').textContent=e.message; }
    finally { $('#guestGo').disabled=false; }
  });
}
async function boot() {
  try { await loadSession(); }
  catch (e) { $('#loading').textContent = `HeroGen is not reachable: ${e.message}`; return; }
  const signinRoute = ['/admin/signin'].includes(location.pathname);
  $('#headerSignIn').textContent = 'Admin sign in';
  initSignin(); initJobs(); initMenu(); initGuestAccess();
  bindOrbitDrag($('#preview'), 'workbench');
  document.querySelectorAll('a[href="/terms"],a[href="/privacy"]').forEach((link) => link.addEventListener('click', () => track('page_view', { page: 'legal' }, { flush: true })));
  $('#backToIdeas').addEventListener('click', () => {
    history.pushState(null, '', '/'); returnToEntry();
    $('#entryH').tabIndex = -1; $('#entryH').focus();
  });
  $('#showLibrary').addEventListener('click', () => showHeroes());
  document.querySelectorAll('a[href="#gallery"],a[href="/#gallery"]').forEach(link => link.addEventListener('click', event => {
    if (location.pathname !== '/') return;
    event.preventDefault(); history.pushState(null, '', '/#gallery'); returnToEntry();
    $('#gallery').scrollIntoView({block:'start'});
  }));
  window.addEventListener('popstate', () => {
    if (location.pathname !== '/') return;
    const id = new URLSearchParams(location.hash.slice(1)).get('job');
    if (id) openJob(id, true);
    else if (location.hash === '#heroes') showHeroes({ push: false });
    else { returnToEntry(); if (location.hash === '#gallery') $('#gallery').scrollIntoView({block:'start'}); }
  });
  $('#headerSignIn').addEventListener('click', signIn);
  $('#closeSignin').addEventListener('click', () => $('#signinDialog').close());
  $('#signOut').addEventListener('click', async () => {
    try { await api('/api/auth/logout', { method: 'POST', json: {} }); location.replace('/'); }
    catch (e) { banner(e.message); }
  });
  $('#headerSignIn').disabled = false;
  show('#headerSignIn', signinRoute);
  window.addEventListener('hashchange', () => {
    if (location.pathname === '/admin/signin' && location.hash.startsWith('#t=')) initVerify().catch(e => banner(e.message));
  });
  if (location.pathname === '/admin/signin') { show('#loading', false); if (location.hash) return initVerify(); signIn(); return; }
  show('#loading', false);
  const heroesRoute = location.hash === '#heroes';
  if (!heroesRoute) { show('#entry'); trackOnce('page:landing', 'page_view', { page: 'landing' }); }
  landing = landingGallery({ visible: !heroesRoute });
  if (heroesRoute) { setScreen('heroes'); heroes.loading(); $('#heroesH').focus({ preventScroll: true }); }
  try { if (S.session.signed_in) await signedIn(); else if (heroesRoute) heroes.update([]); }
  catch (e) { if (heroesRoute) heroes.fail(e); else throw e; }
  if (location.hash === '#gallery') $('#gallery').scrollIntoView({block:'start'});
  window.addEventListener('focus', async () => {
    if (S.session.signed_in) return;
    try {
      await loadSession();
      if (S.session.signed_in) { $('#signinDialog').close(); await signedIn(); }
    } catch (e) { banner(e.message); }
  });
}

boot().catch((e) => banner(e.message));
