// HeroGen service page: start a guest hero from a sketch or description, follow its
// stages, download the GLB and the Roblox bundle. Every value from the server is shown with textContent.
import { developerMode } from '/assets/developer.js';
import { entryPage } from '/assets/entry.js';
import { landingGallery } from '/assets/landing.js';
let landing;
import { generationView } from '/assets/generation.js';
import { bindOrbitDrag, knownCreateResult, setCsrf, track, trackOnce } from './analytics.js';

const $ = (s) => document.querySelector(s);
const S = { csrf: null, session: null, current: null, timer: null, revision: 0 };
let entry;
let screen = 'entry';
function setScreen(next) {
  const enteringWorkbench = next === 'generation' && screen !== 'generation';
  screen = next; show('#entry', next === 'entry'); show('#detail', next === 'generation');
  landing?.setVisible(next === 'entry');
  $('#app').classList.toggle('generation-screen', next === 'generation');
  if (enteringWorkbench) track('page_view', { page: 'workbench' });
}
const generation = generationView($, (url) => preview(url));
function returnToEntry() {
  ++S.revision; clearTimeout(S.timer); S.current = null;
  if ($('#developerMode').getAttribute('aria-pressed') === 'true') $('#developerMode').click();
  developer.update(null); generation.reset(); setScreen('entry'); show('#jobLoading', false);
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
function banner(text) { const b = $('#banner'); b.textContent = text || ''; b.hidden = !text; }
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'text') e.textContent = v; else if (k === 'cls') e.className = v; else e.setAttribute(k, v);
  }
  for (const k of kids) if (k) e.append(k);
  return e;
}
function fmtTime(t) { return t ? new Date(t * 1000).toLocaleString() : ''; }
function fmtUsd(v) { return typeof v === 'number' ? `$${v.toFixed(v < 0.1 ? 4 : 2)}` : ''; }
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
    const err = new Error(msg); err.status = r.status; err.code = data && data.error && data.error.code; throw err;
  }
  return data;
}

async function loadSession() {
  S.session = await api('/api/session');
  S.csrf = S.session.csrf;
  setCsrf(S.csrf);
  return S.session;
}

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
  const ul = $('#jobs');
  ul.replaceChildren(...jobs.map((j) => {
    const b = el('button', { type: 'button', cls: 'jobbtn' }, el('b', { text: j.name || (j.kind === 'prompt' ? 'Described hero' : 'Sketched hero') }),
      el('span', { cls: `state s-${j.state}`, text: j.state }), el('small', { text: fmtTime(j.created_at) }));
    b.addEventListener('click', () => openJob(j.id, true));
    return el('li', {}, b);
  }));
  show('#noJobs', jobs.length === 0);
  return jobs;
}

function renderJob(j) {
  developer.update(j);
  show('#detail', screen === 'generation');
  $('#detailH').textContent = j.name || (j.kind === 'prompt' ? 'Described hero' : 'Sketched hero');
  const st = $('#state'); st.textContent = j.state; st.className = `state s-${j.state}`;
  const bits = [`started ${fmtTime(j.created_at)}`, `profile ${j.profile}`];
  if (Number.isFinite(j.seconds)) bits.push(`recorded duration ${fmtSecs(j.seconds)}`);
  if (Number.isFinite(j.cost_usd)) bits.push(`${j.cost_is_estimate === false ? 'recorded' : 'estimated'} cost ${fmtUsd(j.cost_usd)}`);
  $('#detailSub').textContent = bits.join(' · ') + (j.prompt ? ` · “${j.prompt}”` : '');
  const err = $('#detailErr');
  err.hidden = !j.error; err.textContent = j.error ? j.error.message : '';
  generation.update(j);
  const dl = $('#downloads');
  const items = [];
  const g = j.downloads.glb;
  if (g.available) {
    const glbLink = el('a', { href: g.url, cls: 'button', download: '', text: 'Download GLB' });
    glbLink.addEventListener('click', () => track('download_click', { format: 'glb' }, { flush: true }));
    items.push(glbLink);
  } else {
    items.push(el('span', { cls: 'muted small', text: `GLB: ${g.reason}` }));
  }
  const r = j.downloads.roblox;
  if (r.available) {
    const rbxLink = el('a', { href: r.url, cls: 'button', download: '', text: 'Download Roblox bundle (.zip)' });
    rbxLink.addEventListener('click', () => track('download_click', { format: 'roblox' }, { flush: true }));
    items.push(rbxLink);
  } else {
    items.push(el('span', { cls: 'muted small', text: `Roblox bundle: ${r.reason}` }));
  }
  dl.replaceChildren(...items);
  show('#studio', r.available);
  $('#readme').textContent = j.roblox_readme || '';
  $('#viewerLink').href = j.viewer;
  show('#cancelJob', j.state === 'queued' || j.state === 'running');
  $('#viewerLink').hidden = !g.available;
}

let previewUrl = null, previewRequest = 0, previewModule = null;
async function preview(url) {
  if (url && screen !== 'generation') return;
  if (!url) { ++previewRequest; previewUrl = null; previewModule?.hide(); show('#orbitHint', false); show('#previewError', false); return; }
  if (previewUrl === url) return;
  previewUrl = url;
  const token = ++previewRequest;
  show('#previewError', false); show('#orbitHint', false);
  try {
    previewModule ||= await import('/assets/preview.js');
    if (token !== previewRequest) return;
    await previewModule.show($('#preview'), url);
    if (token === previewRequest) show('#orbitHint');
  } catch {
    if (token !== previewRequest) return;
    previewUrl = null; show('#previewError');
  }
}

// Poll while the job is queued or running.
function polls(j) {
  return j.state === 'queued' || j.state === 'running';
}

function followJob(j, reveal = false) {
  clearTimeout(S.timer);
  if (S.current !== j.id) { developer.update(null); generation.reset(); }
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
  if (polls(j)) S.timer = setTimeout(() => openJob(j.id).then(refreshList).catch((e) => banner(e.message)), 3000);
}
async function openJob(id, reveal = false) {
  const revision = ++S.revision;
  clearTimeout(S.timer);
  if (S.current !== id) { developer.update(null); generation.reset(); show('#detail', false); }
  S.current = id;
  if (reveal) show('#jobLoading');
  try {
    const j = await api(`/api/jobs/${encodeURIComponent(id)}`);
    if (S.current !== id || revision !== S.revision) return;
    banner(''); followJob(j, reveal);
  } catch (e) { if (revision === S.revision) banner(e.message); }
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
    count.className = `small msg ${over ? 'err' : 'muted'}`;
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
  $('#jobForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if ($('#go').disabled) return;
    track('create_clicked');
    const msg = $('#jobMsg');
    const fd = new FormData();
    const kind = document.querySelector('input[name=kind]:checked').value;
    fd.append('kind', kind);
    fd.append('profile', sel.value);
    fd.append('name', $('#name').value);
    let file = null;
    if (kind === 'sketch') {
      file = entry.file();
      if (!file) { track('create_result', { result: 'no_file' }); msg.textContent = 'Choose an image to upload.'; return; }
      if (file.size > S.session.limits.upload_max_bytes) { track('create_result', { result: 'upload_too_large' }); msg.textContent = `Choose an image smaller than ${Math.round(S.session.limits.upload_max_bytes / 1024 / 1024)} MB.`; return; }
      fd.append('sketch', file, 'sketch');
    } else {
      if (!prompt.value.trim()) { track('create_result', { result: 'empty_prompt' }); msg.textContent = 'Describe your hero or choose an image to start.'; prompt.focus(); return; }
      if (Array.from(prompt.value).length > limit) { track('create_result', { result: 'prompt_too_long' }); msg.textContent = `Keep your idea within ${limit} characters.`; prompt.focus(); return; }
      fd.append('prompt', prompt.value);
    }
    $('#go').disabled = true;
    const key = idemKey();
    try {
      if (!S.session.signed_in) { await openGuestWorkspace(); await signedIn(); }
    } catch (e) { msg.textContent = e.message; $('#go').disabled = false; return; }
    msg.textContent = 'Starting…';
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
      msg.textContent = '';
      refreshList().catch((e) => banner(e.message));
    } catch (e) {
      track('create_result', { result: knownCreateResult(e.code, e.status) });
      msg.textContent = e.message;
    }
    finally { $('#go').disabled = false; }
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
  $('#deleteJob').addEventListener('click', async () => {
    if (!confirm('Delete this hero and all of its files?')) return;
    const id = S.current; ++S.revision; clearTimeout(S.timer);
    try {
      await api(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE', json: {} });
      if (S.current === id) {
        show('#detail', false); developer.update(null); generation.reset(); S.current = null;
        history.replaceState(null, '', '/'); setScreen('entry'); show('#heroLibrary'); $('#showLibrary').setAttribute('aria-expanded', 'true');
      }
      await refreshList();
    } catch (e) { banner(e.message); if (S.current === id) openJob(id); }
  });

}

async function signedIn() {
  const guest = S.session.user.kind === 'guest';
  $('#who').textContent = guest ? 'Private guest workspace' : S.session.user.email; show('#who'); show('#signOut', !guest);
  updateAccessSummary();
  show('#developerMode'); show('#app');
  const jobs = await refreshList();
  const want = new URLSearchParams(location.hash.slice(1)).get('job');
  if (want) await openJob(want, true);
}
function updateAccessSummary() {
  const guest = S.session?.user?.kind === 'guest';
  // Limit and recovery copy is a refusal notice: it appears only while this workspace cannot start generation.
  const refused = guest && S.session.generation?.accepting === false;
  show('#guestAccessSummary', refused);
  if (!refused) return;
  const reasons = { generation_paused: 'Live generation is paused right now.',
    generation_budget_exhausted: 'The live-generation budget is used up for now.' };
  $('#guestAccessText').textContent = `${reasons[S.session.generation.reason] || reasons.generation_paused} You can still explore the generated heroes.`;
}
function expireAccess() {
  if (S.session) S.session.signed_in = false;
  returnToEntry(); show('#app', false); show('#who', false); show('#signOut', false); show('#developerMode', false);
  $('#heroLibrary').hidden = true; $('#jobs').replaceChildren();
  show('#guestAccessSummary', false); show('#headerSignIn', false);
  $('#jobMsg').textContent = 'Your workspace session has expired. Create your hero to open guest access.';
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
  initSignin(); initJobs(); initGuestAccess();
  bindOrbitDrag($('#preview'), 'workbench');
  document.querySelectorAll('a[href="/terms"],a[href="/privacy"]').forEach((link) => link.addEventListener('click', () => track('page_view', { page: 'legal' }, { flush: true })));
  $('#backToIdeas').addEventListener('click', () => {
    history.pushState(null, '', '/'); returnToEntry();
    $('#entryH').tabIndex = -1; $('#entryH').focus();
  });
  $('#showLibrary').addEventListener('click', async () => {
    const open = $('#heroLibrary').hidden; show('#heroLibrary', open); $('#showLibrary').setAttribute('aria-expanded', String(open));
    if (open) await refreshList().catch((e) => banner(e.message));
  });
  document.querySelectorAll('a[href="#gallery"],a[href="/#gallery"]').forEach(link => link.addEventListener('click', event => {
    if (location.pathname !== '/') return;
    event.preventDefault(); history.pushState(null, '', '/#gallery'); returnToEntry();
    $('#gallery').scrollIntoView({block:'start'});
  }));
  window.addEventListener('popstate', () => {
    if (location.pathname !== '/') return;
    const id = new URLSearchParams(location.hash.slice(1)).get('job');
    if (id) openJob(id, true); else { returnToEntry(); if (location.hash === '#gallery') $('#gallery').scrollIntoView({block:'start'}); }
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
  show('#loading', false); show('#entry');
  trackOnce('page:landing', 'page_view', { page: 'landing' });
  landing = landingGallery();
  if (S.session.signed_in) await signedIn();
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
