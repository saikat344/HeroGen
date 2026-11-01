// HeroGen service page: sign in with an emailed link, start a hero from a sketch or a description, follow its
// stages, download the GLB and the Roblox bundle. Every value from the server is shown with textContent.
import { developerMode } from '/assets/developer.js';
import { entryPage } from '/assets/entry.js';
import { draftStore } from '/assets/draft.js';
import { publicGallery } from '/assets/public-gallery.js';
import { generationView } from '/assets/generation.js';

const $ = (s) => document.querySelector(s);
const S = { csrf: null, session: null, current: null, timer: null, revision: 0 };
let entry;
let screen = 'entry';
function setScreen(next) {
  screen = next; show('#entry', next === 'entry'); show('#detail', next === 'generation');
  $('#app').classList.toggle('generation-screen', next === 'generation');
}
const generation = generationView($, (url) => preview(url));
function returnToEntry() {
  ++S.revision; clearTimeout(S.timer); S.current = null;
  if ($('#developerMode').getAttribute('aria-pressed') === 'true') $('#developerMode').click();
  developer.update(null); generation.reset(); setScreen('entry'); show('#jobLoading', false);
}
function signIn() {
  if (location.pathname === '/signin' || location.pathname === '/admin/signin') { show('#signin'); $('#signinDialog').showModal(); }
  else {
    $('#accessIntro').textContent = S.session?.public_demo ? 'No account needed. One active generation per workspace, with up to five globally.' : 'Live guest access is not open yet. You can inspect the public gallery.';
    $('#guestGo').disabled = !S.session?.public_demo;
    $('#guestDialog').showModal();
  }
}
async function keepDraft(values = null) {
  const fields = values || { prompt: $('#prompt').value, name: $('#name').value,
    profile: $('#profile').value, file: entry.file() };
  const pending = loadPending();
  if (!fields.prompt.trim() && !fields.name.trim() && !fields.file && !pending) return draftStore('delete');
  await draftStore('put', { ...fields, pending, saved: Date.now() });
}
async function restoreDraft() {
  try {
    const d = await draftStore('get');
    if (!d || Date.now() - d.saved > 24 * 3600 * 1000) { if (d) await draftStore('delete'); return; }
    const pending = loadPending();
    // A failed storage write must never restore an older input under a newer request key.
    if (pending && (!d.pending || d.pending.key !== pending.key || d.pending.hash !== pending.hash || d.pending.revision !== pending.revision)) return;
    $('#prompt').value = typeof d.prompt === 'string' ? d.prompt : '';
    $('#name').value = typeof d.name === 'string' ? d.name : '';
    if (S.session.limits.profiles.includes(d.profile)) $('#profile').value = d.profile;
    if (d.file instanceof Blob) entry.setFile(d.file);
    if (d.pending && !loadPending()) savePending(d.pending);
    $('#prompt').dispatchEvent(new Event('input'));
    $('#jobMsg').textContent = loadPending()
      ? 'An earlier hero was not confirmed. Retry this saved input to recover it without starting a second hero.'
      : 'Your draft is ready. Review it, then create your hero.';
  } catch { /* A fresh form remains usable without browser storage. */ }
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
const PENDING = 'hg-pending-job';
let memPending = null;   // used when sessionStorage is unavailable (then the key lasts until a reload)
function loadPending() {
  let v = memPending;
  try { v = JSON.parse(sessionStorage.getItem(PENDING) || 'null') || memPending; } catch { /* storage off */ }
  return v && typeof v.key === 'string' && /^[A-Za-z0-9_-]{8,64}$/.test(v.key) ? v : null;
}
function savePending(v) { memPending = v; try { sessionStorage.setItem(PENDING, JSON.stringify(v)); } catch { /* storage off */ } }
function clearPending() { memPending = null; try { sessionStorage.removeItem(PENDING); } catch { /* storage off */ } }
async function formHash(kind, profile, name, prompt, file) {
  try {
    const parts = [new TextEncoder().encode(JSON.stringify([kind, profile, name, kind === 'prompt' ? prompt : '']))];
    if (file) parts.push(new Uint8Array(await file.arrayBuffer()));
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) { all.set(p, at); at += p.length; }
    const d = new Uint8Array(await crypto.subtle.digest('SHA-256', all));
    return Array.from(d, (x) => x.toString(16).padStart(2, '0')).join('');
  } catch { return ''; }
}
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
      const r = await api('/api/auth/request', { method: 'POST', json: { email: $('#email').value, invite_code: $('#invite').value } });
      msg.textContent = r.message;
    } catch (e) { msg.textContent = e.message; }
  });
}

async function initVerify() {
  const token = new URLSearchParams(location.hash.slice(1)).get('t');
  history.replaceState(null, '', '/signin');   // the token leaves the address bar and history
  show('#verify');
  if (!token) { $('#verifyWho').textContent = 'This sign-in link is incomplete. Ask for a new one.'; show('#verifyForm', false); return; }
  let info;
  try { info = await api('/api/auth/link', { method: 'POST', json: { token } }); }
  catch (e) { $('#verifyWho').textContent = e.message; show('#verifyForm', false); return; }
  $('#verifyWho').textContent = info.new_account ? `Create the HeroGen account for ${info.email}.` : `Sign in as ${info.email}.`;
  show('#terms', info.needs_terms);
  if (!info.terms_ready) { $('#verifyMsg').textContent = 'Sign-in is not open yet: the terms have not been published.'; $('#verifyBtn').disabled = true; }
  $('#verifyForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    try {
      const r = await api('/api/auth/verify', { method: 'POST', json: { token, accept_terms: $('#accept').checked, adult: $('#adult').checked } });
      S.csrf = r.csrf;
      location.replace('/');
    } catch (e) { $('#verifyMsg').textContent = e.message; }
  });
}

// ---- jobs ----------------------------------------------------------------------------------------------

function renderQuota(u) {
  if (S.session?.user?.kind === 'guest') {
    $('#quota').textContent = 'One active generation per workspace.';
    return;
  }
  if (!u) return;
  $('#quota').textContent = `${u.jobs_today} of ${u.jobs_per_day} heroes in the last 24 hours, ${u.jobs_total} of ${u.jobs_total_limit} in total.`;
}

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
  items.push(g.available ? el('a', { href: g.url, cls: 'button', download: '' , text: 'Download GLB' }) : el('span', { cls: 'muted small', text: `GLB: ${g.reason}` }));
  const r = j.downloads.roblox;
  items.push(r.available ? el('a', { href: r.url, cls: 'button', download: '', text: 'Download Roblox bundle (.zip)' }) : el('span', { cls: 'muted small', text: `Roblox bundle: ${r.reason}` }));
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

// Keep polling while the job is queued or running, and after it only while a stage beside it (the quality check)
// still runs. A job's own stage never keeps a finished job polling: one that shows as running there is a stale
// row, and polling would never end.
function polls(j) {
  return j.state === 'queued' || j.state === 'running' || j.stages.some((s) => s.side && s.state === 'running');
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
  $('#retention').textContent = String(S.session.limits.retention_days);
  // One Idempotency-Key per hero, kept in this tab's sessionStorage (with a hash of the form) from before the
  // request is sent until the server gives a definite answer for that key: the job (201 started, 200 already
  // saved) or a 409 (the key belongs to a saved job with a different form). On anything else (503 busy, another
  // 5xx, a lost connection, 401, 429, a refused upload) the earlier attempt may still have been saved, so the same
  // key is sent with the next attempt from this tab, also after a reload, and the server returns that job instead
  // of starting a second one. Other tabs and devices make their own keys.
  const pending = loadPending();
  if (pending) $('#jobMsg').textContent = 'An earlier hero from this tab was not confirmed. Make it again: if it was saved, you get that hero, not a second one.';
  $('#jobForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if ($('#go').disabled) return;
    const msg = $('#jobMsg');
    const fd = new FormData();
    const kind = document.querySelector('input[name=kind]:checked').value;
    fd.append('kind', kind);
    fd.append('profile', sel.value);
    fd.append('name', $('#name').value);
    let file = null;
    if (kind === 'sketch') {
      file = entry.file();
      if (!file) { msg.textContent = 'Choose an image to upload.'; return; }
      if (file.size > S.session.limits.upload_max_bytes) { msg.textContent = `Choose an image smaller than ${Math.round(S.session.limits.upload_max_bytes / 1024 / 1024)} MB.`; return; }
      fd.append('sketch', file, 'sketch');
    } else {
      if (!prompt.value.trim()) { msg.textContent = 'Describe your hero or choose an image to start.'; prompt.focus(); return; }
      if (Array.from(prompt.value).length > limit) { msg.textContent = `Keep your idea within ${limit} characters.`; prompt.focus(); return; }
      fd.append('prompt', prompt.value);
    }
    const values = { prompt: prompt.value, name: $('#name').value, profile: sel.value, file };
    $('#go').disabled = true;
    if (!S.session.signed_in) {
      try { await keepDraft(values); msg.textContent = 'Your draft is saved.'; }
      catch { msg.textContent = 'Keep this tab open to preserve your draft.'; }
      if (!S.session.public_demo) { $('#go').disabled = false; signIn(); return; }
      $('#go').disabled = true;
      try { await openGuestWorkspace(); await signedIn(); }
      catch(e) { msg.textContent = e.message; $('#go').disabled = false; return; }
    }
    $('#go').disabled = true; msg.textContent = 'Starting…';
    const hash = await formHash(kind, values.profile, values.name, values.prompt, file);
    const earlier = loadPending();
    const key = earlier ? earlier.key : idemKey();
    savePending({ key, hash: earlier ? earlier.hash : hash, revision: idemKey() });
    await keepDraft(values).catch(() => banner('Your retry draft could not be saved. Keep this tab open until your hero is confirmed.'));
    try {
      const j = await api('/api/jobs', { method: 'POST', form: fd, headers: { 'Idempotency-Key': key } });
      clearPending(); ++S.revision; followJob(j, true);
      await draftStore('delete').catch(() => {});
      msg.textContent = '';
      await Promise.all([refreshList(), loadSession().then((session) => { renderQuota(session.usage); updateAccessSummary(); })])
        .catch((e) => banner(`Your hero is saved. Could not refresh account or library details: ${e.message}`));
    } catch (e) {
      if (e.status === 409 && (!e.code || e.code === 'conflict')) {
        clearPending();
        await keepDraft(values).catch(() => draftStore('delete').catch(() => {}));
        msg.textContent = earlier && earlier.hash !== hash
          ? 'Your earlier, unconfirmed hero from this tab was saved, so this different one was not started: check Your heroes, then make it again.'
          : `${e.message} Check Your heroes.`;
        refreshList().catch(() => {});
      } else if (e.status === 503 && (!e.code || e.code === 'busy')) {
        msg.textContent = 'The service is busy and could not confirm your hero. Make it again in a moment from this tab: if it was saved, you get that hero, not a second one.';
      } else {
        msg.textContent = e.message;
        if (e.status === 401) {
          S.session.signed_in = false;
          try { await keepDraft(values); } catch { msg.textContent += ' Keep this tab open to preserve your draft.'; }
          signIn();
        }
      }
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
  $('#deleteAccount').addEventListener('click', async () => {
    try { await api('/api/account', { method: 'DELETE', json: { confirm: $('#confirmDelete').value } }); clearPending(); await draftStore('delete').catch(() => {}); location.replace('/'); }
    catch (e) { banner(e.message); }
  });
}

async function signedIn() {
  const guest = S.session.user.kind === 'guest';
  $('#who').textContent = guest ? 'Private guest workspace' : S.session.user.email; show('#who'); show('#signOut', !guest);
  show('#accountSettings', !guest);
  updateAccessSummary();
  show('#headerSignIn', false); show('#developerMode'); show('#app');
  renderQuota(S.session.usage);
  const jobs = await refreshList();
  const want = new URLSearchParams(location.hash.slice(1)).get('job');
  if (want) await openJob(want, true);
}
function updateAccessSummary() {
  const guest = S.session?.user?.kind === 'guest';
  show('#guestAccessSummary', guest);
  if (!guest) return;
  const reasons = { generation_paused: 'Live generation is temporarily paused.', cost_bound_unverified: 'Live generation is paused while its cost bound is verified.',
    generation_budget_exhausted: 'The live-generation budget is currently allocated.', generation_capacity: 'All live-generation slots are occupied. Try again shortly.' };
  $('#guestAccessText').textContent = `${reasons[S.session.generation?.reason] || 'Your private workspace is ready.'} One active generation per workspace. Existing results remain available.`;
}
function expireAccess() {
  if (S.session) S.session.signed_in = false;
  returnToEntry(); show('#app', false); show('#who', false); show('#signOut', false); show('#developerMode', false);
  $('#heroLibrary').hidden = true; $('#jobs').replaceChildren();
  show('#guestAccessSummary', false); show('#headerSignIn');
  $('#jobMsg').textContent = 'Your access has expired. Your draft is kept in this browser. Open your guest workspace to continue.';
}
async function refreshAccess() {
  await loadSession();
  if (S.session.signed_in) { updateAccessSummary(); renderQuota(S.session.usage); return; }
  expireAccess();
}
let guestOpening = null;
async function openGuestWorkspace() {
  if (!guestOpening) guestOpening = (async () => {
    const result = await api('/api/demo/access', {method:'POST',json:{}});
    S.csrf = result.csrf; await loadSession();
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
  $('#headerSignIn').textContent = 'Open workspace';
  initSignin(); initJobs(); initGuestAccess(); publicGallery();
  $('#backToIdeas').addEventListener('click', () => {
    history.pushState(null, '', '/'); returnToEntry();
    $('#entryH').tabIndex = -1; $('#entryH').focus();
  });
  $('#showLibrary').addEventListener('click', async () => {
    const open = $('#heroLibrary').hidden; show('#heroLibrary', open); $('#showLibrary').setAttribute('aria-expanded', String(open));
    if (open) await refreshList().catch((e) => banner(e.message));
  });
  document.querySelectorAll('a[href="#gallery"],a[href="/#gallery"]').forEach(link => link.addEventListener('click', event => {
    event.preventDefault(); history.pushState(null, '', '/#gallery'); returnToEntry();
    $('#gallery').scrollIntoView({block:'start'});
  }));
  window.addEventListener('popstate', () => {
    const id = new URLSearchParams(location.hash.slice(1)).get('job');
    if (id) openJob(id, true); else { returnToEntry(); if (location.hash === '#gallery') $('#gallery').scrollIntoView({block:'start'}); }
  });
  $('#headerSignIn').addEventListener('click', async () => {
    try { if (!['/signin','/admin/signin'].includes(location.pathname)) await keepDraft(); } catch { $('#jobMsg').textContent = 'Keep this tab open to preserve your draft during sign-in.'; }
    signIn();
  });
  $('#closeSignin').addEventListener('click', () => $('#signinDialog').close());
  $('#signOut').addEventListener('click', async () => {
    try { await api('/api/auth/logout', { method: 'POST', json: {} }); clearPending(); await draftStore('delete').catch(() => {}); location.replace('/'); }
    catch (e) { banner(e.message); }
  });
  $('#headerSignIn').disabled = false;
  if (location.pathname === '/signin') { show('#loading', false); return initVerify(); }
  if (location.pathname === '/admin/signin') { show('#loading', false); signIn(); return; }
  await restoreDraft();
  show('#loading', false); show('#entry');
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
