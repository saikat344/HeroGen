// HeroGen service page: sign in with an emailed link, start a hero from a sketch or a description, follow its
// stages, download the GLB and the Roblox bundle. Every value from the server is shown with textContent.

const $ = (s) => document.querySelector(s);
const S = { csrf: null, session: null, current: null, timer: null };

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
  if (!u) return;
  $('#quota').textContent = `${u.jobs_today} of ${u.jobs_per_day} heroes in the last 24 hours, ${u.jobs_total} of ${u.jobs_total_limit} in total.`;
}

async function refreshList() {
  const { jobs } = await api('/api/jobs');
  const ul = $('#jobs');
  ul.replaceChildren(...jobs.map((j) => {
    const b = el('button', { type: 'button', cls: 'jobbtn' }, el('b', { text: j.name || (j.kind === 'prompt' ? 'Described hero' : 'Sketched hero') }),
      el('span', { cls: `state s-${j.state}`, text: j.state }), el('small', { text: fmtTime(j.created_at) }));
    b.addEventListener('click', () => openJob(j.id));
    return el('li', {}, b);
  }));
  show('#noJobs', jobs.length === 0);
  return jobs;
}

function renderJob(j) {
  show('#detail');
  $('#detailH').textContent = j.name || (j.kind === 'prompt' ? 'Described hero' : 'Sketched hero');
  const st = $('#state'); st.textContent = j.state; st.className = `state s-${j.state}`;
  const bits = [`started ${fmtTime(j.created_at)}`, `profile ${j.profile}`];
  if (j.seconds) bits.push(`took ${fmtSecs(j.seconds)}`);
  if (typeof j.cost_usd === 'number') bits.push(`cost ${fmtUsd(j.cost_usd)} (estimate)`);
  if (j.code_commit) bits.push(`code ${j.code_commit.slice(0, 10)}`);
  $('#detailSub').textContent = bits.join(' · ') + (j.prompt ? ` · “${j.prompt}”` : '');
  const err = $('#detailErr');
  err.hidden = !j.error; err.textContent = j.error ? j.error.message : '';
  $('#stages').replaceChildren(...j.stages.map((s) => el('li', { cls: `st-${s.state}` },
    el('span', { cls: 'dot', 'aria-hidden': 'true' }), el('b', { text: s.label }), el('span', { cls: 'muted', text: ` ${s.state}` }),
    s.seconds ? el('small', { text: ` ${fmtSecs(s.seconds)}` }) : null,
    s.message && s.state !== 'passed' ? el('p', { cls: 'small', text: s.message }) : null)));
  const imgs = [['sketch', 'Sketch'], ['cover', 'Cover art'], ['front', 'Front'], ['left', 'Side'], ['back', 'Back']];
  $('#images').replaceChildren(...imgs.filter(([k]) => j.images[k]).map(([k, label]) =>
    el('figure', {}, el('img', { src: j.images[k], alt: label, loading: 'lazy' }), el('figcaption', { text: label }))));
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
  if (g.available) preview(g.url); else show('#previewWrap', false);
}

let previewUrl = null;
async function preview(url) {
  if (previewUrl === url) return;
  previewUrl = url;
  show('#previewWrap');
  try { (await import('/assets/preview.js')).show($('#preview'), url); }
  catch (e) { show('#previewWrap', false); console.warn('preview unavailable', e); }
}

async function openJob(id) {
  clearTimeout(S.timer);
  S.current = id;
  try {
    const j = await api(`/api/jobs/${encodeURIComponent(id)}`);
    if (S.current !== id) return;
    renderJob(j);
    if (j.state === 'queued' || j.state === 'running') S.timer = setTimeout(() => openJob(id).then(refreshList), 3000);
  } catch (e) { banner(e.message); }
}

function initJobs() {
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
    const msg = $('#jobMsg');
    const fd = new FormData();
    const kind = document.querySelector('input[name=kind]:checked').value;
    fd.append('kind', kind);
    fd.append('profile', sel.value);
    fd.append('name', $('#name').value);
    let file = null;
    if (kind === 'sketch') {
      file = $('#sketch').files[0];
      if (!file) { msg.textContent = 'Choose a sketch to upload.'; return; }
      fd.append('sketch', file, 'sketch');
    } else fd.append('prompt', $('#prompt').value);
    $('#go').disabled = true; msg.textContent = 'Starting…';
    const hash = await formHash(kind, sel.value, $('#name').value, $('#prompt').value, file);
    const earlier = loadPending();
    const key = earlier ? earlier.key : idemKey();
    savePending({ key, hash: earlier ? earlier.hash : hash });
    try {
      const j = await api('/api/jobs', { method: 'POST', form: fd, headers: { 'Idempotency-Key': key } });
      clearPending();
      msg.textContent = '';
      await refreshList();
      renderQuota((await loadSession()).usage);
      openJob(j.id);
    } catch (e) {
      if (e.status === 409) {
        clearPending();
        msg.textContent = earlier && earlier.hash !== hash
          ? 'Your earlier, unconfirmed hero from this tab was saved, so this different one was not started: check Your heroes, then make it again.'
          : `${e.message} Check Your heroes.`;
        refreshList().catch(() => {});
      } else if (e.status === 503) {
        msg.textContent = 'The service is busy and could not confirm your hero. Make it again in a moment from this tab: if it was saved, you get that hero, not a second one.';
      } else {
        msg.textContent = e.message;
      }
    }
    finally { $('#go').disabled = false; }
  });
  $('#cancelJob').addEventListener('click', async () => {
    try { renderJob(await api(`/api/jobs/${encodeURIComponent(S.current)}/cancel`, { method: 'POST', json: {} })); await refreshList(); }
    catch (e) { banner(e.message); }
  });
  $('#deleteJob').addEventListener('click', async () => {
    if (!confirm('Delete this hero and all of its files?')) return;
    try { await api(`/api/jobs/${encodeURIComponent(S.current)}`, { method: 'DELETE', json: {} }); show('#detail', false); clearTimeout(S.timer); await refreshList(); }
    catch (e) { banner(e.message); }
  });
  $('#deleteAccount').addEventListener('click', async () => {
    try { await api('/api/account', { method: 'DELETE', json: { confirm: $('#confirmDelete').value } }); location.replace('/'); }
    catch (e) { banner(e.message); }
  });
}

async function boot() {
  try { await loadSession(); }
  catch (e) { $('#loading').textContent = `HeroGen is not reachable: ${e.message}`; return; }
  show('#loading', false);
  if (location.pathname === '/signin') return initVerify();
  if (!S.session.signed_in) { show('#signin'); initSignin(); return; }
  $('#who').textContent = S.session.user.email; show('#who'); show('#signOut');
  $('#signOut').addEventListener('click', async () => { await api('/api/auth/logout', { method: 'POST', json: {} }).catch(() => {}); location.replace('/'); });
  show('#app');
  renderQuota(S.session.usage);
  initJobs();
  const jobs = await refreshList();
  const want = new URLSearchParams(location.hash.slice(1)).get('job');
  if (want) openJob(want); else if (jobs.length) openJob(jobs[0].id);
}

boot();
