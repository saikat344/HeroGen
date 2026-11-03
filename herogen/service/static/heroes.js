// Your heroes: the card grid behind /#heroes. Server values only ever reach the page through textContent
// or attributes. The list comes from app.js (GET /api/jobs); this module groups it, draws the cards, and
// fetches GET /api/jobs/{id} for the few cards that need more than the summary (stage progress for running
// heroes, a thumbnail when the list has none, the Roblox flag).

import { track } from './analytics.js';

const SVG = 'http://www.w3.org/2000/svg';
const ACTIVE = new Set(['queued', 'running']);
const IMAGE_ORDER = ['cover', 'front', 'sketch'];
export const DETAIL_LIMIT = 12;      // cards beyond this fall back to the monogram tile and cost no request

// ---- pure helpers (unit-tested in tests/service_heroes.test.mjs) ----------------------------------------

// The one name for an unnamed hero of each kind: the cards, the workbench heading and the delete dialog use it.
export const kindName = (job) => job?.kind === 'prompt' ? 'Described hero' : 'Sketched hero';

export function defaultName(job, detail) {
  if (job.name) return job.name;
  const prompt = typeof detail?.prompt === 'string' ? detail.prompt.trim() : '';
  if (prompt) return prompt;
  return kindName(job);
}

// Root heroes, newest first. A variation (Retry or Refine) points at its parent; walk up to the root and count
// it there, so the card says "3 variations" and the grid never lists a branch as a hero of its own.
export function rootHeroes(jobs) {
  const byId = new Map(jobs.map((job) => [job.id, job]));
  const rootOf = (job) => {
    const seen = new Set();
    let at = job;
    while (at.parent_job && byId.has(at.parent_job) && !seen.has(at.id)) { seen.add(at.id); at = byId.get(at.parent_job); }
    return at;
  };
  const counts = new Map();
  for (const job of jobs) {
    if (!job.parent_job) continue;
    const root = rootOf(job);
    if (root !== job) counts.set(root.id, (counts.get(root.id) || 0) + 1);
  }
  return jobs.filter((job) => !job.parent_job || !byId.has(job.parent_job))
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
    .map((job) => ({ job, variations: counts.get(job.id) || 0 }));
}

function span(seconds) {
  if (!Number.isFinite(seconds) || seconds < 1) return '';
  if (seconds < 90) return `${Math.round(seconds)} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

export function relativeTime(seconds, now = Date.now() / 1000) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '';
  const age = Math.max(0, now - seconds);
  if (age < 45) return 'Just now';
  if (age < 90) return '1 min ago';
  if (age < 3600) return `${Math.round(age / 60)} min ago`;
  if (age < 86400) return `${Math.round(age / 3600)} h ago`;
  const days = Math.floor(age / 86400);
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return new Date(seconds * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// One segment per main stage (side stages have `side`), like the workspace progress bar.
export function stageProgress(detail) {
  const stages = Array.isArray(detail?.stages) ? detail.stages.filter((stage) => !stage.side) : [];
  if (!stages.length) return null;
  const kind = (state) => ({ passed: 'done', running: 'run', skipped: 'skip', unavailable: 'na', failed: 'fail',
    rejected: 'fail', canceled: 'na' })[state] || 'wait';
  const segments = stages.map((stage) => ({ kind: kind(stage.state), label: stage.label || '' }));
  const done = stages.filter((stage) => stage.state === 'passed').length;
  const skipped = stages.filter((stage) => stage.state === 'skipped').length;
  const now = stages.find((stage) => stage.state === 'running');
  return { segments, done, skipped, total: stages.length, now: now ? now.label : '' };
}

// The status pill. Colour is never the only signal: every state also has an icon and a label.
// A succeeded job's model is ready at `finished_at`; the Roblox steps then run on for `export_seconds`. While they
// run, the note gives the model time and says the Roblox files are still coming; afterwards it gives the full time.
// Run time counts from `started_at`: the time spent queued is not run time, and with no start there is no time to give.
export function finishedNote(job, format, start = job.started_at) {
  const known = Number.isFinite(start) && Number.isFinite(job.finished_at);
  const model = known ? format(job.finished_at - start) : '';
  if (job.export_state === 'running') return model ? `Model ready in ${model} · Roblox files in progress` : 'Roblox files in progress';
  const total = known ? format(job.finished_at + (Number.isFinite(job.export_seconds) ? job.export_seconds : 0) - start) : '';
  return total ? `Finished in ${total}` : '';
}

// "Roblox ready" appears only when the detail says the Roblox steps passed.
export function heroStatus(job, detail) {
  const progress = stageProgress(detail);
  switch (job.state) {
    case 'queued': {
      const ahead = Number.isFinite(job.queued_ahead) ? job.queued_ahead : (Number.isFinite(detail?.queued_ahead) ? detail.queued_ahead : null);
      const note = ahead === null ? 'Starts when a worker is free'
        : ahead === 0 ? 'Next to start' : `${ahead} ${ahead === 1 ? 'hero' : 'heroes'} ahead of you`;
      return { tone: 'wait', icon: 'timer', label: 'Waiting', note, progress, active: true };
    }
    case 'running': {
      const label = progress?.now || job.stage_label || '';
      const note = progress ? `${progress.done} of ${progress.total} stages complete` : (label ? `Now: ${label}` : 'Starting');
      return { tone: 'run', icon: 'loader-circle', label: 'Running', note, now: label, progress, active: true };
    }
    case 'succeeded': {
      // The list carries the export fields too, so a card without its detail record still knows the Roblox steps are
      // running. The list is the fresher of the two (it is polled; the detail is fetched once), so it wins.
      const exporting = (job.export_state ?? detail?.export_state) === 'running';
      const note = finishedNote({ ...job, export_state: job.export_state ?? detail?.export_state,
        export_seconds: job.export_seconds ?? detail?.export_seconds }, span, job.started_at ?? detail?.started_at);
      if (detail?.roblox_state === 'ready' && !exporting) return { tone: 'ok', icon: 'person-standing', label: 'Roblox ready', note, progress, active: false };
      return { tone: 'ok', icon: 'circle-check', label: 'Ready', note, progress, active: false };
    }
    case 'failed':
      return { tone: 'fail', icon: 'triangle-alert', label: 'Needs attention', note: 'A step did not finish. Open it to retry.', progress, active: false };
    case 'rejected':
      return { tone: 'fail', icon: 'triangle-alert', label: 'Needs attention', note: 'The input was not accepted. Open it to change it.', progress, active: false };
    case 'canceled':
      return { tone: 'off', icon: 'x', label: 'Cancelled', note: 'You stopped this run.', progress, active: false };
    case 'stopped':
      return { tone: 'stop', icon: 'pause', label: 'Saved at this step', note: '', progress, active: false };
    default:
      return { tone: 'off', icon: 'circle-dashed', label: job.state || 'Unknown', note: '', progress, active: false };
  }
}

// A finished, Roblox-ready hero's one-click Roblox download (the card's status pill says "Roblox ready" too), or null.
export function robloxDownload(job, detail) {
  if (heroStatus(job, detail).label !== 'Roblox ready') return null;
  const roblox = detail?.downloads?.roblox;
  return roblox?.available && typeof roblox.url === 'string' && roblox.url ? roblox.url : null;
}

// The summary's `thumbnail_url` first when the server provides one, then the images on the job detail, then
// nothing: the card draws a monogram tile.
export function thumbnailOf(job, detail) {
  if (typeof job.thumbnail_url === 'string' && job.thumbnail_url) return job.thumbnail_url;
  const images = detail?.images;
  if (images && typeof images === 'object') {
    for (const key of IMAGE_ORDER) if (typeof images[key] === 'string' && images[key]) return images[key];
    const first = Object.values(images).find((value) => typeof value === 'string' && value);
    if (first) return first;
  }
  return null;
}

// A hero is still changing while it waits or runs, and for a succeeded hero while its Roblox steps run on.
export function stillWorking(job) {
  return ACTIVE.has(job.state) || (job.state === 'succeeded' && job.export_state === 'running');
}

// Which cards need GET /api/jobs/{id}. Running heroes always do (live progress); finished ones only when the
// list could not give a thumbnail or a name.
export function needsDetail(job, index) {
  if (index >= DETAIL_LIMIT) return false;
  if (ACTIVE.has(job.state)) return true;
  return !job.thumbnail_url || (!job.name && job.kind === 'prompt') || job.state === 'succeeded';
}

export function summaryLine(roots) {
  if (!roots.length) return '';
  const running = roots.filter(({ job }) => job.state === 'running').length;
  const waiting = roots.filter(({ job }) => job.state === 'queued').length;
  const bits = [`${roots.length} ${roots.length === 1 ? 'hero' : 'heroes'}`];
  if (running) bits.push(`${running} running`);
  if (waiting) bits.push(`${waiting} waiting`);
  return bits.join(' · ');
}

// ---- DOM -----------------------------------------------------------------------------------------------

function icon(name, cls = 'icon icon--s') {
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', cls); svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(SVG, 'use');
  use.setAttribute('href', `/assets/icons/sprite.svg#i-${name}`);
  svg.append(use);
  return svg;
}
function node(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function card(entry, detail, now) {
  const { job, variations } = entry;
  const status = heroStatus(job, detail);
  const name = defaultName(job, detail);
  const link = node('a', `hero-card card card--interactive is-${status.tone}`);
  link.href = `/#job=${encodeURIComponent(job.id)}`;
  link.dataset.jobId = job.id;
  link.dataset.state = job.state;

  const thumb = node('div', 'hero-card__thumb');
  const url = thumbnailOf(job, detail);
  if (url) {
    const img = node('img', 'hero-card__img');
    img.alt = ''; img.loading = 'lazy'; img.decoding = 'async'; img.src = url;
    img.addEventListener('error', () => { img.remove(); thumb.classList.add('is-empty'); thumb.append(monogram(name)); }, { once: true });
    thumb.append(img);
  } else {
    thumb.classList.add('is-empty');
    thumb.append(monogram(name));
    if (status.active) thumb.classList.add('is-busy');
  }
  link.append(thumb);

  const body = node('div', 'hero-card__body');
  body.append(node('h2', 'hero-card__name', name));
  const meta = node('p', 'hero-card__meta');
  const pill = node('span', `hero-pill hero-pill--${status.tone}`);
  pill.append(icon(status.icon), node('span', '', status.label));
  const time = node('time', 'hero-card__time faint', relativeTime(job.created_at, now));
  if (job.created_at) { time.dateTime = new Date(job.created_at * 1000).toISOString(); time.title = new Date(job.created_at * 1000).toLocaleString(); time.dataset.at = String(job.created_at); }
  meta.append(pill, time);
  body.append(meta);
  const notes = [];
  if (status.note) notes.push(status.note);
  if (variations) notes.push(`${variations} ${variations === 1 ? 'variation' : 'variations'}`);
  if (notes.length) body.append(node('p', 'hero-card__note small muted', notes.join(' · ')));
  link.append(body);

  if (status.active && status.progress) {
    const line = node('span', 'hero-card__progress');
    line.setAttribute('role', 'img');
    line.setAttribute('aria-label', `${status.progress.done} of ${status.progress.total} stages complete`);
    for (const segment of status.progress.segments) line.append(node('span', `seg seg--${segment.kind}`));
    link.append(line);
  } else if (status.active) {
    const line = node('span', 'hero-card__progress hero-card__progress--indeterminate');
    line.setAttribute('aria-hidden', 'true'); line.append(node('span', 'seg seg--run'));
    link.append(line);
  }
  return link;
}

function signature({ job, variations }, detail) {
  const status = heroStatus(job, detail);
  return JSON.stringify([job.state, defaultName(job, detail), thumbnailOf(job, detail), status.label, status.note,
    status.progress ? status.progress.segments.map((segment) => segment.kind).join('') : '', variations, job.created_at]);
}

function monogram(name) {
  const letter = Array.from(String(name).trim())[0] || 'H';
  return node('span', 'hero-card__mono', letter.toLocaleUpperCase());
}

function skeletonCard() {
  const li = node('li', 'hero-item hero-item--skeleton');
  li.setAttribute('aria-hidden', 'true');
  const shell = node('div', 'hero-card card');
  const thumb = node('div', 'hero-card__thumb skeleton');
  const body = node('div', 'hero-card__body');
  body.append(node('span', 'skeleton hero-skel hero-skel--name'), node('span', 'skeleton hero-skel hero-skel--meta'));
  shell.append(thumb, body); li.append(shell);
  return li;
}

// heroesView($, {api, open, create, refresh}) wires the #heroLibrary region.
//   api(path)      GET helper from app.js (JSON, credentials, errors with .message)
//   open(id)       open a hero in the workspace
//   create()       go to the composer
//   refresh()      reload the list (the polling tick)
export function heroesView($, { api, open, create, refresh, clock = () => Date.now() / 1000 }) {
  const root = $('#heroLibrary'), grid = $('#jobs'), empty = $('#noJobs'), banner = $('#heroesBanner'),
    summary = $('#heroesSummary');
  const details = new Map();      // job id -> {state, data}
  let jobs = [], visible = false, timer = null, tick = null, generation = 0, status = 'idle';

  function setBusy(on) { grid.setAttribute('aria-busy', String(on)); }
  function showBanner(message) {
    banner.hidden = !message;
    if (message) $('#heroesBannerText').textContent = message;
  }

  function render() {
    if (status === 'loading' && !jobs.length) {
      grid.replaceChildren(...Array.from({ length: 6 }, skeletonCard));
      grid.hidden = false; empty.hidden = true; setBusy(true);
      summary.textContent = 'Loading your heroes…';
      return;
    }
    setBusy(false);
    const roots = rootHeroes(jobs);
    if (!roots.length) {
      grid.replaceChildren(); grid.hidden = true;
      empty.hidden = status === 'error'; root.classList.add('is-empty');
      summary.textContent = '';
      return;
    }
    root.classList.remove('is-empty');
    empty.hidden = true; grid.hidden = false;
    const now = clock();
    // Reconcile in place so a poll never rebuilds a card that did not change (keeps keyboard focus and images).
    roots.forEach((entry, index) => {
      const data = details.get(entry.job.id)?.data;
      const download = robloxDownload(entry.job, data);
      const sig = JSON.stringify([signature(entry, data), download]);
      const current = grid.children[index];
      if (current && current.dataset.key === entry.job.id && current.dataset.sig === sig) return;
      const li = node('li', 'hero-item');
      li.dataset.key = entry.job.id; li.dataset.sig = sig;
      li.append(card(entry, data, now));
      // A sibling of the card link (a link cannot hold another link), placed over the card's lower corner.
      if (download) {
        const get = node('a', 'hero-card__download btn btn--primary btn--s');
        get.href = download; get.setAttribute('download', '');
        get.setAttribute('aria-label', `Download ${defaultName(entry.job, data)} for Roblox (.zip)`);
        get.append(icon('download'), node('span', '', 'Roblox'));
        get.addEventListener('click', () => track('download_click', { format: 'roblox' }, { flush: true }));
        li.classList.add('has-download'); li.append(get);
      }
      if (!current) { grid.append(li); return; }
      const hadFocus = current.contains(document.activeElement);
      current.replaceWith(li);
      if (hadFocus) li.querySelector('a')?.focus({ preventScroll: true });
    });
    while (grid.children.length > roots.length) grid.lastElementChild.remove();
    for (const t of grid.querySelectorAll('time[data-at]')) t.textContent = relativeTime(Number(t.dataset.at), now);
    summary.textContent = summaryLine(roots);
  }

  async function fetchDetails() {
    const mine = ++generation;
    const roots = rootHeroes(jobs);
    const wanted = roots.map(({ job }, index) => ({ job, index })).filter(({ job, index }) => needsDetail(job, index))
      .filter(({ job }) => {
        const cached = details.get(job.id);
        return stillWorking(job) || !cached || cached.state !== job.state || cached.data?.export_state === 'running';
      });
    if (!wanted.length) return;
    await Promise.all(wanted.map(async ({ job }) => {
      try { details.set(job.id, { state: job.state, data: await api(`/api/jobs/${encodeURIComponent(job.id)}`) }); }
      catch { /* a card without detail still shows its summary */ }
    }));
    if (mine === generation && visible) render();
  }

  function schedule() {
    clearTimeout(timer);
    if (!visible || document.hidden || !jobs.some(stillWorking)) return;
    timer = setTimeout(() => { Promise.resolve(refresh()).catch(() => {}); }, 3000);
  }

  grid.addEventListener('click', (event) => {
    const link = event.target.closest('a.hero-card');
    if (!link || event.defaultPrevented || event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); open(link.dataset.jobId);
  });
  for (const id of ['#newHero', '#createFirstHero']) {
    $(id).addEventListener('click', (event) => {
      if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault(); create();
    });
  }
  $('#retryHeroes').addEventListener('click', () => { loading(); Promise.resolve(refresh()).catch(() => {}); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && visible) Promise.resolve(refresh()).catch(() => {}); else clearTimeout(timer); });

  function loading() { status = 'loading'; showBanner(''); render(); }
  function update(next) {
    jobs = Array.isArray(next) ? next : []; status = 'ready'; showBanner('');
    const ids = new Set(jobs.map((job) => job.id));
    for (const id of details.keys()) if (!ids.has(id)) details.delete(id);
    render();
    if (visible) { fetchDetails(); schedule(); }
  }
  function fail(error) {
    status = 'error'; setBusy(false);
    showBanner(error?.message || 'Your heroes could not be loaded.');
    if (!jobs.length) { grid.replaceChildren(); grid.hidden = true; empty.hidden = true; summary.textContent = ''; }
    schedule();
  }
  function setVisible(on) {
    visible = on; root.hidden = !on;
    clearInterval(tick); clearTimeout(timer);
    if (on) {
      tick = setInterval(() => { for (const t of grid.querySelectorAll('time[data-at]')) t.textContent = relativeTime(Number(t.dataset.at), clock()); }, 30000);
      if (status !== 'idle') { render(); fetchDetails(); schedule(); }
    }
  }
  return { loading, update, fail, setVisible, get visible() { return visible; } };
}
