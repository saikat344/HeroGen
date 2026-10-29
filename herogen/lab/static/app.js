// HeroGen Lab: the run folders of one character, read by the lab server (/api/workspace, /api/runs/<root>/<run>,
// /api/versions/<id>) and shown as they are. Every number on the page is a record's own or is counted from a file
// the run holds; anything the run does not hold is said to be missing, never filled in.
import * as THREE from 'three';
import { $, $$, chip, fill, fmt, fmtAuto, getJson, h, icon, loadIcons, mb, reducedMotion, runFile, secs,
  spaced, store, usd, versionFile, when, yaw } from './dom.js';
import { PIVOT_JOINT, POSES, SECTIONS, poseWord } from './poses.js';
import { CompareView, Snapshots, Viewer, instance, loadGlb } from './viewer.js';

const ICONS = ['circle-check', 'triangle-alert', 'circle-x', 'circle-dashed', 'circle-help', 'badge-check', 'ban',
  'eye', 'construction', 'crosshair', 'copy', 'check', 'lock', 'git-commit-horizontal', 'history', 'gauge', 'timer',
  'coins', 'package', 'file-image', 'file-text', 'layers', 'focus', 'columns-2', 'image', 'box', 'rotate-3d', 'sun',
  'moon', 'sparkles', 'pencil-line', 'palette', 'scale', 'info', 'loader-circle', 'alert-octagon',
  'hg-cam-front', 'hg-cam-34', 'hg-cam-side', 'hg-cam-back', 'hg-pose-rest', 'hg-pose-walk', 'hg-pose-arms-up',
  'hg-pose-arms-forward', 'hg-pose-crouch', 'hg-pose-twist', 'hg-shade-textured', 'hg-shade-clay', 'hg-shade-wire',
  'hg-shade-sections', 'hg-profile-ingame', 'hg-profile-avatar', 'hg-file-glb', 'hg-file-rbxm', 'hg-turnaround',
  'hg-heldout'];
const ILLUSTRATIONS = ['hg-body-pending', 'hg-compare-versions', 'hg-offline', 'hg-run-failed', 'hg-no-heroes'];

const PROFILES = ['best', 'roblox-ingame', 'roblox-avatar', 'none'];
const PROFILE_LABEL = { best: 'Best quality', 'roblox-ingame': 'Roblox in-game', 'roblox-avatar': 'Roblox avatar',
  none: 'Profile not recorded' };
const profileKey = (p) => (p === null || p === undefined ? 'none' : p);
const MODEL_NAMES = { 'muse-image-1.0': 'Meta Muse Image', 'sam-3.1': 'SAM 3.1', 'sam-3d-objects': 'SAM 3D Objects',
  'sam-3d-body': 'SAM 3D Body' };
const STAGE_NAMES = { 0: 'Sketch check', 1: 'Artwork', 2: 'Turnaround views', '2h': 'Held-out views', 3: 'Base mesh',
  4: 'Game body', 5: 'Results' };
const GROUP_ICON = { Artwork: 'palette', Views: 'hg-turnaround', Body: 'box', Texture: 'hg-shade-textured',
  'Held-out': 'hg-heldout' };
const VIEW_NAMES = { front: 'Front', back: 'Back', left: 'Left', right: 'Right' };
// Pictures the viewer can show, with the mask that frames the body over each one.
const PICTURES = {
  sketch: { img: 'sketch', mask: 'sketch_mask', label: 'Sketch', file: '00-sketch.png', dir: 'front', exact: false },
  cover: { img: 'cover', mask: 'cover_mask', label: 'Artwork', file: '01-cover.png', dir: 'front', exact: false },
  front: { img: 'view_front', mask: 'mask_front', label: 'Front view', file: 'views/front.png', dir: 'front', exact: true },
  right: { img: 'view_right', mask: 'mask_right', label: 'Right view', file: 'views/right.png', dir: 'right', exact: true },
  back: { img: 'view_back', mask: 'mask_back', label: 'Back view', file: 'views/back.png', dir: 'back', exact: true },
  left: { img: 'view_left', mask: 'mask_left', label: 'Left view', file: 'views/left.png', dir: 'left', exact: true },
};

const S = {
  ws: null, char: null, profile: null, runId: null, run: null, lod: 'high', shading: 'textured', pose: 'rest',
  mode: '3d', img: 'sketch', iso: null, focus: null, view: 'q', cmpSel: [], cmpView: 'q',
};
const details = new Map();
let COPY = { states: {}, icons: {} };
let SEC_INFO = {};
let viewer = null;
let snaps = null;
let cmp = null;
let inst = null;      // the body on screen
let restInst = null;  // the same body at rest, for snapshots
let bodyThumb = null;
let loadToken = 0;
let runToken = 0;
let cmpToken = 0;

const copyName = (k) => (COPY.icons[k] || [])[0];
const charOf = (id) => S.ws.characters.find((c) => c.runs.includes(id));
const entry = (id) => S.ws.runs[id];
const mainBody = () => (S.run ? S.run.bodies.find((b) => b.dir === '04-body') || null : null);
const sectionsOf = (body) => (body && body.sections && !body.sections.error ? body.sections : null);
const pressed = (sel, v) => $$(`${sel} button`).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
const illo = (name, ...text) => h('div', { class: 'illo' }, icon(name, 'illo-svg'), ...text);

function getDetail(id) {
  if (!details.has(id)) {
    const p = getJson(`/api/runs/${id.split('/').map(encodeURIComponent).join('/')}`);
    p.catch(() => details.delete(id));
    details.set(id, p);
  }
  return details.get(id);
}

// ---------------------------------------------------------------------------------------------------------------
// Masks: the figure's box in a mask image (white figure on black), measured here from the file.

const boxes = new Map();
function maskBox(url) {
  if (!boxes.has(url)) {
    boxes.set(url, new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => {
        const c = document.createElement('canvas');
        c.width = im.naturalWidth; c.height = im.naturalHeight;
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(im, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data;
        let x0 = c.width; let y0 = c.height; let x1 = -1; let y1 = -1;
        for (let y = 0; y < c.height; y += 1) {
          for (let x = 0; x < c.width; x += 1) {
            if (d[(y * c.width + x) * 4] > 127) {
              if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
            }
          }
        }
        if (x1 < 0) reject(new Error('the mask is empty'));
        else resolve({ w: c.width, h: c.height, x0, y0, x1: x1 + 1, y1: y1 + 1 });
      };
      im.onerror = () => reject(new Error(`could not load ${url}`));
      im.src = url;
    }));
    boxes.get(url).catch(() => boxes.delete(url));
  }
  return boxes.get(url);
}

// ---------------------------------------------------------------------------------------------------------------
// Header, profile strip, banner

function renderHeader() {
  const c = S.char;
  const sel = $('#charSel');
  fill(sel, S.ws.characters.map((x) => h('option', { value: x.key, text: `${x.title} · ${x.runs.length} run${x.runs.length > 1 ? 's' : ''}` })));
  sel.value = c.key;
  sel.hidden = S.ws.characters.length < 2;
  $('#heroName').hidden = !sel.hidden;
  $('#heroName').textContent = c.title;
  const unread = !c.sketch_sha && c.runs.every((id) => entry(id).error);
  $('#heroSub').textContent = [c.label, c.sketch_sha ? `sketch ${c.sketch_sha.slice(0, 8)}` : unread ? "the run's folder can't be read" : 'no 00-sketch.png in this run',
    `${c.runs.length} run${c.runs.length > 1 ? 's' : ''}`].filter(Boolean).join(' · ');
  const st = S.run ? S.run.quality.status : { kind: 'idle', word: 'Loading' };
  const src = borrowedFrom();
  fill('#heroStatus', chip(st.kind, st.word, { big: true, title: [st.detail, src ? `held-out checks from run ${src}` : null].filter(Boolean).join('; ') }),
    src ? h('small', { class: 'status-src', text: `held-out checks from run ${src}` }) : null);
}

// The run whose held-out record this run shows, when it is another run's (same sketch, identical turnaround).
const borrowedFrom = () => (S.run && S.run.heldout_source && S.run.heldout_source !== S.runId ? S.run.heldout_source : null);

function renderBanner() {
  const run = S.run;
  const problems = [];
  if (run) {
    for (const s of run.stages) if (s.status === 'unreadable') problems.push(s.error);
    for (const b of run.bodies) {
      if (b.sections && b.sections.error) problems.push(`${b.dir}/${b.sections.error}`);
      for (const [lod, g] of Object.entries(b.glb || {})) if (g.error) problems.push(`${b.dir}/${lod}.glb ${g.error}`);
    }
    if (run.scores && run.scores.error) problems.push(run.scores.error);
    if (run.heldout && run.heldout.error) problems.push(run.heldout.error);
    if (run.view_meta && run.view_meta.error) problems.push(`views/${run.view_meta.error}`);
  }
  const el = $('#banner');
  el.hidden = !problems.length;
  if (problems.length) {
    fill(el, icon('alert-octagon'), h('div', {}, h('b', { text: "Some of this run's files can't be read. " }),
      'Everything else is shown as read. ', h('span', { class: 'muted', text: problems.join('; ') })));
  }
}

function renderProfiles() {
  const c = S.char;
  for (const b of $$('#profileSeg button')) {
    const key = b.dataset.v;
    const runs = c.profiles[key === 'none' ? 'null' : key] || [];
    b.disabled = !runs.length;
    b.title = runs.length ? `${runs.length} run${runs.length > 1 ? 's' : ''}` : 'No run of this character records this profile';
    if (key === 'none') b.hidden = !runs.length;
  }
  pressed('#profileSeg', S.profile);
  const ids = c.profiles[S.profile === 'none' ? 'null' : S.profile] || [];
  fill('#runSel', ids.map((id) => {
    const r = entry(id);
    const bits = [r.name, r.root, r.started ? when(r.started) : null, r.error ? "can't read" : r.has_body ? null : 'no body'];
    return h('option', { value: id, text: bits.filter(Boolean).join(' · '), title: id });
  }));
  $('#runSel').value = S.runId;
}

function meter(label, value, limit) {
  const over = value !== null && limit !== null && value > limit;
  const pct = value === null || !limit ? 0 : Math.min(100, (value / limit) * 100);
  return h('div', { class: 'meter' }, h('span', { text: label }),
    h('span', { class: 'bar', role: 'img', 'aria-label': `${fmt(value)} of ${fmt(limit)} triangles` },
      h('i', { class: over ? 'over' : '', style: { width: `${pct.toFixed(1)}%` } })),
    h('span', { class: 'num' }, `${fmt(value)} / ${fmt(limit)} `,
      value === null ? chip('idle', 'Not counted') : over ? chip('bad', 'Over', { ic: 'alert-octagon' }) : chip('ok', 'Within', { ic: 'check' })));
}

function renderMeters() {
  const m = S.run ? S.run.budget_meters : { kind: 'unknown', items: [] };
  if (m.kind === 'unreadable') {
    fill('#meters', h('span', { class: 'small muted', text: `04-body/${m.error}, so its triangles can't be checked against a budget.` }));
  } else if (m.kind === 'unknown') {
    fill('#meters', h('span', { class: 'small muted', text: 'No body in this run, so nothing to count against a budget.' }));
  } else if (m.kind === 'none') {
    fill('#meters', h('span', { class: 'small muted', text: "This run's body records no triangle budget." }),
      m.items.map((i) => h('span', { class: 'num', text: `${i.label} ${fmt(i.value)} triangles` })));
  } else {
    fill('#meters', m.items.map((i) => meter(i.label, i.value, i.limit)));
  }
  $('#meters').classList.toggle('many', m.items.length > 3);
}

// ---------------------------------------------------------------------------------------------------------------
// Pipeline strip

function gates(stage) {
  const g = (stage && stage.checks || []).filter((c) => c.gate);
  return [g.filter((c) => c.pass).length, g.length];
}

// A stage's chip from its gates: unreadable, not run, no gates (checks reported only), pass or needs attention.
// Zero gates is never a pass, as in the quality panel.
function gateChip(stages) {
  if (stages.some((s) => s.status === 'unreadable')) return chip('bad', "Can't read");
  const ran = stages.filter((s) => s.status !== 'not run');
  if (!ran.length) return chip('idle', 'Not run');
  const [a, b] = ran.map(gates).reduce((x, y) => [x[0] + y[0], x[1] + y[1]], [0, 0]);
  if (!b) return chip('idle', ran.some((s) => (s.checks || []).length) ? 'No gates' : 'Not run');
  return a === b ? chip('ok', 'Pass') : chip('warn', 'Needs attention');
}

function heldoutCounts() {
  const views = (S.run.heldout && S.run.heldout.views) || [];
  const n = (grp) => views.filter((v) => v.group === grp).length;
  return { views, accepted: n('accepted'), review: n('needs review'), rejected: n('rejected'), failed: n('failed checks'),
    unrecorded: n('not recorded') };
}

function renderPipeline() {
  const run = S.run;
  const st = Object.fromEntries(run.stages.map((s) => [s.stage, s]));
  const img = (k) => (run.images[k] ? runFile(S.runId, run.images[k].path) : null);
  const models = (s) => [...(s.models || []).map((m) => MODEL_NAMES[m] || m)].join(', ');
  const gatesText = (s) => { const [a, b] = gates(s); return b ? `${a} of ${b} gates` : (s.status === 'not run' ? 'no record' : 'no gates'); };
  const ho = heldoutCounts();
  const hoSrc = S.run.heldout_source;
  const firstHo = ho.views.find((v) => v.group === 'accepted' && v.image) || ho.views.find((v) => v.image);
  const minSide = (st['0'].checks || []).find((c) => c.name === 'min_side_px');
  const nViews = ['front', 'back', 'left', 'right'].filter((v) => run.images[`view_${v}`]).length;
  const body = mainBody();
  const tris = body && body.glb && body.glb.high ? body.glb.high.triangles : null;
  const bodyStatus = body ? gateChip([st['3'], st['4']]) : chip('idle', 'Not run');
  const hoStatus = !ho.views.length ? chip('idle', 'Not run')
    : ho.rejected || ho.review || ho.failed ? chip('warn', 'Needs attention') : ho.unrecorded ? chip('idle', 'Not recorded') : chip('ok', 'Pass');
  const rows = [
    { name: 'Sketch', ic: 'pencil-line', img: img('sketch'), chip: gateChip([st['0']]),
      sub: minSide && typeof minSide.value === 'number' ? `${fmt(minSide.value)} px shortest side` : gatesText(st['0']),
      act: () => { setMode('sketch'); setImg('sketch'); } },
    { name: 'Artwork', ic: 'palette', img: img('cover'), chip: gateChip([st['1']]),
      sub: [gatesText(st['1']), models(st['1'])].filter(Boolean).join(' · '), act: () => { setMode('side'); setImg('cover'); } },
    { name: 'Views', ic: 'hg-turnaround', img: img('turnaround'), contain: true, chip: gateChip([st['2']]),
      sub: `${gatesText(st['2'])} · ${nViews} view${nViews === 1 ? '' : 's'}`, act: () => { setMode('side'); setImg('front'); } },
    { name: 'Held-out views', ic: 'hg-heldout', img: firstHo ? runFile(hoSrc, firstHo.image) : null, contain: true, chip: hoStatus,
      sub: ho.views.length ? [ho.accepted && `${ho.accepted} accepted`, ho.review && `${ho.review} to review`,
        ho.rejected && `${ho.rejected} rejected`, ho.failed && `${ho.failed} failed checks`, ho.unrecorded && `${ho.unrecorded} without a record`,
        borrowedFrom() && `from run ${borrowedFrom()}`].filter(Boolean).join(' · ') : 'none',
      act: () => showMe({ type: 'heldout' }) },
    { name: '3D body', ic: 'box', img: bodyThumb, body: true, chip: bodyStatus,
      sub: body ? `${(() => { const a = gates(st['3']); const b = gates(st['4']); return `${a[0] + b[0]} of ${a[1] + b[1]} gates`; })()}${tris !== null ? ` · ${fmt(tris)} tris` : ''}` : 'stage 4 has not built a body',
      act: () => setMode('3d') },
  ];
  $('#pipeRun').textContent = `run ${S.runId}`;
  const word = (c) => c.querySelector('.w').textContent;
  fill('#pipe', rows.map((r) => h('li', {}, h('button', { type: 'button', onclick: r.act, 'aria-label': `${r.name}, ${word(r.chip)}, ${r.sub}` },
    h('span', { class: `thumb${r.contain ? ' contain' : ''}`, id: r.body ? 'bodyThumb' : null,
      style: r.img ? { backgroundImage: `url("${r.img}")` } : null }, r.img ? null : icon(r.ic)),
    h('span', { class: 'meta' }, h('span', { class: 'name' }, icon(r.ic), r.name), r.chip, h('span', { class: 'sub', text: r.sub }))))));
}

// ---------------------------------------------------------------------------------------------------------------
// Versions list

function versionItems() {
  const c = S.char;
  const items = [];
  // This character's versions, then the versions no character could take, each with the reason.
  for (const v of [...c.versions, ...S.ws.unattached_versions]) {
    items.push({ id: `v:${v.id}`, kind: v.unattached_reason ? 'apart' : 'version', label: v.label, v, why: v.unattached_reason,
      usable: Boolean(v.high) && !v.error,
      lines: [when(v.built), [v.profile, v.triangles != null ? `${fmt(v.triangles)} tris` : null].filter(Boolean).join(' · '),
        [`mesh ${v.mesh_source || '–'}`]],
      code: commitOf(v.code), codeText: v.code, frozen: v.frozen, integrity: v.integrity, error: v.error });
  }
  for (const id of c.runs) {
    const r = entry(id);
    for (const dir of r.body_dirs) {
      const extra = dir === '04-body' ? '' : ` · ${dir.replace(/^04-body-/, '')}`;
      items.push({ id: `b:${id}:${dir}`, kind: 'run', runId: id, dir, label: `${id}${extra}`, usable: true,
        lines: [when(r.started), [PROFILE_LABEL[profileKey(r.profile)], dir === '04-body' && r.triangles !== null ? `${fmt(r.triangles)} tris` : null].filter(Boolean).join(' · '),
          [`mesh ${dir === '04-body' ? r.mesh_source || '–' : dir.replace(/^04-body-/, '')}`]],
        code: r.code ? `${r.code.commit.slice(0, 7)}${r.code.dirty ? ' (uncommitted changes)' : ''}` : null });
    }
  }
  return items;
}

// A version's `code` is a sentence; show the commit it names (the sentence stays in the tooltip).
function commitOf(text) {
  if (!text) return null;
  const m = /\b(?:commit\s+)?([0-9a-f]{7,40})\b/.exec(text);
  return m ? m[1].slice(0, 7) : text.length > 40 ? `${text.slice(0, 38)}…` : text;
}

function renderVersions() {
  const items = versionItems();
  const block = (title, list) => (list.length ? [h('div', { class: 'eyebrow', text: title }), h('ul', { class: 'versions' }, list.map((it) => {
    const on = S.cmpSel.includes(it.id);
    const current = it.kind === 'run' && it.runId === S.runId && it.dir === '04-body';
    const integ = it.integrity && it.integrity.length ? (it.integrity.every((f) => f.matches !== false && f.present)
      ? h('span', { class: 'vm' }, icon('check'), 'files match their recorded SHA-256')
      : h('span', { class: 'vm' }, icon('triangle-alert'), `${it.integrity.filter((f) => !f.present || f.matches === false).length} file(s) missing or changed since frozen`)) : null;
    return h('li', {}, h('label', { class: `${on ? 'on' : ''}${current ? ' current' : ''}` },
      h('input', { type: 'checkbox', id: `ver-${it.id}`, 'data-id': it.id, checked: on || null, disabled: !it.usable || null,
        onchange: (e) => toggleCompare(it.id, e.target.checked) }),
      h('span', {}, h('span', { class: 'vn', text: it.label }),
        it.why ? h('span', { class: 'vm why' }, icon('triangle-alert'), `Not attached: ${it.why}`) : null,
        it.error && !it.why ? h('span', { class: 'vm', text: it.error }) : null,
        it.lines.map((l) => h('span', { class: 'vm', text: Array.isArray(l) ? l.join(' ') : l })),
        h('span', { class: 'vm', title: it.codeText || null }, icon('git-commit-horizontal'), it.code || 'build commit not recorded',
          it.frozen ? [' · ', icon('lock'), `frozen ${it.frozen}`] : null),
        integ)));
  }))] : []);
  const frozen = items.filter((i) => i.kind === 'version');
  const runs = items.filter((i) => i.kind === 'run');
  const apart = items.filter((i) => i.kind === 'apart');
  fill('#versions', block('Frozen versions', frozen), block('Runs', runs), block('Versions of no character here', apart),
    items.length ? null : h('p', { class: 'empty', text: 'No bodies in this character yet.' }));
}

function toggleCompare(id, on) {
  S.cmpSel = S.cmpSel.filter((x) => x !== id);
  if (on) { S.cmpSel.push(id); if (S.cmpSel.length > 2) S.cmpSel.shift(); }
  renderVersions();
  renderCompare();
}

// ---------------------------------------------------------------------------------------------------------------
// The body in the viewer

function sectionInfo(body) {
  const sec = sectionsOf(body);
  const parents = {};
  if (sec) for (const s of sec.sections) if (s.name) parents[s.name] = s.parent;
  return { parents, joints: sec ? sec.joints : null, ends: sec ? sec.ends : null };
}

function vpMessage(...kids) {
  const el = $('#vpload');
  el.hidden = !kids.length;
  fill(el, ...kids);
}

async function showBody(resetCamera = false) {
  const token = ++loadToken;
  const body = mainBody();
  S.focus = null;
  if (!body) {
    viewer.clear(); inst = null; restInst = null; bodyThumb = null;
    vpMessage(illo('hg-body-pending', h('p', { text: COPY.states['body-pending'] || '' }),
      h('p', { class: 'small', text: 'This run has no 04-body/high.glb or low.glb.' })));
    renderStats(); renderSectionSelect(); updateSecCard(); renderLegend(); renderPipeline();
    return;
  }
  if (!body[S.lod]) S.lod = body.high ? 'high' : 'low';
  const path = body[S.lod];
  vpMessage(icon('loader-circle', 'ic spin'), `Loading ${path}…`);
  let gltf;
  try {
    gltf = await loadGlb(runFile(S.runId, path));
  } catch (e) {
    if (token !== loadToken) return;
    viewer.clear(); inst = null;
    // A file the server sends but three.js cannot parse is a broken body file, not a lost connection.
    const served = await fetch(runFile(S.runId, path), { method: 'HEAD' }).then((r) => r.ok, () => false);
    if (token !== loadToken) return;
    vpMessage(served
      ? illo('hg-run-failed', h('p', { text: `Couldn't read this body file (${path.split('/').pop()}).` }), h('p', { class: 'small', text: `${path}: ${e.message || e}` }))
      : illo('hg-offline', h('p', { text: COPY.states.offline || '' }), h('p', { class: 'small', text: `Couldn't load ${path}.` })));
    renderStats();
    return;
  }
  if (token !== loadToken) return;
  const info = sectionInfo(body);
  inst = instance(gltf, info);
  restInst = instance(gltf, info);
  S.iso = null;
  viewer.show(inst);
  viewer.setPose(S.pose, true);
  if (resetCamera) viewer.preset(S.view, true);
  vpMessage();
  renderStats(); renderSectionSelect(); updateSecCard(); renderLegend();
  if (S.mode === 'side') setImg(S.img);
  bodyThumb = snaps.render(restInst, 30);
  const bt = $('#bodyThumb');
  if (bt) { bt.style.backgroundImage = `url("${bodyThumb}")`; bt.replaceChildren(); }
  liveTiles();
}

function renderStats() {
  const body = mainBody();
  if (!inst || !body) { fill('#stats'); $('#lodpick').hidden = true; return; }
  const g = body.glb[S.lod] || {};
  fill('#stats', h('span', { class: 'num', text: `${fmt(inst.total)} triangles` }),
    h('span', { text: `${inst.sectionCount} section${inst.sectionCount === 1 ? '' : 's'}` }),
    g.bytes ? h('span', { class: 'num', text: mb(g.bytes) }) : null,
    h('span', { class: 'faint', text: 'Y up · grid square 0.5 stud' }));
  $('#lodpick').hidden = false;
  const low = $$('#lodSeg button')[1];
  low.disabled = !body.low;
  const d = (S.run.deliverables || []).find((x) => x.kind === 'GLB low');
  low.title = body.low ? '' : (d && d.reason) || 'This body has no low level of detail';
  pressed('#lodSeg', S.lod);
}

function renderSectionSelect() {
  const sel = $('#secSel');
  fill(sel, h('option', { value: '', text: 'All sections' }), inst ? SECTIONS.filter((n) => inst.nodes[n]).map((n) =>
    h('option', { value: n, text: `${SEC_INFO[n] ? SEC_INFO[n].id + ' · ' : ''}${n} · ${fmt(inst.tris[n])} tris` })) : []);
  sel.value = S.iso || '';
}

function swatch(n) {
  const x = SEC_INFO[n] || {};
  return h('span', { class: `sw p-${x.pattern || 'dot'}`, style: { backgroundColor: x.color || '#888' }, 'aria-hidden': 'true' });
}

function renderLegend() {
  const el = $('#legend');
  el.hidden = S.shading !== 'sections' || !inst;
  if (el.hidden) return;
  fill(el, SECTIONS.filter((n) => inst.nodes[n]).map((n) => h('li', {}, h('button', {
    type: 'button', 'aria-pressed': String(S.iso === n), onclick: () => isolate(S.iso === n ? null : n) },
  h('span', { class: 'n', text: SEC_INFO[n] ? SEC_INFO[n].id : '' }), swatch(n), h('span', { text: n })))));
}

function isolate(n) {
  S.iso = n; S.focus = null;
  viewer.setHot(n ? [n] : null);
  $('#secSel').value = n || '';
  updateSecCard(); renderLegend();
}

function gapTable(lod, heldoutPoses = false) {
  const sec = sectionsOf(mainBody());
  const pt = sec && sec.pose_test;
  return pt ? pt[`gaps_${lod}${heldoutPoses ? '_heldout' : ''}`] : null;
}

function updateSecCard() {
  const card = $('#secCard');
  if (!inst || (!S.iso && !S.focus)) { card.hidden = true; return; }
  const sec = sectionsOf(mainBody());
  const pt = sec && sec.pose_test;
  const clear = h('button', { class: 'btn', type: 'button', onclick: () => isolate(null) }, icon('layers'), h('span', { text: 'Show all sections' }));
  if (S.focus) {
    const f = S.focus;
    const table = gapTable(f.lod, !(gapTable(f.lod) || {})[f.pose]);
    const recorded = table && table[f.pose] ? table[f.pose][f.joint] : null;
    const words = spaced(f.joint).toLowerCase();
    fill(card, h('h3', {}, icon('crosshair'), `${words[0].toUpperCase()}${words.slice(1)} joint`),
      h('dl', { class: 'kv' },
        h('dt', { text: 'Sections' }), h('dd', { text: `${f.child} / ${f.parent || '–'}` }),
        h('dt', { text: 'Pose' }), h('dd', { text: `${poseWord(f.pose)}${f.pose !== S.pose ? ` (shown at ${poseWord(S.pose)})` : ''}` }),
        h('dt', { text: 'Gap' }), h('dd', { text: recorded === null || recorded === undefined ? '–' : `${fmt(recorded, 4)} studs (${f.lod} body)` }),
        h('dt', { text: 'Limit' }), h('dd', { text: pt && pt.gate_studs !== null ? `${fmtAuto(pt.gate_studs)} studs` : '–' })),
      h('p', { class: 'faint', text: f.inViewer ? "The worst gap of the check, from the record's pose test." :
        `The viewer can show ${POSES.length} of the test poses; ${f.pose} is not one of them, so the joint is shown at rest.` }),
      clear);
  } else {
    const n = S.iso;
    const s = sec ? sec.sections.find((x) => x.name === n) || {} : {};
    const p = inst.pivots[n];
    const joint = s.joint || PIVOT_JOINT[n];
    const table = gapTable(S.lod);
    const gap = S.pose !== 'rest' && table && table[S.pose] ? table[S.pose][joint] : undefined;
    const x = SEC_INFO[n] || {};
    fill(card, h('h3', {}, icon('focus'), swatch(n), `${x.id ? x.id + ' · ' : ''}${n}`),
      h('dl', { class: 'kv' },
        h('dt', { text: 'Triangles' }), h('dd', { text: `${fmt(inst.tris[n])} (${S.lod}, counted)` }),
        s[`triangles_${S.lod}`] !== undefined && s[`triangles_${S.lod}`] !== null ? [h('dt', { text: 'Recorded' }), h('dd', { text: fmt(s[`triangles_${S.lod}`]) })] : null,
        h('dt', { text: 'Pivot' }), h('dd', { text: p ? `${fmt(p.x, 3)}, ${fmt(p.y, 3)}, ${fmt(p.z, 3)}` : '–' }),
        h('dt', { text: 'Joint' }), h('dd', { text: `${joint || '–'}${inst.parent[n] ? ` → ${inst.parent[n]}` : ' (root)'}` }),
        gap !== undefined ? [h('dt', { text: 'Gap now' }), h('dd', { text: `${gap === null ? '–' : fmt(gap, 4) + ' studs'} in ${poseWord(S.pose)}` })] : null),
      h('p', { class: 'faint', text: 'Pivot in studs, glTF axes (Y up, front +Z).' }), clear);
  }
  card.hidden = false;
}

function setPose(p, instant = false) {
  S.pose = p;
  pressed('#poseSeg', p);
  viewer.setPose(p, instant);
  updateSecCard();
}

function setShading(v) {
  S.shading = v;
  pressed('#shadeSeg', v);
  viewer.setShading(v);
  renderLegend();
}

function setCamera(v) {
  if (v === 'orbit') {
    viewer.controls.autoRotate = !viewer.controls.autoRotate;
    $$('#camSeg button')[0].setAttribute('aria-pressed', String(viewer.controls.autoRotate));
    return;
  }
  S.view = v;
  viewer.preset(v);
  $$('#camSeg button').forEach((b) => { if (b.dataset.v !== 'orbit') b.setAttribute('aria-pressed', String(b.dataset.v === v)); });
}

// Views: 3D, a picture, or a picture beside the body through the same camera.
function availablePictures(needMask) {
  if (!S.run) return [];
  return Object.entries(PICTURES).filter(([, p]) => S.run.images[p.img] && (!needMask || S.run.images[p.mask])).map(([k]) => k);
}

function renderImgSelect() {
  const keys = availablePictures(S.mode === 'side');
  fill('#imgSel', keys.map((k) => h('option', { value: k, text: PICTURES[k].label })));
  if (!keys.includes(S.img) && keys.length) S.img = keys[0];
  $('#imgSel').value = S.img;
  $('#imgSel').disabled = !keys.length;
}

function setMode(m) {
  S.mode = m;
  pressed('#modeSeg', m);
  $('#vp').classList.toggle('side', m === 'side');
  $('#imgpane').hidden = m === '3d';
  $('#glpane').hidden = m === 'sketch';
  $('#imgpick').hidden = m === '3d';
  $$('#camSeg button').forEach((b) => { b.disabled = m !== '3d'; });
  $('#glcap').hidden = m !== 'side';
  renderImgSelect();
  if (m !== 'side') viewer.setMode('3d');
  setImg(S.img);
  requestAnimationFrame(() => viewer.resize());
}

async function setImg(k) {
  S.img = k;
  $('#imgSel').value = k;
  const p = PICTURES[k];
  const im = S.run && p ? S.run.images[p.img] : null;
  if (!im) {
    $('#refimg').removeAttribute('src');
    $('#imgcap').textContent = 'No picture of this kind in the run';
    return;
  }
  $('#refimg').src = runFile(S.runId, im.path);
  $('#refimg').alt = `${p.label} (${im.path})`;
  $('#imgcap').textContent = `${p.label} (${im.path})`;
  if (im.px) $('#vp').style.setProperty('--ar', `${im.px[0]} / ${im.px[1]}`);
  if (S.mode !== 'side') return;
  const mask = S.run.images[p.mask];
  const cap = $('#glcap');
  if (!mask || !inst) { cap.textContent = inst ? `No mask for the ${p.label.toLowerCase()}, so the body can't be framed to it.` : 'No body to frame.'; return; }
  try {
    const box = await maskBox(runFile(S.runId, mask.path));
    if (S.mode !== 'side' || S.img !== k) return;
    const pps = p.exact && S.run.view_meta && S.run.view_meta.pixels_per_unit;
    const scale = pps || (box.y1 - box.y0) / (inst.box.max.y - inst.box.min.y);
    const xc = (box.x0 + box.x1) / 2;
    viewer.setMode('side', { dir: p.dir, left: -xc / scale, right: (box.w - xc) / scale, top: box.y1 / scale,
      bottom: (box.y1 - box.h) / scale });
    cap.textContent = pps
      ? `Body through the same camera as the ${p.label.toLowerCase()}: orthographic, ${fmtAuto(pps)} px per stud (views/meta.json), feet on the mask's bottom edge`
      : `Body from the ${p.dir}, scaled to the figure in ${mask.path} (approximate: the drawing has no calibrated camera)`;
  } catch (e) {
    cap.textContent = `Couldn't measure ${mask.path}: ${e.message || e}`;
  }
}

// "Show me": jump to what a failing check points at.
async function showMe(act) {
  if (!act) return;
  const smooth = reducedMotion() ? 'auto' : 'smooth';
  if (act.type === 'heldout') {
    const panel = $('#heldoutPanel');
    panel.scrollIntoView({ behavior: smooth, block: 'start' });
    panel.focus({ preventScroll: true });
    if (act.view) {
      const card = $(`[data-view="${CSS.escape(act.view)}"]`, panel);
      if (card) card.scrollIntoView({ behavior: smooth, block: 'nearest' });
    }
    return;
  }
  $('#stage').scrollIntoView({ behavior: smooth, block: 'start' });
  if (act.type === 'view') {
    setShading('textured'); setPose('rest', false); isolate(null);
    setMode('side'); setImg(act.view);
    return;
  }
  if (act.type === 'joint') {
    if (S.mode !== '3d') setMode('3d');
    const body = mainBody();
    if (act.lod && act.lod !== S.lod && body && body[act.lod]) { S.lod = act.lod; await showBody(); }
    if (!inst) return;
    const sec = sectionsOf(body);
    const child = (sec && (sec.sections.find((s) => s.joint === act.joint) || {}).name)
      || Object.keys(PIVOT_JOINT).find((n) => PIVOT_JOINT[n] === act.joint);
    if (!child || !inst.nodes[child]) return;
    const parent = inst.parent[child];
    setPose(act.pose_in_viewer ? act.pose : 'rest');
    S.iso = null;
    S.focus = { child, parent, joint: act.joint, pose: act.pose, lod: act.lod || S.lod, inViewer: act.pose_in_viewer };
    viewer.setHot([child, parent].filter(Boolean));
    const jp = viewer.posedPivot(child);
    if (jp) viewer.flyTo(jp.clone().add(new THREE.Vector3(0.55, 0.3, 1).normalize().multiplyScalar(7.5)), jp);
    renderLegend();
    updateSecCard();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Quality

function target(c) {
  if (c.threshold === null || c.threshold === undefined) return 'reported';
  return `${({ '>=': '≥', '<=': '≤' })[c.cmp] || c.cmp} ${fmtAuto(c.threshold)}`;
}

function renderQuality() {
  const q = S.run.quality;
  $('#qProfile').textContent = PROFILE_LABEL[profileKey(S.run.profile)];
  const fl = q.failures;
  const failLi = (f, i) => h('li', {}, h('span', { class: `sev${f.kind === 'warn' ? ' warn' : ''}` }), h('div', {},
    h('p', { class: 'ft', text: f.text }),
    h('div', { class: 'fa' }, h('span', { class: 'faint small', text: f.group }),
      f.act ? h('button', { class: 'btn', type: 'button', 'data-f': i, onclick: () => showMe(f.act) }, icon('crosshair'), h('span', { text: 'Show me' })) : null)));
  const linked = S.run.heldout_source && S.run.heldout_source !== S.runId;
  const groups = ['Artwork', 'Views', 'Body', 'Texture', 'Held-out'];
  const tableRows = [];
  for (const gn of groups) {
    const cs = q.checks.filter((c) => c.group === gn);
    if (!cs.length) continue;
    tableRows.push(h('tr', {}, h('th', { colspan: 4, class: 'grp', text: gn })));
    for (const c of cs) {
      tableRows.push(h('tr', {},
        h('td', { title: `${c.name} (stage ${c.stage})` }, c.label, (c.notes || []).map((n) => h('div', { class: 'faint small', text: n }))),
        h('td', { class: 'r num', text: fmtAuto(c.value) }),
        h('td', { class: 'num' }, target(c), c.gate ? null : h('div', { class: 'faint', text: 'not gated' })),
        h('td', {}, c.pass ? chip('ok', 'Pass') : c.gate ? chip('bad', 'Fail') : c.threshold === null ? chip('idle', 'Reported') : chip('warn', 'Below'))));
    }
  }
  const pct = (n) => (q.gates ? `${((n / q.gates) * 100).toFixed(1)}%` : '0%');
  fill('#qBody',
    h('div', { class: 'qhead' }, chip(q.status.kind, q.status.word, { big: true,
      title: [q.status.detail, borrowedFrom() ? `held-out checks from run ${borrowedFrom()}` : null].filter(Boolean).join('; ') }),
      h('div', { class: 'qcount' }, h('b', { text: String(q.passed) }), h('span', { class: 'muted', text: `of ${q.gates} gates passed` })),
      h('div', { class: 'gbar', role: 'img', 'aria-label': `${q.passed} of ${q.gates} gates passed` },
        h('i', { class: 'pass', style: { width: pct(q.passed) } }), h('i', { class: 'fail', style: { width: pct(q.gates - q.passed) } }))),
    h('ul', { class: 'groups' }, q.groups.map((g) => h('li', {}, h('span', { class: 'gn' }, icon(GROUP_ICON[g.name]), g.name),
      h('span', { class: 'num faint', text: g.gates ? `${g.gates - g.failed}/${g.gates}` : '' }),
      !g.gates ? chip('idle', g.checks ? 'No gates' : 'Not run') : g.failed ? chip(g.name === 'Held-out' ? 'warn' : 'bad', `${g.failed} failing`) : chip('ok', 'Pass')))),
    linked ? h('p', { class: 'faint small', text: `Held-out checks are from run ${S.run.heldout_source}, which has the same turnaround.` }) : null,
    fl.length ? [h('div', { class: 'eyebrow', text: 'Top failures' }), h('ul', { class: 'fails' }, fl.slice(0, 4).map((f, i) => failLi(f, i))),
      fl.length > 4 ? h('details', { class: 'all' }, h('summary', { text: `${fl.length - 4} more` }),
        h('ul', { class: 'fails', style: { marginTop: '8px' } }, fl.slice(4).map((f, i) => failLi(f, i + 4)))) : null] : null,
    h('details', { class: 'all' }, h('summary', { text: `All ${q.checks.length} checks, with ${q.reported_below} reported below target that don't gate` }),
      h('div', { class: 'tbl-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', { text: 'Check' }), h('th', { class: 'r', text: 'Value' }),
        h('th', { text: 'Target' }), h('th', {}, h('span', { class: 'sr', text: 'Result' })))), h('tbody', {}, tableRows)))));
}

// ---------------------------------------------------------------------------------------------------------------
// Held-out views

function currentScoreBody() {
  const sc = S.run.scores;
  const body = mainBody();
  if (!sc || !sc.bodies || !body) return null;
  return sc.bodies.find((b) => !b.variant_of && b.glb === body.high) || null;
}

function metrics(row) {
  return h('div', { class: 'metrics' },
    h('div', {}, h('b', { text: fmt(row.iou, 4) }), h('span', { text: 'Outline IoU' })),
    h('div', {}, h('b', { text: fmt(row.colour_error, 2) }), h('span', { text: 'Colour error' })),
    h('div', {}, h('b', { text: fmt(row.ssim, 4) }), h('span', { text: 'SSIM' })));
}

const VIEW_CHIP = {
  accepted: () => chip('ok', 'Accepted', { ic: 'badge-check' }),
  'passed checks': () => chip('ok', 'Passed checks'),
  'needs review': () => chip('warn', 'Needs review', { ic: 'eye' }),
  rejected: () => chip('warn', 'Rejected', { ic: 'ban' }),
  'failed checks': () => chip('bad', 'Failed checks'),
  'not recorded': () => chip('idle', 'Not recorded'),
};

function renderHeldout() {
  const held = S.run.heldout;
  const src = S.run.heldout_source;
  const sc = S.run.scores;
  const sbody = currentScoreBody();
  const scoredHeld = sc && sc.heldout_views ? sc.heldout_views : [];
  $('#hoSub').textContent = held && held.read_by ? held.read_by : '';
  const parts = [];
  if (held && src && src !== S.runId) {
    parts.push(h('p', { class: 'note', text: `From run ${src}: the same sketch and byte-identical turnaround views.` }));
  }
  const views = held ? held.views : [];
  const figure = (v) => {
    const row = sbody && sbody.views[v.name];
    return h('div', { class: 'ho-meta', 'data-view': v.name },
      h('div', { class: 'panel-h' }, h('h3', { text: v.label }), (VIEW_CHIP[v.status] || VIEW_CHIP['not recorded'])()),
      h('div', { class: 'ho-row' },
        h('figure', { class: 'ho-fig' }, h('div', { class: 'im', role: 'img', 'aria-label': `Held-out image ${v.label}`,
          style: v.image ? { backgroundImage: `url("${runFile(src, v.image)}")` } : null }, v.image ? null : 'image not in the run'),
        h('figcaption', { text: `Generated view, found yaw ${v.found_yaw_deg === null ? 'not recorded' : yaw(v.found_yaw_deg)}` })),
        v.found_yaw_deg !== null && mainBody() ? h('figure', { class: 'ho-fig' }, h('div', { class: 'im live', role: 'img',
          'aria-label': `Body rendered at ${yaw(v.found_yaw_deg)}`, 'data-yaw': v.found_yaw_deg,
          'data-mask': v.mask ? runFile(src, v.mask) : '' }), h('figcaption', { text: `Body at ${yaw(v.found_yaw_deg)}, rendered here` })) : null),
      v.status === 'needs review' || v.status === 'failed checks' ? h('p', { class: 'small muted', text: reason(v) }) : null,
      row ? [h('p', { class: 'eyebrow' }, icon('scale'), 'Scores against this view'), metrics(row)]
        : h('p', { class: 'small faint', text: 'Consistency scores not computed yet for this view.' }));
  };
  const reason = (v) => {
    const c = v.checks.find((x) => /status_accepted$/.test(x.name));
    return c && c.notes.length ? c.notes.join('; ') : `${v.checks_failed_gate} gate${v.checks_failed_gate === 1 ? '' : 's'} failing`;
  };
  for (const grp of ['accepted', 'needs review', 'failed checks', 'not recorded']) {
    for (const v of views.filter((x) => x.group === grp)) parts.push(figure(v));
  }
  const rej = views.filter((v) => v.group === 'rejected');
  if (rej.length) {
    parts.push(h('div', { class: 'ho-meta' }, h('div', { class: 'panel-h' }, h('h3', { text: 'Rejected' }),
      chip('warn', `${rej.length} view${rej.length === 1 ? '' : 's'}`, { ic: 'ban' })),
    rej.map((v) => h('div', { class: 'rej', 'data-view': v.name },
      h('div', { class: 'im', role: 'img', 'aria-label': `Rejected view ${v.label}`, style: v.image ? { backgroundImage: `url("${runFile(src, v.image)}")` } : null }),
      h('span', { class: 'num', text: yaw(v.found_yaw_deg) }),
      h('span', { class: 'small' }, `${v.label}: `, v.review ? `${v.review.decision || 'rejected'} on review${v.review.by ? ` by ${v.review.by}` : ''}${v.review.at ? `, ${when(v.review.at)}` : ''}${v.review.note ? `: ${v.review.note}` : ''}` : reason(v))))));
  }
  // Views the score file holds as held out that the held-out record does not list (an earlier held-out view).
  const listed = new Set(views.map((v) => v.name));
  for (const name of scoredHeld.filter((n) => !listed.has(n))) {
    const row = sbody && sbody.views[name];
    const info = sc.view_info[name] || {};
    const image = info.image || (name === sc.heldout_view ? sc.heldout_image : null);
    const render = sbody && sbody.renders[name];
    parts.push(h('div', { class: 'ho-meta', 'data-view': name },
      h('div', { class: 'panel-h' }, h('h3', { text: `${heldLabel(name)} (scored)` }), row ? chip('ok', 'Scored') : chip('idle', 'Not scored')),
      h('div', { class: 'ho-row' },
        h('figure', { class: 'ho-fig' }, h('div', { class: 'im', role: 'img', 'aria-label': `Held-out image ${name}`,
          style: image ? { backgroundImage: `url("${runFile(S.runId, image)}")` } : null }, image ? null : 'image not in the run'),
        h('figcaption', { text: image || 'no image' })),
        h('figure', { class: 'ho-fig' }, h('div', { class: 'im dark', role: 'img', 'aria-label': 'Scorer render of the body',
          style: render ? { backgroundImage: `url("${runFile(S.runId, render)}")` } : null }, render ? null : 'no scorer render'),
        h('figcaption', { text: `Scorer render${row && row.yaw_deg !== null ? ` at ${yaw(row.yaw_deg)}` : ''}` }))),
      row ? [h('p', { class: 'eyebrow' }, icon('scale'), `Scores against this view (${sc.file})`), metrics(row)] : null));
  }
  if (held && held.reference_note) parts.push(h('p', { class: 'note', text: held.reference_note }));
  if (!parts.length) parts.push(h('p', { class: 'empty', text: 'No held-out views in this run, or in a run with the same turnaround.' }));
  fill('#hoBody', parts);
  liveTiles();
}

const heldLabel = (name) => { const m = /^(front|back)-(left|right)-(\d+)$/.exec(name); return m ? `${m[1][0].toUpperCase()}${m[1].slice(1)}-${m[2]} ${m[3]}°` : name; };

async function liveTiles() {
  if (!restInst) return;
  const body = restInst;
  for (const el of $$('#hoBody .im.live')) {
    let box = null;
    if (el.dataset.mask) { try { box = await maskBox(el.dataset.mask); } catch (e) { box = null; } }
    if (body !== restInst) return;
    el.style.backgroundImage = `url("${snaps.render(body, Number(el.dataset.yaw), box)}")`;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Deliverables and generation details

function renderDeliv() {
  $('#dProfile').textContent = PROFILE_LABEL[profileKey(S.run.profile)];
  const items = S.run.deliverables || [];
  fill('#deliv', items.map((d, i) => {
    if (d.status !== 'built') {
      const c = d.status === 'not built' ? chip('idle', 'Not built yet', { ic: 'construction' })
        : d.status === 'not in profile' ? chip('idle', 'Not in profile') : chip('idle', 'Missing');
      const ic = /^GLB/.test(d.kind) ? 'hg-file-glb' : /^R15 model/.test(d.kind) ? 'hg-file-rbxm' : 'construction';
      return h('li', { class: 'nb' }, h('span', { class: 'dic' }, icon(ic)),
        h('div', {}, h('p', { class: 'dn', text: d.kind }), h('p', { class: 'small faint', text: d.reason })), h('span', { class: 'act' }, c));
    }
    const file = d.path.split('/').pop();
    const ext = file.split('.').pop().toLowerCase();
    const thumb = ext === 'png' ? runFile(S.runId, d.path) : null;
    const ic = ext === 'glb' ? 'hg-file-glb' : ext === 'png' ? 'file-image' : ext === 'rbxm' ? 'hg-file-rbxm' : 'file-text';
    const full = `${S.run.path}/${d.path}`;
    const code = h('code', { id: `path${i}`, text: full });
    const btn = h('button', { class: 'btn', type: 'button', 'aria-label': `Copy path ${full}` }, icon('copy'), h('span', { text: 'Copy path' }));
    btn.addEventListener('click', () => copyPath(btn, code));
    return h('li', {}, h('span', { class: 'dic', style: thumb ? { backgroundImage: `url("${thumb}")` } : null }, thumb ? null : icon(ic)),
      h('div', { style: { minWidth: '0' } }, h('p', { class: 'dn' }, `${d.kind} `, h('span', { class: 'faint mono small', text: file })),
        h('p', { class: 'small muted num', text: [mb(d.bytes), d.triangles !== undefined && d.triangles !== null ? `${fmt(d.triangles)} triangles` : null,
          d.px ? `${d.px[0]} × ${d.px[1]} px` : null].filter(Boolean).join(' · ') }), code),
      h('span', { class: 'act' }, btn));
  }));
}

function copyPath(btn, code) {
  const txt = code.textContent;
  const label = btn.lastElementChild;
  const select = () => {
    const r = document.createRange(); r.selectNodeContents(code);
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    label.textContent = 'Selected: press Copy';
  };
  const done = () => {
    btn.replaceChildren(icon('check'), h('span', { text: 'Copied' }));
    setTimeout(() => btn.replaceChildren(icon('copy'), h('span', { text: 'Copy path' })), 1600);
  };
  try { navigator.clipboard.writeText(txt).then(done, select); } catch (e) { select(); }
}

function renderGen() {
  const run = S.run;
  const gen = run.generation || { rows: [], total: {}, with_borrowed: null };
  const rows = gen.rows.map((r) => {
    const rec = r.record;
    const tools = [...(rec.models || []).map((m) => MODEL_NAMES[m] || m), ...(rec.tools || []).map((t) => t.replace(/^blender/i, 'Blender'))];
    const calls = rec.calls ? `${rec.calls.count} model call${rec.calls.count === 1 ? '' : 's'}${rec.calls.count ? `, ${rec.calls.replayed} replayed from the cache` : ''}` : null;
    return { key: r.stage, rec, from: r.from_run, tools, calls };
  });
  const max = Math.max(0, ...rows.map((r) => r.rec.seconds || 0));
  const totalRow = (label, t, note) => h('tr', {}, h('td', {}, h('b', { text: label }),
    t.seconds_missing || t.cost_usd_missing ? h('div', { class: 'faint small', text: 'over the stages that record a number' }) : null,
    note ? h('div', { class: 'faint small', text: note }) : null),
  h('td', { class: 'num' }, h('b', { text: secs(t.seconds) })), h('td', { class: 'r num' }, h('b', { text: usd(t.cost_usd) })));
  const borrowed = rows.find((r) => r.from);
  fill('#genTbl', h('thead', {}, h('tr', {}, h('th', { text: 'Stage' }), h('th', { text: 'Time' }), h('th', { class: 'r' }, icon('coins'), ' Est. cost'))),
    h('tbody', {}, rows.map((r) => h('tr', { class: r.from ? 'borrowed' : null },
      h('td', {}, h('b', { text: `${r.key} · ${STAGE_NAMES[r.key] || r.rec.name}` }),
        r.from ? h('div', { class: 'small', text: `from run ${r.from}, not counted in this run's total` }) : null,
        h('div', { class: 'faint small', text: r.tools.join(', ') || 'local checks' }),
        r.calls ? h('div', { class: 'faint small', text: r.calls }) : null,
        r.rec.code ? h('div', { class: 'faint small', text: `code ${r.rec.code.commit.slice(0, 7)}${r.rec.code.dirty ? ' with uncommitted changes' : ''}` }) : null,
        r.rec.notes ? (r.rec.notes.length > 140
          ? h('details', { class: 'gen-notes faint small' }, h('summary', { text: `${r.rec.notes.slice(0, 120).trim()}…` }), h('p', { text: r.rec.notes }))
          : h('div', { class: 'faint small', text: r.rec.notes })) : null),
      h('td', {}, h('div', { class: 'tcell' }, h('span', { class: 'track' }, h('span', { class: 'tbar', style: { width: `${max && r.rec.seconds ? ((r.rec.seconds / max) * 100).toFixed(1) : 0}%` } })),
        h('span', { class: 'num', text: secs(r.rec.seconds) }))),
      h('td', { class: 'r num', text: usd(r.rec.cost_usd) }))),
    totalRow('Total for this run', gen.total),
    gen.with_borrowed ? totalRow('Including the borrowed stage', gen.with_borrowed, borrowed ? `stage 2h of run ${borrowed.from}` : null) : null));
  const bases = [...new Set(rows.map((r) => r.rec.cost_basis).filter(Boolean))];
  fill('#genNote', h('span', { text: 'Costs are estimates, as each record states:' }), bases.map((b) => h('span', { text: `· ${b}` })));
}

// ---------------------------------------------------------------------------------------------------------------
// Compare

async function compareItem(id) {
  if (id.startsWith('v:')) {
    const v = [...S.char.versions, ...S.ws.unattached_versions].find((x) => `v:${x.id}` === id);
    if (!v || !v.high) return { label: id.slice(2), error: 'no high.glb in this version' };
    const sc = v.scores;
    return { label: v.label, url: versionFile(v.id, v.high), info: { parents: Object.fromEntries((v.sections && v.sections.sections || []).map((s) => [s.name, s.parent])),
      joints: v.sections && v.sections.joints, ends: v.sections && v.sections.ends },
    triangles: v.triangles, triangles_low: v.triangles_low,
    scores: sc ? { body: sc.body, views: sc.views, view_info: sc.view_info, file: `${sc.file} (matched by ${sc.matched_by})` } : null };
  }
  const [, rest] = id.split(/^b:/);
  const dir = rest.slice(rest.lastIndexOf(':') + 1);
  const runId = rest.slice(0, rest.lastIndexOf(':'));
  const d = await getDetail(runId);
  const body = d.bodies.find((b) => b.dir === dir);
  if (!body) return { label: runId, error: `no ${dir} in the run` };
  const sc = d.scores && !d.scores.error ? d.scores : null;
  const sb = sc ? sc.bodies.find((b) => !b.variant_of && b.glb === body.high) : null;
  return { label: `${runId}${dir === '04-body' ? '' : ` · ${dir.replace(/^04-body-/, '')}`}`, url: runFile(runId, body.high || body.low), info: sectionInfo(body),
    triangles: body.glb.high ? body.glb.high.triangles : null, triangles_low: body.glb.low ? body.glb.low.triangles : null,
    scores: sb ? { body: sb, views: sc.views, view_info: sc.view_info, file: `${runId}/05-scores.json` } : null };
}

function delta(a, b, higherBetter) {
  if (typeof a !== 'number' || typeof b !== 'number') return h('span', { class: 'faint', text: '–' });
  const d = b - a;
  if (Math.abs(d) < 1e-9) return h('span', { class: 'delta same', text: 'same' });
  const good = higherBetter ? d > 0 : d < 0;
  return h('span', { class: `delta ${good ? 'better' : 'worse'}`, text: `${good ? '▲' : '▼'} ${d > 0 ? '+' : '−'}${fmtAuto(+Math.abs(d).toFixed(4))} ${good ? 'better' : 'worse'}` });
}

async function renderCompare() {
  const body = $('#cmpBody');
  const token = ++cmpToken;
  if (S.cmpSel.length < 2) {
    if (cmp) { cmp.dispose(); cmp = null; }
    fill(body, h('div', { class: 'empty illo' }, icon('hg-compare-versions', 'illo-svg'),
      h('p', { text: 'Select two versions in the Versions list to compare them here with the same camera.' })));
    return;
  }
  let items;
  try {
    items = await Promise.all(S.cmpSel.map(compareItem));
  } catch (e) {
    if (token !== cmpToken) return;
    if (cmp) { cmp.dispose(); cmp = null; }
    fill(body, h('p', { class: 'empty', text: `Couldn't read a version to compare: ${e.message || e}` }));
    return;
  }
  if (token !== cmpToken) return;
  const [a, b] = items;
  if (!$('#cmp')) {
    if (cmp) { cmp.dispose(); cmp = null; }
    fill(body, h('div', { class: 'cmp-grid' }, h('div', { class: 'cmp-view' }, h('canvas', { id: 'cmp', 'aria-label': 'Two versions side by side, same camera' }),
      h('span', { class: 'cmp-mid' }), h('span', { class: 'cmp-lbl a', id: 'cmpA' }), h('span', { class: 'cmp-lbl b', id: 'cmpB' })),
    h('div', { class: 'tbl-wrap', id: 'cmpTbl' })));
    cmp = new CompareView($('#cmp'));
  }
  $('#cmpA').textContent = a.label; $('#cmpA').title = a.label;
  $('#cmpB').textContent = b.label; $('#cmpB').title = b.label;
  const views = [...new Set([...(a.scores ? a.scores.views : []), ...(b.scores ? b.scores.views : [])])];
  const rows = [];
  for (const v of views) {
    const x = a.scores && a.scores.body.views[v]; const y = b.scores && b.scores.body.views[v];
    if (!x && !y) continue;
    const held = (a.scores && (a.scores.view_info[v] || {}).heldout) || (b.scores && (b.scores.view_info[v] || {}).heldout);
    [['IoU', 'iou', true, 4], ['Colour error', 'colour_error', false, 2], ['SSIM', 'ssim', true, 4]].forEach(([lab, k, hib, dp], i) => {
      rows.push(h('tr', {}, i === 0 ? h('td', { rowspan: 3 }, h('b', { text: VIEW_NAMES[v] || heldLabel(v) }), held ? h('div', { class: 'faint small', text: 'held out: no body was built from it' }) : null) : null,
        h('td', { text: lab }), h('td', { class: 'r num', text: x ? fmt(x[k], dp) : 'not scored' }), h('td', { class: 'r num', text: y ? fmt(y[k], dp) : 'not scored' }),
        h('td', {}, x && y ? delta(x[k], y[k], hib) : null)));
    });
  }
  const tri = (n) => (n === null || n === undefined ? 'none' : fmt(n));
  rows.push(h('tr', {}, h('td', {}, h('b', { text: 'Triangles' })), h('td', { text: 'High' }), h('td', { class: 'r num', text: tri(a.triangles) }),
    h('td', { class: 'r num', text: tri(b.triangles) }), h('td', { class: 'faint', text: typeof a.triangles === 'number' && typeof b.triangles === 'number' ? `${b.triangles >= a.triangles ? '+' : '−'}${fmt(Math.abs(b.triangles - a.triangles))}` : '' })));
  rows.push(h('tr', {}, h('td', {}), h('td', { text: 'Low' }), h('td', { class: 'r num', text: tri(a.triangles_low) }), h('td', { class: 'r num', text: tri(b.triangles_low) }), h('td', {})));
  const sources = [a, b].map((it) => `${it.label}: ${it.scores ? it.scores.file : 'no score row'}`);
  fill('#cmpTbl', h('table', {}, h('thead', {}, h('tr', {}, h('th', { text: 'View' }), h('th', { text: 'Measure' }), h('th', { class: 'r', text: a.label }),
    h('th', { class: 'r', text: b.label }), h('th', { text: 'Change' }))), h('tbody', {}, rows)),
  h('p', { class: 'faint small', style: { marginTop: '8px' }, text: `Triangles are counted from each GLB. Scores from ${sources.join('; ')}. Higher IoU and SSIM are better; lower colour error is better.` }));
  const gl = await Promise.all([a, b].map((it) => (it.url ? loadGlb(it.url).catch(() => null) : null)));
  if (token !== cmpToken || !cmp) return;
  gl.forEach((g, i) => cmp.set(i, g ? instance(g, items[i].info) : null));
  cmp.preset(S.cmpView);
}

// ---------------------------------------------------------------------------------------------------------------
// About

async function renderAbout() {
  const pre = (id) => h('pre', { id, text: 'Loading…' });
  const roots = S.ws ? S.ws.roots.map((r) => h('li', {}, h('code', { text: r.key }), ` ${r.label}${r.exists ? '' : ' (not found)'}`)) : [];
  fill('#aboutBody',
    h('p', { text: 'HeroGen Lab reads the run folders below and shows what their files hold. It never writes to them. Parts HeroGen does not make yet are marked "Not built yet".' }),
    h('div', {}, h('b', { text: 'Runs roots' }), h('ul', {}, roots),
      S.ws && S.ws.versions_root ? h('p', {}, h('b', { text: 'Versions: ' }), S.ws.versions_root.label) : h('p', { class: 'faint', text: 'No versions folder given (--versions).' })),
    h('div', {}, h('b', { text: 'Keyboard' }), h('ul', {},
      h('li', {}, h('kbd', { text: '1' }), ' to ', h('kbd', { text: '4' }), ': front, 3/4, side and back cameras; ', h('kbd', { text: 'O' }), ': orbit'),
      h('li', {}, 'In the 3D view: arrow keys turn the camera, ', h('kbd', { text: '+' }), ' and ', h('kbd', { text: '−' }), ' zoom, ', h('kbd', { text: 'Esc' }), ' shows every section'))),
    h('p', { text: 'Icons: Lucide 1.64.0 (ISC License); some Lucide icons derive from Feather (MIT License). HeroGen camera, pose, shading, profile and file icons, the H mark and the illustrations are original. The wordmark is outlined from the Inter typeface (SIL Open Font License 1.1). Interface type: Bricolage Grotesque, Figtree and JetBrains Mono (SIL Open Font License 1.1), served by the lab itself; nothing is loaded from a font service.' }),
    h('details', {}, h('summary', { text: 'Lucide licence (LUCIDE-LICENSE.txt)' }), pre('licLucide')),
    h('details', {}, h('summary', { text: 'Inter licence (INTER-OFL.txt)' }), pre('licInter')),
    h('details', {}, h('summary', { text: 'Font licences (OFL-BricolageGrotesque.txt, OFL-Figtree.txt, OFL-JetBrainsMono.txt)' }),
      pre('licFontBricolage'), pre('licFontFigtree'), pre('licFontMono')),
    h('details', {}, h('summary', { text: 'Third-party notices (THIRD-PARTY-NOTICES.txt)' }), pre('licNotices')));
  for (const [id, f] of [['licLucide', 'licenses/LUCIDE-LICENSE.txt'], ['licInter', 'licenses/INTER-OFL.txt'],
    ['licNotices', 'licenses/THIRD-PARTY-NOTICES.txt'], ['licFontBricolage', 'fonts/OFL-BricolageGrotesque.txt'],
    ['licFontFigtree', 'fonts/OFL-Figtree.txt'], ['licFontMono', 'fonts/OFL-JetBrainsMono.txt']]) {
    fetch(`static/${f}`).then((r) => (r.ok ? r.text() : Promise.reject(r.status)))
      .then((t) => { $(`#${id}`).textContent = t; }, () => { $(`#${id}`).textContent = `Couldn't load ${f}.`; });
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Choosing a character, a profile and a run

function defaultProfile(c) {
  for (const p of PROFILES) {
    const ids = c.profiles[p === 'none' ? 'null' : p] || [];
    if (ids.some((id) => entry(id).has_body)) return p;
  }
  return PROFILES.find((p) => (c.profiles[p === 'none' ? 'null' : p] || []).length) || 'none';
}

function selectCharacter(key, runId = null) {
  S.char = S.ws.characters.find((c) => c.key === key) || S.ws.characters[0];
  const r = runId ? entry(runId) : null;
  S.profile = r ? profileKey(r.profile) : defaultProfile(S.char);
  S.cmpSel = [];
  selectProfile(S.profile, runId);
}

function selectProfile(p, runId = null) {
  S.profile = p;
  const ids = S.char.profiles[p === 'none' ? 'null' : p] || [];
  // The newest run that went the whole way (a body and its scores), else the newest with a body.
  const pick = runId && ids.includes(runId) ? runId
    : ids.find((id) => entry(id).has_body && entry(id).has_scores) || ids.find((id) => entry(id).has_body) || ids[0];
  if (pick) loadRun(pick);
}

// A run that could not be read: say so in the viewer and leave every panel empty rather than show stale data.
function showUnreadable(id, why) {
  S.run = null;
  viewer.clear(); inst = null; restInst = null;
  vpMessage(illo('hg-run-failed', h('p', { text: `Couldn't read run ${id}.` }), h('p', { class: 'small', text: why })));
  for (const sel of ['#pipe', '#qBody', '#hoBody', '#deliv', '#genTbl', '#genNote', '#meters', '#stats', '#versions']) fill(sel);
  renderHeader();
  $('#pipeRun').textContent = `run ${id}`;
  fill('#heroStatus', chip('bad', "Can't read", { big: true, title: why }));
  $('#banner').hidden = true;
}

async function loadRun(id) {
  const token = ++runToken;
  S.runId = id;
  history.replaceState(null, '', `#run=${encodeURIComponent(id)}`);
  renderProfiles();
  vpMessage(icon('loader-circle', 'ic spin'), `Loading run ${id}…`);
  let run;
  try {
    run = await getDetail(id);
  } catch (e) {
    if (token !== runToken) return;
    S.run = null;
    viewer.clear(); inst = null; restInst = null;
    showUnreadable(id, `The lab server didn't return it (${e.message || e}).`);
    return;
  }
  if (token !== runToken) return;
  if (run.error) { showUnreadable(id, `${run.path}: the ${run.error}.`); return; }
  S.run = run;
  S.iso = null; S.focus = null;
  if (!S.cmpSel.length) {
    const mine = run.bodies.some((b) => b.dir === '04-body') ? `b:${id}:04-body` : null;
    const firstFrozen = S.char.versions.find((v) => v.high && !v.error);
    const frozen = firstFrozen ? `v:${firstFrozen.id}` : null;
    if (mine && frozen) S.cmpSel = [frozen, mine];
  }
  try {
    renderHeader(); renderBanner(); renderProfiles(); renderMeters(); renderPipeline(); renderVersions();
    renderQuality(); renderHeldout(); renderDeliv(); renderGen(); renderImgSelect();
  } catch (e) {
    console.error(e);
    fill('#banner', icon('alert-octagon'), h('div', {}, h('b', { text: "This run can't be shown in full. " }), String(e.message || e)));
    $('#banner').hidden = false;
  }
  renderCompare();
  await showBody(true);
}

// ---------------------------------------------------------------------------------------------------------------
// Controls

function seg(sel, fn) {
  $(sel).addEventListener('click', (e) => { const b = e.target.closest('button'); if (b && !b.disabled) fn(b.dataset.v, b); });
}

function setTheme(v) {
  if (v === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = v;
  pressed('#themeSeg', v);
  viewer.theme();
  if (cmp) cmp.theme();
}

function decorate() {
  const BTN_ICONS = {
    themeSeg: { dark: 'moon', light: 'sun' }, profileSeg: { best: 'sparkles', 'roblox-ingame': 'hg-profile-ingame', 'roblox-avatar': 'hg-profile-avatar', none: 'circle-help' },
    modeSeg: { '3d': 'box', sketch: 'image', side: 'columns-2' },
    camSeg: { orbit: 'rotate-3d', front: 'hg-cam-front', q: 'hg-cam-34', side: 'hg-cam-side', back: 'hg-cam-back' },
    cmpCam: { front: 'hg-cam-front', q: 'hg-cam-34', side: 'hg-cam-side', back: 'hg-cam-back' },
    shadeSeg: { textured: 'hg-shade-textured', clay: 'hg-shade-clay', wire: 'hg-shade-wire', sections: 'hg-shade-sections' },
  };
  fill('#poseSeg', POSES.map((p) => h('button', { type: 'button', 'data-v': p.key, 'aria-pressed': String(p.key === 'rest'), title: p.label }, p.short || p.label)));
  BTN_ICONS.poseSeg = Object.fromEntries(POSES.map((p) => [p.key, p.icon]));
  for (const [id, m] of Object.entries(BTN_ICONS)) {
    for (const b of $$(`#${id} button`)) {
      const n = m[b.dataset.v]; const text = b.textContent.trim();
      // The accessible name starts with the visible label; a longer description, when there is one, follows it.
      const desc = ((n && n.startsWith('hg-') && id !== 'poseSeg' && copyName(n.slice(3))) || b.title || '').replace(/ \(\w\)$/, '');
      b.replaceChildren(n ? icon(n) : '', h('span', { class: 'bl', text }));
      b.setAttribute('aria-label', desc && desc.toLowerCase() !== text.toLowerCase() ? `${text}, ${desc[0].toLowerCase()}${desc.slice(1)}` : text);
      if (!b.title) b.title = desc || text;
    }
  }
  for (const [id, n] of Object.entries({ pipeH: 'layers', verH: 'history', qH: 'gauge', hoH: 'hg-heldout', dH: 'package', gH: 'timer', cH: 'columns-2', aH: 'info' })) {
    $(`#${id}`).prepend(icon(n));
  }
  const wm = icon('wordmark', 'wm-svg');
  $('#wordmark').replaceChildren(wm);
}

function bind() {
  seg('#themeSeg', (v) => { setTheme(v); store.set('hgl-theme', v); });
  seg('#profileSeg', (v) => selectProfile(v));
  seg('#modeSeg', setMode);
  seg('#shadeSeg', setShading);
  seg('#poseSeg', (v) => setPose(v));
  seg('#camSeg', setCamera);
  seg('#lodSeg', (v) => { if (v !== S.lod) { S.lod = v; pressed('#lodSeg', v); showBody(); } });
  seg('#cmpCam', (v) => { S.cmpView = v; pressed('#cmpCam', v); if (cmp) cmp.preset(v); });
  $('#runSel').addEventListener('change', (e) => loadRun(e.target.value));
  $('#charSel').addEventListener('change', (e) => selectCharacter(e.target.value));
  $('#imgSel').addEventListener('change', (e) => setImg(e.target.value));
  $('#secSel').addEventListener('change', (e) => isolate(e.target.value || null));
  viewer.onPick = (n) => isolate(n && n === S.iso ? null : n);
  matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => { viewer.theme(); if (cmp) cmp.theme(); });
  document.addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target;
    if (t instanceof Element && (t.closest('input, select, textarea') || t.isContentEditable)) return;
    if (e.key === 'Escape') { if (S.iso || S.focus) { isolate(null); e.preventDefault(); } return; }
    if (S.mode !== '3d') return;
    const cam = { 1: 'front', 2: 'q', 3: 'side', 4: 'back', o: 'orbit', O: 'orbit' }[e.key];
    if (cam) { setCamera(cam); e.preventDefault(); }
  });
}

async function boot() {
  const [copy, secTokens] = await Promise.all([getJson('static/ui-copy.json').catch(() => null),
    getJson('static/tokens/sections.json').catch(() => null)]);
  if (copy) COPY = copy;
  if (secTokens) SEC_INFO = Object.fromEntries(secTokens.sections.map((x) => [x.label, x]));
  const iconErrors = await loadIcons(ICONS, ILLUSTRATIONS, 'hg-wordmark-color-dark');
  if (iconErrors.length) console.warn('icons not loaded:', iconErrors.join('; '));
  viewer = new Viewer({ canvas: $('#gl'), gizmo: $('#gizmo'), sectionColours: Object.fromEntries(Object.entries(SEC_INFO).map(([k, v]) => [k, v.color])) });
  snaps = new Snapshots(360);
  decorate();
  bind();
  const saved = store.get('hgl-theme');
  setTheme(saved === 'dark' || saved === 'light' ? saved : 'auto');
  window.heroLab = { S, viewer: () => viewer, inst: () => inst, compare: () => cmp, showMe };
  try {
    S.ws = await getJson('/api/workspace');
  } catch (e) {
    vpMessage(illo('hg-offline', h('p', { text: COPY.states.offline || '' }), h('p', { class: 'small', text: String(e.message || e) })));
    $('#heroName').textContent = 'Lab server not reachable';
    return;
  }
  renderAbout();
  if (!S.ws.characters.length) {
    $('#heroName').textContent = 'No runs';
    vpMessage(illo('hg-no-heroes', h('p', { text: `No run folders in ${S.ws.roots.map((r) => r.label).join(', ')}.` })));
    renderCompare();
    return;
  }
  const want = new URLSearchParams(location.hash.slice(1)).get('run');
  const start = want && S.ws.runs[want] ? want : null;
  selectCharacter(start ? charOf(start).key : S.ws.characters[0].key, start);
  // A link to #run=<root>/<run> opens that run, also when only the address's fragment changes.
  window.addEventListener('hashchange', () => {
    const id = new URLSearchParams(location.hash.slice(1)).get('run');
    if (id && id !== S.runId && S.ws.runs[id]) selectCharacter(charOf(id).key, id);
  });
}

boot();
