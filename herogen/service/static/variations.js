import { stageTiming } from './generation.js';
import { acceptPlaceholderOnTab } from './placeholder-tab.js';
// Version actions and owner-scoped lineage share the selected workbench job.
const STAGE = {'1': 'cover', '2': 'turnaround', '3f': 'texture'};
export const versionLabel = (v, index = 1) => v.note || (!v.parent_job ? 'Original texture' : `Variation ${Math.max(1, index)}`);
export const versionAction = (v, parent, known) => known || (v.note && v.note !== parent?.note ? 'Refine' : 'Retry / Refine');
// Tree side for a child version: a new note is a Refine (left), anything else re-runs a step (Retry, right).
export const versionSide = (v, parent, known) => versionAction(v, parent, known) === 'Refine' ? 'refine' : 'retry';
export function versionMetrics(v) {
  const seconds = Number.isFinite(v.seconds) ? (v.seconds < 90 ? `${v.seconds.toFixed(1)} s` : `${(v.seconds / 60).toFixed(1)} min`) : 'Time pending';
  return `${seconds} · ${Number.isFinite(v.cost_usd) ? `$${v.cost_usd.toFixed(2)}` : 'Cost pending'}`;
}
export function versionRows(versions) {
  const ids = new Set(versions.map(v => v.id)), rows = [], seen = new Set();
  function visit(v, depth) {
    if (seen.has(v.id)) return;
    seen.add(v.id); rows.push({...v, depth});
    versions.filter(child => child.parent_job === v.id).forEach(child => visit(child, depth + 1));
  }
  versions.filter(v => !ids.has(v.parent_job)).forEach(v => visit(v, 0));
  versions.forEach(v => visit(v, 0));
  return rows;
}
// The family as nested branches: each node keeps its Refine and Retry children apart, in creation order.
export function variationTree(versions, known = new Map()) {
  const seen = new Set();
  function build(v, side) {
    seen.add(v.id);
    const branch = {version: v, side, refine: [], retry: []};
    for (const child of versions) {
      if (child.parent_job !== v.id || seen.has(child.id)) continue;
      const childSide = versionSide(child, v, known.get(child.id));
      branch[childSide].push(build(child, childSide));
    }
    return branch;
  }
  const byId = new Map(versions.map(v => [v.id, v]));
  return versionRows(versions).filter(r => r.depth === 0 && !seen.has(r.id)).map(r => build(byId.get(r.id), null));
}
// The time to what a game engine needs: the 3D model (a GLB is enough for Unity and the rest), or for a texture-only
// redo the new texture. The Roblox body and rig run in the background after it, so they do not count.
export function expectedTime(job, stage) {
  const keys = stage === 'texture' ? ['3f'] : stage === 'turnaround' ? ['2','3'] : ['1','2','3'];
  const target = stage === 'texture' ? 'the new texture' : 'the 3D model';
  const stages = keys.map(key => job.stages?.find(s => s.key === key)).filter(s => !['skipped', 'unavailable'].includes(s?.state));
  if (!stages.length) return 'Time estimate needs more real runs.';
  if (!stages.every(s => Number.isFinite(s?.typical_seconds))) return 'Time estimate needs more real runs.';
  const seconds = Math.max(1, Math.round(stages.reduce((sum, s) => sum + s.typical_seconds, 0)));
  return `Usually about ${seconds >= 90 ? `${Math.round(seconds / 60)} min` : `${seconds} s`} to ${target}`;
}
function node(tag, value, cls) {
  const n = document.createElement(tag); if (value != null) n.textContent = value; if (cls) n.className = cls; return n;
}
function icon(name, cls = 'icon icon--s') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'), use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  svg.setAttribute('class', cls); svg.setAttribute('aria-hidden', 'true'); use.setAttribute('href', `/assets/icons/sprite.svg#i-${name}`);
  svg.append(use); return svg;
}
// Node status: [label, icon, tone]. Tones map to .node--{tone} and .badge--{tone}.
const STATUS = {
  done: ['Completed', 'circle-check', 'ok'], run: ['Running', 'loader-circle', 'run'], wait: ['Waiting', 'circle-dashed', 'wait'],
  fail: ['Failed', 'circle-x', 'fail'], attn: ['Needs attention', 'triangle-alert', 'fail'], skip: ['Skipped', 'circle-dashed', 'skip'],
  stop: ['Stopped', 'ban', 'wait'],
};
const VERSION_STATUS = {succeeded: 'done', running: 'run', queued: 'wait', failed: 'fail', rejected: 'attn', canceled: 'stop', stopped: 'stop'};
const STEP_STATUS = {passed: 'done', running: 'run', pending: 'wait', failed: 'fail', rejected: 'attn', skipped: 'skip', unavailable: 'skip', canceled: 'stop'};
const FAILED = ['failed', 'rejected', 'canceled'];
const SCOPE = {
  texture: 'Keeps the geometry. Only the texture is repainted.',
  turnaround: 'Redraws the character views, then rebuilds the 3D shape and texture.',
};

// The Refine box's example per step: HeroGen's suggestion for this hero when it has one (a fix the step can make,
// from comparing the artwork with the finished model), otherwise a fixed example of what the step honours.
export const REFINE_EXAMPLES = {texture: 'Make the jacket deep red leather with brass buttons.',
  turnaround: 'Add a hooded cloak and make the boots dark brown.'};
export const refineExample = (job, stage) => job?.refine_suggestions?.[stage] || REFINE_EXAMPLES[stage] || REFINE_EXAMPLES.texture;
export function variationsView($, {api, select, created, changeIdea, key, textureNotes}) {
  let job = null, versions = [], signature = '', request = 0, busy = false, family = '', panel = 'live';
  let compare = null, compareRequest = 0, rootJob = null, nodes = [], focusKey = null, failedChild = null;
  const actionLabels = new Map(), lastTone = new Map();
  const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  $('#inlineRefine').append($('#refineForm'));
  const restoreRefine = () => { $('#inlineRefine').append($('#refineForm')); };
  $('#refineDialog').addEventListener('close', () => { if (!$('#refineDialog').open) restoreRefine(); });
  const message = (id, value) => { $(id).textContent = value; $(id).hidden = !value; };
  const terminal = () => job && !['queued', 'running'].includes(job.state) && job.export_state !== 'running';
  const retryStage = () => STAGE[job?.from_stage] || 'cover';
  const labelOf = v => versionLabel(v, versions.findIndex(x => x.id === v.id));
  function setPanel(value) {
    panel = value;
    for (const name of ['live', 'recipe', 'branches']) {
      $(`#${name}Panel`).hidden = value !== name;
      $(`#${name}Tab`).setAttribute('aria-selected', String(value === name));
      $(`#${name}Tab`).tabIndex = value === name ? 0 : -1;
    }
  }
  for (const name of ['live', 'recipe', 'branches']) {
    $(`#${name}Tab`).addEventListener('click', () => setPanel(name));
    $(`#${name}Tab`).addEventListener('keydown', event => {
      const names = ['live', 'recipe', 'branches'];
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const i = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (names.indexOf(name) + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
      setPanel(names[i]); $(`#${names[i]}Tab`).focus();
    });
  }
  // The newest attempt made from the selected version, when it did not finish (its parent stays usable).
  function latestFailedChild() {
    if (!job) return null;
    const children = versions.filter(v => v.parent_job === job.id);
    const latest = children[children.length - 1];
    return latest && FAILED.includes(latest.state) ? latest : null;
  }
  function actions() {
    const allowed = terminal() && !job.input_rejected;
    $('#retryVersion').disabled = busy || !allowed || !job.can_redo?.[retryStage()];
    $('#refineVersion').disabled = busy || !allowed || !job.can_redo?.turnaround;
    $('#submitRefine').disabled = busy || !allowed || !job.can_redo?.turnaround;
    $('#inlineRefine').hidden = !!job.input_rejected || !terminal();
    if (!$('#refineDialog').open) $('#refineHeading').textContent = `Refine ${labelOf(job)}`;
    $('#attemptTime').textContent = job ? expectedTime(job, retryStage()) : '';
    const waiting = !!job && !terminal();
    $('#retryHelp').textContent = waiting ? 'Available when this run finishes.' : 'Run this step again.';
    $('#refineHelp').textContent = waiting ? 'Available when this run finishes.' : 'Change the result with your instructions.';
    const failed = job?.parent_job && (FAILED.includes(job.state) || job.stages?.some(s => s.key === job.from_stage && ['skipped','failed','rejected'].includes(s.state)));
    failedChild = failed ? null : latestFailedChild();
    const subject = failed ? job : failedChild;
    $('#variationFailure').hidden = !subject;
    if (subject) {
      const parent = versions.find(v => v.id === subject.parent_job);
      const action = versionSide(subject, parent, actionLabels.get(subject.id)) === 'refine' ? 'Refine' : 'Retry';
      $('#variationTitle').textContent = `${action} attempt ${subject.state === 'canceled' ? 'stopped' : 'failed'}`;
    }
    $('#variationReason').textContent = failed ? job.error?.message || job.stages?.find(s => s.key === job.from_stage)?.message || 'This attempt did not finish.' : '';
    $('#variationReason').hidden = !failed;
    $('#retryFailedVersion').disabled = busy || !terminal() || (failed ? !job.can_redo?.[retryStage()] : !failedChild || !job.can_redo?.[STAGE[failedChild.from_stage] || 'cover']);
    $('#inputRefusal').hidden = !job?.input_rejected;
    $('#inputReason').textContent = job?.input_rejected ? job.error?.message || 'This image could not be used.' : '';
    $('#retryInput').disabled = busy || !terminal();
    $('#retryVersion').hidden = !!job?.input_rejected;
    $('#refineVersion').hidden = !!job?.input_rejected;
    $('#attemptActions').hidden = !!job?.input_rejected;
  }

  // ---------- Variations tree ----------
  // Load the version behind a node. Keyboard users keep focus on the node once it reloads; on phones a tap
  // scrolls back to the stage instead.
  function choose(rec, viaPointer) {
    const phoneTap = viaPointer && matchMedia('(max-width: 767px)').matches;
    focusKey = phoneTap ? null : rec.key;
    if ($('#treeDialog').open) $('#treeDialog').close();
    if (rec.version !== job?.id) select(rec.version, true);
    if (phoneTap) {
      document.querySelector('.result-column')?.scrollIntoView({behavior: reduceMotion() ? 'auto' : 'smooth', block: 'start'});
    }
  }
  function treeNode(rec, {num, title, status, image, clay, placeholder, meta}) {
    const [label, iconName, tone] = STATUS[status];
    const prev = lastTone.get(rec.key);
    lastTone.set(rec.key, tone);
    const el = node('div', null, `node vt-node node--${tone}`);
    if (prev && prev !== 'ok' && tone === 'ok') { el.classList.add('is-completing'); rec.completing = true; }
    el.setAttribute('role', 'treeitem'); el.tabIndex = -1; el.dataset.key = rec.key; el.dataset.version = rec.version;
    if (rec.version === job?.id && !rec.step) { el.setAttribute('aria-current', 'true'); el.setAttribute('aria-selected', 'true'); }
    else el.setAttribute('aria-selected', 'false');
    el.title = meta ? `${title} · ${meta}` : title;
    const head = node('span', null, 'node__head');
    const status_icon = icon(iconName, `icon status-icon badge--${tone}${status === 'run' ? ' is-spinning' : ''}`);
    head.append(node('span', String(num).padStart(2, '0'), 'num'), node('span', title, 'title'), status_icon);
    const thumb = node('span', null, `node__thumb${clay ? ' node__thumb--clay' : ''}`);
    if (image) { const img = node('img'); img.src = image; img.alt = ''; img.loading = 'lazy'; img.decoding = 'async'; thumb.append(img); }
    else thumb.append(icon(placeholder || 'image', 'icon icon--l node__placeholder'));
    el.append(head, thumb, node('span', label, `badge badge--${tone} node__status`));
    el.addEventListener('click', () => choose(rec, true));
    rec.el = el; nodes.push(rec);
    return el;
  }
  function connector(drawing) { const c = node('span', null, `connector${drawing ? ' is-drawing' : ''}`); c.setAttribute('aria-hidden', 'true'); return c; }
  function railLabel(side) { const l = node('span', side === 'refine' ? 'Refine' : 'Retry', 'vt-label vt-rail'); l.setAttribute('aria-hidden', 'true'); return l; }
  function versionRec(branch, level, parent, counter) {
    const v = branch.version;
    const rec = {key: v.id, version: v.id, side: branch.side, parent, level, refine: [], retry: []};
    treeNode(rec, {num: ++counter.n, title: labelOf(v), status: VERSION_STATUS[v.state] || 'wait', image: v.cover_thumbnail_url,
      placeholder: 'sparkles', meta: versionMetrics(v)});
    rec.el.setAttribute('aria-level', String(level));
    return rec;
  }
  // A version below the root fork: its own children hang from a rail underneath it.
  function nested(branch, level, parent, counter) {
    const rec = versionRec(branch, level, parent, counter), wrap = node('div', null, 'vt-sub');
    wrap.append(rec.el);
    if (branch.refine.length || branch.retry.length) {
      const nest = node('div', null, 'vt-nest'); nest.setAttribute('role', 'group');
      nest.id = `vt-group-${rec.key}`; rec.el.setAttribute('aria-owns', nest.id); rec.el.setAttribute('aria-expanded', 'true');
      for (const side of ['refine', 'retry']) {
        if (!branch[side].length) continue;
        nest.append(railLabel(side));
        for (const child of branch[side]) {
          const item = node('div', null, 'vt-item vt-rail'), sub = nested(child, level + 1, rec, counter);
          item.append(sub.el); nest.append(item); rec[side].push(sub.rec);
        }
      }
      nest.lastElementChild.classList.add('vt-end');
      wrap.append(nest);
    }
    return {el: wrap, rec};
  }
  // The original: Refine children branch to the left, Retry children to the right (stacked in a narrow panel).
  function fork(branch, level, counter) {
    const rec = versionRec(branch, level, null, counter), el = node('div', null, 'vt-fork'), center = node('div', null, 'vt-center');
    center.append(rec.el); el.append(center);
    if (branch.refine.length || branch.retry.length) {
      const sides = node('div', null, 'vt-branches');
      rec.el.setAttribute('aria-expanded', 'true');
      const owns = [];
      for (const side of ['refine', 'retry']) {
        const column = node('div', null, `vt-side vt-side--${side}`);
        column.setAttribute('role', 'group'); column.setAttribute('aria-label', side === 'refine' ? 'Refine branches' : 'Retry branches');
        column.id = `vt-${side}-${rec.key}`; owns.push(column.id);
        if (!branch[side].length) column.classList.add('is-empty');
        else column.append(railLabel(side));
        for (const child of branch[side]) {
          const item = node('div', null, 'vt-item vt-rail'), sub = nested(child, level + 1, rec, counter);
          item.append(sub.el); column.append(item); rec[side].push(sub.rec);
        }
        sides.append(column);
      }
      rec.el.setAttribute('aria-owns', owns.join(' '));
      const last = [...sides.querySelectorAll(':scope > .vt-side:not(.is-empty)')].pop();
      last?.lastElementChild.classList.add('vt-end');
      el.append(sides);
    }
    return {el, rec};
  }
  function renderTree() {
    const tree = $('#versionTree');
    const active = tree.contains(document.activeElement) ? document.activeElement.dataset.key : null;
    nodes = [];
    const counter = {n: 0}, parts = [];
    const [first, ...others] = variationTree(versions, actionLabels);
    if (first) {
      const spine = node('div', null, 'vt-spine'), steps = [];
      const root = rootJob?.id === first.version.id ? rootJob : null;
      for (const [stageKey, title, imageKey, clay, placeholder] of [['1', 'Artwork', 'cover', false, 'image'], ['3', '3D shape', 'front', true, 'box']]) {
        const stage = root?.stages?.find(s => s.key === stageKey);
        if (!stage) continue;
        const rec = {key: `step-${stageKey}`, version: first.version.id, step: true, level: 1};
        treeNode(rec, {num: ++counter.n, title, status: STEP_STATUS[stage.state] || 'wait', clay, placeholder,
          image: stage.state === 'passed' ? root.images?.[imageKey] || (clay ? null : root.images?.cover) : null,
          meta: stageTiming(stage) || undefined});
        rec.el.setAttribute('aria-level', '1');
        steps.push(rec);
        spine.append(rec.el, connector(rec.completing));
      }
      const top = fork(first, 1, counter);
      steps.forEach((rec, i) => { rec.down = steps[i + 1] || top.rec; (steps[i + 1] || top.rec).up = rec; });
      const all = [...steps.map(s => s.el), top.rec.el];
      all.forEach((el, i) => { el.setAttribute('aria-posinset', String(i + 1)); el.setAttribute('aria-setsize', String(all.length)); });
      parts.push(spine, top.el);
    }
    for (const branch of others) {
      const extra = node('div', null, 'vt-extra');
      extra.append(node('p', 'Continued from a deleted version', 'vt-extra__label'), fork(branch, 1, counter).el);
      parts.push(extra);
    }
    tree.replaceChildren(...parts);
    for (const rec of nodes) for (const side of ['refine', 'retry']) {
      rec[side]?.forEach((child, i, list) => { child.prev = list[i - 1]; child.next = list[i + 1]; });
      const kids = rec[side] || [];
      kids.forEach(child => { child.el.setAttribute('aria-setsize', String(kids.length)); child.el.setAttribute('aria-posinset', String(kids.indexOf(child) + 1)); });
    }
    const target = nodes.find(r => r.key === (active || focusKey)) || nodes.find(r => r.el.getAttribute('aria-current') === 'true') || nodes[0];
    if (target) target.el.tabIndex = 0;
    const idle = [document.body, $('#detail'), null].includes(document.activeElement);
    if (active) target?.el.focus({preventScroll: true});
    else if (focusKey && target?.key === focusKey && idle) target.el.focus();
    if (document.activeElement === target?.el) focusKey = null;
    $('#treeHint').hidden = versions.length !== 1;
  }
  function move(rec, key) {
    const firstChild = r => r.refine?.[0] || r.retry?.[0];
    const after = r => { for (let p = r; p; p = p.parent) if (p.next) return p.next; return null; };
    switch (key) {
      case 'ArrowUp': return rec.prev || rec.up || rec.parent;
      case 'ArrowDown': return rec.down || firstChild(rec) || after(rec);
      case 'ArrowLeft': return rec.side === 'retry' ? rec.parent : rec.refine?.[0];
      case 'ArrowRight': return rec.side === 'refine' ? rec.parent : rec.retry?.[0];
      case 'Home': return nodes[0];
      case 'End': return nodes[nodes.length - 1];
      default: return null;
    }
  }
  $('#versionTree').addEventListener('keydown', event => {
    const rec = nodes.find(r => r.el === event.target);
    if (!rec) return;
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(rec, false); return; }
    const next = move(rec, event.key);
    if (!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    if (!next) return;
    for (const r of nodes) r.el.tabIndex = r === next ? 0 : -1;
    next.el.focus(); next.el.scrollIntoView({block: 'nearest', inline: 'nearest'});
  });
  $('#expandTree').addEventListener('click', () => {
    $('#treeDialogBody').append($('#versionTree')); $('#treeDialog').showModal();
    $('#expandTree').setAttribute('aria-expanded', 'true');
    $('#versionTree').querySelector('[tabindex="0"]')?.focus();
  });
  $('#closeTree').addEventListener('click', () => $('#treeDialog').close());
  $('#treeDialog').addEventListener('close', () => {
    $('#treeHome').append($('#versionTree')); $('#expandTree').setAttribute('aria-expanded', 'false');
  });

  function renderVersions() {
    const options = versions.map((v, i) => {
      const opt = node('option', versionLabel(v, i)); opt.value = v.id; return opt;
    });
    $('#versionSelect').replaceChildren(...options); $('#versionSelect').value = job.id;
    renderTree();
    $('#compareVersions').disabled = versions.filter(v => v.model_url).length < 2;
  }
  async function loadVersions() {
    const id = job.id, token = ++request;
    message('#branchesStatus', versions.length ? '' : 'Loading versions…');
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(id)}/versions`);
      if (token !== request || job?.id !== id) return;
      const root = versionRows(result.versions)[0];
      const original = root?.id === job.id ? job : root ? await api(`/api/jobs/${encodeURIComponent(root.id)}`) : null;
      if (token !== request || job?.id !== id) return;
      versions = result.versions; family = versions.map(v => v.id).join(','); rootJob = original;
      renderVersions(); actions();
      message('#branchesStatus', '');
    } catch (e) { if (token === request) message('#branchesStatus', e.message); }
  }
  async function attempt(stage, note = '', same = false, input = false, action = 'Retry') {
    if (busy || !terminal()) return;
    const selected = job;
    busy = true; actions(); $('#submitRefine').disabled = true; message('#variationError', ''); message('#refineError', '');
    try {
      const next = await api(`/api/jobs/${encodeURIComponent(selected.id)}/${input ? 'retry-input' : 'redo'}`, {
        method: 'POST', headers: {'Idempotency-Key': key()},
        json: input ? {} : {stage, note, build_all: same ? selected.build_all : true},
      });
      actionLabels.set(next.id, action); $('#refineDialog').close(); created(next);
    } catch (e) { message($('#refineDialog').open ? '#refineError' : '#variationError', e.message); }
    finally { busy = false; actions(); }
  }
  $('#versionSelect').addEventListener('change', event => select(event.target.value, true));
  $('#retryVersion').addEventListener('click', () => attempt(retryStage()));
  $('#retryFailedVersion').addEventListener('click', () => {
    if (failedChild) {
      const side = versionSide(failedChild, job, actionLabels.get(failedChild.id));
      attempt(STAGE[failedChild.from_stage] || 'cover', failedChild.note || '', false, false, side === 'refine' ? 'Refine' : 'Retry');
    } else attempt(retryStage(), job.note || '', true);
  });
  $('#retryInput').addEventListener('click', () => attempt(null, '', false, true));
  $('#changeIdea').addEventListener('click', () => changeIdea(job));

  // ---------- Refine box ----------
  function refineChoice() {
    const stage = $('#refineStage').value, notesOff = stage === 'texture' && !textureNotes();
    $('#refineNoteWrap').hidden = notesOff;
    $('#refineNotesOff').hidden = !notesOff;
    $('#submitLabel').textContent = notesOff ? 'Retry texture' : 'Create variation';
    $('#refineScope').textContent = SCOPE[stage] || '';
    $('#refineNote').placeholder = refineExample(job, stage);
    $('#refineTime').textContent = job ? expectedTime(job, stage) : '';
    for (const chip of document.querySelectorAll('[data-refine-stage]')) {
      chip.setAttribute('aria-pressed', String(chip.dataset.refineStage === stage));
      chip.disabled = chip.dataset.refineStage === 'texture' && !job?.can_texture;
    }
  }
  $('#refineVersion').addEventListener('click', () => {
    $('#refineStage').options[0].disabled = !job.can_texture;
    $('#refineStage').value = job.can_texture ? 'texture' : 'turnaround';
    $('#refineNote').value = ''; $('#refineCount').textContent = '0 / 300 characters';
    message('#refineError', ''); refineChoice(); $('#refineDialog').append($('#refineForm')); $('#refineDialog').showModal();
  });
  $('#refineStage').addEventListener('change', refineChoice);
  for (const chip of document.querySelectorAll('[data-refine-stage]')) chip.addEventListener('click', () => {
    $('#refineStage').value = chip.dataset.refineStage; refineChoice();
  });
  acceptPlaceholderOnTab($('#refineNote'));
  $('#refineNote').addEventListener('input', () => {
    const n = Array.from($('#refineNote').value).length;
    $('#refineCount').textContent = `${n} / 300 characters`;
    $('#refineNote').setAttribute('aria-invalid', String(n > 300));
  });
  $('#refineForm').addEventListener('submit', event => {
    event.preventDefault();
    const stage = $('#refineStage').value, note = stage === 'texture' && !textureNotes() ? '' : $('#refineNote').value;
    if (Array.from(note).length > 300) { message('#refineError', 'Use at most 300 characters.'); return; }
    attempt(stage, note, false, false, stage === 'texture' && !textureNotes() ? 'Retry' : 'Refine');
  });
  $('#closeRefine').addEventListener('click', () => $('#refineDialog').close());

  function recipe() {
    const r = job.recipe || {}, dl = node('dl', null, 'recipe-facts');
    for (const [label, value] of [['Profile', r.profile || job.profile], ['Continued from', STAGE[job.from_stage] || 'Original input'], ['Note used', r.note || 'No note']]) dl.append(node('dt', label), node('dd', value));
    $('#recipePanel').replaceChildren(dl);
    for (const stage of r.stages || []) {
      const p = stage.prompt_version, card = node('section', null, 'recipe-stage');
      card.append(node('h3', `Stage ${stage.stage}`), node('p', `Models: ${(stage.models || []).join(', ') || 'Not recorded'}`, 'small'), node('p', `Prompt version: ${p ? (typeof p === 'string' ? p : JSON.stringify(p)) : 'Not recorded'}`, 'small'));
      if (stage.profile) card.append(node('p', `Profile: ${typeof stage.profile === 'string' ? stage.profile : JSON.stringify(stage.profile)}`, 'small'));
      $('#recipePanel').append(card);
    }
  }

  // ---------- Compare ----------
  async function compareModels() {
    const token = ++compareRequest;
    message('#compareError', '');
    message('#compareStatus', 'Loading models…');
    const selected = ['A','B'].map(side => versions.find(v => v.id === $(`#compare${side}`).value));
    const valid = selected[0] && selected[1] && selected[0].id !== selected[1].id;
    for (const side of ['A','B']) $(`#compareCanvas${side}`).hidden = !valid;
    if (!valid) { message('#compareStatus', ''); message('#compareError', 'Choose two different versions.'); return; }
    for (const [i, side] of ['A','B'].entries()) $(`#compareLabel${side}`).textContent = `${versionLabel(selected[i], versions.indexOf(selected[i]))} · ${versionMetrics(selected[i])}${selected[i].note ? '' : ' · No note'}`;
    try {
      const {createPreview} = await import('./preview.js');
      if (token !== compareRequest || !$('#compareDialog').open) return;
      if (!compare) {
        compare = [createPreview({damping: false}), createPreview({damping: false})];
        compare[0].onView(value => compare[1].setView(value)); compare[1].onView(value => compare[0].setView(value));
      }
      await Promise.all(compare.map((viewer, i) => viewer.show($(`#compareCanvas${['A','B'][i]}`), selected[i].model_url)));
      if (token === compareRequest && $('#compareDialog').open) { compare[1].setView(compare[0].view()); message('#compareStatus', 'Cameras linked'); }
    } catch { if (token === compareRequest) { message('#compareStatus', ''); message('#compareError', 'The comparison model could not be loaded. Choose another version or try again.'); } }
  }
  function closeCompare() { ++compareRequest; compare?.forEach(viewer => viewer.dispose()); compare = null; }
  $('#compareVersions').addEventListener('click', () => {
    const available = versions.filter(v => v.model_url);
    for (const side of ['A','B']) $('#compare' + side).replaceChildren(...available.map(v => { const o = node('option', versionLabel(v, versions.indexOf(v))); o.value = v.id; return o; }));
    $('#compareA').value = available.some(v => v.id === job.id) ? job.id : available[0].id;
    $('#compareB').value = available.find(v => v.id !== $('#compareA').value).id;
    $('#compareDialog').showModal(); compareModels();
  });
  for (const side of ['A','B']) $('#compare' + side).addEventListener('change', compareModels);
  $('#closeCompare').addEventListener('click', () => $('#compareDialog').close());
  $('#compareDialog').addEventListener('close', () => { if (!$('#compareDialog').open) closeCompare(); });
  return {
    update(next) {
      const changed = job?.id !== next.id;
      if (changed) { ++request; $('#refineDialog').close(); if (family && !family.split(',').includes(next.id)) { setPanel('live'); lastTone.clear(); } }
      job = next;
      if (changed) {
        $('#refineStage').options[0].disabled = !job.can_texture;
        $('#refineStage').value = job.can_texture ? 'texture' : 'turnaround';
        $('#refineNote').value = ''; $('#refineCount').textContent = '0 / 300 characters';
        message('#refineError', ''); refineChoice();
      }
      actions(); recipe();
      const nextSignature = JSON.stringify([next.id, next.state, next.stages, next.cost_usd, next.export_state]);
      if (signature !== nextSignature) { signature = nextSignature; loadVersions(); }
      else if (changed) renderVersions();
    },
    reset() { ++request; signature = ''; job = null; versions = []; family = ''; nodes = []; lastTone.clear(); $('#refineDialog').close(); $('#compareDialog').close(); $('#treeDialog').close(); closeCompare(); setPanel('live'); }
  };
}
