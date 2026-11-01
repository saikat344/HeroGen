// Version actions and owner-scoped lineage share the selected workbench job.
const STAGE = {'1': 'cover', '2': 'turnaround', '3f': 'texture'};
export const versionLabel = (v, index = 1) => v.note || (!v.parent_job ? 'Original texture' : `Variation ${Math.max(1, index)}`);
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
export function expectedTime(job, stage) {
  const keys = stage === 'texture' ? ['3f','4','6','5'] : stage === 'turnaround' ? ['2','3','3f','4','6','5'] : ['1','2','3','3f','4','6','5'];
  const stages = keys.map(key => job.stages?.find(s => s.key === key)).filter(s => !['skipped', 'unavailable'].includes(s?.state));
  if (!stages.length) return 'Time estimate needs more real runs.';
  if (!stages.every(s => Number.isFinite(s?.typical_seconds))) return 'Time estimate needs more real runs.';
  const seconds = Math.max(1, Math.round(stages.reduce((sum, s) => sum + s.typical_seconds, 0)));
  return `Usually about ${seconds >= 90 ? `${Math.round(seconds / 60)} min` : `${seconds} s`}`;
}
function node(tag, value, cls) {
  const n = document.createElement(tag); if (value != null) n.textContent = value; if (cls) n.className = cls; return n;
}
export function variationsView($, {api, select, created, changeIdea, key, textureNotes}) {
  let job = null, versions = [], signature = '', request = 0, busy = false, family = '', panel = 'live';
  let compare = null, compareRequest = 0;
  const message = (id, value) => { $(id).textContent = value; $(id).hidden = !value; };
  const terminal = () => job && !['queued', 'running'].includes(job.state) && job.export_state !== 'running';
  const retryStage = () => STAGE[job?.from_stage] || 'cover';
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
  function actions() {
    const allowed = terminal() && !job.input_rejected;
    $('#retryVersion').disabled = busy || !allowed || !job.can_redo?.[retryStage()];
    $('#refineVersion').disabled = busy || !allowed || !job.can_redo?.turnaround;
    $('#attemptTime').textContent = job ? expectedTime(job, retryStage()) : '';
    const failed = job?.parent_job && (['failed','rejected','canceled'].includes(job.state) || job.stages?.some(s => s.key === job.from_stage && ['skipped','failed','rejected'].includes(s.state)));
    $('#variationFailure').hidden = !failed;
    $('#variationReason').textContent = failed ? job.error?.message || job.stages?.find(s => s.key === job.from_stage)?.message || 'This attempt did not finish.' : '';
    $('#retryFailedVersion').disabled = busy || !terminal() || !job.can_redo?.[retryStage()];
    $('#inputRefusal').hidden = !job?.input_rejected;
    $('#inputReason').textContent = job?.input_rejected ? job.error?.message || 'This image could not be used.' : '';
    $('#retryInput').disabled = busy || !terminal();
    $('#retryVersion').hidden = !!job?.input_rejected;
    $('#refineVersion').hidden = !!job?.input_rejected;
  }
  function renderVersions() {
    const options = versions.map((v, i) => {
      const opt = node('option', versionLabel(v, i)); opt.value = v.id; return opt;
    });
    $('#versionSelect').replaceChildren(...options); $('#versionSelect').value = job.id;
    const rows = versionRows(versions);
    const list = node('ul', null, 'version-list');
    for (const v of rows) {
      const li = node('li'); li.style.setProperty('--depth', v.depth);
      const b = node('button', null, 'version-node'); b.type = 'button'; b.dataset.version = v.id;
      b.setAttribute('aria-pressed', String(v.id === job.id));
      if (v.cover_thumbnail_url) { const img = node('img'); img.src = v.cover_thumbnail_url; img.alt = ''; img.loading = 'lazy'; b.append(img); }
      const copy = node('span'); copy.append(node('b', versionLabel(v, versions.findIndex(x => x.id === v.id))), node('small', `${v.state} · ${versionMetrics(v)}`));
      b.append(copy); b.addEventListener('click', () => select(v.id, true)); li.append(b); list.append(li);
    }
    $('#versionTree').replaceChildren(list);
    $('#compareVersions').disabled = versions.filter(v => v.model_url).length < 2;
  }
  async function loadVersions() {
    const id = job.id, token = ++request;
    message('#branchesStatus', 'Loading versions…');
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(id)}/versions`);
      if (token !== request || job?.id !== id) return;
      versions = result.versions; family = versions.map(v => v.id).join(','); renderVersions();
      message('#branchesStatus', '');
    } catch (e) { if (token === request) message('#branchesStatus', e.message); }
  }
  async function attempt(stage, note = '', same = false, input = false) {
    if (busy || !terminal()) return;
    const selected = job;
    busy = true; actions(); $('#submitRefine').disabled = true; message('#variationError', ''); message('#refineError', '');
    try {
      const next = await api(`/api/jobs/${encodeURIComponent(selected.id)}/${input ? 'retry-input' : 'redo'}`, {
        method: 'POST', headers: {'Idempotency-Key': key()},
        json: input ? {} : {stage, note, build_all: same ? selected.build_all : true},
      });
      $('#refineDialog').close(); created(next);
    } catch (e) { message($('#refineDialog').open ? '#refineError' : '#variationError', e.message); }
    finally { busy = false; actions(); $('#submitRefine').disabled = false; }
  }
  $('#versionSelect').addEventListener('change', event => select(event.target.value, true));
  $('#retryVersion').addEventListener('click', () => attempt(retryStage()));
  $('#retryFailedVersion').addEventListener('click', () => attempt(retryStage(), job.note || '', true));
  $('#retryInput').addEventListener('click', () => attempt(null, '', false, true));
  $('#changeIdea').addEventListener('click', () => changeIdea(job));
  function refineChoice() {
    const isTexture = $('#refineStage').value === 'texture';
    $('#refineNoteWrap').hidden = isTexture && !textureNotes();
    $('#submitRefine').textContent = isTexture && !textureNotes() ? 'Retry texture' : 'Create variation';
    $('#refineTime').textContent = expectedTime(job, $('#refineStage').value);
  }
  $('#refineVersion').addEventListener('click', () => {
    $('#refineStage').options[0].disabled = !job.can_texture;
    $('#refineStage').value = job.can_texture ? 'texture' : 'turnaround';
    $('#refineNote').value = ''; $('#refineCount').textContent = '0 / 300 characters';
    message('#refineError', ''); refineChoice(); $('#refineDialog').showModal();
  });
  $('#refineStage').addEventListener('change', refineChoice);
  $('#refineNote').addEventListener('input', () => {
    const n = Array.from($('#refineNote').value).length;
    $('#refineCount').textContent = `${n} / 300 characters`;
    $('#refineNote').setAttribute('aria-invalid', String(n > 300));
  });
  $('#refineForm').addEventListener('submit', event => {
    event.preventDefault();
    const stage = $('#refineStage').value, note = stage === 'texture' && !textureNotes() ? '' : $('#refineNote').value;
    if (Array.from(note).length > 300) { message('#refineError', 'Use at most 300 characters.'); return; }
    attempt(stage, note);
  });
  $('#closeRefine').addEventListener('click', () => $('#refineDialog').close());
  $('#viewTree').addEventListener('click', () => {
    const on = $('#versionTree').classList.toggle('show-tree');
    $('#viewTree').setAttribute('aria-pressed', String(on)); $('#viewTree').textContent = on ? 'View list' : 'View tree';
  });
  function recipe() {
    const r = job.recipe || {}, dl = node('dl', null, 'recipe-facts');
    for (const [label, value] of [['Profile', r.profile || job.profile], ['Continued from', STAGE[job.from_stage] || 'Original input'], ['Note used', r.note || 'No note']]) dl.append(node('dt', label), node('dd', value));
    $('#recipePanel').replaceChildren(dl);
    for (const stage of r.stages || []) {
      const p = stage.prompt_version;
      $('#recipePanel').append(node('h3', `Stage ${stage.stage}`), node('p', `Models: ${(stage.models || []).join(', ') || 'Not recorded'}`, 'small'), node('p', `Prompt version: ${p ? (typeof p === 'string' ? p : JSON.stringify(p)) : 'Not recorded'}`, 'small'));
      if (stage.profile) $('#recipePanel').append(node('p', `Profile: ${typeof stage.profile === 'string' ? stage.profile : JSON.stringify(stage.profile)}`, 'small'));
    }
  }
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
  $('#compareDialog').addEventListener('close', closeCompare);
  return {
    update(next) {
      const changed = job?.id !== next.id;
      if (changed) { ++request; $('#refineDialog').close(); if (family && !family.split(',').includes(next.id)) setPanel('live'); }
      job = next; actions(); recipe();
      const nextSignature = JSON.stringify([next.id, next.state, next.stages, next.cost_usd, next.export_state]);
      if (signature !== nextSignature) { signature = nextSignature; loadVersions(); }
      else if (changed) renderVersions();
    },
    reset() { ++request; signature = ''; job = null; versions = []; family = ''; $('#refineDialog').close(); $('#compareDialog').close(); closeCompare(); setPanel('live'); }
  };
}
