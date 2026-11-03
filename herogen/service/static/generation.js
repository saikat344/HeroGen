// Presentation follows reported model and export stage states.
import { kindName } from './heroes.js';
import { stageSource } from './workbench.js';
import { track, trackOnce } from './analytics.js';
import { thumbButton, viewerDock } from './viewer-dock.js';
const COPY = {
  p: ['Your idea', 'Sketching your idea', 'Turn your description into a character sketch.'],
  0: ['Prepare input', 'Preparing your image', 'Check the input before creating the character artwork.'],
  1: ['Artwork', 'Painting your hero', 'Create the coloured character artwork that guides the next steps.'],
  2: ['Character views', 'Seeing every side', 'Create front, back and side views for reconstruction.'],
  3: ['3D shape', 'Building the 3D shape', 'Reconstruct the character in three dimensions.'],
  '3f': ['Texture refine', 'Refining the texture', "Repaint the texture on your hero's 3D model."],
  4: ['Game body', 'Preparing the game body', 'Build the body parts and textures for the selected target.'],
  6: ['Roblox rig', 'Preparing the Roblox rig', 'Prepare the Roblox rig when this export is available.'],
  5: ['Results', 'Collecting your results', 'Prepare the results and available downloads.'],
};
export const TERMINAL = new Set(['succeeded', 'failed', 'rejected', 'canceled', 'stopped']);
const INTERRUPTED = new Set(['failed', 'rejected', 'canceled']);
const STATE = { passed: 'Passed', running: 'Running', pending: 'Waiting', failed: 'Failed', rejected: 'Needs attention',
  canceled: 'Cancelled', skipped: 'Skipped', unavailable: 'Unavailable' };
const ISSUE_STATES = ['failed', 'rejected', 'unavailable', 'skipped'];
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
// "Roblox ready" is shown only for a finished job whose Roblox files are ready (the service reports `roblox_state`
// as ready only when the game body and rig steps passed; `export_state` is the older name for the same report).
export const robloxStatus = (job) => job?.roblox_state || job?.export_state;
export const robloxReady = (job) => job?.state === 'succeeded' && robloxStatus(job) === 'ready';
// What a queued hero waits behind. `queued_ahead` counts the waiting heroes in front of it; 0 means a worker is
// starting it; a service that does not report a place gets the plain sentence.
export function queueNote(job) {
  if (job?.state !== 'queued') return '';
  let ahead = job.queued_ahead;
  if (!Number.isInteger(ahead) && Number.isInteger(job.queue_position)) ahead = Math.max(0, job.queue_position - 1);
  if (!Number.isInteger(ahead) || ahead < 0) return 'Starts when a worker is free';
  return ahead === 0 ? 'Starting…' : `${ahead} ${ahead === 1 ? 'hero' : 'heroes'} ahead of you`;
}
function timeLeft(steps) {
  let seconds = 0;
  for (const s of steps) {
    if (!['pending', 'running'].includes(s.state)) continue;
    if (!Number.isFinite(s.typical_seconds)) return null;
    seconds += s.state === 'running' && Number.isFinite(s.elapsed_seconds) ? Math.max(0, s.typical_seconds - s.elapsed_seconds) : s.typical_seconds;
  }
  return seconds <= 0 ? null : seconds < 60 ? 'under a minute left' : `about ${Math.ceil(seconds / 60)} min left`;
}
export function generationState(job) {
  const stages = (Array.isArray(job.stages) ? job.stages : []).filter((s) => s && typeof s === 'object');
  const main = stages.filter((s) => !s.side);
  const active = main.find((s) => s.state === 'running');
  const focus = active || main.find((s) => s.key === job.stage) || main.find((s) => s.state === 'pending');
  const issues = main.filter((s) => ISSUE_STATES.includes(s.state));
  const title = { queued: 'Waiting for a free worker', running: 'Your hero is taking shape.',
    succeeded: issues.length ? 'Meet your hero.' : 'Your hero. Your next idea.', stopped: 'Your version is saved.', failed: 'Your hero needs attention.', rejected: 'Let’s revisit your input.', canceled: 'Generation cancelled.' }[job.state] || 'Your hero’s progress';
  const subtitle = job.state === 'succeeded' ? (issues.length ? 'Your model is ready. See the notes below.' : 'Explore the result and the available downloads.') :
    job.state === 'stopped' ? 'Inspect this version, then start a new attempt with Retry or Refine.' :
    job.state === 'queued' ? 'Your job is saved. It will begin when a worker is available.' :
    job.state === 'canceled' ? 'The job has stopped. Any saved images remain available to inspect.' :
    INTERRUPTED.has(job.state) ? 'Review the reported issue below. Saved outputs remain available.' :
    (COPY[focus?.key]?.[1] || focus?.label || 'Following the pipeline');
  const steps = main.map((s) => ({ ...s, title: COPY[s.key]?.[0] || s.label || s.key,
    description: COPY[s.key]?.[2] || 'Follow the status reported by the service.',
    status: s.state === 'pending' && TERMINAL.has(job.state) ? (job.state === 'succeeded' && ['4', '6', '5'].includes(s.key) && job.export_state === 'running' ? 'Waiting' : 'Not reached') : STATE[s.state] || s.state || 'Not reported' }));
  const passed = main.filter((s) => s.state === 'passed').length;
  const terminal = TERMINAL.has(job.state);
  const exporting = job.state === 'succeeded' && job.export_state === 'running';
  const current = steps.find((s) => s.state === 'running') || steps.find((s) => s.state === 'pending');
  const phase = COPY[focus?.key]?.[1] || focus?.label || 'Following the pipeline';
  const left = job.state === 'running' ? timeLeft(steps) : null;
  const eyebrow = { queued: 'Queued · your hero is saved', running: left ? `${phase} · ${left}` : phase,
    succeeded: issues.length ? 'Hero ready · see the notes' : exporting ? 'Hero ready · preparing Roblox files' : robloxReady(job) ? 'Hero ready · Roblox R15' : 'Hero ready',
    stopped: 'Saved at this step', failed: 'Needs attention', rejected: 'Input needs attention', canceled: 'Cancelled' }[job.state] || 'Following the pipeline';
  return { title, subtitle, eyebrow, main, focus, issues, terminal, passed, steps, label: progressLabel(job, steps, passed, current, exporting) };
}
// The progress label as plain text pieces: `lead` then the emphasised `strong` part (never both empty).
function progressLabel(job, steps, passed, current, exporting) {
  const total = steps.length;
  if (job.state === 'queued') return { lead: '', strong: 'Waiting to start' };
  if (!total) return { lead: '', strong: 'Preparing the steps' };
  const count = (state) => steps.filter((s) => s.state === state).length;
  const bad = count('failed') + count('rejected'), skipped = count('skipped'), unavailable = count('unavailable');
  const notes = [bad && (job.state === 'rejected' ? 'input needs attention' : `${bad} failed`),
    skipped && `${skipped} skipped`, unavailable && `${unavailable} not available`,
    job.state === 'canceled' && 'cancelled', job.state === 'stopped' && 'saved at this step'].filter(Boolean);
  if (current && (!TERMINAL.has(job.state) || exporting))
    return { lead: `${passed} of ${total} stages complete · `, strong: `${current.state === 'running' ? 'Now' : 'Next'}: ${current.title}` };
  if (!notes.length && passed === total) return { lead: '', strong: `${passed} of ${total} stages complete` };
  return { lead: `${passed} of ${total} complete${notes.length ? ' · ' : ''}`, strong: notes.join(' · ') };
}
// Pipeline order; the stage-2 front, left and back views lead the views.
const IMAGES = [['sketch', 'Starting image'], ['cover', 'Artwork'], ['turnaround', 'Character views'],
  ['front', 'Front'], ['left', 'Left'], ['back', 'Back'], ['right', 'Right'], ['top', 'Top']];
const DOCK_LABEL = {sketch: 'Sketch', turnaround: 'All views'};
// The image the stage shows by default: the front view once it exists, otherwise the newest image.
const defaultImage = (job, available) => (job?.images?.front ? 'front' : available.at(-1)?.[0]) || null;
const imageChoices = job => [...IMAGES.filter(([k]) => job?.images?.[k]),
  ...Object.keys(job?.images || {}).filter(k => !IMAGES.some(([known]) => known === k)).map(k => [k, k.replaceAll('_', ' ')])];
export function stageTiming(s) {
  const parts = [];
  if (s.state === 'running' && Number.isFinite(s.elapsed_seconds)) parts.push(`${Math.floor(s.elapsed_seconds)} s elapsed`);
  else if (Number.isFinite(s.seconds)) parts.push(`${s.seconds.toFixed(1)} s`);
  if (s.state !== 'running' && Number.isFinite(s.cost_usd)) parts.push(`$${s.cost_usd.toFixed(2)}`);
  if (Number.isFinite(s.typical_seconds)) {
    const seconds = Math.max(1, Math.round(s.typical_seconds));
    parts.push(`usually about ${seconds >= 90 ? `${Math.round(seconds / 60)} min` : `${seconds} s`}`);
  }
  return parts.join(' · ');
}
const text = (e, value) => { if (e.textContent !== value) e.textContent = value; };
function node(tag, content, cls) {
  const n = document.createElement(tag); if (content != null) n.textContent = content; if (cls) n.className = cls; return n;
}
const SVG = 'http://www.w3.org/2000/svg';
function icon(name, cls = 'icon icon--s') {
  const svg = document.createElementNS(SVG, 'svg'), use = document.createElementNS(SVG, 'use');
  svg.setAttribute('class', cls); svg.setAttribute('aria-hidden', 'true');
  use.setAttribute('href', `/assets/icons/sprite.svg#i-${name}`); svg.append(use); return svg;
}
// Each step state reads three ways: icon, word and colour.
const STEP_ICON = { passed: 'circle-check', running: 'loader-circle', pending: 'circle-dashed', skipped: 'info',
  unavailable: 'ban', failed: 'circle-x', rejected: 'circle-x', canceled: 'x' };
const SEGMENT = { passed: 'seg--done', running: 'seg--run', skipped: 'seg--skip', unavailable: 'seg--na',
  failed: 'seg--fail', rejected: 'seg--fail' };
// Each progress-bar segment names its stage and state (on hover, on focus and to a screen reader), so the bar
// never relies on colour alone. A step the run will not reach takes the step list's own word.
const SEG_WORD = { passed: 'done', running: 'in progress', skipped: 'skipped', unavailable: 'not available',
  failed: 'failed', rejected: 'needs attention', canceled: 'cancelled' };
export function segmentLabels(steps, terminal) {
  const next = terminal || steps.some((x) => x.state === 'running') ? null : steps.find((x) => x.state === 'pending');
  const after = steps.findIndex((x) => x.state === 'running');
  return steps.map((x, i) => `${x.title} · ${x.state === 'pending'
    ? (terminal ? String(x.status).toLowerCase() : x === next || (after >= 0 && i === after + 1) ? 'up next' : 'not started')
    : SEG_WORD[x.state] || String(x.status).toLowerCase()}`);
}
// Messages the service sends for a step that did not simply pass. They are always shown, never hidden behind a click.
const NOTED = new Set(['skipped', 'unavailable', 'failed', 'rejected']);
export function generationView($, select3D) {
  const dock = viewerDock($);
  let jobId = null, selected = null, manual = false, model = false, job = null, signature = '', activeKey = null;
  let chosenModel = null, shownModel = null, touched = false, modelRequest = 0, loading = null, shownUrl = null, lastImage = null;
  const models = () => job?.models?.length ? job.models : job?.downloads?.glb?.available ? [job.downloads.glb] : [];
  const latestModel = () => models().at(-1);
  function modelCopy() {
    const first = chosenModel?.n === 0;
    text($('#modelTabText'), first ? 'First 3D shape' : '3D preview');
    if (model) text($('#artifactLabel'), first ? 'First 3D shape' : chosenModel?.n ? `Refined version ${chosenModel.n}` : '3D preview');
    $('#modelNote').hidden = !model || !first;
    text($('#modelNote'), first ? 'The texture is still being refined.' : '');
    const latest = latestModel();
    const newer = latest && chosenModel && latest.url !== chosenModel.url &&
      (latest.n == null || latest.n > (chosenModel.n ?? -1));
    $('#modelUpdate').hidden = !model || !newer;
    const added = (latest?.accepted || []).filter(v => !(chosenModel?.accepted || []).includes(v)).length;
    text($('#modelUpdateNote'), added ? `${added} more ${added === 1 ? 'view' : 'views'} added` : 'A sharper texture is ready to inspect.');
  }
  function loadModel(forced = false) {
    if (!chosenModel || loading === chosenModel.url || shownUrl === chosenModel.url) return;
    const candidate = chosenModel, id = jobId, token = ++modelRequest;
    loading = candidate.url;
    Promise.resolve(select3D(candidate.url, {preserveCamera: shownUrl !== null, canReplace: () => forced || !touched})).then(loaded => {
      if (token !== modelRequest || id !== jobId || !model) return;
      if (loaded === false) {
        if (shownModel) { chosenModel = shownModel; modelCopy(); }
        return;
      }
      shownUrl = candidate.url; shownModel = candidate;
      if (candidate.n === 0) trackOnce(`model_first_shown:${id}`, 'model_first_shown');
      else if (candidate.n) trackOnce(`refined_version_shown:${id}:${candidate.n}`, 'refined_version_shown', {n: String(candidate.n)});
    }).finally(() => { if (token === modelRequest) loading = null; });
  }
  const interact = () => { if (model) touched = true; };
  for (const type of ['pointerdown', 'wheel', 'keydown']) $('#preview').addEventListener(type, interact, {passive: true});
  $('#updateModel').addEventListener('click', () => {
    const latest = latestModel();
    if (!latest || latest.url === chosenModel?.url) return;
    if (latest.n) track('update_clicked', {n: String(latest.n)});
    chosenModel = latest; modelCopy(); loadModel(true);
  });
  function pick(key, chosen = false, retry = false) {
    selected = key; manual = chosen; model = key === 'model'; if (key && !model && chosen) lastImage = key;
    $('#previewWrap').hidden = !model;
    $('#viewer').dataset.mode = model ? 'model' : key ? 'image' : 'empty';
    dock.mode(model);
    $('#artboardImage').hidden = model || !key;
    $('#artboardEmpty').hidden = model || !!key;
    $('#modelTab').setAttribute('aria-pressed', String(model));
    $('#artTab').setAttribute('aria-pressed', String(!model));
    if (model) {
      $('#imageError').hidden = true; $('#artboardImage').removeAttribute('src');
      modelCopy(); loadModel();
    }
    else {
      const image = imageChoices(job).find(([k]) => k === key);
      if (image) {
        const url = job.images[key];
        if (retry || $('#artboardImage').getAttribute('src') !== url) {
          $('#imageError').hidden = true; $('#artboardImage').src = url;
          $('#artboardImage').classList.remove('reveal'); void $('#artboardImage').offsetWidth; $('#artboardImage').classList.add('reveal');
        }
        if (!$('#imageError').hidden) $('#artboardImage').hidden = true;
        $('#artboardImage').alt = `${image[1]} for ${job.name || 'your hero'}`;
        text($('#artifactLabel'), image[1]);
      } else { $('#artboardImage').removeAttribute('src'); text($('#artifactLabel'), 'Your character workbench'); }
      ++modelRequest; loading = null; shownUrl = null; select3D(null);
    }
    modelCopy();
    for (const b of $('#images').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.image === key));
  }
  $('#artTab').addEventListener('click', () => {
    const available = imageChoices(job);
    pick(available.some(([k]) => k === lastImage) ? lastImage : defaultImage(job, available), true);
  });
  $('#modelTab').addEventListener('click', () => { if (latestModel()) pick('model', true); });
  $('#artboardImage').addEventListener('error', () => {
    if (model || !selected || $('#artboardImage').getAttribute('src') !== job?.images?.[selected]) return;
    $('#artboardImage').hidden = true; $('#imageError').hidden = false;
  });
  $('#retryImage').addEventListener('click', () => {
    if (!selected || model) return;
    pick(selected, true, true);
  });
  // The segments are one tab stop: arrow keys, Home and End move between them, and the focused or hovered one
  // shows its label in a tooltip under the bar.
  const bar = $('#progressSegments'), tip = $('#progressTip');
  let tipFor = null;
  function showTip(seg) {
    tipFor = seg || null; tip.hidden = !seg;
    if (!seg) return;
    text(tip, seg.getAttribute('aria-label'));
    const box = bar.parentElement.getBoundingClientRect(), at = seg.getBoundingClientRect();
    tip.style.left = `${at.left - box.left + at.width / 2}px`; tip.style.top = `${at.bottom - box.top}px`;
  }
  function renderSegments(steps, fresh, terminal) {
    const keys = steps.map((x) => x.key).join();
    if (bar.dataset.keys !== keys) {
      bar.dataset.keys = keys; showTip(null);
      bar.replaceChildren(...steps.map((x) => {
        const n = node('span', null, 'seg'); n.dataset.key = x.key; n.setAttribute('role', 'img'); n.tabIndex = -1; return n;
      }));
    }
    const labels = segmentLabels(steps, terminal), focused = [...bar.children].indexOf(document.activeElement);
    const current = steps.findIndex((x) => x.state === 'running' || x.state === 'pending');
    const stop = focused >= 0 ? focused : current >= 0 ? current : steps.length - 1;
    steps.forEach((x, i) => {
      const n = bar.children[i], became = !fresh && n.dataset.state && n.dataset.state !== 'passed' && x.state === 'passed';
      n.className = `seg ${SEGMENT[x.state] || ''}${became ? ' is-new' : ''}`.trim();
      n.dataset.state = x.state; n.setAttribute('aria-label', labels[i]); n.tabIndex = i === stop ? 0 : -1;
      if (x.state === 'running') n.setAttribute('aria-current', 'step'); else n.removeAttribute('aria-current');
    });
    bar.dataset.count = String(steps.length);
    if (tipFor) showTip(tipFor.isConnected ? tipFor : null);
  }
  const segOf = (e) => e.target.closest?.('.seg');
  bar.addEventListener('pointerover', (e) => { if (segOf(e)) showTip(segOf(e)); });
  bar.addEventListener('pointerleave', () => showTip(bar.contains(document.activeElement) ? document.activeElement : null));
  bar.addEventListener('focusin', (e) => { if (segOf(e)) showTip(segOf(e)); });
  bar.addEventListener('focusout', (e) => { if (!bar.contains(e.relatedTarget)) showTip(null); });
  bar.addEventListener('keydown', (e) => {
    const segs = [...bar.children], at = segs.indexOf(document.activeElement);
    if (e.key === 'Escape') { showTip(null); return; }
    const to = { ArrowRight: at + 1, ArrowDown: at + 1, ArrowLeft: at - 1, ArrowUp: at - 1, Home: 0, End: segs.length - 1 }[e.key];
    if (at < 0 || to == null || !segs[to]) return;
    e.preventDefault();
    segs[at].tabIndex = -1; segs[to].tabIndex = 0; segs[to].focus();
  });
  function stepRow(s) {
    const li = node('li', null, `st-${s.state}`);
    li.dataset.stage = s.key; li.setAttribute('role', 'listitem');
    if (s.state === 'running') li.setAttribute('aria-current', 'step');
    const head = node('span', null, 'step-head');
    head.append(node('b', s.title), node('span', s.status, 'step-status'));
    li.append(icon(STEP_ICON[s.state] || 'circle-dashed', 'icon icon--s step-icon'), head);
    const timing = stageTiming(s);
    if (timing) li.append(node('small', timing, 'step-duration'));
    if (s.message && (NOTED.has(s.state) || s.state === 'passed' || s.state === 'running')) li.append(node('p', s.message, 'step-note'));
    const source = stageSource(s.key);
    if (source) { const a = node('a', 'View stage source ↗', 'stage-source'); a.href = source; a.target = '_blank'; a.rel = 'noopener'; li.append(a); }
    return li;
  }
  // Skipped and unavailable steps are always listed with the service's own sentence; failed steps join them only when
  // the model itself was kept (a failed hero gets its own banner instead).
  function renderIssues(state, job) {
    const kept = ['succeeded', 'stopped'].includes(job.state);
    const list = state.steps.filter((x) => x.state === 'skipped' || x.state === 'unavailable' || kept && ['failed', 'rejected'].includes(x.state));
    const box = $('#stageIssues');
    box.hidden = !list.length;
    if (!list.length) { box.dataset.signature = ''; return; }
    const fallback = { skipped: 'was skipped.', unavailable: 'is not available for this target.' };
    const signature = JSON.stringify(list.map((x) => [x.key, x.state, x.message]));
    if (box.dataset.signature === signature) return;
    box.dataset.signature = signature;
    const skipped = list.every((x) => x.state === 'skipped');
    text($('#stageIssuesTitle'), skipped ? (list.length === 1 ? '1 step was skipped' : `${list.length} steps were skipped`) : (list.length === 1 ? '1 step needs attention' : `${list.length} steps need attention`));
    $('#stageIssuesList').replaceChildren(...list.map((x) => {
      const li = node('li'); li.append(node('b', x.title), document.createTextNode(` ${x.message || `${x.title} ${fallback[x.state] || 'did not finish.'}`}`)); return li;
    }));
  }
  const toggle = $('#viewSteps'), stepsPanel = $('#progressSteps');
  const open = (on) => { toggle.setAttribute('aria-expanded', String(on)); stepsPanel.inert = !on; $('.progress').classList.toggle('is-open', on); };
  toggle.addEventListener('click', () => open(toggle.getAttribute('aria-expanded') !== 'true'));
  open(false);
  return {
    update(next) {
      const changed = jobId !== next.id;
      if (changed) { $('#imageError').hidden = true; selected = null; lastImage = null; manual = false; model = false; signature = ''; activeKey = null; jobId = next.id;
        chosenModel = null; shownModel = null; touched = false; shownUrl = null; loading = null; ++modelRequest; }
      job = next;
      const latest3D = latestModel();
      if (!chosenModel || !touched && latest3D && (latest3D.n == null || latest3D.n > (chosenModel.n ?? -1))) chosenModel = latest3D;
      const state = generationState(job);
      if (job.state === 'succeeded' && !$('#detail')?.hidden) trackOnce(`result_viewed:${job.id}`, 'result_viewed');
      const running = job.state === 'running' ? state.steps.find((x) => x.key === state.focus?.key) : null;
      text($('#generationEyebrow'), state.eyebrow); text($('#generationTitle'), state.title);
      text($('#generationSubtitle'), running ? running.description : state.subtitle);
      const queue = queueNote(job);
      text($('#queueNoteText'), queue); $('#queueNote').hidden = !queue;
      $('#robloxReady').hidden = !robloxReady(job);
      const label = $('#progressCount'), lead = state.label.lead + '\u0000' + state.label.strong;
      if (label.dataset.label !== lead) {
        label.dataset.label = lead;
        label.replaceChildren(...[state.label.lead && document.createTextNode(state.label.lead), node('b', state.label.strong)].filter(Boolean));
      }
      renderSegments(state.steps, changed, state.terminal);
      $('#detail').dataset.jobState = job.state;
      text($('#workbenchName'), job.name || kindName(job));
      const stageSignature = JSON.stringify(state.steps);
      if ($('#stages').dataset.signature !== stageSignature) {
        $('#stages').dataset.signature = stageSignature;
        $('#stages').replaceChildren(...state.steps.map(stepRow));
      }
      const focus = state.steps.find((s) => s.key === state.focus?.key);
      text($('#currentStep'), job.state === 'queued' ? 'Waiting for a free worker' : state.terminal ? (job.state === 'succeeded' ? 'Your next adventure' : 'What happened') : (focus?.title || 'Waiting for a worker'));
      text($('#stepDescription'), state.terminal ? (job.state === 'succeeded' ? 'Inspect the model, review the recorded checks, then bring the available files into your game.' : state.subtitle) : (focus?.description || 'Your saved job will start when the service is ready.'));
      text($('#stepMessage'), state.focus?.message || '');
      const robloxState = robloxStatus(job);
      const robloxReason = (job.roblox_message || job.export_message || 'the export could not be prepared.').replace(/^Roblox (?:body|files) unavailable for this hero: /, '');
      const exportNote = job.state === 'succeeded' ? (robloxState === 'running' ? 'Roblox files: being prepared' : robloxState === 'ready' ? 'Roblox files: ready' : robloxState === 'unavailable' ? `Roblox files: not available for this hero: ${robloxReason}` : '') : '';
      text($('#exportStatus'), exportNote);
      $('#exportStatus').hidden = !exportNote;
      $('#exportStatus').classList.toggle('sr', robloxState !== 'running');   // only "being prepared" needs words on screen: the pill and the banner say the rest
      renderIssues(state, job);
      const available = imageChoices(job);
      const nextSignature = JSON.stringify(available.map(([k]) => [k, job.images[k]]));
      if (signature !== nextSignature) {
        signature = nextSignature;
        dock.images(available.map(([key, label]) => {
          const b = thumbButton(DOCK_LABEL[key] || label, job.images[key], {image: key});
          b.addEventListener('click', () => pick(key, true, !$('#imageError').hidden)); return b;
        }));
      }
      dock.cameras(job.images);
      $('#modelTab').hidden = !job.downloads?.glb?.available; $('#modelTab').parentElement.hidden = $('#modelTab').hidden;
      const latest = defaultImage(job, available);
      if (!manual || (selected !== 'model' && !job.images?.[selected]) || (selected === 'model' && !job.downloads?.glb?.available)) {
        pick(job.downloads?.glb?.available ? 'model' : latest);
      } else pick(selected, true);
      if (changed) $('#imageError').hidden = true;
      text($('#artboardEmptyText'), state.terminal ? 'No images were saved for this job.' :
        job.state === 'queued' ? 'Starts when a worker is free.' : 'Your first image appears here in about a minute.');
      const idea = !state.terminal && job.kind === 'prompt' && job.prompt ? `“${job.prompt}”` : '';
      text($('#artboardIdea'), idea); $('#artboardIdea').hidden = !idea;
      $('#modelPending').hidden = state.terminal || !!job.downloads?.glb?.available || model || state.focus?.key !== '3' || !available.length;
      return state;
    },
    reset() { jobId = null; job = null; selected = null; signature = ''; chosenModel = null; shownModel = null; shownUrl = null; loading = null; ++modelRequest; select3D(null); }
  };
}
