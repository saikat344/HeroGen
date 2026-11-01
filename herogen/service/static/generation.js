// Presentation follows reported model and export stage states.
import { trackOnce } from './analytics.js';
const COPY = {
  p: ['Your idea', 'Sketching your idea', 'Turn your description into a character sketch.'],
  0: ['Prepare input', 'Preparing your image', 'Check the input before creating the character artwork.'],
  1: ['Artwork', 'Painting your hero', 'Create the coloured character artwork that guides the next steps.'],
  2: ['Character views', 'Seeing every side', 'Create front, back and side views for reconstruction.'],
  3: ['3D shape', 'Building the 3D shape', 'Reconstruct the character in three dimensions.'],
  '3f': ['Texture refine', 'Refining the texture', 'Improve the texture on your website model.'],
  4: ['Game body', 'Preparing the game body', 'Build the body parts and textures for the selected target.'],
  6: ['Roblox rig', 'Preparing the Roblox rig', 'Prepare the Roblox rig when this export is available.'],
  5: ['Results', 'Collecting your results', 'Prepare the results and available downloads.'],
};
const TERMINAL = new Set(['succeeded', 'failed', 'rejected', 'canceled']);
const INTERRUPTED = new Set(['failed', 'rejected', 'canceled']);
const STATE = { passed: 'Passed', running: 'Running', pending: 'Waiting', failed: 'Failed', rejected: 'Needs attention',
  canceled: 'Cancelled', skipped: 'Skipped', unavailable: 'Unavailable' };
export function generationState(job) {
  const stages = (Array.isArray(job.stages) ? job.stages : []).filter((s) => s && typeof s === 'object');
  const main = stages.filter((s) => !s.side);
  const active = main.find((s) => s.state === 'running');
  const focus = active || main.find((s) => s.key === job.stage) || main.find((s) => s.state === 'pending');
  const issues = main.filter((s) => ['failed', 'rejected', 'unavailable', 'skipped'].includes(s.state));
  const title = { queued: 'Waiting for a free worker', running: 'Your hero is taking shape.',
    succeeded: 'Meet your hero.', failed: 'Your hero needs attention.', rejected: 'Let’s revisit your input.', canceled: 'Generation cancelled.' }[job.state] || 'Your hero’s progress';
  const subtitle = job.state === 'succeeded' ? (issues.length ? 'Your model is ready. See the notes below.' : 'Explore the result and the available downloads.') :
    job.state === 'queued' ? 'Your job is saved. It will begin when a worker is available.' :
    job.state === 'canceled' ? 'The job has stopped. Any saved images remain available to inspect.' :
    INTERRUPTED.has(job.state) ? 'Review the reported issue below. Saved outputs remain available.' :
    (COPY[focus?.key]?.[1] || focus?.label || 'Following the pipeline');
  return { title, subtitle, main, focus, issues, terminal: TERMINAL.has(job.state),
    passed: main.filter((s) => s.state === 'passed').length,
    steps: main.map((s) => ({ ...s, title: COPY[s.key]?.[0] || s.label || s.key,
      description: COPY[s.key]?.[2] || 'Follow the status reported by the service.',
      status: s.state === 'pending' && TERMINAL.has(job.state) ? (job.state === 'succeeded' && ['4', '6', '5'].includes(s.key) && job.export_state === 'running' ? 'Waiting' : 'Not reached') : STATE[s.state] || s.state || 'Not reported' })) };
}
const IMAGES = [['sketch', 'Starting image'], ['cover', 'Artwork'], ['turnaround', 'Character views'],
  ['front', 'Front'], ['back', 'Back'], ['left', 'Left'], ['right', 'Right']];
const text = (e, value) => { if (e.textContent !== value) e.textContent = value; };
function node(tag, content, cls) {
  const n = document.createElement(tag); if (content != null) n.textContent = content; if (cls) n.className = cls; return n;
}
export function generationView($, select3D) {
  let jobId = null, selected = null, manual = false, model = false, job = null, signature = '', activeKey = null;
  function pick(key, chosen = false, retry = false) {
    selected = key; manual = chosen; model = key === 'model';
    $('#previewWrap').hidden = !model;
    $('#artboardImage').hidden = model || !key;
    $('#artboardEmpty').hidden = model || !!key;
    $('#modelTab').setAttribute('aria-pressed', String(model));
    $('#artTab').setAttribute('aria-pressed', String(!model));
    if (model) {
      $('#imageError').hidden = true; $('#artboardImage').removeAttribute('src');
      text($('#artifactLabel'), '3D preview'); select3D(job.downloads.glb.url);
    }
    else {
      const image = IMAGES.find(([k]) => k === key);
      if (image) {
        const url = job.images[key];
        if (retry || $('#artboardImage').getAttribute('src') !== url) { $('#imageError').hidden = true; $('#artboardImage').src = url; }
        if (!$('#imageError').hidden) $('#artboardImage').hidden = true;
        $('#artboardImage').alt = `${image[1]} for ${job.name || 'your hero'}`;
        text($('#artifactLabel'), image[1]);
      } else { $('#artboardImage').removeAttribute('src'); text($('#artifactLabel'), 'Your character workbench'); }
      select3D(null);
    }
    for (const b of $('#images').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.image === key));
  }
  $('#artTab').addEventListener('click', () => {
    const available = IMAGES.filter(([k]) => job?.images?.[k]);
    pick(available.at(-1)?.[0] || null, true);
  });
  $('#modelTab').addEventListener('click', () => { if (job?.downloads?.glb?.available) pick('model', true); });
  $('#artboardImage').addEventListener('error', () => {
    if (model || !selected || $('#artboardImage').getAttribute('src') !== job?.images?.[selected]) return;
    $('#artboardImage').hidden = true; $('#imageError').hidden = false;
  });
  $('#retryImage').addEventListener('click', () => {
    if (!selected || model) return;
    pick(selected, true, true);
  });
  return {
    update(next) {
      const changed = jobId !== next.id;
      if (changed) { $('#imageError').hidden = true; selected = null; manual = false; model = false; signature = ''; activeKey = null; jobId = next.id; }
      job = next;
      const state = generationState(job);
      if (job.state === 'succeeded' && !$('#detail')?.hidden) trackOnce(`result_viewed:${job.id}`, 'result_viewed');
      text($('#generationTitle'), state.title); text($('#generationSubtitle'), state.subtitle);
      text($('#progressCount'), `${state.passed} of ${state.main.length} stages passed`);
      text($('#workbenchName'), job.name || (job.kind === 'prompt' ? 'Your imagined hero' : 'Your uploaded character'));
      const stageSignature = JSON.stringify(state.steps);
      if ($('#stages').dataset.signature !== stageSignature) {
        $('#stages').dataset.signature = stageSignature;
        $('#stages').replaceChildren(...state.steps.map((s, i) => {
          const li = node('li', null, `st-${s.state}`);
          li.dataset.stage = s.key; li.setAttribute('role', 'listitem');
          if (s.state === 'running') li.setAttribute('aria-current', 'step');
          li.append(node('span', s.state === 'passed' ? '✓' : String(i + 1).padStart(2, '0'), 'stage-number'),
            node('b', s.title), node('span', s.status, 'step-status'));
          if (typeof s.seconds === 'number' && Number.isFinite(s.seconds)) li.append(node('small', `${s.seconds.toFixed(1)} s recorded`, 'step-duration'));
          return li;
        }));
      }
      if (state.focus?.key !== activeKey) {
        activeKey = state.focus?.key;
        const active = [...$('#stages').children].find((n) => n.dataset.stage === activeKey);
        if (active) $('#stages').scrollLeft = active.offsetLeft - $('#stages').offsetLeft - ($('#stages').clientWidth - active.clientWidth) / 2;
      }
      const focus = state.steps.find((s) => s.key === state.focus?.key);
      text($('#currentStep'), job.state === 'queued' ? 'Waiting for a free worker' : state.terminal ? (job.state === 'succeeded' ? 'Your next adventure' : 'What happened') : (focus?.title || 'Waiting for a worker'));
      text($('#stepDescription'), state.terminal ? (job.state === 'succeeded' ? 'Inspect the model, review the recorded checks, then bring the available files into your game.' : state.subtitle) : (focus?.description || 'Your saved job will start when the service is ready.'));
      text($('#stepMessage'), state.focus?.message || '');
      const issues = state.steps.filter((s) => ['failed', 'rejected', 'unavailable', 'skipped'].includes(s.state));
      const robloxState = job.roblox_state || job.export_state;
      const robloxReason = (job.roblox_message || job.export_message || 'the export could not be prepared.').replace(/^Roblox (?:body|files) unavailable for this hero: /, '');
      const exportNote = job.state === 'succeeded' ? (robloxState === 'running' ? 'Roblox files: being prepared' : robloxState === 'ready' ? 'Roblox files: ready' : robloxState === 'unavailable' ? `Roblox files: not available for this hero: ${robloxReason}` : '') : '';
      text($('#exportStatus'), exportNote);
      $('#exportStatus').hidden = !exportNote;
      $('#stageIssues').replaceChildren(...issues.map((s) => node('p', s.message || `${s.title} was skipped.`)));
      $('#stageIssues').hidden = !issues.length;
      const available = IMAGES.filter(([k]) => job.images?.[k]);
      const nextSignature = JSON.stringify(available.map(([k]) => [k, job.images[k]]));
      if (signature !== nextSignature) {
        signature = nextSignature;
        $('#images').replaceChildren(...available.map(([key, label]) => {
          const b = node('button'); b.type = 'button'; b.dataset.image = key; b.setAttribute('aria-label', `View ${label.toLowerCase()}`);
          const image = node('img'); image.src = job.images[key]; image.alt = ''; image.loading = 'lazy';
          b.append(image, node('span', label)); b.addEventListener('click', () => pick(key, true, !$('#imageError').hidden)); return b;
        }));
      }
      $('#modelTab').hidden = !job.downloads?.glb?.available;
      const latest = available.at(-1)?.[0] || null;
      if (!manual || (selected !== 'model' && !job.images?.[selected]) || (selected === 'model' && !job.downloads?.glb?.available)) {
        pick(job.downloads?.glb?.available ? 'model' : latest);
      } else pick(selected, true);
      if (changed) $('#imageError').hidden = true;
      text($('#artboardEmptyText'), state.terminal ? 'No images were saved for this job.' : 'Your first image will appear here when it is ready.');
      return state;
    },
    reset() { jobId = null; job = null; selected = null; signature = ''; select3D(null); }
  };
}
