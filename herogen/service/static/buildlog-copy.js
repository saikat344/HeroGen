// Plain sentences for the Build log. Events carry only structured fields (kind, who, what, numbers); every word a
// person reads comes from this file, so the copy can change without touching recorded runs. An id that is not
// listed here falls back to a neutral sentence: a raw id, reason code or model message is never shown.

export const WHO = {
  'muse-image': 'Muse Image', 'sam-3.1': 'SAM 3.1', 'muse-spark': 'Muse Spark',
  'sam-3d-objects': 'SAM 3D Objects', 'sam-3d-body': 'SAM 3D Body', herogen: 'HeroGen',
};
const GPU = new Set(['sam-3d-objects', 'sam-3d-body']);
export const isGpu = (who) => GPU.has(who);
export function whoName(who) {
  const name = WHO[who];
  if (!name) return 'HeroGen';
  return GPU.has(who) ? `${name} (GPU)` : name;
}

export const STEPS = {
  p: 'Sketch from your description', 0: 'Prepare input', 1: 'Artwork', 2: 'Character views', 3: '3D shape',
  '3f': 'Texture refine', 4: 'Game body', 6: 'Roblox rig', 5: 'Results page',
};
export const stepName = (key) => STEPS[key] || 'Pipeline step';

// Texture regions the refine step repaints. Which side is the character's left is not recorded, so side views
// keep their angle.
export const REGIONS = {
  'head+0': 'Face close-up', 'body+0': 'Front', 'body+90': 'Side view +90°', 'body-90': 'Side view −90°',
  'body+180': 'Back', 'hair-upper+180': 'Upper hair, back',
};
export const regionName = (view) => REGIONS[view] || 'One view';

// Reason ids (`why`) as plain text. Anything unlisted reads as the neutral fallback.
export const REASONS = {
  outline_moved: 'outline moved too far',
  second_face: 'a second face appeared',
  hair_mismatch: 'hair no longer matches',
  lower_try_accepted: 'not needed, the other try passed',
  outline_not_measurable: 'the outline could not be measured',
  missing_features: 'parts of the character went missing',
  spend_cap: 'the spending cap was reached',
  model_error: 'the model call failed',
  face_unavailable: 'no face was found to check against',
  head_not_found: 'no head found, skipped the face close-up',
  no_usable_views: 'no usable views could be cut from it',
  fast_lane: 'the fast lane leaves it out',
  slot: 'waiting for a free slot',
};
export const reasonText = (why) => REASONS[why] || 'it did not pass the checks';

// Failure codes (`code`) on calls, errors, requeues and step ends.
export const CODES = {
  http_408: 'the model service timed out', http_409: 'the model service was busy', http_425: 'the model service was busy',
  http_429: 'the model service asked HeroGen to slow down', http_4xx: 'the model service refused the request',
  http_5xx: 'the model service returned an error', timeout: 'it took too long and was stopped',
  network: 'the connection to the model service dropped', unreadable: 'the answer could not be read',
  no_slot: 'no free slot opened in time', replay_miss: 'no saved result was found to reuse',
  remote_error: 'the GPU service returned an error', error: 'it stopped with an internal error',
  model_unavailable: 'the model service is not available right now', gpu_unavailable: 'the GPU service did not answer',
  stage_error: 'the step stopped with an internal error', rejected: 'it did not pass its checks',
  model_missing: 'there is no 3D model to work from', not_available: 'it is not available for this target',
  input_missing: 'the input could not be found', restarts: 'the worker restarted too often',
  worker_error: 'the worker stopped unexpectedly', canceled: 'it was cancelled',
};
export const codeText = (code) => CODES[code] || 'it stopped before finishing';

// SAM 3.1 concepts (`target`) and where it looks (`what` of the step that asked).
const TARGETS = {character: 'the character', person: 'a person', creature: 'a creature', feet: 'the feet', foot: 'the feet',
  boot: 'the boots', shoe: 'the shoes', leg: 'the legs', face: 'the face', head: 'the head', hair: 'the hair'};
const PLACES = {prompt_sketch: 'the sketch', sketch_checks: 'the sketch', cover_art: 'the drawing',
  turnaround: 'the turnaround sheet', top_view: 'the top view', repaint: 'the repainted view',
  texture_refine: 'the repainted view'};
const target = (e) => TARGETS[e.target] || 'the outline';
const place = (e) => e.view && REGIONS[e.view] ? `the ${REGIONS[e.view].toLowerCase()}` : PLACES[e.what] || 'the image';
const tryText = (e) => Number.isInteger(e.try) ? `, try ${e.try}` : '';
const view = (e) => regionName(e.view).toLowerCase();

// One entry per model and purpose: `start` while it runs (after "Who: "), `done` when it returns. A purpose a
// model does not list falls back to the step's name.
export const CALLS = {
  'muse-image': {
    sketch_design: {start: 'drawing a character design from your idea', done: 'design drawing ready'},
    full_body_sketch: {start: 'redrawing it as a full-body pencil sketch (the design is the reference)', done: 'full-body sketch ready'},
    cover_art: {start: 'painting the colour artwork from the sketch (same pose and outline)', done: 'colour artwork ready'},
    turnaround: {start: 'drawing one turnaround sheet with front, right, back and left side by side', done: 'turnaround sheet ready'},
    top_view: {start: 'drawing the top view', done: 'top view ready'},
    repaint: {start: (e) => `repainting the ${view(e)}${tryText(e)}`, done: (e) => `repaint ready: ${view(e)}${tryText(e)}`},
  },
  'sam-3.1': {
    segment: {start: (e) => `finding ${target(e)} on the 3D renders`, done: (e) => `looked for ${target(e)} on the 3D renders`},
    repaint: {start: (e) => `checking ${target(e)} on the repaint of the ${view(e)}${tryText(e)}`,
      done: (e) => `checked ${target(e)} on the repaint of the ${view(e)}${tryText(e)}`},
    '*': {start: (e) => `looking for ${target(e)} in ${place(e)}`, done: (e) => `looked for ${target(e)} in ${place(e)}`},
  },
  'muse-spark': {
    cover_art: {start: 'checking whether this is a photo of a real person', done: 'photo check done'},
    prompt_sketch: {start: 'checking the sketch shows the same character as the design', done: 'same-character check done'},
    same_character: {start: 'checking the sketch shows the same character as the design', done: 'same-character check done'},
  },
  'sam-3d-objects': {'3d_shape': {start: 'building a textured 3D mesh from the front view and its outline', done: (e) => {
    const m = e.m || {}, bits = [Number.isFinite(m.vertices) && `${n(m.vertices)} vertices`, Number.isFinite(m.faces) && `${n(m.faces)} faces`,
      Number.isFinite(m.texture_px) && `${n(m.texture_px)} px texture`].filter(Boolean);
    return `returned a mesh${bits.length ? ` (${bits.join(', ')})` : ''}`;
  }}},
  'sam-3d-body': {body_joints: {start: 'finding the body joints from the front view', done: (e) =>
    `returned the body skeleton${Number.isFinite(e.m?.people) ? ` (${n(e.m.people)} ${e.m.people === 1 ? 'person' : 'people'})` : ''}`}},
};
const STAGE_WHAT = {prompt_sketch: 'p', input: '0', cover_art: '1', turnaround: '2', texture_refine: '3f', game_body: '4',
  roblox_rig: '6', results: '5'};
const pick = (value, e) => typeof value === 'function' ? value(e) : value;
const entryFor = (e) => CALLS[e.who]?.[e.what] || CALLS[e.who]?.['*'] || null;
export const upper = (s) => s ? s[0].toUpperCase() + s.slice(1) : s;
const GPU_NAMES = {'a100-40gb': 'A100', 'a100-80gb': 'A100', a100: 'A100', a10g: 'A10G', l4: 'L4', h100: 'H100', t4: 'T4'};

export function callStart(e) {
  const entry = entryFor(e);
  if (entry) return pick(entry.start, e);
  const step = STAGE_WHAT[e.what] ?? e.step;
  return isGpu(e.who) ? 'working on the GPU' : STEPS[step] ? `working on ${STEPS[step].toLowerCase()}` : 'working';
}
// How a finished call ended. Only `ok` and `reused` are successes; a cancelled hedged try is normal and quiet.
// Every other outcome, including ones this file does not know (an exception name, `unreadable`), is a failure.
export function callState(outcome) {
  return outcome === 'ok' || outcome === 'reused' ? 'passed' : outcome === 'cancelled' ? 'kept' : 'failed';
}
export function callDone(e) {
  const entry = entryFor(e);
  switch (e.outcome) {
    case 'reused': return `${callStart(e)}: reused a saved result`;
    case 'cancelled': return `${callStart(e)}: not needed, the other try passed${e.usd > 0 ? ' (already paid for)' : ''}`;
    case 'unknown': return `${callStart(e)}: outcome unknown (may be billed)`;
    case 'ok': {
      let done = entry ? pick(entry.done, e) : 'done';
      const m = e.m || {};
      if (Number.isFinite(m.load_s) && m.load_s > 0) done += ` · cold start ${m.load_s.toFixed(1)} s`;
      if (Number.isFinite(m.infer_s)) done += ` + ${m.infer_s.toFixed(1)} s inference${GPU_NAMES[e.gpu] ? ` on an ${GPU_NAMES[e.gpu]}` : ''}`;
      return done;
    }
    default: return `${callStart(e)} failed: ${codeText(e.code ?? (e.outcome === 'unreadable' ? 'unreadable' : null))}`;
  }
}
export const GPU_STILL = 'Still waiting for the GPU. If it was asleep the first call can take about a minute longer.';

// Numbers: seconds to 0.1, dollars to the cent (four places only below a cent), times as m:ss.
export function seconds(s) {
  if (!Number.isFinite(s)) return '';
  return `${s.toFixed(1)} s`;
}
// A typical duration: whole seconds for one call, minutes for a long step.
export function about(s, {minutesFrom = 90} = {}) {
  if (!Number.isFinite(s)) return '';
  if (s < 1) return 'under 1 s';
  if (s < minutesFrom) return `${Math.round(s)} s`;
  return `${Math.round(s / 60)} min`;
}
export const aboutCall = (s) => about(s, {minutesFrom: 600});
export const usually = (s) => s < 1 ? 'usually under 1 s' : `usually about ${about(s)}`;
export function usd(v) {
  if (!Number.isFinite(v)) return '';
  if (v === 0) return '$0';
  if (v < 0.01) return `$${v.toFixed(4).replace(/0+$/, '')}`;
  const three = v.toFixed(3);
  return `$${three.endsWith('0') ? v.toFixed(2) : three}`;
}
export function clock(ms) {
  if (!Number.isFinite(ms)) return '';
  // Round to tenths first, so 59.96 s reads 01:00.0, never 00:60.0.
  const tenths = Math.round(Math.max(0, ms) / 100), m = Math.floor(tenths / 600), s = (tenths - m * 600) / 10;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}
export function elapsed(ms) {
  const total = Math.max(0, Math.round(ms / 1000));
  return total < 60 ? `${total} s` : `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}
const n = (v) => Number.isFinite(v) ? v.toLocaleString('en-US', {maximumFractionDigits: 2}) : null;
const px = (v) => Number.isFinite(v) ? `${v.toFixed(1)} px` : null;
const pct = (v) => Number.isFinite(v) ? `${Math.round(v <= 1 ? v * 100 : v)} %` : null;

const listOf = (v) => Array.isArray(v) ? v.length : 0;

// Local work and the pipeline's own decisions. Each returns one sentence, or null when there is nothing honest to
// say (the row is then left out rather than invented).
const LOCAL = {
  input_checked: (m) => `checked the sketch size and line contrast${Number.isFinite(m.min_side_px) ? ` (short side ${n(m.min_side_px)} px${Number.isFinite(m.contrast) ? `, contrast ${n(m.contrast)}` : ''})` : ''}, no model call`,
  sheet_split: (m) => `split the sheet into front, right, back and left${Number.isFinite(m.figures) ? ` (${n(m.figures)} of 4 figures)` : ''}`,
  facing_search: (m) => `faced the mesh forward${Number.isFinite(m.tested) ? ` (tested ${n(m.tested)} rotations${Number.isFinite(m.facing_deg) ? `, turned ${n(m.facing_deg)}°` : ''})` : ''}${
    Number.isFinite(m.front_overlap) ? ` and matched its outline to the front view (overlap ${pct(m.front_overlap)})` : ''}`,
  mesh_placed: (m) => `placed the mesh in HeroGen's frame${[Number.isFinite(m.triangles) && `${n(m.triangles)} triangles`,
    m.open_edges === 0 ? 'no open edges' : Number.isFinite(m.open_edges) && `${n(m.open_edges)} open edges`,
    Number.isFinite(m.front_overlap) && `outline overlap ${pct(m.front_overlap)}`].filter(Boolean).join(', ').replace(/^(.+)$/, ' ($1)')}`,
  joints_mapped: (m) => `mapped ${n(m.joints) || 'the'} joints into the front view${Number.isFinite(m.from_model) && Number.isFinite(m.from_outline)
    ? `: ${n(m.from_model)} from SAM 3D Body, ${n(m.from_outline)} from the outline` : ''}${Number.isFinite(m.flagged)
    ? `; ${n(m.flagged)} differ from the outline estimate by more than ${n(m.flag_studs ?? 0.5)} studs` : ''}`,
  landmarks_loaded: (m) => `loaded ${n(m.joints) || 'the'} joint landmarks${Number.isFinite(m.from_model) ? ` (${n(m.from_model)} from SAM 3D Body)` : ''}`,
  decimated: (m) => `reduced the ${n(m.source_triangles) || ''} triangle mesh to fit the in-game budget${Number.isFinite(m.high) && Number.isFinite(m.low)
    ? `: ${n(m.high)} triangles (high), ${n(m.low)} (low)` : ''}`.replace('the  triangle', 'the'),
  cuts_searched: () => 'searched for cleaner cut positions at the neck, waist, elbows, wrists, knees and ankles',
  parts_split: (m) => `split the mesh into ${n(m.parts) || 15} Roblox R15 parts${Number.isFinite(m.high_triangles) && Number.isFinite(m.low_triangles)
    ? ` (${n(m.high_triangles)} triangles high, ${n(m.low_triangles)} low)` : ''}`,
  rig_written: (m) => `wrote the Roblox rig${Number.isFinite(m.parts) ? `: ${n(m.parts)} parts, ${n(m.joints)} joints, ${n(m.attachments)} accessory attachments` : ''}${
    Number.isFinite(m.hip_height) ? `, hip height ${m.hip_height.toFixed(2)} studs` : ''}`,
  results_written: () => 'wrote the results page',
  views_rendered: (m) => `rendered the 3D shape from ${n(m.cameras ?? m.views) || 'several'} ${m.cameras === 1 ? 'camera' : 'cameras'}${Number.isFinite(m.wave) ? ` for wave ${n(m.wave)}` : ''}`,
};
// Local lines are written without the speaker; the panel shows a HeroGen chip and "Copy log" adds "HeroGen (local): ".
const jointName = (name) => typeof name === 'string' && /^[a-z_]+$/.test(name) ? name.replaceAll('_', ' ') : 'a joint';
export function localLine(e) {
  if (e.what === 'joint_from_model' || e.what === 'joint_from_outline') {
    const m = e.m || {};
    return `${jointName(e.name)} from ${e.what === 'joint_from_model' ? 'SAM 3D Body' : 'the outline'}${Number.isFinite(m.off_outline_studs)
      ? ` · ${m.off_outline_studs.toFixed(2)} studs from the outline estimate` : ''}${m.flagged ? ' (differs by more than 0.5 studs)' : ''}`;
  }
  if (e.what === 'head_probe') return `${e.ok === false ? 'no head found on the 3D renders' : 'found the face and head on the 3D renders'}${
    Number.isFinite(e.s) && e.s >= 0.05 ? ` · ${seconds(e.s)}` : ''}`;
  const make = LOCAL[e.what];
  const said = make ? make(e.m || {}) : 'worked on this step (no model call)';
  return Number.isFinite(e.s) && e.s >= 0.05 ? `${said} · ${seconds(e.s)}` : said;
}

export function decisionLine(e) {
  const m = e.m || {};
  switch (e.what) {
    case 'view_result': {
      const name = `${regionName(e.view)}${tryText(e)}`;
      // Accepted: the headline measures. Not used: the measures that missed their limit, each with that limit.
      const fin = Number.isFinite;
      const checks = [
        [m.overlap, m.overlap_min, (v, l) => v < l, (v, l) => `outline overlap ${pct(v)}${fin(l) ? ` (needs ${pct(l)})` : ''}`, true],
        [m.median_px, m.median_limit, (v, l) => v > l, (v, l) => `median drift ${px(v)}${fin(l) ? ` (limit ${n(l)})` : ''}`, true],
        [m.p90_px, m.p90_limit, (v, l) => v > l, (v, l) => `90th percentile ${px(v)}${fin(l) ? ` (limit ${n(l)})` : ''}`, false],
        [m.agreement, m.agreement_min, (v, l) => v < l, (v, l) => `hair agreement ${v.toFixed(2)}${fin(l) ? ` (needs ${n(l)})` : ''}`, true],
        [m.scale_change, m.scale_change_max, (v, l) => v > l, (v, l) => `scale change ${pct(v)}${fin(l) ? ` (at most ${pct(l)})` : ''}`, false],
      ].filter(([v]) => fin(v));
      const missed = checks.filter(([v, l, miss]) => fin(l) && miss(v, l));
      const measures = e.ok ? checks.filter(c => c[4]).map(([v, , , say]) => say(v, null))
        : (missed.length ? missed : checks).map(([v, l, , say]) => say(v, l));
      if (e.ok) return `${name}: accepted${measures.length ? ` (${measures.join(', ')})` : ''}`;
      if (e.why === 'lower_try_accepted') return `${name}: ${reasonText(e.why)}${m.paid === 1 ? ' (already paid for)' : m.paid === 0 ? ' (stopped before it was paid for)' : ''}`;
      return `${name}: not used (${reasonText(e.why)}${measures.length ? `: ${measures.join(', ')}` : ''})`;
    }
    case 'view_selected': return `${regionName(e.view)}: kept the repaint${Number.isInteger(e.try) ? ` from try ${e.try}` : ''}`;
    case 'view_kept': return e.why === 'spend_cap' || e.why === 'model_error'
      ? `${regionName(e.view)}: kept the original paint (${reasonText(e.why)}).`
      : `${regionName(e.view)}: neither try passed${e.why ? ` (${reasonText(e.why)})` : ''}. Keeping the original paint for this view.`;
    case 'wave': return `Wave ${n(m.wave) || 1}: asking Muse Image to repaint ${m.views === 1 ? '1 view' : `${n(m.views) || 'several'} views`}${
      Number.isFinite(m.from_version) ? ` from version ${n(m.from_version)}` : ''}, ${n(m.tries) || 2} tries each, all at once${Number.isFinite(e.n) ? ` (${n(e.n)} requests)` : ''}`;
    case 'recovery_sheet': return `The first sheet had no usable views. Asking Muse Image for one more sheet; the fast lane makes one extra attempt only.`;
    case 'complete_figure': return 'The drawing is cropped at the feet, so Muse Image is asked to complete the figure.';
    default: return null;
  }
}

// A step's check summary: `n` checks, `passed`, and the names of failed hard and informational checks.
export function checksLine(e) {
  if (!Number.isFinite(e.n)) return null;
  const failed = listOf(e.failed), advisory = listOf(e.advisory);
  const passed = Number.isFinite(e.passed) ? e.passed : e.n - failed - advisory;
  const bits = [failed && `${n(failed)} hard ${failed === 1 ? 'limit' : 'limits'} failed`,
    advisory && `${n(advisory)} informational only in the fast lane`].filter(Boolean);
  return `${n(passed)} of ${n(e.n)} checks passed${bits.length ? ` (${bits.join(', ')})` : ''}`;
}
const OPS = {'<': 'under', '<=': 'at most', '==': 'exactly', '>=': 'at least', '>': 'over', '!=': 'not'};
// One recorded check, for a failure or a near miss: the measured value against its limit.
export function checkLine(e) {
  const m = e.m || {}, name = typeof e.name === 'string' ? e.name.replaceAll('_', ' ') : 'a check';
  const measured = Number.isFinite(m.value) ? `measured ${n(m.value)}` : 'not measured';
  const limit = Number.isFinite(m.limit) ? `, needs ${OPS[e.op] || ''} ${n(m.limit)}`.replace('  ', ' ') : '';
  return `${upper(name)}: ${e.ok ? 'passed' : e.gate === false ? 'did not pass (informational)' : 'failed'} · ${measured}${limit}`;
}

export function versionLine(e) {
  const m = e.m || {};
  const views = Number.isFinite(e.passed) && Number.isFinite(e.of) ? `${n(e.passed)} of ${n(e.of)} views repainted` : '';
  const pending = Number.isFinite(m.pending) && m.pending > 0 ? `${n(m.pending)} still being judged` : '';
  const bits = [views, pending].filter(Boolean).join('; ');
  return `Your model now shows version ${n(e.n) || 'a new'}${bits ? ` (${bits})` : ''}.`;
}

const STEP_SKIPS = {
  spend_cap: 'Texture refine reached its spending cap. Showing the best texture so far.',
  rough_shape: 'Texture refine was skipped because the 3D shape is rough.',
  views_rejected: 'Kept the unrefined texture because its views did not pass the checks.',
};
export function skipLine(e) {
  if (e.what === 'identity_checks') return 'The fast lane keeps this sketch without the extra identity checks.';
  if (e.what === 'acceptance_judges') return 'The fast lane skips the visual acceptance judges; structural checks are informational.';
  if (e.what === 'view') return `${regionName(e.view)}: skipped (${reasonText(e.why)}).`;
  if (e.what === 'top_view') return 'No usable top view: continuing with the four side views.';
  if (STEP_SKIPS[e.why]) return STEP_SKIPS[e.why];
  return e.why === 'fast_lane' ? 'The fast lane leaves this part out.' : 'This part was skipped.';
}

const RECOVERS = {merged_limb: 'A limb had merged into the body; its part was rebuilt from the nearest bone.'};
export const recoverLine = (e) => RECOVERS[e.what] || 'HeroGen recovered and carried on.';

// An error row: the step's own plain message when the job record has one, else the code in words.
export function errorLine(e, stage) {
  if (stage?.message && ['failed', 'rejected'].includes(stage.state)) return stage.message;
  return e.code && CODES[e.code] ? `This step stopped: ${CODES[e.code]}.` : 'This step stopped. Technical details are kept in the server logs.';
}

const IMAGES = {design: 'Design drawing', full_body: 'Full-body sketch', sketch: 'Sketch', cover: 'Artwork', turnaround: 'Turnaround sheet', front: 'Front view',
  right: 'Right view', back: 'Back view', left: 'Left view', top: 'Top view'};
export const imageLine = (e) => `${IMAGES[e.what] || 'A new image'} ready`;

export function jobLine(e, job) {
  switch (e.kind) {
    case 'job_start': {
      const attempt = Number.isFinite(e.a) && e.a > 1 ? ` Attempt ${n(e.a)}${Number.isFinite(e.of) ? ` of ${n(e.of)}` : ''}.` : '';
      if (job?.kind === 'sketch' || !job?.prompt) return `Job started.${attempt}`;
      const idea = Array.from(job.prompt);
      return `Job started. Idea: "${idea.length > 160 ? `${idea.slice(0, 160).join('')}…` : job.prompt}"${attempt}`;
    }
    case 'queue': return Number.isFinite(e.s) ? `Worker started after ${seconds(e.s)} in the queue` : 'Worker started';
    case 'requeue': return `This step stopped (${codeText(e.code)}). Starting attempt ${n(e.next) || 2}${Number.isFinite(e.of) ? ` of ${n(e.of)}` : ''}; finished work is kept.`;
    case 'cancel': return 'Cancelled. Everything finished so far stays saved.';
    case 'model_ready': return 'Your website model is ready. Roblox files continue in the background.';
    case 'truncated': return 'The log reached its size limit. From here it shows only step changes, versions and errors.';
    case 'job_end': {
      const totals = [Number.isFinite(e.s) && `Total ${elapsed(e.s * 1000)}`, Number.isFinite(e.usd) && `${usd(e.usd)} (estimate)`].filter(Boolean).join(', ');
      const tail = totals ? ` ${totals}.` : '';
      if (e.state === 'succeeded') return e.export === 'ready' ? `Roblox files ready.${tail}`
        : e.export === 'unavailable' ? `Your model is ready. Roblox files are not available for this hero.${tail}` : `Run finished.${tail}`;
      if (e.state === 'canceled') return `Cancelled. Everything finished so far stays saved.${tail}`;
      if (e.state === 'stopped') return `Saved at this step.${tail}`;
      return `The run stopped. Everything finished so far stays saved.${tail}`;
    }
    default: return null;
  }
}

