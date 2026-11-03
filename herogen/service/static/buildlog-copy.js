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
// Why a repainted view was not used, in words a non-specialist follows. The numbers follow in brackets.
const VERDICTS = {
  outline_moved: "the repaint drifted off the model's outline",
  second_face: "the repainted face spills outside the model's face",
  hair_mismatch: "the repainted hair no longer matches the model's hair shape, size or colour",
  outline_not_measurable: 'the outline could not be measured, so the repaint could not be checked',
  missing_features: "the face's eyes, eyebrows or mouth could not be confirmed in the repaint",
  spend_cap: 'the spending cap was reached',
  model_error: 'the model call failed, so there was no repaint to check',
  face_unavailable: 'no face was found to check the repaint against',
};
const verdictText = (why) => VERDICTS[why] || REASONS[why] || 'it did not pass its checks';

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
    case 'cancelled': return e.sibling ? `${callStart(e)}: not needed, the other try passed${e.usd > 0 ? ' (already paid for)' : ''}`
      : `${callStart(e)}: cancelled (step stopped)${e.usd > 0 ? ' (already paid for)' : ''}`;
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

export const listOf = (v) => Array.isArray(v) ? v.length : 0;

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
  body_start: (m) => `turning the 3D shape into the ${n(m.parts) || 15} rigid parts Roblox R15 expects${Number.isFinite(m.lods) ? `, in ${n(m.lods)} levels of detail` : ''}. No model is called here`,
  landmarks: (m) => `loaded ${n(m.joints) || 'the'} joint landmarks${Number.isFinite(m.from_body) ? ` (${n(m.from_body)} from SAM 3D Body${Number.isFinite(m.from_outline) ? `, ${n(m.from_outline)} from the outline` : ''})` : ''}${
    m.flagged ? `; ${n(m.flagged)} differ from the outline estimate` : ''}`,
  colour_sources: (m) => `gathered the colour sources (${m.refined ? 'refined texture' : 'original texture'}${m.refined && m.source_texture ? ' and the original texture' : ''})`,
  decimate: (m) => `reduced the ${Number.isFinite(m.source_triangles) ? `${n(m.source_triangles)}-triangle ` : ''}mesh to candidate sizes${
    Number.isFinite(m.candidates_high) ? `: ${n(m.candidates_high)} for the high-detail model (${n(m.high_min)} to ${n(m.high_max)})` : ''}${
    Number.isFinite(m.candidates_low) ? ` and ${n(m.candidates_low)} for the low-detail model (${n(m.low_min)} to ${n(m.low_max)})` : ''}`,
  parts_have_surface: (m) => `checked that all ${n(m.parts) || 15} parts own mesh surface${m.recoveries ? ` (${n(m.recoveries)} recovered)` : ' (no recovery needed)'}`,
  cut_search: (m) => `searched for cleaner cut positions${Number.isFinite(m.triangles) ? ` on a ${n(m.triangles)}-triangle copy` : ''}${
    Number.isFinite(m.moved) ? `: ${n(m.moved)} of ${n(m.joints)} cuts moved` : ''}`,
  pose_test: (m) => `posed the body in ${n(m.poses) || 'several'} poses${Number.isFinite(m.extra_poses) ? ` (${n(m.extra_poses)} of them only to check the gaps)` : ''}${
    Number.isFinite(m.gap_high) ? `: joint gaps up to ${studs(m.gap_high)} (high) and ${studs(m.gap_low)} (low)${Number.isFinite(m.limit) ? `, limit ${n(m.limit)}, informational in the fast lane` : ''}` : ''}`,
  atlas: (m) => `packed the texture atlases${Number.isFinite(m.high) ? ` (${n(m.high)} px high, ${n(m.low)} px low)` : ''}`,
  texture_transfer: (m) => `transferred the texture to ${m.lods === 2 ? 'both models' : 'the models'}${Number.isFinite(m.colour_views) ? ` from ${n(m.colour_views)} colour views` : ''}`,
  export: (m) => `exported ${m.lods === 2 ? 'both models' : 'the models'}${Number.isFinite(m.pose_renders) ? ` and ${n(m.pose_renders)} pose renders` : ''}`,
  parts_total: (m) => `${n(m.high)} of ${n(m.high_budget)} triangles (high), ${n(m.low)} of ${n(m.low_budget)} (low); largest part ${n(m.largest)} (limit ${n(m.limit)})`,
  rig_write: (m) => `wrote HeroGen-R15.rbxmx: ${n(m.parts) || 15} MeshParts, ${n(m.motors) || 15} Motor6D joints${Number.isFinite(m.attachments) ? `, ${n(m.attachments)} accessory attachments` : ''}${
    Number.isFinite(m.hip_height) ? `, HipHeight ${m.hip_height.toFixed(3)} studs` : ''}`,
  place_write: () => 'wrote HeroGen-Starter.rbxlx: a starter place with a baseplate, a spawn and the rig',
  studio_scripts: (m) => `wrote ${n(m.files) || 3} Studio helper files (bind_meshes.luau, rig_test.server.luau, README.txt)`,
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
const words = (id) => typeof id === 'string' && /^[a-z0-9_]+$/.test(id) ? id.replaceAll('_', ' ') : null;
const LODS = {high: 'high-detail', low: 'low-detail'};
const lodName = (id) => LODS[id] || 'one';
const studs = (v) => Number.isFinite(v) ? `${v.toFixed(2)} studs` : null;
const jointName = (name) => typeof name === 'string' && /^[a-z_]+$/.test(name) ? name.replaceAll('_', ' ') : 'a joint';
const RUNNING = {decimate: 'reducing the mesh to fit the in-game budget', cut_search: 'searching for cleaner cut positions',
  pose_test: 'posing the body to test the joints', atlas: 'packing the texture atlases', texture_transfer: 'transferring the texture to the game body',
  export: 'exporting the game body'};
export const isLongStep = (name) => Boolean(RUNNING[name]) || /^split_(high|low)$/.test(name);
// Rows that fold together: the 15 parts (with their total), and the joints (with the mapping summary).
export function foldOf(e) {
  const what = String(e.what || '');
  if (/^part_[a-z_]+$/.test(what)) return 'parts';
  if (what === 'parts_total') return 'parts_summary';
  if (/^joint_from_(model|outline)$/.test(what) || (/^joint_[a-z_]+$/.test(what) && Number.isFinite(e.m?.x))) return 'joints';
  if (what === 'joints_mapped') return 'joints_summary';
  return null;
}
export function foldTitle(kind, ms, summary) {
  if (kind === 'parts') return `${n(ms.length)} parts${summary ? ` · ${summary}` : ''}`;
  if (summary) return `${n(ms.length)} joints · ${summary.replace(/^Mapped \d+ joints into the front view: /, '')}`;
  const fromBody = ms.filter(m => m.from_body).length, differs = ms.filter(m => m.differs).length;
  return `${n(ms.length)} joints · ${n(fromBody)} from SAM 3D Body, ${n(ms.length - fromBody)} from the outline${differs ? ` · ${n(differs)} differ from the outline estimate by more than 0.5 studs` : ''}`;
}
// The running sentence of a long local sub-step (`<name>_start`), shown until `<name>` arrives.
export function localRunning(name) {
  const split = /^split_(high|low)$/.exec(name);
  if (split) return `splitting the ${lodName(split[1])} model into its 15 parts`;
  return RUNNING[name] || 'working on this step (no model call)';
}
const PROPORTION = {
  clamped: (who, m) => `${who} height clamped from ${m.from_h?.toFixed(3)} H to ${m.to_h?.toFixed(3)} H`,
  wrist_narrowing: (who, m) => `${who} placed where the arm narrows${Number.isFinite(m.fraction) ? ` (${pct(m.fraction)} along it)` : ''}`,
  waist_prior: (who, m) => `no clear ${who.toLowerCase()} found; placed ${Number.isFinite(m.percent) ? `${n(m.percent)} %` : 'part'} of the way from crotch to armpit`,
  no_side_view: (who, m) => `${who}: no side view, estimated${Number.isFinite(m.h) ? ` at ${m.h.toFixed(3)} H` : ''}`,
  arm_few_pixels: (who, m) => `${who}: too few arm pixels${Number.isFinite(m.pixels) ? ` (${n(m.pixels)})` : ''}, estimated from typical proportions`,
};
// Part, joint, cut and level-of-detail names travel in `what` after a prefix.
function prefixed(e) {
  const m = e.m || {}, what = String(e.what || '');
  let x;
  if ((x = /^part_([a-z_]+)$/.exec(what)) && words(x[1])) return `${words(x[1])}: ${n(m.high)} / ${n(m.low)} triangles (high / low)`;
  if ((x = /^joint_([a-z_]+)$/.exec(what)) && words(x[1]) && Number.isFinite(m.x)) return `${words(x[1])} joint at (${m.x.toFixed(2)}, ${m.y.toFixed(2)}, ${
    m.z.toFixed(2)}) from ${m.from_body ? 'SAM 3D Body' : 'the outline'}${m.differs ? ' · differs from the outline estimate by more than 0.5 studs' : ''}`;
  if ((x = /^cut_([a-z_]+)$/.exec(what)) && words(x[1]) && x[1] !== 'search') return `moved the ${words(x[1])} cut ${Number.isFinite(m.fraction)
    ? `${m.fraction > 0 ? '+' : '−'}${Math.abs(m.fraction).toFixed(2)} radius` : ''}${Number.isFinite(m.studs) ? ` (${m.studs > 0 ? '+' : '−'}${Math.abs(m.studs).toFixed(2)} studs)` : ''}`;
  if ((x = /^split_(high|low)$/.exec(what))) return `split the ${lodName(x[1])} model into 15 parts${Number.isFinite(m.triangles) ? `: ${n(m.triangles)}${
    Number.isFinite(m.budget) ? ` of ${n(m.budget)}` : ''} triangles` : ''}${Number.isFinite(m.body_triangles) ? ` (${n(m.body_triangles)} body, ${n(m.cap_triangles)} caps, ${n(m.cover_triangles)} shoulder covers)` : ''}${
    Number.isFinite(m.largest_part) ? `, largest part ${n(m.largest_part)}` : ''}`;
  if ((x = /^proportion_([a-z_]+)$/.exec(what)) && words(x[1])) {
    const who = upper(words(x[1])), say = PROPORTION[e.why];
    return say ? say(who, m) : `${who} estimated from typical proportions`;
  }
  return null;
}
export function localLine(e) {
  const named = prefixed(e);
  if (named) return `${named}${Number.isFinite(e.s) && e.s >= 0.05 ? ` · ${seconds(e.s)}` : ''}`;
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
      // A plain verdict first, then the numbers behind it. Accepted: the headline measures. Not used: the
      // measures that missed their limit, each with that limit.
      const fin = Number.isFinite;
      const checks = [
        [m.overlap, m.overlap_min, (v, l) => v < l, (v, l) => `outline overlap ${pct(v)}${fin(l) ? ` (needs ${pct(l)})` : ''}`, true],
        [m.median_px, m.median_limit, (v, l) => v > l, (v, l) => `median drift ${px(v)}${fin(l) ? ` (limit ${n(l)})` : ''}`, true],
        [m.p90_px, m.p90_limit, (v, l) => v > l, (v, l) => `90th percentile ${px(v)}${fin(l) ? ` (limit ${n(l)})` : ''}`, false],
        [m.agreement, m.agreement_min, (v, l) => v < l, (v, l) => `hair agreement ${v.toFixed(2)}${fin(l) ? ` (needs ${n(l)})` : ''}`, true],
        [m.scale_change, m.scale_change_max, (v, l) => v > l, (v, l) => `scale change ${pct(v)}${fin(l) ? ` (at most ${pct(l)})` : ''}`, false],
        [m.face_outside, m.face_outside_max, (v, l) => v > l, (v, l) => `face outside the model's face ${pct(v)}${fin(l) ? ` (at most ${pct(l)})` : ''}`, false],
      ].filter(([v]) => fin(v));
      const missed = checks.filter(([v, l, miss]) => fin(l) && miss(v, l));
      const measures = e.ok ? checks.filter(c => c[4]).map(([v, , , say]) => say(v, null))
        : (missed.length ? missed : checks).map(([v, l, , say]) => say(v, l));
      const numbers = measures.length ? ` (${measures.join(', ')})` : '';
      if (e.ok) {
        const matches = fin(m.overlap) || fin(m.median_px) ? `matches the model's outline${fin(m.agreement) ? ' and hair' : ''}` : 'passed its checks';
        return `${name}: accepted, the repaint ${matches}${numbers}`;
      }
      if (e.why === 'lower_try_accepted') return `${name}: ${reasonText(e.why)}${m.paid === 1 ? ' (already paid for)' : m.paid === 0 ? ' (stopped before it was paid for)' : ''}`;
      return `${name}: not used, ${verdictText(e.why)}${numbers}`;
    }
    // What the view uses once its tries are known. The lowest accepted try wins, so every lower try was not used.
    case 'view_selected': {
      if (!Number.isInteger(e.try)) return `${regionName(e.view)}: using a repaint that passed its checks.`;
      const before = e.try === 2 ? 'try 1 was not used' : e.try === 3 ? 'tries 1 and 2 were not used'
        : e.try > 3 ? `tries 1 to ${e.try - 1} were not used` : '';
      return `${regionName(e.view)}: using the repaint from try ${e.try}${before ? `; ${before}` : ''}.`;
    }
    case 'view_kept': {
      const tries = m.tries === 1 ? 'its one try was not used' : m.tries === 2 ? 'neither try was used'
        : Number.isInteger(m.tries) && m.tries > 2 ? `none of its ${n(m.tries)} tries was used` : 'no try passed';
      if (e.why === 'spend_cap' || e.why === 'model_error') return `${regionName(e.view)}: keeping the original paint (${reasonText(e.why)}).`;
      return `${regionName(e.view)}: keeping the original paint; ${tries}${e.why ? ` (${verdictText(e.why)})` : ''}.`;
    }
    case 'face_source': {
      const from = Number.isInteger(e.try) ? `, try ${e.try}` : '';
      if (e.why === 'beauty') return `Face: using the beauty repaint (${regionName(e.view)}${from}).`;
      if (e.why === 'default') return `Face: using the ordinary repaint (${regionName(e.view)}${from}).`;
      if (e.why === 'default_fast_fallback') return 'Face: no face repaint was accepted, so the face keeps the paint from the ordinary texture.';
      return null;
    }
    case 'extra_try': return `${regionName(e.view)}: the earlier tries were not used, so Muse Image gets one more try${Number.isInteger(e.try) ? ` (try ${e.try})` : ''}.`;
    case 'wave': return `Wave ${n(m.wave) || 1}: asking Muse Image to repaint ${m.views === 1 ? '1 view' : `${n(m.views) || 'several'} views`}${
      Number.isFinite(m.from_version) ? ` from version ${n(m.from_version)}` : ''}, ${n(m.tries) || 2} tries each, all at once${Number.isFinite(e.n) ? ` (${n(e.n)} requests)` : ''}`;
    case 'recovery_sheet': return `The first sheet had no usable views. Asking Muse Image for one more sheet; the fast lane makes one extra attempt only.`;
    case 'complete_figure': return 'The drawing is cropped at the feet, so Muse Image is asked to complete the figure.';
    case 'over_budget_high': case 'over_budget_low': return `The ${lodName(e.what.slice(12))} model is over its budget (${n(m.triangles)} against ${n(m.budget)} triangles); the fast lane kept the smallest complete body.`;
    default: return null;
  }
}

// Plain meanings for the rig checks an independent reader runs on the Roblox file.
const CHECK_MEANINGS = {
  xml_well_formed: 'rig file is valid Roblox XML', one_root_model: 'one root Model', humanoid_r15: 'Humanoid is R15',
  humanoid_scale_values: 'body-scale values present', root_part: 'HumanoidRootPart is an invisible Part and the PrimaryPart',
  sections_r15_names: 'exactly the 15 R15 parts', joints_named_and_linked: 'Motor6D joints link each part to its parent',
  rig_attachment_pairs: 'joint attachment pairs', attachments_at_pivots: 'attachments sit on the joint pivots',
  attachment_pairs_coincide: 'both attachments of each joint coincide', motor_c0_c1_equal_attachments: 'Motor6D C0/C1 equal the attachments',
  joint_chain_reachable: 'every part is reachable from the root', joints_inside_parts: 'every joint pivot lies inside both its parts',
  rest_pose_position: 'rest pose reproduces part positions', rest_pose_rotation: 'rest pose reproduces part rotations',
  section_extents_match_glb: 'part sizes match the model file', section_centres_match_glb: 'part centres match the model file',
  rotations_proper: 'no mirrored rotations', triangles_per_mesh: 'largest part under the per-mesh triangle limit',
  hip_height: 'HipHeight puts the feet on the floor', feet_lowest: 'nothing reaches below the feet',
  faces_minus_z: 'character faces forward', not_mirrored: "right arm is on the character's right",
  face_front_on_front: 'face attachment is on the front of the head', accessory_attachments: 'accessory attachments present and inside their parts',
  avatar_part_scale_type: 'all parts use the Classic scale type', meshes_bound: 'meshes bound in Studio; 0 until the script runs',
  place_xml_well_formed: 'starter place is valid Roblox XML', place_referents_resolve: 'every reference in the place resolves',
  place_services: 'Workspace, Lighting and StarterPlayer present', place_baseplate: 'an anchored baseplate',
  place_spawn: 'one anchored spawn', place_spawn_on_baseplate: 'the spawn sits on the baseplate',
  place_marker: 'the starter-place marker is set', place_starter_player_empty: 'StarterPlayer starts empty',
  place_rig_present: 'the rig is in Workspace', place_rig_matches_file: 'the place holds the same rig as the rig file',
  place_rig_rigid_move_position: 'the rig was moved as one piece', place_rig_rigid_move_rotation: 'the rig was turned as one piece',
  place_rig_upright: 'the rig stands upright', place_feet_on_baseplate: 'the feet stand on the baseplate',
  place_rig_clear_of_spawn: 'the rig stands clear of the spawn', place_rig_faces_spawn: 'the rig faces the spawn',
};
const CHECK_GROUP = {4: 'body', 6: 'rig'};
const OPS = {'<': 'under', '<=': 'at most', '==': 'exactly', '>=': 'at least', '>': 'over', '!=': 'not'};
const value = (v) => !Number.isFinite(v) ? null : Math.abs(v) >= 1000 ? n(v) : Number(v.toPrecision(4)).toString();
const checkName = (name) => typeof name === 'string' && /^[a-z0-9_]+$/.test(name) ? name : 'check';
// One recorded check: its name, what it means when known, the measured value against its limit, and the verdict.
export function checkLine(e) {
  const m = e.m || {}, name = checkName(e.name), meaning = CHECK_MEANINGS[name];
  const measured = value(m.value) != null ? `measured ${value(m.value)}` : 'not measured';
  const limit = value(m.limit) != null ? `, needs ${OPS[e.op] ? `${OPS[e.op]} ` : ''}${value(m.limit)}` : '';
  const verdict = e.ok ? 'passed' : e.gate === false ? 'not met (informational)' : 'failed';
  return `${name}${meaning ? ` (${meaning})` : ''}: ${verdict} · ${measured}${limit}`;
}
// A step's check summary, from its individual checks when they were recorded, else from the summary event.
export function checksLine(e, list = [], step = null) {
  const what = CHECK_GROUP[step] ? `${CHECK_GROUP[step]} checks` : 'checks';
  let gated, gatedFailed, info, infoMissed;
  if (list.length && !(e && Number.isFinite(e.n) && e.n !== list.length)) {
    gated = list.filter(c => c.gate === true).length; gatedFailed = list.filter(c => c.gate === true && !c.ok).length;
    info = list.length - gated; infoMissed = list.filter(c => c.gate !== true && !c.ok).length;
  } else if (e && Number.isFinite(e.n)) {
    const failed = listOf(e.failed);
    gated = Number.isFinite(e.gated) ? e.gated : e.n - listOf(e.advisory); gatedFailed = failed;
    info = e.n - gated; infoMissed = Number.isFinite(e.passed) ? Math.max(0, e.n - e.passed - failed) : listOf(e.advisory);
  } else return null;
  const head = gated ? (gatedFailed ? `${n(gatedFailed)} of ${n(gated)} required ${what} failed` : `${n(gated)} of ${n(gated)} required ${what} passed`)
    : `${n(info)} informational ${what}`;
  const tail = gated && info ? ` · ${n(info)} informational${infoMissed ? ` (${n(infoMissed)} not met in the fast lane)` : ''}`
    : !gated && infoMissed ? ` (${n(infoMissed)} not met)` : '';
  return `${head}${tail}`;
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
const SEAM = {no_shared_rim: 'no shared rim between its parts, so its seam cannot be measured (shown as unverified, not a failure)',
  empty_section: 'one of its parts is empty, so its seam cannot be measured', unmeasurable: 'its seam cannot be measured on this character'};
export function skipLine(e) {
  const seam = /^seam_([a-z_]+)$/.exec(String(e.what || ''));
  if (seam && words(seam[1])) return `${upper(words(seam[1]))}: ${SEAM[e.why] || 'its seam was not measured'}.`;
  if (e.what === 'cut_search') return 'The cut search is turned off; cuts stay at the joints.';
  if (e.what === 'identity_checks') return 'The fast lane keeps this sketch without the extra identity checks.';
  if (e.what === 'acceptance_judges') return 'The fast lane skips the visual acceptance judges; structural checks are informational.';
  if (e.what === 'view') return `${regionName(e.view)}: skipped (${reasonText(e.why)}).`;
  if (e.what === 'top_view') return 'No usable top view: continuing with the four side views.';
  if (STEP_SKIPS[e.why]) return STEP_SKIPS[e.why];
  return e.why === 'fast_lane' ? 'The fast lane leaves this part out.' : 'This part was skipped.';
}

const LANDMARK_FIX = {hips_cross_pelvis: 'the hips crossed the pelvis', shoulder_elbow_order: 'the elbow sat above the shoulder',
  hand_direction: 'the hand pointed back at the body', radius_below_forearm: 'the arm radius was below the forearm'};
export function recoverLine(e) {
  const what = String(e.what || ''), m = e.m || {};
  let x;
  if ((x = /^merged_limb_([a-z_]+)$/.exec(what)) && words(x[1])) return `The ${words(x[1])} had merged into the body; its parts were rebuilt from the nearest bone.`;
  if (what === 'merged_limb') return 'A limb had merged into the body; its part was rebuilt from the nearest bone.';
  if ((x = /^limb_root_([a-z_]+)$/.exec(what)) && words(x[1])) return `Repaired the ${words(x[1])} joint (${e.why === 'reversed_root' ? 'it pointed the wrong way' : 'its parts were empty'}).`;
  if ((x = /^sleeve_([a-z_]+)$/.exec(what)) && words(x[1])) return `The ${words(x[1])} was a hollow sleeve; the arm was found inside it${Number.isFinite(m.offset_studs) ? ` (${studs(m.offset_studs)} in)` : ''}.`;
  if ((x = /^landmark_([a-z_]+)$/.exec(what)) && words(x[1])) return `Repaired the ${words(x[1])} landmark (${LANDMARK_FIX[e.why] || 'it was out of place'}).`;
  if ((x = /^pivot_([a-z_]+)_(high|low)$/.exec(what)) && words(x[1])) return `Placed the ${words(x[1])} pivot in the ${lodName(x[2])} model another way${Number.isFinite(m.distance_studs) ? ` (${studs(m.distance_studs)} away)` : ''}.`;
  return 'HeroGen recovered and carried on.';
}

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
    case 'model_ready': return 'Your 3D model is ready. Roblox files continue in the background.';
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

