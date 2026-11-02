// Read-only presentation of the selected job's recorded stage and publication times.
const SOURCE = {p: 'pipeline/prompt_sketch.py', 0: 'pipeline/inputs.py', 1: 'pipeline/stages.py',
  2: 'pipeline/stages.py', 3: 'blender/stage3.py', '3f': 'blender/refine/__init__.py',
  4: 'blender/stage4.py', 6: 'roblox/export.py', 5: 'pipeline/results.py'};
// Plain-language names and descriptions for each step, and for the models and calls the records name.
const STEP = {
  p: ['Sketch from your description', 'Muse Image draws a character design from your idea, then redraws it as a full-body pencil sketch. The sketch gives the rest of the build a clear outline to follow.'],
  0: ['Prepare input', 'HeroGen (local) checks that the sketch is large enough and its lines are clear enough to work from. No model is called.'],
  1: ['Artwork', 'Muse Image paints the colour artwork from your sketch, keeping its pose and outline. SAM 3.1 finds the character and its feet so HeroGen can confirm the whole body is in frame. For an uploaded picture, Muse Spark first checks that it is a drawing and not a photo of a real person.'],
  2: ['Character views', 'Muse Image draws a turnaround sheet (front, right, back and left) and a top view. SAM 3.1 outlines each figure so HeroGen can split the sheet, line the views up and check that they agree.'],
  3: ['3D shape', 'SAM 3D Objects builds a textured 3D mesh from the front view and SAM 3D Body finds the body joints, both on cloud GPUs. HeroGen then puts the mesh in its own frame and turns it to face forward.'],
  '3f': ['Texture refine', 'HeroGen renders the 3D shape from several cameras, asks Muse Image to repaint each render in better detail, and uses SAM 3.1 to check that each repaint still matches the outline. Accepted paint goes back onto the texture. A view that does not pass keeps its original paint.'],
  4: ['Game body', 'HeroGen (local) turns the 3D shape into the 15 rigid parts Roblox R15 expects, placing the cuts at the joints and fitting the in-game triangle budget. No model is called here, and this is usually the longest step.'],
  6: ['Roblox rig', 'HeroGen (local) writes the Roblox R15 rig and checks it with an independent reader. No model is called. If this step fails, your website model is unaffected.'],
  5: ['Results page', 'HeroGen (local) writes the results page for this build. No model is called.'],
};
const LOCAL_STEPS = new Set(['0', '4', '6', '5']);
const LOCAL = 'HeroGen (local)';
const MODELS = [['sam-3d-objects', 'SAM 3D Objects'], ['sam-3d-body', 'SAM 3D Body'], ['sam-3.1', 'SAM 3.1'],
  ['muse-image', 'Muse Image'], ['muse-spark', 'Muse Spark']];
const STATE_WORD = {running: 'running', passed: 'done', failed: 'failed', rejected: 'stopped by a check', skipped: 'skipped'};
const modelName = id => id ? (MODELS.find(([prefix]) => String(id).toLowerCase().startsWith(prefix))?.[1] || String(id)) : LOCAL;
const sign = a => a < 0 ? `−${-a}` : `+${a}`;
function regionName(region) {
  const m = /^(head|body|hair-head|hair-upper)([+-]\d+)$/.exec(region || '');
  if (!m) return 'an extra view';
  const part = m[1], angle = Number(m[2]);
  if (part === 'head' && angle === 0) return 'the face close-up';
  if (part === 'body' && angle === 0) return 'the front view';
  if (part === 'body' && angle === 180) return 'the back view';
  if (part === 'body' && Math.abs(angle) === 90) return `the side view at ${sign(angle)}°`;
  if (part === 'hair-upper' && angle === 180) return 'the upper hair, back';
  return `${{head: 'the head', body: 'the body', 'hair-head': 'the hair around the head', 'hair-upper': 'the upper hair'}[part]} at ${sign(angle)}°`;
}
const PURPOSE = {
  'sketch design': 'Drew the character design from your idea', 'full-body sketch': 'Redrew it as a full-body sketch',
  'cover art': 'Painted the colour artwork from the sketch', 'turnaround': 'Drew the turnaround sheet: front, right, back and left',
  'top view': 'Drew the top view', 'segment character (turnaround)': 'Found the four figures on the sheet',
  'segment character (top view)': 'Cut the top view out of its background', '3D shape': 'Built a textured 3D mesh from the front view',
  'body joints': 'Found the body joints from the front view'};
// A call becomes a one-sentence label; calls that repeat many times can also name the group they belong to.
function describe(call) {
  const purpose = String(call.purpose || '');
  if (String(call.model).startsWith('muse-spark')) return {label: 'Checked that the image is a drawing, not a photo of a real person'};
  if (purpose === 'edit') return {label: 'Edited an image', gkey: 'edit', group: n => `Edited ${n} images`};
  if (PURPOSE[purpose]) return {label: PURPOSE[purpose]};
  let m;
  if ((m = /^repaint (\S+)$/.exec(purpose))) return {label: `Repainted ${regionName(m[1])}`, gkey: 'repaint', group: n => `Repainted ${n} views`};
  if ((m = /^segment (face|hair|head) \(repaint (\S+)\)$/.exec(purpose))) return {
    label: `Checked the ${m[1]} on the repaint of ${regionName(m[2])}`, gkey: `check-${m[1]}`, group: n => `Checked the ${m[1]} on ${n} repaints`};
  if ((m = /^segment (character|creature|person|feet) \(cover art\)$/.exec(purpose))) return {label: `Looked for the ${m[1]} in the sketch`};
  if ((m = /^segment (face|head) \((?:turnaround|texture refine)\)$/.exec(purpose))) return {label: `Outlined the ${m[1]}`};
  if ((m = /^segment (face|hair|head) \(segment\)$/.exec(purpose))) return {label: `Outlined the ${m[1]}`, gkey: `outline-${m[1]}`, group: n => `Outlined the ${m[1]} ${n} times`};
  return {label: purpose ? purpose[0].toUpperCase() + purpose.slice(1) : 'Model call'};
}
export function formatSeconds(s) {
  if (!Number.isFinite(s)) return null;
  return s >= 120 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${s.toFixed(1)} s`;
}
export function formatCost(usd) {
  if (!Number.isFinite(usd)) return null;
  if (usd === 0) return '$0';
  return usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
}
const usually = s => Number.isFinite(s) ? `usually about ${Math.max(1, Math.round(s)) >= 90 ? `${Math.round(s / 60)} min` : `${Math.max(1, Math.round(s))} s`}` : null;
export function checksSummary(checks) {
  if (!Array.isArray(checks) || !checks.length) return null;
  const required = checks.filter(c => c?.gate === true), advisory = checks.length - required.length;
  const unmet = list => list.filter(c => c?.pass === false).length;
  const parts = [];
  if (required.length) parts.push(unmet(required) ? `${unmet(required)} of ${required.length} required ${required.length === 1 ? 'check' : 'checks'} not met`
    : `${required.length} required ${required.length === 1 ? 'check' : 'checks'} passed`);
  const others = checks.filter(c => c?.gate !== true);
  if (advisory) parts.push(`${advisory} advisory${unmet(others) ? ` (${unmet(others)} not met)` : ''}`);
  return parts.join(' · ');
}
function callEvent(call, floor) {
  const start = Date.parse(call.started), {label, gkey, group} = describe(call);
  const reused = call.cached === true || call.outcome === 'reused';
  const number = v => typeof v === 'number' && Number.isFinite(v) ? v : null;
  const seconds = reused ? null : number(call.wall_seconds), cost = reused ? 0 : number(call.cost_usd);
  const failed = call.outcome === 'failed', cancelled = call.outcome === 'cancelled';
  return {kind: 'call', time: Math.max(Number.isFinite(start) ? start : floor, floor), who: modelName(call.model),
    state: reused ? 'reused' : failed ? 'failed' : cancelled ? 'cancelled' : 'passed',
    text: failed ? `${label} (failed)` : cancelled ? `${label} (cancelled)` : label, gkey, group,
    seconds, cost,
    meta: [reused ? 'reused a saved result' : formatSeconds(seconds), formatCost(cost)].filter(Boolean).join(' · ')};
}
// Three or more repeats of the same kind of call fold into one row that opens to list them.
function foldCalls(calls, key) {
  const groups = new Map();
  const id = c => `${c.who}|${c.gkey}`;
  for (const c of calls) if (c.gkey) groups.set(id(c), [...(groups.get(id(c)) || []), c]);
  const rows = [], folded = new Set();
  for (const c of calls) {
    const same = c.gkey && groups.get(id(c));
    if (!same || same.length < 3) { rows.push(c); continue; }
    if (folded.has(id(c))) continue;
    folded.add(id(c));
    const longest = Math.max(...same.map(x => x.seconds ?? 0)), cost = same.reduce((sum, x) => sum + (x.cost ?? 0), 0);
    const allReused = same.every(x => x.state === 'reused'), failed = same.filter(x => x.state === 'failed').length;
    rows.push({kind: 'group', id: `${key}:${id(c)}`, time: Math.min(...same.map(x => x.time)), who: c.who,
      state: failed ? 'failed' : allReused ? 'reused' : 'passed', text: c.group(same.length), children: same,
      meta: [allReused ? 'reused saved results' : `longest ${formatSeconds(longest)}`, formatCost(cost)].filter(Boolean).join(' · ')});
  }
  return rows;
}
export function stageEvents(job, run, now = Date.now()) {
  const events = [], total = (job.stages || []).length;
  (job.stages || []).forEach((stage, index) => {
    const record = run?.stages?.find(r => String(r.stage) === stage.key);
    let start = Date.parse(record?.started);
    if (!Number.isFinite(start)) {
      if (stage.state === 'running' && Number.isFinite(stage.elapsed_seconds)) start = now - stage.elapsed_seconds * 1000;
      else return;
    }
    const [title, about] = STEP[stage.key] || [stage.label || `Step ${stage.key}`, null];
    const running = stage.state === 'running';
    const seconds = Number.isFinite(record?.seconds) ? record.seconds : stage.seconds;
    const cost = Number.isFinite(record?.cost_usd) ? record.cost_usd : stage.cost_usd;
    const meta = [STATE_WORD[stage.state] || stage.state,
      running ? (Number.isFinite(stage.elapsed_seconds) ? `${Math.floor(stage.elapsed_seconds)} s so far` : null) : formatSeconds(seconds),
      !running && Number.isFinite(cost) ? `${formatCost(cost)} (estimate)` : null, usually(stage.typical_seconds)].filter(Boolean).join(' · ');
    events.push({kind: 'step', time: start, state: stage.state, text: `Step ${index + 1} of ${total} · ${title}`, meta,
      checks: running ? null : checksSummary(record?.checks), note: about,
      message: ['failed', 'rejected', 'skipped'].includes(stage.state) && stage.message ? stage.message : null});
    if (running || !record) return;
    const calls = (Array.isArray(record.calls) ? record.calls : []).filter(c => c && typeof c === 'object')
      .map(c => callEvent(c, start)).sort((a, b) => a.time - b.time);
    if (!calls.length && LOCAL_STEPS.has(stage.key)) calls.push({kind: 'call', time: start, who: LOCAL, state: 'passed', text: 'no model call',
      seconds: Number.isFinite(seconds) ? seconds : null, cost: 0, meta: [formatSeconds(seconds), '$0'].filter(Boolean).join(' · ')});
    events.push(...foldCalls(calls, stage.key));
  });
  for (const model of job.models || []) if (model.n && Number.isFinite(Date.parse(model.time))) {
    events.push({kind: 'version', time: Date.parse(model.time), text: `Refined version ${model.n} ready`, state: 'passed'});
  }
  return events.map((e, i) => [e, i]).sort((a, b) => a[0].time - b[0].time || a[1] - b[1]).map(([e]) => e);
}
const lineText = e => `${e.who ? `${e.who}: ` : ''}${e.text}${e.meta ? ` · ${e.meta}` : ''}`;
// The copied log keeps every line, including the calls inside folded rows.
export function logText(rows) {
  const stamp = e => new Date(e.time).toISOString();
  return rows.flatMap(e => [`${stamp(e)}  ${lineText(e)}`, ...(e.checks ? [`${' '.repeat(26)}${e.checks}`] : []),
    ...(e.message ? [`${' '.repeat(26)}${e.message}`] : []), ...(e.note ? [`${' '.repeat(26)}${e.note}`] : []),
    ...(e.children || []).map(c => `${stamp(c)}    ${lineText(c)}`)]).join('\n');
}
function el(tag, className, text) {
  const n = document.createElement(tag); if (className) n.className = className; if (text != null) n.textContent = text; return n;
}
function timeNode(ms) {
  const t = el('time', null, new Date(ms).toLocaleTimeString([], {hour12: false})); t.dateTime = new Date(ms).toISOString(); return t;
}
function lineNode(e) {
  const line = el('span', 'log-line');
  if (e.who) line.append(el('span', e.who === LOCAL ? 'log-who log-who--local' : 'log-who', e.who));
  line.append(el('span', e.kind === 'step' ? 'log-title' : 'log-text', e.text));
  if (e.meta) line.append(el('span', 'log-meta', e.meta));
  return line;
}
function rowNode(e, open) {
  const li = el('li', `log-${e.kind} st-${e.state}`);
  if (e.kind === 'group') {
    const details = el('details', 'log-fold'), summary = el('summary');
    summary.append(timeNode(e.time), lineNode(e));
    const list = el('ul', 'log-children');
    for (const child of e.children) { const c = el('li', `log-call st-${child.state}`); c.append(timeNode(child.time), lineNode(child)); list.append(c); }
    details.append(summary, list); details.open = open.has(e.id);
    details.addEventListener('toggle', () => { if (details.open) open.add(e.id); else open.delete(e.id); });
    li.append(details); return li;
  }
  const body = el('span', 'log-body');
  body.append(lineNode(e));
  for (const [cls, value] of [['log-checks', e.checks], ['log-message', e.message], ['log-note', e.note]]) if (value) body.append(el('span', cls, value));
  li.append(timeNode(e.time), body);
  return li;
}
export function workbenchLog($, api) {
  let job = null, run = null, signature = '', request = 0, log = '';
  const records = new Map(), open = new Set();
  function render() {
    const list = $('#stageLog'), follow = $('#followLive').checked;
    const position = list.scrollTop;
    const rows = stageEvents(job, run);
    log = logText(rows);
    list.replaceChildren(...rows.map(row => rowNode(row, open)));
    if (!rows.length) { const li = document.createElement('li'); li.textContent = 'Waiting for recorded stage events.'; list.append(li); }
    list.scrollTop = follow ? list.scrollHeight : position;
  }
  $('#followLive').addEventListener('change', () => { if (job) render(); });
  $('#copyLog').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(log); $('#logStatus').textContent = 'Log copied'; }
    catch { $('#logStatus').textContent = 'Copy unavailable. Select the log text to copy it.'; }
  });
  return {update(next) {
    if (job?.id !== next.id) { run = null; signature = ''; records.clear(); }
    job = next; render();
    const current = JSON.stringify([job.id, job.stages?.map(s => [s.key, s.state]), job.models]);
    if (current === signature) return;
    signature = current;
    const token = ++request;
    $('#logStatus').textContent = 'Loading recorded events…';
    const id = job.id;
    const stages = (job.stages || []).filter(s => !['pending','running'].includes(s.state) && !records.has(s.key));
    Promise.allSettled(stages.map(s => api(`/runs/r1/${encodeURIComponent(id)}/stage-${encodeURIComponent(s.key)}.json`)
      .then(record => ({...record, stage: s.key})))).then(results => {
      if (token !== request) return;
      const failures = [];
      for (const result of results) {
        if (result.status === 'fulfilled') records.set(result.value.stage, result.value);
        else if (result.reason.status !== 404) failures.push(result.reason.message);
      }
      run = {stages: [...records.values()]}; render();
      $('#logStatus').textContent = failures.length ? `Could not load stage events: ${failures.join('; ')}` :
        results.some(r => r.status === 'rejected') ? 'Some stage times were not recorded.' : 'Recorded stage times';
    });
  }, reset() { ++request; job = null; run = null; signature = ''; records.clear(); }};
}
export function stageSource(key) {
  return SOURCE[key] ? `https://github.com/saikat344/HeroGen/blob/main/herogen/${SOURCE[key]}` : null;
}
