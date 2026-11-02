import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildModel, copyText, orderEvents, rowMeta, stepMeta, typicalFor, buildLog, POLL_MS, IDLE_MS }
  from '../herogen/service/static/buildlog.js';
import * as copy from '../herogen/service/static/buildlog-copy.js';
import { sampleEvents, SAMPLE_STAGES } from './buildlog_fixture.mjs';

const TYPICAL = JSON.parse(readFileSync(new URL('../herogen/service/static/typical-calls.json', import.meta.url)));
const START = Date.UTC(2026, 10, 2, 2, 55, 10);
const job = (states = {}, extra = {}) => ({id: 'j_1', state: 'running', started_at: START / 1000, prompt: 'A fox samurai',
  stages: SAMPLE_STAGES.map(s => ({...s, state: states[s.key] || 'pending', seconds: null, cost_usd: null})), ...extra});
const ev = (s, kind, step, extra = {}) => ({v: 1, t: START + s * 1000, kind, ...(step == null ? {} : {step}), ...extra});
const rowsOf = (model, key) => model.steps.find(s => s.key === key).rows;

test('parallel calls keep one row each, in start order, and copy both lines in time order', () => {
  const events = [ev(1, 'step_start', '1'),
    ev(1.1, 'call_start', '1', {id: 'a', who: 'sam-3.1', what: 'cover_art', target: 'character'}),
    ev(1.1, 'call_start', '1', {id: 'b', who: 'sam-3.1', what: 'cover_art', target: 'feet'}),
    ev(4.9, 'call_end', '1', {id: 'b', who: 'sam-3.1', what: 'cover_art', target: 'feet', outcome: 'ok', s: 3.8, usd: 0.0025}),
    ev(6.5, 'call_end', '1', {id: 'a', who: 'sam-3.1', what: 'cover_art', target: 'character', outcome: 'ok', s: 5.4, usd: 0.0025})];
  // Delivered out of order across two writers: sorting by time restores the story, ties keep file order.
  const shuffled = [events[0], events[4], events[1], events[3], events[2]];
  assert.deepEqual(orderEvents(shuffled).map(e => e.id ?? e.kind), ['step_start', 'a', 'b', 'b', 'a']);
  const model = buildModel(job({1: 'running'}), shuffled, {typical: TYPICAL, now: START + 7000});
  const rows = rowsOf(model, '1');
  assert.deepEqual(rows.map(r => [r.key, r.state, r.text]), [
    ['c:a', 'passed', 'Looked for the character in the drawing'], ['c:b', 'passed', 'Looked for the feet in the drawing']]);
  assert.equal(rowMeta(rows[0]), '5.4 s · $0.0025');
  assert.deepEqual(copyText(model).split('\n'), [
    '+00:01.0  ▸ Step 3 of 9 · Artwork  (usually about 33 s)',
    '+00:01.1  … SAM 3.1: looking for the character in the drawing…',
    '+00:01.1  … SAM 3.1: looking for the feet in the drawing…',
    '+00:04.9  ✓ SAM 3.1: looked for the feet in the drawing · 3.8 s · $0.0025 (usually 3 s)',
    '+00:06.5  ✓ SAM 3.1: looked for the character in the drawing · 5.4 s · $0.0025 (usually 4 s)']);
  assert.equal(stepMeta(model.steps.find(s => s.key === '1')), 'running 6 s · $0.005 est. · usually about 33 s');
});

test('a running call shows its live time and usual time, then turns amber only well past its slow end', () => {
  const start = [ev(0, 'step_start', '2'), ev(0, 'call_start', '2', {id: 'x', who: 'muse-image', what: 'turnaround'})];
  let [row] = rowsOf(buildModel(job({2: 'running'}), start, {typical: TYPICAL, now: START + 9000}), '2');
  assert.equal(row.state, 'running');
  assert.equal(row.text, 'Drawing one turnaround sheet with front, right, back and left side by side…');
  assert.equal(rowMeta(row), '9 s · usually 22 s');
  assert.ok(!row.late);
  [row] = rowsOf(buildModel(job({2: 'running'}), start, {typical: TYPICAL, now: START + 43000}), '2');
  assert.ok(row.late, '1.5 x the p90 of 28.2 s is 42.3 s');
  // Its step is over but no finish was written: it is never left spinning.
  [row] = rowsOf(buildModel(job({2: 'failed'}), start, {typical: TYPICAL}), '2');
  assert.equal(row.state, 'unknown');
  assert.equal(rowMeta(row), 'not recorded');
});

test('GPU calls say they are still waiting after 20 s and reused results are never timed', () => {
  const events = [ev(0, 'step_start', '3'), ev(0, 'gpu_wait', '3', {id: 'g', who: 'sam-3d-objects', what: '3d_shape'}),
    ev(0, 'call_start', '3', {id: 'h', who: 'sam-3d-body', what: 'body_joints'}),
    ev(0.1, 'call_end', '3', {id: 'h', who: 'sam-3d-body', what: 'body_joints', outcome: 'reused', s: 0.1, usd: 0.012})];
  const model = buildModel(job({3: 'running'}), events, {typical: TYPICAL, now: START + 25000});
  const [gpu, reused] = rowsOf(model, '3');
  assert.ok(gpu.gpuWait);
  assert.equal(copy.whoName(gpu.who), 'SAM 3D Objects (GPU)');
  assert.equal(reused.text, 'Reused a saved result');
  assert.equal(rowMeta(reused), '$0');
  assert.match(copyText(model), /SAM 3D Body \(GPU\): reused a saved result · \$0$/m);
});

test('typical times need at least eight past calls', () => {
  const table = {min_n: 8, calls: {'1|sam-3.1|cover_art|feet': {n: 7, p50: 2.6, p90: 4.1},
    '1|sam-3.1|cover_art|character': {n: 8, p50: 3.7, p90: 5.9}, '3f|sam-3.1|*|hair': {n: 9, p50: 1.9, p90: 3.3}}};
  assert.equal(typicalFor(table, {step: '1', who: 'sam-3.1', what: 'cover_art', target: 'feet'}), null);
  assert.equal(typicalFor(table, {step: '1', who: 'sam-3.1', what: 'cover_art', target: 'character'}).p50, 3.7);
  assert.equal(typicalFor(table, {step: '3f', who: 'sam-3.1', what: 'texture_refine', target: 'hair'}).p50, 1.9);
  assert.equal(typicalFor(TYPICAL, {step: '3', who: 'sam-3d-objects', what: '3d_shape'}).p50, 174.9);
  assert.equal(typicalFor(null, {step: '1'}), null);
  for (const row of Object.values(TYPICAL.calls)) assert.ok(row.n >= 8 && row.p50 <= row.p90);
});

test('texture views, reasons and speakers read as plain text', () => {
  const events = [ev(0, 'step_start', '3f'),
    ev(1, 'decision', '3f', {what: 'wave', n: 10, m: {wave: 1, views: 5, tries: 2}}),
    ev(2, 'call_start', '3f', {id: 'r', who: 'muse-image', what: 'repaint', view: 'body-90', try: 2}),
    ev(3, 'call_end', '3f', {id: 'r', who: 'muse-image', what: 'repaint', view: 'body-90', try: 2, outcome: 'cancelled', s: 1, usd: 0.01}),
    ev(4, 'decision', '3f', {what: 'view_result', view: 'body+0', try: 1, ok: false, why: 'outline_moved',
      m: {median_px: 4.67, median_limit: 6, p90_px: 18, p90_limit: 16}}),
    ev(4, 'decision', '3f', {what: 'view_result', view: 'body+90', try: 1, ok: true, m: {overlap: 0.93, median_px: 3.6}}),
    ev(4, 'decision', '3f', {what: 'view_kept', view: 'head+0', ok: false}),
    ev(5, 'local', '3f', {what: 'views_rendered', m: {cameras: 5}}),
    ev(6, 'version', '3f', {n: 1, passed: 3, of: 6}),
    ev(7, 'skip', '3f', {what: 'spend_cap'})];
  const model = buildModel(job({'3f': 'running'}), events);
  assert.deepEqual(rowsOf(model, '3f').map(r => [r.state, r.text]), [
    ['info', 'Wave 1: asking Muse Image to repaint 5 views, 2 tries each, all at once (10 requests)'],
    ['kept', 'Repainting the side view −90°, try 2: not needed, the other try passed (already paid for)'],
    ['kept', 'Front, try 1: not used (outline moved too far: median 4.7 px against 6, p90 18.0 px against 16)'],
    ['passed', 'Side view +90°, try 1: accepted (outline overlap 93 %, drift 3.6 px)'],
    ['kept', 'Face close-up: neither try passed. Keeping the original paint for this view.'],
    ['info', 'Rendered the 3D shape from 5 cameras'],
    ['passed', 'Your model now shows version 1 (3 of 6 views repainted).'],
    ['skipped', 'Texture refine reached its spending cap. Showing the best texture so far.']]);
  assert.equal(rowsOf(model, '3f')[5].who, 'herogen');
  assert.match(copyText(model), /i HeroGen \(local\): rendered the 3D shape from 5 cameras$/m);
  assert.equal(copy.regionName('hair-upper+180'), 'Upper hair, back');
  assert.equal(copy.regionName('body+45'), 'One view');
  assert.equal(copy.usd(0.101), '$0.101');
  assert.equal(copy.usd(0.02), '$0.02');
  assert.equal(copy.usd(0.0034), '$0.0034');
  assert.equal(copy.usd(0), '$0');
  assert.equal(copy.about(240), '4 min');
  assert.equal(copy.clock(467600), '07:47.6');
});

test('unknown codes, model messages and stray fields never reach the page or the copied log', () => {
  const raw = ['Traceback (most recent call last)', '/data/users/u_1/runs/j_1', 'sk-live-123', 'The image shows a real man',
    'weird_reason_code', 'mystery_call', 'unlisted_view'];
  const events = [ev(0, 'step_start', '1'),
    ev(1, 'call_start', '1', {id: 'q', who: 'not-a-model', what: 'mystery_call', prompt: raw[2]}),
    ev(2, 'call_end', '1', {id: 'q', who: 'not-a-model', what: 'mystery_call', outcome: 'failed', code: raw[4], reason: raw[3]}),
    ev(3, 'decision', '1', {what: 'view_result', view: raw[6], ok: false, why: raw[0]}),
    ev(4, 'skip', '1', {why: raw[1]}), ev(5, 'mystery_kind', '1', {msg: raw[0]}),
    ev(6, 'checks', '1', {n: 2, passed: 1, failed: [raw[4]], advisory: []}),
    ev(7, 'error', '1', {code: raw[4]}), ev(8, 'check', '1', {name: 'min_side_px', ok: false, gate: true, op: '>=', m: {value: 200, limit: 256}})];
  const model = buildModel(job({1: 'failed'}), events);
  const shown = JSON.stringify([model.lead, model.tail, model.steps.map(s => s.rows.map(r => r.text))]) + copyText(model);
  for (const text of raw) assert.ok(!shown.includes(text), text);
  assert.deepEqual(rowsOf(model, '1').map(r => r.text), ['Working on artwork failed: it stopped before finishing',
    'One view: not used (it did not pass the checks)', 'This part was skipped.', '1 of 2 checks passed (1 hard limit failed)',
    'This step stopped. Technical details are kept in the server logs.', 'Min side px: failed · measured 200, needs at least 256']);
  // A failed step's own plain message (already shown on the page) is preferred over the code.
  const failed = job({1: 'failed'}); failed.stages[2].message = 'The character is cut off.';
  assert.equal(rowsOf(buildModel(failed, [ev(0, 'error', '1', {code: 'rejected'})]), '1')[0].text, 'The character is cut off.');
  const skipped = job({4: 'skipped', 6: 'skipped'}); skipped.stages[7].message = 'Roblox rig was skipped because the game body was not usable.';
  const rows = rowsOf(buildModel(skipped, [ev(0, 'step_start', '6'), ev(1, 'step_end', '6', {outcome: 'skipped', why: 'earlier_step', s: 0, usd: 0})]), '6');
  assert.deepEqual(rows.map(r => [r.state, r.text]), [['skipped', 'Roblox rig was skipped because the game body was not usable.']]);
});

test('only ok and reused count as success; any other outcome is a loud failure in plain words', () => {
  const start = (id) => ev(0, 'call_start', '2', {id, who: 'muse-image', what: 'turnaround'});
  const end = (id, outcome, extra = {}) => ev(1, 'call_end', '2', {id, who: 'muse-image', what: 'turnaround', outcome, ...extra});
  const events = [ev(0, 'step_start', '2'), start('a'), start('b'), start('c'), start('d'), start('e'), start('f'),
    end('a', 'ApiError'), end('b', 'unknown', {code: 'timeout', usd: 0.01}), end('c', 'unreadable'),
    end('d', 'failed', {code: 'http_5xx'}), end('e', 'reused'), end('f', 'ok', {s: 16.3, usd: 0.01})];
  const model = buildModel(job({2: 'running'}), events);
  assert.deepEqual(rowsOf(model, '2').map(r => [r.state, r.text]), [
    ['failed', 'Drawing one turnaround sheet with front, right, back and left side by side failed: it stopped before finishing'],
    ['failed', 'Drawing one turnaround sheet with front, right, back and left side by side: outcome unknown (may be billed)'],
    ['failed', 'Drawing one turnaround sheet with front, right, back and left side by side failed: the answer could not be read'],
    ['failed', 'Drawing one turnaround sheet with front, right, back and left side by side failed: the model service returned an error'],
    ['passed', 'Reused a saved result'], ['passed', 'Turnaround sheet ready']]);
  assert.ok(!copyText(model).includes('ApiError'));
  assert.equal(copyText(model).split('\n').filter(l => l.includes('  ✗ ')).length, 4);
  assert.ok(model.steps.find(s => s.key === '2').attention, 'a failed call keeps its step open');
  assert.equal(copy.callState('cancelled'), 'kept');
});

test('times round before minutes and seconds are split', () => {
  assert.equal(copy.clock(59960), '01:00.0');
  assert.equal(copy.clock(179600), '02:59.6');
  assert.equal(copy.clock(179960), '03:00.0');
  assert.equal(copy.elapsed(179600), '3:00');
  assert.equal(copy.elapsed(59600), '1:00');
  assert.equal(copy.about(179.6), '3 min');
});

test('a capped log says so, wherever the cap was recorded', () => {
  const events = [ev(0, 'step_start', '1'), ev(1, 'truncated', null)];
  let model = buildModel(job({1: 'running'}), events);
  assert.ok(model.truncated);
  assert.equal(rowsOf(model, '1').at(-1).text, copy.jobLine({kind: 'truncated'}));
  model = buildModel(job({1: 'running'}), [ev(0, 'step_start', '1')], {truncated: true});
  assert.ok(model.truncated);
  assert.match(model.tail.rows[0].text, /size limit/);
});

test('the whole sample reads as the run: steps, totals and the idea', () => {
  const j = job({p: 'passed', 0: 'passed', 1: 'passed', 2: 'passed', 3: 'passed', '3f': 'passed', 4: 'running'});
  j.stages[0].seconds = 48.7; j.stages[0].cost_usd = 0.02;
  const model = buildModel(j, sampleEvents(START), {typical: TYPICAL, now: START + 470000});
  const lines = copyText(model).split('\n');
  assert.equal(lines[0], '+00:00.0  i Job started. Idea: "A fox samurai"');
  assert.equal(lines[1], '+00:00.0  ▸ Step 1 of 9 · Sketch from your description  (usually about 45 s)');
  assert.ok(lines.includes('+00:35.0  ✓ Muse Image: design drawing ready · 34.2 s · $0.01 (usually 25 s)'));
  assert.ok(lines.includes('+00:48.7  ✓ Sketch from your description done · step took 48.7 s · $0.02 · running total $0.02'));
  assert.ok(lines.includes('+05:04.1  ✓ SAM 3D Objects (GPU): returned a mesh · cold start 60.4 s + 96.1 s inference on an A100 · 172.9 s · $0.101 (usually 175 s)'));
  assert.ok(lines.includes('+02:31.2  i SAM 3D Objects (GPU): still waiting for the GPU after 20 s. If it was asleep the first call can take about a minute longer.'));
  assert.ok(lines.includes('+06:39.0  ✗ Muse Image: repainting the back, try 2: not needed, the other try passed (already paid for) · 15.7 s · $0.01'));
  assert.equal(stepMeta(model.steps[0]), '48.7 s · $0.02 est.');
  assert.equal(model.steps.find(s => s.key === '4').rows.length, 0);
  assert.ok(model.steps.find(s => s.key === '3f').defaultOpen, 'a step with a kept view stays open');
  assert.ok(!model.steps.find(s => s.key === '2').defaultOpen);
  assert.equal(model.lead.rows.length, 1);
  assert.equal(rowsOf(model, '3f').at(-1).text, 'Your website model is ready. Roblox files continue in the background.');
});

// ---------- The panel, with a small DOM and manual timers ----------
class El {
  constructor(tag) { this.tagName = tag; this.children = []; this.attrs = {}; this.events = {}; this.own = ''; this.className = '';
    this.hidden = false; this.open = false; this.checked = false; this.scrollTop = 0; this.clientHeight = 100; this.dataset = {};
    const self = this, names = () => self.className.split(' ').filter(Boolean);
    this.classList = {contains: c => names().includes(c), add: c => { if (!names().includes(c)) self.className = [...names(), c].join(' '); },
      remove: c => { self.className = names().filter(x => x !== c).join(' '); }}; }
  querySelectorAll(sel) { const tag = sel.split('[')[0]; return this.all().filter(c => c.tagName === tag); }
  get scrollHeight() { return 100 + 20 * this.count(); }
  count() { return this.children.reduce((n, c) => n + 1 + c.count(), 0); }
  set textContent(v) { this.own = String(v); this.children = []; }
  get textContent() { return this.own + this.children.map(c => c.textContent).join(''); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(k, f) { (this.events[k] ||= []).push(f); }
  fire(k) { for (const f of this.events[k] || []) f({}); }
  detach(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parent = null; }
  append(...kids) { for (const c of kids) { c.parent?.detach(c); c.parent = this; this.children.push(c); } }
  insertBefore(c, ref) { c.parent?.detach(c); c.parent = this; const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c); }
  removeChild(c) { this.detach(c); }
  get isConnected() { return this.root === true || !!this.parent?.isConnected; }
  contains(n) { for (let x = n; x; x = x.parent) if (x === this) return true; return false; }
  focus() { globalThis.document.activeElement = this; }
  replaceChildren(...kids) { for (const c of this.children) c.parent = null; this.children = []; this.own = ''; this.append(...kids); }
  all() { return this.children.flatMap(c => [c, ...c.all()]); }
  find(cls) { return this.all().filter(c => c.className.split(' ').includes(cls)); }
}
globalThis.document = {createElement: t => new El(t), createElementNS: (_, t) => new El(t), hidden: false};

function harness({responses, motion = false}) {
  const els = Object.fromEntries(['#stageLog', '#followLive', '#copyLog', '#logStatus', '#livePanel'].map(k => [k, new El(k)]));
  for (const e of Object.values(els)) e.root = true;
  document.activeElement = null;
  els['#followLive'].checked = true;
  const calls = [], pending = [];
  let clock = START;
  const api = async (path) => {
    calls.push(path);
    const next = responses.shift();
    if (typeof next === 'function') return next(path);
    if (next instanceof Error) throw next;
    return next ?? {events: [], cursor: null};
  };
  const timers = {setTimeout: (f, ms) => { const t = {f, ms}; pending.push(t); return t; },
    clearTimeout: (t) => { const i = pending.indexOf(t); if (i >= 0) pending.splice(i, 1); }};
  const log = buildLog((s) => els[s], api, {typical: TYPICAL, timers, now: () => clock, motion: {matches: motion}});
  const settle = () => new Promise(r => setImmediate(r));
  return {els, calls, pending, log, settle, advance(ms) { clock += ms; },
    async runNext(match = () => true) { const i = pending.findIndex(t => match(t)); const [t] = pending.splice(i, 1); t.f(); await settle(); return t; }};
}
const notFound = () => Object.assign(new Error('Not found.'), {status: 404});

test('rows change in place as calls finish, and the cursor carries on from where it was', async () => {
  const h = harness({responses: [
    {events: [ev(0, 'step_start', '2'), ev(0.5, 'call_start', '2', {id: 'v1', who: 'muse-image', what: 'turnaround'})], cursor: 'o:120'},
    {events: [ev(16.8, 'call_end', '2', {id: 'v1', who: 'muse-image', what: 'turnaround', outcome: 'ok', s: 16.3, usd: 0.01})], cursor: 'o:240', more: true},
    {events: [], cursor: 'o:240'}, {events: [], cursor: 'o:240'}]});
  h.log.update(job({2: 'running'}));
  await h.settle();
  assert.equal(h.calls[0], '/api/jobs/j_1/events');
  const [row] = h.els['#stageLog'].find('bl-call');
  assert.match(row.className, /st-running/);
  assert.match(row.textContent, /Drawing one turnaround sheet/);
  assert.equal(h.els['#logStatus'].textContent, 'Following live');
  const poll = await h.runNext(t => t.ms === POLL_MS);
  assert.equal(poll.ms, POLL_MS);
  assert.equal(h.calls[1], '/api/jobs/j_1/events?after=o%3A120');
  const [same] = h.els['#stageLog'].find('bl-call');
  assert.equal(same, row, 'the same row element is updated, not replaced');
  assert.match(row.className, /st-passed/);
  assert.match(row.textContent, /Turnaround sheet ready16\.3 s · \$0\.01/);
  // More lines were waiting: the next read goes at once. Then two empty replies in a row slow the loop down.
  assert.ok(h.pending.some(t => t.ms === 0));
  await h.runNext(t => t.ms === 0);
  const slow = await h.runNext(t => t.ms === POLL_MS);
  assert.ok(slow);
  assert.ok(h.pending.some(t => t.ms === IDLE_MS));
});

test('a finished job reads once and stops; a busy server slows the loop instead of failing', async () => {
  const h = harness({responses: [Object.assign(new Error('Too many requests.'), {status: 429}),
    {events: sampleEvents(START), cursor: 'o:9', done: true}]});
  h.log.update(job({p: 'passed'}, {state: 'succeeded'}));
  await h.settle();
  const retry = h.pending.find(t => t.f.name === 'poll');
  assert.equal(retry.ms, IDLE_MS);
  await h.runNext(t => t === retry);
  assert.equal(h.pending.filter(t => t.f.name === 'poll').length, 0, 'done: no further reads');
  assert.equal(h.els['#logStatus'].textContent, 'Recorded build log');
  // A collapsed step stays as the person left it after the next update.
  const step = h.els['#stageLog'].find('bl-step')[0], details = step.children[0];
  assert.equal(details.open, false);
  details.open = true; details.fire('toggle');
  h.log.update(job({p: 'passed'}, {state: 'succeeded'}));
  assert.equal(details.open, true);
});

test('polling keeps the row in focus and the steps the reader opened or closed', async () => {
  const first = sampleEvents(START, 130), rest = sampleEvents(START).slice(first.length);
  const h = harness({responses: [{events: first, cursor: 'o:1'}, {events: rest, cursor: 'o:2'}]});
  h.log.update(job({p: 'passed', 0: 'passed', 1: 'passed', 2: 'running'}));
  await h.settle();
  const steps = () => h.els['#stageLog'].find('bl-step');
  const [sketch, input, artwork] = steps();
  const artworkDetails = artwork.children[0], sketchDetails = sketch.children[0];
  assert.equal(artworkDetails.open, false, 'finished steps start closed');
  artworkDetails.open = true; artworkDetails.fire('toggle');           // the reader opens one step
  assert.equal(input.children[0].open, false);
  const summary = artworkDetails.children[0];
  summary.focus();
  const rowsBefore = artwork.find('bl-row');
  await h.runNext(t => t.f.name === 'poll');
  assert.equal(steps()[2], artwork, 'same step element');
  assert.equal(artwork.children[0], artworkDetails);
  assert.equal(artworkDetails.open, true, 'still open after the update');
  assert.equal(document.activeElement, summary, 'focus stays where it was');
  const rowsAfter = artwork.find('bl-row');
  assert.ok(rowsBefore.length > 0 && rowsAfter.length === rowsBefore.length && rowsAfter.every((r, i) => r === rowsBefore[i]),
    'unchanged rows are the same elements');
  assert.equal(sketchDetails.open, false);
  // Closing the running step by hand also sticks, although running steps open by default.
  const views = steps()[3].children[0];
  views.open = false; views.fire('toggle');
  h.log.update(job({p: 'passed', 0: 'passed', 1: 'passed', 2: 'running'}));
  assert.equal(views.open, false);
});

test('older runs without events keep the recorded stage times', async () => {
  const records = (path) => ({started: '2026-11-02T02:55:10Z', seconds: 2, path});
  const h = harness({responses: [notFound(), records]});
  h.log.update(job({p: 'passed'}, {state: 'failed'}));
  await h.settle(); await h.settle();
  assert.equal(h.calls[0], '/api/jobs/j_1/events');
  assert.equal(h.calls[1], '/runs/r1/j_1/stage-p.json');
  assert.equal(h.log.mode, 'fallback');
  assert.ok(h.els['#stageLog'].classList.contains('is-records'));
  assert.match(h.els['#stageLog'].textContent, /Step 1 of 9 · Sketch from your description/);
  assert.equal(h.els['#logStatus'].textContent, 'Recorded stage times');
  assert.equal(h.pending.filter(t => t.f.name === 'poll').length, 0);
  h.els['#copyLog'].fire('click');
  assert.match(h.log.text, /Step 1 of 9 · Sketch from your description · done/);
});

test('a run with no events file reads as finished and falls back too', async () => {
  const h = harness({responses: [{v: 1, events: [], cursor: 'o:0', more: false, done: true},
    () => ({started: '2026-11-02T02:55:10Z', seconds: 1})]});
  h.log.update(job({p: 'passed'}, {state: 'succeeded'}));
  await h.settle(); await h.settle();
  assert.equal(h.log.mode, 'fallback');
  assert.deepEqual(h.calls, ['/api/jobs/j_1/events', '/runs/r1/j_1/stage-p.json']);
  assert.equal(h.els['#stageLog'].find('log-step').length, 1);
});

test('copy log writes the two-line form; following stops when the reader scrolls up', async () => {
  let copied = null;
  Object.defineProperty(globalThis, 'navigator', {value: {clipboard: {writeText: async (t) => { copied = t; }}}, configurable: true});
  const h = harness({responses: [{events: sampleEvents(START, 60), cursor: 'o:1'}]});
  h.log.update(job({p: 'passed', 0: 'passed', 1: 'running'}));
  await h.settle();
  const list = h.els['#stageLog'];
  assert.equal(list.scrollTop, list.scrollHeight, 'follows the newest row');
  list.scrollTop = 0; list.fire('scroll');
  assert.equal(h.els['#followLive'].checked, false);
  h.els['#followLive'].checked = true; h.els['#followLive'].fire('change');
  assert.equal(list.scrollTop, list.scrollHeight);
  for (const f of h.els['#copyLog'].events.click) await f();
  assert.equal(h.els['#logStatus'].textContent, 'Log copied');
  const lines = copied.split('\n');
  assert.ok(lines.includes('+00:51.1  … SAM 3.1: looking for the character in the drawing…'));
  assert.ok(lines.includes('+00:56.5  ✓ SAM 3.1: looked for the character in the drawing · 5.4 s · $0.0025 (usually 4 s)'));
  assert.ok(lines.every(l => /^\+\d\d:\d\d\.\d  […✓✗!i▸] /.test(l)), lines.find(l => !/^\+\d\d:\d\d\.\d  […✓✗!i▸] /.test(l)));
});

test('reduced motion: no spinner is required and the live counter ticks every 5 s', async () => {
  const h = harness({responses: [{events: [ev(0, 'step_start', '1'), ev(0, 'call_start', '1', {id: 'a', who: 'sam-3.1', what: 'cover_art', target: 'feet'})], cursor: 'o:1'}],
    motion: true});
  h.log.update(job({1: 'running'}));
  await h.settle();
  const [row] = h.els['#stageLog'].find('bl-call');
  assert.equal(row.find('bl-dots')[0].textContent, '…');
  assert.ok(h.pending.some(t => t.ms === 5000));
  h.log.reset();
  assert.equal(h.pending.length, 0, 'leaving the hero stops every timer');
});

test('the page labels the log and keeps the follow and copy controls', () => {
  const html = readFileSync(new URL('../herogen/service/static/index.html', import.meta.url), 'utf8');
  assert.match(html, /<ol id="stageLog" class="build-log" role="log" aria-live="off" aria-label="[^"]+"/);
  assert.match(html, /id="liveTab"[^>]*>Build log</);
  assert.match(html, /id="followLive" type="checkbox" checked/);
  assert.match(html, /id="copyLog"/);
  const css = readFileSync(new URL('../herogen/service/static/styles/workspace-tree.css', import.meta.url), 'utf8');
  const block = css.slice(css.indexOf('/* Build log:'), css.indexOf('.side-panel .log-footer'));
  assert.ok(!/#[0-9a-fA-F]{3,8}\b|\d+ms\b|font-size:\s*\d+px/.test(block), 'tokens only');
  assert.match(block, /prefers-reduced-motion: reduce\)[^]*bl-dots[^]*display: inline/);
});
