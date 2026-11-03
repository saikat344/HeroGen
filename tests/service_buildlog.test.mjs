import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildModel, copyText, orderEvents, rowMeta, stepMeta, typicalFor, buildLog, POLL_MS, IDLE_MS, MAX_MS, PAUSED }
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
  assert.equal(reused.text, 'Finding the body joints from the front view: reused a saved result');
  assert.equal(rowMeta(reused), '$0');
  assert.match(copyText(model), /SAM 3D Body \(GPU\): finding the body joints from the front view: reused a saved result · \$0$/m);
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
    ev(4, 'decision', '3f', {what: 'extra_try', view: 'body+0', try: 3}),
    ev(5, 'local', '3f', {what: 'views_rendered', m: {cameras: 5}}),
    ev(6, 'version', '3f', {n: 1, passed: 3, of: 6}),
    ev(7, 'skip', '3f', {what: 'texture_refine', why: 'spend_cap'})];
  const model = buildModel(job({'3f': 'running'}), events);
  assert.deepEqual(rowsOf(model, '3f').map(r => [r.state, r.text]), [
    ['info', 'Wave 1: asking Muse Image to repaint 5 views, 2 tries each, all at once (10 requests)'],
    ['kept', 'Repainting the side view −90°, try 2: cancelled (step stopped) (already paid for)'],
    ['kept', "Front, try 1: not used, the repaint drifted off the model's outline (90th percentile 18.0 px (limit 16))"],
    ['passed', "Side view +90°, try 1: accepted, the repaint matches the model's outline (outline overlap 93 %, median drift 3.6 px)"],
    ['kept', 'Face close-up: keeping the original paint; no try passed.'],
    ['info', 'Front: the earlier tries were not used, so Muse Image gets one more try (try 3).'],
    ['info', 'Rendered the 3D shape from 5 cameras'],
    ['passed', 'Your model now shows version 1 (3 of 6 views repainted).'],
    ['skipped', 'Texture refine reached its spending cap. Showing the best texture so far.']]);
  assert.equal(rowsOf(model, '3f')[6].who, 'herogen');
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
    'One view: not used, it did not pass its checks', 'This part was skipped.', '1 of 2 required checks failed',
    'This step stopped. Technical details are kept in the server logs.']);
  const sum = rowsOf(model, '1').find(r => r.checks);
  assert.equal(sum.state, 'failed');
  assert.deepEqual(sum.checks.map(c => [c.state, c.text]), [['failed', 'min_side_px: failed · measured 200, needs at least 256']]);
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
    ['passed', 'Drawing one turnaround sheet with front, right, back and left side by side: reused a saved result'], ['passed', 'Turnaround sheet ready']]);
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
  assert.equal(copy.usually(0.04), 'usually under 1 s');
  assert.equal(copy.usually(240), 'usually about 4 min');
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
  assert.equal(rowsOf(model, '3f').at(-1).text, 'Your 3D model is ready. Roblox files continue in the background.');
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
globalThis.document = {createElement: t => new El(t), createElementNS: (_, t) => new El(t), hidden: false, listeners: {},
  addEventListener(k, f) { (this.listeners[k] ||= []).push(f); }, fire(k) { for (const f of this.listeners[k] || []) f(); }};

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
    async runNext(match = () => true) { const i = pending.findIndex(t => match(t)); const [t] = pending.splice(i, 1); clock += t.ms || 0; t.f(); await settle(); return t; }};
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

test('the replayed fox-samurai run (3D shape to Roblox rig) renders end to end in plain words', () => {
  const lines = readFileSync(new URL('./buildlog_t08_events.jsonl', import.meta.url), 'utf8').trim().split('\n');
  const events = lines.map(l => JSON.parse(l)).map(e => ({...e, t: START + e.t}));
  const model = buildModel(job({p: 'passed', 0: 'passed', 1: 'passed', 2: 'passed', 3: 'passed', '3f': 'passed', 4: 'passed', 6: 'passed'}),
    events, {typical: TYPICAL});
  const shape = rowsOf(model, '3'), refine = rowsOf(model, '3f'), body = rowsOf(model, '4'), rig = rowsOf(model, '6');
  const texts = rows => rows.map(r => r.text);
  // One row per call id.
  const ids = new Set(events.filter(e => e.id).map(e => e.id));
  assert.equal([...shape, ...refine].filter(r => r.call).length, ids.size);
  // 3D shape: GPU results, facing search, and the 16 joints folded under the mapping summary.
  assert.ok(texts(shape).includes('Returned a mesh (8,223 vertices, 10,161 faces, 1,024 px texture) · cold start 60.4 s + 96.1 s inference on an A100'));
  assert.ok(texts(shape).some(t => t.startsWith('Faced the mesh forward (tested 4 rotations, turned 0°)')));
  const joints3 = shape.find(r => r.key === 'g:3:joints');
  assert.equal(joints3.text, '16 joints · 12 from SAM 3D Body, 4 from the outline; 7 differ from the outline estimate by more than 0.5 studs');
  assert.ok(joints3.checks.some(c => c.text === 'Left shoulder from the outline · 0.66 studs from the outline estimate (differs by more than 0.5 studs)'));
  // Texture refine: every try and view decision in words.
  for (const want of ["Face close-up, try 1: not used, the repaint drifted off the model's outline (median drift 11.0 px (limit 10), 90th percentile 36.1 px (limit 24))",
    'Side view +90°: using the repaint from try 1.', "Front: keeping the original paint; neither try was used (the repaint drifted off the model's outline).",
    'Side view −90°, try 2: not needed, the other try passed (stopped before it was paid for)',
    'Your model now shows version 1 (3 of 6 views repainted; 3 still being judged).']) assert.ok(texts(refine).includes(want), want);
  assert.ok(texts(refine).some(t => t.startsWith("Upper hair, back, try 1: accepted, the repaint matches the model's outline and hair (outline overlap 99 %, median drift 1.0 px, hair agreement 1.00)")));
  // Game body: long sub-steps finished in place, names in words, parts folded with their total, 49 checks folded.
  assert.ok(!body.some(r => r.state === 'running' || r.state === 'unknown'));
  for (const want of ['Crotch height clamped from 0.328 H to 0.380 H', 'Moved the neck cut −0.50 radius (−0.24 studs)',
    'Right elbow: no shared rim between its parts, so its seam cannot be measured (shown as unverified, not a failure).',
    'Split the high-detail model into 15 parts: 23,758 of 24,000 triangles (14,400 body, 5,158 caps, 1,016 shoulder covers), largest part 6,298 · 11.4 s'])
    assert.ok(texts(body).includes(want), want);
  const parts = body.find(r => r.key === 'g:4:parts');
  assert.equal(parts.text, '15 parts · 23,758 of 24,000 triangles (high), 5,998 of 6,000 (low); largest part 6,298 (limit 20,000)');
  assert.equal(parts.checks[1].text, 'Upper torso: 6,298 / 1,836 triangles (high / low)');
  const joints4 = body.find(r => r.key === 'g:4:joints');
  assert.equal(joints4.checks.length, 15);
  assert.ok(joints4.checks.some(c => c.text === 'Left shoulder joint at (0.94, 0.17, 3.61) from the outline · differs from the outline estimate by more than 0.5 studs'));
  const bodyChecks = body.find(r => r.checks && r.text.includes('body checks'));
  assert.equal(bodyChecks.text, '8 of 8 required body checks passed · 41 informational (23 not met in the fast lane)');
  assert.equal(bodyChecks.checks.length, 49);
  assert.equal(bodyChecks.state, 'passed');
  assert.equal(bodyChecks.checks.filter(c => c.state === 'muted').length, 23);
  // Roblox rig: 27 checks behind one line, each with its meaning, value and limit.
  const rigChecks = rig.find(r => r.checks);
  assert.equal(rigChecks.text, '26 of 26 required rig checks passed · 1 informational');
  assert.equal(rigChecks.checks.length, 27);
  assert.ok(rigChecks.checks.some(c => c.text === 'xml_well_formed (rig file is valid Roblox XML): passed · measured 1, needs exactly 1'));
  assert.ok(texts(rig).includes('Wrote HeroGen-R15.rbxmx: 15 MeshParts, 15 Motor6D joints, 19 accessory attachments, HipHeight 1.719 studs · 0.1 s'));
  // The copied log keeps one line per event, never a raw id.
  const text = copyText(model);
  assert.equal(text.split('\n').length, events.length);
  for (const raw of ['lower_try_accepted', 'outline_moved', 'joint_from', 'head+0', 'body-90', 'a100-40gb', 'part_left', 'split_high',
    'no_shared_rim', 'proportion_', 'undefined', 'NaN', 'null']) assert.ok(!text.includes(raw), raw);
});

test('a check that missed inside a step that passed is a quiet note, not a red failure', () => {
  const events = [ev(0, 'step_start', '3f'), ev(1, 'check', '3f', {name: 'outline_p90_px', ok: false, gate: true, op: '<=', m: {value: 18, limit: 16}})];
  const [row] = rowsOf(buildModel(job({'3f': 'passed'}), events), '3f');
  assert.equal(row.state, 'noted');
  assert.equal(row.text, '1 of 1 required checks failed');
  assert.deepEqual(row.checks.map(c => c.text), ['outline_p90_px: failed · measured 18, needs at most 16']);
  assert.equal(rowsOf(buildModel(job({'3f': 'failed'}), events), '3f')[0].state, 'failed');
});

test('a busy server or a lost connection pauses the log quietly and backs off, then it resumes at 1.5 s', async () => {
  const busy = (after) => Object.assign(new Error('Too many requests.'), {status: 429, retryAfter: after});
  const offline = () => Object.assign(new TypeError('Failed to fetch'), {});
  const h = harness({responses: [{events: [ev(0, 'step_start', '1')], cursor: 'o:1', done: false},
    busy('7'), busy(null), offline(), offline(), offline(), {events: [], cursor: 'o:1', done: false}]});
  h.log.update(job({1: 'running'}));
  await h.settle();
  const next = () => h.runNext(t => t.f.name === 'poll');
  const pollDelay = () => h.pending.find(t => t.f.name === 'poll')?.ms;
  assert.equal(pollDelay(), POLL_MS);
  await next();
  assert.equal(pollDelay(), 7000, 'Retry-After is honoured');
  assert.equal(h.els['#logStatus'].textContent, PAUSED);
  await next();
  assert.equal(pollDelay(), 14000, 'no Retry-After: twice as long');
  await next();
  assert.equal(pollDelay(), MAX_MS, 'never more than 15 s');
  await next(); await next();
  assert.equal(pollDelay(), MAX_MS);
  assert.equal(h.els['#logStatus'].textContent, PAUSED);
  assert.ok(h.els['#stageLog'].find('bl-step').length > 0, 'the rows already shown stay');
  await next();
  assert.equal(pollDelay(), POLL_MS, 'back to 1.5 s after a success');
  assert.equal(h.els['#logStatus'].textContent, 'Following live');
});

test('a long local sub-step runs as one row until it ends, and a check list keeps the reader\'s open state', async () => {
  const start = [ev(0, 'step_start', '4'), ev(1, 'local', '4', {what: 'body_start', m: {parts: 15, lods: 2}}),
    ev(2, 'local', '4', {what: 'decimate_start'})];
  let rows = rowsOf(buildModel(job({4: 'running'}), start, {now: START + 12000}), '4');
  assert.deepEqual(rows.map(r => [r.key, r.state, r.text]), [
    ['e:1', 'info', 'Turning the 3D shape into the 15 rigid parts Roblox R15 expects, in 2 levels of detail. No model is called here'],
    ['l:4:decimate', 'running', 'Reducing the mesh to fit the in-game budget…']]);
  assert.equal(rowMeta(rows[1]), '10 s');
  const end = [...start, ev(44, 'local', '4', {what: 'decimate', s: 42, m: {source_triangles: 217836}})];
  rows = rowsOf(buildModel(job({4: 'running'}), end), '4');
  assert.equal(rows[1].key, 'l:4:decimate');
  assert.equal(rows[1].text, 'Reduced the 217,836-triangle mesh to candidate sizes · 42.0 s');
  // The step ended without the finish: never left spinning.
  assert.equal(rowsOf(buildModel(job({4: 'failed'}), start), '4')[1].state, 'unknown');

  const checks = [ev(0, 'step_start', '6'), ev(1, 'check', '6', {name: 'xml_well_formed', ok: true, gate: true, op: '==', m: {value: 1, limit: 1}}),
    ev(1, 'check', '6', {name: 'meshes_bound', ok: false, gate: false, m: {value: 0}}),
    ev(2, 'checks', '6', {n: 2, passed: 1, gated: 1, failed: [], advisory: ['meshes_bound']})];
  const h = harness({responses: [{events: checks, cursor: 'o:1', done: false}, {events: [ev(3, 'step_end', '6', {outcome: 'passed', s: 0.1, usd: 0})], cursor: 'o:2', done: false}]});
  h.log.update(job({6: 'running'}));
  await h.settle();
  const [fold] = h.els['#stageLog'].find('bl-check-sum');
  assert.equal(fold.find('bl-text')[0].textContent, '1 of 1 required rig checks passed · 1 informational (1 not met in the fast lane)');
  assert.deepEqual(fold.find('bl-check').map(c => c.className), ['bl-check st-passed', 'bl-check st-muted']);
  const details = fold.children[0];
  assert.equal(details.open, false);
  details.open = true; details.fire('toggle');
  await h.runNext(t => t.f.name === 'poll');
  assert.equal(h.els['#stageLog'].find('bl-check-sum')[0], fold);
  assert.equal(details.open, true, 'still open after the next update');
});

// ---------- Review fixes ----------
test('a step end with an outcome this page does not know reads as unknown, never as done', () => {
  const events = [ev(0, 'step_start', '2'), ev(5, 'step_end', '2', {outcome: 'unknown', s: 5, usd: 0.01})];
  const model = buildModel(job({2: 'pending'}), events);
  const step = model.steps.find(s => s.key === '2');
  assert.equal(step.state, 'failed');
  assert.ok(step.defaultOpen);
  const last = copyText(model).split('\n').at(-1);
  assert.match(last, /✗ Character views ended, outcome unknown/);
  assert.ok(!copyText(model).includes('Character views done'));
});

test('a requeue or a later attempt starts the step over', () => {
  const first = [ev(0, 'step_start', '3', {a: 1}), ev(1, 'gpu_wait', '3', {a: 1, id: 'g1', who: 'sam-3d-objects', what: '3d_shape'}),
    ev(60, 'step_end', '3', {a: 1, outcome: 'failed', s: 60, usd: 0})];
  for (const restart of [[ev(61, 'requeue', '3', {a: 1, next: 2, of: 2, code: 'gpu_unavailable'})], []]) {
    const events = [...first, ...restart, ev(70, 'step_start', '3', {a: 2}),
      ev(71, 'gpu_wait', '3', {a: 2, id: 'g2', who: 'sam-3d-objects', what: '3d_shape'})];
    const model = buildModel(job({3: 'running'}), events, {now: START + 80000, typical: TYPICAL});
    const step = model.steps.find(s => s.key === '3');
    assert.equal(step.state, 'running');
    const rows = step.rows.filter(r => r.call);
    assert.deepEqual(rows.map(r => [r.key, r.state]), [['c:g1', 'unknown'], ['c:g2', 'running']]);
    assert.equal(rowMeta(rows[1]), '9 s · usually 175 s');
    // Its record still lagging at "pending" makes no difference.
    assert.equal(buildModel(job({3: 'pending'}), events).steps.find(s => s.key === '3').state, 'running');
  }
});

test('a cancelled try is "not needed" only when another try of its view was accepted', () => {
  const repaint = (id, view, n) => [ev(0, 'call_start', '3f', {id, who: 'muse-image', what: 'repaint', view, try: n}),
    ev(5, 'call_end', '3f', {id, who: 'muse-image', what: 'repaint', view, try: n, outcome: 'cancelled', s: 5, usd: 0})];
  const events = [ev(0, 'step_start', '3f'), ...repaint('a', 'body+0', 2), ...repaint('b', 'body+180', 2),
    ev(6, 'decision', '3f', {what: 'view_result', view: 'body+0', try: 1, ok: true, m: {overlap: 0.95}})];
  const model = buildModel(job({'3f': 'failed'}), events);
  const [a, b] = rowsOf(model, '3f').filter(r => r.call);
  assert.equal(a.text, 'Repainting the front, try 2: not needed, the other try passed');
  assert.equal(b.text, 'Repainting the back, try 2: cancelled (step stopped)');
  assert.match(copyText(model), /✗ Muse Image: repainting the back, try 2: cancelled \(step stopped\) · 5\.0 s · \$0$/m);
  assert.match(copyText(model), /Muse Image: repainting the front, try 2: not needed, the other try passed · 5\.0 s · \$0$/m);
});

test('coming back to the tab waits out the back-off, and Retry-After: 0 still waits 1.5 s', async () => {
  const busy = (after) => Object.assign(new Error('Too many requests.'), {status: 429, retryAfter: after});
  const h = harness({responses: [{events: [ev(0, 'step_start', '1')], cursor: 'o:1', done: false}, busy('10'), busy('0'),
    {events: [], cursor: 'o:1', done: false}]});
  h.log.update(job({1: 'running'}));
  await h.settle();
  await h.runNext(t => t.f.name === 'poll');
  assert.equal(h.calls.length, 2);
  h.advance(2000);
  document.fire('visibilitychange');
  await h.settle();
  assert.equal(h.calls.length, 2, 'no read before Retry-After has passed');
  const wait = h.pending.find(t => t.f.name === 'poll');
  assert.equal(wait.ms, 8000);
  await h.runNext(t => t === wait);
  assert.equal(h.calls.length, 3);
  assert.equal(h.pending.find(t => t.f.name === 'poll').ms, POLL_MS, 'Retry-After: 0 is floored to 1.5 s');
});

test('when the route keeps failing, an ended run shows its stage records and stops asking', async () => {
  const broken = () => Object.assign(new Error('Server error.'), {status: 500});
  const record = () => ({started: '2026-11-02T02:55:10Z', seconds: 1});
  const h = harness({responses: [broken(), broken(), broken(), record]});
  h.log.update(job({p: 'passed'}, {state: 'succeeded'}));
  await h.settle();
  await h.runNext(t => t.f.name === 'poll');
  await h.runNext(t => t.f.name === 'poll');
  await h.settle();
  assert.equal(h.log.mode, 'fallback');
  assert.ok(h.calls.includes('/runs/r1/j_1/stage-p.json'));
  assert.equal(h.pending.filter(t => t.f.name === 'poll').length, 0, 'no more reads for an ended run');
  assert.notEqual(h.els['#logStatus'].textContent, PAUSED);
  // A running job keeps trying, but shows the stage records meanwhile instead of "Loading".
  const g = harness({responses: [broken(), broken(), broken(), record]});
  g.log.update(job({p: 'running'}));
  await g.settle();
  await g.runNext(t => t.f.name === 'poll');
  await g.runNext(t => t.f.name === 'poll');
  await g.settle();
  assert.ok(g.els['#stageLog'].classList.contains('is-records'));
  assert.equal(g.pending.filter(t => t.f.name === 'poll').length, 1);
});

test('a missed required check in a step that passed is described as not met, not "not used"', async () => {
  const events = [ev(0, 'step_start', '3f'), ev(1, 'check', '3f', {name: 'outline_p90_px', ok: false, gate: true, op: '<=', m: {value: 18, limit: 16}})];
  const h = harness({responses: [{events, cursor: 'o:1', done: false}]});
  h.log.update(job({'3f': 'passed'}));
  await h.settle();
  const [fold] = h.els['#stageLog'].find('bl-check-sum');
  assert.match(fold.className, /st-noted/);
  assert.equal(fold.find('sr')[0].textContent, 'Not met; the step still passed');
});

test('row keys follow arrival, so a late event with an earlier time does not rename a row', () => {
  const events = [ev(0, 'step_start', '2'), ev(5, 'local', '2', {what: 'sheet_split', m: {figures: 4}}),
    ev(6, 'checks', '2', {n: 13, passed: 10, gated: 4, failed: [], advisory: ['a', 'b', 'c']})];
  const keys = (list) => rowsOf(buildModel(job({2: 'running'}), list), '2').map(r => r.key);
  const before = keys(events);
  const after = keys([...events, ev(1, 'local', '2', {what: 'sheet_split', m: {figures: 3}})]);
  assert.deepEqual(after.filter(k => before.includes(k)), before);
  assert.equal(after.length, before.length + 1);
  assert.equal(after[0], 'e:3', 'the late row sorts first by time but keeps its own key');
});

test('the face line says where the face paint comes from', () => {
  const line = (why, extra = {}) => copy.decisionLine({kind: 'decision', what: 'face_source', why, ...extra});
  assert.equal(line('beauty', {view: 'head+0', try: 3}), 'Face: using the beauty repaint (Face close-up, try 3).');
  assert.equal(line('default', {view: 'head+0', try: 1}), 'Face: using the ordinary repaint (Face close-up, try 1).');
  assert.equal(line('default_fast_fallback'), 'Face: no face repaint was accepted, so the face keeps the paint from the ordinary texture.');
  assert.equal(line('something_else'), null);
  assert.equal(copy.decisionLine({kind: 'decision', what: 'view_result', view: 'body+0', try: 1, ok: false, why: 'model_error'}),
    'Front, try 1: not used, the model call failed, so there was no repaint to check');
});

test('a view decision leads with a plain verdict, keeps its numbers and says which try the view uses', () => {
  const line = (e) => copy.decisionLine({kind: 'decision', view: 'body+0', ...e});
  const result = (why, m) => line({what: 'view_result', try: 2, ok: false, why, m});
  assert.equal(result('outline_moved', {overlap: 0.87, overlap_min: 0.9, median_px: 8, median_limit: 6, p90_px: 22.7, p90_limit: 16}),
    "Front, try 2: not used, the repaint drifted off the model's outline (outline overlap 87 % (needs 90 %), median drift 8.0 px (limit 6), 90th percentile 22.7 px (limit 16))");
  assert.equal(result('second_face', {overlap: 0.95, overlap_min: 0.9, face_outside: 0.31, face_outside_max: 0.25}),
    "Front, try 2: not used, the repainted face spills outside the model's face (face outside the model's face 31 % (at most 25 %))");
  assert.equal(result('hair_mismatch', {agreement: 0.62, agreement_min: 0.8}),
    "Front, try 2: not used, the repainted hair no longer matches the model's hair shape, size or colour (hair agreement 0.62 (needs 0.8))");
  assert.equal(result('outline_not_measurable'), 'Front, try 2: not used, the outline could not be measured, so the repaint could not be checked');
  assert.equal(result('missing_features'), "Front, try 2: not used, the face's eyes, eyebrows or mouth could not be confirmed in the repaint");
  assert.equal(result('face_unavailable'), 'Front, try 2: not used, no face was found to check the repaint against');
  assert.equal(result('spend_cap'), 'Front, try 2: not used, the spending cap was reached');
  assert.equal(line({what: 'view_result', try: 1, ok: true, m: {}}), 'Front, try 1: accepted, the repaint passed its checks');
  assert.equal(line({what: 'view_selected', try: 1, ok: true, m: {tries: 2}}), 'Front: using the repaint from try 1.');
  assert.equal(line({what: 'view_selected', try: 2, ok: true, m: {tries: 2}}), 'Front: using the repaint from try 2; try 1 was not used.');
  assert.equal(line({what: 'view_selected', try: 3, ok: true, m: {tries: 3}}), 'Front: using the repaint from try 3; tries 1 and 2 were not used.');
  assert.equal(line({what: 'view_kept', ok: false, why: 'hair_mismatch', m: {tries: 3}}),
    "Front: keeping the original paint; none of its 3 tries was used (the repainted hair no longer matches the model's hair shape, size or colour).");
  assert.equal(line({what: 'view_kept', ok: false, why: 'spend_cap'}), 'Front: keeping the original paint (the spending cap was reached).');
});

test('the starter place reads in plain words: its line in the rig step and its checks', () => {
  const events = [ev(0, 'step_start', '6'), ev(0.1, 'local', '6', {who: 'herogen', what: 'place_write'}),
    ev(0.2, 'check', '6', {who: 'herogen', name: 'place_marker', ok: true, gate: true, op: '==', m: {value: 1, limit: 1}})];
  const rows = rowsOf(buildModel(job({6: 'running'}), events), '6');
  assert.ok(rows.some(r => r.text === 'Wrote HeroGen-Starter.rbxlx: a starter place with a baseplate, a spawn and the rig'),
    JSON.stringify(rows.map(r => r.text)));
  assert.equal(copy.checkLine({name: 'place_feet_on_baseplate', ok: true, gate: true, op: '<=', m: {value: 0, limit: 0.001}}),
    'place_feet_on_baseplate (the feet stand on the baseplate): passed · measured 0, needs at most 0.001');
});

test('a reworded description is named in the Build log, quoting the rewrite from the job', () => {
  const reworded = 'An original armoured robot in red plating with a plain glowing chest light';
  const withText = job({p: 'passed'}, {kind: 'prompt', prompt: 'A transformer style robot, Ironman inspired',
    reword: {status: 'used', stage: 'p', text: reworded}});
  const events = [ev(1, 'step_start', 'p', {i: 1, of: 9}),
    ev(1.1, 'call_start', 'p', {id: 'a', who: 'muse-image', what: 'sketch_design'}),
    ev(3, 'call_end', 'p', {id: 'a', who: 'muse-image', what: 'sketch_design', outcome: 'failed', code: 'http_4xx', s: 1.9, usd: 0.01}),
    ev(3.1, 'call_start', 'p', {id: 'b', who: 'muse-spark', what: 'reword_brief'}),
    ev(5, 'call_end', 'p', {id: 'b', who: 'muse-spark', what: 'reword_brief', outcome: 'ok', s: 1.9, usd: 0.0012}),
    ev(5.1, 'decision', 'p', {what: 'brief_reworded', ok: true}),
    ev(5.2, 'call_start', 'p', {id: 'c', who: 'muse-image', what: 'sketch_design'}),
    ev(9, 'call_end', 'p', {id: 'c', who: 'muse-image', what: 'sketch_design', outcome: 'ok', s: 3.8, usd: 0.01})];
  // A reworded retry summarises its checks again; only the newest summary row stays.
  const cover = [ev(1, 'step_start', '1'), ev(2, 'checks', '1', {n: 5, passed: 5, gated: 2, failed: [], advisory: []}),
    ev(3, 'decision', '1', {what: 'brief_reworded', ok: true}), ev(4, 'checks', '1', {n: 5, passed: 5, gated: 2, failed: [], advisory: []})];
  const coverRows = rowsOf(buildModel(withText, cover, {now: START + 5000}), '1');
  assert.deepEqual(coverRows.map(r => r.key), ['e:2', 'e:3']);
  assert.equal(coverRows[1].summary.t, START + 4000);
  const rows = rowsOf(buildModel(withText, events, {now: START + 10000}), 'p');
  assert.deepEqual(rows.map(r => [r.state, r.text]), [
    ['failed', 'Drawing a character design from your idea failed: the model service refused the request'],
    ['passed', 'Reworded description ready'],
    ['passed', `The image model refused your description under its content policy (brand or character names usually cause this). Reworded it to “${reworded}” and trying this step once more.`],
    ['passed', 'Design drawing ready']]);
  assert.equal(copy.callStart({who: 'muse-spark', what: 'reword_brief'}),
    'rewording your description without brand or character names, held items or pets');
  // Without the job's text the line still says what happened, never a raw id.
  assert.equal(copy.decisionLine({what: 'brief_reworded', ok: true}),
    'The image model refused your description under its content policy (brand or character names usually cause this). Reworded it and trying this step once more.');
  assert.equal(copy.decisionLine({what: 'reworded_brief_refused', ok: false}), 'The image model refused the reworded description too. The run stops here.');
  assert.match(copy.decisionLine({what: 'brief_reword_failed', ok: false}), /could not be reworded automatically/);
  assert.equal(copy.decisionLine({what: 'sketch_redrawn', ok: true}),
    'Redrew the sketch from the reworded description, so the artwork no longer follows the original look.');
  assert.match(copy.decisionLine({what: 'sketch_redraw_failed', ok: false}), /could not be redrawn/);
});

test('the hero card says the description was reworded, and the edit box gets the rewrite', () => {
  const text = 'An original armoured robot in red plating with a plain glowing chest light';
  const used = {kind: 'prompt', prompt: 'A robot, Ironman inspired', reword: {status: 'used', stage: 'p', text}};
  assert.deepEqual(copy.rewordNotice(used), {tone: 'info', title: 'We reworded your description',
    text: `The image model refused your description under its content policy, which usually means it named a brand, franchise or trademarked character. We reworded it to: “${text}”`});
  const again = copy.rewordNotice({...used, reword: {status: 'refused_again', stage: 'p', text}});
  assert.equal(again.tone, 'warning');
  assert.equal(again.text, `We reworded it once to “${text}”, but the image model refused that as well. Edit it and try again.`);
  const failed = copy.rewordNotice({...used, reword: {status: 'rewrite_failed', stage: 'p', text: null}});
  assert.equal(failed.text, 'HeroGen could not reword it automatically. Brand, franchise or character names usually cause this. '
    + 'Edit the description to remove them and try again.');
  // A failed rewrite is never quoted or pre-filled, even if text arrives: the box gets the user's own description.
  assert.doesNotMatch(copy.rewordNotice({...used, reword: {status: 'rewrite_failed', stage: '1', text: 'A robot like Ironman'}}).text, /Ironman/);
  assert.equal(copy.descriptionDraft({...used, reword: {status: 'rewrite_failed', stage: '1', text: 'A robot like Ironman'}}),
    'A robot, Ironman inspired');
  assert.equal(copy.rewordNotice({kind: 'prompt', prompt: 'a knight', reword: null}), null);
  assert.equal(copy.rewordNotice({kind: 'sketch', reword: {status: 'used', text}}), null);
  assert.equal(copy.rewordNotice({...used, reword: {status: 'something_else', text}}), null);
  assert.equal(copy.descriptionDraft({...used, reword: {status: 'refused_again', text}}), text);
  assert.equal(copy.descriptionDraft({kind: 'prompt', prompt: 'a knight', reword: null}), 'a knight');
  assert.equal(copy.descriptionDraft({kind: 'sketch', prompt: ''}), '');
  // A long rewrite is cut on whole characters with an ellipsis.
  assert.equal(copy.quoted('ab😀cd', 3), '“ab😀…”');
});

test('the toolbar chip names the rewrite in a few words', () => {
  const text = 'An original armoured robot in red plating';
  const job = (reword) => ({kind: 'prompt', prompt: 'A robot, Ironman inspired', reword});
  assert.equal(copy.rewordLabel(job({status: 'used', stage: 'p', text, reason: 'precheck'})), 'Reworded to avoid a brand name');
  assert.equal(copy.rewordLabel(job({status: 'used', stage: 'p', text})), 'Reworded to pass the content policy');
  assert.equal(copy.rewordLabel(job({status: 'refused_again', stage: 'p', text})), 'Reworded description refused too');
  assert.equal(copy.rewordLabel(job({status: 'rewrite_failed', stage: 'p', text: null})), 'Description could not be reworded');
  assert.equal(copy.rewordLabel(job(null)), null);
  assert.equal(copy.rewordLabel({kind: 'sketch', reword: {status: 'used', text}}), null);
});

test('an up-front rewrite says the description named a brand, and its Build log rows are distinct', () => {
  const text = 'A street racer in a plain dark tracksuit with striped trainers, holding a helmet';
  const used = {kind: 'prompt', prompt: 'A street racer in a Nike tracksuit', reword: {status: 'used', stage: 'p', text, reason: 'precheck'}};
  assert.deepEqual(copy.rewordNotice(used), {tone: 'info', title: 'We reworded your description',
    text: `Your description named a brand or character, so we reworded it to: “${text}”`});
  const again = {...used, reword: {...used.reword, status: 'refused_again'}};
  assert.match(copy.rewordNotice(again).text, /Your own description is ready to edit/);
  // A refused up-front rewrite gives the box the user's own description back; a refusal-path rewrite keeps the rewrite.
  assert.equal(copy.descriptionDraft(again), 'A street racer in a Nike tracksuit');
  assert.equal(copy.descriptionDraft({...again, reword: {...again.reword, reason: 'content_policy'}}), text);
  assert.equal(copy.decisionLine({what: 'brief_prechecked', ok: true, n: 0}),
    'Checked your description for brand, logo, character or real-person names: none found.');
  assert.equal(copy.decisionLine({what: 'brief_prechecked', ok: true, n: 2}),
    'Checked your description: it names 2 brands or characters, so it is reworded before anything is drawn.');
  assert.equal(copy.decisionLine({what: 'brief_reworded_upfront', ok: true}, used),
    `Reworded your description up front, without brand or character names, to “${text}”. The sketch is drawn from it.`);
  assert.equal(copy.decisionLine({what: 'brief_precheck_failed', ok: false, code: 'http_5xx'}),
    'The brand and character check did not finish (the model service returned an error). Continuing with your description as written.');
  assert.match(copy.decisionLine({what: 'brief_reword_upfront_failed', ok: false}), /Continuing with it as written/);
  assert.equal(copy.callStart({who: 'muse-spark', what: 'precheck_brief'}),
    'checking your description for brand, logo, character or real-person names, held items and pets');
  const rows = rowsOf(buildModel(job({p: 'passed'}, used), [ev(1, 'step_start', 'p'),
    ev(2, 'decision', 'p', {what: 'brief_precheck_failed', ok: false, code: 'http_5xx'})], {now: START + 3000}), 'p');
  assert.equal(rows[0].state, 'kept', 'a failed check is a warning row, not a failure');
});
