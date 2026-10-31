import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectionStages, developerMode } from '../herogen/service/static/developer.js';

test('service-only stages and unknown measurements remain visible without invented zeroes', () => {
  const stages = inspectionStages({ stages: [{ key: 'p', label: 'Sketch', state: 'running' },
    { key: '6', label: 'Export', state: 'pending' }] }, null);
  assert.deepEqual(stages.map((s) => s.key), ['p', '6']);
  assert.equal(stages[0].seconds, null);
  assert.equal(stages[0].recordStatus, 'No record supplied');
  assert.deepEqual(stages[0].models, []);
});

test('zero measurements, diagnostics, review, and unknown verdicts retain their meaning', () => {
  const [stage] = inspectionStages({ stages: [{ key: '3', state: 'running', seconds: 0, cost_usd: 0 }] },
    { stages: [{ stage: '3', status: 'pass', seconds: 10, cost_usd: 2, checks: [
      { name: 'gap', value: 0, threshold: 0.1, cmp: '<=', pass: false, gate: false },
      { name: 'view', review: true, pass: false }, { name: 'unknown' },
    ] }] });
  assert.equal(stage.state, 'running');
  assert.equal(stage.recordStatus, 'pass');
  assert.equal(stage.seconds, 0);
  assert.equal(stage.cost, 0);
  assert.deepEqual(stage.checks.map((c) => [c.verdict, c.kind]),
    [['Failed', 'Diagnostic'], ['Needs review', 'Gate'], ['Not recorded', 'Gate']]);
});

test('inspection-only stages and recorded provenance survive incomplete service data', () => {
  const stages = inspectionStages({ stages: [null] }, { stages: [null,
    { stage: '2h', name: 'Held out', models: ['sam-3.1', null], calls: { count: 2, replayed: 0 },
      code: { commit: 'abc', dirty: false } }] });
  assert.equal(stages.length, 1);
  assert.deepEqual(stages[0].models, ['sam-3.1']);
  assert.equal(stages[0].calls.replayed, 0);
  assert.equal(stages[0].code.dirty, false);
});

test('review-tier checks can pass; a review flag alone is not a failing verdict', () => {
  const [stage] = inspectionStages({ stages: [{ key: '2h' }] }, { stages: [{ stage: '2h', checks: [
    { name: 'accepted_yaw', pass: true, review: true, gate: false },
    { name: 'pose_drift', pass: false, review: true, gate: false },
    { name: 'unknown', review: true, gate: false },
  ] }] });
  assert.deepEqual(stage.checks.map((c) => c.verdict), ['Passed', 'Needs review', 'Not recorded']);
});

// Small DOM harness exercises requests and stale responses without a browser or model calls.
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.events = {}; this.value = ''; }
  set textContent(value) { this.value = value; this.children = []; }
  get textContent() { return this.value + this.children.map((c) => c.textContent).join(' '); }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(k, f) { this.events[k] = f; }
  append(...children) { for (const c of children) { c.parent = this; this.children.push(c); } }
  replaceChildren() { for (const c of this.children) c.parent = null; this.children = []; }
  contains(n) { return !!n && (n === this || this.children.some((c) => c.contains(n))); }
  get isConnected() { return this.root || !!this.parent?.isConnected; }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  querySelectorAll(selector) {
    return this.children.flatMap((c) => [
      ...((selector === 'h2' ? c.tag === 'h2' : c.attrs['data-dev-focus'] != null) ? [c] : []),
      ...c.querySelectorAll(selector),
    ]);
  }
  focus() { document.activeElement = this; }
}
function setup(getRun) {
  globalThis.document = { activeElement: null, createElement: (tag) => new Element(tag) };
  const button = new Element('button'), panel = new Element('section');
  panel.root = true;
  return { button, panel, mode: developerMode(button, panel, getRun) };
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
const job = (id) => ({ id, name: id, state: 'running', stages: [{ key: '3', label: 'Mesh', state: 'running' }] });

test('normal mode makes no inspection requests; toggle and Escape preserve accessibility state', async () => {
  const calls = [];
  const { button, panel, mode } = setup(async (id) => { calls.push(id); return { stages: [] }; });
  mode.update(job('one'));
  assert.deepEqual(calls, []);
  assert.equal(panel.hidden, true);
  button.events.click();
  await tick();
  assert.deepEqual(calls, ['one']);
  assert.equal(button.attrs['aria-pressed'], 'true');
  assert.equal(panel.hidden, false);
  assert.match(panel.textContent, /Not reported/);
  panel.events.keydown({ key: 'Escape', preventDefault() {} });
  assert.equal(panel.hidden, true);
  assert.equal(document.activeElement, button);
});

test('switching heroes rejects a delayed response from the earlier hero', async () => {
  const waiting = {};
  const { button, panel, mode } = setup((id) => new Promise((resolve) => { waiting[id] = resolve; }));
  mode.update(job('one')); button.events.click();
  mode.update(job('two'));
  waiting.two({ stages: [{ stage: '3', models: ['new-model'] }] });
  await tick();
  waiting.one({ stages: [{ stage: '3', models: ['old-model'] }] });
  await tick();
  assert.match(panel.textContent, /new-model/);
  assert.doesNotMatch(panel.textContent, /old-model/);
});

test('missing records and fetch failures are explicit and refreshable', async () => {
  let count = 0;
  const { button, panel, mode } = setup(async () => {
    if (++count === 1) throw Object.assign(new Error('missing'), { status: 404 });
    throw new Error('offline');
  });
  mode.update(job('one')); button.events.click(); await tick();
  assert.match(panel.textContent, /not available yet/);
  panel.querySelectorAll('[data-dev-focus]').find((n) => n.attrs['data-dev-focus'] === 'refresh').events.click();
  await tick();
  assert.match(panel.textContent, /offline/);
  assert.match(panel.textContent, /retry/);
});

test('closing or clearing the selection discards an in-flight inspection response', async () => {
  let finish;
  const { button, panel, mode } = setup(() => new Promise((resolve) => { finish = resolve; }));
  mode.update(job('one')); button.events.click();
  button.events.click();
  finish({ stages: [{ stage: '3', models: ['late-result'] }] }); await tick();
  assert.equal(panel.hidden, true);
  mode.update(null);
  button.events.click();
  assert.match(panel.textContent, /Select a hero/);
  assert.doesNotMatch(panel.textContent, /late-result/);
});

test('recorded text is rendered literally and focus survives refreshed data', async () => {
  const payload = '<img src=x onerror=alert(1)>';
  const { button, panel, mode } = setup(async () => ({ stages: [{ stage: '3', models: [payload] }] }));
  mode.update(job('one')); button.events.click(); await tick();
  assert.match(panel.textContent, /<img src=x onerror=alert\(1\)>/);
  const summary = panel.querySelectorAll('[data-dev-focus]').find((n) => n.attrs['data-dev-focus'] === 'stage:3');
  summary.focus();
  mode.update(job('one')); await tick();
  assert.equal(document.activeElement.attrs['data-dev-focus'], 'stage:3');
  assert.equal(panel.children.some((c) => c.tag === 'img'), false);
});
