// The Build log: the selected job's recorded events, grouped by step, one row per model call that changes in place
// from running to done. Events are read with a cursor while the job runs; a run without events (older runs) keeps
// the recorded stage times from the stage records instead.
import * as copy from './buildlog-copy.js';
import { recordLog } from './workbench.js';

export const POLL_MS = 1500, IDLE_MS = 3000, MAX_MS = 15000;
export const PAUSED = 'Log paused, retrying…';
const FALLBACK_AFTER = 3;
const GPU_WAIT_NOTE_MS = 20000;
const CALL_START = new Set(['call_start', 'gpu_wait']), CALL_END = new Set(['call_end', 'gpu_end']);
const CALL_NOTE = new Set(['call_wait', 'call_retry']);
const JOB_KINDS = new Set(['job_start', 'queue', 'requeue', 'cancel', 'model_ready', 'truncated', 'job_end']);
const RUNNING = new Set(['queued', 'running']);
const STATE_TEXT = {muted: 'Not met', noted: 'Not met; the step still passed', passed: 'Passed', running: 'Running', pending: 'Waiting', failed: 'Failed', rejected: 'Needs attention',
  skipped: 'Skipped', unavailable: 'Not available', canceled: 'Cancelled', info: 'Note', kept: 'Not used',
  unknown: 'No finish recorded'};
const ICON = {muted: 'info', noted: 'info', passed: 'circle-check', running: 'loader-circle', pending: 'circle-dashed', failed: 'circle-x',
  rejected: 'circle-x', skipped: 'info', unavailable: 'ban', canceled: 'x', info: 'info', kept: 'circle-dashed',
  unknown: 'circle-dashed'};
const GLYPH = {muted: 'i', noted: 'i', running: '…', passed: '✓', failed: '✗', rejected: '✗', kept: '✗', skipped: '!', info: 'i', unknown: 'i',
  canceled: '✗'};

// The typical duration of one call, only when enough past calls back it up. SAM 3.1 calls are told apart by what
// they look for, so the most specific key wins.
export function typicalFor(table, e) {
  const calls = table?.calls || {}, base = `${e.step}|${e.who}`;
  const keys = e.target ? [`${base}|${e.what}|${e.target}`, `${base}|${e.what}`, `${base}|*|${e.target}`] : [`${base}|${e.what}`];
  const row = keys.map(k => calls[k]).find(Boolean);
  return row && Number.isFinite(row.p50) && row.n >= Math.max(8, table.min_n || 0) ? row : null;
}

// File order is append order; `t` is the writer's clock. Sort by time and keep file order for ties.
export function orderEvents(events) {
  return orderWithArrival(events).map(x => x.e);
}
// Each event with its place in arrival (file) order, which never changes as more events arrive; row keys use it, so a
// late event with an earlier time cannot shift the key of a row the reader opened.
function orderWithArrival(events) {
  return events.map((e, i) => ({e, i, t: Number.isFinite(e?.t) ? e.t : Infinity}))
    .sort((a, b) => a.t - b.t || a.i - b.i);
}

const STEP_END = {unknown: ['✗', 'ended, outcome unknown'], passed: ['✓', 'done'], kept: ['✓', 'reused from the earlier attempt'], skipped: ['!', 'skipped'],
  rejected: ['✗', 'stopped'], failed: ['✗', 'stopped']};
const isTerminal = (job) => !!job && !RUNNING.has(job.state) && job.export_state !== 'running';
const stepTitle = (number, total, key) => `Step ${number} of ${total} · ${copy.stepName(key)}`;
const metaJoin = (...parts) => parts.filter(Boolean).join(' · ');

// Turn the job and its events into the rows the panel shows and the lines "Copy log" writes.
export function buildModel(job, events, {typical = null, now = Date.now(), truncated = false} = {}) {
  const stages = job?.stages || [];
  const origin = Number.isFinite(job?.started_at) ? job.started_at * 1000
    : events.find(e => Number.isFinite(e?.t))?.t ?? now;
  const steps = new Map(), lead = {key: 'lead', rows: []}, tail = {key: 'tail', rows: []};
  stages.forEach((s, i) => steps.set(String(s.key), {key: String(s.key), number: i + 1, total: stages.length,
    title: stepTitle(i + 1, stages.length, s.key), stage: s, state: s.state || 'pending', rows: [], usd: 0, calls: 0,
    started: null, ended: null, endS: null, endUsd: null, attention: false, checkList: [], checkRow: null, attempt: null}));
  const calls = new Map(), lines = [];
  let current = null, total = 0, cut = truncated;
  const stepFor = (e) => {
    const key = e.step == null ? null : String(e.step);
    if (key == null) return null;
    const title = Number.isInteger(e.i) && Number.isInteger(e.of) ? stepTitle(e.i, e.of, key) : copy.stepName(key);
    if (!steps.has(key)) steps.set(key, {key, number: e.i ?? null, total: e.of ?? stages.length, title, stage: null,
      state: 'pending', rows: [], usd: 0, calls: 0, started: null, ended: null, endS: null, endUsd: null, attention: false, checkList: [], checkRow: null, attempt: null});
    return steps.get(key);
  };
  function newAttempt(step, attempt) {
    step.attempt = attempt;
    step.endOutcome = null; step.ended = null; step.endS = null; step.endUsd = null; step.started = null;
    for (const row of step.rows) if ((row.call || row.timed) && row.state === 'running') row.stale = true;
  }
  const line = (t, glyph, text) => {
    if (!text) return null;
    const entry = {t, glyph, text: `+${copy.clock(t - origin)}  ${glyph} ${text}`}; lines.push(entry); return entry;
  };
  const accepted = new Map();   // view -> tries the pipeline accepted or kept
  ordered: for (const {e, i: index} of orderWithArrival(events)) {
    if (!e || typeof e.kind !== 'string') continue;
    const t = Number.isFinite(e.t) ? e.t : origin, step = stepFor(e);
    if (step) current = step;
    // A requeue, or the first event of a later attempt, starts the step over: the old attempt's end no longer holds,
    // and its unfinished calls will never finish.
    if (step && e.kind === 'requeue') newAttempt(step, Number.isInteger(e.next) ? e.next : (step.attempt ?? 1) + 1);
    else if (step && Number.isInteger(e.a)) {
      if (step.attempt != null && e.a > step.attempt) newAttempt(step, e.a);
      step.attempt ??= e.a;
    }
    const group = step || (e.kind === 'job_end' ? tail : current || lead);
    switch (true) {
      case e.kind === 'step_start': {
        step.started ??= t;
        const usual = step.stage?.typical_seconds;
        line(t, '▸', `${step.title}${Number.isFinite(usual) ? `  (${copy.usually(usual)})` : ''}`);
        break;
      }
      case e.kind === 'step_end': {
        // An outcome this page does not know never reads as done.
        step.endOutcome = ['passed', 'kept', 'skipped', 'rejected', 'failed'].includes(e.outcome) ? e.outcome : 'unknown';
        step.ended = t; step.endS = Number.isFinite(e.s) ? e.s : null; step.endUsd = Number.isFinite(e.usd) ? e.usd : null;
        // The step's own total also counts work that is not a single call; the running total follows it.
        if (step.endUsd != null) total += step.endUsd - step.usd;
        const cost = step.endUsd ?? step.usd, [glyph, said] = STEP_END[step.endOutcome];
        if (['failed', 'rejected', 'unknown'].includes(step.endOutcome)) step.attention = true;
        line(t, glyph, metaJoin(`${copy.stepName(step.key)} ${said}`,
          Number.isFinite(step.endS) && `step took ${copy.seconds(step.endS)}`, copy.usd(cost),
          `running total ${copy.usd(total)}`));
        // A step that stopped or was skipped explains itself with the plain message the page already shows.
        const message = ['skipped', 'failed', 'rejected'].includes(e.outcome) && step.stage?.message;
        if (message) {
          const state = e.outcome === 'skipped' ? 'skipped' : 'failed';
          step.rows.push({key: `e:${index}`, call: false, who: null, state, text: message, at: t});
          line(t, GLYPH[state], message);
        }
        break;
      }
      case CALL_START.has(e.kind) || CALL_END.has(e.kind): {
        const id = e.id == null ? `#${index}` : String(e.id);
        let row = calls.get(id);
        if (!row) {
          row = {key: `c:${id}`, call: true, who: e.who, what: e.what, target: e.target, step: e.step, view: e.view, try: e.try,
            state: 'running', started: CALL_START.has(e.kind) ? t : null, notes: [], typical: typicalFor(typical, e)};
          calls.set(id, row); group.rows.push(row);
        }
        const name = copy.whoName(row.who), full = {...e, what: e.what ?? row.what, target: e.target ?? row.target,
          view: e.view ?? row.view, try: e.try ?? row.try};
        if (e.kind === 'gpu_wait' && Number.isFinite(e.s)) {
          // A second wait on the same call: the GPU has not answered after `s` seconds.
          row.gpuWait = true;
          line(t, 'i', `${name}: still waiting for the GPU after ${copy.aboutCall(e.s)}. If it was asleep the first call can take about a minute longer.`);
        } else if (CALL_START.has(e.kind)) {
          row.started ??= t;
          row.text = `${copy.upper(copy.callStart(full))}…`;
          line(t, '…', `${name}: ${copy.callStart(full)}…`);
        } else {
          row.ended = t; row.outcome = typeof e.outcome === 'string' ? e.outcome : 'failed';
          row.state = copy.callState(row.outcome);
          row.s = row.outcome === 'reused' ? null : Number.isFinite(e.s) ? e.s : null;
          row.usd = row.outcome === 'reused' ? 0 : Number.isFinite(e.usd) ? e.usd : null;
          row.text = copy.upper(copy.callDone(full)); row.gpuWait = false;
          if (Number.isFinite(row.usd)) { (step || current || {usd: 0}).usd += row.usd; total += row.usd; }
          if (step) step.calls += 1;
          if (row.state === 'failed' && step) step.attention = true;
          const numbers = row.outcome === 'reused' ? '$0' : metaJoin(copy.seconds(row.s), copy.usd(row.usd));
          const usual = row.typical && row.outcome === 'ok' ? ` (usually ${copy.aboutCall(row.typical.p50)})` : '';
          row.full = full; row.name = name; row.numbers = numbers;
          row.line = line(t, GLYPH[row.state], `${metaJoin(`${name}: ${copy.callDone(full)}`, numbers)}${usual}`);
        }
        break;
      }
      case CALL_NOTE.has(e.kind): {
        const row = calls.get(String(e.id));
        const text = e.kind === 'call_retry'
          ? `Attempt ${Number.isFinite(e.attempt) ? e.attempt : 1}${Number.isFinite(e.of) ? ` of ${e.of}` : ''} failed (${
            copy.codeText(e.code)}); trying again${Number.isFinite(e.s) ? ` in ${copy.seconds(e.s)}` : ''}`
          : `Waited ${Number.isFinite(e.s) ? `${copy.seconds(e.s)} ` : ''}for a free slot`;
        if (row) row.notes.push(text);
        line(t, 'i', `${row ? `${copy.whoName(row.who)}: ` : ''}${text}`);
        break;
      }
      default: {
        let text = null, state = 'info', said = null;
        if (e.kind === 'local' && step && /^[a-z0-9_]+_start$/.test(String(e.what)) && copy.isLongStep(e.what.slice(0, -6))) {
          // A long local sub-step: a running row now, finished in place when `<name>` arrives.
          const name = e.what.slice(0, -6), key = `l:${step.key}:${name}`;
          if (calls.has(key)) continue ordered;
          const row = {key, call: false, timed: true, who: 'herogen', state: 'running', started: t, notes: [],
            text: `${copy.upper(copy.localRunning(name))}…`};
          calls.set(key, row); step.rows.push(row);
          line(t, '…', `HeroGen (local): ${copy.localRunning(name)}…`);
          continue ordered;
        }
        if (e.kind === 'local' && step && calls.get(`l:${step.key}:${e.what}`)?.state === 'running') {
          const row = calls.get(`l:${step.key}:${e.what}`), done = copy.localLine(e);
          Object.assign(row, {state: 'info', text: copy.upper(done), ended: t});
          line(t, 'i', `HeroGen (local): ${done}`);
          continue ordered;
        }
        if (e.kind === 'local') { said = copy.localLine(e); text = copy.upper(said); }
        else if (e.kind === 'decision') {
          text = copy.decisionLine(e, job); state = e.ok === false ? 'kept' : e.ok ? 'passed' : 'info';
          // Which tries of each view were accepted ("other" when a try was dropped because another one passed).
          const tries = e.view && (accepted.get(e.view) || accepted.set(e.view, new Set()).get(e.view));
          if (tries && ['view_result', 'view_selected'].includes(e.what) && e.ok === true) tries.add(e.try);
          if (tries && e.what === 'view_result' && e.why === 'lower_try_accepted') tries.add('other');
        }
        else if (e.kind === 'check') {
          // Each check joins its step's list, shown under the step's check summary.
          const said = copy.checkLine(e), cstate = e.ok ? 'passed' : e.gate === true ? 'failed' : 'muted';
          if (step) step.checkList.push({key: `k:${index}`, state: cstate, text: said, ok: e.ok === true, gate: e.gate === true});
          line(t, e.ok ? '✓' : e.gate === true ? '✗' : 'i', `HeroGen (local): ${said}`);
          continue ordered;
        } else if (e.kind === 'checks') {
          if (!step) continue ordered;
          const row = {key: `e:${index}`, call: false, who: 'herogen', state: 'passed', at: t, summary: e, checks: step.checkList};
          // A step run again (a reworded retry) summarises its checks again: the newest summary stands for the step.
          if (step.checkRow) step.rows.splice(step.rows.indexOf(step.checkRow), 1);
          step.checkRow = row; step.rows.push(row);
          line(t, 'i', `HeroGen (local): ${copy.checksLine(e, step.checkList, step.key)}`);
          continue ordered;
        } else if (e.kind === 'version') { text = copy.versionLine(e); state = 'passed'; }
        else if (e.kind === 'skip') { text = copy.skipLine(e); state = 'skipped'; }
        else if (e.kind === 'error') { text = copy.errorLine(e, step?.stage); state = 'failed'; }
        else if (e.kind === 'image') { text = copy.imageLine(e); state = 'passed'; }
        else if (e.kind === 'recover') { text = copy.recoverLine(e); }
        else if (JOB_KINDS.has(e.kind)) {
          text = copy.jobLine(e, job);
          if (e.kind === 'truncated') { cut = true; state = 'skipped'; }
          if (e.kind === 'requeue') state = 'skipped';
          if (e.kind === 'job_end') state = e.state === 'succeeded' ? 'passed' : e.state === 'canceled' ? 'canceled' : 'failed';
        }
        if (!text) continue ordered;
        if (step && (state === 'failed' || state === 'kept')) step.attention = true;
        const fold = e.kind === 'local' && step ? copy.foldOf(e) : null;
        group.rows.push({key: `e:${index}`, call: false, who: said ? 'herogen' : null, state, text, at: t, fold, m: e.m});
        line(t, GLYPH[state], said ? `HeroGen (local): ${said}` : text);
      }
    }
  }
  // The job record decides each step's state; a call whose step is over but whose finish was never written is
  // shown as unrecorded, never as still running.
  for (const step of steps.values()) {
    const s = step.stage;
    // The job record is read every few seconds; until it catches up, the step's own events say where it is.
    if (step.state === 'pending' || (step.state === 'running' && step.endOutcome)) {
      if (step.endOutcome) step.state = {kept: 'passed', passed: 'passed', unknown: 'failed'}[step.endOutcome] || step.endOutcome;
      else if (step.started != null) step.state = 'running';
    }
    step.seconds = Number.isFinite(s?.seconds) && step.state !== 'running' ? s.seconds : step.endS;
    step.cost = step.state !== 'running' && Number.isFinite(s?.cost_usd) ? s.cost_usd : step.endUsd ?? step.usd;
    step.running = step.state === 'running';
    step.elapsed = step.running && step.started != null ? now - step.started : null;
    step.typicalSeconds = Number.isFinite(s?.typical_seconds) ? s.typical_seconds : null;
    if (['failed', 'rejected'].includes(step.state)) step.attention = true;
    // A check that missed inside a step that still passed (one texture try, say) is a note, not a failure.
    // Parts and joints: one row each that opens into the list, so 15 of each never flood the step.
    for (const kind of ['parts', 'joints']) {
      const members = step.rows.filter(r => r.fold === kind || r.fold === `${kind}_summary`);
      const items = members.filter(r => r.fold === kind);
      if (items.length < 3) continue;
      const summary = members.find(r => r.fold === `${kind}_summary`);
      const row = {key: `g:${step.key}:${kind}`, call: false, who: 'herogen', state: 'info', summary: null,
        text: copy.foldTitle(kind, items.map(r => r.m || {}), summary?.text),
        checks: items.map(r => ({key: r.key, state: 'info', text: r.text}))};
      const at = step.rows.indexOf(items[0]);
      step.rows = step.rows.filter(r => !members.includes(r));
      step.rows.splice(Math.min(at, step.rows.length), 0, row);
    }
    // Checks: one summary row per step that opens into the list. A required check that failed is red unless the
    // step still passed (one texture try, say), where it is a quiet note.
    if (step.checkList.length && !step.checkRow) {
      step.checkRow = {key: `k:${step.key}`, call: false, who: 'herogen', state: 'passed', summary: null, checks: step.checkList};
      step.rows.push(step.checkRow);
    }
    if (step.checkRow) {
      const row = step.checkRow, gateFailed = step.checkList.some(c => c.state === 'failed') || copy.listOf(row.summary?.failed) > 0;
      row.text = copy.upper(copy.checksLine(row.summary, step.checkList, step.key) || 'Checks recorded');
      row.state = gateFailed ? (step.state === 'passed' ? 'noted' : 'failed') : 'passed';
      if (row.state === 'failed') step.attention = true;
    }
    for (const row of step.rows) if ((row.call || row.timed) && row.state === 'running' && (row.stale || (!step.running && step.state !== 'pending'))) row.state = 'unknown';
  }
  // A cancelled try was "not needed" only when another try of its view was accepted; otherwise the step stopped it.
  for (const row of calls.values()) if (row.outcome === 'cancelled' && row.full) {
    const tries = accepted.get(row.full.view);
    const sibling = !!tries && [...tries].some(x => x === 'other' || x !== row.full.try);
    const said = copy.callDone({...row.full, sibling});
    row.text = copy.upper(said);
    if (row.line) row.line.text = `+${copy.clock(row.ended - origin)}  ${row.line.glyph} ${metaJoin(`${row.name}: ${said}`, row.numbers)}`;
  }
  for (const row of calls.values()) if (row.state === 'running') {
    row.elapsed = row.started != null ? now - row.started : null;
    if (row.elapsed != null && row.typical && row.elapsed > 1.5 * row.typical.p90 * 1000) row.late = true;
    if (row.elapsed != null && copy.isGpu(row.who) && row.elapsed > GPU_WAIT_NOTE_MS) row.gpuWait = true;
  }
  if (truncated && !events.some(e => e?.kind === 'truncated')) tail.rows.push({key: 'e:cut', call: false, state: 'skipped',
    text: copy.jobLine({kind: 'truncated'}, job)});
  const order = [...steps.values()];
  const newest = [...order].reverse().find(s => s.rows.length || s.running);
  for (const s of order) s.defaultOpen = s.running || s.attention || s === newest;
  return {origin, lead, tail, steps: order, lines, truncated: cut, total,
    usual: [...calls.values()].some(r => r.typical) || order.some(s => s.typicalSeconds != null)};
}

export function copyText(model) {
  return model.lines.map(l => l.text).join('\n');
}

// The right-hand numbers of a row: time and cost when done, a live counter and the usual time while it runs.
export function rowMeta(row) {
  if (row.timed && row.state === 'running') return row.elapsed != null ? copy.elapsed(row.elapsed) : '';
  if (!row.call) return '';
  if (row.state === 'running') return metaJoin(row.elapsed != null && copy.elapsed(row.elapsed),
    row.typical && `usually ${copy.aboutCall(row.typical.p50)}`);
  if (row.outcome === 'reused') return '$0';
  if (row.state === 'unknown') return 'not recorded';
  return metaJoin(copy.seconds(row.s), copy.usd(row.usd));
}
export function rowNote(row) {
  if (row.state !== 'running') return row.notes?.at(-1) || '';
  if (row.gpuWait) return copy.GPU_STILL;
  if (row.late) return `Taking longer than usual (usually ${copy.aboutCall(row.typical.p50)})`;
  return row.notes?.at(-1) || '';
}
export function stepMeta(step) {
  const time = step.running ? (step.elapsed != null ? `running ${copy.elapsed(step.elapsed)}` : 'running')
    : Number.isFinite(step.seconds) ? copy.seconds(step.seconds) : '';
  const cost = Number.isFinite(step.cost) && (step.cost > 0 || step.calls || step.state === 'passed') ? `${copy.usd(step.cost)} est.` : '';
  const usual = step.typicalSeconds != null && (step.running || step.state === 'pending') ? copy.usually(step.typicalSeconds) : '';
  return metaJoin(time, cost, usual);
}

// ---------- DOM ----------
function node(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'), use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  svg.setAttribute('class', 'icon icon--s'); svg.setAttribute('aria-hidden', 'true');
  use.setAttribute('href', `/assets/icons/sprite.svg#i-${name}`); svg.append(use); return svg;
}
// The state marker: an icon (a spinner while running), a static "…" that replaces the spinner under reduced
// motion, and the state in words for screen readers.
function marker(state) {
  const box = node('span', 'bl-mark');
  box.append(icon(ICON[state] || 'info'));
  if (state === 'running') { const dots = node('span', 'bl-dots', '…'); dots.setAttribute('aria-hidden', 'true'); box.append(dots); }
  box.append(node('span', 'sr', STATE_TEXT[state] || ''));
  return box;
}
// Put `nodes` in `parent` in order, moving only what is out of place, so open rows and focus survive an update.
function sync(parent, nodes) {
  nodes.forEach((n, i) => { if (parent.children[i] !== n) parent.insertBefore(n, parent.children[i] || null); });
  while (parent.children.length > nodes.length) parent.removeChild(parent.children[parent.children.length - 1]);
}

export function buildLog($, api, {typical = null, timers = globalThis, now = () => Date.now(), motion = null} = {}) {
  let job = null, events = [], cursor = null, done = false, truncated = false, mode = 'waiting', request = 0;
  let timer = null, tick = null, empties = 0, delay = POLL_MS, offset = 0, model = null, busy = false;
  let serverClock = false, answered = false, paused = false, failures = 0, retryAt = 0;
  let text = '', announced = new Map();
  const rows = new Map(), sections = new Map(), userOpen = new Map();
  // Runs without events show their stage records the way the page always has.
  const records = recordLog($, api);
  let table = typical, tableLoad = null;
  const reduced = () => (motion ?? globalThis.matchMedia?.('(prefers-reduced-motion: reduce)'))?.matches === true;
  const list = () => $('#stageLog'), status = (s) => { $('#logStatus').textContent = s; };
  const serverNow = () => now() + offset;

  function loadTable() {
    if (table || tableLoad || typeof fetch !== 'function') return;
    tableLoad = fetch('/assets/typical-calls.json', {credentials: 'same-origin'}).then(r => r.ok ? r.json() : null)
      .then(t => { table = t; if (job && events.length) render(); }).catch(() => {});
  }

  function renderFallback() {
    const ol = list();
    if (!ol.classList?.contains('is-records')) {
      rows.clear(); sections.clear(); ol.replaceChildren(); ol.classList?.add('is-records');
      records.reset();
    }
    records.update(job);
  }

  // A check summary opens into the full list: name, value against limit, verdict.
  function checksElement(row) {
    let li = rows.get(row.key);
    if (!li) {
      li = node('li'); const details = node('details', 'bl-fold'), summary = node('summary', 'bl-fold-head'), list = node('ol', 'bl-checks');
      details.addEventListener('toggle', () => {
        if (details.open !== li.expect) { userOpen.set(row.key, details.open); li.expect = details.open; }
      });
      details.append(summary, list); li.append(details); li.parts = {details, summary, list}; rows.set(row.key, li);
    }
    const {details, summary, list} = li.parts;
    const sig = JSON.stringify([row.state, row.text, row.checks.length]);
    if (li.sig !== sig) {
      li.sig = sig;
      li.className = `bl-row bl-check-sum st-${row.state}`;
      const body = node('span', 'bl-body'), chev = icon('chevron-right');
      chev.setAttribute('class', 'icon icon--s bl-chev');
      body.append(node('span', 'bl-who', copy.whoName('herogen')), node('span', 'bl-text', row.text));
      summary.replaceChildren(marker(row.state), body, chev);
      list.replaceChildren(...row.checks.map(c => {
        const item = node('li', `bl-check st-${c.state}`); item.append(marker(c.state), node('span', 'bl-text', c.text)); return item;
      }));
    }
    const open = userOpen.has(row.key) ? userOpen.get(row.key) : row.state === 'failed';
    li.expect = open;
    if (details.open !== open) details.open = open;
    return li;
  }

  function rowElement(row) {
    if (row.checks) return checksElement(row);
    let li = rows.get(row.key);
    const meta = rowMeta(row), note = rowNote(row);
    const sig = JSON.stringify([row.state, row.text, row.who, meta, note]);
    if (li && li.sig === sig) return li;
    if (!li) { li = node('li'); rows.set(row.key, li); }
    li.sig = sig;
    li.className = `bl-row st-${row.state}${row.call ? ' bl-call' : ''}${row.late || row.gpuWait ? ' is-late' : ''}`;
    const body = node('span', 'bl-body');
    if (row.who) body.append(node('span', 'bl-who', copy.whoName(row.who)));
    body.append(node('span', 'bl-text', row.text));
    li.replaceChildren(marker(row.state), body);
    if (meta) li.append(node('span', 'bl-meta', meta));
    if (note) li.append(node('span', 'bl-note', note));
    return li;
  }

  function sectionElement(step) {
    let li = sections.get(step.key);
    if (!li) {
      li = node('li', 'bl-step');
      const details = node('details'), summary = node('summary', 'bl-head'), body = node('ol', 'bl-rows');
      // Any open or close the page did not make itself is the reader's choice, and later updates keep it.
      details.addEventListener('toggle', () => {
        if (details.open !== li.expect) { userOpen.set(step.key, details.open); li.expect = details.open; }
      });
      details.append(summary, body); li.append(details);
      li.parts = {details, summary, body}; sections.set(step.key, li);
    }
    const {details, summary, body} = li.parts, meta = stepMeta(step);
    const sig = JSON.stringify([step.state, step.title, meta]);
    li.className = `bl-step st-${step.state}${step.rows.length ? '' : ' is-empty'}`;
    if (li.sig !== sig) {
      li.sig = sig;
      const head = node('span', 'bl-body'), chev = icon('chevron-right');
      chev.setAttribute('class', 'icon icon--s bl-chev');
      head.append(node('span', 'bl-title', step.title));
      summary.replaceChildren(chev, marker(step.state), head);
      if (meta) summary.append(node('span', 'bl-meta', meta));
    }
    const open = userOpen.has(step.key) ? userOpen.get(step.key) : step.defaultOpen;
    li.expect = open;
    if (details.open !== open) details.open = open;
    sync(body, step.rows.map(rowElement));
    body.hidden = !step.rows.length;
    return li;
  }

  function groupElement(group) {
    let li = sections.get(group.key);
    if (!li) { li = node('li', 'bl-group'); const body = node('ol', 'bl-rows'); li.append(body); li.parts = {body}; sections.set(group.key, li); }
    sync(li.parts.body, group.rows.map(rowElement));
    return li;
  }

  function renderEvents() {
    const ol = list(), follow = $('#followLive').checked, position = ol.scrollTop;
    const focused = typeof document !== 'undefined' ? document.activeElement : null;
    model = buildModel(job, events, {typical: table, now: serverNow(), truncated});
    text = copyText(model);
    if (mode !== 'events') { records.reset(); ol.classList?.remove('is-records'); ol.replaceChildren(); mode = 'events'; }
    const parts = [];
    if (model.lead.rows.length) parts.push(groupElement(model.lead));
    for (const step of model.steps) parts.push(sectionElement(step));
    if (model.tail.rows.length) parts.push(groupElement(model.tail));
    sync(ol, parts);
    announce(model);
    // Rows are updated in place, so focus normally stays put; if a reorder moved the focused element, put it back.
    if (focused && focused !== document.activeElement && focused.isConnected && ol.contains(focused)) focused.focus({preventScroll: true});
    scroll(follow ? ol.scrollHeight : position);
    schedule('tick');
  }

  // Step changes and failures go to the polite status line; the log itself stays quiet.
  function announce(m) {
    const first = announced.size === 0;
    for (const s of m.steps) {
      const before = announced.get(s.key);
      if (before !== s.state && !first) {
        if (s.state === 'running') status(`${s.title} started`);
        else if (['failed', 'rejected'].includes(s.state)) status(`${copy.stepName(s.key)} stopped`);
        else if (s.state === 'passed' && before === 'running') status(`${copy.stepName(s.key)} done`);
      }
      announced.set(s.key, s.state);
    }
    if (first) status(done || isTerminal(job) ? 'Recorded build log' : 'Following live');
  }

  function scroll(top) {
    const ol = list();
    if (ol.scrollTop !== top) ol.scrollTop = top;
  }

  function render() {
    if (!job) return;
    if (events.length) renderEvents();
    else if (mode === 'fallback' || answered) renderFallback();
    else if (!list().children.length) list().replaceChildren(node('li', 'bl-wait', 'Loading the build log…'));
  }

  function schedule(what) {
    if (what === 'tick') {
      timers.clearTimeout(tick); tick = null;
      if (model && model.steps.some(s => s.running) && !isTerminal(job)) tick = timers.setTimeout(render, reduced() ? 5000 : 1000);
      return;
    }
    timers.clearTimeout(timer); timer = null;
    if (job && !done && !busy && mode !== 'fallback') timer = timers.setTimeout(poll, what);
  }

  const hidden = () => (typeof document !== 'undefined' && document.hidden) || $('#livePanel')?.hidden === true;

  async function poll() {
    timer = null;
    if (!job || busy) return;
    if (hidden()) { schedule(POLL_MS); return; }
    // Never earlier than a busy server asked, or than the back-off allows (also when the tab comes back).
    if (now() < retryAt) { schedule(retryAt - now()); return; }
    const id = job.id, token = request;
    busy = true;
    try {
      const query = cursor == null ? '' : `?after=${encodeURIComponent(cursor)}`;
      const res = await api(`/api/jobs/${encodeURIComponent(id)}/events${query}`);
      if (token !== request) return;
      const fresh = Array.isArray(res?.events) ? res.events.filter(e => e && typeof e === 'object') : [];
      if (Number.isFinite(res?.now)) { offset = res.now - now(); serverClock = true; }
      for (const e of fresh) events.push(e);
      if (res?.cursor != null) cursor = res.cursor;
      truncated ||= res?.truncated === true;
      empties = fresh.length ? 0 : empties + 1;
      delay = res?.more === true ? 0 : empties >= 2 ? IDLE_MS : POLL_MS;
      // The server says when the record is complete; only a reply without that answer falls back to the job state.
      done = typeof res?.done === 'boolean' ? res.done : !fresh.length && isTerminal(job);
      answered = true; failures = 0; retryAt = 0;
      if (paused) { paused = false; status(done || isTerminal(job) ? 'Recorded build log' : 'Following live'); }
      if (!events.length && done) mode = 'fallback';
    } catch (e) {
      if (token !== request) return;
      // No events route, or no events for this job: the stage records tell the story instead.
      if (e.status === 404 && !events.length) mode = 'fallback';
      else {
        // Busy server or no connection: wait (as long as the server asks, else twice as long, up to 15 s) and
        // say so quietly. The page never raises an error banner for the log.
        const asked = typeof e.retryAfter === 'string' && e.retryAfter.trim() ? Number(e.retryAfter) : NaN;
        delay = e.status === 429 && Number.isFinite(asked) && asked >= 0 ? Math.min(Math.max(asked * 1000, POLL_MS), 60000)
          : Math.min(Math.max(delay * 2, POLL_MS * 2), MAX_MS);
        retryAt = now() + delay; failures += 1;
        paused = true; status(PAUSED);
        // The route keeps failing and nothing has arrived: show the stage records, and stop asking for a run
        // that has already ended.
        if (failures >= FALLBACK_AFTER && !events.length) {
          answered = true;
          if (isTerminal(job)) { mode = 'fallback'; done = true; paused = false; }
        }
      }
    } finally { if (token === request) busy = false; }
    if (token !== request) return;
    render();
    schedule(delay);
  }

  function clear() {
    ++request; records.reset(); timers.clearTimeout(timer); timers.clearTimeout(tick); timer = tick = null;
    job = null; events = []; cursor = null; done = false; truncated = false; mode = 'waiting'; empties = 0;
    delay = POLL_MS; offset = 0; model = null; text = ''; busy = false;
    serverClock = false; answered = false; paused = false; failures = 0; retryAt = 0;
    rows.clear(); sections.clear(); userOpen.clear(); announced = new Map();
    const ol = list(); ol.classList?.remove('is-records'); ol.replaceChildren();
  }

  $('#followLive').addEventListener('change', () => { if ($('#followLive').checked) scroll(list().scrollHeight); });
  // Scrolling up to read stops following; ticking the box again jumps back to the newest row.
  list().addEventListener('scroll', () => {
    const ol = list();
    if (ol.scrollHeight - ol.scrollTop - ol.clientHeight > 8 && $('#followLive').checked) $('#followLive').checked = false;
  });
  $('#copyLog').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(events.length ? text : records.text); status('Log copied'); }
    catch { status('Copy unavailable. Select the log text to copy it.'); }
  });
  if (typeof document !== 'undefined') document.addEventListener?.('visibilitychange', () => {
    if (!document.hidden && job && !done && mode !== 'fallback') { timers.clearTimeout(timer); poll(); }
  });

  // Live counters run on the server's clock. Without a server time in the reply, the running step's elapsed time in
  // the job record and its recorded start give the difference.
  function estimateClock() {
    if (serverClock || !job) return;
    const s = (job.stages || []).find(x => x.state === 'running' && Number.isFinite(x.elapsed_seconds));
    const start = s && events.find(e => e?.kind === 'step_start' && String(e.step) === String(s.key) && Number.isFinite(e.t));
    if (start) offset = start.t + s.elapsed_seconds * 1000 - now();
  }

  return {
    update(next) {
      if (job?.id !== next.id) { clear(); job = next; loadTable(); render(); poll(); return; }
      const wasTerminal = isTerminal(job);
      job = next; estimateClock();
      // A job that just finished gets one more read, so the last events land.
      if (done && !wasTerminal && isTerminal(job)) done = false;
      if (!done && mode !== 'fallback' && !timer) schedule(0);
      render();
    },
    reset() { clear(); },
    get text() { return events.length ? text : records.text; },
    get mode() { return mode; },
  };
}
