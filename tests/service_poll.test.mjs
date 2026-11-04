import test from 'node:test';
import assert from 'node:assert/strict';
import { POLL_TIMEOUT_MS, retryable, retryDelay, within } from '../herogen/service/static/poll.js';

function fakeTimers() {
  const pending = [];
  return { pending, setTimeout: (f, ms) => { const t = { f, ms }; pending.push(t); return t; },
    clearTimeout: (t) => { const i = pending.indexOf(t); if (i >= 0) pending.splice(i, 1); } };
}

test('a read that never answers is aborted at its deadline and fails like a lost connection', async () => {
  const timers = fakeTimers();
  let signal;
  const read = within(POLL_TIMEOUT_MS, (s) => { signal = s; return new Promise(() => {}); }, timers);
  assert.equal(timers.pending[0].ms, POLL_TIMEOUT_MS);
  timers.pending[0].f();
  await assert.rejects(read, (e) => e.status === 0 && e.code === 'timeout' && retryable(e));
  assert.equal(signal.aborted, true, 'the request itself is cancelled');
});

test('a read that answers in time keeps its answer and clears its deadline', async () => {
  const timers = fakeTimers();
  assert.deepEqual(await within(1000, async () => ({ ok: true }), timers), { ok: true });
  assert.equal(timers.pending.length, 0);
  assert.equal(await within(0, async (s) => s), undefined, 'no deadline: no signal');
});

test('busy, rate-limited, server errors and no answer are retried; a missing hero or a refusal is not', () => {
  for (const status of [undefined, 0, 408, 429, 500, 502, 503, 504]) assert.equal(retryable({ status }), true, String(status));
  for (const status of [400, 401, 403, 404, 409]) assert.equal(retryable({ status }), false, String(status));
});

test('the next read waits as long as the server asked, never less than base nor more than max', () => {
  assert.equal(retryDelay({ status: 429, retryAfter: '12' }, 3000), 12000);
  assert.equal(retryDelay({ status: 503, retryAfter: '1' }, 3000), 3000);
  assert.equal(retryDelay({ status: 429, retryAfter: '600' }, 3000), 60000);
  assert.equal(retryDelay({ status: 0 }, 3000), 3000);
  assert.equal(retryDelay({ status: 429, retryAfter: 'soon' }, 3000), 3000);
});
