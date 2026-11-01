import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ALLOWED_CREATE_RESULTS,
  detectDevice,
  flush,
  getVisitorId,
  isGpcEnabled,
  knownCreateResult,
  resetAnalyticsForTest,
  setCsrf,
  track,
  trackOnce,
} from '../herogen/service/static/analytics.js';
import { formatAge, formatDuration, formatUsd, tabFromHash } from '../herogen/service/static/admin.js';

test('Global Privacy Control is honoured strictly', () => {
  assert.equal(isGpcEnabled({ globalPrivacyControl: true }), true);
  assert.equal(isGpcEnabled({ globalPrivacyControl: false }), false);
  assert.equal(isGpcEnabled({}), false);
  assert.equal(isGpcEnabled(null), false);
});

test('visitor id is 32-hex in sessionStorage and falls back cleanly when blocked', () => {
  resetAnalyticsForTest();
  const map = new Map();
  const storage = {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, v),
  };
  const first = getVisitorId(storage);
  assert.match(first, /^[0-9a-f]{32}$/);
  assert.equal(getVisitorId(storage), first);

  map.set('hg-vid', 'not-valid');
  const replaced = getVisitorId(storage);
  assert.match(replaced, /^[0-9a-f]{32}$/);
  assert.notEqual(replaced, 'not-valid');

  const throwingStorage = {
    getItem() { throw new Error('blocked'); },
    setItem() { throw new Error('blocked'); },
  };
  const mem1 = getVisitorId(throwingStorage);
  const mem2 = getVisitorId(throwingStorage);
  assert.match(mem1, /^[0-9a-f]{32}$/);
  assert.equal(mem1, mem2);
});

test('coarse device detection reports only mobile or desktop', () => {
  assert.equal(detectDevice({ matchMedia: () => ({ matches: true }) }), 'mobile');
  assert.equal(detectDevice({ matchMedia: () => ({ matches: false }) }), 'desktop');
  assert.equal(detectDevice({ matchMedia: () => { throw new Error('bad'); } }), 'desktop');
  assert.equal(detectDevice(null), 'desktop');
});

test('knownCreateResult maps only allow-listed codes and never raw messages', () => {
  for (const code of ['accepted', 'generation_paused', 'generation_budget_exhausted', 'rate_limited', 'storage_full']) {
    assert.equal(knownCreateResult(code, 503), code);
  }
  assert.equal(knownCreateResult('unexpected raw message', 401), 'signed_out');
  assert.equal(knownCreateResult(null, 403), 'csrf');
  assert.equal(knownCreateResult(null, 409), 'conflict');
  assert.equal(knownCreateResult(null, 413), 'too_large');
  assert.equal(knownCreateResult(null, 415), 'media_type');
  assert.equal(knownCreateResult(null, 429), 'rate_limited');
  assert.equal(knownCreateResult(null, 503), 'busy');
  assert.equal(knownCreateResult(null, 502), 'internal');
  assert.equal(knownCreateResult(null, 0), 'network_error');
  assert.equal(knownCreateResult(null, 422), 'bad_request');
  assert.ok(ALLOWED_CREATE_RESULTS.has(knownCreateResult('anything', 418)));
});

test('client batching sends CSRF header, honours GPC, and swallows network errors', async () => {
  const prevWin = globalThis.window;
  const prevNav = globalThis.navigator;
  const prevFetch = globalThis.fetch;
  const calls = [];
  try {
    resetAnalyticsForTest();
    Object.defineProperty(globalThis, 'window', {
      value: { matchMedia: () => ({ matches: false }) },
      configurable: true,
    });
    Object.defineProperty(globalThis, 'navigator', {
      value: { globalPrivacyControl: false },
      configurable: true,
    });
    globalThis.fetch = async (url, opt) => {
      calls.push({ url, opt });
      return { ok: true, status: 200, json: async () => ({ ok: true }) };
    };

    setCsrf('csrf-tok-1');
    assert.equal(track('page_view', { page: 'landing' }), true);
    assert.equal(trackOnce('drag', 'viewer_drag', { surface: 'landing' }), true);
    assert.equal(trackOnce('drag', 'viewer_drag', { surface: 'landing' }), false);
    await flush();
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/events');
    assert.equal(calls[0].opt.headers['X-CSRF-Token'], 'csrf-tok-1');
    const body = JSON.parse(calls[0].opt.body);
    assert.equal(body.device, 'desktop');
    assert.deepEqual(body.events, [
      { name: 'page_view', props: { page: 'landing' } },
      { name: 'viewer_drag', props: { surface: 'landing' } },
    ]);

    // Network failure is swallowed cleanly
    globalThis.fetch = async () => { throw new Error('offline'); };
    track('create_clicked');
    await assert.doesNotReject(() => flush());

    // GPC suppresses tracking completely
    calls.length = 0;
    Object.defineProperty(globalThis, 'navigator', {
      value: { globalPrivacyControl: true },
      configurable: true,
    });
    assert.equal(track('page_view', { page: 'landing' }), false);
    await flush();
    assert.equal(calls.length, 0);
  } finally {
    resetAnalyticsForTest();
    if (prevWin === undefined) delete globalThis.window;
    else Object.defineProperty(globalThis, 'window', { value: prevWin, configurable: true });
    if (prevNav === undefined) delete globalThis.navigator;
    else Object.defineProperty(globalThis, 'navigator', { value: prevNav, configurable: true });
    globalThis.fetch = prevFetch;
  }
});

test('admin panel formatting and hash helpers', () => {
  assert.equal(tabFromHash('#analytics'), 'analytics');
  assert.equal(tabFromHash('#settings'), 'settings');
  assert.equal(tabFromHash(''), 'settings');
  assert.equal(tabFromHash('#unknown'), 'settings');
  assert.equal(formatUsd(null), '—');
  assert.equal(formatUsd(0), '$0.00');
  assert.equal(formatUsd(0.0025, { precise: true }), '$0.0025');
  assert.equal(formatDuration(null), '—');
  assert.equal(formatDuration(12.34), '12.3s');
  assert.equal(formatDuration(125), '2m 5s');
  assert.equal(formatAge(45), '45s');
  assert.equal(formatAge(130), '2m 10s');
});
