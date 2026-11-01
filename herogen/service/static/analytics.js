// First-party, privacy-preserving analytics client.
// Never sends or stores IPs, User-Agents, emails, prompts, character names, file names or image data.
// Honours Global Privacy Control (navigator.globalPrivacyControl) and swallows all network/storage errors.

const VISITOR_KEY = 'hg-vid';
const MAX_BATCH = 25;
const MAX_QUEUE = 100;
const FLUSH_DELAY_MS = 800;

export const ALLOWED_CREATE_RESULTS = new Set([
  'accepted',
  'generation_paused',
  'generation_budget_exhausted',
  'conflict',
  'busy',
  'upload_rejected',
  'upload_too_large',
  'no_file',
  'empty_prompt',
  'prompt_too_long',
  'bad_kind',
  'bad_profile',
  'bad_form',
  'multipart',
  'media_type',
  'too_large',
  'idempotency_key',
  'rate_limited',
  'signed_out',
  'signin_required',
  'public_access_closed',
  'csrf',
  'account',
  'storage_full',
  'bad_request',
  'internal',
  'network_error',
]);

let csrfToken = null;
let csrfPromise = null;
let memVisitorId = null;
let timer = null;
const queue = [];
const onceKeys = new Set();

export function isGpcEnabled(nav = typeof navigator !== 'undefined' ? navigator : null) {
  return Boolean(nav && nav.globalPrivacyControl === true);
}

function randomHex32() {
  const bytes = new Uint8Array(16);
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function getVisitorId(storage = typeof sessionStorage !== 'undefined' ? sessionStorage : null) {
  if (storage) {
    try {
      const existing = storage.getItem(VISITOR_KEY);
      if (typeof existing === 'string' && /^[0-9a-f]{32}$/.test(existing)) return existing;
      const fresh = randomHex32();
      storage.setItem(VISITOR_KEY, fresh);
      return fresh;
    } catch {
      // Fall back to tab-lifetime memory when sessionStorage is blocked.
    }
  }
  if (!memVisitorId) memVisitorId = randomHex32();
  return memVisitorId;
}

export function detectDevice(win = typeof window !== 'undefined' ? window : null) {
  try {
    if (win && typeof win.matchMedia === 'function') {
      if (win.matchMedia('(max-width: 767px), (pointer: coarse)').matches) return 'mobile';
    }
  } catch {
    // Default to desktop if matchMedia is unavailable.
  }
  return 'desktop';
}

export function knownCreateResult(code, status = 0) {
  if (typeof code === 'string' && ALLOWED_CREATE_RESULTS.has(code)) return code;
  if (status === 401) return 'signed_out';
  if (status === 403) return 'csrf';
  if (status === 409) return 'conflict';
  if (status === 413) return 'too_large';
  if (status === 415) return 'media_type';
  if (status === 429) return 'rate_limited';
  if (status === 503) return 'busy';
  if (status >= 500) return 'internal';
  if (!status) return 'network_error';
  return 'bad_request';
}

export function setCsrf(token) {
  if (isGpcEnabled()) return;
  if (typeof token === 'string' && token) {
    csrfToken = token;
    if (queue.length && !timer) scheduleFlush();
  }
}

async function ensureCsrf() {
  if (csrfToken) return csrfToken;
  if (typeof fetch !== 'function') return null;
  if (!csrfPromise) {
    csrfPromise = fetch('/api/session', { credentials: 'same-origin', cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && typeof d.csrf === 'string' && d.csrf) csrfToken = d.csrf;
        return csrfToken;
      })
      .catch(() => null)
      .finally(() => { csrfPromise = null; });
  }
  return csrfPromise;
}

export function primeAnalytics() {
  if (typeof window === 'undefined' || isGpcEnabled()) return;
  ensureCsrf().catch(() => {});
}

function scheduleFlush() {
  if (timer || typeof setTimeout !== 'function') return;
  timer = setTimeout(() => {
    timer = null;
    flush().catch(() => {});
  }, FLUSH_DELAY_MS);
}

function sendPayload(body, token) {
  if (typeof fetch !== 'function' || !token) return Promise.resolve(null);
  return fetch('/api/events', {
    method: 'POST',
    credentials: 'same-origin',
    keepalive: true,
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': token },
    body,
  }).catch(() => null);
}

export async function flush({ sync = false } = {}) {
  if (typeof window === 'undefined' || isGpcEnabled() || !queue.length) return;
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
  const batch = queue.splice(0, MAX_BATCH);
  let body;
  try {
    body = JSON.stringify({
      visitor_id: getVisitorId(),
      device: detectDevice(),
      events: batch,
    });
  } catch {
    return;
  }
  if (sync) {
    if (csrfToken) sendPayload(body, csrfToken);
    return;
  }
  const token = csrfToken || (await ensureCsrf());
  if (!token) return;
  const resp = await sendPayload(body, token);
  if (resp && resp.status === 403) {
    csrfToken = null;
    const refreshed = await ensureCsrf();
    if (refreshed) await sendPayload(body, refreshed);
  }
  if (queue.length) scheduleFlush();
}

export function track(name, props = null, options = {}) {
  if (typeof window === 'undefined' || isGpcEnabled()) return false;
  if (typeof name !== 'string' || !name) return false;
  const item = { name };
  if (props && typeof props === 'object') {
    const clean = {};
    for (const [k, v] of Object.entries(props)) {
      if (typeof v === 'string' && v) clean[k] = v;
    }
    if (Object.keys(clean).length) item.props = clean;
  }
  if (queue.length >= MAX_QUEUE) queue.shift();
  queue.push(item);
  if (options.flush || queue.length >= MAX_BATCH) {
    flush().catch(() => {});
  } else {
    scheduleFlush();
  }
  return true;
}

export function trackOnce(key, name, props = null, options = {}) {
  if (typeof window === 'undefined' || isGpcEnabled()) return false;
  if (onceKeys.has(key)) return false;
  onceKeys.add(key);
  return track(name, props, options);
}

export function bindOrbitDrag(canvas, surface) {
  if (!canvas || typeof window === 'undefined') return;
  let start = null;
  canvas.addEventListener('pointerdown', (ev) => {
    if (isGpcEnabled() || onceKeys.has('viewer_drag')) return;
    start = { x: ev.clientX, y: ev.clientY, id: ev.pointerId };
  }, { passive: true });
  canvas.addEventListener('pointermove', (ev) => {
    if (!start || ev.pointerId !== start.id) return;
    const dx = ev.clientX - start.x;
    const dy = ev.clientY - start.y;
    if (dx * dx + dy * dy >= 36) {
      start = null;
      trackOnce('viewer_drag', 'viewer_drag', surface ? { surface } : null);
    }
  }, { passive: true });
  const clear = () => { start = null; };
  canvas.addEventListener('pointerup', clear, { passive: true });
  canvas.addEventListener('pointercancel', clear, { passive: true });
}

export function resetAnalyticsForTest() {
  csrfToken = null;
  csrfPromise = null;
  memVisitorId = null;
  if (timer) { clearTimeout(timer); timer = null; }
  queue.length = 0;
  onceKeys.clear();
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('hg:theme-change', (ev) => {
    const theme = ev?.detail;
    if (theme === 'light' || theme === 'dark' || theme === 'system') {
      track('theme_change', { theme });
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush({ sync: Boolean(csrfToken) }).catch(() => {});
  });
  window.addEventListener('pagehide', () => {
    flush({ sync: Boolean(csrfToken) }).catch(() => {});
  });
}
