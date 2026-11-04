// Reads the job page repeats while a hero builds. Each one gives up after POLL_TIMEOUT_MS, so a slow or lost reply (a
// hosted container that is still starting, a dropped connection) never stops the loop, and a failed read that is worth
// repeating says how long to wait first.
export const POLL_TIMEOUT_MS = 30000;

// read(signal) with a deadline: after ms the request is aborted and the read fails like a lost connection (no status).
export async function within(ms, read, timers = globalThis) {
  if (!ms) return read(undefined);
  const ctrl = new AbortController();
  let timer;
  const late = new Promise((_, reject) => {
    timer = timers.setTimeout(() => {
      ctrl.abort();
      reject(Object.assign(new Error('HeroGen took too long to answer.'), { status: 0, code: 'timeout' }));
    }, ms);
  });
  try { return await Promise.race([read(ctrl.signal), late]); } finally { timers.clearTimeout(timer); }
}

// No answer, a server error, or a busy or rate-limited reply: the same read will likely work in a moment.
export const retryable = (e) => !e?.status || e.status >= 500 || e.status === 408 || e.status === 429;

// The wait before the next read: what the server asked for (Retry-After, in seconds) if longer than base, at most max.
export function retryDelay(e, base, max = 60000) {
  const asked = typeof e?.retryAfter === 'string' && e.retryAfter.trim() ? Number(e.retryAfter) * 1000 : NaN;
  return Math.min(Number.isFinite(asked) && asked >= 0 ? Math.max(asked, base) : base, max);
}
