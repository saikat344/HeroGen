// The split 3D stage: SAM 3D's raw shape in one panel and HeroGen's model in the other, under one camera, with a
// divider the person drags (or moves with the arrow keys) to share the stage between them. preview.js draws the two
// panels; this module decides when the split applies and owns the divider.
export const SPLIT_MIN = 0.2, SPLIT_MAX = 0.8;
const STEP = 0.05, BIG_STEP = 0.1;
// Below this stage width the panels stack, SAM 3D on top, with a horizontal divider.
export const STACK_BELOW = 700;
const DOUBLE_TAP_MS = 320;

export const clampRatio = (ratio) => Number.isFinite(ratio) ? Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, ratio)) : 0.5;
// 'columns' (side by side) or 'rows' (stacked). A stage that is not laid out yet (width 0) keeps columns.
export const splitAxis = (width) => width > 0 && width < STACK_BELOW ? 'rows' : 'columns';
// Stage 3f is repainting the texture right now.
export function textureRefining(job) {
  if (job?.state !== 'running') return false;
  return (job.stages || []).some((s) => s?.key === '3f' && s.state === 'running');
}
// The split applies while the texture refines (unless the person picked one model on the switch), and on any hero
// with SAM 3D's raw shape when the person picks "Side by side".
export function splitWanted({ refining, raw, choice }) {
  if (!raw) return false;
  return choice === 'split' || (choice == null && Boolean(refining));
}
// The divider's keys: the arrows move it (Shift for bigger steps), Home and End send it to the limits.
// Columns: Left/Down shrink the SAM 3D panel. Rows: Up/Left shrink it (the divider moves up).
export function keyRatio(key, ratio, axis = 'columns', big = false) {
  const step = big ? BIG_STEP : STEP;
  const less = axis === 'rows' ? ['ArrowUp', 'ArrowLeft'] : ['ArrowLeft', 'ArrowDown'];
  const more = axis === 'rows' ? ['ArrowDown', 'ArrowRight'] : ['ArrowRight', 'ArrowUp'];
  if (less.includes(key)) return clampRatio(ratio - step);
  if (more.includes(key)) return clampRatio(ratio + step);
  if (key === 'Home') return SPLIT_MIN;
  if (key === 'End') return SPLIT_MAX;
  return null;
}
// The divider under a pointer at (x, y) on a stage whose box is `rect`. Stacked panels share the height above the
// dock (`inset` px at the bottom).
export function ratioAt(x, y, rect, axis = 'columns', inset = 0) {
  const rows = axis === 'rows', size = rows ? rect.height - inset : rect.width;
  if (!(size > 0)) return 0.5;
  return clampRatio(((rows ? y - rect.top : x - rect.left)) / size);
}

/** Wire the divider `handle` (a role="separator" element) on `stage`, whose canvas wrapper `wrap` is measured; the
 *  stacked panels stop above `dock`. `onChange()` runs when the ratio or the axis changes. Returns the live
 *  {ratio, axis, inset}; reading `inset` measures the dock again. */
export function splitDivider(handle, stage, wrap, dock, onChange = () => {}) {
  let ratio = 0.5, axis = 'columns', drag = null, lastTap = 0, inset = 0;
  function measureInset() {
    let next = 0;
    if (axis === 'rows' && dock && !dock.hidden) {
      const box = wrap.getBoundingClientRect(), top = dock.getBoundingClientRect().top;
      next = Math.max(0, Math.min(box.height * 0.6, Math.round(box.bottom - top + 8)));
    }
    if (next !== inset) { inset = next; stage.style.setProperty('--split-inset', `${inset}px`); }
    return inset;
  }
  function paint() {
    const pct = Math.round(ratio * 100);
    stage.style.setProperty('--split', `${(ratio * 100).toFixed(2)}%`);
    stage.style.setProperty('--split-r', ratio.toFixed(4));
    handle.setAttribute('aria-valuenow', String(pct));
    handle.setAttribute('aria-valuetext', `SAM 3D ${pct}%, HeroGen ${100 - pct}%`);
    handle.setAttribute('aria-orientation', axis === 'rows' ? 'horizontal' : 'vertical');
    stage.dataset.splitAxis = axis;
  }
  function set(next) {
    const value = clampRatio(next);
    if (value === ratio) return;
    ratio = value; paint(); onChange();
  }
  function measure() {
    const next = splitAxis(wrap.clientWidth);
    if (next === axis) return;
    axis = next; paint(); measureInset(); onChange();
  }
  handle.addEventListener('pointerdown', (event) => {
    if (event.button > 0) return;
    event.preventDefault();
    if (event.pointerType === 'touch') {
      const at = Date.now();
      if (at - lastTap < DOUBLE_TAP_MS) { lastTap = 0; set(0.5); return; }
      lastTap = at;
    }
    drag = event.pointerId; handle.setPointerCapture?.(event.pointerId); handle.classList.add('is-dragging');
  });
  handle.addEventListener('pointermove', (event) => {
    if (drag !== event.pointerId) return;
    set(ratioAt(event.clientX, event.clientY, wrap.getBoundingClientRect(), axis, inset));
  });
  const end = (event) => { if (drag === event.pointerId) { drag = null; handle.classList.remove('is-dragging'); } };
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) handle.addEventListener(type, end);
  handle.addEventListener('dblclick', () => set(0.5));
  handle.addEventListener('keydown', (event) => {
    const next = keyRatio(event.key, ratio, axis, event.shiftKey);
    if (next == null) return;
    event.preventDefault(); event.stopPropagation(); set(next);
  });
  if (globalThis.ResizeObserver) new ResizeObserver(measure).observe(wrap);
  paint(); measure();
  return { get ratio() { return ratio; }, get axis() { return axis; }, get inset() { return measureInset(); } };
}
