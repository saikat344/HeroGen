// DOM helpers for the lab page. Everything from a run folder goes on the page as text, never as HTML; icons are
// rebuilt element by element from the vendored SVGs through an allow-list.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// h('div', {class: 'x', text: 'y', style: {width: '10px'}, onclick: fn}, ...children)
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'style') Object.assign(el.style, v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function fill(target, ...kids) {
  const el = typeof target === 'string' ? $(target) : target;
  el.replaceChildren();
  append(el, kids);
  return el;
}

// ---------------------------------------------------------------------------------------------------------------
// Icons and illustrations

const SVG_NS = 'http://www.w3.org/2000/svg';
const SVG_TAGS = new Set(['svg', 'g', 'path', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'rect']);
const SVG_ATTRS = new Set(['viewBox', 'fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin',
  'stroke-dasharray', 'd', 'cx', 'cy', 'r', 'rx', 'ry', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 'points', 'transform',
  'opacity', 'vector-effect', 'width', 'height']);
const templates = new Map();

function rebuild(node, root) {
  const tag = node.localName;
  if (!SVG_TAGS.has(tag)) return null;
  const el = document.createElementNS(SVG_NS, tag);
  for (const a of node.attributes) {
    if (!SVG_ATTRS.has(a.name)) continue;
    if (root && (a.name === 'width' || a.name === 'height')) continue;
    el.setAttribute(a.name, a.value);
  }
  for (const child of node.children) {
    const c = rebuild(child, false);
    if (c) el.append(c);
  }
  return el;
}

async function loadSvg(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const doc = new DOMParser().parseFromString(await r.text(), 'image/svg+xml');
  const svg = doc.documentElement;
  if (svg.localName !== 'svg') throw new Error(`${url} is not an SVG`);
  return rebuild(svg, true);
}

export async function loadIcons(names, illustrations, wordmark) {
  const jobs = names.map((n) => loadSvg(`static/icons/${n}.svg`).then((t) => templates.set(n, t)));
  jobs.push(...illustrations.map((n) => loadSvg(`static/illustrations/${n}.svg`).then((t) => templates.set(n, t))));
  if (wordmark) jobs.push(loadSvg(`static/brand/${wordmark}.svg`).then((t) => templates.set('wordmark', t)));
  const results = await Promise.allSettled(jobs);
  return results.filter((r) => r.status === 'rejected').map((r) => String(r.reason));
}

// An inline icon: decorative (the label is on the control or in the text next to it).
export function icon(name, cls = 'ic') {
  const t = templates.get(name);
  if (!t) return h('span', { class: cls, 'aria-hidden': 'true' });
  const el = t.cloneNode(true);
  el.setAttribute('class', cls);
  el.setAttribute('aria-hidden', 'true');
  el.setAttribute('focusable', 'false');
  return el;
}

// ---------------------------------------------------------------------------------------------------------------
// Numbers and words

export const fmt = (n, d = 0) => (n === null || n === undefined || Number.isNaN(n) ? '–'
  : Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }));
export function fmtAuto(n) {
  if (n === null || n === undefined) return '–';
  if (typeof n === 'boolean') return String(n);
  if (typeof n !== 'number') return String(n);
  if (Math.abs(n) >= 1000) return fmt(n);
  return Number.isInteger(n) ? String(n) : String(+n.toPrecision(4));
}
export const mb = (b) => (b === null || b === undefined ? '–' : b < 1e6 ? `${Math.max(1, Math.round(b / 1e3))} KB` : `${(b / 1e6).toFixed(2)} MB`);
export const secs = (s) => (s === null || s === undefined ? '–' : s < 10 ? `${s.toFixed(2)} s` : s < 600 ? `${s.toFixed(1)} s` : `${(s / 60).toFixed(1)} min`);
export const usd = (v) => (v === null || v === undefined ? '–' : `$${v.toFixed(4)}`);
export const yaw = (y) => (y === null || y === undefined ? '–' : `${y > 0 ? '+' : y < 0 ? '−' : ''}${Math.abs(y)}°`);
export const spaced = (s) => s.replace(/([a-z])([A-Z])/g, '$1 $2');

// "2026-10-28T21:58:35-07:00" -> "28 Oct 2026, 21:58 (UTC−07)": the recorded wall clock and offset, unconverted.
export function when(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}).*?(Z|[+-]\d{2}):?(\d{2})?$/.exec(iso || '');
  if (!m) return iso || '–';
  const mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][+m[2] - 1];
  const zone = m[6] === 'Z' ? 'UTC' : `UTC${m[6].replace('-', '−')}`;
  return `${+m[3]} ${mon} ${m[1]}, ${m[4]}:${m[5]} (${zone})`;
}

// A status chip: icon + word + colour, never colour alone.
const CHIP_ICON = { ok: 'circle-check', warn: 'triangle-alert', bad: 'circle-x', idle: 'circle-dashed' };
const SHORT = { 'Needs attention': 'Check', 'Not run': 'Not run', Pass: 'Pass', "Can't read": 'Error' };
export function chip(kind, word, { big = false, ic = null, title = null } = {}) {
  return h('span', { class: `st ${kind}${big ? ' big' : ''}`, title },
    icon(ic || CHIP_ICON[kind]), h('span', { class: 'w', text: word }),
    SHORT[word] ? h('span', { class: 'ws', hidden: true, text: SHORT[word] }) : null);
}

export const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage blocked: the choice lasts this visit */ } },
};

export async function getJson(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// /runs/<root>/<run>/<path> and /versions/<group>/<version>/<file>, each segment encoded.
const enc = (p) => p.split('/').map(encodeURIComponent).join('/');
export const runFile = (runId, path) => `/runs/${enc(runId)}/${enc(path)}`;
export const versionFile = (versionId, name) => `/versions/${enc(versionId)}/${enc(name)}`;

export const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
export const cssToken = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
