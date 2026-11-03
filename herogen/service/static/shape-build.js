// While the 3D shape is built: a figure of dots sampled from the hero's own outline (the stage-2 masks), the
// progress of the step and a tip about what is happening. preview.js draws the dots; this module has no 3D code.

// What the step is doing, in order. Each line is true of the real pipeline (stage 3, then texture refine).
export const SHAPE_TIPS = [
  'SAM 3D Objects is reconstructing the shape on a GPU',
  'SAM 3D Body is finding the joints for the rig',
  'Next, HeroGen lines the shape up with your character views',
  'Then HeroGen repaints the texture from every side',
  'Drag to look around while it builds',
];
export const TIP_EVERY_MS = 6000;
export const shapeTip = (i) => SHAPE_TIPS[((Math.floor(i) % SHAPE_TIPS.length) + SHAPE_TIPS.length) % SHAPE_TIPS.length];

// The step's progress from its elapsed and typical seconds, 0 to 0.95: never full until the shape is there. Linear
// to 80% of the typical time, then it slows towards 95%; without a typical time it eases towards 95% over minutes.
export function buildProgress(elapsed, typical) {
  const e = Number.isFinite(elapsed) ? Math.max(0, elapsed) : 0;
  if (!(Number.isFinite(typical) && typical > 0)) return 0.95 * (1 - Math.exp(-e / 120));
  const p = e / typical;
  return p <= 0.8 ? p : Math.min(0.95, 0.8 + 0.15 * (1 - Math.exp(-(p - 0.8) / 0.3)));
}

// Which pixels of an RGBA image are the figure: a mask's white on black, a cut-out's opaque pixels or a view's
// pixels that differ from its backdrop. The backdrop is read from the four corners.
export function figureMask(data, width, height) {
  const corners = [0, width - 1, (height - 1) * width, height * width - 1].map(i => i * 4);
  const bg = [0, 1, 2, 3].map(c => corners.reduce((sum, i) => sum + data[i + c], 0) / 4);
  const clear = bg[3] < 128, inside = new Uint8Array(width * height);
  let count = 0;
  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    const on = data[o + 3] >= 128 && (clear || Math.abs(data[o] - bg[0]) + Math.abs(data[o + 1] - bg[1]) + Math.abs(data[o + 2] - bg[2]) > 60);
    if (on) { inside[i] = 1; count++; }
  }
  return {width, height, inside, count};
}

function rows(mask) {
  const out = [];
  let top = -1, bottom = -1, left = Infinity, right = -Infinity;
  for (let y = 0; y < mask.height; y++) {
    const xs = [];
    for (let x = 0; x < mask.width; x++) if (mask.inside[y * mask.width + x]) xs.push(x);
    out.push(xs);
    if (xs.length) { if (top < 0) top = y; bottom = y; left = Math.min(left, xs[0]); right = Math.max(right, xs.at(-1)); }
  }
  return top < 0 ? null : {rows: out, top, bottom, centre: (left + right + 1) / 2};
}

/**
 * `count` points inside the figure, height 1 and centred on the origin, facing +Z: x and y from the front mask and,
 * row by row, the depth from the side mask (the left view, whose figure faces the image's left), so the points
 * fill the shape both views agree on. Without a side mask the depth is a modest slab. A third of the points sit on
 * the front outline so the figure reads at once. Returns {positions (x, y, z per point), seeds (0..1)}, or null.
 */
export function sampleFigure(front, side, count, random = Math.random) {
  const f = front && rows(front);
  if (!f || count <= 0) return null;
  const s = side && rows(side);
  const height = f.bottom - f.top + 1, unit = 1 / height;
  const filled = [], edges = [];
  for (let y = f.top; y <= f.bottom; y++) {
    const xs = f.rows[y];
    for (let i = 0; i < xs.length; i++) {
      filled.push([xs[i], y]);
      if (i === 0 || i === xs.length - 1 || xs[i - 1] !== xs[i] - 1 || xs[i + 1] !== xs[i] + 1) edges.push([xs[i], y]);
    }
  }
  const positions = new Float32Array(count * 3), seeds = new Float32Array(count);
  for (let n = 0; n < count; n++) {
    const pool = n % 3 === 0 && edges.length ? edges : filled;
    const [px, py] = pool[Math.floor(random() * pool.length)];
    const fy = py + random(), t = (fy - f.top) / height;
    let z;
    const zs = s && s.rows[Math.min(s.bottom, Math.max(s.top, Math.floor(s.top + t * (s.bottom - s.top + 1))))];
    if (zs?.length) {
      const sideUnit = 1 / (s.bottom - s.top + 1);
      z = (s.centre - (zs[Math.floor(random() * zs.length)] + random())) * sideUnit;
    } else {
      const xs = f.rows[py], half = xs.length ? (xs.at(-1) - xs[0] + 1) * unit / 2 : 0.05;
      z = (random() * 2 - 1) * Math.min(0.12, half * 0.6);
    }
    positions[n * 3] = (px + random() - f.centre) * unit;
    positions[n * 3 + 1] = ((f.top + f.bottom + 1) / 2 - fy) * unit;
    positions[n * 3 + 2] = z;
    seeds[n] = random();
  }
  return {positions, seeds};
}

// ---- in the browser ---------------------------------------------------------------------------------------------
const SAMPLE_PX = 160;   // masks are read at this height: plenty for a few thousand dots
const masks = new Map();
function readMask(url) {
  if (!url) return Promise.resolve(null);
  if (!masks.has(url)) {
    const loading = new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => {
        const k = SAMPLE_PX / Math.max(1, img.naturalHeight);
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(img.naturalWidth * k)); canvas.height = Math.max(1, Math.round(img.naturalHeight * k));
        const context = canvas.getContext('2d', {willReadFrequently: true});
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(figureMask(context.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height));
      };
      img.onerror = () => reject(new Error('mask unavailable'));
      img.src = url;
    });
    masks.set(url, loading);
    loading.catch(() => masks.delete(url));
  }
  return masks.get(url);
}
// The figure for `info` ({key, front, side, count}): a side mask that cannot be read leaves a modest depth.
export async function figurePoints(info) {
  const [front, side] = await Promise.all([readMask(info.front), readMask(info.side).catch(() => null)]);
  const points = sampleFigure(front, side, info.count);
  if (!points) throw new Error('no figure in the mask');
  return {key: info.key, ...points};
}
