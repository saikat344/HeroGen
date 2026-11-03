// The level of detail of HeroGen's model on the stage: the refined mesh, or the Roblox game body at high or low
// detail (the files the Roblox bundle ships). Pure helpers for generation.js; the triangle line names the mesh on
// screen and holds it against its Roblox budget.
export const DETAILS = [['refined', 'Refined'], ['high', 'Roblox high'], ['low', 'Roblox low']];
const NAME = Object.fromEntries(DETAILS);
const count = (n) => Number(n).toLocaleString('en-US');

// The switch's choices for a job's `roblox_bodies` ({high, low}: each {url}); none until stage 4 has made a body.
export function detailChoices(bodies) {
  const has = (k) => typeof bodies?.[k]?.url === 'string';
  if (!has('high') && !has('low')) return [];
  return DETAILS.map(([k]) => k).filter((k) => k === 'refined' || has(k));
}

/** The triangle line for the mesh on screen. `kind`: 'raw' (SAM 3D), 'model' (HeroGen's model on a hero without a
 *  game body), 'refined', 'high' or 'low'. `tris`: the loaded mesh's count. `roblox`: the job's `roblox_triangles`
 *  ({high, high_budget, low, low_budget}). A Roblox body takes stage 4's own count, the number its budget check
 *  used (the same mesh the viewer loaded), and falls back to the loaded count. Returns {text, over}. */
export function meshStats(kind, tris, roblox) {
  if (kind === 'high' || kind === 'low') {
    const recorded = roblox?.[kind], budget = roblox?.[`${kind}_budget`];
    const n = Number.isInteger(recorded) && recorded > 0 ? recorded : tris;
    if (!(n > 0)) return { text: '', over: false };
    if (!(Number.isInteger(budget) && budget > 0)) return { text: `${NAME[kind]} · ${count(n)} triangles`, over: false };
    const over = n > budget;
    return { text: `${NAME[kind]} · ${count(n)} / ${count(budget)} triangles${over ? ' (over budget)' : ''}`, over };
  }
  if (!(Number.isFinite(tris) && tris > 0)) return { text: '', over: false };
  return { text: `${kind === 'refined' ? 'Refined · ' : ''}${count(tris)} triangles`, over: false };
}

// Animate plays on a Roblox body: the low one when it is on screen and has the 15 R15 parts, otherwise the high one.
export function animateDetail(detail, bodies) {
  return detail === 'low' && bodies?.low?.animates === true ? 'low' : 'high';
}
// What the stage says when Animate moves it to another body (nothing when it stays).
export function animateNote(from, to) {
  if (from === to) return '';
  return from === 'low' ? 'The low body has no joints to move, so Animate plays on Roblox high.'
    : 'Animate plays on the Roblox game body, so the stage shows Roblox high.';
}
