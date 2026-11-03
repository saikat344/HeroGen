import test from 'node:test';
import assert from 'node:assert/strict';
import { animateDetail, animateNote, detailChoices, meshStats } from '../herogen/service/static/detail.js';

const roblox = { high: 23630, high_budget: 24000, low: 7690, low_budget: 6000 };

test('the detail switch appears once stage 4 has made a game body, with the levels that exist', () => {
  assert.deepEqual(detailChoices(null), []);
  assert.deepEqual(detailChoices({}), []);
  assert.deepEqual(detailChoices({ high: { url: 'h.glb' }, low: { url: 'l.glb' } }), ['refined', 'high', 'low']);
  assert.deepEqual(detailChoices({ high: { url: 'h.glb' } }), ['refined', 'high']);
  assert.deepEqual(detailChoices({ high: { url: null }, low: { url: 'l.glb' } }), ['refined', 'low']);
});
test('the triangle line counts only the mesh on screen, against its own budget', () => {
  assert.deepEqual(meshStats('high', 23630, roblox), { text: 'Roblox high · 23,630 / 24,000 triangles', over: false });
  assert.deepEqual(meshStats('low', 7690, roblox), { text: 'Roblox low · 7,690 / 6,000 triangles (over budget)', over: true });
  assert.deepEqual(meshStats('refined', 9530, roblox), { text: 'Refined · 9,530 triangles', over: false }, 'no Roblox budget on the refined mesh');
  assert.deepEqual(meshStats('raw', 25863, roblox), { text: '25,863 triangles', over: false });
  assert.deepEqual(meshStats('model', 9530, null), { text: '9,530 triangles', over: false });
  assert.equal(meshStats('low', 6000, { ...roblox, low: 6000 }).over, false, 'exactly on budget is within it');
});
test('a Roblox body takes stage 4’s count, falling back to the loaded mesh', () => {
  assert.equal(meshStats('high', 1, roblox).text, 'Roblox high · 23,630 / 24,000 triangles');
  assert.deepEqual(meshStats('low', 5720, null), { text: 'Roblox low · 5,720 triangles', over: false }, 'no budget known');
  assert.equal(meshStats('high', 0, null).text, '');
  assert.equal(meshStats('refined', 0, roblox).text, '');
});
test('Animate plays on the selected Roblox body when it can, otherwise on Roblox high', () => {
  const bodies = { high: { url: 'h' }, low: { url: 'l', animates: true } };
  assert.equal(animateDetail('low', bodies), 'low');
  assert.equal(animateDetail('high', bodies), 'high');
  assert.equal(animateDetail('refined', bodies), 'high');
  assert.equal(animateDetail('low', { ...bodies, low: { url: 'l', animates: false } }), 'high');
  assert.equal(animateNote('low', 'low'), '');
  assert.match(animateNote('low', 'high'), /low body has no joints/);
  assert.match(animateNote('refined', 'high'), /Roblox high/);
});
