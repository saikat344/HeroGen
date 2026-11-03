import test from 'node:test';
import assert from 'node:assert/strict';
import { clampRatio, keyRatio, ratioAt, splitAxis, splitWanted, textureRefining, SPLIT_MAX, SPLIT_MIN } from '../herogen/service/static/split-view.js';
import { splitCopy } from '../herogen/service/static/generation.js';

const job = (state, stages) => ({ state, stages: Object.entries(stages).map(([key, s]) => ({ key, state: s })) });

test('the divider stays between 20% and 80% and falls back to 50/50', () => {
  assert.equal(clampRatio(0.05), SPLIT_MIN); assert.equal(clampRatio(0.97), SPLIT_MAX); assert.equal(clampRatio(0.42), 0.42);
  assert.equal(clampRatio(NaN), 0.5); assert.equal(clampRatio(undefined), 0.5);
  const rect = { left: 100, top: 50, width: 800, height: 400 };
  assert.equal(ratioAt(500, 0, rect), 0.5); assert.equal(ratioAt(0, 0, rect), SPLIT_MIN); assert.equal(ratioAt(2000, 0, rect), SPLIT_MAX);
  assert.equal(ratioAt(0, 150, rect, 'rows'), 0.25);
  assert.equal(ratioAt(0, 200, rect, 'rows', 100), 0.5, 'stacked panels share the height above the dock');
  assert.equal(ratioAt(10, 10, { left: 0, top: 0, width: 0, height: 0 }), 0.5);
});
test('arrow keys move the divider the way it points; Home and End reach the limits', () => {
  assert.equal(keyRatio('ArrowLeft', 0.5), 0.45); assert.equal(keyRatio('ArrowRight', 0.5), 0.55);
  assert.equal(keyRatio('ArrowDown', 0.5), 0.45); assert.equal(keyRatio('ArrowUp', 0.5), 0.55);
  assert.equal(keyRatio('ArrowUp', 0.5, 'rows'), 0.45, 'stacked: up moves the divider up');
  assert.equal(keyRatio('ArrowDown', 0.5, 'rows'), 0.55);
  assert.equal(keyRatio('ArrowRight', 0.5, 'columns', true), 0.6);
  assert.equal(keyRatio('ArrowLeft', 0.22), SPLIT_MIN); assert.equal(keyRatio('ArrowRight', 0.78), SPLIT_MAX);
  assert.equal(keyRatio('Home', 0.5), SPLIT_MIN); assert.equal(keyRatio('End', 0.5), SPLIT_MAX);
  assert.equal(keyRatio('Enter', 0.5), null); assert.equal(keyRatio('a', 0.5), null);
});
test('a stage narrower than 700 px stacks the panels', () => {
  assert.equal(splitAxis(1200), 'columns'); assert.equal(splitAxis(700), 'columns'); assert.equal(splitAxis(699), 'rows');
  assert.equal(splitAxis(360), 'rows'); assert.equal(splitAxis(0), 'columns', 'an unlaid stage keeps columns');
});
test('the split applies while the texture refines, or when Side by side is picked', () => {
  const refining = job('running', { 3: 'passed', '3f': 'running' });
  assert.equal(textureRefining(refining), true);
  assert.equal(textureRefining(job('running', { 3: 'passed', '3f': 'passed', 4: 'running' })), false);
  assert.equal(textureRefining(job('succeeded', { '3f': 'running' })), false);
  assert.equal(textureRefining(null), false);
  const raw = { url: 'source.glb', raw: true };
  assert.equal(splitWanted({ refining: true, raw, choice: null }), true);
  assert.equal(splitWanted({ refining: true, raw: null, choice: null }), false, 'no SAM 3D shape, no split');
  assert.equal(splitWanted({ refining: true, raw, choice: 'herogen' }), false, 'one model picked on the switch');
  assert.equal(splitWanted({ refining: true, raw, choice: 'raw' }), false);
  assert.equal(splitWanted({ refining: false, raw, choice: null }), false, 'a finished hero is one model');
  assert.equal(splitWanted({ refining: false, raw, choice: 'split' }), true, 'Side by side after the fact');
});
test('the HeroGen panel says it is refining until a version lands, then names the version', () => {
  assert.deepEqual(splitCopy(null, true), { label: 'HeroGen · refining…', working: 'Repainting the texture from every side…', waiting: true });
  assert.equal(splitCopy({ raw: true, n: 0 }, true).waiting, true, 'SAM 3D shape stands in while it repaints');
  assert.deepEqual(splitCopy({ n: 2 }, true), { label: 'HeroGen · version 2', working: 'Still refining…', waiting: false });
  assert.deepEqual(splitCopy({ n: 3 }, false), { label: 'HeroGen · version 3', working: '', waiting: false });
  assert.deepEqual(splitCopy({ n: null }, false), { label: 'HeroGen', working: '', waiting: false });
  assert.equal(splitCopy({ raw: true }, false).waiting, false, 'a finished hero never shows the repaint light');
});
