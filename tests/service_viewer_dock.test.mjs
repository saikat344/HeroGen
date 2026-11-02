import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraImages } from '../herogen/service/static/viewer-dock.js';

test('Front, Side and Back use the stage-2 views and fall back to the artwork', () => {
  assert.deepEqual(cameraImages({front: 'f.png', left: 'l.png', right: 'r.png', back: 'b.png', cover: 'c.png'}).map(v => [v.key, v.url]),
    [['front', 'f.png'], ['side', 'l.png'], ['back', 'b.png']]);
  assert.deepEqual(cameraImages({right: 'r.png', cover: 'c.png'}).map(v => v.url), ['c.png', 'r.png', 'c.png']);
  assert.deepEqual(cameraImages(undefined).map(v => [v.label, v.url]), [['Front', null], ['Side', null], ['Back', null]]);
});
