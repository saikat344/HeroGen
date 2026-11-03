import test from 'node:test';
import assert from 'node:assert/strict';
import {downloadState, GLB_HINT, ROBLOX_HINT} from '../herogen/service/static/downloads.js';
import {robloxDownload} from '../herogen/service/static/heroes.js';

const GLB = {available: true, url: '/api/jobs/j/download/glb?artifact=a.glb&v=1'};
const ROBLOX = {available: true, url: '/api/jobs/j/download/roblox'};
const job = (extra = {}) => ({id: 'j', kind: 'prompt', state: 'succeeded', export_state: 'ready', roblox_state: 'ready',
  downloads: {glb: GLB, roblox: ROBLOX}, ...extra});

test('a Roblox-ready hero leads with the Roblox bundle and keeps the GLB beside it', () => {
  const view = downloadState(job());
  assert.equal(view.state, 'ready');
  assert.deepEqual(view.roblox, {url: ROBLOX.url});
  assert.deepEqual(view.glb, {url: GLB.url, primary: false});
  assert.equal(view.note, '');
});

test('while the Roblox files are prepared the GLB stays available and the Roblox button waits', () => {
  const view = downloadState(job({export_state: 'running', roblox_state: 'running',
    downloads: {glb: GLB, roblox: {available: false, reason: 'Roblox files are being prepared.'}}}));
  assert.equal(view.state, 'exporting');
  assert.deepEqual(view.roblox, {busy: true});
  assert.deepEqual(view.glb, {url: GLB.url, primary: false});
});

test('a failed Roblox export says why and makes the GLB the primary download', () => {
  const view = downloadState(job({export_state: 'unavailable', roblox_state: 'unavailable',
    downloads: {glb: GLB, roblox: {available: false, reason: 'The Roblox rig did not pass its checks for this hero.'}}}));
  assert.equal(view.state, 'unavailable');
  assert.equal(view.roblox, null);
  assert.deepEqual(view.glb, {url: GLB.url, primary: true});
  assert.equal(view.note, 'Roblox version unavailable. The Roblox rig did not pass its checks for this hero.');
});

test('the note gives the cause once, as a sentence', () => {
  const view = downloadState(job({roblox_state: 'unavailable',
    downloads: {glb: GLB, roblox: {available: false, reason: 'Roblox files unavailable for this hero: the Roblox rig stage stopped unexpectedly'}}}));
  assert.equal(view.note, 'Roblox version unavailable. The Roblox rig stage stopped unexpectedly.');
});

test('a GLB-only hero (no Roblox files at all) offers the GLB as the primary download', () => {
  const view = downloadState(job({roblox_state: 'unavailable', roblox_message: '',
    downloads: {glb: GLB, roblox: {available: false}}}));
  assert.equal(view.state, 'unavailable');
  assert.deepEqual(view.glb, {url: GLB.url, primary: true});
  assert.match(view.note, /^Roblox version unavailable\. Roblox files are not available for this hero\.$/);
});

test('a hero still being built shows nothing until a model exists, then the GLB with Roblox waiting', () => {
  const none = {glb: {available: false, reason: 'The body is not finished yet.'}, roblox: {available: false, reason: 'The body is not finished yet.'}};
  assert.equal(downloadState(job({state: 'running', export_state: 'pending', roblox_state: 'pending', downloads: none})), null);
  const early = downloadState(job({state: 'running', export_state: 'pending', roblox_state: 'pending', downloads: {...none, glb: GLB}}));
  assert.equal(early.state, 'building');
  assert.deepEqual(early.roblox, {waiting: true});
  assert.deepEqual(early.glb, {url: GLB.url, primary: false});
});

test('nothing is offered for a refused input or a failed hero with no model', () => {
  assert.equal(downloadState(job({input_rejected: true})), null);
  assert.equal(downloadState(job({state: 'failed', downloads: {glb: {available: false, reason: 'x'}, roblox: {available: false, reason: 'y'}}})), null);
  assert.equal(downloadState(null), null);
});

test('the hints describe what each file is for, without promising a setup-free import', () => {
  assert.match(ROBLOX_HINT, /R15 rig/);
  assert.match(ROBLOX_HINT, /Roblox Studio/);
  assert.doesNotMatch(ROBLOX_HINT, /no setup|one click|instantly/i);
  assert.match(GLB_HINT, /Blender/);
});

test('a hero card offers the one-click Roblox download only when it is Roblox ready', () => {
  const card = {id: 'j', kind: 'prompt', state: 'succeeded', export_state: 'ready'};
  assert.equal(robloxDownload(card, {roblox_state: 'ready', downloads: {roblox: ROBLOX}}), ROBLOX.url);
  assert.equal(robloxDownload({...card, export_state: 'running'}, {roblox_state: 'ready', downloads: {roblox: ROBLOX}}), null);
  assert.equal(robloxDownload(card, {roblox_state: 'unavailable', downloads: {roblox: {available: false}}}), null);
  assert.equal(robloxDownload(card, null), null);
  assert.equal(robloxDownload({...card, state: 'running'}, {roblox_state: 'ready', downloads: {roblox: ROBLOX}}), null);
});
