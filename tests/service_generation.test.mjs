import { readFileSync } from 'node:fs';
import { SourceTextModule } from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import { generationState, segmentLabels, stageTiming, viewsReveal } from '../herogen/service/static/generation.js';
const stage = (key, state, side=false) => ({key, state, side, label:`Stage ${key}`});
test('prompt and uploaded inputs use only the supplied pipeline, in its real order', () => {
  const stages=[stage('p','passed'),stage('0','passed'),stage('1','running'),stage('6','pending'),stage('5','pending')];
  const v=generationState({state:'running',stages});
  assert.deepEqual(v.steps.map(s=>s.key),['p','0','1','6','5']);
  assert.equal(v.passed,2); assert.equal(v.focus.key,'1'); assert.equal(v.subtitle,'Painting your hero');
  assert.equal(generationState({state:'running',stages:stages.slice(1)}).steps.length,4);
});
test('optional unavailable export is never passed or hidden', () => {
  const v=generationState({state:'succeeded',stages:[stage('4','passed'),stage('6','unavailable'),stage('5','passed')]});
  assert.equal(v.passed,2); assert.equal(v.issues[0].key,'6'); assert.equal(v.steps[1].status,'Unavailable');
});
test('interrupted jobs do not pretend pending steps completed', () => {
  for(const state of ['failed','rejected','canceled']) {
    const v=generationState({state,stages:[stage('1',state),stage('2','pending')]});
    assert.equal(v.terminal,true); assert.equal(v.steps[1].status,'Not reached'); assert.equal(v.passed,0);
  }
});
test('queued retry and missing schema stay honest', () => {
  assert.equal(generationState({state:'queued',error:{message:'Busy'}}).title,'Waiting for a free worker');
  const v=generationState({state:'queued'}); assert.equal(v.steps.length,0); assert.equal(v.passed,0);
  const custom=generationState({state:'running',stages:[stage('future','running')]});
  assert.equal(custom.steps[0].title,'Stage future');
});
test('single-step versions are terminal saved results', () => {
  const state = generationState({state:'stopped',stages:[stage('2','passed'),stage('3','pending')]});
  assert.equal(state.terminal,true);
  assert.equal(state.title,'Your version is saved.');
  assert.equal(state.steps[1].status,'Not reached');
});

test('the entry and admin browser modules parse as ES modules', () => {
  for (const name of ['app', 'admin']) {
    const source = readFileSync(new URL(`../herogen/service/static/${name}.js`, import.meta.url), 'utf8');
    assert.doesNotThrow(() => new SourceTextModule(source));
  }
});
test('stage timing reports running elapsed, finished measurements and supplied medians only',()=>{
  assert.equal(stageTiming({state:'running',elapsed_seconds:65.9}),'65 s elapsed');
  assert.equal(stageTiming({state:'passed',seconds:120,cost_usd:.125,typical_seconds:119}),
    '120.0 s · $0.13 · usually about 2 min');
  assert.equal(stageTiming({state:'passed',seconds:0,cost_usd:0}),'0.0 s · $0.00');
  assert.equal(stageTiming({state:'pending'}),'');
  assert.equal(stageTiming({state:'running',seconds:10,cost_usd:.1,elapsed_seconds:12,typical_seconds:60}),
    '12 s elapsed · usually about 60 s');
});

test('healthy finished heroes get the new title while degraded results keep their copy', () => {
  assert.equal(generationState({state:'succeeded',stages:[stage('3f','passed')]}).title,'Your hero. Your next idea.');
  assert.equal(generationState({state:'succeeded',stages:[stage('3f','skipped')]}).title,'Meet your hero.');
  assert.equal(generationState({state:'failed'}).title,'Your hero needs attention.');
});
test('each progress segment names its stage and state, so the bar never relies on colour alone', () => {
  const running=generationState({state:'running',stages:[stage('p','passed'),stage('3f','running'),stage('4','pending'),stage('5','pending')]});
  assert.deepEqual(segmentLabels(running.steps,running.terminal),
    ['Your idea · done','Texture refine · in progress','Game body · up next','Results · not started']);
  const queued=generationState({state:'running',stages:[stage('p','passed'),stage('0','pending'),stage('1','pending')]});
  assert.deepEqual(segmentLabels(queued.steps,queued.terminal),['Your idea · done','Prepare input · up next','Artwork · not started']);
  const failed=generationState({state:'failed',stages:[stage('1','failed'),stage('2','pending'),stage('3f','skipped')]});
  assert.deepEqual(segmentLabels(failed.steps,failed.terminal),['Artwork · failed','Character views · not reached','Texture refine · skipped']);
});

test('a hero whose Roblox files are ready is not called "preparing" while the export finishes', () => {
  const stages = [stage('4','passed'),stage('6','passed'),stage('5','running')];
  assert.equal(generationState({state:'succeeded',export_state:'running',stages}).eyebrow,'Hero ready · preparing Roblox files');
  assert.equal(generationState({state:'succeeded',export_state:'running',roblox_state:'ready',stages}).eyebrow,'Hero ready · Roblox R15');
});

test('the character views show as drafts while drawn, then as the turntable until the first 3D shape', () => {
  const at = (two, three) => [stage('1','passed'), stage('2',two), stage('3',three)];
  const sides = {front:'f', right:'r', back:'b', left:'l'};
  assert.equal(viewsReveal({state:'running', stages:at('passed','passed'), images:{}}), null);
  const start = viewsReveal({state:'running', stages:at('running','pending'), images:{cover:'c'}, draft_images:{}});
  assert.equal(start.drawing, true); assert.equal(start.sheet, null); assert.equal(start.turn, false);
  const sheet = viewsReveal({state:'running', stages:at('running','pending'), images:{cover:'c'},
    draft_images:{turnaround:'t', top_drawn:'td', front:'df'}});
  assert.equal(sheet.sheet, 't'); assert.equal(sheet.views.top, 'td'); assert.equal(sheet.cut.top, false);
  assert.equal(sheet.cut.front, true); assert.equal(sheet.turn, false);
  const cut = viewsReveal({state:'running', stages:at('running','pending'), images:{}, draft_images:{...sides, top:'dt', top_drawn:'td'}});
  assert.equal(cut.turn, true); assert.equal(cut.views.top, 'dt'); assert.equal(cut.cut.top, true);
  const shaping = viewsReveal({state:'running', stages:at('passed','running'), images:{...sides, turnaround:'t'}, draft_images:{front:'stale'}});
  assert.deepEqual([shaping.drawing, shaping.sculpting, shaping.turn, shaping.views.front, shaping.cut.front], [false, true, true, 'f', false]);
  assert.equal(viewsReveal({state:'running', stages:at('passed','running'), images:sides, downloads:{glb:{available:true}}}), null);
  for (const state of ['failed', 'succeeded', 'canceled'])
    assert.equal(viewsReveal({state, stages:at('running','pending'), images:{}, draft_images:sides}), null, state);
});
