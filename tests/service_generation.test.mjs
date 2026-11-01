import { readFileSync } from 'node:fs';
import { SourceTextModule } from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import { generationState, stageTiming } from '../herogen/service/static/generation.js';
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
