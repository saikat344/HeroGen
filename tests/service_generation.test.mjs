import test from 'node:test';
import assert from 'node:assert/strict';
import { generationState } from '../herogen/service/static/generation.js';
const stage = (key, state, side=false) => ({key, state, side, label:`Stage ${key}`});
test('prompt and uploaded inputs use only the supplied pipeline, in its real order', () => {
  const stages=[stage('p','passed'),stage('0','passed'),stage('1','running'),stage('6','pending'),stage('5','pending')];
  const v=generationState({state:'running',stages});
  assert.deepEqual(v.steps.map(s=>s.key),['p','0','1','6','5']);
  assert.equal(v.passed,2); assert.equal(v.focus.key,'1'); assert.equal(v.subtitle,'Painting your hero');
  assert.equal(generationState({state:'running',stages:stages.slice(1)}).steps.length,4);
});
test('a side review cannot block completion or count as a main stage', () => {
  const v=generationState({state:'succeeded',stages:[stage('3','passed'),stage('2h','running',true)]});
  assert.equal(v.title,'Meet your hero.'); assert.equal(v.main.length,1); assert.equal(v.passed,1); assert.equal(v.reviewRunning,true);
});
test('a failed review remains visible without relabelling the successful job', () => {
  const v=generationState({state:'succeeded',stages:[stage('3','passed'),stage('2h','rejected',true)]});
  assert.equal(v.title,'Meet your hero.'); assert.match(v.subtitle,/attention/); assert.equal(v.side[0].state,'rejected');
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
  assert.equal(generationState({state:'queued',error:{message:'Busy'}}).title,'Waiting to retry');
  const v=generationState({state:'queued'}); assert.equal(v.steps.length,0); assert.equal(v.passed,0);
  const custom=generationState({state:'running',stages:[stage('future','running')]});
  assert.equal(custom.steps[0].title,'Stage future');
});
