import test from 'node:test';
import assert from 'node:assert/strict';
import {stageEvents, stageSource} from '../herogen/service/static/workbench.js';

test('live log uses recorded starts, durations and refinement publication times', () => {
  const started = '2026-01-01T00:00:00Z', time = Date.parse(started);
  const job = {stages: [{key:'1',label:'Artwork',state:'passed'},
    {key:'2',label:'Views',state:'running',elapsed_seconds:5}, {key:'3',state:'pending'}],
    models:[{n:1,time:'2026-01-01T00:00:03Z'}, {n:2}]};
  const rows = stageEvents(job, {stages:[{stage:'1',started,seconds:2}]}, time+10000);
  assert.deepEqual(rows.map(r=>[r.time-time,r.text]), [[0,'Artwork started'],[2000,'Artwork · passed'],
    [3000,'Refined version 1 ready'],[5000,'Views started']]);
  assert.deepEqual(stageEvents({stages:[{key:'1',state:'failed'}]}, null), []);
});
test('log keeps failures explicit and never invents a missing finish time', () => {
  const stages = [{key:'1',label:'Artwork',state:'failed'},{key:'2',label:'Views',state:'running'}];
  const rows = stageEvents({stages}, {stages:[{stage:'1',started:'2026-01-01',seconds:1},
    {stage:'2',started:'2026-01-02',seconds:2}]});
  assert.equal(rows[1].text,'Artwork · failed');
  assert.equal(rows.length,3);
  assert.equal(stageSource('3'),'https://github.com/saikat344/HeroGen/blob/main/herogen/blender/stage3.py');
  assert.equal(stageSource('unknown'),null);
});
