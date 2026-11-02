import test from 'node:test';
import assert from 'node:assert/strict';
import {stageEvents, stageSource, checksSummary, formatCost, formatSeconds, logText} from '../herogen/service/static/workbench.js';

const started = '2026-01-01T00:00:00Z', time = Date.parse(started);
const call = (extra) => ({model: 'sam-3.1', kind: 'segment', cached: false, started, wall_seconds: 3.84, cost_usd: 0.0025,
  outcome: 'ok', purpose: 'segment feet (cover art)', ...extra});

test('a started step shows what it does, its numbering and the usual time', () => {
  const job = {stages: [{key:'1',label:'cover art',state:'running',elapsed_seconds:5.6,typical_seconds:33.3},
    {key:'2',label:'turnaround',state:'pending'}]};
  const rows = stageEvents(job, null, time + 10000);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].kind, 'step');
  assert.equal(rows[0].text, 'Step 1 of 2 · Artwork');
  assert.equal(rows[0].meta, 'running · 5 s so far · usually about 33 s');
  assert.match(rows[0].note, /Muse Image paints the colour artwork.*SAM 3.1.*Muse Spark/);
  assert.equal(rows[0].time, time + 4400);
  assert.equal(stageEvents({stages: [{key:'1',state:'failed'}]}, null).length, 0);
});

test('a finished step lists its recorded calls with model, purpose, duration and cost', () => {
  const job = {stages: [{key:'1',label:'cover art',state:'passed',typical_seconds:150}]};
  const record = {stage:'1', started, seconds: 28.6, cost_usd: 0.015,
    checks: [{name:'a',gate:true,pass:true},{name:'b',gate:false,pass:false},{name:'c',gate:false,pass:true}],
    calls: [call({started:'2026-01-01T00:00:01Z'}),
      call({model:'muse-image-1.0',purpose:'cover art',started:'2026-01-01T00:00:06Z',wall_seconds:17.3,cost_usd:0.01}),
      call({model:'muse-spark-1.3',kind:'compare',purpose:'cover art',cached:true,outcome:null,started:'2026-01-01T00:00:00.5Z',cost_usd:0})]};
  const rows = stageEvents(job, {stages:[record]});
  assert.deepEqual(rows.map(r => [r.kind, r.who, r.text, r.meta]), [
    ['step', undefined, 'Step 1 of 1 · Artwork', 'done · 28.6 s · $0.01 (estimate) · usually about 3 min'],
    ['call', 'Muse Spark', 'Checked that the image is a drawing, not a photo of a real person', 'reused a saved result · $0'],
    ['call', 'SAM 3.1', 'Looked for the feet in the sketch', '3.8 s · $0.0025'],
    ['call', 'Muse Image', 'Painted the colour artwork from the sketch', '17.3 s · $0.01']]);
  assert.equal(rows[0].checks, '1 required check passed · 2 advisory (1 not met)');
  assert.equal(rows[1].state, 'reused');
});

test('steps without model calls are marked as HeroGen local work', () => {
  const job = {stages: [{key:'4',label:'game body',state:'passed'},{key:'1',label:'cover art',state:'passed'}]};
  const rows = stageEvents(job, {stages:[{stage:4,started,seconds:287.4,cost_usd:0,calls:[]},
    {stage:'1',started:'2026-01-01T00:10:00Z',seconds:3,calls:[]}]});
  assert.deepEqual(rows.map(r => [r.kind, r.who, r.text, r.meta]), [
    ['step', undefined, 'Step 1 of 2 · Game body', 'done · 4 min 47 s · $0 (estimate)'],
    ['call', 'HeroGen (local)', 'no model call', '4 min 47 s · $0'],
    ['step', undefined, 'Step 2 of 2 · Artwork', 'done · 3.0 s']]);
});

test('repeated calls fold into one row whose children keep every call', () => {
  const job = {stages: [{key:'3f',label:'texture refine',state:'passed'}]};
  const regions = ['head+0','body+0','body+90','body-90'];
  const calls = regions.map((r, i) => call({model:'muse-image-1.0',purpose:`repaint ${r}`,
    started:`2026-01-01T00:00:0${i + 1}Z`,wall_seconds:10 + i,cost_usd:0.01}));
  const rows = stageEvents(job, {stages:[{stage:'3f',started,seconds:60,calls}]});
  const group = rows[1];
  assert.equal(rows.length, 2);
  assert.equal(group.kind, 'group');
  assert.equal(group.text, 'Repainted 4 views');
  assert.equal(group.meta, 'longest 13.0 s · $0.04');
  assert.deepEqual(group.children.map(c => c.text), ['Repainted the face close-up', 'Repainted the front view',
    'Repainted the side view at +90°', 'Repainted the side view at −90°']);
  assert.match(logText(rows), /Repainted the side view at −90° · 13\.0 s · \$0\.01/);
});

test('older image-edit calls fold per model, not across models', () => {
  const calls = [...Array(3)].map(() => call({model:'muse-image-1.0',purpose:'edit'})).concat([...Array(2)].map(() => call({purpose:'edit'})));
  const rows = stageEvents({stages:[{key:'3f',state:'passed'}]}, {stages:[{stage:'3f',started,seconds:9,calls}]});
  assert.deepEqual(rows.slice(1).map(r => [r.kind, r.who, r.text]), [['group','Muse Image','Edited 3 images'],
    ['call','SAM 3.1','Edited an image'],['call','SAM 3.1','Edited an image']]);
});

test('failed and cancelled calls stay explicit and unknown records still render', () => {
  const job = {stages: [{key:'2',label:'turnaround',state:'failed',message:'The views could not be drawn.'}, {key:'x',label:'new step',state:'passed'}]};
  const rows = stageEvents(job, {stages:[{stage:'2',started,seconds:4,calls:[call({outcome:'failed',purpose:'turnaround',model:'muse-image-1.0'}),
    null, call({outcome:'cancelled',purpose:'something new'})]},
    {stage:'x',started,seconds:1,calls:[call({model:null,purpose:'',cost_usd:null,wall_seconds:null})]}]});
  assert.equal(rows[0].message, 'The views could not be drawn.');
  const texts = rows.map(r => r.text);
  assert.ok(texts.includes('Drew the turnaround sheet: front, right, back and left: failed'));
  assert.ok(texts.includes('Something new: cancelled'));
  const unknown = rows.find(r => r.who === 'model not recorded');
  assert.equal(unknown.text, 'Model call');
  assert.equal(unknown.meta, '');
  assert.equal(rows.find(r => r.text.startsWith('Step 2 of 2')).text, 'Step 2 of 2 · new step');
});

test('every outcome other than ok or reused is a failure, shown in plain words', () => {
  const run = outcome => stageEvents({stages:[{key:'1',state:'passed'}]}, {stages:[{stage:'1',started,seconds:3,calls:[
    {model:'muse-image-1.0',purpose:'cover art',outcome,wall_seconds:3,cost_usd:0.04,started}]}]})[1];
  const probe = run('ApiError');
  assert.deepEqual([probe.state, probe.text, probe.meta], ['failed', 'Painted the colour artwork from the sketch: failed', '3.0 s \u00b7 $0.04']);
  for (const [outcome, text] of [['failed','failed'],['ApiError','failed'],['TimeoutError','failed'],['cancelled','cancelled'],
    ['unknown','outcome unknown (may be billed)'],['unreadable','answer unreadable (may be billed)'],['anything else','failed']]) {
    const row = run(outcome);
    assert.equal(row.state, 'failed', outcome);
    assert.equal(row.text, `Painted the colour artwork from the sketch: ${text}`, outcome);
  }
  for (const outcome of ['ok', undefined, null, '']) assert.equal(run(outcome).state, 'passed', String(outcome));
  assert.equal(run('reused').state, 'reused');
  const cached = stageEvents({stages:[{key:'1',state:'passed'}]}, {stages:[{stage:'1',started,seconds:3,calls:[
    {model:'sam-3.1',purpose:'segment feet (cover art)',outcome:'ok',cached:true,wall_seconds:0.1,cost_usd:0}]}]})[1];
  assert.deepEqual([cached.state, cached.meta], ['reused', 'reused a saved result \u00b7 $0']);
});

test('a failed call inside a fold marks the fold failed and is listed in the copied log', () => {
  const calls = ['ok','ApiError','ok','unknown'].map((outcome, i) => call({model:'muse-image-1.0',purpose:'texture refine',outcome,
    started:`2026-01-01T00:00:0${i}Z`}));
  const rows = stageEvents({stages:[{key:'3f',state:'passed'}]}, {stages:[{stage:'3f',started,seconds:9,calls}]});
  assert.deepEqual([rows[1].kind, rows[1].state, rows[1].failed, rows[1].text], ['group', 'failed', 2, 'Repainted 4 views']);
  assert.equal(rows[1].meta, '2 failed \u00b7 longest 3.8 s \u00b7 $0.01');
  assert.match(logText(rows), /Repainted a view of the 3D shape: outcome unknown \(may be billed\)/);
});

test('fast texture refine calls fold, and a fold with no durations omits "longest"', () => {
  const refine = [...Array(5)].map(() => call({model:'muse-image-1.0',purpose:'texture refine'}))
    .concat([...Array(4)].map(() => call({purpose:'segment face (texture refine)'})));
  const rows = stageEvents({stages:[{key:'3f',state:'passed'}]}, {stages:[{stage:'3f',started,seconds:9,calls:refine}]});
  assert.deepEqual(rows.slice(1).map(r => [r.kind, r.who, r.text]), [['group','Muse Image','Repainted 5 views'],
    ['group','SAM 3.1','Found the face on 4 renders']]);
  const bare = [...Array(3)].map(() => ({model:'muse-image-1.0',purpose:'texture refine',started,outcome:'ok'}));
  const folded = stageEvents({stages:[{key:'3f',state:'passed'}]}, {stages:[{stage:'3f',started,seconds:9,calls:bare}]})[1];
  assert.equal(folded.meta, '');
});

test('a very large call list neither overflows the stack nor takes quadratic time', () => {
  const calls = Array.from({length: 250000}, (_, i) => call({model:'muse-image-1.0',purpose:'texture refine',started:new Date(time + i).toISOString()}));
  const begun = Date.now();
  const rows = stageEvents({stages:[{key:'3f',state:'passed'}]}, {stages:[{stage:'3f',started,seconds:9,calls}]});
  assert.equal(rows[1].children.length, 250000);
  assert.ok(Date.now() - begun < 5000);
});

test('refinement publication times sit in time order among the step rows', () => {
  const job = {stages: [{key:'3f',label:'texture refine',state:'running',elapsed_seconds:5}],
    models:[{n:1,time:'2026-01-01T00:00:03Z'}, {n:2}]};
  const rows = stageEvents(job, null, time + 10000);
  assert.deepEqual(rows.map(r => [r.time - time, r.text]), [[3000,'Refined version 1 ready'],[5000,'Step 1 of 1 · Texture refine']]);
});

test('formatting and check summaries', () => {
  assert.equal(formatCost(0), '$0'); assert.equal(formatCost(0.0025), '$0.0025'); assert.equal(formatCost(0.1009), '$0.10');
  assert.equal(formatCost(NaN), null); assert.equal(formatSeconds(0.04), '0.0 s'); assert.equal(formatSeconds(172.9), '2 min 53 s');
  assert.equal(formatSeconds(179.6), '3 min 0 s'); assert.equal(formatSeconds(119.4), '119.4 s'); assert.equal(formatSeconds(119.7), '2 min 0 s');
  assert.equal(checksSummary([]), null); assert.equal(checksSummary(undefined), null);
  assert.equal(checksSummary([{gate:true,pass:false},{gate:true,pass:true}]), '1 of 2 required checks not met');
  assert.equal(checksSummary([{pass:true},{pass:true}]), '2 advisory');
  assert.equal(stageSource('3'),'https://github.com/saikat344/HeroGen/blob/main/herogen/blender/stage3.py');
  assert.equal(stageSource('unknown'),null);
});
