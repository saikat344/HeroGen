import { readFileSync } from 'node:fs';
import { SourceTextModule } from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import { generationState, modelLabel, segmentLabels, shapeBuild, stageTiming, viewsReveal } from '../herogen/service/static/generation.js';
import { buildProgress, figureMask, sampleFigure, shapeTip, SHAPE_TIPS } from '../herogen/service/static/shape-build.js';
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

test('the character views show as drafts while they are drawn', () => {
  const at = (two, three) => [stage('1','passed'), stage('2',two), stage('3',three)];
  const sides = {front:'f', right:'r', back:'b', left:'l'};
  assert.equal(viewsReveal({state:'running', stages:at('passed','passed'), images:{}}), null);
  const start = viewsReveal({state:'running', stages:at('running','pending'), images:{cover:'c'}, draft_images:{}});
  assert.equal(start.drawing, true); assert.equal(start.sheet, null);
  const sheet = viewsReveal({state:'running', stages:at('running','pending'), images:{cover:'c'},
    draft_images:{turnaround:'t', top_drawn:'td', front:'df'}});
  assert.equal(sheet.sheet, 't'); assert.equal(sheet.views.top, 'td'); assert.equal(sheet.cut.top, false);
  assert.equal(sheet.cut.front, true);
  const cut = viewsReveal({state:'running', stages:at('running','pending'), images:{}, draft_images:{...sides, top:'dt', top_drawn:'td'}});
  assert.equal(cut.views.top, 'dt'); assert.equal(cut.cut.top, true);
  const shaping = viewsReveal({state:'running', stages:at('passed','running'), images:{...sides, turnaround:'t'}, draft_images:{front:'stale'}});
  assert.deepEqual([shaping.drawing, shaping.sculpting, shaping.views.front, shaping.cut.front], [false, true, 'f', false]);
  assert.equal(viewsReveal({state:'running', stages:at('passed','running'), images:sides, downloads:{glb:{available:true}}}), null);
  for (const state of ['failed', 'succeeded', 'canceled'])
    assert.equal(viewsReveal({state, stages:at('running','pending'), images:{}, draft_images:sides}), null, state);
});

test('while the 3D shape is built the viewer shows a figure of dots until any 3D model is out', () => {
  const at = (two, three, extra={}) => [stage('2',two), {...stage('3',three), ...extra}];
  const masks = {front:'mf', left:'ml'};
  assert.equal(shapeBuild({state:'running', stages:at('running','pending'), masks}), null, 'not while the views are drawn');
  const next = shapeBuild({state:'running', stages:at('passed','pending'), masks});
  assert.deepEqual([next.front, next.side, next.elapsed], ['mf', 'ml', 0]);
  const running = shapeBuild({state:'running', stages:at('passed','running',{elapsed_seconds:42, typical_seconds:230}), masks});
  assert.deepEqual([running.elapsed, running.typical], [42, 230]);
  assert.equal(shapeBuild({state:'running', stages:at('passed','running'), images:{front:'vf'}}).front, 'vf', 'the front view stands in for its mask');
  assert.equal(shapeBuild({state:'running', stages:at('passed','running'), images:{front:'vf'}}).side, null);
  assert.equal(shapeBuild({state:'running', stages:at('passed','running')}), null, 'nothing to sample');
  assert.equal(shapeBuild({state:'running', stages:at('passed','running'), masks, draft_models:[{url:'d', raw:true, draft:true}]}), null, 'SAM 3D shape is in');
  assert.equal(shapeBuild({state:'running', stages:at('passed','passed'), masks, models:[{url:'s', n:0, raw:true}]}), null);
  for (const state of ['failed', 'canceled', 'stopped']) assert.equal(shapeBuild({state, stages:at('passed','running'), masks}), null, state);
});

test('models are labelled as SAM 3D raw output or HeroGen work', () => {
  assert.equal(modelLabel({n:0, raw:true, draft:true}), 'SAM 3D raw shape');
  assert.equal(modelLabel({n:0, raw:true}), 'SAM 3D raw shape');
  assert.equal(modelLabel({n:3}), 'HeroGen refined · version 3');
  assert.equal(modelLabel({n:null}), 'HeroGen model');
});

test('the build progress never fills before the shape and the tips cycle', () => {
  assert.equal(buildProgress(0, 230), 0);
  assert.ok(Math.abs(buildProgress(115, 230) - 0.5) < 1e-9);
  assert.ok(buildProgress(230, 230) < 0.95 && buildProgress(230, 230) > 0.8);
  assert.ok(buildProgress(5000, 230) <= 0.95);
  assert.ok(buildProgress(60, null) > 0 && buildProgress(1e6, undefined) <= 0.95);
  for (let i = 1; i < 100; i += 7) assert.ok(buildProgress(i, 100) >= buildProgress(i - 1, 100), 'never goes back');
  assert.equal(shapeTip(0), SHAPE_TIPS[0]); assert.equal(shapeTip(SHAPE_TIPS.length + 1), SHAPE_TIPS[1]); assert.equal(shapeTip(-1), SHAPE_TIPS.at(-1));
});

test('the figure of dots is sampled inside the front outline with depth from the side view', () => {
  // A mask image: white figure on black. Front: a 4-wide, 10-tall bar; side: 2 wide.
  const image = (w, h, inside, rgba = false) => {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4, on = inside(x, y);
      data.set(rgba ? [200, 50, 50, on ? 255 : 0] : on ? [255, 255, 255, 255] : [0, 0, 0, 255], o);
    }
    return figureMask(data, w, h);
  };
  const front = image(20, 20, (x, y) => x >= 8 && x < 12 && y >= 5 && y < 15);
  const side = image(20, 20, (x, y) => x >= 9 && x < 11 && y >= 5 && y < 15, true);
  assert.equal(front.count, 40); assert.equal(side.count, 20, 'a cut-out reads its opaque pixels');
  let seed = 1; const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const {positions, seeds} = sampleFigure(front, side, 3000, random);
  assert.equal(positions.length, 9000); assert.equal(seeds.length, 3000);
  let [minY, maxY, maxX, maxZ] = [Infinity, -Infinity, 0, 0];
  for (let i = 0; i < 3000; i++) {
    const [x, y, z] = positions.slice(i * 3, i * 3 + 3);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y); maxX = Math.max(maxX, Math.abs(x)); maxZ = Math.max(maxZ, Math.abs(z));
  }
  assert.ok(minY >= -0.5 && maxY <= 0.5 && maxY - minY > 0.95, 'height 1, centred');
  assert.ok(maxX <= 0.2 + 1e-6 && maxX > 0.18, 'x within the front outline (4 of 10 px)');
  assert.ok(maxZ <= 0.1 + 1e-6 && maxZ > 0.08, 'z within the side outline (2 of 10 px)');
  const slab = sampleFigure(front, null, 500, random);
  assert.ok(slab.positions.every((v, i) => i % 3 !== 2 || Math.abs(v) <= 0.12), 'a modest depth without a side view');
  assert.equal(sampleFigure(image(8, 8, () => false), null, 10), null);
});
