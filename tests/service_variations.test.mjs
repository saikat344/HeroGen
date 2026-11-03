import test from 'node:test';
import assert from 'node:assert/strict';
import { expectedTime, variationTree, versionAction, versionLabel, versionMetrics, versionRows, versionSide } from '../herogen/service/static/variations.js';

test('version tree preserves parent depth, siblings and children of deleted ancestors', () => {
  const versions = [{id:'root'}, {id:'leaf',parent_job:'child'}, {id:'child',parent_job:'root'},
    {id:'sibling',parent_job:'root'}, {id:'orphan',parent_job:'deleted'}];
  assert.deepEqual(versionRows(versions).map(v=>[v.id,v.depth]),
    [['root',0],['child',1],['leaf',2],['sibling',1],['orphan',0]]);
  assert.equal(versionRows([{id:'a',parent_job:'b'},{id:'b',parent_job:'a'}]).length,2);
});
test('version labels and measurements preserve notes and distinguish missing measurements', () => {
  assert.equal(versionLabel({id:'root'}),'Original texture');
  assert.equal(versionLabel({parent_job:'root',note:'Weathered brass'}),'Weathered brass');
  assert.equal(versionMetrics({seconds:228,cost_usd:.12}),'3.8 min · $0.12');
  assert.equal(versionMetrics({seconds:1.1,cost_usd:0}),'1.1 s · $0.00');
  assert.equal(versionMetrics({}),'Time pending · Cost pending');
});
test('expected time uses measured medians for the continued stages and never fills missing values', () => {
  const stages = ['1','2','3','3f','4','6','5'].map(key=>({key,typical_seconds:60}));
  // Only up to what a game engine needs: the 3D model, or the new texture; the Roblox steps run after it.
  assert.equal(expectedTime({stages},'texture'),'Usually about 60 s to the new texture');
  assert.equal(expectedTime({stages},'turnaround'),'Usually about 2 min to the 3D model');
  assert.equal(expectedTime({stages},'cover'),'Usually about 3 min to the 3D model');
  stages[2].typical_seconds=null;
  assert.equal(expectedTime({stages},'texture'),'Usually about 60 s to the new texture');
  assert.equal(expectedTime({stages},'turnaround'),'Time estimate needs more real runs.');
});

test('branch actions use known submissions and never invent historical provenance', () => {
  assert.equal(versionAction({}, {}, 'Retry'),'Retry');
  assert.equal(versionAction({}, {}, 'Refine'),'Refine');
  assert.equal(versionAction({note:'Blue coat'}, {}),'Refine');
  assert.equal(versionAction({note:'Blue coat'}, {note:'Blue coat'}),'Retry / Refine');
  assert.equal(versionAction({}, {}),'Retry / Refine');
});

test('tree sides put new notes on the Refine side and re-runs on the Retry side', () => {
  assert.equal(versionSide({note:'Weathered brass'}, {}), 'refine');
  assert.equal(versionSide({note:'Blue coat'}, {note:'Blue coat'}), 'retry');
  assert.equal(versionSide({}, {}), 'retry');
  assert.equal(versionSide({}, {}, 'Refine'), 'refine');
  const versions = [{id:'root'}, {id:'a',parent_job:'root',note:'Weathered brass'}, {id:'b',parent_job:'root'},
    {id:'c',parent_job:'root',note:'Moss green'}, {id:'d',parent_job:'a',note:'Weathered brass'}, {id:'orphan',parent_job:'gone'}];
  const [root, orphan] = variationTree(versions);
  assert.equal(root.version.id, 'root');
  assert.deepEqual(root.refine.map(n => n.version.id), ['a', 'c']);
  assert.deepEqual(root.retry.map(n => n.version.id), ['b']);
  assert.deepEqual(root.refine[0].retry.map(n => [n.version.id, n.side]), [['d', 'retry']]);
  assert.equal(orphan.version.id, 'orphan');
  assert.equal(variationTree([{id:'x',parent_job:'y'},{id:'y',parent_job:'x'}]).length, 1, 'cycles never repeat a version');
});
