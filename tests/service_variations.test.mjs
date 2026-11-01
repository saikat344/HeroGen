import test from 'node:test';
import assert from 'node:assert/strict';
import { expectedTime, versionAction, versionLabel, versionMetrics, versionRows } from '../herogen/service/static/variations.js';

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
  assert.equal(expectedTime({stages},'texture'),'Usually about 4 min');
  assert.equal(expectedTime({stages},'turnaround'),'Usually about 6 min');
  stages[2].typical_seconds=null;
  assert.equal(expectedTime({stages},'texture'),'Usually about 4 min');
  assert.equal(expectedTime({stages},'turnaround'),'Time estimate needs more real runs.');
});

test('branch actions use known submissions and never invent historical provenance', () => {
  assert.equal(versionAction({}, {}, 'Retry'),'Retry');
  assert.equal(versionAction({}, {}, 'Refine'),'Refine');
  assert.equal(versionAction({note:'Blue coat'}, {}),'Refine');
  assert.equal(versionAction({note:'Blue coat'}, {note:'Blue coat'}),'Retry / Refine');
  assert.equal(versionAction({}, {}),'Retry / Refine');
});
