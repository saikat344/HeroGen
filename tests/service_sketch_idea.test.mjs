import test from 'node:test';
import assert from 'node:assert/strict';
import {nextSketch, sketchQueue, suggestedNote, undoPlan} from '../herogen/service/static/sketch-idea.js';

const item = (id) => ({id, title: ` ${id} hero `, url: `/sketch-ideas/${id}?v=abc`});
const DATA = {items: ['a', 'b', 'c', 'd'].map(item)};

test('the listed sketches come once each, shuffled, and bad entries are dropped', () => {
  const data = {items: [item('a'), {...item('b'), url: 'https://elsewhere.example/b.png'}, {...item('c'), title: '  '},
    {id: '../x', title: 'x', url: '/sketch-ideas/x'}, item('a'), null, item('d')]};
  assert.deepEqual(sketchQueue(data, () => 0.99).map((i) => i.id), ['a', 'd']);
  assert.deepEqual(sketchQueue(data, () => 0).map((i) => i.id), ['d', 'a']);
  assert.deepEqual(sketchQueue(data, () => 0.99)[0], {id: 'a', title: 'a hero', url: '/sketch-ideas/a?v=abc'});
  for (const bad of [null, {}, {items: 'x'}, {items: []}]) assert.deepEqual(sketchQueue(bad), []);
});

test('every sketch is shown before any repeats, and a new round never opens with the last one', () => {
  for (let seed = 0; seed < 20; seed++) {
    let n = seed;
    const random = () => ((n = (n * 9301 + 49297) % 233280) / 233280);
    const queue = [], shown = [];
    let last = null;
    for (let i = 0; i < 12; i++) { const next = nextSketch(queue, DATA, last, random); shown.push(next.id); last = next.id; }
    for (const round of [shown.slice(0, 4), shown.slice(4, 8), shown.slice(8, 12)]) {
      assert.deepEqual([...round].sort(), ['a', 'b', 'c', 'd']);
    }
    for (let i = 1; i < shown.length; i++) assert.notEqual(shown[i], shown[i - 1]);
  }
  assert.equal(nextSketch([], {items: []}), null);
  assert.equal(nextSketch([], {items: [item('a')]}, 'a').id, 'a');   // one sketch: it simply comes again
});

test("Undo keeps the person's own image across several suggestions", () => {
  const theirs = {name: 'mine.png'}, first = {name: 'a.png'}, second = {name: 'b.png'};
  assert.equal(undoPlan(null, null, null), null);              // nothing chosen: nothing to undo
  assert.equal(undoPlan(theirs, null, null), theirs);         // their upload is kept for Undo
  assert.equal(undoPlan(first, first, theirs), theirs);       // replacing our own suggestion keeps theirs
  assert.equal(undoPlan(second, second, null), null);         // suggestions only: still nothing of theirs
  assert.equal(undoPlan(theirs, first, null), theirs);
});

test('screen readers hear the title, and whether Undo is offered', () => {
  assert.equal(suggestedNote('Iron Warden', null), 'Suggested sketch added: Iron Warden.');
  assert.match(suggestedNote('Iron Warden', {}), /Undo brings your image back\.$/);
});
