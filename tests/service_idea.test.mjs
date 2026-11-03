import test from 'node:test';
import assert from 'node:assert/strict';
import {fillPlan, filledNote, ideaFailure} from '../herogen/service/static/idea.js';

const IDEA = 'A sleepy cloud shepherd in a white wool poncho. Full body, standing, plain background.';

test('an empty box takes the idea with nothing to undo', () => {
  for (const current of ['', '   ', undefined, null]) {
    assert.deepEqual(fillPlan(current, IDEA), {text: IDEA, undo: null});
  }
  assert.equal(filledNote(fillPlan('', IDEA)), 'Idea added to the description.');
});

test("the person's own text is replaced but kept for one Undo", () => {
  const plan = fillPlan('A knight with a  blue cape', IDEA);
  assert.deepEqual(plan, {text: IDEA, undo: 'A knight with a  blue cape'});
  assert.match(filledNote(plan), /Undo puts your text back/);
});

test('the same idea again offers no undo', () => {
  assert.equal(fillPlan(IDEA, IDEA).undo, null);
});

test('failures are short fixed lines by code, never the server text', () => {
  const err = (code, message = 'model said: secret brief') => Object.assign(new Error(message), {code});
  assert.equal(ideaFailure(err('idea_refused')), "That idea didn't make it through. Try again for a new one.");
  assert.equal(ideaFailure(err('rate_limited')), "That's a lot of ideas! Wait a moment, then try again.");
  assert.match(ideaFailure(err('generation_paused')), /paused/);
  assert.match(ideaFailure(err('generation_budget_exhausted')), /paused/);
  assert.match(ideaFailure(err('ideas_unavailable')), /not available/);
  for (const e of [err('idea_failed'), err(undefined), err('internal'), new Error('empty'), null]) {
    assert.equal(ideaFailure(e), 'No idea came back this time. Try again in a moment.');
  }
  assert.match(ideaFailure(new TypeError('Failed to fetch')), /could not be reached/);
  for (const code of ['idea_refused', 'idea_failed', 'rate_limited', 'x']) assert.doesNotMatch(ideaFailure(err(code)), /secret/);
});

test('the proven ideas come once each, in a shuffled order, and bad entries are dropped', async () => {
  const { provenQueue } = await import('../herogen/service/static/idea.js');
  const data = {ideas: [{text: 'A'}, {text: ' B '}, {text: ''}, {text: 'x'.repeat(301)}, {}, {text: 'C'}]};
  const q = provenQueue(data, () => 0);
  assert.deepEqual([...q].sort(), ['A', 'B', 'C']);
  assert.notDeepEqual(q, ['A', 'B', 'C'], 'shuffled');
  assert.deepEqual(provenQueue(null), []);
});
