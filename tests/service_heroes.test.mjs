import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultName, finishedNote, heroStatus, needsDetail, relativeTime, rootHeroes, stageProgress, summaryLine, thumbnailOf}
  from '../herogen/service/static/heroes.js';

const job = (id, extra = {}) => ({id, kind: 'prompt', name: id, state: 'succeeded', created_at: 100, finished_at: 160, parent_job: null, ...extra});

test('only root heroes are listed, newest first, with variations counted at the root', () => {
  const jobs = [job('a', {created_at: 10}), job('b', {created_at: 30}), job('a1', {parent_job: 'a', created_at: 40}),
    job('a2', {parent_job: 'a1', created_at: 50}), job('orphan', {parent_job: 'gone', created_at: 5})];
  const roots = rootHeroes(jobs);
  assert.deepEqual(roots.map(({job: j, variations}) => [j.id, variations]), [['b', 0], ['a', 2], ['orphan', 0]]);
  assert.equal(summaryLine(roots), '3 heroes');
  assert.equal(summaryLine([]), '');
});

test('the status pill never claims Roblox ready without the detail saying so', () => {
  assert.equal(heroStatus(job('a'), null).label, 'Ready');
  assert.equal(heroStatus(job('a'), {roblox_state: 'unavailable'}).label, 'Ready');
  assert.equal(heroStatus(job('a'), {roblox_state: 'ready'}).label, 'Roblox ready');
  assert.equal(heroStatus(job('a'), null).note, 'Finished in 60 s');
  assert.equal(heroStatus(job('a', {state: 'failed'}), null).label, 'Needs attention');
  assert.equal(heroStatus(job('a', {state: 'rejected'}), null).tone, 'fail');
  assert.equal(heroStatus(job('a', {state: 'canceled'}), null).label, 'Cancelled');
});

test('the finish note waits for the Roblox steps before giving the full time', () => {
  const secs = (v) => `${Math.round(v)} s`;
  const ready = {state: 'succeeded', created_at: 100, started_at: 100, finished_at: 628};
  assert.equal(finishedNote({...ready, export_state: 'running'}, secs), 'Model ready in 528 s · Roblox files in progress');
  assert.equal(finishedNote({...ready, export_state: 'ready', export_seconds: 709}, secs), 'Finished in 1237 s');
  assert.equal(finishedNote({...ready, export_state: 'unavailable', export_seconds: 12}, secs), 'Finished in 540 s');
  assert.equal(finishedNote(ready, secs), 'Finished in 528 s');
  assert.equal(heroStatus(job('a'), {export_state: 'running'}).note, 'Model ready in 60 s · Roblox files in progress');
  assert.equal(heroStatus(job('a'), {export_state: 'ready', export_seconds: 60, roblox_state: 'ready'}).note, 'Finished in 2 min');
});

test('queued and running heroes explain where they are', () => {
  assert.equal(heroStatus(job('a', {state: 'queued'}), null).note, 'Starts when a worker is free');
  assert.equal(heroStatus(job('a', {state: 'queued', queued_ahead: 2}), null).note, '2 heroes ahead of you');
  assert.equal(heroStatus(job('a', {state: 'queued', queued_ahead: 0}), null).note, 'Next to start');
  const detail = {stages: [{label: 'Artwork', state: 'passed'}, {label: 'Views', state: 'running'}, {label: 'Shape', state: 'pending'},
    {label: 'Side', state: 'pending', side: true}]};
  const running = heroStatus(job('a', {state: 'running'}), detail);
  assert.equal(running.note, '1 of 3 stages complete');
  assert.deepEqual(running.progress.segments.map((s) => s.kind), ['done', 'run', 'wait']);
  assert.equal(running.progress.now, 'Views');
  assert.equal(heroStatus(job('a', {state: 'running', stage_label: 'Views'}), null).note, 'Now: Views');
  assert.equal(stageProgress({stages: []}), null);
});

test('thumbnails prefer the server thumbnail, then the job images, then nothing', () => {
  assert.equal(thumbnailOf(job('a', {thumbnail_url: '/t.png'}), {images: {cover: '/c.png'}}), '/t.png');
  assert.equal(thumbnailOf(job('a'), {images: {sketch: '/s.png', cover: '/c.png'}}), '/c.png');
  assert.equal(thumbnailOf(job('a'), {images: {sketch: '/s.png'}}), '/s.png');
  assert.equal(thumbnailOf(job('a'), {images: {}}), null);
  assert.equal(thumbnailOf(job('a'), null), null);
});

test('names fall back to the prompt, then a plain label', () => {
  assert.equal(defaultName(job('a', {name: ''}), {prompt: ' A mage '}), 'A mage');
  assert.equal(defaultName(job('a', {name: ''}), null), 'Described hero');
  assert.equal(defaultName(job('a', {name: '', kind: 'sketch'}), null), 'Sketched hero');
});

test('only the cards that need more than the summary cost a request', () => {
  assert.equal(needsDetail(job('a', {state: 'running'}), 0), true);
  assert.equal(needsDetail(job('a', {state: 'failed', thumbnail_url: '/t.png'}), 0), false);
  assert.equal(needsDetail(job('a', {state: 'failed'}), 0), true);
  assert.equal(needsDetail(job('a', {state: 'failed'}), 12), false);
});

test('relative times are short and honest', () => {
  const now = 1_000_000;
  assert.equal(relativeTime(now - 5, now), 'Just now');
  assert.equal(relativeTime(now - 300, now), '5 min ago');
  assert.equal(relativeTime(now - 7200, now), '2 h ago');
  assert.equal(relativeTime(now - 90000, now), 'Yesterday');
  assert.equal(relativeTime(now - 3 * 86400, now), '3 days ago');
  assert.equal(relativeTime(0, now), '');
});
