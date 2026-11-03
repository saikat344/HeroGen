import test from 'node:test';
import assert from 'node:assert/strict';
import { reasonText } from '../herogen/service/static/buildlog-copy.js';

test('a content-filter refusal reads as a refusal, not as a failed check', () => {
  assert.equal(reasonText('content_refused'), 'the image model refused this view (content filter)');
  assert.equal(reasonText('something_new'), 'it did not pass the checks');
});
