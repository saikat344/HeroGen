import test from 'node:test';
import assert from 'node:assert/strict';
import {fallbackArt} from '../herogen/service/static/error-art.js';

// Just enough of an <img> and its <figure>: events, src, dataset, complete and naturalWidth.
function fake({src = '/thumb.webp', full = '/art', complete = false, naturalWidth = 0} = {}) {
  const listeners = {};
  const figure = {hidden: false};
  const img = {
    dataset: full === null ? {} : {full}, complete, naturalWidth, _src: src,
    getAttribute(name) { return name === 'src' ? this._src : null; },
    set src(v) { this._src = v; },
    addEventListener(type, fn) { listeners[type] = fn; },
    fire(type) { listeners[type](); },
    closest: () => figure,
  };
  return {img, figure};
}

test('a failed thumbnail falls back to the full artwork', () => {
  const {img, figure} = fake();
  fallbackArt(img);
  img.fire('error');
  assert.equal(img._src, '/art');
  assert.equal(figure.hidden, false, 'the figure stays while the artwork loads');
});

test('the figure is hidden when the full artwork fails too', () => {
  const {img, figure} = fake();
  fallbackArt(img);
  img.fire('error');
  img.fire('error');
  assert.equal(img._src, '/art');
  assert.equal(figure.hidden, true);
});

test('with no other artwork to try the figure is hidden at once', () => {
  const noFull = fake({full: null});
  fallbackArt(noFull.img);
  noFull.img.fire('error');
  assert.equal(noFull.figure.hidden, true);
  const same = fake({src: '/art'});   // the thumbnail already is the full artwork
  fallbackArt(same.img);
  same.img.fire('error');
  assert.equal(same.figure.hidden, true);
});

test('a thumbnail that failed before the script ran is handled too', () => {
  const {img, figure} = fake({complete: true, naturalWidth: 0});
  fallbackArt(img);
  assert.equal(img._src, '/art');
  assert.equal(figure.hidden, false);
  const loaded = fake({complete: true, naturalWidth: 320});
  fallbackArt(loaded.img);
  assert.equal(loaded.img._src, '/thumb.webp');
  assert.equal(loaded.figure.hidden, false);
});
