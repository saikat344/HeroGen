import test from 'node:test';
import assert from 'node:assert/strict';
import { acceptPlaceholderOnTab, placeholderText } from '../herogen/service/static/placeholder-tab.js';

function field(placeholder, value = '') {
  const f = { placeholder, value, events: [], listeners: {},
    addEventListener(type, fn) { this.listeners[type] = fn; },
    setSelectionRange(a, b) { this.selection = [a, b]; },
    dispatchEvent(e) { this.events.push(e.type); } };
  acceptPlaceholderOnTab(f);
  return f;
}
const press = (f, extra = {}) => {
  const e = { key: 'Tab', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
  f.listeners.keydown(e);
  return e;
};

test('the example text drops a trailing ellipsis', () => {
  assert.equal(placeholderText('A forest courier with an orange coat…'), 'A forest courier with an orange coat');
  assert.equal(placeholderText('Keep the face unchanged.'), 'Keep the face unchanged.');
  assert.equal(placeholderText(''), '');
});

test('Tab in an empty box takes the example text and stays in the box', () => {
  const f = field('Add a subtle green patina to the brass.');
  const e = press(f);
  assert.equal(f.value, 'Add a subtle green patina to the brass.');
  assert.ok(e.defaultPrevented);
  assert.deepEqual(f.selection, [39, 39]);
  assert.deepEqual(f.events, ['input'], 'the character count updates');
});

test('Tab moves focus as usual when the box has text, or with Shift', () => {
  const typed = field('Example', 'my own note');
  assert.equal(press(typed).defaultPrevented, false);
  assert.equal(typed.value, 'my own note');
  const shifted = field('Example');
  assert.equal(press(shifted, { shiftKey: true }).defaultPrevented, false);
  assert.equal(shifted.value, '');
});
