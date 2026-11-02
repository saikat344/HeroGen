import { track, trackOnce } from './analytics.js';

const IDEAS = {
  forest: 'A brave forest courier in an orange coat and a teal scarf, worn leather satchel, sturdy boots, warm expressive face. Full body, standing, plain background.',
  clockwork: 'A friendly clockwork gardener robot, round brass head, expressive eyes, moss-green overalls, tiny seedlings in a backpack and a watering can. Full body, standing, plain background.',
  moon: 'A moonlit mage with silver hair, layered midnight-blue robes, leafy shoulder cape, crescent pendant and embroidered stars. Full body, standing, plain background.'
};
// The two image modes send the same `sketch` input; they differ only in what the drop zone asks for.
const MODES = {
  sketch: { title: 'Drop your sketch here, or browse', hint: 'A pencil sketch or line drawing of one full-body character · PNG, JPEG or WebP' },
  upload: { title: 'Drop your artwork here, or browse', hint: 'Coloured art, an illustration or a photo of one full-body character · PNG, JPEG or WebP' }
};
const TYPES = ['image/png', 'image/jpeg', 'image/webp'];

export function entryPage($, onChange) {
  let selected = null, preview = null, mode = 'prompt';
  const form = $('#jobForm');

  // One message line under the composer. Errors carry an icon and the danger colour; progress notes do not.
  function say(text, tone = 'error') {
    const msg = $('#jobMsg');
    msg.textContent = text || '';
    if (text) msg.dataset.tone = tone; else delete msg.dataset.tone;
    form.classList.toggle('has-error', Boolean(text) && tone === 'error');
    // A refused create while generation is closed repeats the notice for screen readers and draws the eye to it.
    if (tone === 'closed') { const notice = $('#generationClosed'); notice.classList.remove('is-flash'); void notice.offsetWidth; notice.classList.add('is-flash'); }
  }
  function setMode(next, { focus = false } = {}) {
    mode = next;
    for (const b of document.querySelectorAll('[data-start]')) b.setAttribute('aria-pressed', String(b.dataset.start === next));
    const image = next !== 'prompt';
    document.querySelector(`input[name=kind][value=${image ? 'sketch' : 'prompt'}]`).checked = true;
    if (image && MODES[next]) { $('#dropTitle').textContent = MODES[next].title; $('#dropHint').textContent = MODES[next].hint; }
    $('#promptIn').hidden = image;
    $('#promptCount').hidden = image;
    $('#sketchIn').hidden = !(image && selected);
    $('#dropZone').hidden = !image || Boolean(selected);
    form.dataset.mode = next;
    if ($('#jobMsg').dataset.tone === 'error') say('');
    if (focus) (image ? (selected ? $('#removeUpload') : $('#browseUpload')) : $('#prompt')).focus();
  }
  function setFile(file) {
    selected = file;
    if (preview) URL.revokeObjectURL(preview);
    preview = file ? URL.createObjectURL(file) : null;
    if (preview) $('#uploadPreview').src = preview; else $('#uploadPreview').removeAttribute('src');
    $('#uploadName').textContent = file ? (file.name || 'Your character image') : '';
    $('#sketch').value = '';
    for (const b of document.querySelectorAll('[data-idea]')) b.setAttribute('aria-pressed', 'false');
    // Choosing a file switches to an image mode; removing it keeps the current image mode so the drop zone returns.
    setMode(file ? (mode === 'prompt' ? 'upload' : mode) : (mode === 'prompt' ? 'prompt' : mode));
    onChange();
  }
  function accept(file) {
    if (!file) return;
    if (!TYPES.includes(file.type)) { say('Choose a PNG, JPEG or WebP image.'); return; }
    say(''); setFile(file); track('upload_selected');
  }

  for (const b of document.querySelectorAll('[data-start]')) b.addEventListener('click', () => {
    if (b.dataset.start === 'prompt' && selected) setFile(null);
    setMode(b.dataset.start, { focus: true });
  });
  // Arrow keys move between the three modes, like a radio group.
  $('.composer__modes').addEventListener('keydown', event => {
    const buttons = [...document.querySelectorAll('[data-start]')], i = buttons.indexOf(document.activeElement);
    if (i < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (i + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].focus(); buttons[next].click();
  });
  $('#sketch').addEventListener('change', () => {
    const f = $('#sketch').files[0];
    if (f && !TYPES.includes(f.type)) { say('Choose a PNG, JPEG or WebP image.'); $('#sketch').value = ''; return; }
    if (f) say('');
    setFile(f || null);
    if (f) track('upload_selected');
  });
  $('#browseUpload').addEventListener('click', () => $('#sketch').click());
  const zone = $('#dropZone');
  for (const type of ['dragenter', 'dragover']) zone.addEventListener(type, event => {
    event.preventDefault(); zone.classList.add('is-over');
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
  });
  for (const type of ['dragleave', 'dragend']) zone.addEventListener(type, event => {
    if (!zone.contains(event.relatedTarget)) zone.classList.remove('is-over');
  });
  zone.addEventListener('drop', event => {
    event.preventDefault(); zone.classList.remove('is-over');
    accept(event.dataTransfer?.files?.[0]);
  });
  $('#removeUpload').addEventListener('click', () => { setFile(null); $('#browseUpload').focus(); });
  for (const b of document.querySelectorAll('[data-idea]')) b.addEventListener('click', () => {
    if (selected) setFile(null);
    setMode('prompt');
    $('#prompt').value = IDEAS[b.dataset.idea]; onChange();
    for (const other of document.querySelectorAll('[data-idea]')) other.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-pressed', 'true'); $('#prompt').focus();
    track('suggestion_click', { id: b.dataset.idea });
  });
  $('#prompt').addEventListener('input', () => {
    for (const b of document.querySelectorAll('[data-idea]')) b.setAttribute('aria-pressed', 'false');
    if ($('#jobMsg').dataset.tone === 'error' && $('#prompt').getAttribute('aria-invalid') !== 'true') say('');
    if (!$('#entry')?.hidden && $('#prompt').value.trim()) trackOnce('idea_typed', 'idea_typed');
  });
  $('#profile').addEventListener('change', () => { $('#targetSummary').textContent = `· ${$('#profile').selectedOptions[0]?.textContent || ''}`; });
  $('#howButton').addEventListener('click', () => {
    if ($('#entry').hidden) { $('#howDialog').showModal(); return; }
    $('#howHeading').tabIndex = -1; $('#howHeading').focus({preventScroll: true});
    $('#howItWorks').scrollIntoView({block: 'start'});
  });
  $('#closeHow').addEventListener('click', () => $('#howDialog').close());

  // The live-generation switch: when the service is not accepting heroes, say so with the server's own sentence.
  function generation(state) {
    const closed = state?.accepting === false;
    $('#generationClosed').hidden = !closed;
    if (closed) $('#generationSentence').textContent = state.sentence || 'Live generation is paused right now.';
    form.classList.toggle('is-closed', closed);
  }
  return { setFile, setMode, say, generation, file: () => selected || $('#sketch').files[0] || null };
}
