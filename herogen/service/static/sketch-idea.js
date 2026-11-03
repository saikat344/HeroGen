// "Suggest a sketch": put a sketch that already made a good hero in as the chosen image. The sketches come from a
// read-only folder the service lists at /api/sketch-ideas (each PNG at /sketch-ideas/<id>, hash-checked); with
// none listed the button stays hidden. Each one is used exactly as an upload would be: the same preview, type
// check and multipart `sketch` field (`entry.js`). An image the person chose themselves is replaced, never lost:
// one Undo brings it back until they choose or remove an image, or switch to Describe.
import { imageProblem } from './entry.js';

// The listed sketches in a random order (a Fisher-Yates shuffle), each one once; bad entries are dropped.
export function sketchQueue(data, random = Math.random) {
  const seen = new Set();
  const list = (Array.isArray(data?.items) ? data.items : []).filter((i) => typeof i?.id === 'string'
    && /^[a-z0-9][a-z0-9-]{0,63}$/.test(i.id) && !seen.has(i.id) && seen.add(i.id)
    && typeof i.title === 'string' && i.title.trim() && typeof i.url === 'string' && i.url.startsWith('/sketch-ideas/'))
    .map((i) => ({ id: i.id, title: i.title.trim(), url: i.url }));
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}

// The next sketch to show, taken from `queue` (changed in place). Once every sketch has been shown, a new shuffle
// starts, never with the one just shown, so clicking on keeps giving a different sketch.
export function nextSketch(queue, data, last = null, random = Math.random) {
  if (!queue.length) {
    queue.push(...sketchQueue(data, random));
    if (queue.length > 1 && queue[0].id === last) queue.push(queue.shift());
  }
  return queue.shift() || null;
}

// The image an Undo brings back after a suggestion replaces `current`: the person's own image when that is what
// is being replaced; when it is an earlier suggestion (`ours`), whatever Undo already held.
export function undoPlan(current, ours, held = null) {
  return current && current !== ours ? current : held;
}

// What a successful suggestion says to screen readers.
export function suggestedNote(title, undo) {
  return `Suggested sketch added: ${title}.${undo ? ' Undo brings your image back.' : ''}`;
}

const loadList = () => fetch('/api/sketch-ideas', { credentials: 'omit', cache: 'no-store' })
  .then((r) => (r.ok ? r.json() : null)).catch(() => null);

export function sketchButton($, { entry, track = () => {}, list = loadList() }) {
  const tools = $('#sketchTools'), button = $('#suggestSketch'), undoButton = $('#undoSketch'), status = $('#sketchStatus');
  const queue = [], files = new Map();   // id -> Promise<File>, fetched once
  let data = null, ours = null, held = null, last = null, busy = false, applying = false;

  function offerUndo(file) { held = file || null; undoButton.hidden = held === null; }
  function announce(text) { status.textContent = ''; if (text) requestAnimationFrame(() => { status.textContent = text; }); }
  function fetchFile(item) {
    if (!files.has(item.id)) {
      const pending = fetch(item.url, { credentials: 'omit' }).then(async (r) => {
        if (!r.ok) throw new Error('unavailable');
        return new File([await r.blob()], `${item.title}.png`, { type: 'image/png' });
      });
      pending.catch(() => files.delete(item.id));
      files.set(item.id, pending);
    }
    return files.get(item.id);
  }
  function use(file) {
    applying = true;
    try { entry.setFile(file); if (file && entry.mode() === 'upload') entry.setMode('sketch'); } finally { applying = false; }
  }

  // Any image change we did not make (a new upload, Remove image, Describe) makes the input the person's again.
  entry.onFile(() => { if (!applying) { ours = null; offerUndo(null); } });

  list.then((result) => {
    data = result;
    tools.hidden = !sketchQueue(data).length;
  });

  button.addEventListener('click', async () => {
    if (busy || !data) return;
    const item = nextSketch(queue, data, last);
    if (!item) return;
    busy = true; button.disabled = true; button.setAttribute('aria-busy', 'true'); announce('Finding a sketch…');
    if ($('#jobMsg').dataset.tone === 'error') entry.say('');
    try {
      const file = await fetchFile(item);
      const problem = imageProblem(file);
      if (problem) throw new Error(problem);
      const undo = undoPlan(entry.file(), ours, held);
      use(file); ours = file; last = item.id; offerUndo(undo);
      announce(suggestedNote(item.title, undo));
      track('suggestion_click', { id: 'proven-sketch' });
      if (queue[0]) fetchFile(queue[0]).catch(() => {});   // the next click is then instant
    } catch {
      announce(''); entry.say('That sketch could not be loaded. Try again for another one.');
    } finally {
      busy = false; button.disabled = false; button.removeAttribute('aria-busy');
    }
  });
  undoButton.addEventListener('click', () => {
    if (!held) return;
    const file = held;
    ours = null; use(file); offerUndo(null); announce('Your image is back.');
    $('#removeUpload').focus();
  });
  return { reset: () => { ours = null; offerUndo(null); } };
}
