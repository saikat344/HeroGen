// "Suggest an idea": put a hero description in the Describe box. The first ideas come instantly, in a random order,
// from descriptions that already made good heroes (/assets/proven-ideas.json); once those are used up, HeroGen asks
// Muse Spark for a new one (POST /api/ideas).
// Text the person already wrote is replaced, never lost: one Undo puts it back until they type or ask again.
// Failures show one short friendly line by error code; the server's or model's own text is never shown.

const FAILURES = {
  idea_refused: "That idea didn't make it through. Try again for a new one.",
  ideas_unavailable: 'Idea suggestions are not available right now. Describe your own hero.',
  generation_paused: 'Idea suggestions are paused right now. Describe your own hero.',
  generation_budget_exhausted: 'Idea suggestions are paused right now. Describe your own hero.',
  rate_limited: "That's a lot of ideas! Wait a moment, then try again.",
};
const FALLBACK = 'No idea came back this time. Try again in a moment.';
const OFFLINE = 'HeroGen could not be reached. Check your connection and try again.';

// The line the composer shows for a failed request.
export function ideaFailure(err) {
  if (err instanceof TypeError) return OFFLINE;
  return FAILURES[err?.code] || FALLBACK;
}

// What a new idea does to the box: its text, and the text an Undo restores (null when there was nothing of note).
export function fillPlan(current, idea) {
  const before = typeof current === 'string' ? current : '';
  return { text: idea, undo: before.trim() && before !== idea ? before : null };
}

// What a successful fill says to screen readers.
export function filledNote(plan) {
  return plan.undo ? 'Idea added to the description. Undo puts your text back.' : 'Idea added to the description.';
}

// The proven ideas in a random order (a Fisher-Yates shuffle), each one shown once.
export function provenQueue(data, random = Math.random) {
  const list = (Array.isArray(data?.ideas) ? data.ideas : [])
    .map((i) => (typeof i?.text === 'string' ? i.text.trim() : '')).filter((t) => t && t.length <= 300);
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}

const loadProven = () => fetch('/assets/proven-ideas.json').then((r) => (r.ok ? r.json() : null))
  .then((data) => provenQueue(data)).catch(() => []);

export function ideaButton($, { request, say, track = () => {}, proven = loadProven() }) {
  const prompt = $('#prompt'), button = $('#suggestIdea'), undoButton = $('#undoIdea'), status = $('#ideaStatus');
  let undoText = null, busy = false;

  // The same path as the "Need a spark?" chips, as one input event so the count, the chips and the message follow.
  function setText(text) {
    prompt.value = text;
    prompt.dispatchEvent(new InputEvent('input', { inputType: 'insertReplacementText' }));
    prompt.focus();
    prompt.setSelectionRange(prompt.value.length, prompt.value.length);
  }
  function offerUndo(text) { undoText = text; undoButton.hidden = text === null; }
  function announce(text) { status.textContent = ''; if (text) requestAnimationFrame(() => { status.textContent = text; }); }

  button.addEventListener('click', async () => {
    if (busy) return;
    busy = true; button.disabled = true; button.setAttribute('aria-busy', 'true'); announce('Finding an idea…');
    if ($('#jobMsg').dataset.tone === 'error') say('');
    try {
      const queued = (await proven).shift();
      const { idea } = queued ? { idea: queued } : await request();
      if (typeof idea !== 'string' || !idea.trim()) throw new Error('empty');
      const plan = fillPlan(prompt.value, idea);
      setText(plan.text); offerUndo(plan.undo); announce(filledNote(plan));
      track('suggestion_click', { id: queued ? 'proven' : 'muse' });
    } catch (e) {
      announce(''); say(ideaFailure(e));
    } finally {
      busy = false; button.disabled = false; button.removeAttribute('aria-busy');
    }
  });
  undoButton.addEventListener('click', () => {
    if (undoText === null) return;
    const text = undoText;
    offerUndo(null); setText(text); announce('Your description is back.');
  });
  // Typing (anything but our own fills) makes the box the person's again, so the Undo goes away.
  prompt.addEventListener('input', (event) => { if (event.inputType !== 'insertReplacementText') offerUndo(null); });
  return { reset: () => offerUndo(null) };
}
