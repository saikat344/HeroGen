// Tab in an empty text box takes its grey example text, like an editor's inline suggestion. A box that already
// holds text, or Shift+Tab, keeps the browser's own focus move.
export const placeholderText = (placeholder) => String(placeholder || '').trim().replace(/\s*(…|\.\.\.)$/, '');

export function acceptPlaceholderOnTab(field) {
  field.addEventListener('keydown', (event) => {
    if (event.key !== 'Tab' || event.shiftKey || event.altKey || event.ctrlKey || event.metaKey || event.isComposing) return;
    const text = placeholderText(field.placeholder);
    if (field.value.trim() || !text) return;
    event.preventDefault();
    field.value = text;
    field.setSelectionRange(text.length, text.length);
    field.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
