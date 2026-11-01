// Apply the preference before CSS loads, including when storage is unavailable.
(() => {
  const root = document.documentElement;
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const key = 'hg-theme';
  let preference = 'system';
  try { const saved = localStorage.getItem(key); if (['light', 'dark', 'system'].includes(saved)) preference = saved; } catch { /* Use the system preference. */ }
  function apply() {
    root.dataset.theme = preference === 'system' ? (media.matches ? 'dark' : 'light') : preference;
    root.style.colorScheme = root.dataset.theme;
    document.dispatchEvent(new CustomEvent('hg:theme', {detail: root.dataset.theme}));
    const control = document.querySelector('#themePreference');
    if (control) control.value = preference;
  }
  apply();
  media.addEventListener('change', () => { if (preference === 'system') apply(); });
  window.addEventListener('storage', (event) => {
    if (event.key !== key && event.key !== null) return;
    preference = ['light', 'dark', 'system'].includes(event.newValue) ? event.newValue : 'system'; apply();
  });
  document.addEventListener('DOMContentLoaded', () => {
    const header = document.querySelector('.top');
    if (!header) return;
    const label = document.createElement('label'); label.className = 'theme-control'; label.textContent = 'Theme';
    const select = document.createElement('select'); select.id = 'themePreference'; select.setAttribute('aria-label', 'Color theme');
    for (const [value, text] of [['system', 'System'], ['light', 'Light'], ['dark', 'Dark']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = text; select.append(option);
    }
    select.value = preference;
    select.addEventListener('change', () => {
      preference = select.value;
      try { localStorage.setItem(key, preference); } catch { /* Keep this tab usable without storage. */ }
      apply();
      document.dispatchEvent(new CustomEvent('hg:theme-change', {detail: preference}));
    });
    label.append(select); header.append(label);
  });
})();
