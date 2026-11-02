// Apply the theme before CSS loads, including when storage is unavailable. Dark is the default for a first visit;
// the header toggle cycles dark -> light -> system and remembers the choice in this browser.
(() => {
  const root = document.documentElement;
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const key = 'hg-theme';
  const order = ['dark', 'light', 'system'];
  const names = {dark: 'Dark', light: 'Light', system: 'System'};
  const icons = {dark: 'moon', light: 'sun', system: 'monitor'};
  const valid = (value) => order.includes(value);
  let preference = 'dark';
  try { const saved = localStorage.getItem(key); if (valid(saved)) preference = saved; } catch { /* Keep the default. */ }

  function render(button) {
    const next = order[(order.indexOf(preference) + 1) % order.length];
    const label = `Theme: ${names[preference]}. Switch to ${names[next].toLowerCase()}.`;
    button.setAttribute('aria-label', label);
    button.title = label;
    button.dataset.preference = preference;
    button.querySelector('use')?.setAttribute('href', `/assets/icons/sprite.svg#i-${icons[preference]}`);
  }
  function apply() {
    root.dataset.theme = preference === 'system' ? (media.matches ? 'dark' : 'light') : preference;
    root.dataset.themePreference = preference;
    root.style.colorScheme = root.dataset.theme;
    document.dispatchEvent(new CustomEvent('hg:theme', {detail: root.dataset.theme}));
    document.querySelectorAll('[data-theme-toggle]').forEach(render);
  }
  function choose(value) {
    preference = valid(value) ? value : 'dark';
    try { localStorage.setItem(key, preference); } catch { /* Keep this tab usable without storage. */ }
    apply();
    document.dispatchEvent(new CustomEvent('hg:theme-change', {detail: preference}));
  }
  function toggleButton() {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn--ghost btn--icon theme-toggle';
    button.dataset.themeToggle = '';
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'icon'); svg.setAttribute('aria-hidden', 'true');
    svg.append(document.createElementNS('http://www.w3.org/2000/svg', 'use'));
    button.append(svg);
    return button;
  }

  apply();
  media.addEventListener('change', () => { if (preference === 'system') apply(); });
  window.addEventListener('storage', (event) => {
    if (event.key !== key && event.key !== null) return;
    preference = valid(event.newValue) ? event.newValue : 'dark'; apply();
  });
  document.addEventListener('DOMContentLoaded', () => {
    // Pages with the new header carry their own toggle; older headers (.top) get one appended.
    if (!document.querySelector('[data-theme-toggle]')) document.querySelector('.top')?.append(toggleButton());
    document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
      render(button);
      button.addEventListener('click', () => choose(order[(order.indexOf(preference) + 1) % order.length]));
    });
  });
  window.herogenTheme = {get preference() { return preference; }, choose};
})();
