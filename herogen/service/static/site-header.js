// Site header: the phone menu sheet, and which nav item is current on the single-page app.
(() => {
  const header = document.querySelector('.site-header');
  if (!header) return;
  const toggle = header.querySelector('.nav-toggle');
  const nav = header.querySelector('.nav');

  function setOpen(open, {focus = false} = {}) {
    header.classList.toggle('is-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    toggle.querySelector('use')?.setAttribute('href', `/assets/icons/sprite.svg#i-${open ? 'x' : 'menu'}`);
    if (open && focus) nav.querySelector('.nav__link')?.focus();
  }
  if (toggle && nav) {
    toggle.addEventListener('click', () => setOpen(!header.classList.contains('is-open'), {focus: true}));
    nav.addEventListener('click', (event) => { if (event.target.closest('.nav__link')) setOpen(false); });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || !header.classList.contains('is-open')) return;
      setOpen(false); toggle.focus();
    });
    document.addEventListener('click', (event) => {
      if (header.classList.contains('is-open') && !header.contains(event.target)) setOpen(false);
    });
    matchMedia('(min-width: 900px)').addEventListener('change', (event) => { if (event.matches) setOpen(false); });
  }

  // On "/", the landing makes "Create" current; the heroes grid and a hero's workspace (both inside #app) make
  // "Your heroes" current. #app stays visible on the landing for a signed-in visitor, so the landing is judged
  // by #entry.
  const app = document.querySelector('#app');
  const entry = document.querySelector('#entry');
  const create = document.querySelector('#workspaceNav');
  const heroes = document.querySelector('#heroesNav');
  if (app && create && heroes) {
    const mark = (link, current) => (current ? link.setAttribute('aria-current', 'page') : link.removeAttribute('aria-current'));
    const sync = () => { const inApp = !app.hidden && !(entry && !entry.hidden); mark(create, !inApp); mark(heroes, inApp); };
    const watch = new MutationObserver(sync);
    watch.observe(app, {attributes: true, attributeFilter: ['hidden']});
    if (entry) watch.observe(entry, {attributes: true, attributeFilter: ['hidden']});
    sync();
  }
})();
