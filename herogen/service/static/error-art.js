// The hero on the 404 page: when its small thumbnail fails to load, try the full artwork named in data-full,
// and when that fails too, hide the whole figure so the page shows no broken image.
export function fallbackArt(img, figure = img.closest('figure') || img) {
  let tried = false;
  const fail = () => {
    const full = img.dataset.full;
    if (full && !tried && img.getAttribute('src') !== full) { tried = true; img.src = full; return; }
    figure.hidden = true;
  };
  img.addEventListener('error', fail);
  // The failure may have happened before this module ran (it loads after the page's markup).
  if (img.complete && img.naturalWidth === 0 && img.getAttribute('src')) fail();
}

if (typeof document !== 'undefined') {
  const img = document.querySelector('.error-art img');
  if (img) fallbackArt(img);
}
