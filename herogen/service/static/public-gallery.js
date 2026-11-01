export async function publicGallery() {
  const cards = document.querySelector('#publicGalleryCards'), status = document.querySelector('#publicGalleryStatus');
  try {
    const response = await fetch('/api/gallery', {credentials:'omit',cache:'no-store'});
    if (!response.ok) throw new Error('unavailable');
    const data = await response.json();
    if (!Array.isArray(data.items)) throw new Error('unavailable');
    cards.replaceChildren();
    for (const item of data.items) {
      const card = document.createElement('a'); card.className = 'public-character'; card.href = `/gallery/${encodeURIComponent(item.id)}`;
      const image = document.createElement('img'); image.src = item.assets.artwork.url; image.alt = `${item.title} — generated character artwork`; image.loading='lazy';
      const body = document.createElement('div'), title = document.createElement('h3'), label = document.createElement('p'), action=document.createElement('span');
      title.textContent=item.title; label.textContent='Previously generated with HeroGen'; action.textContent='Inspect in 3D ↗';
      body.append(title,label,action); card.append(image,body); cards.append(card);
    }
    status.textContent = data.items.length ? 'Select a character to inspect the actual 3D model and its starting sketch.' : 'No public results have been published yet.';
  } catch { status.textContent='The public gallery could not be loaded. Please reload to try again.'; }
}
