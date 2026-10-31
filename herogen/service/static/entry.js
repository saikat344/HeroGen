const IDEAS = {
  forest: 'A brave forest courier in an orange coat and a teal scarf, worn leather satchel, sturdy boots, warm expressive face. Full body, standing, plain background.',
  clockwork: 'A friendly clockwork gardener robot, round brass head, expressive eyes, moss-green overalls, tiny seedlings in a backpack and a watering can. Full body, standing, plain background.',
  moon: 'A moonlit mage with silver hair, layered midnight-blue robes, leafy shoulder cape, crescent pendant and embroidered stars. Full body, standing, plain background.'
};
export const TEMPLATES = [
  ['trail-keeper', 'Trail keeper', 'Sketch'], ['little-inventor', 'Little inventor', 'Artwork'],
  ['cloud-sailor', 'Cloud sailor', 'Sketch'], ['garden-guardian', 'Garden guardian', 'Artwork']
];
export function entryPage($, onChange) {
  let selected = null, preview = null, selection = 0;
  function setFile(file) {
    selection++;
    selected = file;
    if (preview) URL.revokeObjectURL(preview);
    preview = file ? URL.createObjectURL(file) : null;
    if (preview) $('#uploadPreview').src = preview; else $('#uploadPreview').removeAttribute('src');
    $('#uploadName').textContent = file ? (file.name || 'Your character image') : '';
    $('#sketchIn').hidden = !file;
    $('#promptIn').hidden = !!file;
    $('#promptCount').hidden = !!file;
    document.querySelector(`input[name=kind][value=${file ? 'sketch' : 'prompt'}]`).checked = true;
    $('#sketch').value = '';
    for (const b of document.querySelectorAll('[data-idea]')) b.setAttribute('aria-pressed', 'false');
    onChange();
  }
  $('#sketch').addEventListener('change', () => {
    const f = $('#sketch').files[0];
    if (f && !['image/png', 'image/jpeg', 'image/webp'].includes(f.type)) {
      $('#jobMsg').textContent = 'Choose a PNG, JPEG or WebP image.'; $('#sketch').value = ''; return;
    }
    setFile(f || null);
  });
  $('#removeUpload').addEventListener('click', () => { setFile(null); $('#prompt').focus(); });
  for (const b of document.querySelectorAll('[data-idea]')) b.addEventListener('click', () => {
    setFile(null); $('#prompt').value = IDEAS[b.dataset.idea]; onChange();
    b.setAttribute('aria-pressed', 'true'); $('#prompt').focus();
  });
  $('#prompt').addEventListener('input', () => {
    for (const b of document.querySelectorAll('[data-idea]')) b.setAttribute('aria-pressed', 'false');
  });
  $('#heroArtwork').addEventListener('error', () => { $('#heroArtwork').hidden = true; });
  for (const [id, name, kind] of TEMPLATES) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'template';
    button.setAttribute('aria-label', `Start from ${name}, ${kind.toLowerCase()}`);
    const image = document.createElement('img'); image.alt = ''; image.src = `/assets/art/${id}.png`;
    const title = document.createElement('b'); title.textContent = name;
    const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = kind;
    const arrow = document.createElement('span'); arrow.className = 'arrow'; arrow.textContent = '›'; arrow.setAttribute('aria-hidden', 'true');
    button.append(image, title, tag, arrow); button.hidden = true;
    image.addEventListener('load', () => { button.hidden = false; $('#gallery').hidden = false; });
    button.addEventListener('click', async () => {
      const current = ++selection;
      button.disabled = true; $('#templateMsg').textContent = `Selecting ${name}…`;
      try {
        const response = await fetch(image.src, { credentials: 'same-origin' });
        if (!response.ok) throw new Error('The image could not be loaded. Please try again.');
        const blob = await response.blob();
        if (current !== selection) return;
        setFile(new File([blob], `${id}.png`, { type: 'image/png' }));
        $('#name').value = name;
        $('#templateMsg').textContent = `${name} selected. Review it above, then create your hero.`;
        $('#entry').scrollIntoView({ block: 'start' }); $('#go').focus({ preventScroll: true });
      } catch (e) { if (current === selection) $('#templateMsg').textContent = e.message; }
      finally { button.disabled = false; }
    });
    $('#templates').append(button);
  }
  $('#howButton').addEventListener('click', () => $('#howDialog').showModal());
  $('#closeHow').addEventListener('click', () => $('#howDialog').close());
  return { setFile, file: () => selected || $('#sketch').files[0] || null };
}
