// The minimal viewer, used until the Studio interface is deployed: it reads the same Lab API the Studio reads
// (/api/workspace, /api/runs/<root>/<run>, /runs/<root>/<run>/<path>) and opens #run=<root>/<run>.
const $ = (s) => document.querySelector(s);
const enc = (p) => p.split('/').map(encodeURIComponent).join('/');
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };

async function getJson(url) {
  const r = await fetch(url, { cache: 'no-store', credentials: 'same-origin' });
  if (!r.ok) throw new Error(r.status === 401 ? 'Sign in first.' : `${url}: ${r.status}`);
  return r.json();
}

async function open(id) {
  const run = await getJson(`/api/runs/${enc(id)}`);
  $('#msg').textContent = run.name;
  $('#stages').replaceChildren(...run.stages.map((s) => {
    const tr = el('tr');
    tr.append(el('td', s.name), el('td', s.status), el('td', typeof s.seconds === 'number' ? `${s.seconds.toFixed(1)} s` : ''),
      el('td', typeof s.cost_usd === 'number' ? `$${s.cost_usd.toFixed(4)}` : ''));
    return tr;
  }));
  $('#images').replaceChildren(...Object.entries(run.images || {}).map(([k, f]) => {
    const fig = el('figure'); const img = el('img'); img.src = `/runs/${enc(id)}/${enc(f.path)}`; img.alt = k;
    fig.append(img, el('figcaption', k)); return fig;
  }));
  const glb = (run.deliverables || []).find((d) => d.status === 'built' && d.path && d.path.endsWith('high.glb'));
  if (glb) (await import('/assets/preview.js')).show($('#preview'), `/runs/${enc(id)}/${enc(glb.path)}`);
}

async function boot() {
  let ws;
  try { ws = await getJson('/api/workspace'); } catch (e) { $('#msg').textContent = e.message; return; }
  const ids = Object.keys(ws.runs);
  if (!ids.length) { $('#msg').textContent = 'No heroes yet.'; return; }
  const sel = $('#run');
  sel.replaceChildren(...ids.map((id) => { const o = el('option', id.split('/')[1]); o.value = id; return o; }));
  const want = new URLSearchParams(location.hash.slice(1)).get('run');
  sel.value = want && ws.runs[want] ? want : ids[0];
  sel.addEventListener('change', () => { location.hash = `run=${sel.value}`; });
  window.addEventListener('hashchange', () => { const id = new URLSearchParams(location.hash.slice(1)).get('run'); if (id && ws.runs[id]) { sel.value = id; open(id); } });
  open(sel.value).catch((e) => { $('#msg').textContent = e.message; });
}

boot();
