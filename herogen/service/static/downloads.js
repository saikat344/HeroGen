// The download panel under the result: the hero's main call to action once it is finished. Roblox first when its
// files are ready, the GLB beside it; while the Roblox files are prepared the GLB stays available and the Roblox
// button shows its progress; when they are unavailable the panel says why and the GLB leads. Server values reach the
// page only through textContent and attributes.

const SVG = 'http://www.w3.org/2000/svg';
export const ROBLOX_HINT = 'R15 rig and a starter place for Roblox Studio';
export const GLB_HINT = 'Standard 3D file for Blender, Unity, Unreal and other tools';
const BUILDING = new Set(['queued', 'running']);

// What the panel shows for a job: null when there is nothing to offer yet (still building with no model, or an input
// that was not accepted). `state` is one of ready | exporting | building | unavailable.
//   roblox: {url} when it can be downloaded, {busy: true} while its files are prepared, {waiting: true} while the
//           hero is still being built, null when it is unavailable (the note says why)
//   glb:    {url, primary} when it can be downloaded, else null
export function downloadState(job) {
  if (!job || job.input_rejected) return null;
  const glbInfo = job.downloads?.glb, rbxInfo = job.downloads?.roblox;
  const glbUrl = glbInfo?.available && glbInfo.url ? glbInfo.url : null;
  const glb = (primary) => (glbUrl ? { url: glbUrl, primary } : null);
  if (rbxInfo?.available && rbxInfo.url) return { state: 'ready', roblox: { url: rbxInfo.url }, glb: glb(false), note: '' };
  if (job.state === 'succeeded' && (job.export_state === 'running' || job.roblox_state === 'running')) {
    return { state: 'exporting', roblox: { busy: true }, glb: glb(false), note: '' };
  }
  if (BUILDING.has(job.state)) {
    return glbUrl ? { state: 'building', roblox: { waiting: true }, glb: glb(false), note: '' } : null;
  }
  // The service's reason may repeat the headline ("Roblox files unavailable for this hero: …"); keep only the cause.
  let reason = String(rbxInfo?.reason || job.roblox_message || 'Roblox files are not available for this hero.').trim()
    .replace(/^Roblox (?:body|files) unavailable for this hero: /, '');
  reason = reason.charAt(0).toUpperCase() + reason.slice(1);
  const note = `Roblox version unavailable. ${/[.!?]$/.test(reason) ? reason : `${reason}.`}`;
  if (!glbUrl && !['succeeded', 'stopped'].includes(job.state)) return null;
  return { state: 'unavailable', roblox: null, glb: glb(true), note: glbUrl ? note : `${note} ${glbInfo?.reason || ''}`.trim() };
}

function icon(name) {
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', 'icon'); svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(SVG, 'use');
  use.setAttribute('href', `/assets/icons/sprite.svg#i-${name}`);
  svg.append(use);
  return svg;
}

function option({ id, href, label, hint, primary, busy = false, onClick }) {
  const wrap = document.createElement('div');
  wrap.className = `deliver__option${primary ? ' deliver__option--primary' : ''}`;
  const control = document.createElement(href ? 'a' : 'button');
  control.className = `btn ${primary ? 'btn--primary' : 'btn--secondary'} btn--l btn--block`;
  control.id = id;
  if (href) { control.href = href; control.setAttribute('download', ''); if (onClick) control.addEventListener('click', onClick); }
  else { control.type = 'button'; control.disabled = true; }   // not aria-busy: the shared busy style hides the label
  control.append(icon(busy ? 'loader-circle' : href ? 'download' : 'package'), document.createTextNode(label));
  if (busy) control.classList.add('is-busy');
  const help = document.createElement('p');
  help.className = 'deliver__hint'; help.id = `${id}Hint`; help.textContent = hint;
  control.setAttribute('aria-describedby', help.id);
  wrap.append(control, help);
  return wrap;
}

// Fills #deliver for `job`; returns the state so the caller can rank Retry and Refine below it.
export function renderDownloads($, job, { track = () => {} } = {}) {
  const view = downloadState(job);
  const box = $('#deliver');
  box.hidden = !view;
  box.dataset.state = view?.state || '';
  const signature = JSON.stringify(view);
  if (box.dataset.signature === signature) return view;
  box.dataset.signature = signature;
  const items = [];
  if (view?.roblox) {
    const r = view.roblox;
    items.push(option({ id: 'robloxDownload', href: r.url, primary: true,
      label: r.busy ? 'Preparing Roblox files…' : 'Download for Roblox (.zip)', busy: r.busy,
      hint: r.waiting ? 'Ready when your hero finishes' : r.busy ? 'The Roblox bundle appears here when it is ready' : ROBLOX_HINT,
      onClick: () => track('download_click', { format: 'roblox' }, { flush: true }) }));
  }
  if (view?.glb) {
    items.push(option({ id: 'glbDownload', href: view.glb.url, primary: view.glb.primary, label: 'Download 3D model (.glb)',
      hint: GLB_HINT, onClick: () => track('download_click', { format: 'glb' }, { flush: true }) }));
  }
  $('#downloads').replaceChildren(...items);
  $('#deliverNoteText').textContent = view?.note || '';
  $('#deliverNote').hidden = !view?.note;
  return view;
}
