import { show, hide } from '/assets/preview.js';
const $ = (s) => document.querySelector(s);
let character = null, sketch = false, loading = false;
async function model() {
  if (!character || loading) return;
  loading=true; $('#reloadPublicModel').disabled=true;
  $('#publicModelStatus').textContent='Loading the saved 3D model…';
  try { await show($('#publicCanvas'),character.assets.model.url); $('#publicModelStatus').textContent='Drag to orbit · Scroll to zoom'; }
  catch { $('#publicModelStatus').textContent='The 3D model could not be loaded. Use Reload 3D to try again.'; }
  finally { loading=false; $('#reloadPublicModel').disabled=false; }
}
function reference() {
  $('#referenceStatus').textContent='';
  $('#publicReference').src=character.assets[sketch?'sketch':'artwork'].url;
  $('#publicReference').alt=`${sketch?'Generated starting sketch':'Generated character artwork'} for ${character.title}`;
  $('#publicReference').hidden=false;
  $('#referenceLabel').textContent=sketch?'Starting sketch':'Character artwork';
  $('#referenceToggle').textContent=sketch?'Show artwork':'Show starting sketch';
}
$('#reloadPublicModel').addEventListener('click',model);
$('#referenceToggle').addEventListener('click',()=>{if(character){sketch=!sketch;reference();}});
$('#publicReference').addEventListener('error',()=>{ $('#publicReference').hidden=true;$('#referenceStatus').textContent='This reference image could not be loaded. Toggle the reference to retry.'; });
window.addEventListener('pagehide',hide);
async function boot() {
  const id = location.pathname.split('/').at(-1);
  const response=await fetch(`/api/gallery/${encodeURIComponent(id)}`,{credentials:'omit',cache:'no-store'});
  if(!response.ok) throw new Error('This public character is unavailable. Return to the gallery to choose another.');
  character=await response.json();
  $('#publicTitle').textContent=character.title; $('#publicDescription').textContent=character.description;
  $('#publicVersion').textContent=`Refined 3D mesh · Refinement record version ${character.code_commit.slice(0,12)} · Input: ${character.input_kind.toLowerCase()}`;
  document.title=`${character.title} · HeroGen`;
  const record = character.refinement;
  if (record) {
    const entries = [['Meta models', record.models.join(', ')], ['Refinement views accepted', `${record.accepted_views} / ${record.attempted_views}`],
      ['Refinement duration', `${record.seconds} seconds`], ['Refinement cost estimate', `$${record.estimated_cost_usd.toFixed(3)} · stage only, not a provider invoice`]];
    for (const [label, value] of entries) { const dt=document.createElement('dt'), dd=document.createElement('dd'); dt.textContent=label; dd.textContent=value; $('#publicRecord').append(dt,dd); }
  } else $('#publicRecordStatus').textContent='No published refinement measurements are available for this character.';
  reference();await model();
}
boot().catch((error)=>{ $('#publicTitle').textContent='Character unavailable';$('#publicModelStatus').textContent=error.message;$('#reloadPublicModel').disabled=true;$('#referenceToggle').disabled=true; });
