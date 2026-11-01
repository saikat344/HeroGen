// Read-only presentation of the selected job's recorded stage and publication times.
const SOURCE = {p: 'pipeline/prompt_sketch.py', 0: 'pipeline/inputs.py', 1: 'pipeline/stages.py',
  2: 'pipeline/stages.py', 3: 'blender/stage3.py', '3f': 'blender/refine/__init__.py',
  4: 'blender/stage4.py', 6: 'roblox/export.py', 5: 'pipeline/results.py'};
export function stageEvents(job, run, now = Date.now()) {
  const events = [];
  for (const stage of job.stages || []) {
    const record = run?.stages?.find(r => String(r.stage) === stage.key);
    const start = Date.parse(record?.started);
    if (Number.isFinite(start)) {
      events.push({time: start, text: `${stage.label} started`, state: 'started'});
      if (stage.state !== 'running' && Number.isFinite(record.seconds)) events.push({
        time: start + record.seconds * 1000, text: `${stage.label} · ${stage.state}`, state: stage.state});
    } else if (stage.state === 'running' && Number.isFinite(stage.elapsed_seconds)) {
      events.push({time: now - stage.elapsed_seconds * 1000, text: `${stage.label} started`, state: 'running'});
    }
  }
  for (const model of job.models || []) if (model.n && Number.isFinite(Date.parse(model.time))) {
    events.push({time: Date.parse(model.time), text: `Refined version ${model.n} ready`, state: 'passed'});
  }
  return events.sort((a, b) => a.time - b.time);
}
export function workbenchLog($, api) {
  let job = null, run = null, signature = '', request = 0, log = '';
  const records = new Map();
  function render() {
    const list = $('#stageLog'), follow = $('#followLive').checked;
    const position = list.scrollTop;
    const rows = stageEvents(job, run);
    log = rows.map(row => `${new Date(row.time).toISOString()}  ${row.text}`).join('\n');
    list.replaceChildren(...rows.map(row => {
      const li = document.createElement('li'); li.className = `st-${row.state}`;
      const time = document.createElement('time'); time.dateTime = new Date(row.time).toISOString();
      time.textContent = new Date(row.time).toLocaleTimeString([], {hour12: false});
      const text = document.createElement('span'); text.textContent = row.text;
      li.append(time, text); return li;
    }));
    if (!rows.length) { const li = document.createElement('li'); li.textContent = 'Waiting for recorded stage events.'; list.append(li); }
    list.scrollTop = follow ? list.scrollHeight : position;
  }
  $('#followLive').addEventListener('change', () => { if (job) render(); });
  $('#copyLog').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(log); $('#logStatus').textContent = 'Log copied'; }
    catch { $('#logStatus').textContent = 'Copy unavailable. Select the log text to copy it.'; }
  });
  return {update(next) {
    if (job?.id !== next.id) { run = null; signature = ''; records.clear(); }
    job = next; render();
    const current = JSON.stringify([job.id, job.stages?.map(s => [s.key, s.state]), job.models]);
    if (current === signature) return;
    signature = current;
    const token = ++request;
    $('#logStatus').textContent = 'Loading recorded events…';
    const id = job.id;
    const stages = (job.stages || []).filter(s => !['pending','running'].includes(s.state) && !records.has(s.key));
    Promise.allSettled(stages.map(s => api(`/runs/r1/${encodeURIComponent(id)}/stage-${encodeURIComponent(s.key)}.json`)
      .then(record => ({...record, stage: s.key})))).then(results => {
      if (token !== request) return;
      const failures = [];
      for (const result of results) {
        if (result.status === 'fulfilled') records.set(result.value.stage, result.value);
        else if (result.reason.status !== 404) failures.push(result.reason.message);
      }
      run = {stages: [...records.values()]}; render();
      $('#logStatus').textContent = failures.length ? `Could not load stage events: ${failures.join('; ')}` :
        results.some(r => r.status === 'rejected') ? 'Some stage times were not recorded.' : 'Recorded stage times';
    });
  }, reset() { ++request; job = null; run = null; signature = ''; records.clear(); }};
}
export function stageSource(key) {
  return SOURCE[key] ? `https://github.com/saikat344/HeroGen/blob/main/herogen/${SOURCE[key]}` : null;
}
