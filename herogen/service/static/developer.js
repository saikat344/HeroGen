// Read-only inspection of the selected hero through the service's authenticated Lab API.
const list = (value) => Array.isArray(value) ? value : [];
const object = (value) => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const number = (value) => typeof value === 'number' && Number.isFinite(value) ? value : null;
const text = (value) => value == null ? 'Not reported' :
  typeof value === 'object' ? JSON.stringify(value) : String(value);

export function inspectionStages(job, run) {
  const records = new Map(list(run?.stages).filter((s) => s && typeof s === 'object').map((s) => [String(s.stage), s]));
  const stages = new Map(list(job?.stages).filter((s) => s && typeof s === 'object').map((s) => [String(s.key), s]));
  for (const [key, record] of records) if (!stages.has(key)) stages.set(key, { key, label: record.name });
  return [...stages].map(([key, stage]) => {
    const record = object(records.get(key));
    return {
      key, label: stage.label || record.name || key, state: stage.state || 'Not reported by service',
      recordStatus: record.status || 'No record supplied',
      seconds: number(stage.seconds) ?? number(record.seconds),
      cost: number(stage.cost_usd) ?? number(record.cost_usd),
      models: list(record.models).filter((m) => typeof m === 'string'),
      mesh: record.mesh_source, joints: record.joints_source, code: object(record.code),
      calls: object(record.calls),
      checks: list(record.checks).filter((c) => c && typeof c === 'object').map((c) => ({
        name: c.name, value: c.value, threshold: c.threshold,
        comparison: c.cmp || c.op || null, assumed: c.cmp_assumed === true,
        verdict: c.pass === true ? 'Passed' : c.pass === false ?
          (c.review === true ? 'Needs review' : 'Failed') : 'Not recorded',
        kind: c.gate === false ? 'Diagnostic' : 'Gate',
      })),
    };
  });
}

function node(tag, content, className) {
  const element = document.createElement(tag);
  if (content != null) element.textContent = text(content);
  if (className) element.className = className;
  return element;
}

function facts(entries) {
  const dl = node('dl', null, 'developer-facts');
  for (const [label, value] of entries) dl.append(node('dt', label), node('dd', text(value)));
  return dl;
}

function table(checks) {
  const wrap = node('div', null, 'developer-table');
  wrap.tabIndex = 0;
  wrap.setAttribute('role', 'region');
  wrap.setAttribute('aria-label', 'Recorded checks');
  const table = node('table', null, 'checks');
  const head = node('thead');
  const row = node('tr');
  for (const label of ['Check', 'Measured', 'Requirement', 'Result', 'Type']) {
    const th = node('th', label); th.scope = 'col'; row.append(th);
  }
  head.append(row); table.append(head);
  const body = node('tbody');
  for (const c of checks) {
    const tr = node('tr');
    const requirement = c.threshold == null ? 'Not recorded' :
      `${c.comparison || 'Operator not recorded'} ${text(c.threshold)}${c.assumed ? ' (assumed operator)' : ''}`;
    for (const value of [c.name, c.value, requirement, c.verdict, c.kind]) tr.append(node('td', text(value)));
    body.append(tr);
  }
  table.append(body); wrap.append(table); return wrap;
}

export function developerMode(button, panel, getRun) {
  let enabled = false, job = null, run = null, request = 0, busy = false, message = '';
  const expanded = new Set();

  function render() {
    const active = panel.contains(document.activeElement) ? document.activeElement.getAttribute('data-dev-focus') : null;
    panel.hidden = !enabled;
    button.setAttribute('aria-pressed', String(enabled));
    if (!enabled) return;
    panel.replaceChildren();
    const header = node('div', null, 'row');
    const title = node('h2', 'Developer mode'); title.id = 'developerH'; title.tabIndex = -1;
    title.setAttribute('data-dev-focus', 'heading');
    header.append(title);
    const refresh = node('button', busy ? 'Loading records…' : 'Refresh records');
    refresh.type = 'button'; refresh.disabled = !job || busy;
    refresh.setAttribute('data-dev-focus', 'refresh');
    refresh.addEventListener('click', load);
    const close = node('button', 'Close developer mode'); close.type = 'button';
    close.setAttribute('data-dev-focus', 'close'); close.addEventListener('click', toggle);
    header.append(refresh, close); panel.append(header);
    panel.append(node('p', 'Inspect the recorded pipeline, measurements and provenance for your selected hero.', 'muted'));
    if (!job) { panel.append(node('p', 'Select a hero to inspect its pipeline.')); return; }
    panel.append(node('h3', job.name || 'Selected hero'));
    panel.append(facts([
      ['Job', job.id], ['State', job.state], ['Profile', job.profile],
      ['Job attempts', number(job.attempts)], ['Build commit', job.code_commit],
      ['Recorded duration (seconds)', number(job.seconds)],
      [job.cost_is_estimate === false ? 'Recorded cost (USD)' : 'Estimated cost (USD)', number(job.cost_usd)],
    ]));
    panel.append(node('p', 'Attempts are counted per job. Complete per-stage retry histories are not supplied by this API.', 'small muted'));
    const status = node('p', message, 'small muted'); status.setAttribute('role', 'status'); panel.append(status);
    if (job.error) panel.append(node('p', `${text(job.error.stage_label)}: ${text(job.error.code)} — ${text(job.error.message)}`, 'err'));
    for (const stage of inspectionStages(job, run)) {
      const details = node('details', null, 'developer-stage');
      details.open = expanded.has(stage.key);
      details.addEventListener('toggle', () => {
        if (!details.isConnected) return;
        if (details.open) expanded.add(stage.key); else expanded.delete(stage.key);
      });
      const summary = node('summary', `${stage.label} · ${stage.state} · ${stage.checks.length} reported checks`);
      summary.setAttribute('data-dev-focus', `stage:${stage.key}`);
      details.append(summary);
      details.append(facts([
        ['Stage record', stage.recordStatus], ['Models', stage.models.length ? stage.models.join(', ') : null],
        ['Duration (seconds)', stage.seconds], ['Estimated cost (USD)', stage.cost],
        ['Model calls', number(stage.calls.count)], ['Replayed calls', number(stage.calls.replayed)],
        ['Mesh source', stage.mesh], ['Joints source', stage.joints],
        ['Stage commit', stage.code.commit], ['Modified code', stage.code.dirty],
      ]));
      if (stage.checks.length) details.append(table(stage.checks));
      else details.append(node('p', 'No checks available through this inspection API.', 'muted small'));
      panel.append(details);
    }
    const link = node('a', 'Open full inspection: views, poses, geometry and quality');
    link.setAttribute('data-dev-focus', 'viewer');
    link.href = `/viewer#run=r1/${encodeURIComponent(job.id)}`;
    panel.append(link);
    panel.append(node('h3', 'Explore an improvement'));
    panel.append(node('p', 'Choose a recorded failure, inspect its matching view or pose, then compare frozen versions on the same inputs and reference artwork. Scores against generated views measure consistency with those views. Missing measurements remain unrecorded.'));
    panel.append(node('p', 'Source map: herogen/pipeline (image stages), herogen/blender (body and scores), herogen/roblox (export), herogen/service (jobs and attempts).', 'small muted'));
    if (active) {
      const target = [...panel.querySelectorAll('[data-dev-focus]')].find((n) => n.getAttribute('data-dev-focus') === active);
      (target && !target.disabled ? target : title).focus();
    }
  }

  async function load() {
    if (!enabled || !job || busy) return;
    const token = ++request, id = job.id;
    busy = true; message = 'Loading stage records…'; render();
    try {
      const data = await getRun(id);
      if (token !== request || !enabled || job?.id !== id) return;
      run = data;
      message = data?.error ? `Inspection is incomplete: ${text(data.error)}` :
        'Stage records loaded. Service state and saved stage records are shown separately.';
    } catch (error) {
      if (token !== request || !enabled || job?.id !== id) return;
      run = null;
      message = error.status === 404 ? 'Stage records are not available yet. Job state is shown below.' :
        `Could not load inspection records: ${error.message}. Use Refresh records to retry.`;
    } finally {
      if (token === request) { busy = false; render(); }
    }
  }

  function toggle() {
    enabled = !enabled;
    ++request; busy = false;
    render();
    if (enabled) { panel.querySelector('h2').focus(); load(); }
    else button.focus();
  }
  button.addEventListener('click', toggle);
  panel.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && enabled) { event.preventDefault(); toggle(); }
  });
  return {
    update(next) {
      if (job?.id !== next?.id) {
        ++request; busy = false; run = null; message = ''; expanded.clear();
      }
      job = next;
      render();
      if (enabled) load();
    },
  };
}
