// HeroGen Admin panel: Settings (#settings) and Analytics (#analytics).
// Uses textContent and SVG geometric attributes only (no innerHTML, no inline styles).

const SVG_NS = 'http://www.w3.org/2000/svg';
const TABS = ['settings', 'analytics'];

export function tabFromHash(hash) {
  return hash === '#analytics' ? 'analytics' : 'settings';
}

export function formatUsd(val, { precise = false } = {}) {
  if (typeof val !== 'number' || !Number.isFinite(val)) return '—';
  if (val === 0) return '$0.00';
  if (precise && Math.abs(val) < 0.01) return `$${val.toFixed(4)}`;
  if (precise && Math.abs(val) < 1) return `$${val.toFixed(3)}`;
  return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: precise ? 4 : 2 })}`;
}

export function formatDuration(secs) {
  if (typeof secs !== 'number' || !Number.isFinite(secs)) return '—';
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const mins = Math.floor(secs / 60);
  const rem = Math.round(secs % 60);
  if (mins < 60) return `${mins}m ${rem}s`;
  const hrs = (secs / 3600).toFixed(1);
  return `${hrs}h`;
}

export function formatAge(secs) {
  if (typeof secs !== 'number' || !Number.isFinite(secs) || secs < 0) return '0s';
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
  if (secs < 86400) return `${(secs / 3600).toFixed(1)}h`;
  return `${(secs / 86400).toFixed(1)}d`;
}

function formatLocalTime(epochS) {
  if (typeof epochS !== 'number' || !Number.isFinite(epochS)) return 'Not configured';
  return new Date(epochS * 1000).toLocaleString();
}

function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'cls') node.className = v;
    else if (k === 'text') node.textContent = String(v);
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, String(v));
  }
  for (const kid of kids) {
    if (kid != null) node.append(kid);
  }
  return node;
}

function svgEl(tag, attrs = {}, ...kids) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v != null) node.setAttribute(k, String(v));
  }
  for (const kid of kids) {
    if (kid != null) node.append(kid);
  }
  return node;
}

function svgSingleBar(value, maxValue, variant = 'accent') {
  const W = 240;
  const H = 10;
  const ratio = maxValue > 0 && value > 0 ? Math.min(1, value / maxValue) : 0;
  const fillW = ratio > 0 ? Math.max(3, Math.round(ratio * W * 10) / 10) : 0;
  const track = svgEl('rect', { x: 0, y: 0, width: W, height: H, rx: 4, class: 'svg-track' });
  const children = [track];
  if (fillW > 0) {
    children.push(svgEl('rect', { x: 0, y: 0, width: fillW, height: H, rx: 4, class: `svg-fill svg-fill-${variant}` }));
  }
  return svgEl('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', class: 'bar-svg', 'aria-hidden': 'true' }, ...children);
}

function svgDualBar(v1, v2, maxValue) {
  const W = 240;
  const H = 16;
  const w1 = maxValue > 0 && v1 > 0 ? Math.max(3, Math.round(Math.min(1, v1 / maxValue) * W * 10) / 10) : 0;
  const w2 = maxValue > 0 && v2 > 0 ? Math.max(3, Math.round(Math.min(1, v2 / maxValue) * W * 10) / 10) : 0;
  const kids = [
    svgEl('rect', { x: 0, y: 0, width: W, height: 6, rx: 3, class: 'svg-track' }),
    svgEl('rect', { x: 0, y: 9, width: W, height: 6, rx: 3, class: 'svg-track' }),
  ];
  if (w1 > 0) kids.push(svgEl('rect', { x: 0, y: 0, width: w1, height: 6, rx: 3, class: 'svg-fill svg-fill-ok' }));
  if (w2 > 0) kids.push(svgEl('rect', { x: 0, y: 9, width: w2, height: 6, rx: 3, class: 'svg-fill svg-fill-accent' }));
  return svgEl('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', class: 'bar-svg bar-svg-dual', 'aria-hidden': 'true' }, ...kids);
}

function svgBudgetMeter(limit, used, reserved) {
  const W = 300;
  const H = 14;
  const safeLimit = typeof limit === 'number' && limit > 0 ? limit : 0;
  const uRatio = safeLimit > 0 ? Math.min(1, Math.max(0, (used || 0) / safeLimit)) : 0;
  const rRatio = safeLimit > 0 ? Math.min(1 - uRatio, Math.max(0, (reserved || 0) / safeLimit)) : 0;
  const uW = Math.round(uRatio * W * 10) / 10;
  const rW = Math.round(rRatio * W * 10) / 10;
  const kids = [svgEl('rect', { x: 0, y: 0, width: W, height: H, rx: 5, class: 'svg-track' })];
  if (uW > 0) kids.push(svgEl('rect', { x: 0, y: 0, width: uW, height: H, class: 'svg-fill svg-fill-accent' }));
  if (rW > 0) kids.push(svgEl('rect', { x: uW, y: 0, width: rW, height: H, class: 'svg-fill svg-fill-warn' }));
  return svgEl('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', class: 'budget-svg', 'aria-hidden': 'true' }, ...kids);
}

function kpiCard(label, value, sub = '') {
  return el('div', { cls: 'kpi-card' },
    el('span', { cls: 'kpi-label', text: label }),
    el('strong', { cls: 'kpi-value', text: value }),
    sub ? el('small', { cls: 'kpi-sub muted', text: sub }) : null);
}

function badge(text, tone = 'neutral') {
  return el('span', { cls: `admin-badge badge-${tone}`, text });
}

function barRow(label, countText, value, maxValue, variant = 'accent', sub = '') {
  return el('div', { cls: 'bar-row' },
    el('div', { cls: 'bar-head' },
      el('span', { cls: 'bar-label', text: label }),
      el('span', { cls: 'bar-val', text: countText })),
    svgSingleBar(value, maxValue, variant),
    sub ? el('small', { cls: 'muted bar-sub', text: sub }) : null);
}

function bootAdmin() {
  const $ = (s) => document.querySelector(s);
  const state = {
    csrf: null,
    session: null,
    budget: null,
    analytics: null,
    activeTab: 'settings',
    formDirty: false,
  };

  function setBanner(msg) {
    const b = $('#adminBanner');
    b.textContent = msg || '';
    b.hidden = !msg;
  }

  async function api(path, { method = 'GET', json } = {}) {
    const opt = { method, headers: {}, credentials: 'same-origin', cache: 'no-store' };
    if (method !== 'GET') opt.headers['X-CSRF-Token'] = state.csrf || '';
    if (json !== undefined) {
      opt.headers['Content-Type'] = 'application/json';
      opt.body = JSON.stringify(json);
    }
    const r = await fetch(path, opt);
    let data = null;
    try { data = await r.json(); } catch { data = null; }
    if (!r.ok) {
      const msg = (data && data.error && data.error.message) || `Request failed (${r.status}).`;
      const err = new Error(msg);
      err.status = r.status;
      err.code = data && data.error && data.error.code;
      throw err;
    }
    return data;
  }

  function renderBudgetMeterInto(wrap, b) {
    wrap.replaceChildren();
    if (!b || typeof b.limit_usd !== 'number') {
      wrap.append(el('p', { cls: 'empty-note muted small', text: 'No live-generation budget has been configured yet.' }));
      return;
    }
    wrap.append(
      svgBudgetMeter(b.limit_usd, b.spent_usd || 0, b.running_usd || 0),
      el('div', { cls: 'meter-legend small muted' },
        el('span', { cls: 'legend-item' }, el('i', { cls: 'dot-swatch swatch-accent', 'aria-hidden': 'true' }), `Spent ${formatUsd(b.spent_usd || 0)}`),
        el('span', { cls: 'legend-item' }, el('i', { cls: 'dot-swatch swatch-warn', 'aria-hidden': 'true' }), `Running ${formatUsd(b.running_usd || 0)}`),
        el('span', { cls: 'legend-item' }, el('i', { cls: 'dot-swatch swatch-track', 'aria-hidden': 'true' }), `Remaining ${formatUsd(b.remaining_usd || 0)}`))
    );
  }

  function populateBudgetForm(b) {
    $(b.enabled ? '#modeAccepting' : '#modePaused').checked = true;
    $('#limitUsd').value = String(b.limit_usd);
    $('#runningEstimateUsd').value = String(b.running_estimate_usd);
    $('#maxRunning').value = String(b.max_running);
    state.formDirty = false;
  }

  function renderSettings(b) {
    state.budget = b;
    $('#budgetSentence').textContent = b.sentence;
    $('#quickPauseBtn').textContent = b.enabled ? 'Pause now' : 'Resume';
    $('#budgetKpis').replaceChildren(
      kpiCard('Spent', formatUsd(b.spent_usd)),
      kpiCard('Running', formatUsd(b.running_usd), `${b.active} queued or running`),
      kpiCard('Remaining', formatUsd(b.remaining_usd)));
    renderBudgetMeterInto($('#budgetBarWrap'), b);
    if (!state.formDirty) populateBudgetForm(b);
  }

  function renderStageTable(wrap, stages) {
    wrap.replaceChildren();
    if (!stages || !stages.length) {
      wrap.append(el('p', { cls: 'empty-note muted', text: 'No stage records in this window.' }));
      return;
    }
    const tbody = el('tbody');
    for (const s of stages) {
      tbody.append(el('tr', {},
        el('td', {}, el('code', { cls: 'stage-code', text: s.stage }), ' ', el('span', { text: s.label })),
        el('td', { cls: 'num', text: s.runs }),
        el('td', { cls: 'num ok-num', text: s.passed }),
        el('td', { cls: 'num bad-num', text: s.failed }),
        el('td', { cls: 'num warn-num', text: s.rejected }),
        el('td', { cls: 'num muted', text: s.unavailable }),
        el('td', { cls: 'num', text: formatDuration(s.median_seconds) }),
        el('td', { cls: 'num', text: formatDuration(s.p90_seconds) }),
        el('td', { cls: 'num', text: s.mean_cost_usd != null ? formatUsd(s.mean_cost_usd, { precise: true }) : '—' }),
        el('td', { cls: 'num', text: formatUsd(s.total_cost_usd || 0, { precise: true }) })));
    }
    wrap.append(el('table', { cls: 'checks admin-table stage-table' },
      el('thead', {}, el('tr', {},
        el('th', { scope: 'col', text: 'Stage' }),
        el('th', { scope: 'col', cls: 'num', text: 'Runs' }),
        el('th', { scope: 'col', cls: 'num', text: 'Passed' }),
        el('th', { scope: 'col', cls: 'num', text: 'Failed' }),
        el('th', { scope: 'col', cls: 'num', text: 'Rejected' }),
        el('th', { scope: 'col', cls: 'num', text: 'Unavail.' }),
        el('th', { scope: 'col', cls: 'num', text: 'Median' }),
        el('th', { scope: 'col', cls: 'num', text: 'p90' }),
        el('th', { scope: 'col', cls: 'num', text: 'Mean USD' }),
        el('th', { scope: 'col', cls: 'num', text: 'Total USD' }))),
      tbody));
  }

  function renderAnalytics(rep) {
    state.analytics = rep;
    const days = rep.window?.days || 7;
    $('#analyticsWindowNote').textContent = `Showing the last ${days} day${days === 1 ? '' : 's'} (UTC). Session-scoped visitor IDs only; no cookies, IPs, User-Agents, emails, prompts or file names.`;

    // (a) Daily visitors & page views
    const byPage = rep.daily?.by_page || {};
    $('#dailyKpis').replaceChildren(
      kpiCard('Session visitors', rep.daily?.visitors ?? 0, `last ${days}d`),
      kpiCard('Page views', rep.daily?.page_views ?? 0, `last ${days}d`),
      kpiCard('Landing views', byPage.landing ?? 0),
      kpiCard('Workbench views', byPage.workbench ?? 0),
      kpiCard('Gallery detail', byPage.gallery_detail ?? 0),
      kpiCard('Legal clicks', byPage.legal ?? 0)
    );

    const dailyChart = $('#dailyChart');
    dailyChart.replaceChildren();
    const series = rep.daily?.series || [];
    const maxDaily = Math.max(0, ...series.map((d) => Math.max(d.visitors || 0, d.page_views || 0)));
    if (maxDaily === 0) {
      dailyChart.append(el('p', { cls: 'empty-note muted', text: 'No visitors or page views recorded in this window yet.' }));
    } else {
      dailyChart.append(el('div', { cls: 'meter-legend small muted' },
        el('span', { cls: 'legend-item' }, el('i', { cls: 'dot-swatch swatch-ok', 'aria-hidden': 'true' }), 'Visitors'),
        el('span', { cls: 'legend-item' }, el('i', { cls: 'dot-swatch swatch-accent', 'aria-hidden': 'true' }), 'Page views')));
      for (const row of series) {
        dailyChart.append(el('div', { cls: 'daily-row' },
          el('code', { cls: 'daily-date', text: row.day.slice(5) }),
          svgDualBar(row.visitors, row.page_views, maxDaily),
          el('span', { cls: 'daily-nums small', text: `${row.visitors}v · ${row.page_views}pv` })));
      }
    }

    // (b) Funnel
    const funnelChart = $('#funnelChart');
    const funnelBreakdowns = $('#funnelBreakdowns');
    funnelChart.replaceChildren();
    funnelBreakdowns.replaceChildren();
    const steps = rep.funnel?.steps || [];
    const maxFunnel = Math.max(0, ...steps.map((s) => s.count || 0));
    const topCount = steps[0]?.count || 0;
    if (maxFunnel === 0) {
      funnelChart.append(el('p', { cls: 'empty-note muted', text: 'No funnel activity recorded in this window yet.' }));
    } else {
      steps.forEach((step, idx) => {
        const prev = idx > 0 ? steps[idx - 1].count : null;
        const pctTop = topCount > 0 ? `${Math.round((step.count / topCount) * 100)}% of landing` : '';
        // Steps count events in the window, not one visitor's path, so repeat views can outnumber the step before.
        const pctPrev = !prev ? '' : step.count > prev ? 'includes repeat views'
          : `${Math.round((step.count / prev) * 100)}% of previous`;
        const meta = [pctPrev, idx > 1 ? pctTop : ''].filter(Boolean).join(' · ');
        funnelChart.append(barRow(`${idx + 1}. ${step.label}`, String(step.count), step.count, maxFunnel,
          idx === steps.length - 1 ? 'ok' : 'accent', meta));
      });
    }
    const src = rep.funnel?.idea_sources || {};
    const dl = rep.funnel?.downloads_by_format || {};
    funnelBreakdowns.append(
      badge(`Suggestions: ${src.suggestion ?? 0}`),
      badge(`Starters: ${src.starter ?? 0}`),
      badge(`Uploads: ${src.upload ?? 0}`),
      badge(`Typed ideas: ${src.typed ?? 0}`),
      badge(`GLB downloads: ${dl.glb ?? 0}`),
      badge(`Roblox downloads: ${dl.roblox ?? 0}`)
    );

    // (c) Refusal reasons
    const refWrap = $('#refusalsChart');
    refWrap.replaceChildren();
    const refusals = rep.refusals || [];
    if (!refusals.length) {
      refWrap.append(el('p', { cls: 'empty-note muted', text: 'No create refusals recorded in this window.' }));
    } else {
      const maxRef = Math.max(1, ...refusals.map((r) => r.count));
      for (const r of refusals) {
        refWrap.append(barRow(r.reason, String(r.count), r.count, maxRef, 'bad'));
      }
    }

    // (d) Jobs by final state
    const jobsSummary = $('#jobsStateSummary');
    const jobsChart = $('#jobsStateChart');
    jobsSummary.replaceChildren();
    jobsChart.replaceChildren();
    const jobs = rep.jobs || {};
    const byState = jobs.by_final_state || {};
    const active = jobs.active || {};
    const byKind = jobs.by_user_kind || {};
    jobsSummary.append(
      badge(`Total jobs: ${jobs.total ?? 0}`),
      badge(`Queued: ${active.queued ?? 0}`, (active.queued ?? 0) > 0 ? 'warn' : 'neutral'),
      badge(`Running: ${active.running ?? 0}`, (active.running ?? 0) > 0 ? 'warn' : 'neutral'),
      badge(`Accounts: ${byKind.account ?? 0}`),
      badge(`Guests: ${byKind.guest ?? 0}`)
    );
    const stateEntries = [
      ['succeeded', 'Succeeded', 'ok'],
      ['rejected', 'Rejected', 'warn'],
      ['failed', 'Failed', 'bad'],
      ['canceled', 'Canceled', 'muted'],
    ];
    const maxState = Math.max(0, ...stateEntries.map(([k]) => byState[k] || 0));
    if ((jobs.total ?? 0) === 0) {
      jobsChart.append(el('p', { cls: 'empty-note muted', text: 'No jobs created in this window.' }));
    } else {
      for (const [k, label, tone] of stateEntries) {
        const c = byState[k] || 0;
        jobsChart.append(barRow(label, String(c), c, Math.max(1, maxState), tone));
      }
    }

    // (e) Per-stage table & per-job totals
    const jt = rep.stages?.job_totals || {};
    $('#jobTotalsKpis').replaceChildren(
      kpiCard('Jobs in window', jt.jobs ?? 0, `${jt.timed_jobs ?? 0} finished with timing`),
      kpiCard('Median wall time', formatDuration(jt.median_seconds)),
      kpiCard('p90 wall time', formatDuration(jt.p90_seconds)),
      kpiCard('Median job cost', jt.median_cost_usd != null ? formatUsd(jt.median_cost_usd, { precise: true }) : '—'),
      kpiCard('p90 job cost', jt.p90_cost_usd != null ? formatUsd(jt.p90_cost_usd, { precise: true }) : '—'),
      kpiCard('Total stage/job cost', formatUsd(jt.total_cost_usd || 0, { precise: true }),
        jt.mean_cost_usd != null ? `mean ${formatUsd(jt.mean_cost_usd, { precise: true })}` : 'estimated USD')
    );
    renderStageTable($('#mainStagesWrap'), rep.stages?.main || []);

    // (f) Live-generation budget
    const b = rep.budget || {};
    $('#analyticsBudgetKpis').replaceChildren(
      kpiCard('Total limit', formatUsd(b.limit_usd)),
      kpiCard('Spent', formatUsd(b.spent_usd || 0)),
      kpiCard('Running', formatUsd(b.running_usd || 0), `${b.active ?? 0} queued or running`),
      kpiCard('Remaining', formatUsd(b.remaining_usd))
    );
    renderBudgetMeterInto($('#analyticsBudgetBar'), b);

    // (g) Gallery engagement
    const gal = rep.gallery || {};
    const shades = gal.shade_toggles || {};
    const themes = gal.theme_changes || {};
    $('#gallerySummary').replaceChildren(
      kpiCard('Inspect clicks', gal.inspect_clicks ?? 0),
      kpiCard('3D orbit drags', gal.viewer_drags ?? 0, 'first per page'),
      kpiCard('Shade toggles', shades.total ?? 0),
      kpiCard('Theme changes', themes.total ?? 0, `L:${themes.light ?? 0} D:${themes.dark ?? 0} S:${themes.system ?? 0}`)
    );

    const pickerWrap = $('#pickerChart');
    pickerWrap.replaceChildren();
    const picks = gal.picker_selections || [];
    if (!picks.length) {
      pickerWrap.append(el('p', { cls: 'empty-note muted', text: 'No gallery picker selections recorded in this window.' }));
    } else {
      const maxPick = Math.max(1, ...picks.map((p) => p.count));
      for (const p of picks) {
        pickerWrap.append(barRow(p.id, String(p.count), p.count, maxPick, 'accent'));
      }
    }

    const shadeWrap = $('#shadeChart');
    shadeWrap.replaceChildren();
    const shadeRows = [
      ['Textured', shades.textured ?? 0],
      ['Clay', shades.clay ?? 0],
      ['Wireframe', shades.wire ?? 0],
    ];
    const maxShade = Math.max(0, ...shadeRows.map(([, c]) => c));
    if (maxShade === 0) {
      shadeWrap.append(el('p', { cls: 'empty-note muted', text: 'No shade toggles recorded in this window.' }));
    } else {
      for (const [label, c] of shadeRows) {
        shadeWrap.append(barRow(label, String(c), c, maxShade, 'ok'));
      }
    }

    // (h) Device split
    const devWrap = $('#deviceChart');
    devWrap.replaceChildren();
    const desktop = rep.devices?.desktop || { visitors: 0, events: 0 };
    const mobile = rep.devices?.mobile || { visitors: 0, events: 0 };
    const totalDevVisitors = (desktop.visitors || 0) + (mobile.visitors || 0);
    const totalDevEvents = (desktop.events || 0) + (mobile.events || 0);
    if (totalDevEvents === 0) {
      devWrap.append(el('p', { cls: 'empty-note muted', text: 'No client device events recorded in this window.' }));
    } else {
      const maxVis = Math.max(1, desktop.visitors || 0, mobile.visitors || 0);
      const dPct = totalDevVisitors > 0 ? Math.round((desktop.visitors / totalDevVisitors) * 100) : 0;
      const mPct = totalDevVisitors > 0 ? Math.round((mobile.visitors / totalDevVisitors) * 100) : 0;
      devWrap.append(
        barRow('Desktop', `${desktop.visitors} visitors (${dPct}%)`, desktop.visitors, maxVis, 'accent', `${desktop.events} events`),
        barRow('Mobile', `${mobile.visitors} visitors (${mPct}%)`, mobile.visitors, maxVis, 'ok', `${mobile.events} events`)
      );
    }
  }

  async function loadBudget() {
    const b = await api('/api/admin/generation');
    renderSettings(b);
    return b;
  }

  async function loadAnalytics() {
    const days = Number.parseInt($('#analyticsDays').value, 10) || 7;
    const rep = await api(`/api/admin/analytics?days=${encodeURIComponent(days)}`);
    renderAnalytics(rep);
    return rep;
  }

  function selectTab(name, { focusTab = false, updateHash = true } = {}) {
    const next = TABS.includes(name) ? name : 'settings';
    state.activeTab = next;
    for (const t of TABS) {
      const btn = $(`#tab-${t}`);
      const panel = $(`#panel-${t}`);
      const selected = t === next;
      btn.setAttribute('aria-selected', String(selected));
      btn.tabIndex = selected ? 0 : -1;
      panel.hidden = !selected;
      if (selected && focusTab) btn.focus();
    }
    if (updateHash) {
      const wantedHash = `#${next}`;
      if (location.hash !== wantedHash) history.replaceState(null, '', wantedHash);
    }
    if (next === 'analytics' && !state.analytics) {
      loadAnalytics().catch((e) => setBanner(e.message));
    }
  }

  function initTabs() {
    const tabButtons = TABS.map((t) => $(`#tab-${t}`));
    for (const [idx, btn] of tabButtons.entries()) {
      btn.addEventListener('click', () => selectTab(TABS[idx], { focusTab: false, updateHash: true }));
      btn.addEventListener('keydown', (ev) => {
        let nextIdx = null;
        if (ev.key === 'ArrowRight') nextIdx = (idx + 1) % TABS.length;
        else if (ev.key === 'ArrowLeft') nextIdx = (idx - 1 + TABS.length) % TABS.length;
        else if (ev.key === 'Home') nextIdx = 0;
        else if (ev.key === 'End') nextIdx = TABS.length - 1;
        if (nextIdx !== null) {
          ev.preventDefault();
          selectTab(TABS[nextIdx], { focusTab: true, updateHash: true });
        }
      });
    }
    window.addEventListener('hashchange', () => {
      if ($('#adminShell').hidden) return;
      selectTab(tabFromHash(location.hash), { focusTab: false, updateHash: false });
    });
  }

  async function saveBudgetPayload(payload) {
    const msg = $('#budgetFormMsg');
    msg.className = 'msg small muted';
    msg.textContent = 'Saving…';
    $('#saveBudgetBtn').disabled = true;
    $('#quickPauseBtn').disabled = true;
    try {
      const updated = await api('/api/admin/generation', { method: 'POST', json: payload });
      state.formDirty = false;
      renderSettings(updated);
      if (state.analytics) {
        state.analytics.budget = updated;
        renderAnalytics(state.analytics);
      }
      msg.className = 'msg small ok-msg';
      msg.textContent = updated.enabled ? 'Saved — live generation is active.' : 'Saved — live generation is paused.';
    } catch (e) {
      msg.className = 'msg small err';
      msg.textContent = e.message;
    } finally {
      $('#saveBudgetBtn').disabled = false;
      $('#quickPauseBtn').disabled = false;
    }
  }

  function initControls() {
    for (const input of document.querySelectorAll('#budgetForm input, #budgetForm textarea')) {
      input.addEventListener('input', () => { state.formDirty = true; });
      input.addEventListener('change', () => { state.formDirty = true; });
    }

    $('#budgetForm').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      await saveBudgetPayload({
        enabled: $('#modeAccepting').checked,
        limit_usd: $('#limitUsd').value.trim(),
        running_estimate_usd: $('#runningEstimateUsd').value.trim(),
        max_running: Number($('#maxRunning').value),
      });
    });

    $('#quickPauseBtn').addEventListener('click', async () => {
      await saveBudgetPayload({ enabled: !state.budget.enabled });
    });

    $('#refreshBudgetBtn').addEventListener('click', async () => {
      $('#refreshBudgetBtn').disabled = true;
      setBanner('');
      try { state.formDirty = false; await loadBudget(); }
      catch (e) { setBanner(e.message); }
      finally { $('#refreshBudgetBtn').disabled = false; }
    });

    $('#analyticsDays').addEventListener('change', () => {
      setBanner('');
      loadAnalytics().catch((e) => setBanner(e.message));
    });

    $('#refreshAnalyticsBtn').addEventListener('click', async () => {
      $('#refreshAnalyticsBtn').disabled = true;
      setBanner('');
      try { await loadAnalytics(); }
      catch (e) { setBanner(e.message); }
      finally { $('#refreshAnalyticsBtn').disabled = false; }
    });

    $('#adminSignOut').addEventListener('click', async () => {
      try {
        await api('/api/auth/logout', { method: 'POST', json: {} });
        location.replace('/');
      } catch (e) {
        setBanner(e.message);
      }
    });
  }

  function showForbidden(session) {
    $('#adminLoading').hidden = true;
    $('#adminShell').hidden = true;
    $('#adminForbidden').hidden = false;
    if (session?.signed_in && session?.user?.kind === 'account') {
      $('#adminWho').textContent = session.user.email;
      $('#adminWho').hidden = false;
      $('#adminSignOut').hidden = false;
      $('#forbiddenMsg').textContent = `${session.user.email} is signed in, but is not listed in HEROGEN_SERVICE_ADMIN_EMAILS.`;
      $('#adminSignInLink').hidden = true;
    } else {
      $('#forbiddenMsg').textContent = 'Sign in with an administrator email to access live-generation settings and first-party analytics.';
      $('#adminSignInLink').hidden = false;
    }
  }

  async function boot() {
    initTabs();
    initControls();
    const initialTab = tabFromHash(location.hash);
    try {
      state.session = await api('/api/session');
      state.csrf = state.session.csrf;
    } catch (e) {
      $('#adminLoading').textContent = `HeroGen is not reachable: ${e.message}`;
      return;
    }
    if (!state.session.is_admin) {
      showForbidden(state.session);
      return;
    }
    try {
      await loadBudget();
    } catch (e) {
      if (e.status === 403) {
        showForbidden(state.session);
        return;
      }
      $('#adminLoading').hidden = true;
      setBanner(e.message);
      return;
    }
    $('#adminWho').textContent = state.session.user.email;
    $('#adminWho').hidden = false;
    $('#adminSignOut').hidden = false;
    $('#adminLoading').hidden = true;
    $('#adminShell').hidden = false;
    selectTab(initialTab, { focusTab: false, updateHash: true });
  }

  boot().catch((e) => setBanner(e.message));
}

if (typeof document !== 'undefined') {
  bootAdmin();
}
