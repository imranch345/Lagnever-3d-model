// Lagnav dashboard. Reads /api/status every 15 s and renders every tab from it. Labels that
// come from files are inserted as text; only our own Markdown documents are rendered as HTML.

const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const POLL_MS = 15000;

const NAMES = {
  'heart.left_ventricle': 'Left ventricle',
  'heart.myocardium': 'Myocardium',
  'heart.right_ventricle': 'Right ventricle',
  'heart.left_atrium': 'Left atrium',
  'heart.right_atrium': 'Right atrium',
  'respiratory.left_lung': 'Left lung',
  'respiratory.right_lung': 'Right lung',
};
const ENTITY_ORDER = Object.keys(NAMES);
const STATE_LABEL = { running: 'Running', done: 'Done', waiting: 'Waiting', stopped: 'Stopped' };

let status = null;
let lastOk = 0;
let currentDoc = null;

// ---------------------------------------------------------------------------------------------
// Small helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function s(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function inline(text) {
  // Our own RESULTS.md and evidence captions: bold, code and links rendered; no raw HTML kept.
  const span = el('span');
  const safe = String(text ?? '').replace(/</g, '&lt;');
  span.innerHTML = window.marked ? marked.parseInline(safe) : safe;
  return span;
}

const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(d));
const pct = (x, d = 0) => (x === null || x === undefined ? '–' : `${(x * 100).toFixed(d)}%`);

function ago(seconds) {
  if (seconds === null || seconds === undefined) return '';
  if (seconds < 90) return `${Math.round(seconds)} s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min ago`;
  return `${(seconds / 3600).toFixed(1)} h ago`;
}

function pill(state, label) {
  return el('span', { class: `pill ${state}` }, el('span', { class: 'dot' }), label || STATE_LABEL[state] || state);
}

function legend(items) {
  return el('div', { class: 'legend' }, items.map(([cls, style, label]) =>
    el('span', {}, el('span', { class: `key ${cls || ''}`, style }), label)));
}

// One tooltip for every mark: values lead, labels follow.
const tooltip = $('tooltip');
function showTip(event, title, lines = []) {
  tooltip.replaceChildren(el('strong', {}, title), ...lines.map((l) => el('div', {}, l)));
  tooltip.style.display = 'block';
  const r = event.target.getBoundingClientRect?.() ?? { left: event.clientX, top: event.clientY, width: 0 };
  const x = event.clientX ?? r.left + r.width / 2;
  const y = event.clientY ?? r.top;
  const w = tooltip.offsetWidth;
  tooltip.style.left = `${Math.min(window.innerWidth - w - 8, Math.max(8, x + 14))}px`;
  tooltip.style.top = `${Math.max(8, y - tooltip.offsetHeight - 10)}px`;
}
function hideTip() { tooltip.style.display = 'none'; }
function hover(node, title, lines) {
  node.setAttribute('tabindex', '0');
  node.setAttribute('aria-label', [title, ...lines].join('. '));
  node.addEventListener('pointermove', (e) => showTip(e, title, lines));
  node.addEventListener('pointerleave', hideTip);
  node.addEventListener('focus', (e) => showTip(e, title, lines));
  node.addEventListener('blur', hideTip);
  return node;
}

// ---------------------------------------------------------------------------------------------
// Tabs

const TABS = ['overview', 'research', 'anatomy', 'viewer', 'evidence', 'docs'];
function showTab(name) {
  const tab = TABS.includes(name) ? name : 'overview';
  for (const t of TABS) $(`tab-${t}`).hidden = t !== tab;
  for (const a of document.querySelectorAll('nav a')) {
    if (a.dataset.tab === tab) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  if (tab === 'viewer' && !$('viewer-frame').src) $('viewer-frame').src = '/viewer';
  if (tab === 'docs' && !currentDoc) openDoc('notebook');
  window.scrollTo({ top: 0 });
  hideTip();
}
window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));

// ---------------------------------------------------------------------------------------------
// Overview

function stepTwentyOne() { return status.step21; }

function chamberRange(report) {
  if (!report) return null;
  const heart = ENTITY_ORDER.slice(0, 5).map((e) => report.entities[e]).filter((x) => x && x.generality_mm);
  const lungs = ENTITY_ORDER.slice(5).map((e) => report.entities[e]).filter((x) => x && x.generality_mm);
  const ratios = Object.values(report.entities).filter((x) => x.generality_mm)
    .map((x) => x.mean_baseline_mm / x.generality_mm);
  const span = (xs, k) => [Math.min(...xs.map((x) => x[k])), Math.max(...xs.map((x) => x[k]))];
  return {
    heart: span(heart, 'generality_mm'),
    lungs: span(lungs, 'generality_mm'),
    ratio: [Math.min(...ratios), Math.max(...ratios)],
  };
}

function renderOverview() {
  const l2 = status.level2a;
  const q = l2.quality;
  const report = l2.report;
  const st = stepTwentyOne();
  const done = st.points.reduce((a, p) => a + p.done, 0);
  const total = st.points.reduce((a, p) => a + p.total, 0);
  const decided = st.points.filter((p) => p.decided);
  const unanimous = decided.filter((p) => p.verdict === 'unanimous');
  const range = chamberRange(report);
  const tiles = [
    ['Real CT subjects usable', q.total ? `${q.usable}` : '–',
      q.total ? `of ${q.total} fetched · reference ${q.chosen}` : 'quality gate not run'],
    ['Shape models', report ? `${Object.keys(report.entities).length}` : '–',
      report?.joint ? `joint model: ${report.joint.modes} modes, ${report.joint.train} training subjects` : 'not built yet'],
    ['Held-out error, heart', range ? `${fmt(range.heart[0], 1)}–${fmt(range.heart[1], 1)} mm` : '–',
      range ? `lungs ${fmt(range.lungs[0], 1)}–${fmt(range.lungs[1], 1)} mm · ${fmt(range.ratio[0], 1)}–${fmt(range.ratio[1], 1)}× better than the mean shape` : ''],
    ['Step 21 runs', `${done} / ${total}`, st.stage_two.length ? `Stage 2: points ${st.stage_two.join(' and ')}` : 'Stage 1'],
    ['Step 21 verdicts', decided.length ? `${unanimous.length} of ${decided.length} unanimous` : '–',
      decided.length ? `points ${decided.map((p) => p.point).sort((a, b) => a - b).join(', ')}` : 'no point decided yet'],
  ];
  $('ov-tiles').replaceChildren(...tiles.map(([label, value, hint]) =>
    el('div', { class: 'tile' }, el('div', { class: 'label' }, label), el('div', { class: 'value' }, value),
      el('div', { class: 'hint' }, hint))));

  $('ov-jobs').replaceChildren(
    el('tr', {}, el('th', {}, 'State'), el('th', {}, 'Job'), el('th', {}, 'Last change')),
    ...status.jobs.map((j) => el('tr', {},
      el('td', {}, pill(j.state)),
      el('td', {}, el('div', {}, j.name), el('div', { class: 'muted' }, j.detail || '')),
      el('td', { class: 'num muted' }, ago(j.updated_seconds_ago)))));

  $('ov-notebook').replaceChildren(...status.notebook.slice(0, 6).map((h) =>
    el('li', {}, el('a', { href: '#docs', onclick: (e) => { e.preventDefault(); location.hash = 'docs'; openDoc('notebook', h); } }, h))));

  const rows = (list) => list.map((r) => el('li', {},
    el('div', {}, el('span', { class: 'chip' }, r.id), inline(r.claim)),
    el('div', { class: 'numbers' }, inline(r.numbers))));
  $('ov-research').replaceChildren(...rows(status.results.research));
  $('ov-product').replaceChildren(...rows(status.results.product));
}

// ---------------------------------------------------------------------------------------------
// Research: Step 21

function branchMap(st) {
  const rows = [...st.points].sort((a, b) => a.point - b.point);
  rows.push({ point: st.step20.point, frozen: true, escaped: 8, verdict: 'unanimous' });
  const rowH = 34, left = 104, gap = 26, top = 8;
  const width = left + 8 * gap + 60 + 230;
  const svg = s('svg', { viewBox: `0 0 ${width} ${top * 2 + rows.length * rowH}`, width, role: 'img',
    'aria-label': 'Outcome of each branch at each branch point' });
  rows.forEach((row, i) => {
    const y = top + i * rowH + rowH / 2;
    svg.append(s('text', { x: 0, y: y + 4 }, `step ${row.point}`));
    for (let b = 0; b < 8; b += 1) {
      const x = left + b * gap;
      const run = row.frozen ? { name: `branch_0${b}`, state: 'done', escaped: true } : row.runs.find((r) => r.name === `branch_0${b}`);
      let mark;
      if (run.state === 'done') {
        mark = s('circle', { cx: x, cy: y, r: 6, fill: run.escaped ? 'var(--series-1)' : 'var(--series-2)', stroke: 'var(--surface)', 'stroke-width': 2, class: 'mark' });
      } else {
        mark = s('circle', { cx: x, cy: y, r: 5, fill: 'none', stroke: run.state === 'running' ? 'var(--series-1)' : 'var(--muted)', 'stroke-width': 1.5, class: 'mark' });
      }
      svg.append(mark);
      const lines = row.frozen
        ? ['Step 20, frozen: escaped at 830–840']
        : run.state === 'done'
          ? [run.escaped ? `escaped at step ${run.escape_step}` : 'did not escape by step 1200',
            `final rotation ${fmt(run.final_rotation_deg, 2)}°`, `loader seed ${run.loader_seed}`]
          : [run.state === 'running' ? 'running now' : run.state === 'stopped' ? 'stopped before finishing' : 'waiting to run'];
      svg.append(hover(s('circle', { cx: x, cy: y, r: 12, class: 'hit' }), `step ${row.point} · ${run.name}`, lines));
    }
    const refX = left + 8 * gap + 12;
    const ref = row.frozen ? null : row.runs.find((r) => r.name === 'reference');
    const refDone = row.frozen || ref?.state === 'done';
    svg.append(s('rect', { x: refX - 6, y: y - 6, width: 12, height: 12, rx: 2, transform: `rotate(45 ${refX} ${y})`,
      fill: refDone ? 'var(--ink-2)' : 'none', stroke: refDone ? 'var(--surface)' : 'var(--muted)', 'stroke-width': refDone ? 2 : 1.5 }));
    svg.append(hover(s('circle', { cx: refX, cy: y, r: 12, class: 'hit' }), `step ${row.point} · reference`,
      row.frozen ? ['Step 20 frozen state'] : ref?.state === 'done'
        ? [ref.reproduces ? `reproduces frozen seed 9 (${ref.exact}/${ref.shared} checkpoints exact)` : 'does NOT reproduce frozen seed 9']
        : [ref?.state === 'running' ? 'running now' : 'waiting to run']));
    const text = row.frozen ? '8/8 · unanimous (Step 20)'
      : row.decided ? `${row.escaped}/8 · ${row.verdict}` : `${row.done} of ${row.total} runs done`;
    svg.append(s('text', { x: refX + 24, y: y + 4, style: 'fill: var(--ink)' }, text));
  });
  return svg;
}

function escapeStrip(st) {
  const rows = [...st.points].filter((p) => p.runs.some((r) => r.state === 'done' && r.escaped && r.name.startsWith('branch')))
    .sort((a, b) => a.point - b.point);
  if (!rows.length) return el('p', { class: 'empty' }, 'No escapes recorded yet.');
  const steps = rows.flatMap((p) => p.runs.filter((r) => r.escaped && r.name.startsWith('branch') && !r.name.endsWith('replay')).map((r) => r.escape_step));
  const lo = Math.floor((Math.min(...steps, st.frozen_escape[0]) - 20) / 20) * 20;
  const hi = Math.ceil((Math.max(...steps, st.frozen_escape[1]) + 20) / 20) * 20;
  const stacks = rows.map((p) => {
    const counts = {};
    for (const r of p.runs) if (r.escaped && r.name.startsWith('branch') && !r.name.endsWith('replay')) counts[r.escape_step] = (counts[r.escape_step] || 0) + 1;
    return counts;
  });
  const tallest = Math.max(1, ...stacks.flatMap((c) => Object.values(c)));
  const left = 80, right = 20, top = 22, width = 640, pitch = 8;
  const rowH = Math.max(36, tallest * pitch + 14);
  const plotW = width - left - right;
  const x = (v) => left + ((v - lo) / (hi - lo)) * plotW;
  const height = top + rows.length * rowH + 30;
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, width, role: 'img', 'aria-label': 'Escape step of each branch, by branch point' });
  // Frozen seed 9's own escape, as a band.
  svg.append(s('rect', { x: x(st.frozen_escape[0]), y: top - 6, width: Math.max(2, x(st.frozen_escape[1]) - x(st.frozen_escape[0])),
    height: rows.length * rowH + 6, fill: 'var(--band)' }));
  svg.append(s('text', { x: x(st.frozen_escape[0]) + 4, y: top - 10 }, `seed 9 itself: ${st.frozen_escape[0]}–${st.frozen_escape[1]}`));
  for (let t = lo; t <= hi; t += 20) {
    svg.append(s('line', { x1: x(t), x2: x(t), y1: top - 4, y2: top + rows.length * rowH, class: 'gridline' }));
    svg.append(s('text', { x: x(t), y: height - 8, 'text-anchor': 'middle' }, t));
  }
  svg.append(s('line', { x1: left, x2: width - right, y1: top + rows.length * rowH, y2: top + rows.length * rowH, class: 'axis' }));
  rows.forEach((row, i) => {
    const y = top + i * rowH + rowH / 2;
    svg.append(s('text', { x: 0, y: y + 4 }, `from ${row.point}`));
    const seen = {};
    row.runs.filter((r) => r.name.startsWith('branch') && !r.name.endsWith('replay') && r.state === 'done' && r.escaped)
      .forEach((r) => {
        const k = r.escape_step;
        seen[k] = (seen[k] || 0) + 1;
        // A dot plot: branches escaping at the same step stack, centred on the row.
        const dy = (seen[k] - 1 - (stacks[i][k] - 1) / 2) * pitch;
        svg.append(s('circle', { cx: x(k), cy: y + dy, r: 4.5, fill: 'var(--series-1)', stroke: 'var(--surface)', 'stroke-width': 2 }));
        svg.append(hover(s('circle', { cx: x(k), cy: y + dy, r: 12, class: 'hit' }), `step ${k}`,
          [`${r.name} from step ${row.point}`, `final rotation ${fmt(r.final_rotation_deg, 2)}°`]));
      });
  });
  return el('div', {}, svg, el('p', { class: 'muted', style: 'margin:6px 0 0;font-size:12px' },
    'Training step at which each escaped branch crossed into graph use for good; branches at the same step are stacked. Escapes before step 805 are timed on the 50-step grid; from 805 to 900 on a 5-step grid.'));
}

function renderResearch() {
  const st = stepTwentyOne();
  const decided = st.points.filter((p) => p.decided);
  const table = el('table', {},
    el('tr', {}, ...['Branch point', 'Stage', 'Escaped', 'Verdict', 'Escape step (range · mean)', 'Final rotation (mean ± SD)', 'Reference'].map((h, i) =>
      el('th', { class: i === 2 ? 'num' : '' }, h))),
    ...[...st.points].sort((a, b) => a.point - b.point).map((p) => el('tr', {},
      el('td', {}, `step ${p.point}`), el('td', {}, String(p.stage)),
      el('td', { class: 'num' }, p.decided ? `${p.escaped}/8` : `${p.escaped} so far`),
      el('td', {}, p.verdict ?? `running (${p.done}/${p.total})`),
      el('td', { class: 'num' }, p.escape_steps ? `${p.escape_steps.min}–${p.escape_steps.max} · ${fmt(p.escape_steps.mean, 1)}` : '–'),
      el('td', { class: 'num' }, p.final_rotation ? `${fmt(p.final_rotation.mean, 3)}° ± ${fmt(p.final_rotation.sd, 3)}` : '–'),
      el('td', {}, p.reference_reproduces === true ? 'reproduces ✓' : p.reference_reproduces === false ? 'FAILS ✗' : '–'))),
    el('tr', {}, el('td', {}, 'step 805'), el('td', {}, 'Step 20'), el('td', { class: 'num' }, '8/8'), el('td', {}, 'unanimous'),
      el('td', { class: 'num' }, '830–840'), el('td', { class: 'num' }, '20.988° ± 0.023'), el('td', {}, 'frozen')));

  $('step21').replaceChildren(
    el('div', { style: 'display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px' },
      el('h3', { style: 'margin:0' }, 'Step 21: when is seed 9\'s escape fixed?'), pill(st.job.state, `${STATE_LABEL[st.job.state]} · ${st.job.detail}`)),
    el('p', { class: 'note' }, st.question, ' ',
      decided.length ? `So far ${decided.filter((p) => p.verdict === 'unanimous').length} of ${decided.length} decided points are unanimous; ` : '',
      'step 0 decides between "fixed at initialisation" and "entered during training".'),
    legend([[null, 'background: var(--series-1)', 'Escaped'], [null, 'background: var(--series-2)', 'Did not escape'],
      ['hollow', 'border-color: var(--series-1)', 'Running'], ['hollow', null, 'Waiting'],
      [null, 'background: var(--ink-2); border-radius: 2px; transform: rotate(45deg); width: 9px; height: 9px', 'Reference (must reproduce seed 9)']]),
    el('div', { class: 'scroll' }, branchMap(st)),
    el('h3', { style: 'margin-top:18px' }, 'When each branch escaped'),
    el('div', { class: 'scroll' }, escapeStrip(st)),
    el('h3', { style: 'margin-top:18px' }, 'Table'),
    el('div', { class: 'scroll' }, table),
    el('details', { style: 'margin-top:12px' }, el('summary', {}, 'Sweep log (last lines)'), el('pre', { class: 'log' }, st.log_tail.join('\n'))));

  $('rs-table').replaceChildren(
    el('tr', {}, el('th', {}, ''), el('th', {}, 'Result'), el('th', {}, 'Numbers'), el('th', {}, 'Source')),
    ...status.results.research.map((r) => el('tr', {}, el('td', {}, el('span', { class: 'chip' }, r.id)),
      el('td', {}, inline(r.claim)), el('td', { class: 'ink2' }, inline(r.numbers)), el('td', { class: 'muted' }, inline(r.source)))));
}

// ---------------------------------------------------------------------------------------------
// Generated anatomy

function modelsChart(report) {
  const rows = [...ENTITY_ORDER.map((e) => ({ id: e, name: NAMES[e], ...report.entities[e] })),
    { id: 'joint', name: 'Joint (all seven)', ...report.joint }].filter((r) => r.generality_mm);
  const max = Math.max(...rows.map((r) => r.mean_baseline_mm));
  const top = Math.ceil(max / 5) * 5;
  const left = 140, right = 60, rowH = 30, head = 8, width = 640;
  const plotW = width - left - right;
  const x = (v) => left + (v / top) * plotW;
  const height = head + rows.length * rowH + 28;
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, width, role: 'img', 'aria-label': 'Held-out error of each model against the mean-shape baseline' });
  for (let t = 0; t <= top; t += 5) {
    svg.append(s('line', { x1: x(t), x2: x(t), y1: head, y2: head + rows.length * rowH, class: 'gridline' }));
    svg.append(s('text', { x: x(t), y: height - 8, 'text-anchor': 'middle' }, `${t}`));
  }
  svg.append(s('text', { x: x(top) + 20, y: height - 8 }, 'mm'));
  rows.forEach((r, i) => {
    const y = head + i * rowH + rowH / 2;
    svg.append(s('text', { x: 0, y: y + 4, style: r.id === 'joint' ? 'fill: var(--ink); font-weight: 600' : '' }, r.name));
    svg.append(s('line', { x1: x(r.generality_mm), x2: x(r.mean_baseline_mm), y1: y, y2: y, stroke: 'var(--axis)', 'stroke-width': 2 }));
    svg.append(s('circle', { cx: x(r.mean_baseline_mm), cy: y, r: 5, fill: 'var(--muted)', stroke: 'var(--surface)', 'stroke-width': 2 }));
    svg.append(s('circle', { cx: x(r.generality_mm), cy: y, r: 5, fill: 'var(--series-1)', stroke: 'var(--surface)', 'stroke-width': 2 }));
    svg.append(s('text', { x: x(r.mean_baseline_mm) + 10, y: y + 4 }, `${fmt(r.mean_baseline_mm / r.generality_mm, 1)}×`));
    const lines = [`model ${fmt(r.generality_mm)} mm · mean shape ${fmt(r.mean_baseline_mm)} mm`,
      `${r.modes} modes explain ${pct(r.explained_total, 1)}`, `${r.train} training subjects · ${r.held_out ?? ''} held out`];
    svg.append(hover(s('rect', { x: 0, y: y - rowH / 2, width, height: rowH, class: 'hit' }), r.name, lines));
  });
  return svg;
}

function validityChart(v) {
  const rows = v.spreads;
  const left = 44, right = 20, top = 16, plotH = 150, width = 420;
  const band = (width - left - right) / rows.length;
  const y = (p) => top + plotH * (1 - p);
  const svg = s('svg', { viewBox: `0 0 ${width} ${top + plotH + 30}`, width, role: 'img', 'aria-label': 'Share of generated thoraxes passing every check, by spread' });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(s('line', { x1: left, x2: width - right, y1: y(t), y2: y(t), class: t === 0 ? 'axis' : 'gridline' }));
    svg.append(s('text', { x: left - 8, y: y(t) + 4, 'text-anchor': 'end' }, `${t * 100}%`));
  }
  svg.append(s('line', { x1: left, x2: width - right, y1: y(v.method.target), y2: y(v.method.target), stroke: 'var(--ink-2)', 'stroke-width': 1 }));
  svg.append(s('text', { x: width - right, y: y(v.method.target) - 5, 'text-anchor': 'end' }, `target ${pct(v.method.target)}`));
  rows.forEach((r, i) => {
    const cx = left + band * i + band / 2;
    const w = Math.min(24, band * 0.5);
    const h = plotH * r.pass_rate;
    const yTop = y(r.pass_rate);
    const rr = Math.min(4, h);
    const d = `M${cx - w / 2},${top + plotH} V${yTop + rr} Q${cx - w / 2},${yTop} ${cx - w / 2 + rr},${yTop} H${cx + w / 2 - rr} Q${cx + w / 2},${yTop} ${cx + w / 2},${yTop + rr} V${top + plotH} Z`;
    svg.append(s('path', { d, fill: 'var(--series-1)' }));
    svg.append(s('text', { x: cx, y: yTop - 6, 'text-anchor': 'middle', style: 'fill: var(--ink)' }, pct(r.pass_rate)));
    svg.append(s('text', { x: cx, y: top + plotH + 18, 'text-anchor': 'middle' }, `${r.spread}σ`));
    const lines = [`${r.samples} samples at spread ${r.spread}`,
      ...Object.entries(r.failing).filter(([, share]) => share > 0).map(([k, share]) => `${pct(share)} fail ${k}`)];
    svg.append(hover(s('rect', { x: cx - band / 2, y: top, width: band, height: plotH, class: 'hit' }), `${pct(r.pass_rate)} pass every check`, lines));
  });
  return svg;
}

// Bars as HTML, so labels keep their size in a narrow card. Single series: no legend.
function htmlBars(items) {
  const max = Math.max(...items.map(([, v]) => v), 1);
  return el('div', { class: 'hbars' }, items.map(([label, value]) => hover(
    el('div', { class: 'hbar' }, el('div', { class: 'hlabel' }, label),
      el('div', { class: 'htrack' }, el('div', { class: 'hfill', style: `width:${(value / max) * 85}%` }), el('span', { class: 'hvalue' }, String(value)))),
    `${value} subjects`, [label])));
}

function renderAnatomy() {
  const l2 = status.level2a;
  const report = l2.report;
  const q = l2.quality;
  const flow = [
    ['Fetched', l2.subjects_on_disk || '–', 'by byte range from a 28.4 GB archive'],
    ['Usable', q.total ? q.usable : '–', q.total ? `quality gate · reference ${q.chosen}` : 'quality gate not run'],
    ['Training', report ? report.train.length : '–', 'used to fit the models'],
    ['Held out', report ? report.held_out.length : '–', 'never seen in fitting'],
  ];
  $('an-flow').replaceChildren(...flow.flatMap(([label, value, hint], i) => [
    i ? el('span', { class: 'arrow', 'aria-hidden': 'true' }, '→') : null,
    el('div', { class: 'tile' }, el('div', { class: 'label' }, label), el('div', { class: 'value' }, String(value)), el('div', { class: 'hint' }, hint))]).filter(Boolean));

  if (report) {
    $('an-models').replaceChildren(el('h3', {}, 'Held-out reconstruction error'),
      el('p', { class: 'note' }, `Mean vertex error rebuilding ${report.held_out.length} subjects the models never saw, against predicting the mean shape for everyone. Lower is better; the multiple is the improvement.`),
      legend([[null, 'background: var(--series-1)', 'Shape model'], [null, 'background: var(--muted)', 'Mean-shape baseline']]),
      el('div', { class: 'scroll' }, modelsChart(report)));
    const head = ['Structure', 'Train', 'Excluded', 'Fit (mean)', 'Modes', '1st mode', 'Held-out', 'Mean shape', 'Better', 'Specificity', 'Template'];
    $('an-table').replaceChildren(el('h3', {}, 'Models in detail'),
      el('p', { class: 'note' }, 'Fit: registration surface distance. Specificity: mean distance from 50 random samples to the nearest training shape. Template: faces · closed · Euler number (2 is sphere-like).'),
      el('div', { class: 'scroll' }, el('table', {},
        el('tr', {}, ...head.map((h, i) => el('th', { class: i ? 'num' : '' }, h))),
        ...ENTITY_ORDER.filter((e) => report.entities[e]).map((e) => {
          const r = report.entities[e];
          return el('tr', {}, el('td', {}, NAMES[e]), el('td', { class: 'num' }, r.train), el('td', { class: 'num' }, Object.keys(r.excluded).length),
            el('td', { class: 'num' }, `${fmt(r.fit_mean_mm)} mm`), el('td', { class: 'num' }, r.modes), el('td', { class: 'num' }, pct(r.explained?.[0])),
            el('td', { class: 'num' }, `${fmt(r.generality_mm)} mm`), el('td', { class: 'num' }, `${fmt(r.mean_baseline_mm)} mm`),
            el('td', { class: 'num' }, `${fmt(r.mean_baseline_mm / r.generality_mm, 1)}×`), el('td', { class: 'num' }, `${fmt(r.specificity_mm)} mm`),
            el('td', { class: 'num' }, r.template ? `${r.template.faces} · ${r.template.watertight ? 'yes' : 'no'} · ${r.template.euler}` : '–'));
        }),
        report.joint ? el('tr', {}, el('td', {}, el('strong', {}, 'Joint (all seven)')), el('td', { class: 'num' }, report.joint.train), el('td', { class: 'num' }, '–'),
          el('td', { class: 'num' }, '–'), el('td', { class: 'num' }, report.joint.modes), el('td', { class: 'num' }, pct(report.joint.explained?.[0])),
          el('td', { class: 'num' }, `${fmt(report.joint.generality_mm)} mm`), el('td', { class: 'num' }, `${fmt(report.joint.mean_baseline_mm)} mm`),
          el('td', { class: 'num' }, `${fmt(report.joint.mean_baseline_mm / report.joint.generality_mm, 1)}×`), el('td', { class: 'num' }, `${fmt(report.joint.specificity_mm)} mm`), el('td', { class: 'num' }, '–')) : null)));
  } else {
    $('an-models').replaceChildren(el('h3', {}, 'Models'), el('p', { class: 'empty' }, 'Not built yet.'));
    $('an-table').replaceChildren();
  }

  const v = l2.validity;
  const vJob = l2.jobs.find((j) => j.name.startsWith('Validity'));
  if (v) {
    const c = v.control;
    $('an-validity').replaceChildren(el('h3', {}, 'Do generated thoraxes look like real ones?'),
      el('p', { class: 'note' }, `Checks that can fail, calibrated on ${v.method.calibrated_on} real training subjects: folding, overlap deeper than ${v.method.depth_mm ?? 2} mm, volume range. Weights clipped at ${v.method.clip}σ.`),
      el('div', { class: 'tiles', style: 'margin-bottom:8px' },
        el('div', { class: 'tile' }, el('div', { class: 'label' }, 'Real held-out subjects passing'), el('div', { class: 'value' }, c.pass_rate === null ? '–' : pct(c.pass_rate)), el('div', { class: 'hint' }, `control: ${c.subjects} subjects`)),
        el('div', { class: 'tile' }, el('div', { class: 'label' }, 'Usable sampling spread'), el('div', { class: 'value' }, v.usable_spread === null ? 'none' : `${v.usable_spread}σ`), el('div', { class: 'hint' }, `largest spread with ≥ ${pct(v.method.target)} passing`))),
      el('div', { class: 'scroll' }, validityChart(v)));
  } else {
    $('an-validity').replaceChildren(el('h3', {}, 'Do generated thoraxes look like real ones?'),
      el('p', {}, vJob ? pill(vJob.state) : null, ' ', el('span', { class: 'ink2' }, 'The validity sweep writes its result here when it finishes.')));
  }

  const rej = Object.entries(q.rejections || {});
  $('an-rejections').replaceChildren(el('h3', {}, `Why ${q.total - q.usable} subjects were rejected`),
    el('p', { class: 'note' }, 'A subject can have several reasons. Most scans were taken for other regions and stop short of the chest.'),
    rej.length ? htmlBars(rej) : el('p', { class: 'empty' }, 'No quality report yet.'));

  const samples = l2.samples;
  $('an-samples').replaceChildren(el('h3', {}, 'Generated samples'),
    samples.length ? el('ul', { class: 'rows' }, ...samples.map((x) => el('li', { style: 'display:flex;gap:12px;align-items:center;flex-wrap:wrap' },
      el('span', {}, `Seed ${x.seed}`), x.usable ? pill('done', 'Passes the checks') : pill('stopped', 'Fails a check'),
      el('span', { class: 'muted' }, x.hard.join('; ')),
      el('button', { style: 'margin-left:auto', onclick: () => viewerAction('/sample', { seed: x.seed }) }, 'Open in the 3D viewer'))))
      : el('p', { class: 'empty' }, 'None yet. They are written after the validity sweep.'));
}

// ---------------------------------------------------------------------------------------------
// Viewer

async function viewerAction(path, body) {
  location.hash = 'viewer';
  const note = $('vw-note');
  note.textContent = 'Working…';
  try {
    const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
    const reply = await res.json();
    note.textContent = reply.message || reply.error || '';
    note.className = reply.ok === false || reply.error ? 'muted' : 'ink2';
  } catch (err) {
    note.textContent = `Request failed: ${err}`;
  }
  const frame = $('viewer-frame');
  if (frame.src) frame.contentWindow.location.reload(); else frame.src = '/viewer';
}

function renderViewerToolbar() {
  $('vw-toolbar').replaceChildren(
    el('button', { class: 'primary', onclick: () => viewerAction('/sample') }, 'Generate a new thorax'),
    el('button', { onclick: () => viewerAction('/original') }, 'Back to the real subject'),
    el('button', { onclick: () => viewerAction('/command', { text: 'make the left ventricle 15% larger' }) }, 'Make the left ventricle 15% larger'),
    el('button', { onclick: () => viewerAction('/reset') }, 'Reset the scene'),
    el('a', { href: '/viewer', target: '_blank', rel: 'noopener' }, 'Open in a new tab'),
    el('span', { id: 'vw-note', class: 'ink2', style: 'flex-basis:100%;font-size:13px' }));
}

// ---------------------------------------------------------------------------------------------
// Evidence and documents

function renderEvidence() {
  $('ev-gallery').replaceChildren(...status.evidence.map((item) => {
    const isImage = /\.(jpe?g|png)$/i.test(item.file);
    const media = isImage
      ? el('a', { href: `/evidence/${encodeURIComponent(item.file)}`, target: '_blank', rel: 'noopener' },
        el('img', { src: `/evidence/${encodeURIComponent(item.file)}`, alt: item.file, loading: 'lazy' }))
      : el('a', { href: '#docs', class: 'doc', onclick: (e) => { e.preventDefault(); location.hash = 'docs'; openDoc(`evidence:${item.file}`); } }, `Document: ${item.file}`);
    return el('div', { class: 'shot' }, media, el('div', { class: 'body' },
      el('div', { class: 'date' }, item.date), inline(item.shows), el('div', { class: 'how' }, inline(item.produced))));
  }));
}

function slug(text) { return String(text).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, ''); }

async function openDoc(id, heading) {
  currentDoc = id;
  renderDocList();
  const body = $('doc-body');
  body.replaceChildren(el('p', { class: 'muted' }, 'Loading…'));
  const url = id.startsWith('evidence:') ? `/evidence/${encodeURIComponent(id.slice(9))}` : `/api/doc?id=${encodeURIComponent(id)}`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    body.innerHTML = window.marked ? marked.parse(text) : '';
    if (!window.marked) body.append(el('pre', {}, text));
    for (const h of body.querySelectorAll('h1, h2, h3')) h.id = slug(h.textContent);
    if (heading) {
      const target = [...body.querySelectorAll('h2')].find((h) => h.textContent.trim() === heading.trim());
      target?.scrollIntoView({ block: 'start' });
    } else {
      window.scrollTo({ top: 0 });
    }
  } catch (err) {
    body.replaceChildren(el('p', { class: 'empty' }, `Could not open this document (${err.message}).`));
  }
}

function renderDocList() {
  if (!status) return;
  $('doc-list').replaceChildren(...status.docs.filter((d) => d.available).map((d) =>
    el('a', { href: '#docs', 'aria-current': currentDoc === d.id ? 'true' : null,
      onclick: (e) => { e.preventDefault(); openDoc(d.id); } }, d.title)));
}

// ---------------------------------------------------------------------------------------------
// Polling

function renderAll() {
  renderOverview();
  renderResearch();
  renderAnatomy();
  renderEvidence();
  renderDocList();
}

async function refresh() {
  try {
    const res = await fetch('/api/status', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    status = await res.json();
    lastOk = Date.now();
    renderAll();
    $('updated').textContent = `Live · updated ${status.generated_at.slice(11)}`;
    $('pulse').classList.remove('stale');
  } catch (err) {
    $('updated').textContent = `Not updating (${err.message})`;
    $('pulse').classList.add('stale');
  }
}

renderViewerToolbar();
showTab(location.hash.slice(1) || 'overview');
refresh();
setInterval(refresh, POLL_MS);
