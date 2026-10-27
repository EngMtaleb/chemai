"""The demo page for the when-to-stop service (Project 5).

One page, no build step, no framework, no CDN - the same rule as the other pages.
Three parts: the four stops the project judged, the batch rule run on an example
or on a log the visitor brings, and the square-root law for slow cycles. The
limits come from /health and are printed under everything.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>When to Stop</title>
<style>
  :root {
    --ink: #13252f; --muted: #5b6d77; --line: #dbe4e8; --bg: #f5f8f9;
    --card: #ffffff; --accent: #0f7c8c; --live: #13252f; --warn: #b4531a; --good: #1f6f54;
    --soft: #eef3f5; --orange: #c2610e;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 34px 20px 10px; max-width: 1000px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: -0.2px; }
  h2 { margin: 0 0 4px; font-size: 18px; }
  .sub { color: var(--muted); max-width: 76ch; }
  .rule { font-size: 18px; margin: 12px 0 6px; padding: 12px 14px; border-left: 4px solid var(--accent);
          background: var(--card); border-radius: 0 10px 10px 0; max-width: 76ch; }
  main { max-width: 1000px; margin: 0 auto; padding: 12px 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 18px; margin-bottom: 18px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
  button { font: inherit; padding: 9px 14px; border-radius: 9px; cursor: pointer;
           border: 1px solid var(--line); background: #fff; color: var(--ink); }
  button:hover { border-color: var(--accent); }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .55; cursor: wait; }
  button.go { background: var(--ink); border-color: var(--ink); color: #fff; }
  input[type=file] { display: none; }
  input[type=number] { font: inherit; padding: 7px 8px; border: 1px solid var(--line); border-radius: 8px;
                       width: 100%; background: #fff; color: var(--ink); }
  label.f { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted);
            text-transform: uppercase; letter-spacing: .04em; }
  .hint { color: var(--muted); font-size: 13px; margin-top: 10px; }
  .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .v { margin-top: 2px; font-variant-numeric: tabular-nums; }
  .big { font-size: 26px; font-weight: 650; line-height: 1.15; font-variant-numeric: tabular-nums; }
  .controls { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 14px; }
  .grid > div { border-top: 1px solid var(--line); padding-top: 10px; }
  .verdict { font-size: 19px; font-weight: 600; margin: 14px 0 4px; }
  .flags { margin-top: 10px; color: var(--warn); font-size: 14px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 12px; }
  th { text-align: left; font-size: 12px; color: var(--muted); text-transform: uppercase;
       letter-spacing: .04em; border-bottom: 1px solid var(--line); padding: 8px 6px; }
  td { border-bottom: 1px solid var(--line); padding: 9px 6px; vertical-align: top; }
  td.yes { color: var(--good); font-weight: 600; } td.no { color: var(--warn); font-weight: 600; }
  svg { width: 100%; height: 260px; display: block; margin-top: 12px; }
  .axis { font-size: 11px; fill: var(--muted); }
  .legend { font-size: 13px; color: var(--muted); margin-top: 6px; }
  .legend i { display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin: 0 6px 0 12px; }
  .limits { color: var(--muted); font-size: 13px; }
  .limits li { margin-bottom: 6px; }
  code { background: var(--soft); padding: 1px 5px; border-radius: 5px; font-size: 13px; overflow-wrap: anywhere; }
  a { color: var(--accent); }
  @media (max-width: 760px) {
    .controls { grid-template-columns: 1fr 1fr; }
    .grid { grid-template-columns: 1fr 1fr; }
    table { font-size: 13px; }
  }
</style>
</head>
<body>
<header>
  <h1>When to Stop</h1>
  <p class="sub">Every cycle in a desalination plant gets dearer the longer it runs &mdash; the loop
  concentrates, the membrane fouls &mdash; and ending it has a fixed price: a drain and its brine, a
  backwash, a chemical clean. One rule says when to end any of them:</p>
  <div class="rule"><b>Stop when the live cost of the next cubic metre reaches the average cost of the
  cycle so far.</b> If the live cost climbs in a straight line, the best cycle is
  &radic;(2&times;fixed cost &divide; climb rate): twice as costly to stop, &radic;2 times longer &mdash; not twice.</div>
  <p class="sub"><b>Read this first:</b> the batch rule was declared, then tested once on 153 batches of a
  public closed-circuit RO testbed it had not seen. The testbed's pumps make its energy ten times a
  real plant's: the rule carries over, the numbers do not. Prices below are yours to set.</p>
</header>

<main>
  <div class="card">
    <h2>Four stops, one rule</h2>
    <div class="hint" style="margin-top:2px">What the project found when it asked, at each scale, whether
      the energy (and water) the cycle spends is the reason to stop it.</div>
    <table>
      <tr><th>Decision</th><th>Every</th><th>Energy the reason?</th><th>What the record said</th></tr>
      <tr><td>End a closed-circuit RO batch</td><td>~1 hour</td><td class="yes">Yes</td>
        <td>Where the record shows a held-out batch's best end (97 of 153 at b&nbsp;=&nbsp;1, 10 at b&nbsp;=&nbsp;10), the rule
        ended it a median <b>0.02&ndash;0.07&nbsp;%</b> above that end, found in hindsight.
        The published optimisers ended 90&ndash;100&nbsp;% of theirs while the cost was still falling.</td></tr>
      <tr><td>Backwash a UF element</td><td>~20 minutes</td><td class="no">No</td>
        <td>Energy and water alone would filter <b>~16&times;</b> more per backwash; the pilot spent 29&nbsp;% of
        its feed on backwash. The interval was set by something else.</td></tr>
      <tr><td>Clean a UF membrane chemically</td><td>~4&ndash;5 months</td><td class="yes">About</td>
        <td>The resistance doubles between cleans; energy alone puts the clean at 1.1&ndash;1.3&times; the
        interval actually run.</td></tr>
      <tr><td>Clean an RO train</td><td>weeks&ndash;months</td><td class="no">No</td>
        <td>In 15 cycles of one seawater plant, energy savings alone never covered a cleaning (mid case);
        in 9 the penalty had no rising trend at all. Longer cycles left more pressure drop behind.</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>End a batch</h2>
    <div class="hint" style="margin-top:2px">A batch log from start-up: time, pump power and permeate flow
      (reject conductivity optional). The rule runs on it as if live; the best end in hindsight is shown
      beside it.</div>
    <div class="row">
      <button id="exLong" class="on">Long test batch</button>
      <button id="exShort">Short batch that stopped early</button>
      <button id="upload">Your CSV&hellip;</button><input type="file" id="file" accept=".csv,text/csv">
    </div>
    <div class="controls">
      <label class="f">Brine price b, kWh per m&sup3;<input type="number" id="b" value="1" min="0" step="0.5"></label>
      <label class="f">Last drain, kWh<input type="number" id="drain" value="0.048" min="0" step="0.005"></label>
      <label class="f">Brine per batch, L<input type="number" id="brine" value="14.5" min="0" step="0.5"></label>
      <label class="f">&nbsp;<button class="go" id="runBatch">Run the rule</button></label>
    </div>
    <div class="hint">b = brine price &divide; electricity price: brine at $0.10/m&sup3; with power at
      $0.10/kWh is b = 1; brine at $1/m&sup3; is b = 10.</div>
    <div class="verdict" id="bVerdict"></div>
    <div class="grid">
      <div><div class="k">The rule stops</div><div class="big" id="bStop">&ndash;</div><div class="v" id="bStop2"></div></div>
      <div><div class="k">Best in hindsight</div><div class="big" id="bBest">&ndash;</div><div class="v" id="bBest2"></div></div>
      <div><div class="k">The log ends</div><div class="big" id="bEnd">&ndash;</div><div class="v" id="bEnd2"></div></div>
      <div><div class="k">Square-root forecast</div><div class="big" id="bLaw">&ndash;</div><div class="v" id="bLaw2"></div></div>
    </div>
    <svg id="bChart" viewBox="0 0 900 260" preserveAspectRatio="none"></svg>
    <div class="legend"><i style="background:var(--live)"></i>live cost of the next m&sup3;
      <i style="background:var(--accent)"></i>average cost so far, drain and brine included</div>
    <div class="flags" id="bFlags"></div>
    <div class="hint">CSV columns: <code>t</code> (seconds) or <code>time</code> (a date-time),
      <code>power_kw</code>, <code>permeate_lpm</code>, optional <code>conductivity</code>. The rule assumes
      the batch holds its flux and recirculation setpoints.</div>
  </div>

  <div class="card">
    <h2>Clean or wait</h2>
    <div class="hint" style="margin-top:2px">For slow cycles: what a cleaning costs (per m&sup3;/day of
      capacity), how fast the extra cost of running fouled climbs, and the interval you run.</div>
    <div class="controls">
      <label class="f">Cost of a cleaning, $ per m&sup3;/day<input type="number" id="F" value="1" min="0.001" step="0.1"></label>
      <label class="f">Climb, &cent;/m&sup3; per month<input type="number" id="a" value="0.3" min="0" step="0.05"></label>
      <label class="f">Interval you run, days<input type="number" id="T" value="45" min="1" step="1"></label>
      <label class="f">&nbsp;<button class="go" id="runLaw">Compare</button></label>
    </div>
    <div class="grid">
      <div><div class="k">Best interval</div><div class="big" id="lStar">&ndash;</div><div class="v">days, &radic;(2F/a)</div></div>
      <div><div class="k">Yours costs</div><div class="big" id="lExtra">&ndash;</div><div class="v">more per m&sup3; than the best</div></div>
      <div><div class="k">Implied cleaning cost</div><div class="big" id="lImp">&ndash;</div><div class="v">that makes your interval the best</div></div>
      <div><div class="k">Average extra cost</div><div class="big" id="lJ">&ndash;</div><div class="v">&cent;/m&sup3; at your interval</div></div>
    </div>
    <div class="row">
      <span class="k" style="margin-right:6px">Real cycles:</span>
      <button id="exRO">RO train, 42 days</button>
      <button id="exUF">UF membrane, 132 days</button>
    </div>
    <div class="verdict" id="iVerdict"></div>
    <svg id="iChart" viewBox="0 0 900 220" preserveAspectRatio="none"></svg>
    <div class="legend"><i style="background:var(--live)"></i>live extra cost, $/m&sup3;
      <i style="background:var(--accent)"></i>average cost of the cycle, cleaning included</div>
  </div>

  <div class="card">
    <h2>Limits</h2>
    <ul class="limits" id="limits"><li>loading&hellip;</li></ul>
    <div class="hint" id="sources"></div>
    <div class="hint">API: <a href="docs">docs</a> &middot; <code>POST batch</code> &middot;
      <code>POST interval</code> &middot; <code>GET law</code> &middot; <code>GET health</code></div>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
let EX = null, current = null;

function fmt(x, d) { return (x === null || x === undefined || !isFinite(x)) ? '–' : Number(x).toFixed(d); }

function chart(svg, xs, series, markers, xlabel, ylabel) {
  const W = Math.max(320, svg.clientWidth || 900), H = svg.clientHeight || 260;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const L = 54, R = 12, T = 24, B = 34;
  const ys = [];
  series.forEach(s => s.y.forEach(v => { if (v !== null && isFinite(v)) ys.push(v); }));
  markers.forEach(m => { if (m && isFinite(m.y)) ys.push(m.y); });
  if (!ys.length) { svg.innerHTML = ''; return; }
  ys.sort((a, b) => a - b);
  let lo = ys[Math.floor(ys.length * 0.01)], hi = ys[Math.floor(ys.length * 0.97)];
  markers.forEach(m => { if (m && isFinite(m.y)) { lo = Math.min(lo, m.y); hi = Math.max(hi, m.y); } });
  const pad = (hi - lo) * 0.12 || 1; lo -= pad; hi += pad;
  const xv = xs.filter(v => v !== null && isFinite(v));
  const x0 = Math.min(...xv), x1 = Math.max(...xv);
  const X = v => L + (v - x0) / ((x1 - x0) || 1) * (W - L - R);
  const Y = v => T + (1 - (v - lo) / ((hi - lo) || 1)) * (H - T - B);
  let h = '';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, y = Y(v);
    h += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#e7edf0"/>`;
    h += `<text class="axis" x="${L - 6}" y="${y + 4}" text-anchor="end">${v.toFixed(Math.abs(hi - lo) < 0.05 ? 4 : 2)}</text>`;
  }
  for (let i = 0; i <= 5; i++) {
    const v = x0 + (x1 - x0) * i / 5;
    h += `<text class="axis" x="${X(v)}" y="${H - 14}" text-anchor="middle">${v.toFixed(0)}</text>`;
  }
  h += `<text class="axis" x="${(L + W - R) / 2}" y="${H - 1}" text-anchor="middle">${xlabel}</text>`;
  h += `<text class="axis" x="4" y="12">${ylabel}</text>`;
  series.forEach(s => {
    let d = '', pen = false;
    s.y.forEach((v, i) => {
      if (v === null || !isFinite(v) || xs[i] === null || v < lo || v > hi) { pen = false; return; }
      d += (pen ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true;
    });
    h += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.w || 2}"/>`;
  });
  markers.forEach(m => {
    if (!m || !isFinite(m.x) || !isFinite(m.y)) return;
    h += `<circle cx="${X(m.x)}" cy="${Y(m.y)}" r="6" fill="${m.fill || m.color}" stroke="${m.color}" stroke-width="2"/>`;
    h += `<text class="axis" x="${X(m.x) + 9}" y="${Y(m.y) + (m.dy || -9)}" fill="${m.color}">${m.label}</text>`;
  });
  svg.innerHTML = h;
}

function setActive(id) {
  ['exLong', 'exShort', 'upload'].forEach(k => $(k).classList.toggle('on', k === id));
}

function loadBatch(key) {
  const e = EX[key];
  current = {t: e.t, power_kw: e.power_kw, permeate_lpm: e.permeate_lpm, conductivity: e.conductivity, name: e.name};
  $('drain').value = e.drain_kwh; $('brine').value = e.brine_l;
  runBatch();
}

async function runBatch() {
  if (!current) return;
  const body = {t: current.t, power_kw: current.power_kw, permeate_lpm: current.permeate_lpm,
                conductivity: current.conductivity || null,
                b: Number($('b').value), drain_kwh: Number($('drain').value), brine_l: Number($('brine').value)};
  $('runBatch').disabled = true;
  try {
    const r = await fetch('batch', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) { $('bVerdict').textContent = 'Could not read the log: ' + (j.detail ? JSON.stringify(j.detail) : r.status); return; }
    showBatch(j);
  } finally { $('runBatch').disabled = false; }
}

function cond(p) { return p && p.conductivity !== null && p.conductivity !== undefined ? ` · ${fmt(p.conductivity, 1)} mS/cm` : ''; }

function showBatch(j) {
  $('bVerdict').textContent = (current.name ? current.name + ': ' : '') + j.verdict;
  $('bStop').textContent = j.stop ? fmt(j.stop.minutes, 0) + ' min' : 'not yet';
  $('bStop2').textContent = j.stop ? `${fmt(j.stop.permeate_l, 0)} L${cond(j.stop)} · +${fmt(j.regret_pct, 2)} %` : 'the cost was still falling';
  $('bBest').textContent = j.best_in_hindsight ? fmt(j.best_in_hindsight.minutes, 0) + ' min' : '–';
  $('bBest2').textContent = j.best_in_hindsight ? `${fmt(j.best_in_hindsight.average_kwh_m3, 2)} kWh/m³${cond(j.best_in_hindsight)}` : '';
  $('bEnd').textContent = j.end ? fmt(j.end.minutes, 0) + ' min' : '–';
  $('bEnd2').textContent = j.end ? `+${fmt(j.end_regret_pct, 2)} % above the best${cond(j.end)}` : '';
  if (j.law) {
    $('bLaw').textContent = j.law.forecast_minutes !== null ? fmt(j.law.forecast_minutes, 0) + ' min' : '–';
    $('bLaw2').textContent = `${fmt(j.law.v_star_l, 0)} L, from minutes ${j.law.fitted_minutes[0]}–${j.law.fitted_minutes[1]}`;
  } else { $('bLaw').textContent = '–'; $('bLaw2').textContent = 'needs 20 minutes of log'; }
  const c = j.curve, mk = [];
  if (j.best_in_hindsight) mk.push({x: j.best_in_hindsight.permeate_l, y: j.best_in_hindsight.average_kwh_m3, color: '#1f6f54', fill: '#fff', label: 'best', dy: 18});
  if (j.stop) mk.push({x: j.stop.permeate_l, y: j.stop.average_kwh_m3, color: '#0f7c8c', label: 'rule stops'});
  chart($('bChart'), c.permeate_l, [{y: c.live, color: '#13252f', w: 1.6}, {y: c.average, color: '#0f7c8c', w: 2.4}],
        mk, 'permeate in this batch, L', 'kWh/m³');
  $('bFlags').textContent = (j.notes || []).filter(n => !n.startsWith('fixed cost')).join(' · ');
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  const head = lines[0].split(/[,;\t]/).map(s => s.trim().toLowerCase());
  const col = (n) => head.indexOf(n);
  const it = col('t'), itime = col('time'), ip = col('power_kw'), iq = col('permeate_lpm'), ic = col('conductivity');
  if ((it < 0 && itime < 0) || ip < 0 || iq < 0) throw new Error('need columns t or time, power_kw, permeate_lpm');
  const t = [], p = [], q = [], c = [];
  let t0 = null;
  for (const line of lines.slice(1)) {
    const f = line.split(/[,;\t]/);
    let tv = it >= 0 ? Number(f[it]) : Date.parse(f[itime].trim().replace(' ', 'T')) / 1000;
    if (!isFinite(tv)) continue;
    if (t0 === null) t0 = tv;
    t.push(tv - t0); p.push(Number(f[ip])); q.push(Number(f[iq])); if (ic >= 0) c.push(Number(f[ic]));
  }
  return {t, power_kw: p, permeate_lpm: q, conductivity: ic >= 0 ? c : null, name: 'your log'};
}

function law() {
  const F = Number($('F').value), aMonth = Number($('a').value), T = Number($('T').value);
  const a = aMonth / 100 / 30;                     // $/m3 per day
  if (!(F > 0) || !(a > 0) || !(T > 0)) { $('lStar').textContent = '–'; return; }
  const ts = Math.sqrt(2 * F / a), J = (x) => F / x + a * x / 2;
  $('lStar').textContent = fmt(ts, 0);
  $('lExtra').textContent = '+' + fmt((J(T) - J(ts)) * 100, 2) + ' ¢';
  $('lImp').textContent = '$' + fmt(a * T * T / 2, 3);
  $('lJ').textContent = fmt(J(T) * 100, 2);
}

async function runInterval(key) {
  const e = EX[key];
  ['exRO', 'exUF'].forEach(k => $(k).classList.toggle('on', k === (key === 'ro_cycle' ? 'exRO' : 'exUF')));
  const r = await fetch('interval', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({day: e.day, live_cost: e.live_cost, fixed_cost: e.fixed_cost})});
  const j = await r.json();
  $('iVerdict').textContent = `${e.name}: ${j.verdict}` + (j.t_star_days ? ` (actual ${fmt(e.actual_days, 0)} days; implied cleaning cost $${fmt(j.implied_fixed_cost, 3)} per m³/day against $${e.fixed_cost} assumed)` : '');
  chart($('iChart'), j.curve.day, [{y: j.curve.live, color: '#13252f', w: 1.6}, {y: j.curve.average, color: '#0f7c8c', w: 2.4}], [],
        'days since the last cleaning', '$/m³');
}

async function init() {
  EX = await (await fetch('examples')).json();
  loadBatch('batch_long');
  law();
  runInterval('ro_cycle');
  const h = await (await fetch('health')).json();
  $('limits').innerHTML = h.limits.map(l => `<li>${l}</li>`).join('');
  $('sources').textContent = 'Data: ' + Object.values(h.sources).join(' · ');
}

$('exLong').onclick = () => { setActive('exLong'); $('b').value = 1; loadBatch('batch_long'); };
$('exShort').onclick = () => { setActive('exShort'); $('b').value = 1; loadBatch('batch_short'); };
$('upload').onclick = () => $('file').click();
$('file').onchange = async (ev) => {
  const f = ev.target.files[0]; if (!f) return;
  try { current = parseCSV(await f.text()); setActive('upload'); runBatch(); }
  catch (err) { $('bVerdict').textContent = 'Could not read the file: ' + err.message; }
};
$('runBatch').onclick = runBatch;
['F', 'a', 'T'].forEach(k => $(k).oninput = law);
$('runLaw').onclick = law;
$('exRO').onclick = () => runInterval('ro_cycle');
$('exUF').onclick = () => runInterval('uf_interval');
window.addEventListener('resize', () => { if (current) runBatch(); });
init();
</script>
</body>
</html>
"""
