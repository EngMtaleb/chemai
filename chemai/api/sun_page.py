"""The demo page for the sun-and-moon service (Project 4).

One page, no build step, no framework, no CDN - the same rule as the other pages.
Three parts: what the pilot record answered, a design explorer over the solar
and tidal sweep, and the declared tide (or time-of-day) test run on a log the
visitor brings. The limits come from /health and are printed under everything.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Desalination on the Sun - and the Moon</title>
<style>
  :root {
    --ink: #13252f; --muted: #5b6d77; --line: #dbe4e8; --bg: #f5f8f9;
    --card: #ffffff; --accent: #0f7c8c; --sun: #c27c0e; --warn: #b4531a; --good: #1f6f54;
    --soft: #eef3f5;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 34px 20px 10px; max-width: 1000px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: -0.2px; }
  h2 { margin: 0 0 4px; font-size: 18px; }
  .sub { color: var(--muted); max-width: 74ch; }
  main { max-width: 1000px; margin: 0 auto; padding: 12px 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 18px; margin-bottom: 18px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  button { font: inherit; padding: 9px 14px; border-radius: 9px; cursor: pointer;
           border: 1px solid var(--line); background: #fff; color: var(--ink); }
  button:hover { border-color: var(--accent); }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .55; cursor: wait; }
  button#run { background: var(--ink); border-color: var(--ink); color: #fff; }
  input[type=file] { display: none; }
  select { font: inherit; padding: 7px 8px; border: 1px solid var(--line); border-radius: 8px;
           background: #fff; color: var(--ink); max-width: 100%; }
  label.f { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted);
            text-transform: uppercase; letter-spacing: .04em; }
  .hint { color: var(--muted); font-size: 13px; margin-top: 10px; }
  .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .v { margin-top: 2px; font-variant-numeric: tabular-nums; }
  .big { font-size: 28px; font-weight: 650; line-height: 1.15; font-variant-numeric: tabular-nums; }
  .tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 12px; }
  .tile { border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  .tile .ans { font-size: 20px; font-weight: 650; color: var(--good); margin: 4px 0 2px; }
  .tile .num { font-size: 13px; color: var(--muted); margin-top: 6px; font-variant-numeric: tabular-nums; }
  .controls { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 12px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 16px; }
  .grid > div { border-top: 1px solid var(--line); padding-top: 10px; }
  .bar { height: 12px; background: var(--soft); border-radius: 999px; overflow: hidden; margin-top: 10px; }
  .bar > i { display: block; height: 100%; background: var(--sun); width: 0; transition: width .25s; }
  .verdict { font-size: 24px; font-weight: 650; margin: 2px 0 4px; }
  .pill { display: inline-block; font-size: 12px; padding: 2px 9px; border-radius: 999px;
          border: 1px solid var(--line); color: var(--muted); margin-left: 8px; vertical-align: middle; }
  .flags { margin-top: 12px; color: var(--warn); font-size: 14px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 12px; }
  th { text-align: left; font-size: 12px; color: var(--muted); text-transform: uppercase;
       letter-spacing: .04em; border-bottom: 1px solid var(--line); padding: 8px 6px; }
  td { border-bottom: 1px solid var(--line); padding: 8px 6px; font-variant-numeric: tabular-nums; }
  svg { width: 100%; height: 200px; display: block; margin-top: 10px; }
  .axis { font-size: 11px; fill: var(--muted); }
  .limits { color: var(--muted); font-size: 13px; }
  .limits li { margin-bottom: 6px; }
  code { background: var(--soft); padding: 1px 5px; border-radius: 5px; font-size: 13px; overflow-wrap: anywhere; }
  a { color: var(--accent); }
  @media (max-width: 760px) {
    .tiles { grid-template-columns: 1fr; }
    .controls { grid-template-columns: 1fr 1fr; }
    .grid { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>Desalination on the Sun &mdash; and the Moon</h1>
  <p class="sub">A seawater plant that runs on solar power stops at night and starts again every
  morning. Two questions follow. Does its pre-treatment membrane mind &mdash; the hour, the restart,
  or the tide outside? And how much plant, water tank and solar (or a tidal turbine) buys how much
  renewable energy?</p>
  <p class="sub"><b>Read this first:</b> the membrane answers come from one public seawater pilot
  (ultrafiltration, 2012&ndash;2015), tested the way it was declared before the tests ran. The design
  numbers come from a synthetic year with no site and no prices. The limits are at the bottom.</p>
</header>

<main>
  <div class="card">
    <h2>What 5,106 hours of a seawater pilot said</h2>
    <div class="hint" style="margin-top:2px">107 runs, 14,998 steady filtration stretches; the
      fouling rate of each stretch from the membrane's resistance.</div>
    <div class="tiles">
      <div class="tile"><div class="k">The tide</div>
        <div class="ans">None detected</div>
        <div>The test found no tide-locked rise and fall in the fouling.</div>
        <div class="num">p = 0.24 &middot; a 10&nbsp;% swing would have been caught 99.5&nbsp;% of the time</div></div>
      <div class="tile"><div class="k">The sunny hours</div>
        <div class="ans">No difference detected</div>
        <div>09&ndash;15&nbsp;h fouled like the rest of the day.</div>
        <div class="num">&minus;0.8&nbsp;% (&minus;2.5 to +1.2&nbsp;%) &middot; p = 0.75</div></div>
      <div class="tile"><div class="k">The restart</div>
        <div class="ans">No penalty detected</div>
        <div>The first 3&nbsp;h after a stop: a penalty above about 10&nbsp;% is ruled out.</div>
        <div class="num">+3.1&nbsp;% (&minus;3.9 to +10.0&nbsp;%) &middot; p = 0.38</div></div>
    </div>
    <div class="hint">So on this record the pre-treatment can follow the sun &mdash; stop at night,
      start in the morning &mdash; without a fouling cost bigger than about 10&nbsp;%. Reverse
      osmosis under daily stops is a separate question (see the limits).</div>
  </div>

  <div class="card">
    <h2>Size a plant on the sun</h2>
    <div class="hint" style="margin-top:2px">Per cubic metre a day of average water demand. A bigger
      plant runs fewer hours and makes its water while the sun shines; the tank carries the night.</div>
    <div class="controls">
      <label class="f">Weather<select id="weather"></select></label>
      <label class="f">Tidal turbine<select id="tidal_kw"></select></label>
      <label class="f">Plant size<select id="capacity"></select></label>
      <label class="f">Water tank<select id="storage_h"></select></label>
      <label class="f">Solar<select id="pv_kwp"></select></label>
    </div>
    <div class="grid">
      <div><div class="k">Renewable share</div><div class="big" id="ren">&ndash;</div>
        <div class="bar"><i id="renBar"></i></div></div>
      <div><div class="k">Grid energy</div><div class="big" id="grid">&ndash;</div>
        <div class="v" id="gridFrom"></div></div>
      <div><div class="k">Solar thrown away</div><div class="big" id="curt">&ndash;</div>
        <div class="v">tank full, plant at its limit</div></div>
      <div><div class="k">Plant runs</div><div class="big" id="hours">&ndash;</div>
        <div class="v">hours a day, on average</div></div>
    </div>
    <div class="grid">
      <div><div class="k">Per 1,000 m&sup3;/d: solar</div><div class="v" id="p_pv"></div></div>
      <div><div class="k">Water tank</div><div class="v" id="p_tank"></div></div>
      <div><div class="k">Grid electricity a year</div><div class="v" id="p_grid"></div></div>
      <div><div class="k">CO&#8322; avoided a year</div><div class="v" id="p_co2"></div></div>
    </div>
    <div class="flags" id="ruleNote"></div>
    <div class="hint">The share is the best schedule found with the whole year known in advance. A
      simple rule with no forecast &mdash; run on the sun first, then on the tank, then on the grid
      &mdash; landed within 0.2 points of it in the median design; with a tank under 8&nbsp;h,
      foresight is worth several points. CO&#8322; at 0.4&ndash;0.7&nbsp;kg
      per grid kWh. API: <code id="designUrl"></code></div>
  </div>

  <div class="card">
    <h2>Test your own log</h2>
    <div class="hint" style="margin-top:2px">The same declared test, pointed at any plant signal with a
      time stamp: a fouling rate, a pressure, a turbidity. Is it locked to the tide &mdash; or to the
      clock?</div>
    <div class="row" style="margin-top:12px">
      <button type="button" class="ex" data-ex="tide">Example: a log with a tide in it</button>
      <button type="button" class="ex" data-ex="clock">Example: a log with a daily wave</button>
      <label><input type="file" id="csv" accept=".csv,text/csv,.txt">
        <button type="button" onclick="document.getElementById('csv').click()">Upload a CSV</button></label>
    </div>
    <div class="row" style="margin-top:12px">
      <span class="k">Test for</span>
      <button type="button" class="kind on" data-kind="tide">the tide</button>
      <button type="button" class="kind" data-kind="clock">the time of day</button>
      <button type="button" id="run" disabled>Run the test</button>
      <span class="hint" id="source" style="margin:0">no log loaded</span>
    </div>
    <div class="hint">CSV columns <code>t,value</code> and optionally <code>run</code>; times in local
      standard time. At least 100 points over 3 days; a gap over 6 hours starts a new run. The examples
      are made up in your browser, so the answer is known.</div>

    <div id="result" hidden>
      <div style="margin-top:16px"><span class="verdict" id="verdict"></span><span class="pill" id="pills"></span></div>
      <div class="grid">
        <div><div class="k">Wave's share of the variance</div><div class="v" id="r2"></div></div>
        <div><div class="k">Wave size</div><div class="v" id="size"></div></div>
        <div><div class="k">Placebo periods</div><div class="v" id="plac"></div></div>
        <div><div class="k">Record</div><div class="v" id="rec"></div></div>
      </div>
      <table><thead><tr><th>Component</th><th>Period</th><th>Amplitude</th><th>Phase</th></tr></thead>
        <tbody id="comp"></tbody></table>
      <div class="k" style="margin-top:16px" id="foldTitle"></div>
      <svg id="fold" viewBox="0 0 960 200"></svg>
      <div class="hint">Raw averages, each run's mean taken off. The test itself also removes slow
        drifts (and, for the tide, the time of day) before it looks.</div>
      <div class="flags" id="notes"></div>
    </div>
  </div>

  <div class="card">
    <div class="k">What this cannot do</div>
    <ul class="limits" id="limits"></ul>
    <div class="hint">Method and evidence: <code>projects/p04_tide_fouling/VALIDATION.md</code> in
      <a href="https://github.com/EngMtaleb/chemai">github.com/EngMtaleb/chemai</a>.
      API documentation at <a href="docs">/sun/docs</a>. Pilot record: Zenodo 4630769 (CC0).</div>
  </div>
</main>

<script>
var $ = function (id) { return document.getElementById(id); };
var NS = 'http://www.w3.org/2000/svg';
var PERIODS = {tide: {M2: 12.4206, N2: 12.6583, O1: 25.8193, M4: 6.2103},
               clock: {D24: 24, D12: 12, D8: 8, D6: 6}};
var EPOCH = Date.UTC(2000, 0, 1);
var KEYS = ['weather', 'tidal_kw', 'capacity', 'storage_h', 'pv_kwp'];
var IDX = {weather: 0, capacity: 1, storage_h: 2, pv_kwp: 3, tidal_kw: 4};
var START = {weather: 'sunny', tidal_kw: '0', capacity: '3', storage_h: '16', pv_kwp: '0.7'};
var rows = [], sec = 3.26, kind = 'tide', records = null;

function fmt(x, d) { return (x === null || x === undefined) ? '–' : Number(x).toFixed(d); }
function pct(x, d) { return fmt(100 * x, d === undefined ? 0 : d) + ' %'; }
function el(name, attrs, text) {
  var e = document.createElementNS(NS, name);
  for (var k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

/* ---------- design explorer ---------- */
function label(key, v) {
  var x = parseFloat(v);
  if (key === 'weather') return v === 'sunny' ? 'Sunny year' : 'Cloudier year';
  if (key === 'tidal_kw') return x === 0 ? 'None' : x + ' kW per m³/d';
  if (key === 'capacity') return x === 1 ? '1× demand (runs all day)' : x + '× demand';
  if (key === 'storage_h') return x + ' h of demand';
  if (key === 'pv_kwp') return x + ' kWp per m³/d';
  return v;
}
function optionsFor(i) {
  var seen = {}, out = [];
  rows.forEach(function (r) {
    for (var j = 0; j < i; j++) {
      var kj = KEYS[j];
      if (String(r[IDX[kj]]) !== $(kj).value) return;
    }
    var v = String(r[IDX[KEYS[i]]]);
    if (!seen[v]) { seen[v] = 1; out.push(v); }
  });
  if (KEYS[i] !== 'weather') out.sort(function (a, b) { return parseFloat(a) - parseFloat(b); });
  return out;
}
function cascade(from) {
  for (var i = from; i < KEYS.length; i++) {
    var key = KEYS[i], s = $(key), keep = s.value || START[key], opts = optionsFor(i);
    s.innerHTML = '';
    opts.forEach(function (v) { var o = document.createElement('option'); o.value = v; o.textContent = label(key, v); s.appendChild(o); });
    if (opts.indexOf(keep) < 0) {
      var best = opts[0];
      if (key !== 'weather') opts.forEach(function (v) {
        if (Math.abs(parseFloat(v) - parseFloat(keep)) < Math.abs(parseFloat(best) - parseFloat(keep))) best = v; });
      keep = best;
    }
    s.value = keep;
  }
  showDesign();
}
function showDesign() {
  var q = KEYS.map(function (k) { return k + '=' + encodeURIComponent($(k).value); }).join('&');
  $('designUrl').textContent = 'GET /sun/design?' + q;
  fetch('design?' + q).then(function (r) { return r.json(); }).then(function (d) {
    if (d.detail) return;
    $('ren').textContent = pct(d.renewable_fraction);
    $('renBar').style.width = (100 * d.renewable_fraction).toFixed(1) + '%';
    $('grid').textContent = fmt(d.grid_kwh_m3, 2) + ' kWh/m³';
    $('gridFrom').textContent = 'from ' + fmt(sec, 2) + ' kWh/m³ all on the grid';
    $('curt').textContent = pct(d.curtailed_share);
    $('hours').textContent = fmt(d.run_hours_per_day, 1);
    var p = d.per_1000_m3_day;
    $('p_pv').textContent = p.pv_kwp + ' kWp' + (p.tidal_kw ? ' + ' + p.tidal_kw + ' kW tidal' : '');
    $('p_tank').textContent = p.tank_m3.toLocaleString() + ' m³, plant ' + p.plant_m3_day.toLocaleString() + ' m³/d';
    $('p_grid').textContent = Math.round(p.grid_mwh_year).toLocaleString() + ' MWh (from ' + Math.round(p.grid_mwh_year_without).toLocaleString() + ')';
    var saved = p.grid_mwh_year_without - p.grid_mwh_year;
    $('p_co2').textContent = Math.round(saved * 0.4).toLocaleString() + '–' + Math.round(saved * 0.7).toLocaleString() + ' t';
    if (d.rule_fraction === null) $('ruleNote').textContent = 'The no-forecast rule was not run with a turbine.';
    else {
      var gap = d.renewable_fraction - d.rule_fraction;
      $('ruleNote').textContent = gap > 0.02 ? 'With no forecast, the simple rule reaches ' + pct(d.rule_fraction) +
        ' here: a small tank leaves the schedule little room, so foresight pays.' : '';
    }
  });
}

/* ---------- test your own log ---------- */
function rng(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function iso(ms) { return new Date(ms).toISOString().slice(0, 19); }
function example(which) {
  var u = rng(which === 'tide' ? 7 : 11), out = [], e = 0;
  var gauss = function () { return Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()); };
  for (var run = 0; run < 10; run++) {
    var start = Date.UTC(2014, 0, 6) + run * 9 * 86400e3;
    for (var h = 0; h < 6 * 24; h += 0.5) {
      var ms = start + h * 3600e3, th = (ms - EPOCH) / 3600e3, hod = ((th % 24) + 24) % 24;
      e = 0.8 * e + 0.6 * gauss();
      var v = 2 + 0.002 * h + 0.25 * e;
      v += which === 'tide' ? 0.25 * Math.cos(2 * Math.PI * th / 12.4206 + 1)
                            : 0.25 * Math.cos(2 * Math.PI * (hod - 14) / 24);
      out.push({t: iso(ms), value: Math.round(v * 1e4) / 1e4, run: 'run ' + (run + 1)});
    }
  }
  return out;
}
function parseCsv(text) {
  var lines = text.split(/\r?\n/).filter(function (s) { return s.trim().length; });
  if (!lines.length) throw new Error('the file is empty');
  var sep = lines[0].indexOf(';') >= 0 && lines[0].indexOf(',') < 0 ? ';' : ',';
  var head = lines[0].split(sep).map(function (s) { return s.trim().toLowerCase().replace(/"/g, ''); });
  var it = 0, iv = 1, ir = -1, first = 0;
  if (isNaN(Date.parse(head[0].replace(' ', 'T')))) {
    first = 1;
    it = Math.max(0, head.findIndex(function (h) { return /^(t|time|timestamp|date|datetime)$/.test(h); }));
    var v = head.findIndex(function (h) { return h === 'value'; });
    iv = v >= 0 ? v : (it === 0 ? 1 : 0);
    ir = head.indexOf('run');
  }
  var out = [];
  for (var i = first; i < lines.length; i++) {
    var c = lines[i].split(sep).map(function (s) { return s.trim().replace(/"/g, ''); });
    var val = parseFloat(c[iv]);
    if (!c[it] || !isFinite(val)) continue;
    var p = {t: c[it].replace(/^(\d{4}-\d\d-\d\d) /, '$1T'), value: val};
    if (ir >= 0 && c[ir]) p.run = c[ir];
    out.push(p);
  }
  if (out.length > 200000) throw new Error(out.length + ' rows: resample to 200,000 or fewer');
  return out;
}
function load(recs, label) {
  records = recs;
  $('source').textContent = label + ' · ' + recs.length.toLocaleString() + ' points';
  $('run').disabled = false;
  $('result').hidden = true;
}
function setKind(k) {
  kind = k;
  document.querySelectorAll('.kind').forEach(function (b) { b.classList.toggle('on', b.dataset.kind === k); });
}
function fold() {
  var P = kind === 'tide' ? 12.4206 : 24, nb = 12, sum = [], cnt = [], mean = {}, n = {};
  records.forEach(function (r) { var k = r.run || ''; mean[k] = (mean[k] || 0) + r.value; n[k] = (n[k] || 0) + 1; });
  for (var i = 0; i < nb; i++) { sum.push(0); cnt.push(0); }
  records.forEach(function (r) {
    var ms = Date.parse(r.t.length === 16 ? r.t + ':00Z' : r.t + 'Z');
    if (isNaN(ms)) return;
    var th = (ms - EPOCH) / 3600e3, ph = ((th / P) % 1 + 1) % 1, b = Math.min(nb - 1, Math.floor(ph * nb));
    var k = r.run || '';
    sum[b] += r.value - mean[k] / n[k]; cnt[b] += 1;
  });
  var m = sum.map(function (s, i) { return cnt[i] ? s / cnt[i] : 0; });
  var svg = $('fold'); while (svg.firstChild) svg.removeChild(svg.firstChild);
  var W = Math.max(300, Math.round(svg.getBoundingClientRect().width) || 960), H = 200, L = 52, R = 12, T = 12, B = 28;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var lim = Math.max.apply(null, m.map(Math.abs)) * 1.15 || 1;
  var Y = function (v) { return T + (lim - v) / (2 * lim) * (H - T - B); };
  var bw = (W - L - R) / nb;
  svg.appendChild(el('line', {x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: '#c8d3d8'}));
  m.forEach(function (v, i) {
    var y0 = Y(0), y1 = Y(v);
    svg.appendChild(el('rect', {x: L + i * bw + 3, width: bw - 6, y: Math.min(y0, y1), height: Math.max(1, Math.abs(y1 - y0)),
                                fill: kind === 'tide' ? '#0f7c8c' : '#c27c0e', rx: 2}));
    var lab = kind === 'tide' ? Math.round(i * 30) + '°' : (i * 2) + ' h';
    if (bw < 40 && i % 2) return;
    svg.appendChild(el('text', {x: L + i * bw + bw / 2, y: H - 8, 'text-anchor': 'middle', 'class': 'axis'}, lab));
  });
  svg.appendChild(el('text', {x: L - 6, y: Y(lim / 1.15) + 4, 'text-anchor': 'end', 'class': 'axis'}, '+' + (lim / 1.15).toPrecision(2)));
  svg.appendChild(el('text', {x: L - 6, y: Y(-lim / 1.15) + 4, 'text-anchor': 'end', 'class': 'axis'}, '−' + (lim / 1.15).toPrecision(2)));
  $('foldTitle').textContent = kind === 'tide' ? 'Average by phase of the 12.42-hour lunar tide' : 'Average by hour of the day';
}
function runTest() {
  $('run').disabled = true; $('run').textContent = 'Running…';
  fetch('test', {method: 'POST', headers: {'Content-Type': 'application/json'},
                 body: JSON.stringify({records: records, kind: kind})})
    .then(function (r) { return r.json().then(function (j) { return {ok: r.ok, j: j}; }); })
    .then(function (x) { show(x.ok ? x.j : {status: 'error', detail: x.j.detail}); })
    .catch(function (e) { show({status: 'error', detail: String(e)}); })
    .then(function () { $('run').disabled = false; $('run').textContent = 'Run the test'; });
}
function show(d) {
  $('result').hidden = false;
  var what = kind === 'tide' ? 'the tide' : 'the time of day';
  $('comp').innerHTML = ''; $('notes').innerHTML = '';
  if (d.status === 'error') {
    $('verdict').textContent = 'The log was refused';
    $('pills').textContent = '';
    ['r2', 'size', 'plac', 'rec'].forEach(function (k) { $(k).textContent = '–'; });
    $('notes').textContent = typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail).slice(0, 400);
    return;
  }
  if (d.status === 'insufficient') {
    $('verdict').textContent = 'Not enough data to tell';
    $('pills').textContent = '';
    ['r2', 'size', 'plac'].forEach(function (k) { $(k).textContent = '–'; });
    $('rec').textContent = d.n + ' points, ' + d.n_blocks + ' blocks, ' + fmt(d.span_hours / 24, 1) + ' days';
    $('notes').textContent = d.notes.join(' ');
    return;
  }
  $('verdict').textContent = (d.verdict === 'locked' ? 'Locked to ' : 'Not locked to ') + what;
  $('pills').textContent = 'p = ' + fmt(d.p_value, 3) + ' · declared level 0.01';
  $('r2').textContent = pct(d.partial_r2, 2) + ' (chance alone: up to ' + pct(d.null_partial_r2_q95, 2) + ')';
  $('size').textContent = fmt(d.range_5_95, 3) + ' peak to trough' + (d.relative_range !== null ? ' · ' + pct(d.relative_range, 1) + ' of the median' : '');
  $('plac').textContent = 'p = ' + fmt(d.placebo_p_value, 3) + (d.placebo_p_value > 0.01 ? ' (quiet, as they should be)' : ' (NOT quiet)');
  $('rec').textContent = d.n.toLocaleString() + ' points, ' + d.n_blocks + ' blocks, ' + fmt(d.span_hours / 24, 1) + ' days';
  var per = PERIODS[kind];
  Object.keys(d.amplitudes || {}).forEach(function (k) {
    var tr = document.createElement('tr');
    [k, (+per[k].toFixed(2)) + ' h', fmt(d.amplitudes[k], 4), fmt(d.phases_deg[k], 0) + '°'].forEach(function (c) {
      var td = document.createElement('td'); td.textContent = c; tr.appendChild(td); });
    $('comp').appendChild(tr);
  });
  $('notes').textContent = (d.notes || []).join(' ');
  fold();
}

document.querySelectorAll('.ex').forEach(function (b) {
  b.onclick = function () { load(example(b.dataset.ex), b.textContent.replace('Example: ', 'Example – ')); };
});
document.querySelectorAll('.kind').forEach(function (b) { b.onclick = function () { setKind(b.dataset.kind); }; });
$('run').onclick = runTest;
$('csv').onchange = function (ev) {
  var f = ev.target.files[0]; if (!f) return;
  var rd = new FileReader();
  rd.onload = function () {
    try { load(parseCsv(rd.result), f.name); }
    catch (e) { records = null; $('run').disabled = true; $('source').textContent = String(e.message || e); }
  };
  rd.readAsText(f);
};
KEYS.forEach(function (k, i) { $(k).onchange = function () { cascade(i + 1); }; });

fetch('health').then(function (r) { return r.json(); }).then(function (h) {
  h.limits.forEach(function (s) { var li = document.createElement('li'); li.textContent = s; $('limits').appendChild(li); });
});
fetch('designs').then(function (r) { return r.json(); }).then(function (d) {
  rows = d.rows; sec = d.sec_kwh_m3; cascade(0);
});
</script>
</body>
</html>
"""
