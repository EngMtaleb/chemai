"""The demo page for the RO fouling service.

One page, no build step, no framework, no CDN - the same rule as the loop page.
It calls the endpoints an engineer would call, draws the cycle and the forecast
cone as inline SVG, and prints the limits under the answer.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RO Fouling and Cleaning</title>
<style>
  :root {
    --ink: #13252f; --muted: #5b6d77; --line: #dbe4e8; --bg: #f5f8f9;
    --card: #ffffff; --accent: #0f7c8c; --warn: #b4531a; --bad: #a33246; --good: #1f6f54;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 34px 20px 10px; max-width: 1000px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: -0.2px; }
  .sub { color: var(--muted); max-width: 72ch; }
  main { max-width: 1000px; margin: 0 auto; padding: 12px 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 18px; margin-bottom: 18px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  button { font: inherit; padding: 9px 14px; border-radius: 9px; cursor: pointer;
           border: 1px solid var(--line); background: #fff; color: var(--ink); }
  button:hover { border-color: var(--accent); }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  input[type=file] { display: none; }
  input[type=number] { font: inherit; width: 80px; padding: 6px 8px; border: 1px solid var(--line);
                       border-radius: 8px; }
  .hint { color: var(--muted); font-size: 13px; margin-top: 10px; }
  svg { width: 100%; height: 260px; display: block; }
  .axis { font-size: 11px; fill: var(--muted); }
  .verdict { font-size: 24px; font-weight: 650; margin: 2px 0 4px; }
  .pill { display: inline-block; font-size: 12px; padding: 2px 9px; border-radius: 999px;
          border: 1px solid var(--line); color: var(--muted); margin-left: 8px; vertical-align: middle; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 14px; }
  .grid > div { border-top: 1px solid var(--line); padding-top: 10px; }
  .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .v { margin-top: 2px; font-variant-numeric: tabular-nums; }
  .flags { margin-top: 12px; color: var(--warn); font-size: 14px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { text-align: left; font-size: 12px; color: var(--muted); text-transform: uppercase;
       letter-spacing: .04em; border-bottom: 1px solid var(--line); padding: 8px 6px; }
  td { border-bottom: 1px solid var(--line); padding: 9px 6px; vertical-align: top; }
  td small { color: var(--warn); display: block; margin-top: 4px; }
  .limits { color: var(--muted); font-size: 13px; }
  .limits li { margin-bottom: 6px; }
  code { background: #eef3f5; padding: 1px 5px; border-radius: 5px; font-size: 13px; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr 1fr; } svg { height: 210px; } }
</style>
</head>
<body>
<header>
  <h1>RO Fouling and Cleaning</h1>
  <p class="sub">Give it the normalised feed-channel pressure drop of one RO train since its
  last cleaning. It answers when the train reaches the review level for a cleaning, with a
  P10&ndash;P90 range, and says when the data cannot carry a date at all.</p>
  <p class="sub"><b>Read this first:</b> on the one real plant it was tested on, the date did
  not beat a simple rule &mdash; &ldquo;plan a cleaning when the last week sits within 0.03 of
  the level&rdquo;. The weekly plan uses both. The limits are listed under the answer.</p>
</header>

<main>
  <div class="card">
    <div class="row" id="examples"></div>
    <div class="row" style="margin-top:10px">
      <label><input type="file" id="csv" accept=".csv,text/csv">
        <button type="button" onclick="document.getElementById('csv').click()">Upload a CSV</button></label>
      <span class="hint" id="source">examples are simulated, so the cause is known</span>
    </div>
    <div class="hint">CSV: columns <code>day,dp_norm</code> &mdash; days since the last cleaning
      and DP_norm over start-up (1.15 = +15 %). Review level:
      <input id="lvl" type="number" value="1.15" min="1.01" max="2.5" step="0.01"></div>
  </div>

  <div class="card" id="chartCard" hidden>
    <div class="k">DP_norm since the last cleaning <span class="pill" id="window"></span></div>
    <svg id="chart" viewBox="0 0 960 260" preserveAspectRatio="none"></svg>
    <div class="hint">Dots: daily values. Shaded: the days the line was fitted on. Cone: where the
      fitted line reaches the review level at P10, P50 and P90.</div>
  </div>

  <div class="card" id="result" hidden>
    <div><span class="verdict" id="verdict"></span><span class="pill" id="conf"></span></div>
    <div class="grid">
      <div><div class="k">Now</div><div class="v" id="now"></div></div>
      <div><div class="k">Trend</div><div class="v" id="rate"></div></div>
      <div><div class="k">Review level in</div><div class="v" id="days"></div></div>
      <div><div class="k">Extrapolation</div><div class="v" id="ratio"></div></div>
    </div>
    <div class="flags" id="flags"></div>
  </div>

  <div class="card" id="planCard" hidden>
    <div class="k">This week's CIP plan</div>
    <table><thead><tr><th>Train</th><th>Action</th><th>Week</th><th>Owner</th><th>Confidence</th></tr></thead>
      <tbody id="planBody"></tbody></table>
    <div class="hint" id="planNotes"></div>
  </div>

  <div class="card">
    <div class="k">What this cannot do</div>
    <ul class="limits" id="limits"></ul>
    <div class="hint">Method and evidence: <code>projects/p03_ro_fouling/VALIDATION.md</code> in
      <a href="https://github.com/EngMtaleb/chemai">github.com/EngMtaleb/chemai</a>.
      API documentation at <a href="docs">/ro/docs</a>.</div>
  </div>
</main>

<script>
var $ = function (id) { return document.getElementById(id); };
var NS = 'http://www.w3.org/2000/svg';

function el(name, attrs, text) {
  var e = document.createElementNS(NS, name);
  for (var k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

function draw(cycle, f) {
  var svg = $('chart'); while (svg.firstChild) svg.removeChild(svg.firstChild);
  var W = 960, H = 260, L = 52, R = 18, T = 14, B = 30;
  var lvl = f ? f.review_level : parseFloat($('lvl').value);
  var tMax = Math.max.apply(null, cycle.day);
  var far = tMax;
  if (f && f.days_hi !== null && f.days_hi !== undefined) far = tMax + f.days_hi;
  else if (f && f.days_to_level !== null && f.days_to_level !== undefined) far = tMax + f.days_to_level * 1.6;
  far = Math.max(far, tMax * 1.25, tMax + 7);
  var ys = cycle.dp_norm.concat([lvl]);
  var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
  var pad = (hi - lo) * 0.12 || 0.02; lo -= pad; hi += pad;
  var X = function (t) { return L + t / far * (W - L - R); };
  var Y = function (v) { return H - B - (v - lo) / (hi - lo) * (H - T - B); };
  for (var i = 0; i <= 4; i++) {
    var v = lo + (hi - lo) * i / 4;
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: '#edf2f4' }));
    svg.appendChild(el('text', { x: L - 6, y: Y(v) + 4, 'text-anchor': 'end', 'class': 'axis' }, v.toFixed(2)));
  }
  var step = far > 120 ? 30 : (far > 60 ? 14 : 7);
  for (var d = 0; d <= far; d += step)
    svg.appendChild(el('text', { x: X(d), y: H - 10, 'text-anchor': 'middle', 'class': 'axis' }, d + ' d'));
  if (f && f.fit_window && f.fit_window.length === 2)
    svg.appendChild(el('rect', { x: X(f.fit_window[0]), y: T, width: X(f.fit_window[1]) - X(f.fit_window[0]),
                                 height: H - T - B, fill: '#e8f4f6' }));
  svg.appendChild(el('line', { x1: L, x2: W - R, y1: Y(lvl), y2: Y(lvl), stroke: '#a33246',
                               'stroke-dasharray': '6 5', 'stroke-width': 1.5 }));
  svg.appendChild(el('text', { x: W - R - 4, y: Y(lvl) - 6, 'text-anchor': 'end', fill: '#a33246',
                               'font-size': 12 }, 'review level ' + lvl.toFixed(2)));
  if (f && f.level_now !== null && f.status === 'crossing') {
    var t0 = tMax, y0 = f.level_now;
    var cone = [f.days_lo, f.days_to_level, f.days_hi];
    var pts = [X(t0) + ',' + Y(y0)];
    if (cone[0] !== null) pts.push(X(t0 + cone[0]) + ',' + Y(lvl));
    pts.push(X(t0 + (cone[2] !== null ? cone[2] : far - t0)) + ',' + Y(cone[2] !== null ? lvl : y0 + (lvl - y0) * (far - t0) / Math.max(cone[1], 1)));
    svg.appendChild(el('polygon', { points: pts.join(' '), fill: '#0f7c8c', opacity: 0.12 }));
    svg.appendChild(el('line', { x1: X(t0), y1: Y(y0), x2: X(t0 + cone[1]), y2: Y(lvl), stroke: '#0f7c8c',
                                 'stroke-width': 2 }));
    svg.appendChild(el('circle', { cx: X(t0 + cone[1]), cy: Y(lvl), r: 5, fill: '#fff', stroke: '#0f7c8c',
                                   'stroke-width': 2 }));
  }
  for (var j = 0; j < cycle.day.length; j++)
    svg.appendChild(el('circle', { cx: X(cycle.day[j]), cy: Y(cycle.dp_norm[j]), r: 2.6, fill: '#3d6f80' }));
  $('window').textContent = (cycle.name ? cycle.name + ' · ' : '') + cycle.day.length + ' days recorded, last on day ' + tMax;
  $('chartCard').hidden = false;
}

var WORDS = { at_or_above: 'At or above the review level', crossing: 'Review level ahead',
              not_in_sight: 'Not in sight', falling: 'Falling - no reason to clean',
              insufficient: 'Not enough data yet' };
var COLOUR = { at_or_above: '#a33246', crossing: '#0f7c8c', not_in_sight: '#1f6f54',
               falling: '#1f6f54', insufficient: '#b4531a' };
function num(v, d) { return v === null || v === undefined ? '-' : v.toFixed(d); }

function show(f, ok) {
  if (!ok) { $('verdict').textContent = 'Rejected: ' + (f.detail || 'bad request');
    $('verdict').style.color = '#b4531a'; $('flags').textContent = ''; $('result').hidden = false; return; }
  $('verdict').textContent = WORDS[f.status] || f.status;
  $('verdict').style.color = COLOUR[f.status] || '#13252f';
  $('conf').textContent = f.in_envelope ? 'inside the tested envelope' : 'flagged: weak';
  $('now').textContent = 'DP_norm ' + num(f.level_now, 3);
  $('rate').textContent = f.rate_per_week === null ? '-' :
    (f.rate_per_week * 100).toFixed(2) + ' %/week (' + (f.rate_lo * 100).toFixed(2) + ' to ' + (f.rate_hi * 100).toFixed(2) + ')';
  $('days').textContent = f.status === 'crossing' ? num(f.days_to_level, 0) + ' d  (P10 ' + num(f.days_lo, 0) +
    ', P90 ' + (f.days_hi === null ? '>180' : num(f.days_hi, 0)) + ')' : (f.status === 'at_or_above' ? 'now' : '-');
  $('ratio').textContent = f.extrapolation_ratio === null ? '-' : f.extrapolation_ratio.toFixed(2) + ' x the window';
  $('flags').textContent = f.warnings.length ? f.warnings.join(' · ') : '';
  $('result').hidden = false;
}

function forecast(cycle) {
  $('planCard').hidden = true;
  var body = { cycle: cycle, review_level: parseFloat($('lvl').value) || 1.15 };
  var ok = true;
  fetch('forecast', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    .then(function (r) { ok = r.ok; return r.json(); })
    .then(function (f) { draw(cycle, ok ? f : null); show(f, ok); })
    .catch(function (e) { show({ detail: e.message }, false); });
}

function plan(payload) {
  $('result').hidden = true;
  payload.review_level = parseFloat($('lvl').value) || 1.15;
  fetch('plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    .then(function (r) { return r.json(); })
    .then(function (p) {
      var first = p.rows.length ? p.rows[0].train : null;     // draw the most urgent train
      draw(payload.trains.filter(function (t) { return t.name === first; })[0] || payload.trains[0], null);
      $('planBody').innerHTML = p.rows.map(function (r) {
        return '<tr><td><b>' + r.train + '</b></td><td>' + r.action + '<br><span style="color:#5b6d77">' +
          r.evidence + '</span>' + (r.flags.length ? '<small>' + r.flags.join('<br>') + '</small>' : '') +
          '</td><td>' + (r.week === null ? '-' : r.week) + '</td><td>' + r.owner + '</td><td>' + r.confidence + '</td></tr>';
      }).join('');
      $('planNotes').textContent = p.notes.join(' ');
      $('planCard').hidden = false;
    });
}

$('csv').addEventListener('change', function (e) {
  var file = e.target.files[0]; if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      var rows = reader.result.trim().split(/\r?\n/).map(function (r) { return r.split(/[,;\t]/); });
      var head = rows[0].map(function (h) { return h.trim().toLowerCase(); });
      var di = head.indexOf('day'), xi = head.indexOf('dp_norm');
      if (di < 0 || xi < 0) throw new Error('need columns day, dp_norm');
      var cyc = { name: file.name, day: [], dp_norm: [] };
      rows.slice(1).forEach(function (r) { var d = parseFloat(r[di]), x = parseFloat(r[xi]);
        if (isFinite(d) && isFinite(x)) { cyc.day.push(d); cyc.dp_norm.push(x); } });
      $('source').textContent = file.name + ' - ' + cyc.day.length + ' days';
      forecast(cyc);
    } catch (err) { $('source').textContent = 'could not read that file: ' + err.message; }
  };
  reader.readAsText(file);
});

fetch('health').then(function (r) { return r.json(); }).then(function (h) {
  $('limits').innerHTML = h.limits.map(function (l) { return '<li>' + l + '</li>'; }).join('');
});

fetch('examples').then(function (r) { return r.json(); }).then(function (list) {
  $('examples').innerHTML = list.map(function (e) {
    return '<button type="button" data-key="' + e.key + '" title="' + e.blurb + '">' + e.title + '</button>';
  }).join('');
  $('examples').addEventListener('click', function (ev) {
    var btn = ev.target.closest ? ev.target.closest('button') : null; if (!btn) return;
    var all = document.querySelectorAll('#examples button');
    for (var i = 0; i < all.length; i++) all[i].classList.remove('on');
    btn.classList.add('on');
    fetch('examples/' + btn.getAttribute('data-key')).then(function (r) { return r.json(); }).then(function (ex) {
      $('source').textContent = ex.blurb;
      if (ex.kind === 'plan') plan(ex.payload); else forecast(ex.payload.cycle);
    });
  });
  $('examples').getElementsByTagName('button')[0].click();
});
</script>
</body>
</html>
"""
