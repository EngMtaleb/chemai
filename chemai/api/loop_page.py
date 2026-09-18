"""The demo page for the loop-diagnosis service.

One page, no build step, no framework, no CDN. It calls the same endpoints an
engineer would call, so what a visitor sees is what the service actually says -
including its limits, which are printed at the bottom rather than hidden.

The charts are drawn as inline SVG. A plotting library would be more capable
and would also mean a build step and a third-party script on a page whose whole
point is that it is simple enough to trust.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Control Loop Diagnosis</title>
<style>
  :root {
    --ink: #15211c; --muted: #5d6b64; --line: #dfe5e1; --bg: #f7f9f8;
    --card: #ffffff; --accent: #1f6f54; --warn: #b4531a; --bad: #9c2b2b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 34px 20px 10px; max-width: 980px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: -0.2px; }
  .sub { color: var(--muted); max-width: 70ch; }
  main { max-width: 980px; margin: 0 auto; padding: 12px 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 18px; margin-bottom: 18px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; }
  button { font: inherit; padding: 9px 14px; border-radius: 9px; cursor: pointer;
           border: 1px solid var(--line); background: #fff; color: var(--ink); }
  button:hover { border-color: var(--accent); }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  label.file { display: inline-block; }
  input[type=file] { display: none; }
  .hint { color: var(--muted); font-size: 13px; margin-top: 10px; }
  svg { width: 100%; height: 170px; display: block; }
  .axis { font-size: 11px; fill: var(--muted); }
  .verdict { font-size: 25px; font-weight: 650; margin: 2px 0 4px; }
  .pill { display: inline-block; font-size: 12px; padding: 2px 9px; border-radius: 999px;
          border: 1px solid var(--line); color: var(--muted); margin-left: 8px;
          vertical-align: middle; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
  .grid > div { border-top: 1px solid var(--line); padding-top: 10px; }
  .k { font-size: 12px; color: var(--muted); text-transform: uppercase;
       letter-spacing: 0.04em; }
  .v { margin-top: 2px; }
  .flags { margin-top: 12px; color: var(--warn); font-size: 14px; }
  .limits { color: var(--muted); font-size: 13px; }
  .limits li { margin-bottom: 6px; }
  code { background: #eef2f0; padding: 1px 5px; border-radius: 5px; font-size: 13px; }
  @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Control Loop Diagnosis</h1>
  <p class="sub">Give it a loop's setpoint, measurement and controller output. It answers
  what is wrong, how sure it is, what to do, and who owns it - or says the data cannot
  support an answer.</p>
</header>

<main>
  <div class="card">
    <div class="row" id="examples"></div>
    <div class="row" style="margin-top:10px">
      <label class="file"><input type="file" id="csv" accept=".csv,text/csv">
        <button type="button" onclick="document.getElementById('csv').click()">
          Upload a CSV</button></label>
      <span class="hint" id="source">worked examples are simulated, so the fault is known</span>
    </div>
    <div class="hint">CSV needs columns <code>sp,pv,op</code> and a header row.
      Sampling interval: <input id="ts" type="number" value="1" min="0.001" step="0.1"
      style="width:70px"> s &nbsp; Loop type:
      <select id="lt"><option value="F">flow</option><option value="P">pressure</option>
      <option value="L">level</option><option value="T">temperature</option>
      <option value="Q">quality</option></select></div>
  </div>

  <div class="card" id="charts" hidden>
    <div class="k">Controller output (OP) <span id="window" class="pill"></span></div>
    <svg id="opChart" viewBox="0 0 900 170" preserveAspectRatio="none"></svg>
    <div class="k" style="margin-top:10px">Measurement (PV) and setpoint (SP)</div>
    <svg id="pvChart" viewBox="0 0 900 170" preserveAspectRatio="none"></svg>
  </div>

  <div class="card" id="result" hidden>
    <div><span class="verdict" id="verdict"></span><span class="pill" id="conf"></span></div>
    <div class="k">Evidence</div><div class="v" id="evidence"></div>
    <div class="grid">
      <div><div class="k">Action</div><div class="v" id="action"></div></div>
      <div><div class="k">Owner</div><div class="v" id="owner"></div></div>
    </div>
    <div class="flags" id="flags"></div>
  </div>

  <div class="card">
    <div class="k">What this cannot do</div>
    <ul class="limits" id="limits"></ul>
    <div class="hint">Measured on 141 plant loops from the SACAC and ISDB archives; the
      full record is in <code>projects/p02_control_loops/VALIDATION.md</code>. API docs at
      <a href="docs">/loops/docs</a>.</div>
  </div>
</main>

<script>
const $ = function (id) { return document.getElementById(id); };
var current = null;

function line(svg, series, colour, lo, hi, dashed) {
  var w = 900, h = 170, pad = 8, span = (hi - lo) || 1;
  var step = w / Math.max(series.length - 1, 1), d = '';
  for (var i = 0; i < series.length; i++) {
    var x = (i * step).toFixed(1);
    var y = (h - pad - (series[i] - lo) / span * (h - 2 * pad)).toFixed(1);
    d += (i ? 'L' : 'M') + x + ',' + y;
  }
  var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', colour);
  path.setAttribute('stroke-width', '1.4');
  path.setAttribute('vector-effect', 'non-scaling-stroke');
  if (dashed) path.setAttribute('stroke-dasharray', '5 4');
  svg.appendChild(path);
}

function thin(a) {
  if (a.length <= 900) return a;
  var step = Math.ceil(a.length / 900), out = [];
  for (var i = 0; i < a.length; i += step) out.push(a[i]);
  return out;
}

function window_of(loop, periodSeconds) {
  // Show about eight cycles: the corners of a sticking valve are the evidence,
  // and a thousand cycles squeezed into 900 pixels hides them. A data-quality
  // fault is the opposite - a frozen sensor is only visible in the whole record.
  var n = loop.pv.length;
  if (periodSeconds === -1) return n;
  var wanted = periodSeconds ? Math.round(8 * periodSeconds / (loop.ts || 1)) : 400;
  return Math.min(n, Math.max(120, wanted));
}

function draw(loop, periodSeconds) {
  var end = window_of(loop, periodSeconds);
  var cut = function (a) { return a.slice(0, end); };
  var op = thin(cut(loop.op)), pv = thin(cut(loop.pv)), sp = thin(cut(loop.sp));
  var total = Math.round(loop.pv.length * (loop.ts || 1));
  var shown = Math.round(end * (loop.ts || 1));
  $('window').textContent = shown >= total ? 'whole record, ' + total + ' s'
                                           : 'first ' + shown + ' s of ' + total + ' s';
  var opChart = $('opChart'), pvChart = $('pvChart');
  while (opChart.firstChild) opChart.removeChild(opChart.firstChild);
  while (pvChart.firstChild) pvChart.removeChild(pvChart.firstChild);
  line(opChart, op, '#2d6ea8', Math.min.apply(null, op), Math.max.apply(null, op));
  var both = pv.concat(sp);
  var lo = Math.min.apply(null, both), hi = Math.max.apply(null, both);
  line(pvChart, sp, '#7b8a82', lo, hi, true);
  line(pvChart, pv, '#1f6f54', lo, hi);
  $('charts').hidden = false;
}

var COLOUR = { stiction: '#9c2b2b', saturation: '#b4531a', frozen_sensor: '#9c2b2b',
               manual: '#b4531a', tuning: '#b4531a', undetermined: '#5d6b64' };
var WORDS = { stiction: 'Sticking valve', tuning: 'Tuning problem',
              saturation: 'Valve at its limit', frozen_sensor: 'Frozen transmitter',
              manual: 'Loop not in automatic', undetermined: 'No confident diagnosis' };

function show(out, ok) {
  if (!ok) {
    $('verdict').textContent = 'Rejected: ' + (out.detail || 'bad request');
    $('verdict').style.color = '#b4531a';
    $('evidence').textContent = ''; $('action').textContent = '';
    $('owner').textContent = ''; $('flags').textContent = '';
    $('result').hidden = false; return;
  }
  var word = WORDS[out.diagnosis] || out.diagnosis;
  if (out.diagnosis === 'undetermined' && !out.quality_flags.length) {
    // Be precise about which kind of 'no answer' this is: a quiet loop is not
    // the same as a record too short to judge (VALIDATION.md 15.1).
    word = out.regularity <= 1 ? 'No regular oscillation' : 'No confident diagnosis';
  }
  $('verdict').textContent = word;
  $('verdict').style.color = COLOUR[out.diagnosis] || '#15211c';
  $('conf').textContent = out.confidence_band + ' confidence';
  $('evidence').textContent = out.evidence + '  ·  measured on ' + out.shape_signal;
  $('action').textContent = out.action;
  $('owner').textContent = out.owner || '-';
  $('flags').textContent = out.quality_flags.length
    ? 'Data-quality flags: ' + out.quality_flags.join(', ') : '';
  $('result').hidden = false;
  if (current) {
    // zoom to the waveform, unless the evidence is a data-quality fault
    draw(current, out.quality_flags.length ? -1 : (out.period_s || 0));
  }
}

function analyse(loop) {
  current = loop;
  draw(loop);
  var body = { loop: { name: loop.name || 'loop', loop_type: loop.loop_type,
                       sp: loop.sp, pv: loop.pv, op: loop.op }, ts: loop.ts };
  var ok = true;
  fetch('analyse/loop', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify(body) })
    .then(function (r) { ok = r.ok; return r.json(); })
    .then(function (out) { show(out, ok); })
    .catch(function (e) { show({ detail: e.message }, false); });
}

function parseCsv(text) {
  var rows = text.trim().split(/\r?\n/).map(function (r) { return r.split(/[,;\t]/); });
  var head = rows[0].map(function (h) { return h.trim().toLowerCase(); });
  if (head.indexOf('pv') < 0 || head.indexOf('op') < 0)
    throw new Error('need columns sp, pv, op');
  function col(c) {
    var i = head.indexOf(c);
    return rows.slice(1).map(function (r) { return parseFloat(r[i]); });
  }
  var pv = col('pv'), op = col('op');
  var mean = pv.reduce(function (a, b) { return a + b; }, 0) / pv.length;
  var sp = head.indexOf('sp') >= 0 ? col('sp') : pv.map(function () { return mean; });
  return { name: 'uploaded', loop_type: $('lt').value,
           ts: parseFloat($('ts').value) || 1, sp: sp, pv: pv, op: op };
}

$('csv').addEventListener('change', function (e) {
  var file = e.target.files[0]; if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      var loop = parseCsv(reader.result);
      $('source').textContent = file.name + ' - ' + loop.pv.length + ' samples';
      var buttons = document.querySelectorAll('#examples button');
      for (var i = 0; i < buttons.length; i++) buttons[i].classList.remove('on');
      analyse(loop);
    } catch (err) { $('source').textContent = 'could not read that file: ' + err.message; }
  };
  reader.readAsText(file);
});

fetch('health').then(function (r) { return r.json(); }).then(function (h) {
  $('limits').innerHTML = h.limits.map(function (l) { return '<li>' + l + '</li>'; }).join('');
});

fetch('examples').then(function (r) { return r.json(); }).then(function (list) {
  $('examples').innerHTML = list.map(function (e) {
    return '<button type="button" data-key="' + e.key + '" title="' + e.blurb + '">' +
           e.title + '</button>';
  }).join('');
  $('examples').addEventListener('click', function (ev) {
    var btn = ev.target.closest ? ev.target.closest('button') : null;
    if (!btn) return;
    var buttons = document.querySelectorAll('#examples button');
    for (var i = 0; i < buttons.length; i++) buttons[i].classList.remove('on');
    btn.classList.add('on');
    fetch('examples/' + btn.getAttribute('data-key'))
      .then(function (r) { return r.json(); })
      .then(function (loop) {
        $('source').textContent = loop.blurb;
        $('ts').value = loop.ts; $('lt').value = loop.loop_type;
        analyse(loop);
      })
      .catch(function (e) { $('source').textContent = 'could not load that example'; });
  });
  $('examples').getElementsByTagName('button')[0].click();
});
</script>
</body>
</html>
"""
