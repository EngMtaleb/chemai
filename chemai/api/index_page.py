"""The service's front door.

Two projects share one deployment, and a visitor who opens the root URL should
land somewhere, not on a 404. This lists what is running and links to each.
"""

INDEX = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chemai - industrial AI services</title>
<style>
  body { margin: 0; background: #f7f9f8; color: #15211c;
         font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 720px; margin: 0 auto; padding: 48px 20px 60px; }
  h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.2px; }
  .sub { color: #5d6b64; margin: 0 0 28px; }
  a.card { display: block; background: #fff; border: 1px solid #dfe5e1; border-radius: 12px;
           padding: 18px 20px; margin-bottom: 14px; text-decoration: none; color: inherit; }
  a.card:hover { border-color: #1f6f54; }
  .t { font-weight: 640; font-size: 17px; }
  .d { color: #5d6b64; margin-top: 4px; }
  .m { color: #1f6f54; margin-top: 8px; font-size: 13px; }
  footer { color: #5d6b64; font-size: 13px; margin-top: 26px; }
  a { color: #1f6f54; }
</style>
</head>
<body>
<div class="wrap">
  <h1>chemai</h1>
  <p class="sub">Industrial AI services for chemical engineering problems. Two projects, one
  deployment. Every answer carries its uncertainty, or the reason it cannot be given.</p>

  <a class="card" href="loops/">
    <div class="t">Control loop diagnosis</div>
    <div class="d">Sticking valve, tuning, saturation, frozen transmitter - from setpoint,
      measurement and controller output. Answers with the evidence, the action and the owner.</div>
    <div class="m">Open the demo &rarr;</div>
  </a>

  <a class="card" href="docs">
    <div class="t">Distillation soft sensor</div>
    <div class="d">Estimates laboratory vapour pressure from two process measurements, with a
      95% interval and an operating-envelope check that refuses extrapolation.</div>
    <div class="m">Open the API documentation &rarr;</div>
  </a>

  <footer>
    Source and validation records:
    <a href="https://github.com/EngMtaleb/chemai">github.com/EngMtaleb/chemai</a><br>
    Eng. Mohammed Taleb - chemical engineer ·
    <a href="https://www.linkedin.com/in/mohammed-taleb-4b4195115/">LinkedIn</a><br>
    Free hosting - the first request after idle takes up to a minute to wake.
  </footer>
</div>
</body>
</html>
"""
