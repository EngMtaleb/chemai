"""Descriptive, after the runs: for every method of experiments 11 and 12 (condition B), the score against the rule
(caught - false alarms - 10) and a paired 95 % range for it (2000 resamples of the 58 loops, the same resample for the
method and the rule). Nothing here changes a result; it says how far one score difference can be trusted with 58 loops."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).parent
E11F = json.loads((H.parent / 'exp11' / 'results_exp11_features.json').read_text())
E11D = json.loads((H.parent / 'exp11' / 'results_exp11_dl.json').read_text())
F = json.loads((H / 'results_exp12_features.json').read_text())
N = json.loads((H / 'results_exp12_dl.json').read_text())
P = json.loads((H / 'results_exp12_posthoc.json').read_text()) if (H / 'results_exp12_posthoc.json').exists() else None
names = F['calls']['names']
assert E11F['calls']['names'] == names and E11D['names'] == names and N['names'] == names
lab = pd.read_csv(H.parent / 'exp10' / 'real_meta_exp10.csv').set_index('name').label.reindex(names).values
truth = lab == 'stiction'
rule = np.array(F['calls']['the rule']['B']) == 'stiction'
calls = {}
for k in ['ML · random forest (default)', 'tuning · random forest', 'ensemble · soft voting of 5 tuned', 'physics · random forest', 'physics · hybrid', 'physics · rule as teacher', 'ANN · physics features']:
    calls[f'exp11 · {k}'] = np.array(E11F['calls'][k]['B']) == 'stiction'
for k in ['cnn', 'lstm']:
    calls[f'exp11 · {k}'] = np.array(E11D['networks'][k]['calls_B']) == 'stiction'
SAME = {'random_forest · default', 'random_forest · tuned', 'check · ensemble of the 5 (experiment 11)', 'ANN'}   # experiment 11 again (reproduction)
for k in F['methods']:
    if k != 'the rule' and k not in SAME:
        calls[f'exp12 · {k}'] = np.array(F['calls'][k]['B']) == 'stiction'
for k, v in N['methods'].items():
    calls[f'exp12 · {k}'] = np.array(v['calls_B'], bool)
if P:
    for k, v in P['methods'].items():
        calls[f'post hoc · {k}'] = np.array(v['calls_B'], bool)
score = lambda c, i: int((c[i] & truth[i]).sum()) - int((c[i] & ~truth[i]).sum())          # noqa: E731
rng = np.random.default_rng(0)
idx = rng.integers(0, len(truth), size=(2000, len(truth)))
allr = np.arange(len(truth))
out = {}
for k, c in calls.items():
    d0 = score(c, allr) - score(rule, allr)
    ds = np.array([score(c, i) - score(rule, i) for i in idx])
    out[k] = {'caught': int((c & truth).sum()), 'false_alarms': int((c & ~truth).sum()), 'vs_rule': d0,
              'range95': [int(np.percentile(ds, 2.5)), int(np.percentile(ds, 97.5))], 'share_above_0': round(float((ds > 0).mean()), 3)}
(H / 'paired_exp12.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
tries = [k for k in out if not k.startswith('post hoc')]
print(f'{len(tries)} methods tried against the rule (experiments 11 and 12), {sum(out[k]["vs_rule"] > 2 for k in tries)} clear the bar (> +2)')
for k, v in sorted(out.items(), key=lambda kv: -kv[1]['vs_rule']):
    print(f"{k:58s} {v['caught']:2d}/{v['false_alarms']:2d}  vs rule {v['vs_rule']:+3d}  95% {v['range95']}  P(>0) {v['share_above_0']:.2f}")
