"""Phase 1's results pages (v10, notes 9-10) -> p1_data.js (const P1C). Every number those pages show comes from here.

    python3 make_p1_data.py

Charts (ΔP = dp_pct, 7 days ahead, rolling test, trains A/B/C):
  - XGBoost, hybrid: phase 1's own code (p3lib) and split — fit on the first three quarters of block 1's training pairs,
    the last quarter as validation (the declared way phase 1 sizes its 90 % ranges); test forecasts and ranges as saved.
  - Prophet: addition 2's prophet_chart (addition 1's forecast() unchanged); test forecasts as saved; range computed
    afterwards.
  - LSTM, RNN, hybrid LSTM, hybrid RNN: stage 1's forecasts on its fit rows and validation slice (addition 2's run);
    test forecasts as saved (addition 1) or as declared (addition 2); range computed afterwards from the validation slice.
Verdict: scores_*_addition2.csv (phase 1's six and addition 1's four re-scored identically, plus addition 2's two).
"""
# In this repository the paths point at the project's own folders; the run used the working folders.

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent.parent
P3, AD1, AD2 = S / 'phase1', S / 'phase1' / 'addition', S / 'phase1' / 'addition2'
sys.path.insert(0, str(P3))
import p3lib as L  # noqa: E402

DATA = P3 / 'data'
SHA = {"derived_daily_indicator_table.csv": "50ad81cb632dfc74e710bad70ce9bf08a014061d0ef7b602be25fc465f1f6374",
       "cip_campaign_ledger.csv": "4f65351d8aaa96d8c804c63d3257978dd22628553e7ab4e12d08f1aa0a375ca7"}
for n, s in SHA.items():
    assert hashlib.sha256((DATA / n).read_bytes()).hexdigest() == s, n
d = pd.read_csv(DATA / 'derived_daily_indicator_table.csv').sort_values(['train', 'day_index']).reset_index(drop=True)
led = pd.read_csv(DATA / 'cip_campaign_ledger.csv')
days = L.build_days(d, led); pairs = days[days.valid].reset_index(drop=True); feats = L.features(days)
c, h = 'dp_pct', 7
P = pairs[pairs.h == h]
r2 = lambda a: [round(float(x) * 100, 2) for x in a]          # fraction -> percent, 2 decimals  # noqa: E731
mae = lambda y, p: float(np.mean(np.abs(np.asarray(y) - np.asarray(p))) * 100)  # noqa: E731

# saved rolling test forecasts (phase 1 + addition 1 + addition 2), dp_pct, 7 days
R = pd.read_csv(P3 / 'out' / 'pred_rolling.csv'); R = R[(R.target == c) & (R.h == h)]
PR = pd.read_csv(AD1 / 'pred_prophet.csv'); PR = PR[(PR.target == c) & (PR.h == h)][['train', 't', 'Prophet']]
NN = pd.read_csv(AD1 / 'pred_nets_rolling.csv'); NN = NN[(NN.target == c) & (NN.h == h)][['train', 't', 'LSTM', 'RNN']]
HN = pd.read_csv(AD2 / 'pred_hnets_rolling.csv'); HN = HN[(HN.target == c) & (HN.h == h)][['train', 't', 'hybrid LSTM', 'hybrid RNN']]
T = R.merge(PR, on=['train', 't'], validate='1:1').merge(NN, on=['train', 't'], validate='1:1').merge(HN, on=['train', 't'], validate='1:1')
assert len(T) == len(R)

# the four network re-runs reproduce the saved LSTM / RNN test forecasts
for b in range(4):
    re = pd.read_csv(AD2 / 'parts' / f'chart_rolling_h07_b{b}.csv')
    m = re.merge(NN, on=['train', 't'])
    for k in ('LSTM', 'RNN'):
        dmax = float(np.max(np.abs(m[f'{k}|dp_pct'] - m[k])))
        assert dmax < 1e-12, (b, k, dmax)

# ---------------------------------------------------------------- per-method curves (block 1's training), ranges by block
# phase 1's corrected range split (final notebook, "Corrections after the run"): the last quarter starts at a whole day,
# and the method is refitted on the pairs whose target day is before it. phase3/out/ holds the first run's ranges.
def split(train):
    cal_start = train.t.iloc[int(round(0.75 * len(train)))]
    return train[train.t + h < cal_start], train[train.t >= cal_start]


QB = {m: [] for m in L.METHODS}
for b, (start, _) in enumerate(L.BLOCKS):
    tr_b = P[P.t + h < start].sort_values(['t', 'train']); fit_b, cal_b = split(tr_b)
    (f_c,) = L.forecasts(fit_b, [cal_b], c, h, feats)
    for m in L.METHODS:
        QB[m].append(float(np.quantile(np.abs(cal_b[f'y_{c}'].to_numpy() - f_c[m]), 0.9)))
NB = {'no change': 86.2, 'smoothed no change': 86.7, 'trend': 87.8, 'fouling law': 92.3, 'XGBoost': 93.7, 'hybrid': 92.7}  # final notebook [25], 7 d
for m in L.METHODS:                                      # the recomputed ranges reproduce the final notebook's coverage
    qte = np.array([QB[m][b] for b in R.block])
    cov = 100 * np.mean(np.abs(R.y.to_numpy() - R[m].to_numpy()) <= qte)
    assert abs(cov - NB[m]) < 0.05, (m, cov, NB[m])
train0 = P[P.t + h < L.BLOCKS[0][0]].sort_values(['t', 'train'])
fit0, cal0 = split(train0)
f_fit, f_cal = L.forecasts(fit0, [fit0, cal0], c, h, feats)
PQ = json.loads((AD2 / 'prophet_chart_q90.json').read_text())
PC = pd.read_csv(AD2 / 'prophet_chart_curves.csv')


def curves_nets(method):
    tag = 'chart' if method in ('LSTM', 'RNN') else 'declared'
    cur = pd.concat([pd.read_csv(AD2 / 'parts' / f'{tag}_rolling_h07_b{b}_curves.csv') for b in range(4)])
    cur = cur[(cur.method == method) & (cur.target == c)]
    q = [float(cur[cur.block == b].q90.iloc[0]) for b in range(4)]
    c0 = cur[cur.block == 0]
    return c0[c0.part == 'fit'], c0[c0.part == 'val'], q


METHODS = {
    'XGBoost': {'family': 'ML', 'learns': True, 'range': 'declared'},
    'hybrid': {'family': 'ML', 'learns': True, 'range': 'declared'},
    'Prophet': {'family': 'ML', 'learns': False, 'range': 'afterwards'},
    'LSTM': {'family': 'DL', 'learns': True, 'range': 'afterwards'},
    'RNN': {'family': 'DL', 'learns': True, 'range': 'afterwards'},
    'hybrid LSTM': {'family': 'DL', 'learns': True, 'range': 'afterwards'},
    'hybrid RNN': {'family': 'DL', 'learns': True, 'range': 'afterwards'},
}
out = {'h': h, 'blocks': L.BLOCKS, 'n_test_pairs': int(len(T)), 'trains': {}, 'methods': {}}
for tr in 'ABC':
    g = d[d.train == tr]
    out['trains'][tr] = {'real': {'d': g.day_index.astype(int).tolist(), 'v': r2(g[c])}, 'cip': led[led.train == tr].day_index.astype(int).tolist(),
                         'n_test': int((T.train == tr).sum())}
for m, meta in METHODS.items():
    if m in ('XGBoost', 'hybrid'):
        fit_rows = fit0.assign(pred=f_fit[m]); val_rows = cal0.assign(pred=f_cal[m])
        fit_rows = fit_rows.assign(y=fit_rows[f'y_{c}'], now=fit_rows[c]); val_rows = val_rows.assign(y=val_rows[f'y_{c}'], now=val_rows[c])
        q = QB[m]
    elif m == 'Prophet':
        fit_rows = PC[PC.part == 'fit']; val_rows = PC[PC.part == 'val']
        q = [float(PQ['q90_by_block'][str(b)]) for b in range(4)]
    else:
        fr, vr, q = curves_nets(m)
        nowmap = days[days.h == h].set_index(['train', 't'])[c]
        fit_rows = fr.assign(now=[float(nowmap[(a, b)]) for a, b in zip(fr.train, fr.t)])
        val_rows = vr.assign(now=[float(nowmap[(a, b)]) for a, b in zip(vr.train, vr.t)])
    M = {**meta, 'q_by_block': [round(x * 100, 2) for x in q], 'per': {}}
    for tr in 'ABC':
        fr_, vr_, te = fit_rows[fit_rows.train == tr].sort_values('t'), val_rows[val_rows.train == tr].sort_values('t'), T[T.train == tr].sort_values('t')
        qte = np.array([q[b] for b in te.block])
        M['per'][tr] = {
            'fit': {'d': (fr_.t + h).astype(int).tolist(), 'p': r2(fr_.pred)},
            'val': {'d': (vr_.t + h).astype(int).tolist(), 'p': r2(vr_.pred)},
            'test': {'d': (te.t + h).astype(int).tolist(), 'p': r2(te[m]), 'q': [round(x * 100, 2) for x in qte], 'b': te.block.astype(int).tolist()},
            'miss': {'fit': mae(fr_.y, fr_.pred), 'val': mae(vr_.y, vr_.pred), 'test': mae(te.y, te[m])},
            'miss_nc': {'fit': mae(fr_.y, fr_.now), 'val': mae(vr_.y, vr_.now), 'test': mae(te.y, te['no change'])},
            'cover': float(np.mean(np.abs(te.y.to_numpy() - te[m].to_numpy()) <= qte) * 100),
            'valid_from': int((vr_.t + h).min()) if len(vr_) else None,
        }
    out['methods'][m] = M
# the rival on the test days, per train
out['nochange'] = {tr: {'d': (T[T.train == tr].sort_values('t').t + h).astype(int).tolist(), 'p': r2(T[T.train == tr].sort_values('t')['no change'])} for tr in 'ABC'}

# ---------------------------------------------------------------- the verdict, both tests, every horizon and indicator
V = {}
for test in ('rolling', 'unseen_train'):
    sc = pd.read_csv(AD2 / f'scores_{test}_addition2.csv')
    sc = sc[sc.method != 'no change']
    V[test] = [{'m': r['method'], 'target': r['target'], 'h': int(r['h']), 'skill': round(float(r['skill_pct']), 1), 'pos': int(r['blocks positive']),
                'pass': bool(r['beats no change'] == True)} for r in sc.to_dict('records')]  # noqa: E712
out['verdict'] = V
out['passes'] = {t: [x for x in V[t] if x['pass']] for t in V}
# ---------------------------------------------------------------- the zero readings (flux loss): the share, and the test without them
# the smoothed state takes the last three readings of the cycle within seven days (p3lib.build_days); a pair is left out when one is 0
zero_in = {}
for tr, g in d.groupby('train'):
    g = g.set_index('day_index').sort_index(); cips = np.sort(led[led.train == tr].day_index.values)
    for t in g.index:
        last = cips[cips <= t].max() if (cips <= t).any() else -1
        cyc = g[(g.index > last) & (g.index <= t)]; r3 = cyc[cyc.index > t - 7].tail(3)
        zero_in[(tr, int(t))] = bool(((r3.flux_pct.values if len(r3) else np.array([g.at[t, 'flux_pct']])) == 0).any())
RF = pd.read_csv(P3 / 'out' / 'pred_rolling.csv'); RF = RF[(RF.target == 'flux_pct') & (RF.h == 14)]
RF = RF.assign(zero_in=[zero_in[(a, int(b))] for a, b in zip(RF.train, RF.t)])
skl = lambda g: float(100 * (1 - np.mean(np.abs(g.y - g['smoothed no change'])) / np.mean(np.abs(g.y - g['no change']))))  # noqa: E731
KZ = RF[~RF.zero_in]
out['flux_zero'] = {'share_all': round(100 * float((d.flux_pct == 0).mean()), 2), 'n_rows': int(len(d)),
                    'all': {'n': int(len(RF)), 'skill': round(skl(RF), 2), 'blocks': [round(skl(g), 1) for _, g in RF.groupby('block')]},
                    'no_zero': {'n': int(len(KZ)), 'skill': round(skl(KZ), 2), 'blocks': [round(skl(g), 1) for _, g in KZ.groupby('block')]}}
assert round(out['flux_zero']['share_all']) == 67 and round(out['flux_zero']['no_zero']['skill'], 1) == 12.1, out['flux_zero']
import re  # noqa: E402
txt = (P3 / 'nbrun4' / 'outputs.txt').read_text()        # the final notebook's printed outputs (the corrected ranges)
out['ranges'] = {tg: dict(zip(('held', 'of'), map(int, re.search(rf'{tg}: holds over all blocks in (\d+) of (\d+)', txt).groups()))) for tg in ('dp_pct', 'flux_pct')}
js = '/* generated by make_p1_data.py (v10) — phase 1 results; do not edit by hand */\nconst P1C = ' + json.dumps(out, separators=(',', ':')) + ';\n'
Path(__file__).with_name('p1_data.js').write_text(js)
print('p1_data.js', round(len(js) / 1024), 'KB · test pairs', len(T), '· passes', {t: [(x['m'], x['target'], x['h']) for x in out['passes'][t]] for t in V})
for m in METHODS:
    print(m, {tr: tuple(round(out['methods'][m]['per'][tr]['miss'][p], 1) for p in ('fit', 'val', 'test')) + (round(out['methods'][m]['per'][tr]['cover']),) for tr in 'ABC'})
print('ranges held (declared, phase 1):', out['ranges'])
print('flux zeros:', out['flux_zero'])
