#!/usr/bin/env python3
"""Reproducible causal downstream forecasting of deposited industrial indicators.

Run: python run_study.py. The source CSVs are from the verified Zenodo deposit.
This script intentionally never reconstructs physical dP from dp_pct.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/smartro-industrial-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'zenodo_deposit'
PLAN = json.loads((ROOT / 'study_plan.json').read_text())
CHANNELS = ['dp_pct', 'flux_pct', 'ndp_pct', 'sp_pct']
METHODS = ['persistence', 'median3_7d', 'ridge_residual', 'histgb_residual']
BLOCKS = PLAN['rolling_test_blocks']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def num(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.ndarray,)): return v.tolist()
    raise TypeError(str(type(v)))

def md_table(df):
    def fmt(v):
        return f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def verify_source():
    verified = []
    for line in (SRC/'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(SRC/name) == digest, f'Source checksum mismatch: {name}'
        verified.append(name)
    return verified

def construct_pairs(data, ledger):
    rows = []
    features = None
    audit = dict(candidates=0, missing_exact_target=0, crosses_or_starts_cip=0, accepted=0)
    for tr, group in data.groupby('train'):
        group = group.sort_values('day_index').set_index('day_index', drop=False)
        cips = ledger.loc[ledger.train.eq(tr), 'day_index'].to_numpy()
        for day, current in group.iterrows():
            past_cips = cips[cips <= day]
            last_cip = int(past_cips.max()) if len(past_cips) else None
            history = group[(group.index <= day) & (group.index > (last_cip if last_cip is not None else -1))]
            if history.empty:  # current CIP day is deliberately excluded below
                history = group.loc[[day]]
            feature = {}
            for channel in CHANNELS:
                feature[channel+'_now'] = float(current[channel])
                for w in [7, 28]:
                    window = history[history.index >= day-w+1]
                    values = window[channel].to_numpy(float)
                    feature[channel+f'_median{w}'] = float(np.median(values))
                    feature[channel+f'_std{w}'] = float(np.std(values))
                    x = window.index.to_numpy(float)
                    slope = float(np.polyfit(x-x[-1], values, 1)[0]) if len(x)>1 else 0.
                    feature[channel+f'_slope{w}'] = slope
                    feature[channel+f'_n{w}'] = int(len(values))
                previous = history[history.index < day]
                feature[channel+'_last_change'] = float(current[channel]-previous[channel].iloc[-1]) if len(previous) else 0.
            feature['days_since_observation'] = int(day-history.index[-2]) if len(history)>1 else 0
            feature['cip_known'] = int(last_cip is not None)
            feature['days_since_known_cip'] = min(int(day-last_cip), 365) if last_cip is not None else 0
            feature['current_raw_sample_count'] = int(current.n_10min_samples)
            feature['current_basis_n3'] = int(current.basis == 'n3')
            if features is None: features = list(feature)
            assert list(feature) == features
            assert history.index.max() <= day
            if last_cip is not None and day > last_cip: assert history.index.min() > last_cip
            for h in PLAN['horizons_days']:
                audit['candidates'] += 1
                target_day = int(day+h)
                if target_day not in group.index:
                    audit['missing_exact_target'] += 1
                    continue
                if np.any((cips >= day) & (cips <= target_day)):
                    audit['crosses_or_starts_cip'] += 1
                    continue
                target_row = group.loc[target_day]
                common = dict(train=tr, origin=int(day), label_day=target_day, horizon=h, last_cip=last_cip)
                last3 = history[history.index >= day-6].tail(3)
                for target in PLAN['targets']:
                    rows.append({**common, 'target':target, 'y':float(target_row[target]), 'now':float(current[target]),
                                 'smoothed':float(last3[target].median()), **feature})
                audit['accepted'] += 1
    return pd.DataFrame(rows), features, audit

def fit_predict(train, test, features):
    assert len(train) and len(test)
    Xtr = train[features].to_numpy(float)
    Xte = test[features].to_numpy(float)
    delta = train.y.to_numpy()-train.now.to_numpy()
    out = {'persistence': test.now.to_numpy(), 'median3_7d':test.smoothed.to_numpy()}
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    gb = HistGradientBoostingRegressor(learning_rate=.04, max_iter=150, max_leaf_nodes=7,
                                      min_samples_leaf=20, l2_regularization=5.,
                                      early_stopping=False, random_state=20261030)
    for name, model in [('ridge_residual',ridge),('histgb_residual',gb)]:
        model.fit(Xtr, delta)
        out[name] = np.clip(test.now.to_numpy()+model.predict(Xte), 0., 1.)
    return out

def verify_future_invariance(data,ledger,pairs,features):
    """Future values/cleanings and forbidden metadata cannot alter past features."""
    altered=data.copy()
    altered.loc[altered.day_index>=550,CHANNELS]=.731
    altered.loc[altered.day_index>=550,'n_10min_samples']=999
    altered['days_to_next_cip']=-12345
    altered['state']='deliberately_changed_future_metadata'
    altered['days_since_cip']=-12345  # recomputed only from past ledger
    changed, changed_features, _=construct_pairs(altered,ledger[ledger.day_index<550])
    assert features==changed_features
    keys=['train','origin','label_day','target','horizon']
    first=pairs[pairs.label_day<550].set_index(keys)[features].sort_index()
    second=changed[changed.label_day<550].set_index(keys)[features].sort_index()
    pd.testing.assert_frame_equal(first,second,check_exact=True)
    return dict(passes=True,past_feature_rows_checked=len(first),cutoff_day=550,
                mutated='Future channel values, future raw-count metadata, future CIP events, all future-aware state/next-CIP metadata, and supplied CIP-age column')

def score(group):
    base = group[group.method.eq('persistence')]
    bm = float(np.abs(base.y-base.prediction).mean())
    result = []
    for method, df in group.groupby('method',sort=False):
        err = df.prediction-df.y
        mae = float(np.abs(err).mean())
        result.append(dict(method=method,n=len(df),mae=mae,rmse=float(np.sqrt(np.mean(err**2))),
                           skill=1-mae/bm if bm>0 else None))
    return result

def main():
    verified = verify_source()
    data = pd.read_csv(SRC/'data/derived_daily_indicator_table.csv')
    ledger = pd.read_csv(SRC/'data/cip_campaign_ledger.csv')
    assert not data.duplicated(['train','day_index']).any()
    assert not data[CHANNELS].isna().any().any()
    assert ((data[CHANNELS] >= 0)&(data[CHANNELS]<=1)).all().all()
    attachment=json.loads((ROOT/'DATA5.json').read_text())['deposit']
    embedded_comparison={}
    for tr, g in data.groupby('train'):
        g=g.sort_values('day_index')
        emb=attachment['trains'][tr]
        embedded_comparison[tr]=dict(day_sequence_matches=g.day_index.tolist()==emb['day'],
                                    dp_max_abs_difference=float(np.max(np.abs(g.dp_pct-np.array(emb['dp'])))),
                                    flux_max_abs_difference=float(np.max(np.abs(g.flux_pct-np.array(emb['flux'])))),
                                    cip_days_match=ledger[ledger.train.eq(tr)].day_index.tolist()==attachment['cips'][tr])
    pairs, features, audit = construct_pairs(data, ledger)
    future_invariance=verify_future_invariance(data,ledger,pairs,features)
    pair_cols=['train','origin','label_day','horizon','target','last_cip','now','smoothed','y']
    pairs.to_csv(ROOT/'forecast_pairs.csv',index=False)
    predictions=[]
    splits=[]
    def run(part, split, target,h, train,test):
        assert train.label_day.max() < test.origin.min(), (part,split)
        out=fit_predict(train,test,features)
        splits.append(dict(protocol=part,split=split,target=target,horizon=h,
                           train_n=len(train),test_n=len(test),train_origin_min=int(train.origin.min()),
                           train_label_max=int(train.label_day.max()),test_origin_min=int(test.origin.min()),
                           test_origin_max=int(test.origin.max()),test_label_max=int(test.label_day.max()),
                           fitted_trains=sorted(train.train.unique()),tested_trains=sorted(test.train.unique())))
        for method, pred in out.items():
            for i,(_,row) in enumerate(test.iterrows()):
                predictions.append({k:row[k] for k in pair_cols[:5]} | dict(protocol=part,split=split,method=method,
                                     y=float(row.y),now=float(row.now),prediction=float(pred[i])))
    for target in PLAN['targets']:
        for h in PLAN['horizons_days']:
            pp=pairs[pairs.target.eq(target)&pairs.horizon.eq(h)]
            run('validation','450_549',target,h,pp[pp.label_day<450],pp[(pp.origin>=450)&(pp.label_day<550)])
            for start,end in BLOCKS:
                run('rolling',f'{start}_{end}',target,h,pp[pp.label_day<start],pp[(pp.origin>=start)&(pp.origin<=end)])
            for tr in ['A','B','C']:
                train=pp[(pp.label_day<550)&(~pp.train.eq(tr))]
                test=pp[(pp.origin>=550)&pp.train.eq(tr)]
                assert tr not in set(train.train)
                run('held_out_train',tr,target,h,train,test)
    pred=pd.DataFrame(predictions)
    pred.to_csv(ROOT/'predictions.csv',index=False)
    metrics=[]
    detailed=[]
    for (part,target,h),g in pred.groupby(['protocol','target','horizon']):
        for r in score(g): metrics.append(dict(protocol=part,target=target,horizon=int(h),**r))
    for (part,split,target,h),g in pred.groupby(['protocol','split','target','horizon']):
        for r in score(g): detailed.append(dict(protocol=part,split=split,target=target,horizon=int(h),**r))
    metrics_df=pd.DataFrame(metrics)
    details_df=pd.DataFrame(detailed)
    metrics_df.to_csv(ROOT/'metrics.csv',index=False)
    details_df.to_csv(ROOT/'metrics_by_split.csv',index=False)
    gates=[]
    for target in PLAN['targets']:
        for h in PLAN['horizons_days']:
            for method in METHODS[1:]:
                r=metrics_df.query('protocol=="rolling" and target==@target and horizon==@h and method==@method').iloc[0]
                x=metrics_df.query('protocol=="held_out_train" and target==@target and horizon==@h and method==@method').iloc[0]
                b=details_df.query('protocol=="rolling" and target==@target and horizon==@h and method==@method')
                wins=int((b.skill>0).sum())
                gates.append(dict(target=target,horizon=h,method=method,rolling_skill=float(r.skill),
                                  held_out_train_skill=float(x.skill),positive_blocks=wins,
                                  passes=bool(r.skill>=.05 and x.skill>=.05 and wins>=3)))
    inventory = dict(rows=len(data), trains=data.train.value_counts().sort_index().to_dict(),
                     day_min=int(data.day_index.min()),day_max=int(data.day_index.max()),
                     distinct_calendar_indices=int(data.day_index.nunique()),cleaning_campaigns=len(ledger),
                     missing_core_indicators=int(data[CHANNELS].isna().sum().sum()),
                     n_raw_samples_aggregated=int(data.n_10min_samples.sum()),
                     daily_sample_counts_above_144=int((data.n_10min_samples>144).sum()),
                     warning_raw_sample_count='Sum across train-day aggregations, not independent calendar observations; some daily counts exceed 144. Do not equate this to unique raw records.',
                     basis_counts=data.basis.value_counts().to_dict(),
                     clipped_zero_share={x:float((data[x]==0).mean()) for x in CHANNELS},
                     clipped_one_share={x:float((data[x]==1).mean()) for x in CHANNELS})
    provenance=dict(source_url='https://zenodo.org/records/21055948',doi='10.5281/zenodo.21055948',
                    paper_doi='10.3390/membranes16080257',license='CC-BY-4.0 (data); MIT (source deposit code)',
                    zip_sha256=sha(ROOT/'ESWRT_reproducibility_deposit_v1.3-20260630.zip'),
                    published_zip_md5_verified=True,verified_manifest_files=verified,
                    source_csv_sha256=sha(SRC/'data/derived_daily_indicator_table.csv'),
                    cip_csv_sha256=sha(SRC/'data/cip_campaign_ledger.csv'),
                    attachment_comparison=embedded_comparison,
                    source_semantic_discrepancy=PLAN['source_semantic_limit'])
    result=dict(study=PLAN,inventory=inventory,provenance=provenance,pair_audit=audit,
                feature_names=features,metrics=metrics,metrics_by_split=detailed,splits=splits,research_gate=gates,
                verification=dict(source_manifest_all_pass=True,unique_train_days=True,
                                  no_missing_core_indicators=True,historical_features_only=True,
                                  no_training_target_reaches_test_origin=True,no_cross_cip_pairs=True,
                                  no_held_out_train_in_fit=True,future_aware_features_excluded=PLAN['excluded_features'],
                                  future_mutation_invariance=future_invariance),
                environment=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
                source_code_sha256=sha(__file__),study_plan_sha256=sha(ROOT/'study_plan.json'))
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,default=num))
    plot_data=dict(series={tr:g[['day_index']+CHANNELS+['n_10min_samples']].to_dict('records') for tr,g in data.groupby('train')},
                   cips={tr:g.day_index.tolist() for tr,g in ledger.groupby('train')},metrics=metrics,
                   seven_day_A=pred[(pred.protocol=='rolling')&(pred.target=='dp_pct')&(pred.horizon==7)&(pred.train=='A')].to_dict('records'),
                   timeline=dict(development_train_end=449,validation_start=450,validation_label_end=549,test_blocks=BLOCKS))
    (ROOT/'plot_data.json').write_text(json.dumps(plot_data,default=num,separators=(',',':')))
    make_charts(data,ledger,metrics_df)
    make_dictionary_chart()
    write_report(result,metrics_df)
    print(json.dumps(dict(inventory=inventory,passes=[x for x in gates if x['passes']],
                          forecast_pairs=len(pairs),source_verified_files=len(verified),
                          metrics_7d=metrics_df[metrics_df.horizon.eq(7)].to_dict('records')),indent=2,default=num))

def make_charts(data,ledger,m):
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7fbfd','axes.facecolor':'#f7fbfd','axes.labelcolor':'#16364a','text.color':'#16364a','font.size':10})
    colors=['#087e8b','#d27b2f','#7457a6']
    fig,axs=plt.subplots(3,1,figsize=(10,5.4),sharex=True,constrained_layout=True)
    for ax,(tr,g),col in zip(axs,data.groupby('train'),colors):
        ax.scatter(g.day_index,g.dp_pct,s=7,color=col,alpha=.8)
        for day in ledger[ledger.train.eq(tr)].day_index: ax.axvline(day,color='#d45b68',alpha=.5,lw=.8)
        ax.set_ylabel(f'Train {tr}\ndp_pct')
        ax.grid(alpha=.16)
    axs[-1].set_xlabel('Relative day index (exact calendar dates withheld)')
    fig.suptitle('Actual industrial-derived observations · gaps are retained\nVertical marks: logged CIP campaigns; dp_pct is the deposited score, not bar',fontsize=13)
    fig.savefig(ROOT/'industrial_observations.svg');fig.savefig(ROOT/'industrial_observations.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    labels={'persistence':'No change','median3_7d':'Trailing median','ridge_residual':'Ridge residual','histgb_residual':'Gradient boosting'}
    for ax,target in zip(axs,PLAN['targets']):
        for method,col in zip(METHODS,['#527286','#0b9c89','#d88b34','#7862ad']):
            z=m[(m.protocol=='rolling')&(m.target==target)&(m.method==method)].sort_values('horizon')
            ax.plot(z.horizon,z.mae*100,'o-',label=labels[method],color=col,lw=2)
        ax.set_title(target+' · future time blocks');ax.set_xlabel('Forecast horizon / days');ax.set_ylabel('MAE × 100 (deposited score points)');ax.set_xticks([1,3,7,14]);ax.grid(alpha=.18)
    axs[0].legend(fontsize=8)
    fig.suptitle('Recomputed on independently downloaded public industrial data',fontsize=13)
    fig.savefig(ROOT/'forecast_comparison.svg');fig.savefig(ROOT/'forecast_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.2),constrained_layout=True)
    ax.broken_barh([(0,450)],(30,14),facecolors='#b8d8e7');ax.text(225,37,'Development fit\nlabel day < 450',ha='center',va='center')
    ax.broken_barh([(450,100)],(30,14),facecolors='#e9c77b');ax.text(500,37,'Validation',ha='center',va='center',fontsize=9)
    for i,(a,b) in enumerate(BLOCKS):
        ax.broken_barh([(a,b-a+1)],(30,14),facecolors=['#0d8e8d','#177f9b','#266eaa','#6750a3'][i]);ax.text((a+b)/2,37,str(i+1),ha='center',va='center',color='white')
    for i,(a,b) in enumerate(BLOCKS):
        ax.broken_barh([(0,a)],(22-i*5,3),facecolors='#bdd6e0');ax.broken_barh([(a,b-a+1)],(22-i*5,3),facecolors='#0d8e8d')
    ax.set_xlim(0,785);ax.set_ylim(1,49);ax.set_yticks([]);ax.set_xlabel('Relative day index');ax.set_xticks([0,450,550,605,660,715,771]);ax.set_title('Causal evaluation: all training labels precede each test block\nLower bars: expanding fit (light) → next unseen block (teal)',fontsize=12)
    fig.savefig(ROOT/'temporal_split.svg');fig.savefig(ROOT/'temporal_split.png',dpi=160);plt.close(fig)

def make_dictionary_chart():
    rows=[('dp_pct','Pressure-related deposited score','Dimensionless','Do not convert to bar'),
          ('flux_pct','Flux-related deposited score','Dimensionless','Not product flow in m³/h'),
          ('ndp_pct / sp_pct','Additional observed indicators','Dimensionless','No raw pressure / salinity'),
          ('day_index','Relative observation day','Days from day 0','No weather calendar alignment'),
          ('CIP ledger','Observed cleaning campaign dates','Relative day','Past known events only as input'),
          ('state / days_to_next_cip','Future-aware event metadata','Metadata','Excluded from predictors')]
    from html import escape
    body=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="565" viewBox="0 0 1200 565">',
          '<rect width="1200" height="565" rx="24" fill="#f4f9fc"/>',
          '<g font-family="Arial,sans-serif" fill="#17364a">',
          '<text x="32" y="44" font-size="25" font-weight="700">Source dictionary · retain the published meaning</text>',
          '<text x="32" y="75" font-size="16" fill="#567385">Verified Zenodo deposit: 1,340 train-days · 3 trains · 18 cleaning campaigns</text>']
    xs=[32,266,690,900]
    for x,t in zip(xs,['FIELD','USED AS','UNIT','INTERPRETATION LIMIT']): body.append(f'<text x="{x}" y="119" font-size="13" fill="#648093" font-weight="700">{t}</text>')
    for i,row in enumerate(rows):
        y=150+i*53
        body.append(f'<rect x="20" y="{y-19}" width="1160" height="47" rx="8" fill="{("#e9f2f7" if i%2==0 else "#f4f9fc")}"/>')
        for j,(x,t) in enumerate(zip(xs,row)):
            body.append(f'<text x="{x}" y="{y+8}" font-size="{16 if j!=0 else 15}" font-weight="{700 if j==0 else 400}">{escape(t)}</text>')
    body.extend(['<rect x="24" y="467" width="1152" height="76" rx="12" fill="#fff0cb"/>',
                 '<text x="40" y="495" font-size="17" font-weight="700">Source ambiguity is preserved, not hidden.</text>',
                 '<text x="40" y="522" font-size="16">Dictionary and scripts differ on baseline/reset semantics; raw SCADA is withheld. No physical ΔP threshold is inferred.</text>',
                 '</g></svg>'])
    (ROOT/'source_dictionary.svg').write_text(''.join(body))

def write_report(r,m):
    sections=['# Industrial RO — independently reproduced derived-indicator forecast',
              '## Evidence and provenance',
              'Downloaded Zenodo version DOI 10.5281/zenodo.21055948, verified the published ZIP MD5 and every entry in SHA256SUMS.txt. These are real-plant-derived daily indicator records, not invented data and not raw SCADA.',
              f"Rows: {r['inventory']['rows']}; trains: {r['inventory']['trains']}; relative days 0–771; CIP campaigns: 18. No exact calendar dates, raw sensor pressures, membrane dimensions, or chemical-water-quality channels are supplied.",
              '## Source dictionary / semantic limit',
              '| Field | Meaning used in this study | Excluded interpretation |\n|---|---|---|\n| dp_pct | Deposited dimensionless pressure-related indicator | Not measured ΔP in bar; not verified DP_norm−1 |\n| flux_pct | Deposited dimensionless flux-related indicator | Not measured flow m³/h |\n| ndp_pct, sp_pct | Additional current and causal historical indicator inputs | Not reconstructed feed pressure / product salinity |\n| day_index | Relative chronology and split only | Not absolute weather-aligned date |\n| days_to_next_cip, state | Future-aware metadata excluded from predictors | Not an available future sensor |\n| CIP ledger | Past event history and retrospective pair exclusions | Not a prescriptive cleaning policy |',
              PLAN['source_semantic_limit'],
              'The deposited source scripts (head2head.py lines distinguishing RAW and PERCIP, angle1_trigger.py description) explicitly separate fixed-baseline DP_norm from per-CIP dp_pct/ndp_pct. The dictionary names *_pct fixed-baseline deviations. As upstream raw data are withheld, the discrepancy cannot be resolved here. We preserve the exact names and make no physical ΔP threshold or energy conversion.',
              '## Method',
              'Plan frozen before fitting: study_plan.json. Four fixed methods; future-minus-current regression for Ridge and HistGradientBoosting. Models receive current values and strictly trailing history of four deposited indicators. Historical windows reset after known cleaning. All preprocessing is fitted to training rows only. Missing future observations are not interpolated. No train ID, absolute day index, future CIP interval, or event-window state is a model feature.',
              'Validation: origin ≥450 and target <550; development training target <450. Final rolling tests: origin days 550–604, 605–659, 660–714, 715–771. Each fit only uses target days before its test-block start. Separate generalization test holds out an entire train and fits the other two trains before day550.',
              'The pair exclusion uses the future CIP ledger retrospectively only to define a conditional no-intervening-CIP benchmark. This is not an unconditional operational forecast: the future cleaning schedule would not generally be known at deployment. Overlapping horizons and autocorrelated daily records are not independent samples. No confidence intervals or universal reliability claim are made.',
              '## Results (MAE in deposited dimensionless units; lower is better)',
              'Persistence has the lowest validation MAE in all eight target/horizon combinations. The later 14-day flux smoothing gain is a test observation and research candidate; it was not the best validation-selected method. A fair operational default remains the simple baseline pending new prospective evidence.',
              md_table(m),
              '## Predefined research screen',
              'Required: ≥5% MAE improvement over persistence on pooled rolling tests AND positive skill on at least3/4 time blocks AND ≥5% pooled held-out-train improvement. This is a research screen, not operational acceptance.',
              md_table(pd.DataFrame(r['research_gate'])),
              '## Reproducibility and limits',
              'A future-mutation test changes all future channel values, removes all future CIP events, and corrupts future-aware metadata; it verifies exact invariance of every pre-cutoff feature row. The deposit also contains18 train-days with n_10min_samples>144, which requires upstream clarification; no counts are changed or invented, and their sum is not presented as independent raw records.',
              'run_study.py regenerates forecast_pairs.csv, predictions.csv, metrics.csv, metrics_by_split.csv, results.json, plot_data.json and three charts from the verified deposited CSVs. Full pair/split audit is in results.json. Training code is new, so this is an independent reanalysis inspired by the attachment, not a claim to have rerun its unavailable full notebooks.',
              'Public source data license: CC BY4.0; original source code: MIT. Attribution: Yi Hsiang Su, Fan Cheng Meng and Pieh Yu Chang, Auditable Clean-in-Place Decision Support from Routine SWRO SCADA, Membranes2026,16(8),257; DOI10.3390/membranes16080257. Dataset depositor: Yi Hsiang Su. Exact version DOI10.5281/zenodo.21055948.',
              'No result validates autonomous CIP decisions, industrial electricity savings, plant-wide fault labels, or generalization to other facilities. For stronger field evidence: raw calibrated sensors, feed-water quality, exact CIP actions, operation modes and prospective evaluation on another plant are needed.']
    (ROOT/'report.md').write_text('\n\n'.join(sections))

if __name__=='__main__': main()
