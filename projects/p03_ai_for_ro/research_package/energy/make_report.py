"""Create compact display arrays, English figures and a sourced study report."""
from pathlib import Path
import json,hashlib,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/industrial-ro-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
POL=['fixed_two_trains','daylight_tank_rule','rolling_milp']
LABEL={'fixed_two_trains':'Fixed schedule','daylight_tank_rule':'Daylight + tank rule','rolling_milp':'Rolling optimization'}
COLORS={'fixed_two_trains':'#64748b','daylight_tank_rule':'#06a6b4','rolling_milp':'#153b73'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','axes.labelcolor':'#253858','text.color':'#153453','xtick.color':'#456074','ytick.color':'#456074','svg.fonttype':'none'})

def main():
    files=sorted((HERE/'results').glob('*__*.json'))
    runs=[]
    for p in files:
        r=json.loads(p.read_text())
        # Accounting-only explicit split for an already simulated yield stress.
        for z in r['trajectory']:
            z['scenario_yield_loss_m3']=z['gross_product_m3']-z['routine_start_diversion_m3']-z['accepted_product_m3']
        r['metrics']['scenario_yield_loss_m3']=sum(z['scenario_yield_loss_m3'] for z in r['trajectory'])
        p.write_text(json.dumps(r,indent=2,allow_nan=False));runs.append(r)
    lookup={(r['metrics']['case'],r['metrics']['policy']):r for r in runs}
    summary=json.loads((HERE/'results.json').read_text())
    summary['runs']=[{'configuration':r['configuration'],'metrics':r['metrics'],'audit':r['audit']} for r in runs]
    summary['policy_labels']=LABEL
    nom=['2023-01_nominal','2023-04_nominal','2023-07_nominal','2023-10_nominal']
    seasonal=[]
    for case in nom:
        b=lookup[case,POL[0]]['metrics']
        for pol in POL:
            m=lookup[case,pol]['metrics']
            seasonal.append({'case':case,'policy':pol,'PV_share_pct':m['PV_share_pct'],
                'grid_kWh':m['grid_kWh'],'load_kWh':m['total_electricity_kWh'],
                'grid_reduction_vs_fixed_pct':100*(1-m['grid_kWh']/b['grid_kWh']),
                'total_energy_change_vs_fixed_pct':100*(m['total_electricity_kWh']/b['total_electricity_kWh']-1),
                'pass':m['service_stock_constraints_pass']})
    summary['seasonal_comparison']=seasonal
    summary['scope_note']='Four predetermined72hourwindows; no annual RO savings extrapolation; no tariff/capitaloptimization. Weather-derivedPVdata retained fromyear2023; ROfeedtemperature fixed25C.'
    (HERE/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    figs=HERE/'figures';figs.mkdir(exist_ok=True)
    x=np.arange(4);width=.24
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for i,p in enumerate(POL):
        rr=[lookup[c,p]['metrics'] for c in nom]
        xx=x+(i-1)*width
        axs[0].bar(xx,[r['grid_kWh'] for r in rr],width,label=LABEL[p],color=COLORS[p])
        axs[1].bar(xx,[r['PV_share_pct'] for r in rr],width,label=LABEL[p],color=COLORS[p])
    for ax in axs:
        ax.set_xticks(x,['January','April','July','October']);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    axs[0].set(ylabel='Grid electricity over 72 h [kWh]',title='Same water demand and terminal inventory')
    axs[1].set(ylabel='Solar share of operating electricity [%]',title='Timing benefit, with the same hardware',ylim=(0,80))
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Modeled industrial RO dispatch | 400 m³/day demand, 320 kWp PV, 200 m³ tank',fontsize=16)
    for ext in ['svg','png']:fig.savefig(figs/f'seasonal_dispatch.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(14,9),sharex=True,layout='constrained')
    pv=[r['PV_kW'] for r in lookup['2023-10_nominal',POL[0]]['trajectory']]
    axs[0].fill_between(np.arange(72),pv,color='#f5ca67',alpha=.55,label='Modeled PV available')
    for p in POL:
        tr=lookup['2023-10_nominal',p]['trajectory'];t=[r['hour'] for r in tr]
        axs[0].step(t,[r['load_kWh'] for r in tr],where='post',c=COLORS[p],label=LABEL[p])
        axs[1].step(np.arange(73),[100]+[r['tank_m3'] for r in tr],where='post',c=COLORS[p])
        axs[2].step(t,[r['active_trains'] for r in tr],where='post',c=COLORS[p])
    axs[0].set(ylabel='Hourly average power [kW]');axs[0].legend(loc='upper right',ncol=2,frameon=False,fontsize=10)
    axs[1].axhline(20,c='#b45309',ls='--',alpha=.6);axs[1].axhline(100,c='#64748b',ls=':',alpha=.6);axs[1].axhline(200,c='#b45309',ls='--',alpha=.6)
    axs[1].set(ylabel='Accepted-water stock [m³]',ylim=(0,210))
    axs[2].set(ylabel='Operating trains',yticks=[0,1,2,3],xlabel='Hours from 15 October 2023, 00:00 Kuwait')
    for ax in axs:ax.grid(alpha=.12);ax.set_xlim(0,72)
    fig.suptitle('October replay: shift production while preserving water service',fontsize=17)
    for ext in ['svg','png']:fig.savefig(figs/f'october_timeline.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,case,title in zip(axs,['october_yield90','october_capacity_limit_yield90'],['400 m³/day demand: headroom available','600 m³/day demand: no production headroom']):
        initial=lookup[case,POL[0]]['metrics']['initial_stock_m3']
        for p in POL:
            tr=lookup[case,p]['trajectory']
            ax.step(np.arange(73),[initial]+[r['tank_m3'] for r in tr],where='post',label=LABEL[p],c=COLORS[p])
        ax.axvline(24,color='#b45309',ls='--',label='Accepted-yield step 1.0 → 0.9')
        ax.axhline(initial,color='#64748b',ls=':',alpha=.7)
        ax.set(title=title,xlabel='Elapsed hours',ylabel='Accepted-water stock [m³]',xlim=(0,72));ax.grid(alpha=.12)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Prescribed yield-loss sensitivity |not a learned fouling law',fontsize=17)
    for ext in ['svg','png']:fig.savefig(figs/f'yield_and_capacity_stress.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    plots={'policy_labels':LABEL,'seasonal':seasonal,
           'october_nominal':{p:lookup['2023-10_nominal',p]['trajectory'] for p in POL},
           'yield90':{p:lookup['october_yield90',p]['trajectory'] for p in POL},
           'capacity_limit':{p:lookup['october_capacity_limit_yield90',p]['trajectory'] for p in POL},
           'sensitivity':[r['metrics'] for r in runs if r['metrics']['case'] not in nom]}
    (HERE/'plot_data.json').write_text(json.dumps(plots,allow_nan=False))
    source=json.loads((HERE/'data/solar_source.json').read_text())
    source['new_study_use']={'PV_kWp':320,'scaling':'Originalmodeled4.5kWparraytimes320/4.5; sameorientation/module/loss assumptions. Inverter/landdesignnotoptimized.','RO_feed_temperature_C':25,'weather_air_temperature_is_not_RO_feed_temperature':True,'all8760sourcehours_retained':True,'selected_replay_hours':288}
    source['independently_opened_primary_pages_2026_10_30']=['https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-user-manual_en','https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en']
    (HERE/'sources.json').write_text(json.dumps(source,indent=2))
    table=['|Window|Policy|Grid kWh|Total kWh|PV share|Stock/service pass|','|---|---|---:|---:|---:|---|']
    for c in nom:
        for p in POL:
            m=lookup[c,p]['metrics'];table.append(f"|{c[:7]}|{LABEL[p]}|{m['grid_kWh']:.2f}|{m['total_electricity_kWh']:.2f}|{m['PV_share_pct']:.2f}%|{m['service_stock_constraints_pass']}|")
    stress=['|Case|Policy|Unserved m³|Terminal error m³|Grid kWh|Pass|','|---|---|---:|---:|---:|---|']
    for c in ['october_yield90','october_forecast140','october_capacity_limit_yield90']:
        for p in POL:
            m=lookup[c,p]['metrics'];stress.append(f"|{c}|{LABEL[p]}|{m['unmet_m3']:.3f}|{m['terminal_error_m3']:.3f}|{m['grid_kWh']:.2f}|{m['service_stock_constraints_pass']}|")
    octm=[lookup['2023-10_nominal',p]['metrics'] for p in POL]
    gr=100*(1-octm[2]['grid_kWh']/octm[0]['grid_kWh']);tc=100*(octm[2]['total_electricity_kWh']/octm[0]['total_electricity_kWh']-1)
    report=f'''# Industrial RO solar and storage study — executed 30 October 2026

## Evidence and objective

This is a newly executed engineering scenario study. Water and electricity relations come from the industrial physics model; the solar source is **PVGIS-ERA5 modeled PV**, not a photovoltaic meter. The daylight rule and MILP controller are not learned AI. No real plant was controlled. Four predetermined 72-hour windows do not establish annual savings.

## Common plant and fair comparison

Three switchable trains have 600 m³/day gross nominal capacity; the main demand is 400 m³/day. All policies have the same 200 m³ water tank, 320 kWp PV, no battery, no export and unrestricted grid backup. Initial and required final stock are both 100 m³, with a 20 m³ reserve. The third train provides explicit installed headroom. Capital cost and long-term duty rotation are outside scope.

ON trains modulate only within the physics-checked 6.888–8.333 m³/h interval. The bounded affine power relation multiplies its intercept by the ON state and is never extrapolated to zero flow. Its endpoints require 19.21365–24.01093 kW/train and pass the physics operating/product-salt gates. At the midpoint, affine power overestimates the physical calculation by 0.1761%. The 500 mg/L salt screen is an illustrative process specification, not potable-water certification. Feed remains 25°C and 43 g/L; outdoor temperature is not substituted for feed temperature.

A routine start diverts the first 5 minutes of modeled product while full production-associated electricity is charged. This and a 2-hour minimum ON time are study assumptions, not measured or OEM-validated transitions. No chemical CIP or post-CIP rinse is simulated. OFF production-associated power is zero. Standby loads, water aging, intake/pretreatment yield losses, chemicals and dynamic flushing hydraulics are not modeled. Simplified feed-pump electricity and a 0.45 kWh/m³ auxiliary allowance come from the physics study.

## Policies

1. **Fixed schedule:** two fully loaded trains continuously supply 400 m³/day. The 600 m³/day stress uses three, despite the retained internal identifier `fixed_two_trains`.
2. **Daylight + tank rule:** nominal clock plan operates one train from 23:00–06:00, three from 08:00–17:00, and two otherwise. Its flow trim balances known demand and routine-start losses. Inventory feedback corrects departures from this reference over up to 6 hours, respecting finite storage and minimum ON time. It sees no future weather.
3. **Rolling optimization:** a MILP replans hourly over up to 24 hours. It minimizes forecast grid use, with small total-electricity and startup tie-break terms. PV forecasts repeat the previous observed 24 hours. A median of the last four completed nonzero-output measurements estimates accepted-yield loss. Tank measurements are exact in this simulation. Every horizon ends at the common target stock. A failed solve uses the logged rule fallback.

MILP settings: 0.5% requested relative gap and 3-second time limit per call. Twelve calls returned time-limited feasible incumbents in this run; each was independently checked against the full constraints. These are not proven global optima. Retained trajectories reproduce the stated accounting exactly; a rerun can choose a slightly different feasible schedule depending on solver timing/hardware.

The checks separate unserved water, reserve intrusion, spillage and terminal stock error. A schedule failing any water/stock condition is excluded from energy-saving rankings. Ending with depleted stock cannot count as a saving.

## Seasonal results

{chr(10).join(table)}

For October, rolling optimization reduces grid import by **{gr:.2f}%** versus fixed operation; total modeled electricity changes by **{tc:+.2f}%**. These are different denominators and different claims. Most of the gain comes from shifting production toward PV with installed headroom and storage. The smaller total-energy change comes from the bounded operating point and startup accounting. It is not an AI-forecast achievement or measured industrial saving.

## Disturbances and limits

{chr(10).join(stress)}

The accepted-yield step to 0.9 at hour 24 deliberately removes useful product while retaining the same electrical draw. Extra unavailable/diverted product is accounted separately from routine-start diversion. This is not a fitted fouling law, a new membrane solution at changed recovery, or a rate estimated from the industrial observations.

At 400 m³/day demand, the rolling controller restores the required final stock after this stress. The fixed and simple-rule policies leave terminal deficits, so their energy numbers are not like-for-like savings. At 600 m³/day demand, a 600 m³/day plant has no capacity to recover sustained yield loss. Initial storage hides the shortfall temporarily: all policies serve demand during this 72-hour window, but all end 120 m³ below target. That is not sustainable steady operation.

The +40% PV forecast stress can improve or worsen a particular realized schedule by chance; it does not establish a general benefit from biased forecasts. Other October sensitivities use 6/12/24 hours of storage and 240/320/400 kWp PV. The extra tank has little benefit in this particular replay once storage is sufficient. No tariff, capital-cost optimum or off-grid autonomy is claimed.

## Verification

The separate physics agent independently reconstructed all 33 trajectories, confirming gross water, startup loss, accepted product, tank, PV, grid and load accounting, as well as every routine start's minimum ON time. Independent pass/fail classifications match: 28 pass, 5 fail. Across 25 physical points within the trim interval, every operating gate passes and the largest power interpolation error is 0.17633%. Review code and output are in `physics/review_energy.py` and `physics/energy_independent_review.json`.

## Reproduction and source

Run `python energy/run_study.py`, then `python energy/make_report.py` from the bundled project. Dependencies: NumPy, Pandas, SciPy and Matplotlib. The sibling `physics/model.py` contains the engineering model. All 8,760 PV hours, original API JSON and metadata are retained in `data/`; no network is required. Every trajectory and solver status is in `results/`. `plot_data.json` contains display arrays; `manifest.json` records file hashes.

Source acknowledgment: **Photovoltaic Geographic Information System (PVGIS) of the European Commission Joint Research Centre**. Public Kuwait City reference point 29.3759°N, 47.9774°E; year 2023; fixed 30° south-facing free-standing crystalline silicon; 14% system-loss setting. Original 4.5 kWp modeled output is scaled linearly to each studied array size. ERA5 timestamps are hourly centers at HH:30 UTC, displayed as hourly intervals in Kuwait (UTC+3). `sources.json` retains the request, hashes and interpretation.

- https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en
- https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-user-manual_en
- https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en

**Arabic narration:** «ثبتنا نفس المحطة ونفس الطلب ونفس حجم الخزان عند المقارنة. شغّلناها بثلاث طرق: تشغيل ثابت، وقاعدة بسيطة مرتبطة بالنهار والخزان، وتحسين يعيد حساب الخطة كل ساعة. استفدنا أكثر من الشمس، لكن فرقنا بين تقليل كهرباء الشبكة وتقليل استهلاك التحلية نفسه. وحين نقص الماء المقبول، اختبرنا هل النظام يقدر يعوضه ويرجع الخزان لنفس المستوى. النتائج محاكاة صناعية بشروط واضحة، وتوضح دور التخطيط والمرونة؛ ما ننسب كل فائدة للذكاء الاصطناعي.»
'''
    (HERE/'report.md').write_text(report)
    manifest=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and '__pycache__' not in str(p) and p.name!='manifest.json':
            manifest.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Created',len(runs),'run summaries,3figuresandplot_data.json')
    print('October gridreduction%',gr,'totalenergychange%',tc)

if __name__=='__main__':main()
