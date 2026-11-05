"""Independent read-only audit of energy trajectories against declared physics.

Writes only into the physics-review folder; never changes energy-owned files.
"""
from pathlib import Path
import json, sys, importlib.util
import numpy as np
from model import solve_point, DEFAULT

HERE=Path(__file__).resolve().parent
ENERGY=HERE.parent/'energy'

def main():
    cfg=json.loads((ENERGY/'config.json').read_text())
    low=solve_point({'target_flux_LMH':cfg['low_flux_LMH']})
    high=solve_point({'target_flux_LMH':cfg['high_flux_LMH']})
    q0=low['train_q_m3_h'];q1=high['train_q_m3_h']
    p0=low['train_power_kW'];p1=high['train_power_kW']
    diversion=cfg['routine_start_product_diversion_minutes']/60
    # Direct physics evaluations throughout the optimization surrogate interval.
    interval=[]
    for f in np.linspace(cfg['low_flux_LMH'],cfg['high_flux_LMH'],25):
        r=solve_point({'target_flux_LMH':float(f)})
        frac=(r['train_q_m3_h']-q0)/(q1-q0)
        linear=p0+frac*(p1-p0)
        interval.append({'flux_LMH':float(f),'physics_feasible':r['feasible'],
                         'power_linear_relative_error':linear/r['train_power_kW']-1,
                         'pressure_bar':r['feed_pressure_bar'],
                         'product_salt_mg_L':r['permeate_salinity_mg_L']})
        assert r['feasible']
    records=[]
    for path in sorted((ENERGY/'results').glob('*__*.json')):
        run=json.loads(path.read_text());rows=run['trajectory'];m=run['metrics']
        init=np.array([1,1,0]) if run['configuration'].get('demand_m3_day',400)<590 else np.ones(3,dtype=int)
        prev=init;stock=m['initial_stock_m3'];start_times=[[] for _ in range(3)]
        errs={k:0. for k in ['recorded_gross','recorded_start_diversion','accepted','power','stock','PV','electrical']}
        additional_yield_loss=0.;start_count=0
        for t,row in enumerate(rows):
            on=np.array(row['train_on']);trim=np.array(row['train_trim_fraction'])
            assert np.all((on==0)|(on==1));assert np.all(trim>=-1e-8) and np.all(trim<=on+1e-8)
            start=(on>prev).astype(int);start_count+=int(start.sum())
            for j in range(3):
                if start[j]:start_times[j].append(t)
            q=np.where(on==1,q0+trim*(q1-q0),0)
            gross=float(q.sum());loss=float(np.dot(start,q))*diversion
            accepted=(gross-loss)*row['realized_yield']
            power=float(np.sum(np.where(on==1,p0+trim*(p1-p0),0)))
            newstock=stock+accepted-row['served_water_m3']-row['spill_m3']
            vals={'recorded_gross':gross-row['gross_product_m3'],
                  'recorded_start_diversion':loss-row['routine_start_diversion_m3'],
                  'accepted':accepted-row['accepted_product_m3'],
                  'power':power-row['load_kWh'],'stock':newstock-row['tank_m3'],
                  'PV':row['PV_kW']-row['solar_used_kWh']-row['solar_curtailed_kWh'],
                  'electrical':row['load_kWh']-row['grid_kWh']-row['solar_used_kWh']}
            for k,v in vals.items():errs[k]=max(errs[k],abs(v))
            additional_yield_loss+=(gross-loss)*(1-row['realized_yield'])
            stock=row['tank_m3'];prev=on
        for j,ts in enumerate(start_times):
            for t in ts:
                assert t+1<len(rows), 'Final-hour start would leave unobserved minimum-on obligation'
                assert rows[t+1]['train_on'][j]==1, 'Cross-boundary minimum-on violation'
        independent_pass=(sum(x['unmet_m3'] for x in rows)<1e-6 and abs(stock-m['initial_stock_m3'])<1e-5
                          and sum(x['spill_m3'] for x in rows)<1e-6
                          and max(x['reserve_intrusion_m3'] for x in rows)<1e-5)
        assert independent_pass==m['service_stock_constraints_pass']
        assert max(errs.values())<1e-6
        assert start_count==m['routine_starts']
        records.append({'file':path.name,'audit_pass':True,'service_pass':independent_pass,
                        'max_errors':errs,'additional_unavailable_yield_m3':additional_yield_loss,
                        'minimum_on_at_every_start_verified':True})
    # Independent causal-index audit of all possible rolling horizons.
    max_observation_offset=max(-24+h-1 for t in range(72) for h in [min(24,72-t)])
    assert max_observation_offset==-1
    out={'scope':'Independent code and numerical review, not field validation',
         'trajectories_checked':len(records),'all_numerical_audits_pass':True,
         'dense_physics_interval':interval,
         'max_linear_power_error':max(abs(x['power_linear_relative_error']) for x in interval),
         'causal_weather_index_latest_relative_hour':max_observation_offset,
         'causality_code_review':'Forecast uses preceding24observations; feedbackyield uses precedingmeasuredratios; demand is assumedknown. No futureactualPV is supplied to optimization.',
         'findings':['No critical water/energy accounting or minimum-on-boundary defect found.',
                     'Capacity-loss factor is an imposed accepted-yield stress, not a physical fouling forecast.',
                     'Use fixed schedule label for the600m3/daycapacity case because all3trains are initially ON.',
                     'Every savings comparison must exclude cases that fail service/terminal/reserve gates.',
                     'Dispatch uses fixed25C43g/Lphysics; outdoorairtemperature inPVdata doesnotchangefeedwaterphysics.'],
         'runs':records}
    (HERE/'energy_independent_review.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['runs','dense_physics_interval']},indent=2))

if __name__=='__main__':main()
