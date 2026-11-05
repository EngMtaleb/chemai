"""Reproducible modeled industrial SWRO dispatch; no field control or AI forecast.

Run from any directory: python industrial_ro/energy/run_study.py
Source data and physical model are retained with this project.
"""
from __future__ import annotations
import json, sys, math, hashlib, itertools, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'physics'))
from model import solve_point

CFG=json.loads((HERE/'config.json').read_text())
LOW=solve_point({'target_flux_LMH':CFG['low_flux_LMH']})
HIGH=solve_point({'target_flux_LMH':CFG['high_flux_LMH']})
assert LOW['feasible'] and HIGH['feasible'], (LOW,HIGH)
QL=float(LOW['train_q_m3_h']); QH=float(HIGH['train_q_m3_h'])
PL=float(LOW['train_power_kW']); PH=float(HIGH['train_power_kW'])
DQ=QH-QL; DP=PH-PL; DIV=CFG['routine_start_product_diversion_minutes']/60

def active_h(n,starts): return n-DIV*starts

def nominal_rule(t):
    """Clock rule: one train 23:00-06:00; three08:00-17:00; two otherwise."""
    h=t%24
    return 3 if 8<=h<17 else 1 if (h<6 or h>=23) else 2

def reference_rule(N,demand,I0,initial):
    u=np.array([[int(j<nominal_rule(t)) for j in range(3)] for t in range(N)])
    prev=np.vstack([initial,u[:-1]])
    starts=np.maximum(0,u-prev)
    eff=u.sum()-DIV*starts.sum()
    q=demand.sum()/eff
    # This balanced clock schedule is a nominal reference. Tank feedback may
    # depart from it. It does not see future solar observations or yield faults.
    flow=(u.sum(axis=1)-DIV*starts.sum(axis=1))*q
    stock=I0+np.cumsum(flow-demand)
    return u, q, np.r_[I0,stock]

def physical_action(u,x,previous,factor):
    starts=np.maximum(0,np.asarray(u)-np.asarray(previous))
    raw=np.asarray(u)*QL+np.asarray(x)*DQ
    restart_loss=DIV*np.sum(starts*raw)
    gross=float(raw.sum())
    accepted=factor*(gross-restart_loss)
    power=float(np.sum(np.asarray(u)*PL+np.asarray(x)*DP))
    return accepted,power,int(starts.sum()),float(restart_loss),gross

def rule_action(t,N,I,cap,reserve,demand,initial,age,fhat,ref):
    """Inventory correction around clock rule, with finite-tank/min-on guard.

    Only current stock, past measured yield, clock and known demand enter.
    A shrinking correction horizon targets the shared final inventory.
    """
    horizon=min(6,N-t)
    ref_q=demand[t]+ref[t+1]-ref[t]
    target=ref_q+(ref[t]-I)/horizon
    if t==N-1: target=demand[t]+ref[-1]-I
    # Keep next stock within reserve/capacity whenever feasible.
    target=max(target,demand[t]+reserve-I)
    target=min(target,demand[t]+cap-I)
    choices=[]
    for bits in itertools.product([0,1],repeat=3):
        u=np.array(bits,dtype=int)
        if np.any((initial==1)&(age<2)&(u==0)): continue
        starts=np.maximum(0,u-initial)
        if t==N-1 and starts.sum(): continue
        n=int(u.sum()); e=active_h(n,int(starts.sum()))
        if n==0: x=np.zeros(3); delivered=0
        else:
            qr=float(np.clip(target/max(fhat*e,1e-12),QL,QH))
            x=u*(qr-QL)/DQ
            delivered=fhat*e*qr
        stock=I+delivered-demand[t]
        violations=max(0,reserve-stock)+max(0,stock-cap)
        # Prefer feasible water service, then match target, then fewer switches.
        score=(1e5*violations+abs(delivered-target)+1e-4*starts.sum()+1e-6*np.dot(u,[0,1,2]))
        choices.append((score,u,x))
    if not choices: raise RuntimeError('No admissible rule action')
    _,u,x=min(choices,key=lambda z:z[0])
    return u,x

def mpc_plan(pv,demand,I,cap,reserve,initial,age,fhat,terminal,time_limit=3):
    """MILP: train on/off and starts, continuous narrow-range trim, grid+stock.

    Linearized physical power curve; product loss only at routine restarts.
    No forecast from future realized PV; caller supplies a causal forecast.
    """
    T=len(pv); J=3; n=14*T
    def iu(j,t):return j*T+t
    def ix(j,t):return 3*T+j*T+t
    def ist(j,t):return 6*T+j*T+t
    def iw(j,t):return 9*T+j*T+t
    def ig(t):return 12*T+t
    def iv(t):return 13*T+t
    lo=np.zeros(n); hi=np.ones(n); integ=np.zeros(n,dtype=int)
    integ[:3*T]=1;integ[6*T:9*T]=1
    hi[12*T:13*T]=np.inf;lo[13*T:]=reserve;hi[13*T:]=cap
    lo[iv(T-1)]=hi[iv(T-1)]=terminal
    c=np.zeros(n);c[12*T:13*T]=1
    for j in range(J):
        c[j*T:(j+1)*T]=.001*PL
        c[3*T+j*T:3*T+(j+1)*T]=.001*DP
        c[6*T+j*T:6*T+(j+1)*T]=.02
        if initial[j] and age[j]<2: lo[iu(j,0)]=1
    row=[];col=[];val=[];rl=[];rh=[]
    def add(d,l=-np.inf,h=np.inf):
        r=len(rl)
        for k,v in d.items():
            if v:row.append(r);col.append(k);val.append(v)
        rl.append(l);rh.append(h)
    for t in range(T):
        water={iv(t):1};grid={ig(t):1}
        if t:water[iv(t-1)]=-1
        for j in range(J):
            u,x,s,w=iu(j,t),ix(j,t),ist(j,t),iw(j,t)
            water.update({u:-fhat*QL,x:-fhat*DQ,s:fhat*DIV*QL,w:fhat*DIV*DQ})
            grid.update({u:-PL,x:-DP})
            add({x:1,u:-1},h=0)
            if t:
                prev=iu(j,t-1)
                add({u:1,prev:-1,s:-1},h=0)
                add({s:1,prev:1},h=1)
            else:
                add({u:1,s:-1},h=int(initial[j]))
                hi[s]=1-int(initial[j])
            add({s:1,u:-1},h=0)
            add({w:1,x:-1},h=0);add({w:1,s:-1},h=0)
            add({w:1,x:-1,s:-1},l=-1)
            if t<T-1: add({u:1,iu(j,t+1):1,s:-2},l=0)
            else: hi[s]=0
        add(water,l=(I if t==0 else 0)-demand[t],h=(I if t==0 else 0)-demand[t])
        add(grid,l=-pv[t])
        # Identical units use a fixed priority. Duty rotation is outside scope.
        add({iu(0,t):1,iu(1,t):-1},l=0)
        add({iu(1,t):1,iu(2,t):-1},l=0)
    A=coo_matrix((val,(row,col)),shape=(len(rl),n)).tocsc()
    result=milp(c,integrality=integ,bounds=Bounds(lo,hi),
                constraints=LinearConstraint(A,np.array(rl),np.array(rh)),
                options={'time_limit':time_limit,'mip_rel_gap':.005,'presolve':True})
    if result.x is None:
        return None,{'status':int(result.status),'message':result.message}
    z=result.x;az=A@z
    valid=(np.all(z>=lo-1e-6) and np.all(z<=hi+1e-6)
           and np.all(az>=np.array(rl)-1e-5) and np.all(az<=np.array(rh)+1e-5)
           and np.all(abs(z[integ==1]-np.round(z[integ==1]))<1e-5))
    if not valid:return None,{'status':int(result.status),'message':'incumbent failed independent constraint check'}
    U=np.array([[round(z[iu(j,t)]) for j in range(J)] for t in range(T)],dtype=int)
    X=np.array([[z[ix(j,t)] for j in range(J)] for t in range(T)])
    return (U,X),{'status':int(result.status),'gap':float(getattr(result,'mip_gap',np.nan)),
                 'optimal_within_requested_gap':bool(result.status==0)}

def simulate(case,policy,source):
    N=72;start=pd.Timestamp(case['date'],tz='Asia/Kuwait')
    times=pd.to_datetime(source.interval_start_kuwait)
    ids=np.flatnonzero((times>=start)&(times<start+pd.Timedelta(hours=N)))
    assert len(ids)==N and ids[0]>=24
    scale=case.get('pv_kWp',320)/4.5
    pv_all=source.pv_kw.to_numpy()*scale; pv=pv_all[ids]
    dday=case.get('demand_m3_day',400);demand=np.full(N,dday/24)
    cap=case.get('tank_m3',200);I0=cap/2;reserve=.1*cap
    I=I0;initial=np.array([1,1,0]) if dday<590 else np.ones(3,dtype=int)
    previous=initial.copy();age=np.where(initial,10,0)
    ftrue=np.ones(N)
    if case.get('yield_step'):ftrue[24:]=case['yield_step']
    _,rq,ref=reference_rule(N,demand,I0,initial)
    # A 600m3/day demand cannot use the daytime2xheadroom schedule: use
    # all-trains-on as the causal nominal reference in this capacity stress.
    if rq>QH:
        ref=np.full(N+1,I0)
    ratios=[];rows=[];solver_calls=[];fallbacks=0
    for t in range(N):
        fhat=float(np.median(ratios[-4:])) if ratios else 1.
        if policy=='fixed_two_trains':
            u=initial.copy();x=u.copy().astype(float)
        elif policy=='daylight_tank_rule':
            u,x=rule_action(t,N,I,cap,reserve,demand,previous,age,fhat,ref)
        elif policy=='rolling_milp':
            H=min(24,N-t)
            forecast=pv_all[ids[t]+np.arange(H)-24]*case.get('forecast_multiplier',1)
            plan,meta=mpc_plan(forecast,demand[t:t+H],I,cap,reserve,previous,age,fhat,I0)
            solver_calls.append({'t':t,**meta})
            if plan is None:
                fallbacks+=1
                u,x=rule_action(t,N,I,cap,reserve,demand,previous,age,fhat,ref)
            else:u,x=plan[0][0],plan[1][0]
        else:raise ValueError(policy)
        predicted,_,_,_,_=physical_action(u,x,previous,1)
        q,p,starts,div,gross=physical_action(u,x,previous,ftrue[t])
        before=I+q-demand[t]
        unmet=max(0,-before);spill=max(0,before-cap);I=float(np.clip(before,0,cap))
        if predicted>1e-10:ratios.append(q/predicted)
        grid=max(0,p-pv[t]);solar=min(p,pv[t])
        rows.append({'hour':t,'time_kuwait':str(times.iloc[ids[t]]),'PV_kW':float(pv[t]),
                     'active_trains':int(u.sum()),'starts':starts,'gross_product_m3':gross,
                     'routine_start_diversion_m3':div,'scenario_yield_loss_m3':gross-div-q,
                     'accepted_product_m3':q,'demand_m3':demand[t],
                     'served_water_m3':demand[t]-unmet,'tank_m3':I,'load_kWh':p,
                     'grid_kWh':grid,'solar_used_kWh':solar,'solar_curtailed_kWh':pv[t]-solar,
                     'unmet_m3':unmet,'spill_m3':spill,'reserve_intrusion_m3':max(0,reserve-I),
                     'realized_yield':float(ftrue[t]),'estimated_yield_before_action':fhat,
                     'train_on':u.tolist(),'train_trim_fraction':np.asarray(x).tolist()})
        age=np.where(u==1,np.where(previous==1,age+1,1),0)
        previous=u.copy()
    def total(k):return float(sum(r[k] for r in rows))
    passed=total('unmet_m3')<1e-6 and abs(I-I0)<1e-5 and total('spill_m3')<1e-6 and max(r['reserve_intrusion_m3'] for r in rows)<1e-5
    metrics={'case':case['name'],'policy':policy,'demand_m3':float(demand.sum()),
             'accepted_product_m3':total('accepted_product_m3'),'served_water_m3':total('served_water_m3'),
             'initial_stock_m3':I0,'final_stock_m3':I,'terminal_error_m3':I-I0,
             'total_electricity_kWh':total('load_kWh'),'grid_kWh':total('grid_kWh'),
             'solar_used_kWh':total('solar_used_kWh'),'PV_available_kWh':float(pv.sum()),
             'PV_share_pct':100*total('solar_used_kWh')/total('load_kWh'),
             'SEC_per_served_m3':total('load_kWh')/max(total('served_water_m3'),1e-12),
             'solar_curtailed_kWh':total('solar_curtailed_kWh'),'unmet_m3':total('unmet_m3'),
             'spill_m3':total('spill_m3'),'min_tank_m3':min(r['tank_m3'] for r in rows),
             'max_tank_m3':max(r['tank_m3'] for r in rows),
             'max_reserve_intrusion_m3':max(r['reserve_intrusion_m3'] for r in rows),
             'routine_starts':int(total('starts')),'routine_start_diversion_m3':total('routine_start_diversion_m3'),
             'scenario_yield_loss_m3':total('scenario_yield_loss_m3'),
             'fallback_hours':fallbacks,'service_stock_constraints_pass':bool(passed),
             'MILP_calls':len(solver_calls),'MILP_limited_incumbents':sum(s.get('status')!=0 and 'gap'in s for s in solver_calls)}
    return {'configuration':case,'metrics':metrics,'trajectory':rows,'solver_log':solver_calls}

def audit(run):
    r=run['trajectory'];m=run['metrics']
    errwater=abs(m['initial_stock_m3']+sum(x['accepted_product_m3']-x['served_water_m3']-x['spill_m3'] for x in r)-m['final_stock_m3'])
    errpower=max(abs(x['load_kWh']-x['grid_kWh']-x['solar_used_kWh']) for x in r)
    pvbalance=max(abs(x['PV_kW']-x['solar_used_kWh']-x['solar_curtailed_kWh']) for x in r)
    assert errwater<1e-7 and errpower<1e-8 and pvbalance<1e-8
    assert all(0<=x['active_trains']<=3 for x in r)
    assert all(-1e-9<=f<=1+1e-9 for x in r for f in x['train_trim_fraction'])
    # Only deliberate starts count; verify2hminimum operation inside horizon.
    for t in range(len(r)-1):
        prev=np.array([1,1,0]) if run['configuration'].get('demand_m3_day',400)<590 else np.ones(3)
        if t:prev=np.array(r[t-1]['train_on'])
        start=np.array(r[t]['train_on'])>prev
        assert np.all(np.array(r[t+1]['train_on'])[start]==1)
    return {'water_balance_residual_m3':errwater,'electrical_balance_residual_kWh':errpower,
            'PV_balance_residual_kWh':pvbalance,'within_horizon_minimum_on_verified':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case');ap.add_argument('--policy');args=ap.parse_args()
    src=pd.read_csv(HERE/'data/solar_hourly.csv')
    cases=[{'name':d[:7]+'_nominal','date':d} for d in CFG['seasons']]
    cases += [{'name':'october_yield90','date':'2023-10-15','yield_step':.9},
              {'name':'october_forecast140','date':'2023-10-15','forecast_multiplier':1.4},
              {'name':'october_tank6h','date':'2023-10-15','tank_m3':100},
              {'name':'october_tank24h','date':'2023-10-15','tank_m3':400},
              {'name':'october_PV240','date':'2023-10-15','pv_kWp':240},
              {'name':'october_PV400','date':'2023-10-15','pv_kWp':400},
              {'name':'october_capacity_limit_yield90','date':'2023-10-15','demand_m3_day':600,'tank_m3':300,'yield_step':.9}]
    if args.case:cases=[c for c in cases if c['name']==args.case]
    policies=CFG['policies'] if not args.policy else [args.policy]
    runs=[];HERE.joinpath('results').mkdir(exist_ok=True)
    for case in cases:
        for p in policies:
            path=HERE/'results'/f"{case['name']}__{p}.json"
            began=time.monotonic();run=simulate(case,p,src);run['audit']=audit(run)
            path.write_text(json.dumps(run,indent=2,allow_nan=False));runs.append(run)
            m=run['metrics'];print(case['name'],p,'grid',round(m['grid_kWh'],2),'PV%',round(m['PV_share_pct'],2),'terminal',round(m['terminal_error_m3'],6),'pass',m['service_stock_constraints_pass'],'fallback',m['fallback_hours'],'seconds',round(time.monotonic()-began,1),flush=True)
    # Merge complete runs retained from the same explicitconfiguration.
    allruns=[json.loads(p.read_text()) for p in sorted((HERE/'results').glob('*__*.json'))]
    summary={'scope':'Newly executed modeled industrial dispatch; real climate-derivedPV,reduced-order industrialROphysics,declared operating assumptions. No live plant,learnedsolarforecast,orvalidatedfoulingcontroller.',
             'config':CFG,'physical_endpoints':{'low':LOW,'high':HIGH},
             'runs':[{'configuration':r['configuration'],'metrics':r['metrics'],'audit':r['audit']} for r in allruns]}
    (HERE/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()
