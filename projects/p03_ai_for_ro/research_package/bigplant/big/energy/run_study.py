"""Reproducible modeled industrial SWRO dispatch; no field control or AI forecast.

BIGPLANT VERSION (copied from Smart_RO_Complete_Project/energy/run_study.py).
Generalised from 3 hard-coded trains to J trains. Every change from the
original is marked "# BIGPLANT". Plant sizes, train counts, the clock-rule
train counts, the steady-policy name and the stress-case list are read from
the config file (--config), so the SAME code reproduces the original 3-train
study (config_original_size.json) and runs the 12-train plant (config.json).
Policy logic, horizons, MILP gap/time limit and the fallback are unchanged.

Run: python big/energy/run_study.py [--config F] [--results DIR] [--case C] [--policy P]
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

# BIGPLANT: config file, results folder and (for a sensitivity run only) the
# tie-break scale are selectable. Original: fixed config.json and results/.
_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--config',default=str(HERE/'config.json'))
_pre.add_argument('--results',default=str(HERE/'results'))
_pre.add_argument('--tie-break-scale',type=float,default=None)
_PRE,_=_pre.parse_known_args()
CFG=json.loads(Path(_PRE.config).read_text())
RESULTS=Path(_PRE.results)
# BIGPLANT: train layout passed to the physics model explicitly (original used
# the model DEFAULT of 2 vessels/train, 3 trains).
LAYOUT={'vessels_per_train':CFG['vessels_per_train'],'trains':CFG['trains']}
LOW=solve_point({'target_flux_LMH':CFG['low_flux_LMH'],**LAYOUT})
HIGH=solve_point({'target_flux_LMH':CFG['high_flux_LMH'],**LAYOUT})
assert LOW['feasible'] and HIGH['feasible'], (LOW,HIGH)
QL=float(LOW['train_q_m3_h']); QH=float(HIGH['train_q_m3_h'])
PL=float(LOW['train_power_kW']); PH=float(HIGH['train_power_kW'])
DQ=QH-QL; DP=PH-PL; DIV=CFG['routine_start_product_diversion_minutes']/60

# BIGPLANT: train count and derived constants (original: 3 hard-coded).
J_TRAINS=int(CFG['trains'])
TRAIN_INDEX=np.arange(J_TRAINS)                      # original: [0,1,2]
STEADY=np.array([1]*int(CFG['steady_trains'])+[0]*(J_TRAINS-int(CFG['steady_trains'])),dtype=int)  # original: [1,1,0]
CAPACITY_M3_DAY=J_TRAINS*QH*24
# BIGPLANT: original test "demand < 590 m3/day" (all trains start ON at or above it)
# on a 600 m3/day plant; kept in the same proportion of full capacity.
FULL_START_DEMAND=590/600*CAPACITY_M3_DAY
FIXED_POLICY=CFG['fixed_policy']                     # original: 'fixed_two_trains'
CLOCK=CFG['clock_rule_trains']                       # original: 1 / 3 / 2 trains
# BIGPLANT: three tie-break weights are absolute numbers sized for a 2-vessel
# train (rule: 1e-4 per start and 1e-6 per train index, in m3; MILP: 0.02 per
# start, in kWh). They are multiplied by the per-train size ratio
# (vessels per train / 2) so they keep the same proportion to per-train flow
# and power (the MILP's other tie-break, 0.001*train power, already scales).
# Equals exactly 1 for the original plant. --tie-break-scale overrides it
# (used only for the literal-weights sensitivity run).
TRAIN_SCALE=CFG['vessels_per_train']/2 if _PRE.tie_break_scale is None else float(_PRE.tie_break_scale)
assert sum(STEADY)*QH*24<=CAPACITY_M3_DAY and CLOCK['day_08_17']<=J_TRAINS
if 'initial_on_state' in CFG: assert list(STEADY)==CFG['initial_on_state'], 'config initial_on_state disagrees with steady_trains'

def active_h(n,starts): return n-DIV*starts

def nominal_rule(t):
    """Clock rule: one train 23:00-06:00; three08:00-17:00; two otherwise.

    BIGPLANT: counts come from config clock_rule_trains. For 12 trains they are
    4 / 12 / 8, i.e. the same 1/3, 3/3 and 2/3 of the installed trains."""
    h=t%24
    return CLOCK['day_08_17'] if 8<=h<17 else CLOCK['night_23_06'] if (h<6 or h>=23) else CLOCK['other']

def reference_rule(N,demand,I0,initial):
    u=np.array([[int(j<nominal_rule(t)) for j in range(J_TRAINS)] for t in range(N)])  # BIGPLANT: range(3)
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
    for bits in itertools.product([0,1],repeat=J_TRAINS):  # BIGPLANT: repeat=3 (2^12 = 4096 combinations)
        u=np.array(bits,dtype=int)
        if np.any((initial==1)&(age<2)&(u==0)): continue
        starts=np.maximum(0,u-initial)
        if t==N-1 and starts.sum(): continue
        n=int(u.sum()); e=active_h(n,int(starts.sum()))
        if n==0: x=np.zeros(J_TRAINS); delivered=0  # BIGPLANT: np.zeros(3)
        else:
            qr=float(np.clip(target/max(fhat*e,1e-12),QL,QH))
            x=u*(qr-QL)/DQ
            delivered=fhat*e*qr
        stock=I+delivered-demand[t]
        violations=max(0,reserve-stock)+max(0,stock-cap)
        # Prefer feasible water service, then match target, then fewer switches.
        # BIGPLANT: 1e-4 and 1e-6 multiplied by TRAIN_SCALE; [0,1,2] -> TRAIN_INDEX.
        score=(1e5*violations+abs(delivered-target)+1e-4*TRAIN_SCALE*starts.sum()+1e-6*TRAIN_SCALE*np.dot(u,TRAIN_INDEX))
        choices.append((score,u,x))
    if not choices: raise RuntimeError('No admissible rule action')
    _,u,x=min(choices,key=lambda z:z[0])
    return u,x

def mpc_plan(pv,demand,I,cap,reserve,initial,age,fhat,terminal,time_limit=3):
    """MILP: train on/off and starts, continuous narrow-range trim, grid+stock.

    Linearized physical power curve; product loss only at routine restarts.
    No forecast from future realized PV; caller supplies a causal forecast.
    """
    # BIGPLANT: J taken from the state vector (was J=3); variable blocks are
    # u, x, s, w (J*T each), grid (T), stock (T): n=4*J*T+2*T (was 14*T).
    T=len(pv); J=len(initial); n=4*J*T+2*T
    def iu(j,t):return j*T+t
    def ix(j,t):return J*T+j*T+t        # BIGPLANT: 3*T -> J*T
    def ist(j,t):return 2*J*T+j*T+t     # BIGPLANT: 6*T -> 2*J*T
    def iw(j,t):return 3*J*T+j*T+t      # BIGPLANT: 9*T -> 3*J*T
    def ig(t):return 4*J*T+t            # BIGPLANT: 12*T -> 4*J*T
    def iv(t):return 4*J*T+T+t          # BIGPLANT: 13*T -> 4*J*T+T
    lo=np.zeros(n); hi=np.ones(n); integ=np.zeros(n,dtype=int)
    integ[:J*T]=1;integ[2*J*T:3*J*T]=1                                 # BIGPLANT: [:3*T], [6*T:9*T]
    hi[4*J*T:4*J*T+T]=np.inf;lo[4*J*T+T:]=reserve;hi[4*J*T+T:]=cap     # BIGPLANT: 12*T, 13*T
    lo[iv(T-1)]=hi[iv(T-1)]=terminal
    c=np.zeros(n);c[4*J*T:4*J*T+T]=1                                   # BIGPLANT: 12*T:13*T
    for j in range(J):
        c[j*T:(j+1)*T]=.001*PL
        c[J*T+j*T:J*T+(j+1)*T]=.001*DP                                 # BIGPLANT: 3*T
        c[2*J*T+j*T:2*J*T+(j+1)*T]=.02*TRAIN_SCALE                     # BIGPLANT: 6*T; .02 -> .02*TRAIN_SCALE
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
        # BIGPLANT: priority chain u[0]>=u[1]>=...>=u[J-1] (was the two rows for J=3).
        for j in range(J-1):
            add({iu(j,t):1,iu(j+1,t):-1},l=0)
    A=coo_matrix((val,(row,col)),shape=(len(rl),n)).tocsc()
    began=time.perf_counter()  # BIGPLANT: solve wall time logged (no effect on the solve)
    result=milp(c,integrality=integ,bounds=Bounds(lo,hi),
                constraints=LinearConstraint(A,np.array(rl),np.array(rh)),
                options={'time_limit':time_limit,'mip_rel_gap':.005,'presolve':True})
    wall=time.perf_counter()-began
    if result.x is None:
        return None,{'status':int(result.status),'message':result.message,'wall_s':wall}
    z=result.x;az=A@z
    valid=(np.all(z>=lo-1e-6) and np.all(z<=hi+1e-6)
           and np.all(az>=np.array(rl)-1e-5) and np.all(az<=np.array(rh)+1e-5)
           and np.all(abs(z[integ==1]-np.round(z[integ==1]))<1e-5))
    if not valid:return None,{'status':int(result.status),'message':'incumbent failed independent constraint check','wall_s':wall}
    U=np.array([[round(z[iu(j,t)]) for j in range(J)] for t in range(T)],dtype=int)
    X=np.array([[z[ix(j,t)] for j in range(J)] for t in range(T)])
    return (U,X),{'status':int(result.status),'gap':float(getattr(result,'mip_gap',np.nan)),
                 'optimal_within_requested_gap':bool(result.status==0),'wall_s':wall}

def simulate(case,policy,source):
    N=72;start=pd.Timestamp(case['date'],tz='Asia/Kuwait')
    times=pd.to_datetime(source.interval_start_kuwait)
    ids=np.flatnonzero((times>=start)&(times<start+pd.Timedelta(hours=N)))
    assert len(ids)==N and ids[0]>=24
    scale=case.get('pv_kWp',CFG['PV_kWp'])/4.5   # BIGPLANT: default 320 -> config PV_kWp (4.5 kWp = PVGIS source array, unchanged)
    pv_all=source.pv_kw.to_numpy()*scale; pv=pv_all[ids]
    dday=case.get('demand_m3_day',CFG['demand_m3_day']);demand=np.full(N,dday/24)  # BIGPLANT: default 400 -> config
    cap=case.get('tank_m3',CFG['tank_m3']);I0=cap/2;reserve=.1*cap               # BIGPLANT: default 200 -> config
    # BIGPLANT: was initial=np.array([1,1,0]) if dday<590 else np.ones(3,dtype=int)
    I=I0;initial=STEADY.copy() if dday<FULL_START_DEMAND else np.ones(J_TRAINS,dtype=int)
    previous=initial.copy();age=np.where(initial,10,0)
    ftrue=np.ones(N)
    if case.get('yield_step'):ftrue[24:]=case['yield_step']
    _,rq,ref=reference_rule(N,demand,I0,initial)
    # A 600m3/day demand cannot use the daytime2xheadroom schedule: use
    # all-trains-on as the causal nominal reference in this capacity stress.
    # (BIGPLANT: test unchanged; rq and QH are both per-train flows.)
    if rq>QH:
        ref=np.full(N+1,I0)
    ratios=[];rows=[];solver_calls=[];fallbacks=0
    for t in range(N):
        fhat=float(np.median(ratios[-4:])) if ratios else 1.
        if policy==FIXED_POLICY:   # BIGPLANT: was 'fixed_two_trains'; 'fixed_eight_trains' for 12 trains
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
    assert all(0<=x['active_trains']<=J_TRAINS for x in r)   # BIGPLANT: <=3
    assert all(-1e-9<=f<=1+1e-9 for x in r for f in x['train_trim_fraction'])
    # Only deliberate starts count; verify2hminimum operation inside horizon.
    for t in range(len(r)-1):
        # BIGPLANT: was np.array([1,1,0]) if demand<590 (default 400) else np.ones(3)
        prev=STEADY.copy() if run['configuration'].get('demand_m3_day',CFG['demand_m3_day'])<FULL_START_DEMAND else np.ones(J_TRAINS)
        if t:prev=np.array(r[t-1]['train_on'])
        start=np.array(r[t]['train_on'])>prev
        assert np.all(np.array(r[t+1]['train_on'])[start]==1)
    return {'water_balance_residual_m3':errwater,'electrical_balance_residual_kWh':errpower,
            'PV_balance_residual_kWh':pvbalance,'within_horizon_minimum_on_verified':True}

def main():
    # BIGPLANT: parent parser adds --config/--results/--tie-break-scale.
    ap=argparse.ArgumentParser(parents=[_pre]);ap.add_argument('--case');ap.add_argument('--policy');args=ap.parse_args()
    src=pd.read_csv(HERE/'data/solar_hourly.csv')
    cases=[{'name':d[:7]+'_nominal','date':d} for d in CFG['seasons']]
    cases += CFG['stress_cases']   # BIGPLANT: list moved to config (tank/PV/capacity sizes are plant-specific)
    if args.case:cases=[c for c in cases if c['name']==args.case]
    policies=CFG['policies'] if not args.policy else [args.policy]
    runs=[];RESULTS.mkdir(parents=True,exist_ok=True)
    for case in cases:
        for p in policies:
            path=RESULTS/f"{case['name']}__{p}.json"
            began=time.monotonic();run=simulate(case,p,src);run['audit']=audit(run)
            run['runtime_s']=time.monotonic()-began   # BIGPLANT: wall time stored (original only printed it)
            run['tie_break_scale']=TRAIN_SCALE        # BIGPLANT: provenance of the tie-break weights
            path.write_text(json.dumps(run,indent=2,allow_nan=False));runs.append(run)
            m=run['metrics'];print(case['name'],p,'grid',round(m['grid_kWh'],2),'PV%',round(m['PV_share_pct'],2),'terminal',round(m['terminal_error_m3'],6),'pass',m['service_stock_constraints_pass'],'fallback',m['fallback_hours'],'seconds',round(time.monotonic()-began,1),flush=True)
    # Merge complete runs retained from the same explicitconfiguration.
    allruns=[json.loads(p.read_text()) for p in sorted(RESULTS.glob('*__*.json'))]
    summary={'scope':'Newly executed modeled industrial dispatch; real climate-derivedPV,reduced-order industrialROphysics,declared operating assumptions. No live plant,learnedsolarforecast,orvalidatedfoulingcontroller.',
             'config':CFG,'physical_endpoints':{'low':LOW,'high':HIGH},
             'tie_break_scale':TRAIN_SCALE,
             'runs':[{'configuration':r['configuration'],'metrics':r['metrics'],'audit':r['audit'],'runtime_s':r.get('runtime_s')} for r in allruns]}
    # BIGPLANT: summary written next to the results folder (original: energy/results.json).
    (RESULTS.parent/(RESULTS.name+'_summary.json')).write_text(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()
