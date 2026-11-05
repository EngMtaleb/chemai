"""Reproduce industrial RO physics, envelope and sensitivity. No network required."""
from pathlib import Path
import hashlib, json, sys, platform
import numpy as np
from model import solve_point, calibration, DEFAULT

HERE=Path(__file__).resolve().parent

def compact(r):
    keys=["evidence_type","train_q_m3_h","plant_q_m3_day","train_feed_m3_h","train_brine_m3_h",
          "train_power_kW","plant_power_kW","feed_pressure_bar","vessel_dp_bar",
          "permeate_salinity_mg_L","brine_salinity_g_L","SEC_kWh_m3","RO_SEC_kWh_m3",
          "power_components_train_kW","feasible","violations","water_balance_residual_m3_h",
          "salt_balance_residual_kg_h"]
    return {k:r[k] for k in keys}

def main():
    cases={
       "clean_design":{},
       "winter_15C":{"temperature_C":15},
       "hot_35C":{"temperature_C":35},
       "high_salinity_50gL":{"feed_salinity_g_L":50},
       "permeability_loss_20pct":{"A_factor":.8},
       "channel_resistance_3x":{"dp_factor":3},
       "salt_permeability_2_5x":{"B_factor":2.5},
       "combined_degradation":{"A_factor":.7,"B_factor":1.5,"dp_factor":2},
       "mild_degradation":{"A_factor":.85,"B_factor":1.15,"dp_factor":1.5},
       "hot_salt_passage":{"temperature_C":35,"B_factor":2.5},
       "overproduction_20LMH":{"target_flux_LMH":20},
       "pump_efficiency_75pct":{"hp_pump_efficiency":.75},
       "without_energy_recovery":{"use_erd":False},
    }
    records=[]
    for name,ov in cases.items():
        r=solve_point(ov)
        records.append({"id":name,"overrides":ov,**compact(r),"elements":r["elements"]})
    nominal=solve_point()
    points=[]
    states={"clean":{},"mild":{"A_factor":.85,"B_factor":1.15,"dp_factor":1.5},
            "degraded":{"A_factor":.7,"B_factor":1.5,"dp_factor":2}}
    fluxes=[12.,13.26,DEFAULT["target_flux_LMH"]]
    for state,ov in states.items():
        for T in [15,20,25,30,35]:
            for C in [38,43,48,53]:
                for R in [.35,.4,.45]:
                    for f in fluxes:
                        inp={**ov,"temperature_C":T,"feed_salinity_g_L":C,"recovery":R,"target_flux_LMH":f}
                        r=solve_point(inp)
                        points.append({"state":state,"inputs":inp,**compact(r)})
    print(f'Computed {len(points)} operating points',flush=True)
    sensitivity={
       "A_factor":[.85,1.,1.15],"B_factor":[.75,1.,1.25],
       "mass_transfer_factor":[.75,1.,1.25],"dp_factor":[.7,1.,1.3],
       "hp_pump_efficiency":[.75,.85,.9],"erd_pressure_efficiency":[.93,.96,.98],
       "auxiliary_kWh_per_m3":[.25,.45,.75],
       "B_temperature_coefficient_per_C":[.02,.04,.06],
    }
    sens=[]
    for key,values in sensitivity.items():
        for val in values:
            # B-temperature sensitivity must be assessed away from calibration T.
            ov={key:val}
            if key=="B_temperature_coefficient_per_C":ov["temperature_C"]=35
            sens.append({"parameter":key,"value":val,"overrides":ov,**compact(solve_point(ov))})
    # Numerical and physical verification; no validation against field observations.
    hi32=solve_point({"segments_per_element":32});hi64=solve_point({"segments_per_element":64})
    convergence={k:{"n16":nominal[k],"n32":hi32[k],"n64":hi64[k],
                   "relative_n16_vs_n64":abs(nominal[k]/hi64[k]-1)}
                 for k in ["feed_pressure_bar","permeate_salinity_mg_L","SEC_kWh_m3"]}
    end0=solve_point({"target_flux_LMH":12});end1=solve_point()
    mid=solve_point({"target_flux_LMH":(12+DEFAULT["target_flux_LMH"])/2})
    linear=(end0["train_power_kW"]+end1["train_power_kW"])/2
    interpolation={"low_flux":12,"high_flux":DEFAULT["target_flux_LMH"],
       "low_q":end0["train_q_m3_h"],"high_q":end1["train_q_m3_h"],
       "low_power_kW":end0["train_power_kW"],"high_power_kW":end1["train_power_kW"],
       "midpoint_exact_kW":mid["train_power_kW"],"midpoint_linear_kW":linear,
       "midpoint_relative_overestimate":linear/mid["train_power_kW"]-1,
       "slope_kWh_m3":(end1["train_power_kW"]-end0["train_power_kW"])/(end1["train_q_m3_h"]-end0["train_q_m3_h"]),
       "intercept_kW":end0["train_power_kW"]-end0["train_q_m3_h"]*(end1["train_power_kW"]-end0["train_power_kW"])/(end1["train_q_m3_h"]-end0["train_q_m3_h"])}
    all_r=points+records
    verification={
       "water_balance_max_absolute_m3_h":max(abs(r["water_balance_residual_m3_h"]) for r in all_r),
       "salt_balance_max_absolute_kg_h":max(abs(r["salt_balance_residual_kg_h"]) for r in all_r),
       "calibration":calibration(),"axial_resolution":convergence,
       "dispatch_linearization":interpolation,
       "all_nominal_design_gates_pass":nominal["feasible"],
       "pressure_increases_when_A_falls":records[4]["feed_pressure_bar"]>nominal["feed_pressure_bar"],
       "energy_rises_without_ERD":records[-1]["SEC_kWh_m3"]>nominal["SEC_kWh_m3"],
       "salt_passage_gate_detects_salt_stress":not records[6]["feasible"],
    }
    assert verification["water_balance_max_absolute_m3_h"]<1e-8
    assert verification["salt_balance_max_absolute_kg_h"]<1e-7
    assert max(v["relative_n16_vs_n64"] for v in convergence.values())<.01
    assert verification["all_nominal_design_gates_pass"]
    assert verification["pressure_increases_when_A_falls"]
    assert verification["energy_rises_without_ERD"]
    assert verification["salt_passage_gate_detects_salt_stress"]
    result={"title":"Industrial SWRO physics study","status":"recomputed_scenario_results",
            "config":DEFAULT,"calibration":calibration(),"nominal":nominal,"scenarios":records,
            "sensitivity":sens,"verification":verification,
            "operating_grid_summary":{"points":len(points),"feasible":sum(p["feasible"] for p in points)},
            "runtime":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform()}}
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (HERE/'operating_grid.json').write_text(json.dumps(points,ensure_ascii=False,separators=(',',':')))
    (HERE/'dispatch_endpoints.json').write_text(json.dumps({"clean":{"low":compact(end0),"high":compact(end1)},
        "mild":{"low":compact(solve_point({**states['mild'],'target_flux_LMH':12})),
                "high":compact(solve_point(states['mild']))},
        "degraded":{"low":compact(solve_point({**states['degraded'],'target_flux_LMH':12})),
                    "high":compact(solve_point(states['degraded']))},
        "linearization":interpolation},indent=2))
    print(json.dumps({"nominal":compact(nominal),"verification":verification,"scenarios":[{k:r[k] for k in ['id','feed_pressure_bar','SEC_kWh_m3','permeate_salinity_mg_L','feasible','violations']} for r in records]},indent=2),flush=True)
    from make_plots import make_plots
    make_plots(result,points)
    manifest=[]
    for f in sorted(HERE.iterdir()):
        if f.is_file() and f.name!='manifest.json':
            manifest.append({"name":f.name,"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size})
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
