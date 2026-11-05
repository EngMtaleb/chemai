"""Industrial SWRO study model. All outputs are simulated, not plant measurements.

BIGPLANT VERSION (copied from Smart_RO_Complete_Project/physics/model.py).
The ONLY changes are marked "# BIGPLANT": the train layout in DEFAULT
(110 vessels per train instead of 2, 12 trains instead of 3). Membrane model,
element (41 m2), calibration, correlations, efficiencies, gates and the mean
flux (14.518 LMH) are unchanged.

Seven 8-inch elements per pressure vessel, 110 vessels per train, twelve trains
(original: two vessels per train, three trains).
Solution diffusion + film concentration polarization; explicit differential
water/salt balances. A25/B25 are fitted only to a manufacturer's element point.
Hydraulic/transfer correlations and efficiencies remain disclosed assumptions.
Run `python run_study.py` in this directory to reproduce the study.
"""
from __future__ import annotations
from functools import lru_cache
from math import exp, log
import numpy as np
from scipy.optimize import brentq, least_squares

DEFAULT = {
    "temperature_C": 25.0, "feed_salinity_g_L": 43.0,
    # BIGPLANT: flux deliberately kept at the original 14.518 LMH. The original
    # expression is retained so the value is bit-identical; it equals
    # 132,000 m3/day / (24 h * 12 trains * 110 vessels * 7 elements * 41 m2)
    # and gives 100 m3/day per 7-element vessel (checked in tools/check_layout.py).
    "target_flux_LMH": 600000.0 / (24 * 3 * 2 * 7 * 41),
    "recovery": 0.40, "A_factor": 1.0, "B_factor": 1.0,
    "dp_factor": 1.0, "mass_transfer_factor": 1.0,
    # BIGPLANT: was "vessels_per_train": 2, "trains": 3
    "elements_per_vessel": 7, "vessels_per_train": 110, "trains": 12,
    "element_area_m2": 41.0, "segments_per_element": 16,
    "permeate_pressure_bar": 0.5,
    "k25_LMH_at_10m3h": 180.0, "k_flow_exponent": 0.6,
    "element_dp_bar_at_8m3h": 0.15, "dp_flow_exponent": 1.7,
    "B_temperature_coefficient_per_C": 0.04,
    "hp_pump_efficiency": 0.85, "booster_efficiency": 0.80,
    "erd_pressure_efficiency": 0.96,
    "low_pressure_bar": 3.0, "feed_pump_efficiency": 0.75,
    "auxiliary_kWh_per_m3": 0.45, "use_erd": True,
    "max_feed_pressure_bar": 80.0, "max_vessel_dp_bar": 3.5,
    "max_element_dp_bar": 1.0, "max_element_average_flux_LMH": 32.0,
    "max_element_recovery": 0.13, "max_feed_per_vessel_m3h": 13.0,
    "min_brine_per_vessel_m3h": 3.5,
    "max_product_salinity_mg_L": 500.0,
    "min_temperature_C": 15.0, "max_temperature_C": 35.0,
}

DATASHEET = {"temperature_C":25.0,"feed_salinity_g_L":32.0,
             "pressure_bar":55.0,"recovery":0.08,
             "permeate_m3_day":30.2,"salt_rejection":0.998,
             "area_m2":41.0}

def tcf(T: float) -> float:
    """Manufacturer water-permeability temperature correction (dimensionless)."""
    return exp((2640.0 if T >= 25 else 3020.0) * (1/298.0 - 1/(273.0+T)))

def viscosity_ratio(T: float) -> float:
    """Simplified pure-water viscosity temperature dependence; study assumption."""
    # Andrade-type water viscosity approximation; salinity influence omitted.
    return 10 ** (247.8/(T+133.15) - 247.8/(25+133.15))

def osmotic_bar(c_g_L: float, T: float) -> float:
    """DuPont practical total-salt approximation; not ion-resolved chemistry."""
    c = max(0.0, c_g_L) * 1000.0
    if c < 20000:
        return c * (T + 320.0) / 491000.0
    return (0.0117*c - 34.0) / 14.23 * (T+320.0)/345.0

def local_flux(c: float, p: float, A: float, B: float, k: float, T: float,
               pp: float) -> tuple[float,float,float]:
    """J in LMH; cp/cm in g/L. Film theory and salt balance solved jointly."""
    def concentrations(j):
        e=exp(min(j/k,5.0))
        den=j+B*e
        cp = B*c*e/den if den > 0 else c
        cm = c*e*(j+B)/den if den > 0 else c
        return cp,cm
    def residual(j):
        cp,cm=concentrations(j)
        return j-A*(p-pp-osmotic_bar(cm,T)+osmotic_bar(cp,T))
    if p <= pp: return 0.0,c,c
    # At j=0, finite diffusive salt flux equilibrates both sides: cp=cm=c.
    hi=max(A*(p-pp),1.0)
    j=brentq(residual,0.0,hi,xtol=1e-10)
    cp,cm=concentrations(j)
    return j,cp,cm

def vessel(pin: float, qfeed: float, cfeed: float, T: float,
           A25: float, B25: float, cfg: dict) -> dict:
    n=int(cfg["elements_per_vessel"]); ns=int(cfg["segments_per_element"])
    da=cfg["element_area_m2"]/ns
    A=A25*tcf(T)*cfg["A_factor"]
    B=B25*exp(cfg["B_temperature_coefficient_per_C"]*(T-25))*cfg["B_factor"]
    mu=viscosity_ratio(T)
    q=qfeed; c=cfeed; p=pin; total_qp=0.0; total_salt=0.0
    profiles=[]; el=[]
    for eidx in range(n):
        eq=q; ep=p; eqp=0.0; esalt=0.0
        for sidx in range(ns):
            dp=cfg["element_dp_bar_at_8m3h"]*(q/8)**cfg["dp_flow_exponent"]*mu*cfg["dp_factor"]/ns
            k=cfg["k25_LMH_at_10m3h"]*(q/10)**cfg["k_flow_exponent"]*(1/mu)**0.5*cfg["mass_transfer_factor"]
            j,cp,cm=local_flux(c,p-dp/2,A,B,k,T,cfg["permeate_pressure_bar"])
            qp=min(j*da/1000,q*0.25)
            if q-qp < 1e-8: raise ValueError("Vessel dryout")
            salt=qp*cp
            cnext=(q*c-salt)/(q-qp)
            profiles.append({"element":eidx+1,"axial_fraction":(eidx+(sidx+.5)/ns)/n,
                             "feed_m3h":q,"pressure_bar":p-dp/2,"bulk_g_L":c,
                             "surface_g_L":cm,"flux_LMH":j,"permeate_mg_L":cp*1000})
            q-=qp; c=cnext; p-=dp
            total_qp+=qp; total_salt+=salt; eqp+=qp; esalt+=salt
        el.append({"element":eidx+1,"feed_m3h":eq,"permeate_m3h":eqp,
                   "brine_m3h":q,"average_flux_LMH":eqp*1000/cfg["element_area_m2"],
                   "recovery":eqp/eq,"dp_bar":ep-p,"permeate_mg_L":esalt/eqp*1000 if eqp else 0})
    cp=total_salt/total_qp if total_qp else cfeed
    return {"feed_m3h":qfeed,"permeate_m3h":total_qp,"brine_m3h":q,
            "feed_pressure_bar":pin,"brine_pressure_bar":p,"dp_bar":pin-p,
            "permeate_mg_L":cp*1000,"brine_g_L":c,"recovery":total_qp/qfeed,
            "water_residual_m3h":qfeed-q-total_qp,
            "salt_residual_kg_h":qfeed*cfeed-q*c-total_salt,
            "elements":el,"profile":profiles}

@lru_cache(maxsize=1)
def calibration() -> dict:
    c=dict(DEFAULT);c["elements_per_vessel"]=1;c["permeate_pressure_bar"]=0.0
    target=DATASHEET["permeate_m3_day"]/24
    qfeed=target/DATASHEET["recovery"]
    cp=DATASHEET["feed_salinity_g_L"]*(1-DATASHEET["salt_rejection"])*1000
    def residual(logAB):
        A,B=np.exp(logAB)
        v=vessel(55,qfeed,32,25,float(A),float(B),c)
        return [log(v["permeate_m3h"]/target),log(v["permeate_mg_L"]/cp)]
    fit=least_squares(residual,np.log([1.2,0.055]),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    A,B=map(float,np.exp(fit.x)); v=vessel(55,qfeed,32,25,A,B,c)
    return {"A25_LMH_bar":A,"B25_LMH":B,"reference":DATASHEET,
            "calculated_permeate_m3_day":v["permeate_m3h"]*24,
            "calculated_salt_rejection":1-v["permeate_mg_L"]/32000,
            "relative_flow_error":v["permeate_m3h"]/target-1,
            "absolute_rejection_error":1-v["permeate_mg_L"]/32000-0.998,
            "claim":"One-point manufacturer calibration only; not field validation."}

def solve_point(overrides: dict | None = None) -> dict:
    cfg=dict(DEFAULT); cfg.update(overrides or {})
    if not 0 < cfg["recovery"] < .75: raise ValueError("Recovery must lie in (0,0.75)")
    cal=calibration(); A=cal["A25_LMH_bar"];B=cal["B25_LMH"]
    area=cfg["element_area_m2"]*cfg["elements_per_vessel"]
    qp_target=cfg["target_flux_LMH"]*area/1000
    qfeed=qp_target/cfg["recovery"]
    def flow_resid(p):
        return vessel(p,qfeed,cfg["feed_salinity_g_L"],cfg["temperature_C"],A,B,cfg)["permeate_m3h"]-qp_target
    pin=brentq(flow_resid,1.0,140.0,xtol=1e-8)
    v=vessel(pin,qfeed,cfg["feed_salinity_g_L"],cfg["temperature_C"],A,B,cfg)
    nv=cfg["vessels_per_train"]; nt=cfg["trains"]
    Qp=v["permeate_m3h"]*nv; Qf=qfeed*nv; Qb=v["brine_m3h"]*nv
    low=cfg["low_pressure_bar"]
    if cfg["use_erd"]:
        hp_kw=max(pin-low,0)*Qp/(36*cfg["hp_pump_efficiency"])
        recovered_pressure=low+cfg["erd_pressure_efficiency"]*max(v["brine_pressure_bar"]-low,0)
        booster_kw=max(pin-recovered_pressure,0)*Qb/(36*cfg["booster_efficiency"])
    else:
        hp_kw=max(pin-low,0)*Qf/(36*cfg["hp_pump_efficiency"]); booster_kw=0.0
    feed_kw=low*Qf/(36*cfg["feed_pump_efficiency"])
    aux_kw=cfg["auxiliary_kWh_per_m3"]*Qp
    total_kw=hp_kw+booster_kw+feed_kw+aux_kw
    checks={
      "feed_pressure":pin <= cfg["max_feed_pressure_bar"],
      "vessel_pressure_drop":v["dp_bar"] <= cfg["max_vessel_dp_bar"],
      "element_pressure_drop":max(x["dp_bar"] for x in v["elements"]) <= cfg["max_element_dp_bar"],
      "element_average_flux":max(x["average_flux_LMH"] for x in v["elements"]) <= cfg["max_element_average_flux_LMH"],
      "element_recovery":max(x["recovery"] for x in v["elements"]) <= cfg["max_element_recovery"],
      "feed_flow":qfeed <= cfg["max_feed_per_vessel_m3h"],
      "brine_crossflow":v["brine_m3h"] >= cfg["min_brine_per_vessel_m3h"],
      "salt_quality_gate":v["permeate_mg_L"] <= cfg["max_product_salinity_mg_L"],
      "temperature_study_envelope":cfg["min_temperature_C"] <= cfg["temperature_C"] <= cfg["max_temperature_C"],
    }
    return {"config":cfg,"evidence_type":"simulated_industrial_design",
            "train_q_m3_h":Qp,"plant_q_m3_day":Qp*nt*24,
            "train_feed_m3_h":Qf,"train_brine_m3_h":Qb,
            "train_power_kW":total_kw,"plant_power_kW":total_kw*nt,
            "feed_pressure_bar":pin,"vessel_dp_bar":v["dp_bar"],
            "permeate_salinity_mg_L":v["permeate_mg_L"],
            "brine_salinity_g_L":v["brine_g_L"],
            "SEC_kWh_m3":total_kw/Qp,"RO_SEC_kWh_m3":(hp_kw+booster_kw)/Qp,
            "power_components_train_kW":{"high_pressure_pump":hp_kw,"erd_booster":booster_kw,
                  "feed_pump":feed_kw,"assumed_auxiliaries":aux_kw},
            "feasible":all(checks.values()),"checks":checks,
            "violations":[k for k,vv in checks.items() if not vv],
            "water_balance_residual_m3_h":v["water_residual_m3h"]*nv*nt,
            "salt_balance_residual_kg_h":v["salt_residual_kg_h"]*nv*nt,
            "elements":v["elements"],"vessel_profile":v["profile"]}

if __name__ == "__main__":
    import json
    print(json.dumps({"calibration":calibration(),"nominal":solve_point()},indent=2))
