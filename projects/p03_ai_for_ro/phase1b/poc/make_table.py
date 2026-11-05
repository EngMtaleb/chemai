"""Phase 1B proof of concept - the simulated plant record (declaration_poc.md, approved 3 Nov 2026 22:40 UTC+3).

    python3 -B make_table.py <NASA POWER daily csv> <out folder>

Three plants, one per fouling law (seeds 4101-4103), from simulator version 2 (sim_v2.py, unchanged) with sigma_z = 0.75,
the instruments at s = 0.5, r = 0.5, attempt 2's slow movement off. One row per train per recorded day: every logged
column, with noise; the four indicators built from measured values only. The no-link columns come from a fourth random
stream, default_rng([seed, 4]), so the plant itself is draw for draw what version 2 makes without them.
Writes table.csv (the record), cleanings.csv (the cleaning log), hidden.csv (truths for checks only: never a model input),
plantwide.csv (SDI, turbidity and the no-link columns on every day, for the control's shift) and table.sha256.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sim_v2 as S  # noqa: E402
from chemai.config import ROConfig, ROSimConfig  # noqa: E402
from chemai.data import ro_sim as B  # noqa: E402
from chemai.features.ro_physics import (log_mean_concentration_factor, osmotic_pressure, tcf,  # noqa: E402
                                        water_viscosity)

PLANTS = [(4101, "linear"), (4102, "biofilm"), (4103, "asymptotic")]
V = S.V2(sigma_z=0.75, wander_sd=0.0)          # switches 1-3 and the clock from day 0 on (their defaults)
S_COV, R_NOISE = 0.5, 0.5
BASE_DAYS = 14

CAUSES = ["temperature_c", "flux_lmh", "sdi", "turbidity_ntu"]
OPERATION = ["feed_flow_m3h", "perm_flow_m3h", "conc_flow_m3h", "recovery", "feed_tds_mgl", "perm_pressure_bar"]
RESULTS_RAW = ["feed_pressure_bar", "conc_pressure_bar", "dp_bar", "perm_tds_mgl"]
INDICATORS = ["dp_pct", "flux_pct", "sp_pct", "ndp_pct"]
NO_LINK = ["ph", "orp_mv", "free_chlorine_mgl", "sbs_mgl", "antiscalant_mgl", "coagulant_mgl_fe", "cf_dp_bar"]
GROUPS = {"causes": CAUSES, "operation": OPERATION, "results": RESULTS_RAW + INDICATORS, "no link": NO_LINK}


def no_link(seed: int, n: int) -> pd.DataFrame:
    """The columns that play no part in this simulator: levels ASSUMED, anchored as declared. One value per day (one intake)."""
    rng = np.random.default_rng([seed, 4])
    ph = 8.1 + 0.1 * S.ar1(rng, n, 7.0) + rng.normal(0, 0.02, n)
    orp = 180.0 + 20.0 * S.ar1(rng, n, 7.0) + rng.normal(0, 5.0, n)
    cl = np.maximum(0.0, 0.02 + 0.01 * S.ar1(rng, n, 7.0) + rng.normal(0, 0.005, n))

    def setpoint(lo, hi):
        v, d = np.empty(n), 0
        while d < n:
            span = int(rng.integers(30, 91))
            v[d:d + span] = rng.uniform(lo, hi)
            d += span
        return v * (1.0 + rng.normal(0, 0.03, n))

    sbs, asc, coag = setpoint(2.0, 4.0), setpoint(1.5, 3.0), setpoint(1.0, 4.0)
    cf, d = np.empty(n), 0
    while d < n:                                   # 0.2 bar after a change, 0.7 bar (10 psi) at the change-out
        span = int(rng.integers(30, 91))
        k = np.arange(min(span, n - d))
        cf[d:d + len(k)] = 0.2 + 0.5 * k / span
        d += span
    cf = cf + rng.normal(0, 0.01, n)
    return pd.DataFrame({"day": np.arange(n), "ph": ph, "orp_mv": orp, "free_chlorine_mgl": cl, "sbs_mgl": sbs,
                         "antiscalant_mgl": asc, "coagulant_mgl_fe": coag, "cf_dp_bar": cf})


def indicators(g: pd.DataFrame, area: float, ro: ROConfig) -> pd.DataFrame:
    """The deposit's kind of indicators, from measured values only: rises over the median of the first 14 recorded days,
    never reset at a cleaning, clipped 0-1."""
    T = g["temperature_c"].to_numpy(float)
    qf, qc, qp = (g[c].to_numpy(float) for c in ("feed_flow", "conc_flow", "perm_flow"))
    rec = qp / qf
    flux = qp * 1000.0 / area
    q_avg = (qf + qc) / 2.0
    mu25 = float(water_viscosity(25.0))
    dp = g["dp"].to_numpy(float)
    dp_ref = dp * (q_avg[0] / q_avg) ** ro.vendor_flow_exponent * (mu25 / water_viscosity(T)) ** ro.dp_visc_exponent
    c_fc = g["feed_tds"].to_numpy(float) * log_mean_concentration_factor(rec)        # g/L, feed-concentrate mean
    c_p = g["perm_tds"].to_numpy(float)
    ndp = g["feed_pressure"].to_numpy(float) - dp / 2.0 - g["perm_pressure"].to_numpy(float) \
        - (osmotic_pressure(c_fc, T) - osmotic_pressure(c_p, T))
    a25 = flux / (ndp * tcf(T, ro.tcf_e_water))                                       # 25 C specific flux
    sp25 = (c_p / c_fc) * flux / tcf(T, ro.tcf_e_salt)                                 # salt passage at 25 C, per flux

    def rise(x):
        return np.clip(x / np.median(x[:BASE_DAYS]) - 1.0, 0.0, 1.0)

    return pd.DataFrame({"dp_pct": rise(dp_ref), "flux_pct": np.clip(1.0 - a25 / np.median(a25[:BASE_DAYS]), 0.0, 1.0),
                         "sp_pct": rise(sp25), "ndp_pct": rise(ndp), "recovery": rec, "flux_lmh": flux})


def one_plant(i: int, seed: int, law: str, sea: np.ndarray, cfg: ROSimConfig, ro: ROConfig):
    pp = B.sample_plant(np.random.default_rng(seed), cfg, law)
    rv = S.simulate(pp, cfg, V, sea_t=sea, ro=ro)
    d = rv.run.daily
    rec = d[d["recorded"]].sort_values(["train", "day"]).reset_index(drop=True)
    inst = S.logged(rv, S_COV, R_NOISE).rename(columns={"day_index": "day"})
    nl = no_link(seed, cfg.days)
    area = {p.name: p.area_m2 for p in pp.trains}
    parts = []
    for tr, g in rec.groupby("train", sort=True):
        g = g.reset_index(drop=True)
        ind = indicators(g, area[tr], ro)
        parts.append(pd.DataFrame({
            "plant": f"P{i + 1}", "law": law, "train": tr, "day": g["day"].astype(int),
            "temperature_c": g["temperature_c"], "flux_lmh": ind["flux_lmh"],
            "feed_flow_m3h": g["feed_flow"], "perm_flow_m3h": g["perm_flow"], "conc_flow_m3h": g["conc_flow"],
            "recovery": ind["recovery"], "feed_tds_mgl": 1000.0 * g["feed_tds"], "perm_pressure_bar": g["perm_pressure"],
            "feed_pressure_bar": g["feed_pressure"], "conc_pressure_bar": g["conc_pressure"], "dp_bar": g["dp"],
            "perm_tds_mgl": 1000.0 * g["perm_tds"],
            **{c: ind[c] for c in INDICATORS}}))
    tab = pd.concat(parts, ignore_index=True).merge(inst, on="day", how="left").merge(nl, on="day", how="left")
    tab = tab.rename(columns={"sdi": "sdi", "turbidity_ntu": "turbidity_ntu"})
    cols = ["plant", "law", "train", "day"] + CAUSES + OPERATION + RESULTS_RAW + INDICATORS + NO_LINK
    tab = tab[cols]
    cl = rv.run.cleanings.assign(plant=f"P{i + 1}")[["plant", "train", "day", "legs", "reason"]]
    hid = d[d["recorded"]][["train", "day", "phi", "true_dp", "true_temperature", "dp_true"]].assign(plant=f"P{i + 1}")
    zz = pd.DataFrame({"day": np.arange(cfg.days), "z": rv.z[:cfg.days]})
    hid = hid.merge(zz, on="day", how="left")
    info = {"plant": f"P{i + 1}", "seed": seed, "law": law, "temp_window_start": rv.temp_start,
            "operator_level": pp.operator_level, "max_days_without_cip": pp.max_days_without_cip,
            "rows": len(tab), "cleanings": len(cl), "recorded_share": {tr: round(float(n) / cfg.days, 3) for tr, n
                                                                          in tab.groupby("train").size().items()}}
    pw = inst.merge(nl, on="day").assign(plant=f"P{i + 1}")     # the plant-wide daily series, every day (the control)
    return tab, cl, hid, info, pw


def main():
    sea_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sea = S.load_sea_temperature(sea_path)
    cfg, ro = ROSimConfig(days=S.N_DAYS), ROConfig()
    tabs, cls, hids, infos, pws = [], [], [], [], []
    for i, (seed, law) in enumerate(PLANTS):
        t, c, h, info, pw = one_plant(i, seed, law, sea, cfg, ro)
        tabs.append(t), cls.append(c), hids.append(h), infos.append(info), pws.append(pw)
        print(json.dumps(info))
    tab = pd.concat(tabs, ignore_index=True)
    assert not tab.isna().any().any(), tab.columns[tab.isna().any()].tolist()
    tab.to_csv(out / "table.csv", index=False, float_format="%.6g")
    pd.concat(cls, ignore_index=True).to_csv(out / "cleanings.csv", index=False)
    pd.concat(hids, ignore_index=True).to_csv(out / "hidden.csv", index=False, float_format="%.6g")
    pd.concat(pws, ignore_index=True).to_csv(out / "plantwide.csv", index=False, float_format="%.6g")
    (out / "plants.json").write_text(json.dumps({"plants": infos, "groups": GROUPS,
                                                  "sea_file": sea_path.name,
                                                  "sea_sha256": hashlib.sha256(sea_path.read_bytes()).hexdigest()},
                                                 indent=1))
    lines = [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in ("table.csv", "cleanings.csv")]
    (out / "table.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("rows", len(tab), "columns", len(tab.columns))


if __name__ == "__main__":
    main()
