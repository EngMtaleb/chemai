"""declaration_v15.md: page 39's schedule library (phase39/sched_lib.py) with the calibrated measurement (screen flow
exponent 1.75, transmitter noise, the slow movement w from its own stream as sim_v2 draws it), the new table, and the
causes the model reads set by the run (every column, or the real record's columns only).

Was: page 39 (note 17): a cleaning schedule from the model vs a fixed calendar (declaration_schedule.md).

simulate_policy() is sim_v2.simulate() (phase 3b, version 2) line for line in its physics, with three changes, all in the
declaration: every measurement draw is made before the loop (so a schedule cannot move the plant's noise), a cleaning's own
draws come from their own stream, and the operator's block is replaced by the schedule being tested. features() computes
the model's inputs from a train's recorded days; the same function serves table.csv (training) and the run (the schedule).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SP / "phase3b"))
sys.path.insert(0, str(SP / "phase3"))
import sim_v2 as S  # noqa: E402
import p3lib as L  # noqa: E402
from chemai.config import ROConfig, ROSimConfig  # noqa: E402
from chemai.data import ro_sim as B  # noqa: E402
from chemai.features.ro_physics import tcf, water_viscosity  # noqa: E402

POC = Path(__file__).resolve().parent / "data"
import json as _json, dataclasses as _dc
_CAL = _json.loads((Path(__file__).resolve().parent / "out" / "calibration15.json").read_text())["chosen"]
SEA = SP / "solar" / "data" / "POWER_Point_Daily_20150101_20241231_029d00N_048d75E_LST.csv"
V = S.V2(sigma_z=0.75, wander_sd=_CAL["w"])
S_COV, R_NOISE = 0.5, 0.5
BASE_DAYS, H = 14, 7
CAUSES = ["temperature_c", "flux_lmh", "sdi", "turbidity_ntu"]
FEATS = ["now", "state", "lag1", "lag3", "lag7", "lag14", "slope14", "days_since_cip"] + \
        [f"{c}{s}" for c in CAUSES for s in ("", "_m7")]
EVAL = [(4201 + i, "linear") for i in range(5)] + [(4211 + i, "biofilm") for i in range(5)] + \
       [(4221 + i, "asymptotic") for i in range(5)]


# ------------------------------------------------------------------ the model's inputs, from a train's recorded days
class Rec:
    """One train's recorded days, growing day by day: day, dp_pct (once the baseline is in), the causes."""

    def __init__(self):
        self.days, self.cols, self.pos = [], {c: [] for c in ["dp_pct"] + CAUSES}, {}

    def add(self, day, **vals):
        self.pos[day] = len(self.days)
        self.days.append(day)
        for c in self.cols:
            self.cols[c].append(vals.get(c, np.nan))

    def near(self, c, day, tol=2):                 # p3lib.near: the day itself, or up to `tol` days earlier
        for k in range(tol + 1):
            j = self.pos.get(day - k)
            if j is not None:
                return self.cols[c][j]
        return np.nan


def features(r: Rec, i: int, last_cip):
    """build_days' inputs for the day at index i: p3lib/poc_lib's definitions, for dp_pct and the four causes."""
    days, dp = r.days, r.cols["dp_pct"]
    t = days[i]
    f = {"now": dp[i]}
    lo = i
    while lo > 0 and days[lo - 1] > (last_cip if last_cip is not None else -1):
        lo -= 1
    recent3 = [dp[j] for j in range(lo, i + 1) if days[j] > t - 7][-3:]
    f["state"] = float(np.median(recent3)) if recent3 else dp[i]
    for lag in (1, 3, 7, 14):
        f[f"lag{lag}"] = r.near("dp_pct", t - lag)
    idx = [j for j in range(lo, i + 1) if days[j] > t - 14]
    f["slope14"] = float(np.polyfit([days[j] for j in idx], [dp[j] for j in idx], 1)[0]) if len(idx) >= 4 else np.nan
    f["days_since_cip"] = float(t - last_cip) if last_cip is not None else np.nan
    wk = [j for j in range(max(0, i - 10), i + 1) if days[j] > t - 7]
    for c in CAUSES:
        f[c] = r.cols[c][i]
        f[f"{c}_m7"] = float(np.mean([r.cols[c][j] for j in wk]))
    return [f[k] for k in FEATS]


def training_pairs():
    """table.csv + cleanings.csv (the proof of concept's record): inputs at t, dp_pct at t + 7 (recorded, no cleaning
    in between), exactly as poc_lib.build_days pairs them."""
    t, c = pd.read_csv(POC / "table.csv"), pd.read_csv(POC / "cleanings.csv")
    X, y, now, meta = [], [], [], []
    for (p, tr), g in t.groupby(["plant", "train"]):
        g = g.sort_values("day")
        cips = np.sort(c[(c.plant == p) & (c.train == tr)].day.values)
        r = Rec()
        for _, row in g.iterrows():
            r.add(int(row.day), **{k: float(row[k]) for k in ["dp_pct"] + CAUSES})
        for i, d in enumerate(r.days):
            past = cips[cips <= d]
            last = int(past.max()) if len(past) else None
            target = r.pos.get(d + H)
            if target is None or ((cips > d) & (cips <= d + H)).any():
                continue
            X.append(features(r, i, last)), y.append(r.cols["dp_pct"][target]), now.append(r.cols["dp_pct"][i])
            meta.append((p, tr, d))
    return np.array(X, float), np.array(y), np.array(now), meta


def fit(X, y, now):
    from xgboost import XGBRegressor
    return [XGBRegressor(random_state=s, **L.XGB).fit(X, y - now).get_booster() for s in L.SEEDS]


def forecast(models, x):
    a = np.asarray([x], float)
    return float(np.clip(x[0] + np.mean([m.inplace_predict(a)[0] for m in models]), 0.0, 1.0))


# ------------------------------------------------------------------ the plant, with a schedule
def simulate_policy(seed: int, law: str, policy: str, sea: np.ndarray, N: int | None = None, models=None,
                    cfg: ROSimConfig | None = None, ro: ROConfig | None = None):
    """One plant under one schedule: 'calendar' (every N days), 'model' (the forecast), or 'operator' (the simulator's
    own rule: dP on screen, the longest gap, other reasons; for reference only)."""
    cfg = cfg or _dc.replace(ROSimConfig(days=S.N_DAYS), dp_noise_rel=(0.5 * _CAL["x"], 1.5 * _CAL["x"]),
                             outlier_probability=_CAL["p"])
    ro = ro or _dc.replace(ROConfig(), vendor_flow_exponent=1.75)
    pp = B.sample_plant(np.random.default_rng(seed), cfg, law)
    v2 = V
    rng = np.random.default_rng(pp.seed)
    rng2 = np.random.default_rng([pp.seed, 2])
    rng_cip = np.random.default_rng([pp.seed, 7])       # a cleaning's own draws (efficiency, legs)
    rng3 = np.random.default_rng([pp.seed, 3])          # sim_v2's third stream: the reading's slow movement
    rng_plan = np.random.default_rng([pp.seed, 8])      # the schedule's draws (the operator's delay and other reasons,
    n_days, n_tr = cfg.days, len(pp.trains)             # the model's planning day)
    ext = n_days + B.HORIZON_PAD
    days_ext = np.arange(ext)

    temp_start = int(rng2.integers(0, len(sea) - ext + 1))
    phase = float(rng2.uniform(0.0, 365.0))
    z = S.driver_signal(rng2, ext, v2.seasonal_share, v2.tau_days, phase)
    twin = S.driver_signal(rng2, n_days, v2.seasonal_share, v2.tau_days, phase + 365.0 / 4.0)
    e1, e2 = rng2.normal(size=n_days), rng2.normal(size=n_days)
    temp_ext = B._temperature(days_ext, pp, rng)
    temp_ext = np.asarray(sea[temp_start:temp_start + ext], float).copy()
    temp = temp_ext[:n_days]
    tds = pp.feed_tds + 0.4 * np.sin(2 * np.pi * (np.arange(n_days) - pp.temp_phase) / 365.0) \
        + rng.normal(0, 0.15, n_days)
    demand = 2.0 * (1 + 0.06 * np.sin(2 * np.pi * (np.arange(n_days) - pp.temp_phase) / 365.0)
                    + rng.normal(0, 0.03, n_days))
    on = B._availability(n_days, n_tr, cfg, rng)
    mu25 = float(water_viscosity(25.0))
    seen_inst = np.sqrt(S_COV) * z[:n_days] + np.sqrt(1.0 - S_COV) * twin
    sdi = 3.0 * np.exp(0.25 * (seen_inst + R_NOISE * e1))
    turb = 0.5 * np.exp(0.4 * (seen_inst + R_NOISE * e2))

    drv = np.exp(v2.sigma_z * z - v2.sigma_z ** 2 / 2.0)
    rate_mult, upset = {}, {}
    for p in pp.trains:
        noise = np.exp(rng.normal(0, cfg.rate_noise, ext) - cfg.rate_noise ** 2 / 2)
        rate_mult[p.name] = p.q10 ** ((temp_ext - 25.0) / 10.0) * noise * drv
        up = np.zeros(ext)
        n_up = rng.poisson(cfg.upsets_per_year * ext / 365.0)
        for d in rng.integers(0, ext, n_up):
            Ln = int(rng.integers(3, 11))
            up[d:d + Ln] += B._u(rng, cfg.upset_size) / Ln
        upset[p.name] = up
    drift = {p.name: np.cumsum(rng.normal(0, cfg.pt_drift_bar_per_day, (2, n_days)), axis=1) for p in pp.trains}
    # every measurement draw, before the loop: one set per train per day, used only on the days the train runs
    M = {p.name: dict(rec=rng.normal(0, 0.005, n_days), outl=rng.random(n_days), dpn=rng.normal(0, 1, n_days),
                      pf=rng.normal(0, 1, n_days), pc=rng.normal(0, 1, n_days), tm=rng.normal(0, 0.2, n_days),
                      ff=rng.normal(0, 0.005, n_days), cf=rng.normal(0, 0.005, n_days), pfl=rng.normal(0, 0.005, n_days),
                      pp_=rng.normal(0, 0.02, n_days), ft=rng.normal(0, 0.01, n_days), pt=rng.normal(0, 0.02, n_days))
         for p in pp.trains}

    wander = {p.name: (v2.wander_sd * S.ar1(rng3, n_days, v2.wander_tau) if v2.wander_sd > 0 else np.zeros(n_days))
              for p in pp.trains}
    st = {p.name: np.array([0.0, 0.0, 0.0]) for p in pp.trains}
    if pp.law == "biofilm":
        for p in pp.trains:
            st[p.name][0] = p.biofilm_seed * 0.3
    pending = {p.name: {} for p in pp.trains}
    last_cip = {p.name: 0 for p in pp.trains}           # the clock from day 0 (version 2)
    cip_days = {p.name: [] for p in pp.trains}
    recs = {p.name: Rec() for p in pp.trains}
    dpref = {p.name: [] for p in pp.trains}
    qref = {p.name: None for p in pp.trains}
    base = {p.name: None for p in pp.trains}
    gbase = {p.name: [] for p in pp.trains}
    rows, events, n_pred = [], [], 0

    for d in range(n_days):
        n_on = int(on[:, d].sum())
        share = np.clip(demand[d] / max(n_on, 1), 0.75, 1.10)
        for k, p in enumerate(pp.trains):
            cip_today = d in pending[p.name]
            fac = float(share) if (on[k, d] and not cip_today) else 0.0
            s = B.step_state(st[p.name], p, float(rate_mult[p.name][d]) * fac, float(upset[p.name][d]))
            if cip_today:
                eff = p.cip_efficiency * (rng_cip.uniform(0.85, 1.0))
                s = B.clean(s, p, eff)
                legs = 2 if rng_cip.random() < cfg.two_leg_probability else 1
                if legs == 2:
                    s = B.clean(s, p, eff * 0.5)
                events.append(dict(train=p.name, day=d, legs=legs, reason=pending[p.name].pop(d)))
                last_cip[p.name] = d
                cip_days[p.name].append(d)
            st[p.name] = s
            phi = float(s.sum())
            g = float(B.gap_law(phi, p))
            running = bool(on[k, d]) and not cip_today
            if not running:
                rows.append(dict(train=p.name, day=d, running=False, g=g, cip=cip_today))
                continue
            m = M[p.name]
            q_p = p.q_perm_design * share
            rec = float(np.clip(p.recovery + m["rec"][d], 0.3, 0.6))
            q_f = q_p / rec
            q_c = q_f - q_p
            q_avg = (q_f + q_c) / 2.0
            q_avg_design = (p.q_perm_design / p.recovery + p.q_perm_design / p.recovery - p.q_perm_design) / 2.0
            T = float(temp[d])
            mu = float(water_viscosity(T))
            dp = p.dp_clean * (q_avg / q_avg_design) ** (2 - p.friction_m) * (mu / mu25) ** p.friction_m * g
            sc = cfg.outlier_scale if m["outl"][d] < cfg.outlier_probability else 1.0
            dp_meas = dp * (1 + m["dpn"][d] * p.dp_noise_rel * sc) * (1.0 + float(wander[p.name][d]))
            t_meas = T + m["tm"][d]
            ff, cfl, pfl = q_f * (1 + m["ff"][d]), q_c * (1 + m["cf"][d]), q_p * (1 + m["pfl"][d])
            recorded = n_on >= 2
            rows.append(dict(train=p.name, day=d, running=True, recorded=recorded, g=g, cip=False))
            if recorded:
                q_avg_m = (ff + cfl) / 2.0
                if qref[p.name] is None:
                    qref[p.name] = q_avg_m
                ref = dp_meas * (qref[p.name] / q_avg_m) ** ro.vendor_flow_exponent \
                    * (mu25 / float(water_viscosity(t_meas))) ** ro.dp_visc_exponent
                dpref[p.name].append(ref)
                if len(gbase[p.name]) < BASE_DAYS:
                    gbase[p.name].append(g)
                recs[p.name].add(d, temperature_c=t_meas, flux_lmh=pfl * 1000.0 / p.area_m2, sdi=float(sdi[d]),
                                 turbidity_ntu=float(turb[d]))
                r = recs[p.name]
                if base[p.name] is None and len(dpref[p.name]) >= BASE_DAYS:
                    base[p.name] = float(np.median(dpref[p.name][:BASE_DAYS]))
                    r.cols["dp_pct"] = [float(np.clip(v / base[p.name] - 1.0, 0.0, 1.0)) for v in dpref[p.name]]
                elif base[p.name] is not None:
                    r.cols["dp_pct"][-1] = float(np.clip(ref / base[p.name] - 1.0, 0.0, 1.0))
            if base[p.name] is None:
                continue
            since = d - last_cip[p.name]
            if pending[p.name] or since < cfg.min_days_between_cip:
                continue
            if policy == "calendar":
                if since >= N:
                    pending[p.name] = {d + 1: "calendar"}
            elif policy == "model":
                if recorded:
                    r = recs[p.name]
                    i = len(r.days) - 1
                    last = cip_days[p.name][-1] if cip_days[p.name] else None
                    fc = forecast(models, features(r, i, last))
                    n_pred += 1
                    if fc >= pp.operator_level - 1.0:
                        pending[p.name] = {d + 1 + int(rng_plan.integers(0, 7)): "model forecast"}
            elif policy == "operator":
                r = recs[p.name]                        # the screen since the last cleaning: its last 7 readings
                cyc = [v for dd, v in zip(r.days[-7:], r.cols["dp_pct"][-7:]) if dd > last_cip[p.name]]
                look = float(np.median(cyc)) + 1.0 if len(cyc) >= 3 else 0.0
                if look >= pp.operator_level:
                    delay = int(rng_plan.integers(pp.operator_delay[0], pp.operator_delay[1] + 1))
                    pending[p.name] = {d + delay + 1: "dP on screen"}
                elif since >= pp.max_days_without_cip:
                    pending[p.name] = {d + 1: "time since last cleaning"}
                elif rng_plan.random() < cfg.other_reason_cip_per_year / 365.0:
                    pending[p.name] = {d + 1: "other reason"}
            else:
                raise ValueError(policy)

    daily = pd.DataFrame(rows)
    gb = {tr: float(np.median(v)) for tr, v in gbase.items()}
    daily["ratio"] = daily["g"] / daily["train"].map(gb)
    return dict(seed=seed, law=law, policy=policy, N=N, level=pp.operator_level, daily=daily,
                cleanings=pd.DataFrame(events, columns=["train", "day", "legs", "reason"]), n_pred=n_pred,
                trains=[p.name for p in pp.trains])


def score(run):
    """Per train: cleanings, running days at or above the limit (true ratio), mean true rise on running days."""
    d = run["daily"]
    out = []
    for tr in run["trains"]:
        g = d[(d.train == tr) & d.running]
        out.append(dict(seed=run["seed"], law=run["law"], policy=run["policy"], N=run["N"], train=tr,
                        level=run["level"], cleanings=int((run["cleanings"].train == tr).sum()),
                        days_above=int((g.ratio >= run["level"]).sum()), running_days=int(len(g)),
                        mean_rise=float((g.ratio - 1.0).mean())))
    return out
