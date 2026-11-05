"""Phase 3b, step 3 - calibration, the gate, as declared (declaration.md, "Step 3").

    python calibrate.py <sea temperature csv> <real fingerprint json> <real level-0 csv> <phase3 folder> <out folder>
    python calibrate.py ... --test        (two settings on seeds 3001-3006: checks the code, touches no declared seed)
    python calibrate.py ... --attempt2    (declaration.md, "Attempt 2": 14 quantities, 300 + 200 settings)

300 settings (Latin hypercube, seed 3101), each run on 6 development plants (seeds 3201-3206; linear, biofilm and asymptotic
fouling in turn) and pooled into one fingerprint. Settings that pass the fingerprint go to the forecast gate. The choice: the
smallest and the largest sigma_Z among the settings that pass both.

How the declared ranges map onto the simulator - details the declaration leaves open, fixed here before the run:
  - every tuned quantity takes one value per setting; the base's per-train jitter is kept where the base has it
  - ΔP sensor noise (0.004-0.04) and the fouling-rate factor (0.3-3) each span a factor of ten and are drawn on a log scale;
    every other range linearly
  - the operator's delay is drawn at each decision from 0 to the setting's value, in whole days
  - the longest time without a cleaning and the shortest time between cleanings: the setting's value, in whole days
  - "base fouling rates x f": the linear law's rate and the biofilm law's rate constant times f, the asymptotic law's time
    constant divided by f
  - the cleaning efficiency is the plant's value; each train is jittered by up to 10 % as in the base
  - the on and off blocks: the means of the base's run lengths of running and stopped days
"""
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from chemai.config import ROSimConfig  # noqa: E402
from chemai.data import ro_sim as B  # noqa: E402
import fingerprint as F  # noqa: E402
import sim_v2 as S  # noqa: E402

N_SETTINGS, LHS_SEED = 300, 3101
DEV_SEEDS = tuple(range(3201, 3207))
TEST_SEEDS = tuple(range(3001, 3007))
CORE = ["noise_pp", "lag1_autocorr_of_daily_change", "mean_abs_change_1d_pp", "mean_abs_change_3d_pp",
        "mean_abs_change_7d_pp", "mean_abs_change_14d_pp"]
OTHER = ["cycle_length_days", "share_of_cycles_over_60_days", "rise_pp_per_week", "cip_drop_pp", "level_at_cleaning_pp",
         "level_p10_pp", "level_p50_pp", "level_p90_pp"]
MISSING_RANGE = (0.08, 0.30)
FEATS_DP = ["dp_pct", "dp_pct_lag1", "dp_pct_lag3", "dp_pct_lag7", "dp_pct_lag14", "dp_pct_slope14", "dp_pct_state",
            "dp_pct_law", "days_since_cip"]
RANGES = {  # name: (low, high, scale)
    "sigma_z": (0.0, 1.5, "lin"), "dp_noise_rel": (0.004, 0.04, "log"), "outlier_probability": (0.0, 0.05, "lin"),
    "operator_level": (1.12, 1.40, "lin"), "operator_delay_max": (0.0, 14.0, "lin"),
    "max_days_without_cip": (40.0, 160.0, "lin"), "min_days_between_cip": (10.0, 30.0, "lin"),
    "rate_factor": (0.3, 3.0, "log"), "cip_efficiency": (0.4, 0.95, "lin"), "on_block_days": (5.0, 40.0, "lin"),
    "off_block_days": (1.0, 10.0, "lin")}
RANGES2 = {**RANGES, "wander_sd": (0.0, 0.05, "lin"), "wander_tau": (2.0, 30.0, "log"),
           "irreversible_share": (0.02, 0.60, "lin")}
ATTEMPT2_LHS_SEED, ATTEMPT2_LOCAL_SEED, N_LOCAL, N_BEST = 3102, 3103, 200, 10


def settings(ranges=RANGES, seed=LHS_SEED) -> pd.DataFrame:
    u = qmc.LatinHypercube(d=len(ranges), seed=seed).random(N_SETTINGS)
    out = {}
    for j, (k, (lo, hi, sc)) in enumerate(ranges.items()):
        out[k] = lo * (hi / lo) ** u[:, j] if sc == "log" else lo + (hi - lo) * u[:, j]
    df = pd.DataFrame(out)
    for k in ("operator_delay_max", "max_days_without_cip", "min_days_between_cip"):
        df[k] = df[k].round().astype(int)
    df.insert(0, "setting", range(N_SETTINGS))
    return df


def local_settings(best: pd.DataFrame, ranges=RANGES2, n=N_LOCAL, seed=ATTEMPT2_LOCAL_SEED) -> pd.DataFrame:
    """n settings around the best ones: every quantity moved uniformly by up to +-10 % of its range (log scale where
    log-scaled), clipped to the range."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        b = best.iloc[i % len(best)]
        r = {}
        for k, (lo, hi, sc) in ranges.items():
            if sc == "log":
                x = np.log(b[k]) + rng.uniform(-0.1, 0.1) * (np.log(hi) - np.log(lo))
                r[k] = float(np.exp(np.clip(x, np.log(lo), np.log(hi))))
            else:
                r[k] = float(np.clip(b[k] + rng.uniform(-0.1, 0.1) * (hi - lo), lo, hi))
        rows.append(r)
    df = pd.DataFrame(rows)
    for k in ("operator_delay_max", "max_days_without_cip", "min_days_between_cip"):
        df[k] = df[k].round().astype(int)
    df.insert(0, "around", [int(best.iloc[i % len(best)]["setting"]) for i in range(n)])
    df.insert(0, "setting", range(N_SETTINGS, N_SETTINGS + n))
    return df


def config_for(row) -> tuple[ROSimConfig, S.V2]:
    base = ROSimConfig(days=S.N_DAYS)
    f = float(row["rate_factor"])
    cfg = replace(base,
                  dp_noise_rel=(float(row["dp_noise_rel"]),) * 2,
                  outlier_probability=float(row["outlier_probability"]),
                  operator_level=(float(row["operator_level"]),) * 2,
                  operator_delay_days=(0, int(row["operator_delay_max"])),
                  max_days_without_cip=(int(row["max_days_without_cip"]), int(row["max_days_without_cip"]) + 1),
                  min_days_between_cip=int(row["min_days_between_cip"]),
                  linear_rate=tuple(x * f for x in base.linear_rate),
                  biofilm_k=tuple(x * f for x in base.biofilm_k),
                  asym_tau_days=tuple(x / f for x in base.asym_tau_days),
                  cip_efficiency=(float(row["cip_efficiency"]),) * 2,
                  on_block_days=float(row["on_block_days"]), off_block_days=float(row["off_block_days"]))
    if "irreversible_share" in row and pd.notna(row["irreversible_share"]):            # attempt 2
        cfg = replace(cfg, irreversible_share=(float(row["irreversible_share"]),) * 2)
        return cfg, S.V2(sigma_z=float(row["sigma_z"]), wander_sd=float(row["wander_sd"]),
                         wander_tau=float(row["wander_tau"]))
    return cfg, S.V2(sigma_z=float(row["sigma_z"]))


def plants(row, sea, seeds):
    cfg, v2 = config_for(row)
    out = []
    for i, seed in enumerate(seeds):
        pp = B.sample_plant(np.random.default_rng(seed), cfg, cfg.laws[i % 3])
        out.append(S.deposit_view(S.simulate(pp, cfg, v2, sea)))
    return out


def pooled_fingerprint(views):
    parts, legs = [], {}
    for k, (v, lg) in enumerate(views):
        parts.append(v[["train", "day_index", "dp_pct"]].assign(train=f"p{k}" + v["train"]))
        legs.update({f"p{k}{t}": x for t, x in lg.items()})
    fp = F.fingerprint(pd.concat(parts, ignore_index=True), legs, boot=False)
    vals = dict(fp["value"]); vals["missing_share"] = fp["missing_share_all_trains"]
    vals["complete_cycles"] = fp["counts"]["complete_cycles"]
    return vals


def judge(vals, real):
    inside = {}
    for k in CORE + OTHER:
        lo, hi = real["interval_90"][k]
        v = vals[k]
        inside[k] = bool(np.isfinite(v) and lo <= v <= hi)
    inside["missing_share"] = bool(MISSING_RANGE[0] <= vals["missing_share"] <= MISSING_RANGE[1])
    core_ok = all(inside[k] for k in CORE)
    other_out = [k for k in OTHER + ["missing_share"] if not inside[k]]
    dist = 0.0
    for k in CORE + OTHER:
        lo, hi = real["interval_90"][k]
        dist += ((vals[k] - real["value"][k]) / ((hi - lo) / 2)) ** 2 if np.isfinite(vals[k]) else 1e6
    dist += ((vals["missing_share"] - 0.19) / 0.11) ** 2
    return core_ok and len(other_out) <= 2, core_ok, other_out, inside, float(dist)


_SEA = None


def _init(sea_path):
    global _SEA
    _SEA = S.load_sea_temperature(sea_path)


def _fp_task(args):
    row, seeds, real = args
    vals = pooled_fingerprint(plants(row, _SEA, seeds))
    ok, core_ok, other_out, inside, dist = judge(vals, real)
    return {**row, **{f"sim {k}": v for k, v in vals.items()}, **{f"in {k}": v for k, v in inside.items()},
            "core_inside": core_ok, "others_outside": ";".join(other_out), "fingerprint_pass": ok, "distance": dist}


def forecast_gate(views, L):
    """Phase 3's rolling origin within each plant, ΔP history only, one seed; errors pooled over the plants."""
    rows, skipped = [], 0
    for k, (v, lg) in enumerate(views):
        d = v.assign(flux_pct=0.0, sp_pct=0.0, ndp_pct=0.0, n_10min_samples=144.0)
        led = pd.DataFrame([{"train": t, "day_index": x} for t, xs in lg.items() for x in xs], columns=["train", "day_index"])
        days = L.build_days(d, led)
        pairs = days[days.valid]
        for h in L.HORIZONS:
            P = pairs[pairs.h == h]
            for b, (start, _) in enumerate(L.BLOCKS):
                train, test = P[P.t + h < start].sort_values(["t", "train"]), P[P.block == b]
                if test.empty or len(train) < 20:
                    skipped += int(not test.empty)
                    continue
                (f,) = L.forecasts(train, [test], "dp_pct", h, FEATS_DP)
                rows.append(test[["train", "t", "dp_pct", "y_dp_pct"]].rename(columns={"dp_pct": "now", "y_dp_pct": "y"})
                            .assign(plant=k, h=h, block=b, **f))
    ro = pd.concat(rows, ignore_index=True)
    res = []
    for h, g in ro.groupby("h"):
        pooled = L.scores(g.y.to_numpy(), g.now.to_numpy(), {m: g[m].to_numpy() for m in L.METHODS})
        blocks = [L.scores(gb.y.to_numpy(), gb.now.to_numpy(), {m: gb[m].to_numpy() for m in L.METHODS})
                  for _, gb in g.groupby("block")]
        for m in L.METHODS[1:]:
            bs = [bl[m]["skill_pct"] for bl in blocks]
            res.append({"h": int(h), "method": m, "skill_pct": pooled[m]["skill_pct"],
                        "beats": bool(L.verdict(pooled[m]["skill_pct"], bs)), "n": int(len(g))})
    return pd.DataFrame(res), skipped


def main():
    sea_path, real_path, real0_path, p3, out = sys.argv[1:6]
    test = "--test" in sys.argv
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, p3)
    import p3lib as L
    L.fit_xgb.__defaults__ = ((0,),)                         # one seed, as declared
    assert len(L.fit_xgb(np.zeros((10, 2)), np.arange(10.0))) == 1
    real = json.loads(Path(real_path).read_text())
    real0 = pd.read_csv(real0_path)
    real_verdict = {(int(r.h), r.method): bool(r["beats no change"]) for _, r in real0[real0.method != "no change"].iterrows()}
    t0 = time.time()
    attempt2 = "--attempt2" in sys.argv
    seeds = TEST_SEEDS if test else DEV_SEEDS
    if attempt2:
        # first: attempt 1's plants must be unchanged by attempt 2's code (setting 0, as saved)
        old = pd.read_csv(out / "calibration_fingerprints.csv").iloc[0]
        _init(sea_path)
        again = _fp_task((settings().iloc[0].to_dict(), DEV_SEEDS, real))
        for k in [c for c in old.index if c.startswith("sim ")]:
            assert np.isclose(old[k], again[k], rtol=1e-9, atol=1e-12, equal_nan=True), k
        print("attempt 1's setting 0 reproduced exactly with attempt 2's code", flush=True)
        st = settings(RANGES2, ATTEMPT2_LHS_SEED)
        if test:
            st = st.iloc[:2]
        with Pool(2, initializer=_init, initargs=(sea_path,)) as pool:
            fps = pd.DataFrame(pool.map(_fp_task, [(r.to_dict(), seeds, real) for _, r in st.iterrows()], chunksize=4))
            print(f"latin hypercube: {len(fps)} settings, {int(fps.fingerprint_pass.sum())} pass [{time.time() - t0:.0f} s]", flush=True)
            loc = local_settings(fps.nsmallest(N_BEST, "distance"), n=4 if test else N_LOCAL)
            fl = pd.DataFrame(pool.map(_fp_task, [(r.to_dict(), seeds, real) for _, r in loc.iterrows()], chunksize=4))
        fps = pd.concat([fps.assign(stage="latin hypercube"), fl.assign(stage="around the best 10")], ignore_index=True)
        tag = "_attempt2" + ("_test" if test else "")
    else:
        st = settings()
        if test:
            st = st.iloc[:2]
        with Pool(2, initializer=_init, initargs=(sea_path,)) as pool:
            fps = pool.map(_fp_task, [(r.to_dict(), seeds, real) for _, r in st.iterrows()], chunksize=4)
        fps = pd.DataFrame(fps)
        tag = "_test" if test else ""
    fps.to_csv(out / f"calibration_fingerprints{tag}.csv", index=False)
    print(f"fingerprints: {len(fps)} settings, {int(fps.fingerprint_pass.sum())} pass, "
          f"{int(fps.core_inside.sum())} with all six core statistics inside [{time.time() - t0:.0f} s]", flush=True)
    sea = S.load_sea_temperature(sea_path)
    gate_rows = []
    cand = fps if test else fps[fps.fingerprint_pass]
    for _, r in cand.sort_values("sigma_z").iterrows():
        views = plants(r, sea, seeds)
        g, skipped = forecast_gate(views, L)
        same = all(real_verdict[(int(x.h), x.method)] == bool(x.beats) for _, x in g.iterrows())
        gate_rows.append(g.assign(setting=int(r.setting), sigma_z=float(r.sigma_z), skipped_blocks=skipped, same_as_real=same))
        print(f"setting {int(r.setting)} (sigma_z {r.sigma_z:.3f}): forecast gate {'pass' if same else 'FAIL'}; "
              f"passing methods {g[g.beats][['h', 'method']].values.tolist() or 'none'} [{time.time() - t0:.0f} s]", flush=True)
    gate = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    gate.to_csv(out / f"calibration_gate{tag}.csv", index=False)
    passed = gate.groupby("setting").same_as_real.first() if len(gate) else pd.Series(dtype=bool)
    both = fps[fps.setting.isin(passed[passed].index)] if len(passed) else fps.iloc[:0]
    chosen = []
    if len(both):
        chosen = sorted({int(both.loc[both.sigma_z.idxmin(), "setting"]), int(both.loc[both.sigma_z.idxmax(), "setting"])})
    res = {"settings": int(len(fps)), "fingerprint_pass": int(fps.fingerprint_pass.sum()),
           "core_inside": int(fps.core_inside.sum()), "forecast_gate_tested": int(passed.size),
           "pass_both": int(len(both)), "chosen_settings": chosen,
           "chosen": both[both.setting.isin(chosen)][["setting"] + list(RANGES2 if attempt2 else RANGES)].to_dict("records"),
           "seeds": list(seeds), "seconds": round(time.time() - t0)}
    (out / f"calibration_result{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
