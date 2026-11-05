"""Phase 3b, step 2 - does simulator version 2 do what the declaration says? Checked before any calibration.

    python check_sim_v2.py <sea temperature csv> <real fingerprint json> <out folder>

1. switches off = the base simulator, draw for draw        5. the instruments see sqrt(s) of the driver, less with noise r
2. the same seed gives the same plant                      6. each day's fouling growth follows the declared rate formula
3. temperature = the real series window                    7. the indicator: 0-1, never reset, close to the true dP
4. the driver: unit variance, 30 % yearly, 7-day memory    8. s and r change only the logged columns
Then one plant drawn, and the uncalibrated fingerprint (seeds 3001-3006, outside the declared development and test seeds).
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from chemai.config import ROSimConfig  # noqa: E402
from chemai.data import ro_sim as B  # noqa: E402
import fingerprint as F  # noqa: E402
import sim_v2 as S  # noqa: E402

t0 = time.time()
sea = S.load_sea_temperature(sys.argv[1])
real = json.loads(Path(sys.argv[2]).read_text())
OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
res = {}
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731

# 1 -------------------------------------------------------------- switches off = the base
cfg0 = ROSimConfig()
rng = np.random.default_rng(11)
for law in cfg0.laws:
    pp = B.sample_plant(rng, cfg0, law)
    a = B.simulate_plant(pp, cfg0)
    b = S.simulate(pp, cfg0, S.V2(real_temperature=False, driver=False, flux_factor=False, clock_from_start=False), sea).run
    pd.testing.assert_frame_equal(a.daily, b.daily, check_exact=True)
    pd.testing.assert_frame_equal(a.cleanings, b.cleanings, check_exact=True)
    for t in a.rate_mult:
        assert np.array_equal(a.rate_mult[t], b.rate_mult[t]) and np.array_equal(a.state[t], b.state[t])
res["1 switches off = base, draw for draw"] = "pass (3 plants, one per fouling law, every column and every state)"

# 2 -------------------------------------------------------------- repeatable
cfg = ROSimConfig(days=S.N_DAYS)
pp = B.sample_plant(np.random.default_rng(3001), cfg, "linear")
r1, r2 = S.simulate(pp, cfg, S.V2(), sea), S.simulate(pp, cfg, S.V2(), sea)
pd.testing.assert_frame_equal(r1.run.daily, r2.run.daily, check_exact=True)
res["2 repeatable"] = "pass"

# 3 -------------------------------------------------------------- temperature
dd = r1.run.daily[r1.run.daily["running"]]
win = sea[r1.temp_start:r1.temp_start + S.N_DAYS]
assert np.allclose(dd["true_temperature"].to_numpy(), win[dd["day"].to_numpy()])
res["3 temperature = series window"] = {"window_start_in_series": r1.temp_start,
                                        "range_C": [round(float(win.min()), 2), round(float(win.max()), 2)],
                                        "sensor_noise_sd_C": round(float((dd["temperature_c"] - dd["true_temperature"]).std()), 3)}

# 4 -------------------------------------------------------------- the driver
n = 365 * 400
z = S.driver_signal(np.random.default_rng(13), n, 0.3, 7.0, phase=100.0)
tt = np.arange(n)
X = np.column_stack([np.sin(2 * np.pi * tt / 365), np.cos(2 * np.pi * tt / 365), np.ones(n)])
fit = X @ np.linalg.lstsq(X, z, rcond=None)[0]
rest = z - fit
res["4 driver (400 simulated years)"] = {"variance": round(float(z.var()), 3), "yearly_share": round(float(fit.var() / z.var()), 3),
                                          "lag1_of_the_rest": round(corr(rest[:-1], rest[1:]), 4),
                                          "declared_lag1": round(float(np.exp(-1 / 7)), 4)}

# 5 -------------------------------------------------------------- the instruments
g = np.random.default_rng(14)
fake = S.RunV2(run=None, z=z, twin=S.driver_signal(g, n, 0.3, 7.0, phase=100.0 + 365.0 / 4.0), e1=g.normal(size=n),
              e2=g.normal(size=n), temp_start=None)
rows = []
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    for r in (0.0, 0.5, 1.0):
        lg = S.logged(fake, s, r)
        x1 = np.log(lg["sdi"].to_numpy() / 3.0) / 0.25            # back to the standard scale (the scale is cosmetic)
        x2 = np.log(lg["turbidity_ntu"].to_numpy() / 0.5) / 0.4
        rows.append({"s": s, "r": r, "corr_SDI_Z": round(corr(x1, z), 3), "corr_turbidity_Z": round(corr(x2, z), 3),
                     "declared": round(float(np.sqrt(s) / np.sqrt(1 + r * r)), 3)})
inst = pd.DataFrame(rows)
print(inst.to_string(index=False))
assert (inst[["corr_SDI_Z", "corr_turbidity_Z"]].sub(inst["declared"], axis=0).abs() < 0.02).all().all()
res["5 instruments"] = "pass: correlation with Z = sqrt(s)/sqrt(1+r^2) within 0.02 at all 15 grid points"

# 6 -------------------------------------------------------------- the declared rate formula
cfg6 = ROSimConfig(days=S.N_DAYS, rate_noise=0.0, upsets_per_year=0.0)
pp6 = B.sample_plant(np.random.default_rng(3002), cfg6, "linear")
rv6 = S.simulate(pp6, cfg6, S.V2(sigma_z=0.7), sea)
T6 = sea[rv6.temp_start:rv6.temp_start + S.N_DAYS]
worst, checked = 0.0, 0
for p in pp6.trains:
    st = rv6.run.state[p.name]
    cl = set(rv6.run.cleanings.loc[rv6.run.cleanings["train"] == p.name, "day"])
    expect = p.linear_rate * p.q10 ** ((T6 - 25) / 10) * np.exp(0.7 * rv6.z[:S.N_DAYS] - 0.7 ** 2 / 2) * rv6.flux_fac[p.name]
    for d in range(1, S.N_DAYS):
        if d in cl or st[d].sum() >= B.PHI_MAX - 1e-12:
            continue
        worst = max(worst, abs((st[d, 0] - st[d - 1, 0]) - expect[d]))
        checked += 1
assert worst < 1e-12
res["6 rate formula"] = f"pass: {checked} train-days, largest difference {worst:.1e} (linear law, no daily noise, no upsets)"

# 7 -------------------------------------------------------------- the indicator
view, legs = S.deposit_view(r1)
assert view["dp_pct"].between(0, 1).all()
m = view["dp_true_pct"].between(0, 1)
res["7 indicator"] = {"rows": int(len(view)), "within_0_1": True,
                      "corr_with_true_dP": round(corr(view.loc[m, "dp_pct"], view.loc[m, "dp_true_pct"]), 3),
                      "cleaning_legs": {k: len(v) for k, v in legs.items()}}

# 7b ------------------------------------------------------------- the first cleaning no longer comes right after the baseline
first = []
for i, seed in enumerate(range(3001, 3031)):
    p_ = B.sample_plant(np.random.default_rng(seed), cfg, cfg.laws[i % 3])
    c_ = S.simulate(p_, cfg, S.V2(), sea).run.cleanings
    first += [(g["day"].min(), g.sort_values("day")["reason"].iloc[0]) for _, g in c_.groupby("train")]
first = pd.DataFrame(first, columns=["day", "reason"])
res["7b first cleaning per train (30 plants)"] = {"median_day": float(first["day"].median()),
                                                  "share_before_day_30": round(float((first["day"] < 30).mean()), 2),
                                                  "reasons": first["reason"].value_counts().to_dict()}

# 8 -------------------------------------------------------------- s and r change only the logged columns
v0, _ = S.deposit_view(r1)
for s in (0.0, 1.0):
    for r in (0.0, 1.0):
        S.logged(r1, s, r)
pd.testing.assert_frame_equal(v0, S.deposit_view(r1)[0], check_exact=True)
res["8 the plant does not depend on s, r"] = "pass"

# 9 -------------------------------------------------------------- attempt 2: the reading's slow movement
w = S.ar1(np.random.default_rng(15), 365 * 400, 7.0)
res["9 wander (400 simulated years, unit sd)"] = {"sd": round(float(w.std()), 3), "lag1": round(corr(w[:-1], w[1:]), 4),
                                                 "declared_lag1": round(float(np.exp(-1 / 7)), 4)}
rw = S.simulate(pp, cfg, S.V2(wander_sd=0.03, wander_tau=10.0), sea)
assert np.allclose(rw.wander["A"].std(), 0.03, rtol=0.35)
r0 = S.simulate(pp, cfg, S.V2(), sea)
pd.testing.assert_frame_equal(r0.run.daily, r1.run.daily, check_exact=True)
res["9 wander off = attempt 1's plant"] = "pass"

# ---------------------------------------------------------------- one plant, drawn
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.6]})
fig.patch.set_facecolor(SURF)
for a in ax:
    a.set_facecolor(SURF); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
days = np.arange(S.N_DAYS)
ax[0].plot(days, sea[r1.temp_start:r1.temp_start + S.N_DAYS], color=ORANGE, lw=1.2)
ax[0].set_ylabel("feed temperature, °C")
lg = S.logged(r1, 1.0, 0.5)
ax[1].plot(days, np.log(lg["sdi"] / 3.0) / 0.25, ".", color=MUTED, ms=2.5, label="SDI reading, s = 1, r = 0.5 (standard scale)")
ax[1].plot(days, r1.z[:S.N_DAYS], color=BLUE, lw=1.4, label="fouling driver Z (not logged)")
ax[1].set_ylabel("standard units"); ax[1].legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
for tr, col in zip("ABC", (BLUE, AQUA, ORANGE)):
    v = view[view["train"] == tr]
    ax[2].plot(v["day_index"], 100 * v["dp_pct"], ".", ms=2.5, color=col, label=f"train {tr}")
    for c in legs[tr]:
        ax[2].axvline(c, color=col, lw=0.8, alpha=0.5)
ax[2].set_ylabel("ΔP rise, %"); ax[2].set_xlabel("day"); ax[2].legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
ax[0].set_title("SIMULATION · simulator version 2 before calibration (σ_Z = 0.5, a placeholder) · one plant, linear fouling law",
                loc="left", color=INK, fontsize=11)
plt.tight_layout(); fig.savefig(OUT / "sim_v2_one_plant.png", dpi=110)

# ---------------------------------------------------------------- the uncalibrated fingerprint
parts, all_legs = [], {}
for i, seed in enumerate(range(3001, 3007)):
    p_ = B.sample_plant(np.random.default_rng(seed), cfg, cfg.laws[i % 3])
    v, lg_ = S.deposit_view(S.simulate(p_, cfg, S.V2(), sea))
    v["train"] = f"p{seed}" + v["train"]
    parts.append(v[["train", "day_index", "dp_pct"]])
    all_legs.update({f"p{seed}{k}": x for k, x in lg_.items()})
fp = F.fingerprint(pd.concat(parts, ignore_index=True), all_legs, boot=False)
rows = []
for k, v in real["value"].items():
    lo, hi = real["interval_90"][k]
    sv = fp["value"][k]
    rows.append({"statistic": k, "real": round(v, 3), "real 90 % interval": f"{lo:.3f} to {hi:.3f}",
                 "simulated (uncalibrated)": round(sv, 3), "inside": bool(lo <= sv <= hi)})
rows.append({"statistic": "missing_share", "real": "A 0.30, B 0.08, C 0.09", "real 90 % interval": "0.08 to 0.30 (declared)",
             "simulated (uncalibrated)": round(fp["missing_share_all_trains"], 3),
             "inside": bool(0.08 <= fp["missing_share_all_trains"] <= 0.30)})
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "fingerprint_uncalibrated.csv", index=False)
res["uncalibrated fingerprint (6 plants, 18 trains)"] = {"inside": int(tab["inside"].sum()), "of": int(len(tab)),
                                                         "cycles": fp["counts"]}
res["seconds"] = round(time.time() - t0)
(OUT / "check_sim_v2.json").write_text(json.dumps(res, indent=1, default=str))
pd.set_option("display.width", 200)
print(json.dumps(res, indent=1, default=str))
print(inst.to_string(index=False))
print(tab.to_string(index=False))
