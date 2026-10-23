"""Step 3 - the simulator, and whether its cycles look like the real ones.

Said openly: the fouling-rate ranges in ROSimConfig were set after looking at
the deposit's slopes (step 1), so that simulated cycles cover what the real
plant shows. This script records that comparison. The deposit's forecast
OUTCOMES were not used for anything until step 6.

    python projects/p03_ro_fouling/step3_simulator.py
"""
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from chemai.config import RODepositConfig, ROSimConfig  # noqa: E402
from chemai.data.ro_deposit import download_deposit, load_daily, split_cycles  # noqa: E402
from chemai.data.ro_sim import generate_plants, sample_plant, simulate_plant  # noqa: E402
from chemai.evaluation.fouling_eval import plant_view, sim_cycles  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
OUT, FIG = HERE / "results", HERE / "figures"


def cycle_stats(cycles, min_points=8, window=(3, 45)):
    """Per cycle: length; and, over the first six weeks (what a forecast sees):
    slope, scatter about that line, and the jump between consecutive days.

    Two scatter measures because each is biased one way: scatter about a line
    grows with curvature, consecutive-day jumps shrink when noise is slow."""
    lengths, slopes, scatter, jumps = [], [], [], []
    for c in cycles:
        lengths.append(c.end - c.start)
        m = (c.day > window[0]) & (c.day <= window[1])
        if m.sum() < min_points:
            continue
        d, x = c.day[m], c.dp_norm[m]
        sl, ic, _, _ = stats.theilslopes(x, d)
        r = x - (ic + sl * d)
        slopes.append(sl * 7)
        scatter.append(1.4826 * np.median(np.abs(r - np.median(r))))
        step = np.diff(d) == 1
        dx = np.diff(x)[step]
        if len(dx) >= 5:
            jumps.append(1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(2))
    return np.array(lengths), np.array(slopes), np.array(scatter), np.array(jumps)


def q(a):
    return [round(float(v), 4) for v in np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])]


def main():
    dcfg = RODepositConfig()
    folder = download_deposit(cfg=dcfg)
    real = [c for c in split_cycles(load_daily(folder / dcfg.daily_file), dcfg) if c.kind == "complete"]
    rl, rs, rn, rj = cycle_stats(real)
    runs = generate_plants(15, seed=3003)
    sim = [c for i, r in enumerate(runs) for c in sim_cycles(r, i) if c.closed]
    sl, ss, sn, sj = cycle_stats(sim)
    res = dict(
        note=("Rate ranges in ROSimConfig were set after looking at the deposit's slopes (step 1). "
              "This comparison was written after the external test (step 6); it uses no forecast "
              "outcome. Lengths, scatter and jumps were NOT matched to the deposit before the "
              "settings were frozen."),
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        real=dict(cycles=len(rl), length_days=q(rl), slope_per_week=q(rs), scatter=q(rn),
                  day_to_day_jump=q(rj)),
        simulated=dict(cycles=len(sl), length_days=q(sl), slope_per_week=q(ss), scatter=q(sn),
                       day_to_day_jump=q(sj)),
        ratio_of_medians_sim_over_real=dict(
            length=round(float(np.median(sl) / np.median(rl)), 2),
            slope=round(float(np.median(ss) / np.median(rs)), 2),
            scatter=round(float(np.median(sn) / np.median(rn)), 2),
            day_to_day_jump=round(float(np.median(sj) / np.median(rj)), 2)),
        real_share_falling=round(float(np.mean(rs < 0)), 2),
        sim_share_falling=round(float(np.mean(ss < 0)), 2))
    print(json.dumps(res, indent=1))
    (OUT / "step3_simulator.json").write_text(json.dumps(res, indent=1))

    # ---- figure: one plant per fouling law
    cfg = ROSimConfig()
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    for ax, law in zip(axes, cfg.laws):
        run = simulate_plant(sample_plant(rng, cfg, law), cfg)
        v = plant_view(run, "fit")
        for tr, col in zip("ABC", ("#2a6fb0", "#128c84", "#a8782f")):
            g = run.daily[(run.daily["train"] == tr) & run.daily["recorded"]]
            ax.plot(g["day"], g["dp_true"], color=col, lw=1.6)
            vv = v[v["train"] == tr]
            ax.plot(vv["day"], vv["dp_norm_raw"], ".", ms=2, color=col, alpha=0.45)
        for _, e in run.cleanings.iterrows():
            ax.axvline(e["day"], color="#b53c58", alpha=0.25, lw=1)
        ax.set_ylabel("DP_norm")
        ax.set_title(f"{law}: operator cleans near {run.params.operator_level:.2f}", fontsize=10, loc="left")
        ax.set_ylim(0.9, 1.7)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("day   (lines: true fouling; dots: what the plant reads; red: cleanings)")
    fig.tight_layout()
    fig.savefig(FIG / "step3_simulated_plants.png", dpi=110)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (r, s_, lab) in zip(axes, ((rl, sl, "cycle length, days"),
                                       (rs, ss, "slope, first 6 weeks, per week"),
                                       (rn, sn, "scatter about the line"),
                                       (rj, sj, "jump between consecutive days"))):
        bins = np.linspace(min(r.min(), np.quantile(s_, 0.02)), max(r.max(), np.quantile(s_, 0.98)), 24)
        ax.hist(s_, bins=bins, density=True, color="#9fc6d6", label=f"simulated ({len(s_)})")
        ax.hist(r, bins=bins, density=True, histtype="step", lw=2, color="#b53c58", label=f"deposit ({len(r)})")
        ax.set_xlabel(lab)
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Simulated cycles against the deposit's complete cycles", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "step3_sim_vs_real.png", dpi=110)


if __name__ == "__main__":
    main()
