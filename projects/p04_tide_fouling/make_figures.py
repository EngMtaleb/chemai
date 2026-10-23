"""Figures for Project 4, regenerated from results/ (and the record for fig2).

    python projects/p04_tide_fouling/make_figures.py [names...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GREEN, RED = "#2f6fb2", "#d9822b", "#3f8f5a", "#c23b3b"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_record():
    runs = pd.read_parquet(RES / "runs.parquet")
    fig, ax = plt.subplots(figsize=(9, 2.6))
    colors = {"feed": GREEN, "filtrate": BLUE, "none": MUTED}
    for r in runs.itertuples():
        y = 1 if r.split == "training" else 0
        ax.barh(y, (r.t_end - r.t_start).total_seconds() / 86400, left=r.t_start, height=0.6,
                color=colors[r.turbidity_reads], edgecolor="none")
    ax.set_yticks([0, 1], ["test/", "training/"])
    ax.set_title("107 runs with a trusted clock, 5,106 h  ·  colour = what the turbidity sensor read",
                 loc="left", fontsize=10, color=INK)
    for k, c in colors.items():
        ax.barh(np.nan, 0, color=c, label={"feed": "feed", "filtrate": "filtrate",
                                          "none": "no signal"}[k])
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0, -0.18))
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig1_record.png")


def fig_segments():
    from chemai.data.uf_pilot import read_run, run_root
    from chemai.features.uf_segments import cut_segments, resistance_index
    df = read_run(run_root() / "training" / "20150303.csv")
    df = df[df.index < df.index[0] + pd.Timedelta(hours=3.2)]
    lnr = np.log(resistance_index(df))
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(df.index, lnr, color=MUTED, lw=0.6)
    for a, b in cut_segments(df):
        seg = lnr.iloc[a:b + 1].resample("60s").median().dropna()
        x = (seg.index - seg.index[0]).total_seconds() / 3600
        k, c = np.polyfit(x, seg.to_numpy(), 1)
        ax.plot(seg.index, c + k * x, color=ORANGE, lw=2)
    ax.set_ylabel("ln R (relative)")
    ax.set_title("Each pair of elements filters ~24 min; the orange slope is that segment's fouling rate",
                 loc="left", fontsize=10, color=INK)
    fig.savefig(FIG / "fig2_segments.png")


def fig_power():
    p = json.loads((RES / "step3_power.json").read_text())
    names = {"fouling_pct_h": "fouling rate (primary)", "turbidity": "feed turbidity",
             "pump_speed": "pump speed", "p_mf_in": "MF inlet pressure"}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for (k, v), c in zip(p.items(), [ORANGE, GREEN, BLUE, MUTED]):
        x = [float(s) * 100 for s in v["detection_rate_at_p_le_0.01"]]
        y = [r * 100 for r in v["detection_rate_at_p_le_0.01"].values()]
        ax.plot(x, y, "o-", color=c, label=names.get(k, k))
    ax.axhline(80, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("tide wave, 5-95 % range as % of the median")
    ax.set_ylabel("found at p ≤ 0.01 (%)")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "fig3_power.png")


def fig_test():
    """Observed statistic against its week-shift null, for every declared test."""
    from chemai.evaluation.tide_test import (CONSTITUENTS, PLACEBO, TideTestConfig, block_shifts,
                                             prepare, rotation_stack, tide_design, _fit)
    seg = pd.read_parquet(RES / "segments.parquet")
    cfg = TideTestConfig()
    jobs = [("fouling rate (H1)", seg, "fouling_pct_h", CONSTITUENTS),
            ("placebo periods", seg, "fouling_pct_h", PLACEBO),
            ("pump speed (C1)", seg, "pump_speed", CONSTITUENTS),
            ("pump discharge pressure (C2)", seg, "p_mf_in", CONSTITUENTS),
            ("feed turbidity (S1)", seg[seg.turbidity_reads == "feed"], "turbidity", CONSTITUENTS)]
    fig, axes = plt.subplots(1, len(jobs), figsize=(12, 2.6), sharey=False)
    for ax, (title, data, col, periods) in zip(axes, jobs):
        prep = prepare(data, col, cfg)
        d = tide_design(prep, periods)
        zy = d.zcat.T @ prep.y_res
        k = len(d.blocks)
        tot = float(prep.y_res @ prep.y_res)
        obs = _fit(d, zy, rotation_stack(d, np.zeros(k)))[0] / tot
        rng = np.random.default_rng(cfg.seed)
        null = np.array([_fit(d, zy, rotation_stack(d, block_shifts(k, rng, cfg.shift_weeks)))[0] / tot
                         for _ in range(cfg.n_surrogates)])
        p = (1 + np.sum(null >= obs)) / (1 + len(null))
        ax.hist(null * 100, bins=40, color=GRID, edgecolor=MUTED, lw=0.3)
        ax.axvline(obs * 100, color=RED if p <= 0.01 else INK, lw=2)
        ax.set_title(f"{title}\np = {p:.3f}", fontsize=9, color=INK)
        ax.set_yticks([])
        ax.set_xlabel("variance explained (%)", fontsize=8)
    fig.suptitle("Real clock (line) against 999 clocks moved by whole weeks (grey)", x=0.01, ha="left",
                 fontsize=10, color=INK, y=1.08)
    fig.savefig(FIG / "fig4_test.png")


def fig_phases():
    """Each long episode's M2 phase: a tide would make the arrows point one way."""
    from chemai.evaluation.tide_test import CONSTITUENTS, TideTestConfig, prepare
    seg = pd.read_parquet(RES / "segments.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), subplot_kw={"projection": "polar"},
                             gridspec_kw={"wspace": 0.55})
    for ax, (title, col) in zip(axes, [("fouling rate", "fouling_pct_h"), ("pump speed (control)", "pump_speed")]):
        prep = prepare(seg, col, TideTestConfig())
        w = 2 * np.pi * prep.t_hours / CONSTITUENTS["M2"]
        amps, phs = [], []
        for e in np.unique(prep.episode):
            sel = prep.episode == e
            if np.ptp(prep.t_hours[sel]) < 48:
                continue
            a = np.column_stack([np.cos(w[sel]), np.sin(w[sel])])
            c = np.linalg.lstsq(a, prep.y_res[sel], rcond=None)[0]
            amps.append(np.hypot(*c))
            phs.append(np.arctan2(c[1], c[0]))
        amps, phs = np.array(amps), np.array(phs)
        for a, p in zip(amps / amps.max(), phs):
            ax.plot([p, p], [0, a], color=BLUE, lw=1.2, alpha=0.7)
        r = np.abs(np.mean(np.exp(1j * phs)))
        ax.set_title(f"{title}: {len(phs)} episodes\nmean resultant length {r:.2f}", fontsize=9, color=INK)
        ax.set_yticks([])
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["0°", "90°", "180°", "270°"], fontsize=7)
    fig.savefig(FIG / "fig5_phases.png")


def fig_sun():
    """The two declared sun tests: the daily profile and the restart window."""
    r = json.loads((RES / "step6_sun_tests.json").read_text())
    h2, h3 = r["H2_fouling_daily"], r["H3_restart"]
    med = h2["median_y"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    prof = np.array(h2["profile_by_hour"]) / med * 100
    ax.axvspan(9, 15, color="#fff3c4", lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(np.arange(24), prof, "o-", color=ORANGE, ms=3)
    lo, hi = (np.array(h2["solar_contrast_ci95"]) / med * 100)
    ax.set_ylim(-12, 12)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day (local standard time)")
    ax.set_ylabel("fouling rate vs daily mean (%)")
    ax.set_title(f"Daily pattern: p = {h2['p_value']:.2f}. Daylight 09–15 h: "
                 f"{h2['solar_contrast_relative']*100:+.1f} % ({lo:+.1f} to {hi:+.1f})",
                 loc="left", fontsize=9, color=INK)
    ax = axes[1]
    rows = [("restart (H3)", r["H3_restart"]), ("2012–13", r["H3_2012_13"]), ("2014–15", r["H3_2014_15"]),
            ("placebo 24 h", r["H3_placebo_24h"])]
    for i, (name, x) in enumerate(rows):
        m = x["median_y"]
        ax.errorbar(x["effect"] / m * 100, i, xerr=[[(x["effect"] - x["ci95"][0]) / m * 100],
                                                   [(x["ci95"][1] - x["effect"]) / m * 100]],
                    fmt="o", color=BLUE if i else ORANGE, capsize=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("first 3 h after restart (% of median)")
    ax.set_title(f"Restart: p = {h3['p_value']:.2f}", loc="left", fontsize=9, color=INK)
    fig.savefig(FIG / "fig6_sun.png")


def fig_solar():
    """How much capacity and storage a given solar share needs (sunny synthetic year)."""
    df = pd.read_parquet(RES / "step7_solar_plan.parquet")
    df = df[(df.weather == "sunny") & (df.pv_kwp == 0.9)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for c, col in zip((1.5, 2.0, 2.5, 3.0), [MUTED, BLUE, GREEN, ORANGE]):
        d = df[df.capacity == c].sort_values("storage_h")
        ax.plot(d.storage_h, d.solar_fraction * 100, "o-", color=col, label=f"capacity {c:g}× demand")
    ax.axhline(100, color=GRID)
    ax.set_xlabel("product-water tank (hours of demand)")
    ax.set_ylabel("energy from the sun (%)")
    ax.set_ylim(40, 101)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("PV 0.9 kWp per m³/day · modules at design flux", loc="left", fontsize=9, color=INK)
    fig.savefig(FIG / "fig7_solar_share.png")


def fig_solar_week():
    from chemai.models.solar_plan import SolarPlanConfig, plan_lp, pv_year
    cfg = SolarPlanConfig()
    pv = pv_year(cfg)
    _, s = plan_lp(pv, 3.0, 16, 0.9, cfg, series=True)
    daily = pv.reshape(-1, 24).sum(1)
    start = int(np.argmin([daily[d:d + 7].min() for d in range(200, 300)])) + 200      # a week with a dull day
    sl = slice(start * 24, start * 24 + 7 * 24)
    h = np.arange(7 * 24)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    axes[0].fill_between(h, s["pv_kwh"][sl], color="#ffd66b", lw=0, label="sun available")
    axes[0].plot(h, s["production_m3"][sl] * cfg.sec, color=ORANGE, lw=1.4, label="energy used")
    axes[0].plot(h, s["grid_kwh"][sl], color=RED, lw=1.2, label="from grid")
    axes[0].set_ylabel("kWh per hour")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    axes[1].plot(h, s["tank_m3"][sl] * 24, color=BLUE)
    axes[1].set_ylabel("tank (h of demand)")
    axes[1].set_xlabel("hour of the week")
    axes[0].set_title("A week with a dull day: production in daylight, the tank carries the night",
                      loc="left", fontsize=9, color=INK)
    fig.savefig(FIG / "fig8_solar_week.png")


FIGS = {"record": fig_record, "segments": fig_segments, "power": fig_power, "test": fig_test,
        "phases": fig_phases, "sun": fig_sun, "solar": fig_solar, "solar_week": fig_solar_week}

if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
        print("wrote", name)
