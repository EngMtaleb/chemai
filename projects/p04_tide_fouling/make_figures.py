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


FIGS = {"record": fig_record, "segments": fig_segments, "power": fig_power}

if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
        print("wrote", name)
