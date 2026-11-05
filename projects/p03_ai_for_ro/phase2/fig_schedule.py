"""Figures for step 3.
(1) heat map: how much of the plant's electricity the sun covers, by panel size and tank size, for two plant sizes
    (follow-the-sun rule; the tank-0 row is the same as steady production: with no tank there is nothing to shift);
(2) one summer day and one winter day of the rule strategy: sun and plant electricity, production, tank.

    python fig_schedule.py <work folder> [year] [capacity for the days figure] [kWp per m3/day] [tank hours]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

W = Path(sys.argv[1])
YEAR = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
C_DAY = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
P_DAY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
S_DAY = int(sys.argv[5]) if len(sys.argv) > 5 else 12
sys.path.insert(0, str(W))
import schedule as Sc  # noqa: E402

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]                       # sequential blue, steps 100 -> 700
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ (1) heat map
t = pd.read_csv(W / "out" / f"schedule_{YEAR}.csv")
cmap = LinearSegmentedColormap.from_list("blue", RAMP)
lo, hi = 30.0, 100.0
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
for ax, C in zip(axes, (1.5, 2.0)):
    d = t[(t.strategy == "rule") & (t.capacity == C)].pivot(index="tank_h", columns="kwp_per_m3d", values="solar_share_pct")
    d = d.sort_index()
    P, H = list(d.columns), list(d.index)
    ax.set_facecolor(SURF)
    ax.pcolormesh(np.arange(len(P) + 1), np.arange(len(H) + 1), d.values, cmap=cmap, vmin=lo, vmax=hi,
                  edgecolors=SURF, linewidth=2)
    for i, h in enumerate(H):
        for j, p in enumerate(P):
            v = d.values[i, j]
            dark = (v - lo) / (hi - lo) > 0.55
            ax.text(j + 0.5, i + 0.5, f"{v:.0f} %", ha="center", va="center", fontsize=11,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(np.arange(len(P)) + 0.5, [f"{p:.1f}" for p in P])
    ax.set_yticks(np.arange(len(H)) + 0.5, [f"{h:.0f} h" for h in H])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("panels, kWp per m³/day of demand")
    ax.set_title(f"plant {C:g}× demand", loc="left", color=INK, fontsize=11)
axes[0].set_ylabel("product tank, hours of demand")
fig.suptitle(f"Share of the plant's electricity that comes from the sun · follow-the-sun rule · {YEAR}",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, 0.01, "Tank 0 h = steady production: with no tank there is nothing to shift, so even the most panels "
         "tested cover under 45 %.", color=INK2, fontsize=9)
plt.tight_layout(rect=(0, 0.04, 1, 0.97)); plt.savefig(W / "out" / f"schedule_heatmap_{YEAR}.png", dpi=150, facecolor=SURF)

# ------------------------------------------------------------------ (2) a summer day and a winter day
perf = Sc.load_perf([W / "out" / "perf_table_part0.csv", W / "out" / "perf_table_part1.csv"])
other = Sc.other_energy(perf)
cv = Sc.curves(perf, other, C_DAY)
pv, T = Sc.load_sun_sea(YEAR)
kwp, S_max = P_DAY * 1000.0, S_DAY * Sc.DEMAND
q = Sc.run_rule(pv * kwp, T, cv, S_max)
S = np.cumsum(q - Sc.DEMAND)
E = np.array([Sc.energy(Sc.nearest(cv, T[i]), q[i]) for i in range(len(q))])
idx = pd.date_range(f"{YEAR}-01-01", periods=len(pv), freq="h")
fig, axes = plt.subplots(3, 2, figsize=(13, 8.2), facecolor=SURF, sharex="col")
for j, day in enumerate((f"{YEAR}-07-15", f"{YEAR}-01-15")):
    k = np.where(idx.strftime("%Y-%m-%d") == day)[0]
    k = np.r_[k[0] - 24:k[-1] + 1]                                   # the day and the day before
    h = np.arange(len(k)) - 24
    a, b, c = axes[0, j], axes[1, j], axes[2, j]
    for ax in (a, b, c):
        style(ax)
    a.fill_between(h, pv[k] * kwp, color=YELLOW, alpha=0.35, lw=0, label="sun available")
    a.plot(h, E[k], color=BLUE, lw=2, label="plant electricity")
    a.set_title(("Summer" if j == 0 else "Winter") + f" · {day} and the day before · sea {T[k[-1]]:.1f} °C",
                loc="left", color=INK, fontsize=11)
    b.plot(h, q[k], color=BLUE, lw=2)
    b.axhline(Sc.DEMAND, color=MUTED, lw=1)
    b.text(h[-1], Sc.DEMAND + 2, "demand", color=INK2, fontsize=9, ha="right")
    c.plot(h, S[k], color=AQUA, lw=2)
    c.axhline(S_max, color=MUTED, lw=1)
    c.text(h[-1], S_max - 30, "tank full", color=INK2, fontsize=9, ha="right", va="top")
    c.set_ylim(-10, S_max * 1.08)
    b.set_ylim(0, max(c_["q_max"] for c_ in cv.values()) * 1.08)
    a.set_ylim(0, max((pv[k] * kwp).max(), E[k].max()) * 1.3)
    c.set_xlabel("hour (0 = midnight of the day shown)")
    c.set_xticks(np.arange(-24, 25, 6))
    if j == 0:
        a.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
        a.set_ylabel("kW"); b.set_ylabel("production, m³/h"); c.set_ylabel("water in tank, m³")
for r in range(3):                                                    # same scale left and right
    lo_, hi_ = min(axes[r, 0].get_ylim()[0], axes[r, 1].get_ylim()[0]), max(axes[r, 0].get_ylim()[1], axes[r, 1].get_ylim()[1])
    axes[r, 0].set_ylim(lo_, hi_); axes[r, 1].set_ylim(lo_, hi_)
fig.suptitle(f"Follow-the-sun rule · plant {C_DAY:g}× demand · {P_DAY:.1f} kWp per m³/day · tank {S_DAY} h",
             x=0.01, ha="left", color=INK, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(W / "out" / f"schedule_days_{YEAR}.png", dpi=150, facecolor=SURF)
print("ok")
