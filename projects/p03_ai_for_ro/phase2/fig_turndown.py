"""Figure: one vessel design - energy and permeate salinity as production goes down.

    python fig_turndown.py <work folder> [csv name] [recovery] [membrane label] [png name]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

W = Path(sys.argv[1])
CSV = sys.argv[2] if len(sys.argv) > 2 else "turndown.csv"
R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
LABEL = sys.argv[4] if len(sys.argv) > 4 else "phase-1 membrane"
PNG = sys.argv[5] if len(sys.argv) > 5 else "turndown.png"
t = pd.read_csv(W / "out" / CSV)
d = t[(t.n == 7) & (t.R == R)]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {11.0: "#2a78d6", 25.0: "#1baf7a", 36.0: "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), facecolor=SURF)
for ax in axes:
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for T, g in d.groupby("T"):
    ok = g.note == "ok"
    for ax, col in zip(axes, ("sec_ro", "c_p")):
        ax.plot(g.production_pct, g[col], "-", color=COL[T], lw=2)
        ax.plot(g.production_pct[ok], g[col][ok], "o", color=COL[T], ms=6)
        ax.plot(g.production_pct[~ok], g[col][~ok], "o", mfc=SURF, mec=COL[T], ms=6, mew=1.5)
        last = g.iloc[-1]
        ax.text(last.production_pct + 1.5, last[col], f"{T:.0f} °C", color=INK, va="center", fontsize=9)
q_min = d[d.Q_brine >= 3.4].production_pct.min()
for ax in axes:
    ax.axvline(q_min, color=INK2, lw=1, ls="--")
    ax.set_xlabel("production, % of design (14 LMH)")
axes[0].text(q_min + 1, axes[0].get_ylim()[1] * 0.98, "brine flow\n≥ 3.4 m³/h →", color=INK2, va="top", fontsize=9)
axes[0].set_ylabel("RO energy, kWh per m³"); axes[1].set_ylabel("permeate salinity, mg/L")
axes[0].set_title("Energy per m³ falls as production falls", loc="left", color=INK, fontsize=11)
axes[1].set_title("…but the permeate gets saltier, and more so when the sea is warm", loc="left", color=INK, fontsize=11)
fig.text(0.01, 0.01, f"One vessel, 7 elements, {100 * R:.0f} % recovery, feed 43 g/L, {LABEL} · filled dot: inside every FilmTec limit · open dot: a limit crossed",
         color=INK2, fontsize=9)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(W / "out" / PNG, dpi=150, facecolor=SURF); print("ok")
