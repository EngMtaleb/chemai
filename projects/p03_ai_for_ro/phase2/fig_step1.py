import json, sys
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd
W = Path(sys.argv[1]); r = json.loads((W / "out/sun_step1.json").read_text())
m = pd.DataFrame(r["monthly"])
INK, INK2, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"
names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
x = np.arange(12)
a.bar(x, m.pv, width=0.62, color=BLUE, edgecolor=SURF, linewidth=2)
a.errorbar(x, m.pv, yerr=[m.pv - m.pv_min_year, m.pv_max_year - m.pv], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
a.set_xticks(x, names); a.set_ylabel("kWh per kWp per day")
a.set_title(f"One kWp of panels in Kuwait: {r['annual_kwh_per_kwp']['mean']:,.0f} kWh a year on average\n"
            "monthly average 2015–2024 · whiskers: the lowest and highest year", loc="left", color=INK, fontsize=11)
for i in (0, 5, 11):
    a.text(x[i], 0.25, f"{m.pv[i]:.1f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
b.plot(x, m.cloud_loss_pct, "-o", color=BLUE, lw=2, ms=6, label="clouds (sunshine lost)")
b.plot(x, m.heat_loss_pct, "-o", color=ORANGE, lw=2, ms=6, label="heat (panel output lost)")
b.text(11.25, m.cloud_loss_pct.iloc[-1], "clouds", color=INK, va="center", fontsize=9)
b.text(11.25, m.heat_loss_pct.iloc[-1], "heat", color=INK, va="center", fontsize=9)
b.set_xticks(x, names); b.set_xlim(-0.5, 12.2); b.set_ylabel("% lost")
b.set_title("What the panels lose each month\nclouds: all-sky vs clear-sky sun · heat: hot cells vs 25 °C cells", loc="left", color=INK, fontsize=11)
b.set_ylim(0, 14); b.legend(frameon=False, loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig(W / "out/sun_step1.png", dpi=150, facecolor=SURF); print("ok")
