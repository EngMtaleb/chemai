"""Look first: each train's dP rise, its 7-day median, the 15 % line, cleanings and the onsets to be predicted."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

P = Path(sys.argv[1])
d = pd.read_csv(P / "data" / "derived_daily_indicator_table.csv")
led = pd.read_csv(P / "data" / "cip_campaign_ledger.csv")
ev = pd.read_csv(P / "out" / "onsets.csv")
m7 = pd.read_csv(P / "out" / "dp_median7.csv")
m7.columns = ["train", "day", "m7"]
INK, INK2, MUTED, GRID, SURF, BLUE, ORANGE, RED = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834", "#d03b3b"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
for ax, tr in zip(axes, "ABC"):
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    g = d[d.train == tr]; m = m7[m7.train == tr]
    ax.axvspan(550, 772, color=GRID, alpha=0.5, lw=0)
    ax.plot(g.day_index, 100 * g.dp_pct, ".", color=BLUE, ms=3, alpha=0.5, label="daily value")
    mm = m.set_index("day").m7
    gap = mm.index.to_series().diff() > 7                       # break the line where the record has a gap
    mm = pd.concat([mm, pd.Series(float("nan"), index=mm.index[gap] - 0.5)]).sort_index()
    ax.plot(mm.index, 100 * mm.values, "-", color=BLUE, lw=1.5, label="7-day median")
    ax.axhline(15, color=MUTED, lw=1)
    for c in led[led.train == tr].day_index:
        ax.axvline(c, color=ORANGE, lw=1.2)
    e = ev[(ev.train == tr) & (ev.day >= 300)]                 # the day-11 start-up spikes are not fouling events
    ax.plot(e.day, [15] * len(e), "v", color=RED, ms=9, label="onset to predict")
    ax.set_ylabel(f"train {tr}\ndP rise, %"); ax.set_ylim(0, 60)
axes[0].text(553, 54, "test period", color=INK2, fontsize=9)
axes[0].text(2, 16.5, "+15 %", color=INK2, fontsize=9)
axes[0].plot([], [], color=ORANGE, lw=1.2, label="cleaning (CIP)")
axes[0].legend(frameon=False, fontsize=9, loc="upper left", ncol=4)
axes[-1].set_xlabel("day")
fig.suptitle("Normalised feed-channel dP rise of the three trains · the events phase 3 will try to see coming", x=0.01, ha="left",
             color=INK, fontsize=12)
plt.tight_layout(); plt.savefig(P / "out" / "events.png", dpi=150); print("ok")
