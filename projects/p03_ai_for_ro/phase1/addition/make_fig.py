"""Phase 3 addition - the deck figure: skill against no change, rolling origin, the new methods next to phase 3's one pass.

    python3 -B make_fig.py            (reads scores_rolling_addition.csv, writes p3_skill_addition.png)

Every point is read from scores_rolling_addition.csv (written by score.py); nothing is typed in by hand.
Style copied from phase 3's own skill figure (ro_fouling_phase3.ipynb, section 5).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
VIOLET, GREEN, MAGENTA = "#6250d6", "#008300", "#e87ba4"      # validated as a set (dataviz validate_palette, light, all pairs)
plt.rcParams.update({"font.family": "Inter", "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
YMIN, YMAX = -100, 20
LINES = [("Prophet", VIOLET, "-", "Prophet"),
         ("Prophet + level shift", VIOLET, (0, (4, 2)), "Prophet + a drop at each cleaning (declared variant)"),
         ("LSTM", GREEN, "-", "LSTM"),
         ("RNN", MAGENTA, "-", "RNN"),
         ("smoothed no change", MUTED, "-", "smoothed no change (phase 3’s one pass)")]


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    s = pd.read_csv(HERE / "scores_rolling_addition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, c in zip(axes, ["dp_pct", "flux_pct"]):
        style(ax)
        f = s[s.target == c]
        for m, col, ls, lab in LINES:
            g = f[f.method == m].sort_values("h")
            ax.plot(g.h, g.skill_pct, color=col, ls=ls, lw=2, marker="o", ms=6, label=lab, clip_on=True)
            below = g[g.skill_pct < YMIN]
            if len(below):                                   # off the chart: say where, with the numbers
                lo, hi = below.skill_pct.min(), below.skill_pct.max()
                ax.annotate(f"{m}: {hi:.0f} to {lo:.0f} %, below the chart".replace("-", "−"),
                            xy=(7.5, YMIN), xytext=(7.5, YMIN + 9), ha="center", color=INK, fontsize=9,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.axhline(0, color=INK2, lw=1)
        ax.axhline(5, color=MUTED, lw=1, ls=(0, (2, 2)))
        ax.text(0.6, 5.6, "+5 %: the bar to beat", color=INK2, fontsize=8, va="bottom")
        ax.set_xticks([1, 3, 7, 14])
        ax.set_xlabel("days ahead")
        ax.set_xlim(0.3, 14.7)
        ax.set_ylim(YMIN, YMAX)
        ax.set_title(("ΔP" if c == "dp_pct" else "flux loss") + " · rolling origin · skill against no change",
                     loc="left", color=INK, fontsize=11)
        if c == "flux_pct":                                  # the one new point above +5 % fails on the blocks
            r = f[(f.method == "Prophet + level shift") & (f.h == 14)].iloc[0]
            ax.annotate(f"+{r.skill_pct:.1f} %, but better in only {int(r['blocks positive'])} of 4 blocks: fails",
                        xy=(14, r.skill_pct), xytext=(13.2, 15.5), ha="right", va="center", color=INK, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkA=2, shrinkB=4))
    axes[0].set_ylabel("skill: % less error than no change")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=5)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "p3_skill_addition.png", dpi=100, facecolor=SURF)
    print("wrote", HERE / "p3_skill_addition.png")


if __name__ == "__main__":
    main()
