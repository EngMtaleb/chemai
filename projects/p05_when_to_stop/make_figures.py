"""Figures for Project 5, regenerated from results/ (and the CCRO record for fig1).

    python projects/p05_when_to_stop/make_figures.py [names...]
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


def fig_batch():
    """One long test batch: the live cost climbs, the average cost falls, they cross at the best end."""
    from chemai.config import CCROConfig
    from chemai.data.nawi_ccro import read_dataset
    from chemai.features.ccro_batches import COLUMNS, batch_curve, cost_curve
    from chemai.models.stopping import rolling_median
    cfg = CCROConfig()
    b = pd.read_parquet(RES / "ccro_batches.parquet")
    row = b[(b.dataset == "dataset6") & (b.exp == "Exp1") & (b.seq == 11)].iloc[0]
    df = read_dataset("dataset6", columns=COLUMNS)
    g = df[(df.exp == "Exp1") & (df.seq == 11)]
    cv = batch_curve(g, cfg)
    filt = cv.phase.to_numpy() == "filt"
    t = cv.t_s.to_numpy()
    m = rolling_median(t, cv.live_kwh_m3.to_numpy(), cfg.live_window_s)
    V = cv.V_m3.to_numpy() * 1000
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ok = filt & (cv.elapsed_min.to_numpy() >= 3)
    ax.plot(V[ok], m[ok], color=INK, lw=1.4, label="live cost of the next m³ (2-min median)")
    for bb, col in ((1.0, BLUE), (10.0, ORANGE)):
        J = cost_curve(cv, row.own_drain_kwh, row.brine_l, bb)
        ax.plot(V[ok], J[ok], color=col, lw=1.6, label=f"average cost so far, brine price b = {bb:g}")
        cross = np.where(ok & (m >= J))[0]
        if len(cross):
            i = cross[0]
            ax.plot(V[i], J[i], "o", color=col, ms=7)
            ax.annotate(f"stop: {cv.c_ms_cm.iat[i]:.0f} mS/cm", (V[i], J[i]), xytext=(6, -16),
                        textcoords="offset points", color=col, fontsize=8.5)
    ax.set_ylim(8.2, 12.6)
    ax.set_xlabel("permeate in this batch, L")
    ax.set_ylabel("kWh per m³ (b in kWh per m³ of brine)")
    ax.set_title("Stop when the live cost of the next m³ reaches the average so far  ·  test batch, BO Exp1 no. 11",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(FIG / "fig1_batch_rule.png")
    plt.close(fig)


def fig_test():
    """Held-out batches: regret of the rule against the optimiser's own ends."""
    r = pd.read_parquet(RES / "step3_test_scores.parquet")
    r = r[r.evaluable & r.constant_setpoints]
    dev = pd.read_parquet(RES / "step2_dev_scores.parquet")
    dev = dev[dev.evaluable & (dev.dataset == "dataset3")]
    esc_b = {"Exp1": 1.0, "Exp2": 10.0, "Exp2_continued/Data": 10.0, "Exp3": 10.0}
    bo_b = {"Exp1": 1.0, "Exp2": 10.0, "Exp3A": 10.0, "Exp3B": 10.0}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    data, labels = [], []
    for b in (0.0, 1.0, 3.0, 10.0):
        ins = r[r.inside & (r.b == b)]
        data.append(ins.regret_rule_pct.clip(lower=1e-3).to_numpy())
        labels.append(f"rule\nb={b:g}\n({len(ins)})")
    own = r[r.inside & (r.b == r.exp.map(bo_b))]
    data.append(own.regret_end_pct.clip(lower=1e-3).to_numpy())
    labels.append(f"BO's own\nends\n({len(own)})")
    bp = ax1.boxplot(data, tick_labels=labels, showfliers=True, widths=0.55, patch_artist=True,
                     medianprops={"color": INK}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
    for p, c in zip(bp["boxes"], [BLUE] * 4 + [ORANGE]):
        p.set_facecolor(c)
        p.set_alpha(0.35)
    ax1.set_yscale("log")
    ax1.axhline(0.5, color=RED, lw=0.8, ls="--")
    ax1.text(0.6, 0.56, "declared median limit 0.5 %", color=RED, fontsize=7.5)
    ax1.set_ylabel("cost above the batch's own best, %")
    ax1.set_title("153 held-out batches (Bayesian-optimisation runs)", loc="left", fontsize=10, color=INK)
    falling = []
    for name, d, bmap in (("ESC\n(development)", dev, esc_b), ("BO\n(test)", r, bo_b)):
        own_b = d[d.b == d.exp.map(bmap)]
        falling.append((name, 100 * own_b.falling_at_end.mean(), len(own_b)))
    ax2.bar([f[0] for f in falling], [f[1] for f in falling], color=[MUTED, ORANGE], width=0.55)
    for i, (_, v, n) in enumerate(falling):
        ax2.text(i, v + 2, f"{v:.0f} %\nof {n}", ha="center", fontsize=8.5, color=INK)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("batches ended while the cost was still falling, %")
    ax2.set_title("the optimisers stopped early", loc="left", fontsize=10, color=INK)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_test.png")
    plt.close(fig)


def fig_map():
    """Four stops on one time axis: what was done, and what energy alone would do."""
    d = json.loads((RES / "step4_ro_cip_economics.json").read_text())
    uf = json.loads((RES / "step5_uf_economics.json").read_text())
    ro = pd.read_csv(RES / "step4_cycles_mid.csv")
    bw = pd.read_csv(RES / "step5_backwash_mid.csv")
    test = pd.read_parquet(RES / "step3_test_scores.parquet")
    test = test[test.evaluable & test.constant_setpoints & test.inside & (test.b == 1.0)]
    batch = pd.read_parquet(RES / "ccro_batches.parquet")
    b6 = batch[batch.dataset == "dataset6"]
    minutes_per_l = float((b6.filt_min / b6.filt_Vp_l).median())
    rows = [
        ("CCRO batch\n(testbed, b = 1)", np.nan, float(test.V_best_l.median() * minutes_per_l) / 60 / 24,
         "energy is the decision: the rule ends a batch a median 0.07 % above its own best (b = 1)"),
        ("UF backwash\n(pilot)", float(bw.v_cycle_m3.median() / bw.q_m3h.median()) / 24,
         float(bw.v_star_m3.median() / bw.q_m3h.median()) / 24, f"energy + water alone: {uf['B1']['median_ratio']:.0f}× longer"),
        ("UF chemical clean\n(pilot)", float(np.mean([c["length_d"] for c in uf["C_intervals"]])),
         float(np.mean([c["t_star_d"] for c in uf["C_intervals"]])), "energy alone: about the same"),
        ("RO cleaning\n(plant, 15 cycles)", float(ro.length_d.median()), np.inf,
         f"energy savings alone never covered a cleaning: 0 of {d['complete_cycles']} (mid case); in 9, no rise at all"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for i, (name, actual, star, note) in enumerate(rows):
        y = len(rows) - 1 - i
        if not np.isfinite(actual):                       # the testbed: the rule's own stop
            ax.plot(star, y, "s", color=BLUE, ms=8, zorder=3)
            ax.text(1.4e-3, y + 0.28, note, fontsize=8.5, color=BLUE)
            continue
        ax.plot(actual, y, "o", color=INK, ms=8, zorder=3)
        if np.isfinite(star) and abs(star / actual - 1) > 0.05:
            ax.annotate("", xy=(star, y), xytext=(actual, y),
                        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2})
            ax.plot(star, y, "s", color=ORANGE, ms=7, zorder=3)
        elif not np.isfinite(star):
            ax.annotate("", xy=(2000, y), xytext=(actual, y),
                        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2, "ls": "--"})
        ax.text(1.4e-3, y + 0.28, note, fontsize=8.5, color=ORANGE)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 3000)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ticks = [1 / 1440, 1 / 24, 1, 7, 30, 365]
    ax.set_xticks(ticks, ["1 min", "1 h", "1 day", "1 week", "1 month", "1 year"])
    ax.grid(axis="y", visible=False)
    ax.set_title("One rule, four stops  ·  ● what was done   ■ what energy (and water) alone would do",
                 loc="left", fontsize=10, color=INK)
    fig.savefig(FIG / "fig3_four_stops.png")
    plt.close(fig)


def fig_ro():
    """RO cleaning: the energy penalty stays far below what a cleaning costs."""
    from chemai.data.ro_deposit import load_daily, load_ledger, recovery_by_campaign
    d = json.loads((RES / "step4_ro_cip_economics.json").read_text())
    mid = pd.read_csv(RES / "step4_cycles_mid.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    ax1.barh(range(len(mid)), mid.penalty_end_usd_m3 * 100, color=BLUE, alpha=0.7)
    ax1.set_yticks(range(len(mid)), mid.id, fontsize=7)
    ax1.set_xlabel("energy penalty at the cleaning, US cents per m³ (mid case)")
    ax1.set_title("the penalty never passed 0.6 ¢/m³; a cleaning costs\n100 days of 1 ¢/m³ (mid case)", loc="left", fontsize=10, color=INK)
    ax1.invert_yaxis()
    daily, ledger = load_daily(), load_ledger()
    rec = recovery_by_campaign(daily, ledger)
    xs, ys = [], []
    from step4_ro_cip_economics import cycle_inputs
    for c in cycle_inputs(daily):
        train, end = c["train"], int(c["id"].split(":")[1].split("-")[1])
        r = rec[(rec.train == train) & (rec.day >= end) & (rec.day <= end + 7) & rec.evaluable]
        if len(r):
            xs.append(c["length"])
            ys.append(float(r.post.iloc[-1]) - c["dp_start"])
    ax2.scatter(xs, np.array(ys) * 100, color=ORANGE, s=28, zorder=3)
    ax2.set_xlabel("cycle length, days")
    ax2.set_ylabel("pressure drop left after the\nclosing cleaning, % of clean")
    ax2.set_title(f"waiting longer left more behind  (Spearman {d['D4']['spearman_rho']:.2f}, p = {d['D4']['p']:.3f})",
                  loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_ro_cleaning.png")
    plt.close(fig)


def fig_uf():
    """UF: the backwash water and the slow climb of the post-backwash resistance."""
    uf = json.loads((RES / "step5_uf_economics.json").read_text())
    bw = pd.read_csv(RES / "step5_backwash_mid.csv")
    daily = pd.read_parquet(RES / "step5_uf_daily.parquet")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    years = sorted(bw.year.unique())
    ax1.boxplot([bw[bw.year == y].ratio for y in years], tick_labels=[str(y) for y in years], widths=0.55,
                medianprops={"color": INK}, flierprops={"markersize": 3})
    ax1.axhline(1, color=MUTED, lw=0.8)
    ax1.set_ylabel("energy-only volume per backwash\n÷ volume the pilot filtered")
    ax1.set_title(f"backwash: {uf['B1']['median_ratio']:.0f}× too often for energy alone", loc="left", fontsize=10, color=INK)
    ax2.plot(daily.index, daily.lnr, ".", color=BLUE, ms=3)
    for c in uf["C_cleaning_days"]:
        ax2.axvline(pd.Timestamp(c), color=GREEN, lw=1)
    ax2.set_ylabel("ln R per element (daily median)")
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.set_title("chemical cleans (green): energy alone would wait 1.1-1.3× as long", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_uf.png")
    plt.close(fig)


FIGS = {"batch": fig_batch, "test": fig_test, "map": fig_map, "ro": fig_ro, "uf": fig_uf}

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    FIG.mkdir(exist_ok=True)
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
        print("wrote", name)
