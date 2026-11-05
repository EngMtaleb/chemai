"""The project page's figures, each drawn from results saved in this folder, in a light and a dark version.

    python projects/p03_ai_for_ro/figures/make_figures.py

1 · phase 1   the week-ahead ΔP error of every forecaster on the real record    phase1/addition2/scores_rolling_addition2.csv
2 · 1 and 1B  days above the cleaning limit under three schedules (simulated)  phase1b/schedule/out/sched_*/results.json
3 · phase 2   the share of the electricity from the sun, 2015–2024             the outputs of notebooks/solar_ro_phase2.ipynb
4 · 2C        electricity bought from the grid in one October window            research_package/bigplant/big/energy/results_summary.json
5 · phase 3   a year under the sun, clean and fouled                            phase3/out/fouled_years_2024.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
P = HERE.parent

THEMES = {   # the same two themes as Project 2's figures
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  accent="#2a78d6", context="#898781", soft="#b9d3f2"),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 accent="#3987e5", context="#898781", soft="#27466b"),
}
FONT = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 10.5})


def frame(ax, t, grid="y"):
    ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.grid(axis=grid, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def titles(fig, t, title, sub, x=0.06, y=0.955, gap=0.045):
    fig.text(x, y, title, fontsize=15, fontweight="semibold", color=t["ink"])
    fig.text(x, y - gap, sub, fontsize=10, color=t["ink2"])


def save(fig, t, name, mode):
    out = HERE / f"{name}_{mode}.png"
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 1 · phase 1: ten forecasters against "no change"
S1 = pd.read_csv(P / "phase1" / "addition2" / "scores_rolling_addition2.csv")
S1 = S1[(S1.target == "dp_pct") & (S1.h == 7)].set_index("method")["MAE_pp"]
NO_CHANGE = S1["no change"]
STAGES = [("Rules", [("smoothed state", "smoothed no change"), ("trend", "trend")]),
          ("Physics", [("fouling law", "fouling law")]),
          ("Machine\nlearning", [("XGBoost", "XGBoost"), ("Prophet, level-shifted", "Prophet + level shift")]),
          ("Physics\n+ ML", [("hybrid", "hybrid")]),
          ("Deep\nlearning", [("LSTM", "LSTM"), ("RNN", "RNN")]),
          ("Deep learning\n+ physics", [("hybrid LSTM", "hybrid LSTM"), ("hybrid RNN", "hybrid RNN")])]


PLACE = {"Prophet, level-shifted": ((0, 8), "center", "bottom", "Prophet, level-shifted\n{v:.2f}"),
         "hybrid RNN": ((6, -10), "left", "top", "hybrid RNN {v:.2f}")}


def fig_forecasters(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    frame(ax, t)
    ax.set_xlim(-0.5, len(STAGES) - 0.2)
    ax.set_ylim(2.45, 4.15)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.8, bottom=0.14)
    best = []
    for i, (_, methods) in enumerate(STAGES):
        vals = [(short, S1[key]) for short, key in methods]
        top = min(vals, key=lambda v: v[1])
        offs = [0.0] if len(vals) == 1 else [-0.12, 0.12]
        for dx, (short, v) in zip(offs, sorted(vals, key=lambda v: v[1])):
            if (short, v) == top:
                best.append((i + dx, v, short))
            else:
                ax.scatter(i + dx, v, s=46, color=t["context"], edgecolors=t["bg"], linewidths=2, zorder=3)
                xy, ha, va, text = PLACE.get(short, ((8, 0), "left", "center", f"{short} {v:.2f}"))
                ax.annotate(text.format(v=v), (i + dx, v), xytext=xy, textcoords="offset points", ha=ha, va=va,
                            fontsize=8.5, color=t["muted"])
    ax.scatter([b[0] for b in best], [b[1] for b in best], s=110, color=t["accent"], edgecolors=t["bg"],
               linewidths=2, zorder=5)
    for x, v, short in best:
        up = v >= NO_CHANGE
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 13 if up else -25), textcoords="offset points", ha="center",
                    fontsize=13, fontweight="semibold", color=t["ink"])
        ax.annotate(short, (x, v), xytext=(0, 30 if up else -39), textcoords="offset points", ha="center",
                    fontsize=9.5, color=t["ink2"])
    ax.axhline(NO_CHANGE, color=t["ink2"], linewidth=1.2, zorder=2)
    ax.annotate(f"“no change” {NO_CHANGE:.2f} — the rival", (len(STAGES) - 0.2, NO_CHANGE), xytext=(0, -16),
                textcoords="offset points", ha="right", fontsize=10, color=t["ink"], fontweight="semibold")
    ax.set_yticks([2.5, 3.0, 3.5, 4.0])
    ax.set_yticklabels(["2.5", "3.0", "3.5", "4.0"])
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([s for s, _ in STAGES], color=t["ink2"], fontsize=10.5)
    titles(fig, t, "Forecasting ΔP a week ahead on the real record, stage by stage",
           "Mean absolute error in points of ΔP rise, days 550–771 of all three trains — lower is better. Blue: the "
           "best of each stage.\nNone passed the bar of 5 % below “no change”. Prophet's own forecast, "
           f"{S1['Prophet']:.2f}, is off the scale.", gap=0.075)
    return save(fig, t, "phase1_forecasters", mode)


# ---------------------------------------------------------------- 2 · phases 1 and 1B: three cleaning schedules
SCH = {k: json.loads((P / "phase1b" / "schedule" / "out" / f"sched_{k}" / "results.json").read_text())
       for k in ("routine", "every")}


def schedule_rows():
    cal = SCH["routine"]["total"]["calendar"]
    assert cal == SCH["every"]["total"]["calendar"], "both studies share the calendar"
    rows = [("Calendar · every 151 days", cal["cleanings"], cal["days_above"], "context")]
    for k, label in (("routine", "The model · routine columns\n(ΔP, temperature, flux)"),
                     ("every", "The model · every column")):
        m = SCH[k]["total"]["model"]
        rows.append((label, m["cleanings"], m["days_above"], "accent"))
    return rows


def fig_schedules(mode):
    t = THEMES[mode]
    rows = schedule_rows()
    fig, ax = plt.subplots(figsize=(10, 3.9), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    frame(ax, t, grid="x")
    cal = rows[0][2]
    for k, (label, n, days, col) in enumerate(reversed(rows)):
        ax.barh(k, days, height=0.56, color=t[col], zorder=3)
        txt = f"{days:,} days above  ·  {n} cleanings"
        if days != cal:
            txt += f"  ·  −{round(100 * (1 - days / cal))} %"
        ax.annotate(txt, (days, k), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10.5,
                    fontweight="semibold" if col == "accent" else "normal", color=t["ink"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=t["ink"], fontsize=10.5)
    ax.set_xlim(0, cal * 1.55)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.72, bottom=0.1)
    titles(fig, t, "Cleaning by the model, not the calendar — on simulated plants",
           "Days above the cleaning limit on 15 simulated plants the model never saw: 45 trains, 772 days each.\n"
           "The same plants under each schedule; a simulation, not a real plant.", x=0.03, y=0.93, gap=0.12)
    return save(fig, t, "schedules", mode)


# ---------------------------------------------------------------- 3 · phase 2: the share from the sun, ten years
def solar_table():
    nb = json.loads((P / "notebooks" / "solar_ro_phase2.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code" and 'print("solar share, lowest and highest year:")' in "".join(c["source"])]
    assert len(code) == 1, "the notebook's ten-year cell"
    texts = ["".join(o["data"]["text/plain"]) for o in code[0]["outputs"] if "data" in o]
    mean_txt, range_txt = texts[0], texts[1]
    cols = ["steady", "same 2x plant, steady", "rule (no forecast)", "plan, perfect 48-h forecast", "plan, tomorrow = today"]

    def row(txt, name):
        line = next(ln for ln in txt.splitlines() if ln.startswith(name))
        return [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
    mean = dict(zip(cols, row(mean_txt, "solar share %")))
    lo, hi = dict(zip(cols, row(range_txt, "min"))), dict(zip(cols, row(range_txt, "max")))
    return mean, lo, hi


def fig_solar(mode):
    t = THEMES[mode]
    mean, lo, hi = solar_table()
    rows = [("Steady · the size of the demand", "steady", "context"),
            ("Follows the sun · a simple rule", "rule (no forecast)", "accent"),
            ("Follows the sun · perfect 48-h forecast", "plan, perfect 48-h forecast", "accent"),
            ("Follows the sun · “tomorrow = today”", "plan, tomorrow = today", "context")]
    fig, ax = plt.subplots(figsize=(10, 3.9), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    frame(ax, t, grid="x")
    for k, (label, key, col) in enumerate(reversed(rows)):
        ax.plot([lo[key], hi[key]], [k, k], color=t[col], linewidth=8, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.scatter(mean[key], k, s=110, color=t[col], edgecolors=t["bg"], linewidths=2, zorder=4)
        ax.annotate(f"{mean[key]:.1f} %", (hi[key], k), xytext=(12, 0), textcoords="offset points", va="center",
                    fontsize=11, fontweight="semibold", color=t["ink"])
        ax.annotate(f"years {lo[key]:.1f}–{hi[key]:.1f}", (hi[key], k), xytext=(62, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=t["muted"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=t["ink"], fontsize=10.5)
    ax.set_xlim(0, 122)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100 %"])
    ax.grid(False)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.spines["bottom"].set_bounds(0, 100)
    fig.subplots_adjust(left=0.29, right=0.99, top=0.72, bottom=0.12)
    titles(fig, t, "How much of the plant's electricity comes from the sun, 2015–2024",
           "A plant twice the demand, 0.8 kWp of panels per m³/day and a 12-hour tank, on NASA POWER weather.\n"
           "Dot: the ten-year mean; bar: the lowest to the highest year.", x=0.03, y=0.93, gap=0.12)
    return save(fig, t, "solar_share", mode)


# ---------------------------------------------------------------- 4 · phase 2C: three ways to run the plant in October
RUNS = json.loads((P / "research_package" / "bigplant" / "big" / "energy" / "results_summary.json").read_text())["runs"]
OCT = {r["metrics"]["policy"]: r["metrics"] for r in RUNS if r["configuration"]["name"] == "2023-10_nominal"}


def fig_october(mode):
    t = THEMES[mode]
    rows = [("Fixed schedule · eight trains", "fixed_eight_trains", "context"),
            ("Daylight + tank rule", "daylight_tank_rule", "context"),
            ("Rolling optimisation (MILP)", "rolling_milp", "accent")]
    fixed = OCT["fixed_eight_trains"]["grid_kWh"]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    frame(ax, t, grid="x")
    for k, (label, key, col) in enumerate(reversed(rows)):
        m = OCT[key]
        ax.barh(k, m["grid_kWh"] / 1000, height=0.56, color=t[col], zorder=3)
        txt = f"{m['grid_kWh'] / 1000:,.0f} MWh  ·  {m['PV_share_pct']:.0f} % of the load from PV"
        if key != "fixed_eight_trains":
            txt += f"  ·  −{100 * (1 - m['grid_kWh'] / fixed):.1f} %"
        ax.annotate(txt, (m["grid_kWh"] / 1000, k), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=10.5, fontweight="semibold" if col == "accent" else "normal", color=t["ink"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=t["ink"], fontsize=10.5)
    ax.set_xlim(0, fixed / 1000 * 1.75)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    tot = [OCT[k]["total_electricity_kWh"] / 1000 for _, k, _ in rows]
    fig.subplots_adjust(left=0.25, right=0.98, top=0.7, bottom=0.1)
    titles(fig, t, "Electricity bought from the grid — one October window, the whole plant",
           "72 hours, 12 trains, the same demand and weather (PVGIS 2023). Total electricity barely moves: "
           f"{tot[0]:,.0f}, {tot[1]:,.0f} and {tot[2]:,.0f} MWh.\nA model of the plant and its control, "
           "not metered data.", x=0.03, y=0.93, gap=0.13)
    return save(fig, t, "october_grid", mode)


# ---------------------------------------------------------------- 5 · phase 3: a year under the sun, clean and fouled
FY = pd.read_csv(P / "phase3" / "out" / "fouled_years_2024.csv")


def fig_fouled(mode):
    t = THEMES[mode]
    states = [("clean", "Clean"), ("median day", "The median\nfouling day"),
              ("operators' cleaning level", "At the operators'\ncleaning level"),
              ("both at 90th percentile", "Heavy · both at the\n90th percentile")]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    frame(ax, t)
    ax.set_xlim(-0.5, len(states) - 0.5)
    ax.set_ylim(0, 100)
    clean = FY[FY.state == "clean"].set_index("strategy")["kWh/m3"]
    for strat, col, name in (("steady", "context", "a steady plant"), ("rule", "accent", "the plant that follows the sun")):
        ys = [FY[(FY.state == s) & (FY.strategy == strat)]["solar share %"].iloc[0] for s, _ in states]
        ax.plot(range(len(states)), ys, color=t[col], linewidth=2, zorder=3)
        ax.scatter(range(len(states)), ys, s=90, color=t[col], edgecolors=t["bg"], linewidths=2, zorder=4)
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.0f} %", (i, y), xytext=(0, 11), textcoords="offset points", ha="center",
                        fontsize=11, fontweight="semibold" if col == "accent" else "normal", color=t["ink"])
        ax.annotate(name, (len(states) - 1, ys[-1]), xytext=(0, -20), textcoords="offset points", ha="right",
                    fontsize=9.5, color=t[col] if col == "accent" else t["muted"])
    for i, (s, _) in enumerate(states):
        e = FY[(FY.state == s) & (FY.strategy == "rule")]["kWh/m3"].iloc[0] / clean["rule"] - 1
        ax.annotate("energy: the reference" if i == 0 else f"energy +{100 * e:.1f} %", (i, 0), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9.5, color=t["ink2"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100 %"])
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels([lab for _, lab in states], color=t["ink2"], fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.8, bottom=0.16)
    titles(fig, t, "Fouled or clean, the sun still carries the plant that follows it",
           "Share of the electricity from the sun over 2024, each train held at one fouling state of the real plant;\n"
           "the energy change is the whole plant's, for the plant that follows the sun. A what-if on our model.",
           y=0.95, gap=0.085)
    return save(fig, t, "fouled_year", mode)


if __name__ == "__main__":
    for mode in ("light", "dark"):
        for f in (fig_forecasters, fig_schedules, fig_solar, fig_october, fig_fouled):
            print(f(mode).relative_to(P))
    print("no change", round(NO_CHANGE, 2), "| best of each stage:",
          {s.replace("\n", " "): round(min(S1[k] for _, k in m), 2) for s, m in STAGES})
    print("schedules", [(r[0].split("\n")[0], r[1], r[2]) for r in schedule_rows()])
    print("solar", {k: v for k, v in solar_table()[0].items()})
    print("october", {k: (round(v["grid_kWh"]), round(v["PV_share_pct"], 1)) for k, v in OCT.items()})
