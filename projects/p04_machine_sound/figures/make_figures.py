"""The project page's figures, each drawn from results/summary.json, in a light and a dark version.

    python projects/p04_machine_sound/figures/make_figures.py

1 · three_tests   the four simple families, the best challenge result and our system on the three public tests
2 · test1_types   Test 1 by machine type: our system against the best submission for that type
3 · test3_steps   Test 3: the four steps from 53 % to 63 %
4 · noise_array   the pumps in factory noise: one microphone and eight, two families
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE.parent / "results" / "summary.json").read_text())

THEMES = {   # the same two themes as the other projects' figures
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  ours="#2a78d6", rival="#eb6834", context="#898781"),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 ours="#3987e5", rival="#d95926", context="#898781"),
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


def titles(fig, t, title, sub, x=0.06, y=0.945, gap=0.05):
    fig.text(x, y, title, fontsize=15, fontweight="semibold", color=t["ink"])
    fig.text(x, y - gap, sub, fontsize=10, color=t["ink2"], va="top", linespacing=1.45)


def save(fig, t, name, mode):
    fig.savefig(HERE / f"{name}_{mode}.png", facecolor=t["bg"])
    plt.close(fig)


# ---------------------------------------------------------------- 1 · the three tests
def fig_three_tests(mode):
    t = THEMES[mode]
    F = S["families"]
    tests = [("Test 1 · known machines\nDCASE 2020 · mean AUC", F["test1_mean_auc"], S["test1"]["best_mean"]["mean"], "best of 116\nsubmissions"),
             ("Test 2 · new operating conditions\nDCASE 2022 · official score", F["test2_official"], S["test2"]["best_team"], "best of 31\nteams"),
             ("Test 3 · new machine types\nDCASE 2025 · official score", F["test3_official"], S["test3"]["best_team"], "best of 35\nteams")]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.14)
    frame(ax, t)
    for i, (name, fam, rival, rival_label) in enumerate(tests):
        simple = [fam[f] for f in ("F1", "F2", "F3", "F4")]
        ax.scatter([i - 0.3] * 4, simple, s=42, color=t["context"], zorder=3, linewidths=0)
        ax.annotate(f"four simple families\nbest {max(simple):.1f}", (i - 0.3, min(simple)), xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                    fontsize=9, color=t["muted"], linespacing=1.3)
        ax.scatter([i], [rival], s=120, color=t["rival"], zorder=4, linewidths=0)
        ax.annotate(f"{rival:.2f}", (i, rival), xytext=(0, -13), textcoords="offset points", ha="center", va="top", fontsize=11.5, fontweight="semibold", color=t["ink"])
        ax.annotate(rival_label, (i, rival), xytext=(0, -28), textcoords="offset points", ha="center", va="top", fontsize=9, color=t["ink2"], linespacing=1.3)
        ax.scatter([i + 0.27], [fam["F5"]], s=150, color=t["ours"], zorder=5, linewidths=0)
        ax.annotate(f"{fam['F5']:.2f}", (i + 0.27, fam["F5"]), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=13,
                    fontweight="semibold", color=t["ink"])
        ax.annotate("our system", (i + 0.27, fam["F5"]), xytext=(11, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, color=t["ink2"])
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, *_ in tests], fontsize=10.5, color=t["ink2"], linespacing=1.4)
    ax.set_xlim(-0.7, 2.75)
    ax.set_ylim(33, 104)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_yticklabels([f"{v} %" for v in (40, 50, 60, 70, 80, 90, 100)], fontsize=10.5)
    titles(fig, t, "Three public tests: simple methods, the best challenge result, our system",
           "Each evaluation set as published. Grey: a limit on one number, distance from the normal cloud, Isolation Forest, autoencoder.\n"
           "Orange: the best result of the challenge itself. Blue: our system, scored after the challenges closed — not a challenge entry.")
    save(fig, t, "three_tests", mode)


# ---------------------------------------------------------------- 2 · Test 1 by machine type
def fig_test1_types(mode):
    t = THEMES[mode]
    T1 = S["test1"]["types"]
    order = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    names = {"fan": "Fan", "pump": "Pump", "slider": "Slide rail", "valve": "Valve", "ToyCar": "Toy car", "ToyConveyor": "Toy conveyor"}
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    fig.subplots_adjust(left=0.15, right=0.95, top=0.72, bottom=0.11)
    frame(ax, t, grid="x")
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        a, b = T1[k]["auc"], T1[k]["best_submission"]
        ax.plot([a, b], [y, y], color=t["axis"], linewidth=2, zorder=2, solid_capstyle="round")
        ax.scatter([b], [y], s=110, color=t["rival"], zorder=3, linewidths=0)
        ax.scatter([a], [y], s=130, color=t["ours"], zorder=4, linewidths=0)
        lo, hi = (a, b) if a < b else (b, a)
        ax.annotate(f"{lo:.2f}", (lo, y), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=10.5,
                    fontweight="semibold" if lo == a else "normal", color=t["ink"] if lo == a else t["ink2"])
        ax.annotate(f"{hi:.2f}", (hi, y), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=10.5,
                    fontweight="semibold" if hi == a else "normal", color=t["ink"] if hi == a else t["ink2"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in reversed(order)], fontsize=11, color=t["ink2"])
    ax.set_xlim(86.5, 101.2)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xticks([88, 90, 92, 94, 96, 98, 100])
    ax.set_xticklabels([f"{v} %" for v in (88, 90, 92, 94, 96, 98, 100)], fontsize=10.5)
    ax.scatter([], [], s=90, color=t["ours"], label="our system")
    ax.scatter([], [], s=80, color=t["rival"], label="best submission for that type")
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    n = S["test1"]["highest_on_types"]
    titles(fig, t, f"Test 1 by machine type: the highest AUC on {n} of the 6 types",
           f"DCASE 2020 Task 2 evaluation set, AUC per machine type. The orange point is the best of all {S['test1']['n_submissions']} submissions "
           "for that type,\nnot one system. The toy conveyor is our weakest type.", gap=0.05)
    save(fig, t, "test1_types", mode)


# ---------------------------------------------------------------- 3 · Test 3: four steps
def fig_test3_steps(mode):
    t = THEMES[mode]
    steps = S["test3"]["steps"]
    labels = ["Expert + trainee\nour first system", "Expert alone\nthe trainee taken out", "Expert, taught the\nmachine settings too", "+ two pre-trained\nmodels, equal votes"]
    ys = [s["score"] for s in steps]
    best = S["test3"]["best_team"]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.16)
    frame(ax, t)
    ax.axhline(best, color=t["rival"], linewidth=2, zorder=2)
    ax.annotate(f"best challenge team {best:.2f}", (-0.42, best), xytext=(0, 6), textcoords="offset points", ha="left", va="bottom", fontsize=10,
                fontweight="semibold", color=t["ink"])
    ax.plot(range(4), ys, color=t["ours"], linewidth=2, zorder=3)
    ax.scatter(range(4), ys, s=130, color=t["ours"], zorder=4, edgecolors=t["bg"], linewidths=2)
    for i, v in enumerate(ys):
        up = v > best      # a value above the reference line is labelled above its point, clear of the line
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 12 if up else -14), textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=13,
                    fontweight="semibold", color=t["ink"])
        if i:
            mid = ((2 * i - 1) / 2, (v + ys[i - 1]) / 2)
            ax.annotate(f"+{v - ys[i - 1]:.1f}", mid, xytext=(8, -6) if mid[1] < best else (0, 9), textcoords="offset points",
                        ha="left" if mid[1] < best else "center", va="top" if mid[1] < best else "bottom", fontsize=10, color=t["ink2"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=10.5, color=t["ink2"], linespacing=1.4)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(48, 67)
    ax.set_yticks([50, 55, 60, 65])
    ax.set_yticklabels(["50 %", "55 %", "60 %", "65 %"], fontsize=10.5)
    titles(fig, t, "Test 3 · new machine types: four steps, from 53 % to 63 %",
           "Official score on the DCASE 2025 Task 2 evaluation set (eight new machine types, nothing to tune on). The axis starts at 48 %.\n"
           "The last step is small, and it was chosen on an evaluation set that had been read before.")
    save(fig, t, "test3_steps", mode)


# ---------------------------------------------------------------- 4 · factory noise, one microphone and eight
def fig_noise_array(mode):
    t = THEMES[mode]
    N = S["noise"]
    lv = ["6dB", "0dB", "-6dB"]
    f5_one = [N["family5_fine_tuned"]["MIC"][k]["pump"] for k in lv]
    f5_arr = [N["family5_fine_tuned"]["ARR"][k]["pump"] for k in lv]
    f2_one = [N["families_1_to_4_pumps"]["F2"][k]["CH0"] for k in lv]
    f2_arr = [N["families_1_to_4_pumps"]["F2"][k]["BF"] for k in lv]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    fig.subplots_adjust(left=0.07, right=0.72, top=0.78, bottom=0.13)
    frame(ax, t)
    series = [(f5_arr, t["ours"], "-", "Family 5 · eight microphones"), (f5_one, t["ours"], (0, (4, 3)), "Family 5 · one microphone"),
              (f2_arr, t["rival"], "-", "Family 2 · eight microphones"), (f2_one, t["rival"], (0, (4, 3)), "Family 2 · one microphone")]
    for (ys, col, ls, name), dy in zip(series, (13, -13, 0, 0)):
        ax.plot(range(3), ys, color=col, linewidth=2, linestyle=ls, zorder=3)
        ax.scatter(range(3), ys, s=64, color=col, zorder=4, edgecolors=t["bg"], linewidths=1.5)
        ax.annotate(f"{name}\n{ys[-1]:.3f}", (2, ys[-1]), xytext=(12, dy), textcoords="offset points", ha="left", va="center", fontsize=10,
                    color=t["ink2"], linespacing=1.35)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["6 dB\nmachine louder than the factory", "0 dB\nas loud as the factory", "−6 dB\nfactory louder than the machine"], fontsize=10.5,
                       color=t["ink2"], linespacing=1.4)
    ax.set_xlim(-0.25, 2.12)
    ax.set_ylim(0.70, 1.01)
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_yticklabels(["0.70", "0.80", "0.90", "1.00"], fontsize=10.5)
    titles(fig, t, "The pumps in factory noise: one microphone or eight",
           "Average AUC over the four MIMII pumps at three noise levels. Blue: Family 5, the fine-tuned expert. Orange: Family 2, distance from\n"
           "the normal cloud, the best simple family. Dashed: one microphone. Solid: the eight microphones combined (each with its own array method).")
    save(fig, t, "noise_array", mode)


for m in ("light", "dark"):
    fig_three_tests(m)
    fig_test1_types(m)
    fig_test3_steps(m)
    fig_noise_array(m)
print("figures written to", HERE)
