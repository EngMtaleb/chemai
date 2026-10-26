"""The project page's results: the F1 of every method on the 58 labelled real loops, stage by stage,
and what real loops in training add. Reads the result files of experiments 11 and 12 (written by the
experiment scripts, not committed), draws two figures in a light and a dark version, and prints the
Markdown tables used in the README.

    python projects/p02_control_loops/figures/make_results_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"
E11F = json.loads((EXP / "exp11" / "results_exp11_features.json").read_text())["methods"]
E11D = json.loads((EXP / "exp11" / "results_exp11_dl.json").read_text())["networks"]
E12F = json.loads((EXP / "exp12" / "results_exp12_features.json").read_text())["methods"]
E12D = json.loads((EXP / "exp12" / "results_exp12_dl.json").read_text())["methods"]
PAIRED = json.loads((EXP / "exp12" / "paired_exp12.json").read_text())

FAMILIES = [("logistic_regression", "Logistic regression"), ("svm", "SVM"), ("decision_tree", "Decision tree"),
            ("random_forest", "Random forest"), ("gradient_boosting", "Gradient boosting"),
            ("xgboost", "XGBoost"), ("lightgbm", "LightGBM")]


def b(m):                      # condition B: simulation + the real loops of the other four parts
    return m["sim_plus_real"]


def a(m):                      # condition A: simulation only
    return m["sim_only"]


RULE = b(E12F["the rule"])
RULE_NET = RULE["caught"] - RULE["false_alarms"]

# (stage, name, result in condition B, key in paired_exp12.json)
ROWS = [("The rule", "Regular cycle + triangular shape", RULE, None)]
for key, name in FAMILIES:
    ROWS.append(("Machine learning", f"{name} · default", b(E12F[f"{key} · default"]), f"exp12 · {key} · default"))
    ROWS.append(("Machine learning", f"{name} · tuned", b(E12F[f"{key} · tuned"]), f"exp12 · {key} · tuned"))
ROWS += [
    ("Machine learning", "Ensemble · the 7 tuned vote", b(E12F["ensemble · 7 tuned"]), "exp12 · ensemble · 7 tuned"),
    ("Machine learning", "Ensemble · 5 tuned (experiment 11)", b(E11F["ensemble · soft voting of 5 tuned"]),
     "exp11 · ensemble · soft voting of 5 tuned"),
    ("Physics", "S alone · one threshold", b(E12F["S alone"]), "exp12 · S alone"),
    ("Physics", "Physics forest · S added", b(E11F["physics · random forest"]), "exp11 · physics · random forest"),
    ("Physics", "Hybrid · the rule, then the physics forest", b(E11F["physics · hybrid"]), "exp11 · physics · hybrid"),
    ("Physics", "Rule as teacher · physics forest", b(E11F["physics · rule as teacher"]),
     "exp11 · physics · rule as teacher"),
    ("Deep learning", "ANN · on the numbers, S included", b(E11F["ANN · physics features"]),
     "exp11 · ANN · physics features"),
    ("Deep learning", "1D-CNN · reads CO and CV", b(E11D["cnn"]), "exp11 · cnn"),
    ("Deep learning", "LSTM · reads CO and CV", b(E11D["lstm"]), "exp11 · lstm"),
    ("Deep learning + physics", "CNN · hybrid", b(E12D["cnn · hybrid"]), "exp12 · cnn · hybrid"),
    ("Deep learning + physics", "CNN · rule as teacher", b(E12D["cnn · rule as teacher"]), "exp12 · cnn · rule as teacher"),
    ("Deep learning + physics", "LSTM · hybrid", b(E12D["lstm · hybrid"]), "exp12 · lstm · hybrid"),
    ("Deep learning + physics", "ANN · hybrid", b(E12F["ANN · hybrid"]), "exp12 · ANN · hybrid"),
    ("Deep learning + physics", "ANN · rule as teacher", b(E12F["ANN · rule as teacher"]), "exp12 · ANN · rule as teacher"),
    ("Deep learning + physics", "LSTM · rule as teacher", b(E12D["lstm · rule as teacher"]),
     "exp12 · lstm · rule as teacher"),
]
STAGES = ["The rule", "Machine learning", "Physics", "Deep learning", "Deep learning + physics"]
REAL = [("Random forest", E12F["random_forest · default"]), ("ANN", E11F["ANN · physics features"]),
        ("CNN", E11D["cnn"]), ("LSTM", E11D["lstm"])]

THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  accent="#2a78d6", context="#898781"),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 accent="#3987e5", context="#898781"),
}
FONT = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 10.5})


def net(r):
    return r["caught"] - r["false_alarms"] - RULE_NET


def frame(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def swarm(values, center, ax, fig, diam_px):
    """Beeswarm offsets: each dot as close to the stage's centre as it can sit without touching another."""
    fig.canvas.draw()
    bb = ax.get_window_extent()
    px_x = bb.width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    px_y = bb.height / (ax.get_ylim()[1] - ax.get_ylim()[0])
    placed, out = [], {}
    for k in sorted(range(len(values)), key=lambda k: -values[k]):
        y = values[k]
        for step in [0] + [s * m for m in range(1, 40) for s in (1, -1)]:
            x = center + step * diam_px / px_x * 0.55
            if all(((x - x0) * px_x) ** 2 + ((y - y0) * px_y) ** 2 >= diam_px ** 2 for x0, y0 in placed):
                break
        placed.append((x, y))
        out[k] = x
    return [out[k] for k in range(len(values))]


def fig_stages(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    frame(ax, t)
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    ax.set_ylim(0, 1.0)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.14)
    best = []
    for i, stage in enumerate(STAGES):
        rows = [r for r in ROWS if r[0] == stage]
        top = max(rows, key=lambda r: r[2]["f1"])
        xs = swarm([r[2]["f1"] for r in rows], i, ax, fig, diam_px=17)
        for x, (_, name, res, _) in zip(xs, rows):
            if res is top[2]:
                best.append((x, res["f1"], name))
            else:
                ax.scatter(x, res["f1"], s=46, color=t["context"], edgecolors=t["bg"], linewidths=2, zorder=3)
    xs, ys = [p[0] for p in best], [p[1] for p in best]
    ax.plot(xs, ys, color=t["accent"], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    ax.scatter(xs, ys, s=110, color=t["accent"], edgecolors=t["bg"], linewidths=2, zorder=5)
    short = {"Regular cycle + triangular shape": "the rule", "XGBoost · tuned": "XGBoost, tuned",
             "S alone · one threshold": "S alone", "1D-CNN · reads CO and CV": "1D-CNN", "CNN · hybrid": "CNN hybrid"}
    for i, f1, name in best:
        ax.annotate(f"{f1:.2f}", (i, f1), xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=13, fontweight="semibold", color=t["ink"])
        ax.annotate(short.get(name, name), (i, f1), xytext=(0, 31), textcoords="offset points", ha="center",
                    fontsize=9.5, color=t["ink2"])
    ax.axhline(RULE["f1"], color=t["ink2"], linewidth=1, zorder=2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(["The rule", "Machine\nlearning", "Physics", "Deep\nlearning", "Deep learning\n+ physics"],
                       color=t["ink2"], fontsize=10.5)
    fig.text(0.06, 0.955, "F1 on the 58 real loops, stage by stage", fontsize=15, fontweight="semibold",
             color=t["ink"])
    fig.text(0.06, 0.91, "Each dot is one method; blue is the best of each stage. Every loop is tested once, "
             "by a model that never saw it or its valve.", fontsize=10, color=t["ink2"])
    out = HERE / f"f1_by_stage_{mode}.png"
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)
    return out


def fig_real(mode):
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
    fig.patch.set_facecolor(t["bg"])
    frame(ax, t)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=1)
    for k, (name, m) in enumerate(reversed(REAL)):
        f0, f1 = a(m)["f1"], b(m)["f1"]
        ax.plot([f0, f1], [k, k], color=t["context"], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(f0, k, s=70, color=t["context"], edgecolors=t["bg"], linewidths=2, zorder=3)
        ax.scatter(f1, k, s=110, color=t["accent"], edgecolors=t["bg"], linewidths=2, zorder=4)
        ax.annotate(f"{f0:.2f}", (f0, k), xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=10, color=t["ink2"])
        ax.annotate(f"{f1:.2f}", (f1, k), xytext=(10, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=11, fontweight="semibold", color=t["ink"])
    ax.set_yticks(range(len(REAL)))
    ax.set_yticklabels([n for n, _ in reversed(REAL)], color=t["ink"], fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(REAL) - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.spines["bottom"].set_visible(False)
    fig.text(0.13, 0.93, "What real loops add", fontsize=15, fontweight="semibold", color=t["ink"])
    fig.text(0.13, 0.855, "F1 on the 58 real loops. Trained on simulation only, then with about 40 real loops "
             "added to the training.", fontsize=10, color=t["ink2"])
    ax.scatter([], [], s=70, color=t["context"], label="simulation only")
    ax.scatter([], [], s=110, color=t["accent"], label="+ real loops")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=10,
              labelcolor=t["ink2"], handletextpad=0.3, columnspacing=1.4, borderaxespad=0.2)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.74, bottom=0.1)
    out = HERE / f"real_loops_{mode}.png"
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)
    return out


def tables():
    lines = ["| Stage | Method | Caught (of 26) | False alarms | F1 | vs the rule |", "|---|---|---:|---:|---:|---:|"]
    for stage in STAGES:
        rows = [r for r in ROWS if r[0] == stage]
        top = max(rows, key=lambda r: r[2]["f1"])
        for k, (_, name, res, _) in enumerate(rows):
            d = net(res)
            vs = "—" if stage == "The rule" else (f"**{d:+d} ✓**" if d > 2 else f"{d:+d}".replace("+0", "0"))
            f1 = f"**{res['f1']:.2f}**" if res is top[2] else f"{res['f1']:.2f}"
            lines.append(f"| {stage if k == 0 else ''} | {name} | {res['caught']} | {res['false_alarms']} | {f1} | {vs} |")
    real = ["| | Simulation only · F1 | + real loops · F1 | Caught · false alarms, before → after |",
            "|---|---:|---:|---|"]
    for name, m in REAL:
        real.append(f"| {name} | {a(m)['f1']:.2f} | **{b(m)['f1']:.2f}** | "
                    f"{a(m)['caught']} · {a(m)['false_alarms']} → {b(m)['caught']} · {b(m)['false_alarms']} |")
    ranges = ["| Method | vs the rule | 95 % range | Resamples above 0 |", "|---|---:|---:|---:|"]
    for _, name, res, key in ROWS:
        if key and net(res) > 2:
            p = PAIRED[key]
            ranges.append(f"| {name} | {p['vs_rule']:+d} | {p['range95'][0]:+d} to {p['range95'][1]:+d} | "
                          f"{p['share_above_0']:.0%} |")
    return "\n".join(lines), "\n".join(real), "\n".join(ranges)


if __name__ == "__main__":
    assert len([r for r in ROWS if r[0] != "The rule"]) == 29, "the 29 methods of experiments 11 and 12"
    for mode in THEMES:
        print(fig_stages(mode).name, fig_real(mode).name)
    for tbl in tables():
        print("\n" + tbl)
