"""Step 2 - the physics that decides what a fouling signal looks like.

Three questions, answered on simulated plants (their own seed, 3003, so neither
the development nor the test plants are touched):

  1. How much fouling does a normalisation invent when production changes?
  2. Can a plant measure its own pressure-drop flow exponent?
  3. Does the gap law bend enough, at the levels plants clean at, to matter?

    python projects/p03_ro_fouling/step2_physics.py
"""
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chemai.config import ROConfig  # noqa: E402
from chemai.data.ro_sim import generate_plants  # noqa: E402
from chemai.evaluation.fouling_eval import plant_view  # noqa: E402
from chemai.features.ro_physics import dp_from_blockage, fit_flow_exponent, water_viscosity  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
OUT, FIG = HERE / "results", HERE / "figures"
NORMS = {"linear ratio (1.0)": (1.0, 0.0), "vendor sheet (1.5)": (1.5, 0.25),
         "vendor, no viscosity": (1.5, 0.0), "physics (1.75)": (1.75, 0.25), "plant's own fit": ("fit", None)}


def artefact(run, fe, ve, sensor="dpt") -> np.ndarray:
    """DP_norm as the plant would read it, over the truth on the plant's scale: 1.00 = no error."""
    v = plant_view(run, fe, ve, sensor=sensor)
    out = []
    for tr, g in v.groupby("train"):
        truth = run.daily.set_index(["train", "day"]).loc[list(zip(g["train"], g["day"])), "dp_true"].to_numpy()
        r = g["dp_norm_raw"].to_numpy() / truth
        out.append(r / np.median(r))
    return np.concatenate(out)


def main():
    ro = ROConfig()
    runs = generate_plants(15, seed=3003)
    res = {"plants": 15, "seed": 3003}

    # 1. how much fouling does each normalisation invent?
    res["normalisation_error"] = {}
    for name, (fe, ve) in NORMS.items():
        r = np.concatenate([artefact(run, fe, ve) for run in runs])
        res["normalisation_error"][name] = dict(
            p5_p95_percent=[round(float((np.quantile(r, 0.05) - 1) * 100), 2),
                            round(float((np.quantile(r, 0.95) - 1) * 100), 2)],
            sd_percent=round(float(np.std(r) * 100), 2))
        print(f"{name:24s} sd {np.std(r) * 100:.2f} %   5-95 % band {np.quantile(r, .05):.3f}-{np.quantile(r, .95):.3f}")

    # 2. can the plant measure its own exponent? (first 120 recorded days of each train)
    rows = []
    for run in runs:
        for p in run.params.trains:
            g = run.daily[(run.daily["train"] == p.name) & run.daily["recorded"]]
            g = g[g["day"] <= g["day"].min() + 120]
            a, b = fit_flow_exponent(g), fit_flow_exponent(g.drop(columns=["dp"]))
            rows.append((2 - p.friction_m, a["n"] if a["ok"] else np.nan, b["n"] if b["ok"] else np.nan))
    rows = np.array(rows, float)
    res["exponent_fit"] = dict(
        trains=len(rows),
        true_range=[round(float(rows[:, 0].min()), 3), round(float(rows[:, 0].max()), 3)],
        dp_transmitter_mae=round(float(np.nanmean(np.abs(rows[:, 1] - rows[:, 0]))), 3),
        two_pressure_transmitters_mae=round(float(np.nanmean(np.abs(rows[:, 2] - rows[:, 0]))), 3),
        two_pt_failed=int(np.isnan(rows[:, 2]).sum()))
    print(json.dumps(res["exponent_fit"]))

    # 3. the gap law at the levels plants clean at
    phi = np.linspace(0, 0.45, 200)
    n = ro.gap_exponent
    even = dp_from_blockage(phi, n)
    # straight line through 1.00 and the point where the curve reaches 1.30
    i30 = int(np.argmax(even >= 1.30))
    line = 1 + (even[i30] - 1) * phi / phi[i30]
    gap = float(np.max((line - even)[: i30 + 1]))     # a convex curve sags below its chord
    res["gap_law"] = dict(exponent=n, phi_at_1_30=round(float(phi[i30]), 3),
                          max_bend_below_1_30=round(gap, 4),
                          note="largest gap between the curve and its chord from 1.00 to 1.30, in DP_norm units")
    print(json.dumps(res["gap_law"]))

    # seasons: what viscosity alone does to dP over a +-5 K swing
    mu = water_viscosity(np.array([20.0, 25.0, 30.0])) / water_viscosity(25.0)
    res["viscosity"] = dict(mu_ratio_20_30=[round(float(m), 3) for m in mu],
                            dp_swing_percent=[round(float((m ** ro.dp_visc_exponent - 1) * 100), 2) for m in mu])
    print(json.dumps(res["viscosity"]))
    (OUT / "step2_physics.json").write_text(json.dumps(res, indent=1))

    # ---- figure: one train, three normalisations
    run = runs[0]
    tr = run.params.trains[0].name
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 2]))
    g = run.daily[(run.daily["train"] == tr) & run.daily["recorded"]]
    win = g["day"].between(150, 400)
    axes[0].plot(g.loc[win, "day"], g.loc[win, "perm_flow"], color="#2a6fb0", lw=1.2)
    axes[0].set_ylabel("permeate\nm3/h")
    axes[1].plot(g.loc[win, "day"], g.loc[win, "temperature_c"], color="#a8782f", lw=1.2)
    axes[1].set_ylabel("feed\ntemperature C")
    truth = g.loc[win, "dp_true"].to_numpy()
    for name, col in (("linear ratio (1.0)", "#b53c58"), ("vendor sheet (1.5)", "#c4791b"),
                      ("plant's own fit", "#128c84")):
        fe, ve = NORMS[name]
        v = plant_view(run, fe, ve)
        v = v[(v["train"] == tr) & v["day"].between(150, 400)]
        k = np.median(v["dp_norm_raw"].to_numpy() / truth)
        axes[2].plot(v["day"], v["dp_norm_raw"] / k, ".", ms=3, color=col, label=name)
    axes[2].plot(g.loc[win, "day"], truth, color="#14364c", lw=2, label="true fouling")
    axes[2].set_ylabel("DP_norm")
    axes[2].legend(frameon=False, fontsize=9, ncol=2)
    axes[2].set_xlabel("day")
    axes[0].set_title("When production steps, a wrong flow exponent invents fouling", fontsize=11, loc="left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "step2_normalisation.png", dpi=110)

    # ---- figure: the gap law
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhspan(1.10, 1.35, color="#eef7fb", label="where plants clean")
    ax.plot(phi * 100, even, color="#128c84", lw=2.2, label=f"even fouling, n = {n}")
    ax.plot(phi * 100, dp_from_blockage(phi, 3.0), color="#2a6fb0", lw=1.6, ls="--", label="slit limit, n = 3")
    ax.plot(phi * 100, dp_from_blockage(phi, n, lead_share=0.2, rest_ratio=0.15), color="#b53c58", lw=2,
            label="lead element fouls first")
    ax.plot(phi[: i30 + 1] * 100, line[: i30 + 1], color="#9db3c0", lw=1.2, ls=":", label="straight line")
    ax.set_xlabel("lead element gap blocked, %")
    ax.set_ylabel("DP_norm")
    ax.set_ylim(0.98, 2.0)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("dP ~ gap^-n: it accelerates - but barely below 1.3", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "step2_gap_law.png", dpi=110)


if __name__ == "__main__":
    main()
