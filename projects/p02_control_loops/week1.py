"""Project 2 - Control Loop Performance. Week 1: simulator, faults, Harris index.

Run:  python projects/p02_control_loops/week1.py
      (SACAC part runs only if data/sacac/ is populated)

Answers four questions, each recorded in results.json:
  A. Does each simulated condition behave as its label claims?
  B. What does the Harris index see - and what can it NOT tell apart?
  C. How sensitive is Harris to the assumed dead time?
  D. What does Harris say on the only real loops with a published dead time?
"""
from __future__ import annotations

import dataclasses as dc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from chemai.config import DATA_DIR, ControlLoopConfig
from chemai.data import generate_dataset, load_sacac, sample_loop_spec, simulate_loop
from chemai.features import (delay_samples, harris_index, harris_sensitivity,
                             oscillation_regularity)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("p02.week1")

HERE = Path(__file__).parent
OUT = HERE / "results.json"
FIG = HERE / "figures" / "week1.png"


def sim_delay(spec) -> int:
    """Delay in historian samples as simulated: theta plus one controller step."""
    return delay_samples(spec.theta + spec.ts / spec.substeps, spec.ts)


def run_metrics(spec, df, cfg) -> dict:
    e = (df.sp - df.pv).to_numpy()
    period, r = oscillation_regularity(detrend(e))
    return {**spec.to_dict(), "d": sim_delay(spec),
            "harris": harris_index(e, sim_delay(spec), cfg.harris_ar_order),
            "regularity": r, "period_s": period * spec.ts if np.isfinite(period) else None,
            "oscillating": bool(r > cfg.regularity_threshold)}


# ------------------------------------------------------------------ A + B
def simulated_study(cfg) -> tuple[pd.DataFrame, dict]:
    specs, runs = generate_dataset(n_loops_per_cell=8, seeds_per_loop=2, cfg=cfg)
    by_id = {s.loop_id: s for s in specs}
    rows = [{**run_metrics(by_id[lid], df, cfg), "seed_idx": k} for (lid, k), df in runs.items()]
    res = pd.DataFrame(rows)

    table = (res.groupby(["condition", "loop_type"])
             .agg(harris_median=("harris", "median"),
                  harris_q10=("harris", lambda x: x.quantile(0.1)),
                  harris_q90=("harris", lambda x: x.quantile(0.9)),
                  oscillating=("oscillating", "mean"), n_runs=("harris", "size"))
             .round(3))
    log.info("condition table\n%s", table.to_string())
    return res, {"n_loops": len(specs), "n_runs": len(runs),
                 "table": {f"{c}|{lt}": v for (c, lt), v in table.to_dict("index").items()}}


# ---------------------------------------------------------------------- C
def delay_sensitivity(res: pd.DataFrame, runs_cfg) -> dict:
    """Harris on healthy loops when the delay is under- or over-estimated."""
    rng = np.random.default_rng(1)
    out = {}
    for lt in ("F", "P", "L"):
        ratios = {-1: [], 2: [], 5: []}
        for i in range(10):
            s = sample_loop_spec(lt, "healthy", rng, runs_cfg)
            e = (lambda d: (d.sp - d.pv).to_numpy())(simulate_loop(s, 3000, seed=i, cfg=runs_cfg))
            d0 = sim_delay(s)
            base = harris_index(e, d0)
            for off in ratios:
                if d0 + off >= 1:
                    ratios[off].append(harris_index(e, d0 + off) / base)
        out[lt] = {f"d{off:+d}": round(float(np.median(v)), 2) for off, v in ratios.items() if v}
    log.info("Harris / Harris(true d), median over healthy loops: %s", out)
    return out


# ---------------------------------------------------------------------- A
def tight_gain_margin_sweep(cfg) -> dict:
    """Why tuning_tight is GM 1.03-1.12: regular oscillation needs it."""
    rng = np.random.default_rng(2)
    out = {}
    for lt in ("F", "P", "L"):
        row = {}
        for gm in (1.03, 1.1, 1.2, 1.35, 1.5):
            hits = []
            for i in range(10):
                s = sample_loop_spec(lt, "healthy", rng, cfg)
                s = dc.replace(s, kc=s.kc * s.gain_margin / gm, gain_margin=gm)
                e = (lambda d: (d.sp - d.pv).to_numpy())(simulate_loop(s, 3000, seed=i, cfg=cfg))
                hits.append(oscillation_regularity(detrend(e))[1] > cfg.regularity_threshold)
            row[str(gm)] = float(np.mean(hits))
        out[lt] = row
    log.info("fraction oscillating vs gain margin: %s", out)
    return out


def stiction_visibility(cfg, n=40) -> dict:
    """Does a sticky valve produce a regular oscillation? Depends on slip vs load noise."""
    rng = np.random.default_rng(11)
    rows = []
    for lt in ("F", "P", "L"):
        for i in range(n):
            s = sample_loop_spec(lt, "stiction", rng, cfg)
            e = (lambda d: (d.sp - d.pv).to_numpy())(simulate_loop(s, 3000, seed=i, cfg=cfg))
            rows.append({"loop_type": lt, "slip_to_load": s.slip_to_load,
                         "osc": oscillation_regularity(detrend(e))[1] > cfg.regularity_threshold})
    d = pd.DataFrame(rows)
    bins = [0, 1, 2, np.inf]
    d["band"] = pd.cut(d.slip_to_load, bins, labels=["J/σ<1", "1-2", ">2"])
    tab = d.groupby(["loop_type", "band"], observed=False).osc.agg(["mean", "size"]).round(2)
    log.info("stiction: fraction oscillating by slip / load-noise\n%s", tab.to_string())
    return {"overall": d.groupby("loop_type").osc.mean().round(2).to_dict(),
            "by_band": {f"{lt}|{b}": v for (lt, b), v in tab.to_dict("index").items()},
            "_frame": d}


# ---------------------------------------------------------------------- D
def longest_constant_sp(df: pd.DataFrame) -> pd.DataFrame:
    seg = (df.sp.diff().fillna(0) != 0).cumsum()
    best = seg.value_counts().idxmax()
    return df[seg == best]


def sacac_references(cfg) -> dict | None:
    root = DATA_DIR / "sacac"
    if not root.exists():
        log.warning("data/sacac not found - skipping the real-loop check")
        return None
    out = {}
    for r in load_sacac(root):
        if r.dead_time is None:
            continue
        if r.ts is None:
            log.warning("%s: sampling time not confirmed - skipped", r.name)
            continue
        df = longest_constant_sp(r.data)               # Harris needs regulatory data
        e = (df.sp - df.pv).to_numpy()
        d_lo, d_hi = (delay_samples(t, r.ts) for t in r.dead_time)
        sens = harris_sensitivity(e, range(d_lo, d_hi + 1), cfg.harris_ar_order)
        period, reg = oscillation_regularity(detrend(e))
        out[r.name] = {"label": r.label, "ts": r.ts, "dead_time_s": r.dead_time,
                       "samples_used": int(len(df)), "samples_total": int(len(r.data)),
                       "harris_by_d": {k: round(v, 3) for k, v in sens.items()},
                       "regularity": round(reg, 2),
                       "period_s": round(period * r.ts, 1) if np.isfinite(period) else None}
    log.info("SACAC references: %s", json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------- figure
def figure(res: pd.DataFrame, vis: pd.DataFrame, cfg) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed (pip install -e '.[viz]') - no figure")
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.3, 1.2, 0.8]})

    spec = sample_loop_spec("F", "stiction", np.random.default_rng(4), cfg)
    d = simulate_loop(spec, 400, seed=3, cfg=cfg)
    ax[0].plot(d.t, d.op, lw=1.2, label="OP  (recorded in plants)")
    ax[0].plot(d.t, d.mv, lw=1.2, label="MV  (valve position - not recorded)")
    a2 = ax[0].twinx()
    a2.plot(d.t, d.pv, lw=0.8, color="C2", alpha=0.7)
    a2.set_ylabel("PV", color="C2")
    ax[0].set(title=f"Simulated flow loop, stiction S={spec.stiction_s:.1f}% J={spec.stiction_j:.1f}%",
              xlabel="s", ylabel="% travel")
    ax[0].legend(fontsize=8, loc="upper left")

    conds = list(cfg.conditions)
    for k, lt in enumerate(("F", "P", "L")):
        sub = res[res.loop_type == lt]
        vals = [sub[sub.condition == c].harris for c in conds]
        pos = np.arange(len(conds)) + (k - 1) * 0.25
        ax[1].boxplot(vals, positions=pos, widths=0.2, patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=f"C{k}", alpha=0.5), medianprops=dict(color="k"))
        ax[1].plot([], [], color=f"C{k}", lw=6, alpha=0.5, label=lt)
    ax[1].set_xticks(range(len(conds)), [c.replace("_", "\n") for c in conds], fontsize=8)
    ax[1].set(ylabel="Harris index", ylim=(0, 1.05),
              title="Harris ranks - it does not diagnose")
    ax[1].legend(title="loop type", fontsize=8)

    tab = vis.groupby(["loop_type", "band"], observed=False).osc.mean().unstack()
    tab.T[["F", "P", "L"]].plot(kind="bar", ax=ax[2], rot=0, alpha=0.8, color=["C0", "C1", "C2"])
    ax[2].set(ylabel="fraction with regular oscillation", xlabel="slip J / load-noise σ",
              title="Stiction is visible only above the noise", ylim=(0, 1))
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=110)
    log.info("figure -> %s", FIG)


def main() -> dict:
    cfg = ControlLoopConfig()
    res, study = simulated_study(cfg)
    vis = stiction_visibility(cfg)
    frame = vis.pop("_frame")
    results = {
        "config": {"conditions": list(cfg.conditions), "tight_gain_margin": cfg.tight_gain_margin,
                   "op_noise_limit": cfg.op_noise_limit, "n_samples": cfg.n_samples,
                   "harris_ar_order": cfg.harris_ar_order},
        "simulated": study,
        "harris_delay_sensitivity": delay_sensitivity(res, cfg),
        "tight_gain_margin_sweep": tight_gain_margin_sweep(cfg),
        "stiction_visibility": vis,
        "sacac_references": sacac_references(cfg),
    }
    figure(res, frame, cfg)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
