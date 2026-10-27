"""Collect the three tests into results/summary.json (the numbers the README, the validation
record and the deck quote) and draw the three figures."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
R, FIG = HERE / "results", HERE / "figures"
A = json.loads((R / "stepA2_ro_test.json").read_text())
Bj = json.loads((R / "stepB2_uf_test.json").read_text())
C = json.loads((R / "stepC2_ccro_test.json").read_text())
A1 = json.loads((R / "stepA1_ro_dev.json").read_text())
B1 = json.loads((R / "stepB1_uf_dev.json").read_text())
C1 = json.loads((R / "stepC1_ccro_dev.json").read_text())

PICKS = [("ml", "Machine learning"), ("ml_ensemble", "Ensemble"), ("ml_physics", "ML + physics"),
         ("ml_physics_ensemble", "Ensemble + physics"), ("dl", "Deep learning"), ("dl_physics", "DL + physics")]


def key_of(part_frozen: dict, rung: str) -> str:
    if rung == "ml_ensemble":
        return "ml:ensemble_top3"
    if rung == "ml_physics_ensemble":
        return "ml_physics:ensemble_top3"
    pre = {"ml": "ml", "ml_physics": "ml_physics", "dl": "dl", "dl_physics": "dl"}[rung]
    return f"{pre}:{part_frozen[rung]}"


def pr_point(y, s, recall=None, precision=None):
    o = np.argsort(-s)
    ys = y[o]
    tp = np.cumsum(ys)
    prec, rec = tp / np.arange(1, len(ys) + 1), tp / ys.sum()
    if recall is not None:
        return float(prec[rec >= recall - 1e-9].max())
    ok = prec >= precision - 1e-9
    return float(rec[ok].max()) if ok.any() else 0.0


out = {"commits": {"A": A["declaration_commit"], "B": Bj["declaration_commit"], "C": C["declaration_commit"]}}
# ---- A
fa = A1["frozen"]
rp_real = pd.read_parquet(R / "stepA2_real_predictions.parquet")
yr = rp_real["y"].to_numpy().astype(bool)
a = {"final_exam": {k: A["final_exam"][k] for k in ("plants", "rows", "positives")},
     "real": {k: A["real_exploratory"][k] for k in ("rows", "positives", "groups")},
     "rule": {"exam": A["final_exam"]["margin_rule"], "real": A["real_exploratory"]["margin_rule"]},
     "forecast": {"exam": A["final_exam"]["forecast"], "real": A["real_exploratory"]["forecast"]},
     "picks": {}}
for rung, label in PICKS:
    k = key_of(fa, rung)
    e, r = A["final_exam"]["models"][k], A["real_exploratory"]["models"][k]
    a["picks"][rung] = dict(label=label, model=k, exam=dict(recall=e["recall"], precision=e["precision"], ap=e["ap"],
                                                             d=e["recall_minus_rule"], ci=e["recall_minus_rule_ci95"]),
                            real=dict(recall=r["recall"], precision=r["precision"], ap=r["ap"], d=r["recall_minus_rule"],
                                      ci=r["recall_minus_rule_ci95"],
                                      precision_at_rule_recall=round(pr_point(yr, rp_real[k].to_numpy(), recall=0.426), 3),
                                      recall_at_rule_precision=round(pr_point(yr, rp_real[k].to_numpy(), precision=0.929), 3)))
a["matched_training_real"] = A["real_exploratory_matched_training"]["models"]
out["A"] = a
# ---- B
fb = B1["frozen"]
s = Bj["scores"]
b = {"test": Bj["test"], "dev_rows": B1["data"]["rows"],
     "baselines": {k: s[k]["mae"] for k in ("no_model_dev_median", "pred_line", "pred_hermia", "pred_history")},
     "picks": {}, "best_any": min(((k, v["mae"]) for k, v in s.items() if ":" in k and not k.startswith("clock")), key=lambda x: x[1]),
     "clock": Bj["clock_check"], "hermia_weights": Bj["hermia_weights_test_mean"], "separability": Bj["separability"]}
for rung, label in PICKS:
    k = {"dl": f"dl:{fb['dl']}", "dl_physics": "dl:hermia"}.get(rung, key_of(fb, rung))
    b["picks"][rung] = dict(label=label, model=k, mae=s[k]["mae"], d=s[k]["minus_no_model"], ci=s[k]["minus_no_model_ci95"])
out["B"] = b
# ---- C
fc = C1["frozen"]
c = {"test_batches": C["test_batches"], "dev_batches": C1["data"]["batches"], "r_max_esc": C1["data"]["r_max_esc"],
     "r_max_open_loop": C1["data"]["r_max_open_loop"], "law": {}, "picks": {}, "curve_mae": C["curve_mae_after_minute_20"]}
for bb, d in C["by_b"].items():
    c["law"][bb] = dict(inside=d["inside"], err=d["square_root_law_T4"]["median_abs_err_pct"],
                        ratio=d["square_root_law_T4"]["median_ratio"], regret=d["square_root_law_T4"]["regret_median_pct"])
for rung, label in PICKS:
    k = {"dl": "dl:mlp", "dl_physics": "dl:powerlaw"}.get(rung, key_of(fc, rung))
    c["picks"][rung] = dict(label=label, model=k, by_b={bb: dict(err=d[k]["median_abs_err_pct"], ratio=d[k]["median_ratio"],
                                                                  regret=d[k]["regret_median_pct"], d=d[k]["minus_law"],
                                                                  ci=d[k]["minus_law_ci95"], beyond=d[k]["beyond_batch"])
                                                         for bb, d in C["by_b"].items()})
out["C"] = c
(R / "summary.json").write_text(json.dumps(out, indent=1, default=str))

# ---------------------------------------------------------------- figures
COL = {"rule": "#9fb6c3", "ml": "#2a6fb0", "ml_ensemble": "#5b8fc9", "ml_physics": "#128c84",
       "ml_physics_ensemble": "#4fb3a9", "dl": "#a8782f", "dl_physics": "#b53c58"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, part, title in ((axes[0], "exam", "60 new simulated plants (final exam)"), (axes[1], "real", "the real plant (exploratory)")):
    ax.scatter(a["rule"][part]["recall"], a["rule"][part]["precision"], s=120, color="#14364c", marker="s", label="margin rule", zorder=3)
    ax.scatter(a["forecast"][part]["recall"], a["forecast"][part]["precision"], s=90, color="#9fb6c3", marker="D", label="forecast", zorder=3)
    for rung, _ in PICKS:
        p = a["picks"][rung][part]
        ax.scatter(p["recall"], p["precision"], s=70, color=COL[rung], label=a["picks"][rung]["label"], zorder=3)
    ax.set_xlim(0.15, 0.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("recall: share of real crossings flagged")
    ax.set_ylabel("precision: flags that came true")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.25)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.suptitle("Part A — will a train reach its review level within 28 days?", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "figA_ro.png", dpi=110)

fig, ax = plt.subplots(figsize=(11, 4.8))
labels = ["no model (median)", "Hermia, from 5 min", "history"] + [b["picks"][r]["label"] for r, _ in PICKS]
vals = [b["baselines"]["no_model_dev_median"], b["baselines"]["pred_hermia"], b["baselines"]["pred_history"]] + \
       [b["picks"][r]["mae"] for r, _ in PICKS]
cols = ["#14364c", "#b53c58", "#9fb6c3"] + [COL[r] for r, _ in PICKS]
ax.barh(range(len(vals))[::-1], vals, color=cols)
ax.set_yticks(range(len(vals))[::-1], labels)
for i, v in zip(range(len(vals))[::-1], vals):
    ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
ax.axvline(vals[0], color="#14364c", lw=1, ls="--")
ax.set_xlabel("test MAE of the climb from minute 4 to 11, % of R (2014-15, 10,830 cycles)")
ax.set_title("Part B — how far will a UF cycle's resistance climb?", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "figB_uf.png", dpi=110)

fig, ax = plt.subplots(figsize=(11, 4.6))
bs = ["b=0", "b=1", "b=3", "b=10"]
series = [("square-root law (T4)", [c["law"][x]["err"] for x in bs], "#14364c")] + \
         [(c["picks"][r]["label"], [c["picks"][r]["by_b"][x]["err"] for x in bs], COL[r])
          for r in ("ml", "ml_physics", "ml_physics_ensemble", "dl", "dl_physics")]
w = 0.14
for j, (lab, v, col) in enumerate(series):
    ax.bar(np.arange(4) + (j - 2.5) * w, v, w, color=col, label=lab)
ax.set_xticks(range(4), [f"{x}  ({c['law'][x]['inside']} batches)" for x in bs])
ax.set_ylabel("median |V* / V_best − 1|, %")
ax.set_ylim(0, 20)
ax.legend(frameon=False, fontsize=8.5, ncol=3)
ax.set_title("Part C — where should a CCRO batch end? (forecast from minutes 5–20)", loc="left", fontsize=12)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "figC_ccro.png", dpi=110)
print(json.dumps(out["commits"]), "figures written")
