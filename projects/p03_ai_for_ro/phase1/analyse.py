"""Phase 3 - score the saved predictions exactly as declared: forecasts, 90 % ranges, early warning.

    python analyse.py <out folder>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3lib as L  # noqa: E402

OUT = Path(sys.argv[1])
pd.set_option("display.width", 220)


def forecast_scores(p):
    rows = []
    for (c, h), g in p.groupby(["target", "h"]):
        pooled = L.scores(g.y.to_numpy(), g.now.to_numpy(), {m: g[m].to_numpy() for m in L.METHODS})
        blocks = {b: L.scores(gb.y.to_numpy(), gb.now.to_numpy(), {m: gb[m].to_numpy() for m in L.METHODS})
                  for b, gb in g.groupby("block")}
        for m in L.METHODS:
            bs = [blocks[b][m]["skill_pct"] for b in sorted(blocks)]
            rows.append({"target": c, "h": h, "method": m, "n": len(g), **pooled[m],
                         **{f"skill block {b + 1}": s for b, s in enumerate(bs)},
                         "beats no change": L.verdict(pooled[m]["skill_pct"], bs) if m != "no change" else None})
    return pd.DataFrame(rows)


def coverage(p):
    rows = []
    for (c, h), g in p.groupby(["target", "h"]):
        for m in L.METHODS:
            inside = np.abs(g.y - g[m]) <= g[f"q90 {m}"]
            rows.append({"target": c, "h": h, "method": m, "coverage_pct": 100 * inside.mean(),
                         "width_pp": 100 * 2 * g[f"q90 {m}"].mean(),
                         **{f"cov block {b + 1}": 100 * inside[g.block == b].mean() for b in range(len(L.BLOCKS))}})
    return pd.DataFrame(rows)


def onsets(m7, thr):
    out = []
    for tr, s in m7.groupby("train"):
        s = s.set_index("day").m7
        for t, v in s.items():
            if v < thr:
                continue
            before = s[(s.index >= t - 14) & (s.index < t)]
            if len(before) and (before < thr).all():
                out.append({"train": tr, "day": int(t)})
    return pd.DataFrame(out)


def warnings(wd, m7, thr):
    ev = onsets(m7, thr)
    ev = ev[ev.day >= L.BLOCKS[0][0]]
    w = wd.merge(m7.rename(columns={"day": "t"}), on=["train", "t"], how="inner")
    w = w[w.m7 < thr]
    rows = []
    for h, gh in w.groupby("h"):
        for m in L.METHODS:
            warn = gh[gh[m] >= thr]
            caught, leads = 0, []
            for _, e in ev.iterrows():
                ww = warn[(warn.train == e.train) & (warn.t >= e.day - 14) & (warn.t <= e.day - 1)]
                if len(ww):
                    caught += 1; leads.append(int(e.day - ww.t.min()))
            false = 0
            for _, r in warn.iterrows():
                nxt = ev[(ev.train == r.train) & (ev.day > r.t) & (ev.day <= r.t + h)]
                false += int(nxt.empty)
            episodes = sum(int((g.t.diff().fillna(99) > 1).sum()) for _, g in warn.groupby("train"))
            rows.append({"threshold_pct": 100 * thr, "h": h, "method": m, "onsets": len(ev), "caught": caught,
                         "lead_days": leads, "warning_days": len(warn), "false_warning_days": false,
                         "warning_episodes": episodes, "days_checked": len(gh)})
    return pd.DataFrame(rows), ev


def main():
    ro = pd.read_csv(OUT / "pred_rolling.csv")
    lo = pd.read_csv(OUT / "pred_unseen_train.csv")
    wd = pd.read_csv(OUT / "pred_warning_days.csv")
    m7 = pd.read_csv(OUT / "dp_median7.csv"); m7.columns = ["train", "day", "m7"]
    fs_ro, fs_lo = forecast_scores(ro), forecast_scores(lo)
    cov = coverage(ro)
    warn = {thr: warnings(wd, m7, thr) for thr in (0.15, 0.20)}
    fs_ro.to_csv(OUT / "scores_rolling.csv", index=False); fs_lo.to_csv(OUT / "scores_unseen_train.csv", index=False)
    cov.to_csv(OUT / "coverage.csv", index=False)
    pd.concat([w for w, _ in warn.values()]).to_csv(OUT / "warning.csv", index=False)
    for name, fs in (("FUTURE, ROLLING ORIGIN", fs_ro), ("FUTURE, UNSEEN TRAIN", fs_lo)):
        print(f"\n===== {name}: skill vs no change, %  (MAE of no change in brackets, percentage points)")
        for c in L.TARGETS:
            t = fs[fs.target == c].pivot(index="method", columns="h", values="skill_pct").loc[L.METHODS]
            base = fs[(fs.target == c) & (fs.method == "no change")].set_index("h").MAE_pp
            print(f"-- {c}  [no change MAE: " + ", ".join(f"h{h} {v:.2f}" for h, v in base.items()) + "]")
            print(t.round(1).to_string())
            v = fs[(fs.target == c) & (fs.method != "no change")]
            print("beats no change (declared rule):", v[v["beats no change"] == True][["h", "method"]].values.tolist())
            r2 = fs[fs.target == c].pivot(index="method", columns="h", values="R2_change").loc[L.METHODS]
            print("R2 of the change:\n" + r2.round(2).to_string())
            bl = fs[(fs.target == c)].set_index(["h", "method"])[[f"skill block {b + 1}" for b in range(4)]]
            print("skill by block:\n" + bl.round(1).to_string())
    print("\n===== 90 % RANGES (rolling origin): coverage %, width pp")
    for c in L.TARGETS:
        print(f"-- {c}")
        print(cov[cov.target == c].pivot(index="method", columns="h", values="coverage_pct").loc[L.METHODS].round(1).to_string())
        print(cov[cov.target == c].pivot(index="method", columns="h", values="width_pp").loc[L.METHODS].round(1).to_string())
    print("\n===== EARLY WARNING (descriptive)")
    for thr, (w, ev) in warn.items():
        print(f"-- threshold {100 * thr:.0f} %: onsets in the test period {ev[['train', 'day']].values.tolist()}")
        print(w.drop(columns=["threshold_pct"]).to_string(index=False))
    res = {"rolling": fs_ro.to_dict("records"), "unseen_train": fs_lo.to_dict("records"), "coverage": cov.to_dict("records"),
           "warning": {str(k): v[0].to_dict("records") for k, v in warn.items()}}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
